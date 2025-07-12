import uvicorn
from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Literal, List
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()

embedding_function = OpenAIEmbeddings()

docs = [
    Document(
        page_content="Bella Vista is owned by Antonio Rossi, a renowned chef with over 20 years of experience in the culinary industry. He started Bella Vista to bring authentic Italian flavors to the community.",
        metadata={"source": "owner.txt"},
    ),
    Document(
        page_content="Bella Vista offers a range of dishes with prices that cater to various budgets. Appetizers start at $8, main courses range from $15 to $35, and desserts are priced between $6 and $12.",
        metadata={"source": "dishes.txt"},
    ),
    Document(
        page_content="Bella Vista is open from Monday to Sunday. Weekday hours are 11:00 AM to 10:00 PM, while weekend hours are extended from 11:00 AM to 11:00 PM.",
        metadata={"source": "restaurant_info.txt"},
    ),
    Document(
        page_content="Bella Vista offers a variety of menus including a lunch menu, dinner menu, and a special weekend brunch menu. The lunch menu features light Italian fare, the dinner menu offers a more extensive selection of traditional and contemporary dishes, and the brunch menu includes both classic breakfast items and Italian specialties.",
        metadata={"source": "restaurant_info.txt"},
    ),
]

db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever(search_kwargs={"k": 2})
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-4o-mini")
rag_chain = prompt | llm


class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str


class GradeQuestion(BaseModel):
    score: str = Field(description="Question is about the restaurant? 'yes' or 'no'")


def question_classifier(state: AgentState):
    question = state["messages"][-1].content
    system = """You are a classifier. Is the user's question about Bella Vista restaurant (owner, prices, hours, menu)? Respond 'yes' or 'no'."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{question}")]
    )
    structured_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(
        GradeQuestion
    )
    result = (grade_prompt | structured_llm).invoke({"question": question})
    state["on_topic"] = result.score
    return state


def on_topic_router(state):
    return "on_topic" if state["on_topic"].lower() == "yes" else "off_topic"


def retrieve_docs(state):
    question = state["messages"][-1].content
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state


def generate_answer(state):
    question = state["messages"][-1].content
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    state["messages"].append(generation)
    return state


def off_topic_response(state: AgentState):
    state["messages"].append(
        AIMessage(
            content="I can only answer questions about the Bella Vista restaurant."
        )
    )
    state["documents"] = []
    return state


workflow = StateGraph(AgentState)
workflow.add_node("topic_decision", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate_answer", generate_answer)
workflow.add_conditional_edges(
    "topic_decision",
    on_topic_router,
    {"on_topic": "retrieve", "off_topic": "off_topic_response"},
)
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("off_topic_response", END)
workflow.set_entry_point("topic_decision")
rag_graph = workflow.compile()

T = TypeVar("T")


class JSONRPCRequest(BaseModel, Generic[T]):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: T | None = None
    id: str | int | None = None


class JSONRPCResponse(BaseModel, Generic[T]):
    jsonrpc: Literal["2.0"] = "2.0"
    result: T | None = None
    error: dict | None = None
    id: str | int | None = None


async def get_jsonrpc_request(request: Request) -> JSONRPCRequest:
    return JSONRPCRequest.model_validate(await request.json())


def deserialize_messages(messages_data: list[dict]) -> list[BaseMessage]:
    messages = []
    for msg_data in messages_data:
        role = msg_data.get("role")
        content = msg_data.get("content")
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


app = FastAPI(title="RAG Legacy Bot Service (JSON-RPC)")


@app.post("/jsonrpc", response_model=JSONRPCResponse)
async def jsonrpc_handler(rpc: JSONRPCRequest = Depends(get_jsonrpc_request)):
    if rpc.method == "invoke_rag_graph":
        print("🤖 [RAG Bot] 'invoke_rag_graph' called.")
        try:
            langchain_messages = deserialize_messages(rpc.params["messages"])
            if not any(isinstance(msg, HumanMessage) for msg in langchain_messages):
                raise ValueError("No human message found in request history.")

            final_state = await rag_graph.ainvoke({"messages": langchain_messages})

            answer = final_state["messages"][-1].content
            documents = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in final_state.get("documents", [])
            ]

            return JSONRPCResponse(
                id=rpc.id, result={"answer": answer, "documents": documents}
            )
        except Exception as e:
            print(f"❌ [RAG Bot] Error: {e}")
            return JSONRPCResponse(id=rpc.id, error={"code": -32000, "message": str(e)})

    return JSONRPCResponse(
        id=rpc.id, error={"code": -32601, "message": "Method not found"}
    )


if __name__ == "__main__":
    print("Starting RAG Legacy Bot Service on http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
