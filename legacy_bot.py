# legacy_bot_service.py (Simplified)

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Literal, List
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()

embedding_function = OpenAIEmbeddings()
docs = [
    Document(
        page_content="Bella Vista is owned by Antonio Rossi...",
        metadata={"source": "owner.txt"},
    ),
    Document(
        page_content="Bella Vista offers a range of dishes...",
        metadata={"source": "dishes.txt"},
    ),
    Document(
        page_content="Bella Vista is open from Monday to Sunday...",
        metadata={"source": "restaurant_info.txt"},
    ),
]
db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever(search_kwargs={"k": 2})
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-4o-mini")
rag_chain = prompt | llm


class AgentState(TypedDict):
    messages: List
    documents: List
    on_topic: str


class GradeQuestion(BaseModel):
    score: str = Field(
        description="Is the question about the restaurant? 'yes' or 'no'"
    )


def question_classifier(state: AgentState):
    question = state["messages"][-1].content
    system = "You are a classifier. Is the question about Bella Vista restaurant (owner, prices, hours, menu)? Respond 'yes' or 'no'."
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
    state["documents"] = retriever.invoke(state["messages"][-1].content)
    return state


def generate_answer(state):
    generation = rag_chain.invoke(
        {"context": state["documents"], "question": state["messages"][-1].content}
    )
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


app = FastAPI(title="RAG Legacy Bot Service (Stateless)")


@app.post("/invoke", response_model=JSONRPCResponse)
async def invoke_handler(rpc: JSONRPCRequest):
    if rpc.method == "invoke_rag":
        try:
            print("🤖 [RAG Bot] 'invoke_rag' called.")
            final_state = await rag_graph.ainvoke(
                {"messages": [HumanMessage(content=rpc.params["query"])]}
            )
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
    print("🚀 Starting RAG Legacy Bot Service on http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
