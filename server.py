# a2a_server.py
import asyncio
import uvicorn
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Generic, TypeVar, Literal
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import RequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types import (
    AgentCard, AgentCapabilities, AgentSkill, Task, TaskState, Message, Role, TextPart,
    MessageSendParams, TaskQueryParams, TaskIdParams, Artifact, DataPart
)
from a2a.utils import new_agent_text_message, create_task_obj

T = TypeVar("T")
class JSONRPCRequest(BaseModel, Generic[T]):
    jsonrpc: Literal["2.0"] = "2.0"; method: str; params: T | None = None; id: str | int | None = None

ADAPTER_SERVICE_URL = "http://localhost:8001/jsonrpc"

class RAGAgentExecutor(AgentExecutor):
    def __init__(self, task_store: TaskStore):
        self.task_store = task_store

    async def run_interaction_flow(self, task_id: str):
        # This background task no longer sets the "Thinking..." status.
        # It now focuses solely on the long-running work.
        print(f"⚙️  [A2A Server] Background flow for {task_id} started.")
        task = await self.task_store.get(task_id)
        if not task: return

        try:
            async with httpx.AsyncClient() as client:
                adapter_request = JSONRPCRequest(
                    method="process_and_forward",
                    params={"task": task.model_dump(mode="json")}, id=task_id
                ).model_dump(exclude_none=True)
                response = await client.post(ADAPTER_SERVICE_URL, json=adapter_request, timeout=90.0)
                response.raise_for_status()
                adapter_response = response.json()
                if adapter_response.get("error"): raise Exception(f"Adapter error: {adapter_response['error']}")
            
            result_payload = adapter_response.get("result", {})
            answer_text = result_payload.get("answer", "Error: No answer from adapter.")
            source_documents = result_payload.get("documents", [])
            
            final_task = await self.task_store.get(task_id)
            if final_task and final_task.status.state == TaskState.working:
                final_task.status.state = TaskState.completed
                agent_response_message = Message(role=Role.agent, parts=[TextPart(text=answer_text)], messageId=f"msg-{uuid4().hex}", taskId=task_id)
                final_task.history.append(agent_response_message)
                final_task.status.message = agent_response_message
                
                if source_documents:
                    final_task.artifacts = [Artifact(artifactId=f"sources-{task_id}", name="Source Documents", parts=[DataPart(data={"sources": source_documents})])]
                else:
                    final_task.artifacts = []

                await self.task_store.save(final_task)
                print(f"🏁 [A2A Server] Task {task_id} completed. History updated.")

        except Exception as e:
            print(f"❌ [A2A Server] Error in flow: {e}")
            task_to_fail = await self.task_store.get(task_id)
            if task_to_fail:
                task_to_fail.status.state = TaskState.failed
                task_to_fail.status.message = new_agent_text_message(str(e))
                await self.task_store.save(task_to_fail)

    async def execute(self, c, e): pass
    async def cancel(self, c, e): pass

class RAGRequestHandler(RequestHandler):
    def __init__(self, agent_executor: RAGAgentExecutor, task_store: TaskStore):
        self.agent_executor = agent_executor
        self.task_store = task_store

    async def on_message_send(self, params: MessageSendParams, context=None) -> Task:
        print("✅ [A2A Server] 'on_message_send' received.")
        task = await self.task_store.get(params.message.taskId) if params.message.taskId else None
        
        if not task:
            task = create_task_obj(params)
        
        if not task.history: task.history = []
        task.history.append(params.message)

        # --- CHANGE IS HERE ---
        # 1. Immediately set the status to 'working' and the message to 'Thinking...'.
        # This provides immediate feedback to the user.
        task.status.state = TaskState.working
        task.status.message = new_agent_text_message("Thinking...")
        
        # 2. Save this intermediate state.
        await self.task_store.save(task)
        print(f"   Task {task.id} updated to 'Working' state before starting background job.")

        # 3. Start the long-running process in the background.
        asyncio.create_task(self.agent_executor.run_interaction_flow(task.id))

        # 4. Return the task object that already contains the "Working/Thinking" state.
        return task

    async def on_get_task(self, params: TaskQueryParams, context=None) -> Task | None:
        return await self.task_store.get(params.id)

    # --- Other methods remain unchanged ---
    async def on_cancel_task(self, params: TaskIdParams, context=None) -> Task | None: raise NotImplementedError()
    async def on_message_send_stream(self, params: MessageSendParams, context=None): raise NotImplementedError()
    async def on_resubscribe_to_task(self, params: TaskIdParams, context=None): raise NotImplementedError()
    async def on_set_task_push_notification_config(self, params, context=None): raise NotImplementedError()
    async def on_get_task_push_notification_config(self, params, context=None): raise NotImplementedError()
    async def on_list_task_push_notification_config(self, params, context=None): raise NotImplementedError()
    async def on_delete_task_push_notification_config(self, params, context=None): raise NotImplementedError()

# The build_app and main sections are identical and can be left as they are.
def build_app() -> FastAPI:
    skill = AgentSkill(id="rag_chat", name="RAG Chatbot", description="Answers questions about Bella Vista restaurant using internal documents.", tags=["rag", "chat"], examples=["When does Bella Vista open?"])
    card = AgentCard(name="A2A RAG Proxy Agent", description="An agent that uses RAG to answer questions.", url="http://localhost:8000/", version="3.0", defaultInputModes=["text/plain"], defaultOutputModes=["text/plain"], capabilities=AgentCapabilities(streaming=False), skills=[skill])
    task_store = InMemoryTaskStore()
    agent_executor = RAGAgentExecutor(task_store=task_store)
    http_handler = RAGRequestHandler(agent_executor=agent_executor, task_store=task_store)
    a2a_app = A2AStarletteApplication(agent_card=card, http_handler=http_handler).build()
    api = FastAPI(title="A2A RAG Entrypoint Server")
    api.mount("/", a2a_app)
    return api

app = build_app()

if __name__ == "__main__":
    print("🚀 Starting A2A RAG Entrypoint Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)