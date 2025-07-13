# a2a_server.py (Stateless Version)

import uvicorn
import httpx
from fastapi import FastAPI
from uuid import uuid4

# LangChain Imports

# A2A Core imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    Message,
    Role,
    TextPart,
    DataPart,
)

ADAPTER_SERVICE_URL = "http://localhost:8001/forward"


def get_text_from_a2a_message(message: Message | None) -> str:
    if not message or not message.parts:
        return ""
    for part_wrapper in message.parts:
        actual_part = getattr(part_wrapper, "root", part_wrapper)
        if isinstance(actual_part, TextPart):
            return actual_part.text
    return ""


class RAGProxyExecutor(AgentExecutor):
    """Dieser Executor ruft direkt den Adapter auf und wartet auf die Antwort."""

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        try:
            if not context.message:
                raise ValueError("Executor received no message.")

            print("⚙️  [A2A Server] Forwarding message to adapter...")

            async with httpx.AsyncClient() as client:
                # Wir schicken die A2A-Message direkt an den Adapter
                adapter_request = {
                    "jsonrpc": "2.0",
                    "method": "process_and_forward",
                    "params": {"message": context.message.model_dump(mode="json")},
                    "id": str(uuid4()),
                }
                response = await client.post(
                    ADAPTER_SERVICE_URL, json=adapter_request, timeout=90.0
                )
                response.raise_for_status()
                adapter_response = response.json()

            if adapter_response.get("error"):
                raise Exception(
                    f"Adapter or downstream error: {adapter_response['error']}"
                )

            # Antwort vom Adapter verarbeiten und eine A2A-Message erstellen
            result_payload = adapter_response.get("result", {})
            answer_text = result_payload.get("answer", "Error: No answer from bot.")
            source_documents = result_payload.get("documents", [])

            parts = [TextPart(text=answer_text)]
            if source_documents:
                parts.append(DataPart(data={"sources": source_documents}))

            final_message = Message(
                messageId=f"a2a-response-{uuid4().hex}", role=Role.agent, parts=parts
            )
            await event_queue.enqueue_event(final_message)

        except Exception as e:
            print(f"❌ [A2A Server] Error: {e}")
            error_message = Message(
                messageId=f"a2a-error-{uuid4().hex}",
                role=Role.agent,
                parts=[TextPart(text=str(e))],
            )
            await event_queue.enqueue_event(error_message)
        finally:
            await event_queue.close()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass


def build_app() -> FastAPI:
    skill = AgentSkill(
        id="rag_chat",
        name="RAG Chatbot",
        description="Answers questions about Bella Vista restaurant.",
        tags=["rag", "chat"],
    )
    card = AgentCard(
        name="A2A Stateless RAG Agent",
        description=skill.description,
        url="http://localhost:8000/",
        version="4.0",
        defaultInputModes=["text/plain"],
        defaultOutputModes=["application/json"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    agent_executor = RAGProxyExecutor()
    handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    a2a_app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()

    api = FastAPI(title="A2A Stateless RAG Server")
    api.mount("/", a2a_app)
    return api


app = build_app()

if __name__ == "__main__":
    print("🚀 Starting A2A Stateless RAG Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
