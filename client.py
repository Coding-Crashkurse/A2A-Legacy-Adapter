# client.py
import asyncio
import httpx
import json
from uuid import uuid4
import traceback

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    Message, MessageSendParams, SendMessageRequest,
    GetTaskRequest, Task, TaskState, TextPart, Role, DataPart
)

BASE_URL = "http://localhost:8000"
POLLING_INTERVAL_SECONDS = 0.3

def get_text_from_message(message: Message | None) -> str:
    if not message or not message.parts: return ""
    for part_wrapper in message.parts:
        actual_part = getattr(part_wrapper, "root", part_wrapper)
        if isinstance(actual_part, TextPart):
            return actual_part.text
    return ""

async def send_and_poll(client: A2AClient, user_message: Message) -> Task:
    """Sends a single message and polls until the task is complete."""
    request = SendMessageRequest(params=MessageSendParams(message=user_message), id=f"request-{uuid4().hex}")

    print(f'▶️  Sending: "{get_text_from_message(user_message)}"...\n')
    # The server immediately returns a Task object.
    initial_response = await client.send_message(request)
    task: Task = initial_response.root.result

    # --- CHANGE IS HERE: Display the initial status immediately! ---
    # This is the feedback the user wants to see right away.
    print(f"👍 Task successfully started with ID: {task.id}")
    initial_status_text = get_text_from_message(task.status.message)
    print(f"⏳ INITIAL STATUS | Status: {task.status.state.value.upper()} | Message: {initial_status_text}")
    # --- END OF CHANGE ---

    terminal_states = [TaskState.completed, TaskState.failed, TaskState.canceled, TaskState.rejected]
    while task.status.state not in terminal_states:
        await asyncio.sleep(POLLING_INTERVAL_SECONDS)
        print(f"🔄 Polling for status of task {task.id}...")
        task_response = await client.get_task(
            GetTaskRequest(params={"id": task.id}, id=f"request-{uuid4().hex}")
        )
        task = task_response.root.result
        status_text = get_text_from_message(task.status.message)
        print(f"⏳ UPDATE | Status: {task.status.state.value.upper()} | Message: {status_text}")

    return task

def print_task_results(task: Task):
    """Prints the final message and any artifacts from a completed task."""
    final_message = get_text_from_message(task.status.message)
    print("\n" + "="*50)
    print(f"🤖 Agent response: '{final_message}'")

    if task.artifacts:
        print("\n📄 Artifacts:")
        for artifact in task.artifacts:
            for part_wrapper in artifact.parts:
                part = getattr(part_wrapper, 'root', part_wrapper)
                if isinstance(part, DataPart):
                    print(json.dumps(part.data, indent=2))
    print("="*50 + "\n")

# The main function remains the same as the multi-turn version.
async def main():
    print(f"➡️  Connecting to A2A Agent at {BASE_URL}...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

            print(f"✅ Connected to Agent: '{agent_card.name}'. Type 'exit' to quit.")
            
            current_task_id = None
            
            while True:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("👋 Goodbye!")
                    break
                
                user_message = Message(
                    role=Role.user,
                    parts=[TextPart(text=user_input)],
                    messageId=f"msg-{uuid4().hex}",
                    taskId=current_task_id
                )
                
                completed_task = await send_and_poll(client, user_message)

                if not current_task_id:
                    current_task_id = completed_task.id
                
                print_task_results(completed_task)

    except Exception as e:
        print(f"\n🚨 An unexpected error occurred: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())