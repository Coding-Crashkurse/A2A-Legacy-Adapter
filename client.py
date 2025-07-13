# client.py (Stateless Version)

import asyncio
import httpx
import json
from uuid import uuid4
import traceback

from a2a.client import A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    SendMessageRequest,
    TextPart,
    Role,
    JSONRPCErrorResponse,
    DataPart,
)

BASE_URL = "http://localhost:8000"


def get_text_from_message(message: Message | None) -> str:
    if not message or not message.parts:
        return ""
    for part_wrapper in message.parts:
        actual_part = getattr(part_wrapper, "root", part_wrapper)
        if isinstance(actual_part, TextPart):
            return actual_part.text
    return ""


def print_final_message(final_message: Message):
    text = get_text_from_message(final_message)
    print("\n" + "=" * 50)
    print(f"🤖 Agent response: '{text}'")

    for part_wrapper in final_message.parts:
        part = getattr(part_wrapper, "root", part_wrapper)
        if isinstance(part, DataPart):
            print("\n📄 Sources:")
            print(json.dumps(part.data, indent=2))

    print("=" * 50 + "\n")


async def main():
    print(f"➡️  Connecting to A2A Agent at {BASE_URL}...")
    try:
        async with httpx.AsyncClient(timeout=90.0) as http_client:
            client = await A2AClient.get_client_from_agent_card_url(
                http_client, base_url=BASE_URL
            )
            print(
                f"✅ Connected to Agent: '{client.agent_card.name}'. Type 'exit' to quit."
            )

            while True:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                user_message = Message(
                    role=Role.user,
                    parts=[TextPart(text=user_input)],
                    messageId=f"msg-{uuid4().hex}",
                )
                request = SendMessageRequest(
                    params=MessageSendParams(message=user_message),
                    id=f"request-{uuid4().hex}",
                )

                print(f'▶️  Sending: "{user_input}" and waiting for direct response...')
                response = await client.send_message(request)

                if isinstance(response.root, JSONRPCErrorResponse):
                    print(f"🚨 An error occurred: {response.root.error.message}")
                    continue

                result_message = response.root.result
                if not isinstance(result_message, Message):
                    print(
                        f"🚨 Unexpected result type: {type(result_message)}. Expected a Message."
                    )
                    continue

                print_final_message(result_message)

    except Exception as e:
        print(f"\n🚨 An unexpected error occurred: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
