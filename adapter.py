import uvicorn
import httpx
from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel
from typing import Generic, TypeVar, Literal

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


LEGACY_BOT_URL = "http://localhost:8002/jsonrpc"
app = FastAPI(title="Adapter Service (JSON-RPC)")


def a2a_task_to_message_history(task_data: dict) -> list[dict]:
    messages = []
    if task_data.get("history"):
        for a2a_msg in task_data["history"]:
            text_content = ""
            for part in a2a_msg.get("parts", []):
                if part.get("kind") == "text":
                    text_content = part.get("text", "")
                    break
            if not text_content:
                continue

            role = a2a_msg.get("role")
            if role == "user":
                messages.append({"role": "human", "content": text_content})
            elif role == "agent":
                messages.append({"role": "ai", "content": text_content})

    return messages


@app.post("/jsonrpc", response_model=JSONRPCResponse)
async def jsonrpc_handler(rpc: JSONRPCRequest = Depends(get_jsonrpc_request)):
    if rpc.method == "process_and_forward":
        print("🔄 [Adapter] 'process_and_forward' called.")
        try:
            message_history = a2a_task_to_message_history(rpc.params["task"])
            async with httpx.AsyncClient() as client:
                bot_request = JSONRPCRequest(
                    method="invoke_rag_graph",
                    params={"messages": message_history},
                    id=rpc.id,
                ).model_dump(exclude_none=True)

                print(f"🔄 [Adapter] Forwarding request to RAG Bot: {LEGACY_BOT_URL}")
                response = await client.post(
                    LEGACY_BOT_URL, json=bot_request, timeout=60.0
                )
                response.raise_for_status()
                bot_response_data = response.json()

                if bot_response_data.get("error"):
                    raise Exception(f"RAG bot error: {bot_response_data['error']}")

                final_result = bot_response_data.get("result")
                return JSONRPCResponse(id=rpc.id, result=final_result)

        except Exception as e:
            print(f"❌ [Adapter] Error: {e}")
            return JSONRPCResponse(id=rpc.id, error={"code": -32001, "message": str(e)})

    return JSONRPCResponse(
        id=rpc.id, error={"code": -32601, "message": "Method not found"}
    )


if __name__ == "__main__":
    print("Starting Adapter Service on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
