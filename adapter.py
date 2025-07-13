# adapter_service.py (Simplified)

import uvicorn
import httpx
from fastapi import FastAPI
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


LEGACY_BOT_URL = "http://localhost:8002/invoke"
app = FastAPI(title="Adapter Service (Stateless)")


def get_text_from_a2a_message(a2a_msg: dict) -> str:
    """Extracts text from a single A2A message dictionary."""
    for part in a2a_msg.get("parts", []):
        if part.get("kind") == "text":
            return part.get("text", "")
    return ""


@app.post("/forward", response_model=JSONRPCResponse)
async def forward_handler(rpc: JSONRPCRequest):
    if rpc.method == "process_and_forward":
        print("🔄 [Adapter] 'process_and_forward' called.")
        try:
            query = get_text_from_a2a_message(rpc.params["message"])
            if not query:
                raise ValueError("No text content found in A2A message.")

            async with httpx.AsyncClient() as client:
                bot_request = JSONRPCRequest(
                    method="invoke_rag",
                    params={"query": query},
                    id=rpc.id,
                ).model_dump(exclude_none=True)

                print(f"🔄 [Adapter] Forwarding to RAG Bot: {LEGACY_BOT_URL}")
                response = await client.post(
                    LEGACY_BOT_URL, json=bot_request, timeout=60.0
                )
                response.raise_for_status()
                bot_response_data = response.json()

                if bot_response_data.get("error"):
                    raise Exception(f"RAG bot error: {bot_response_data['error']}")

                return JSONRPCResponse(
                    id=rpc.id, result=bot_response_data.get("result")
                )
        except Exception as e:
            print(f"❌ [Adapter] Error: {e}")
            return JSONRPCResponse(id=rpc.id, error={"code": -32001, "message": str(e)})

    return JSONRPCResponse(
        id=rpc.id, error={"code": -32601, "message": "Method not found"}
    )


if __name__ == "__main__":
    print("🚀 Starting Adapter Service on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
