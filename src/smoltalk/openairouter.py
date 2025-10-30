import asyncio
import json
import os
import time
from typing import List


from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from smoltalk.models import ChatCompletionRequest, ChatMessage

OpenAIRouter = APIRouter()
starttime = int(time.time())
@OpenAIRouter.get("/api/forward-sse")
async def forward_sse():
    ...#return EventSourceResponse(stream_from_third_party_sse())

@OpenAIRouter.post("/v1/chat/completions")
async def create_chat_completion(request: Request, chatRequest: ChatCompletionRequest):
    n = chatRequest.n or 1

    # Handle streaming requests
    if chatRequest.stream:
        async def stream_generator():
            """Generate SSE-formatted streaming responses"""
            try:
                # get_response_stream is a synchronous generator, so we iterate normally
                for chunk in request.app.toolbox.get_response_stream(chatRequest.messages):
                    # Format as SSE data
                    chunk_json = json.dumps(chunk)
                    yield f"data: {chunk_json}\n\n"
                
                # Send the final [DONE] message
                yield "data: [DONE]\n\n"
            except Exception as e:
                # Send error in SSE format
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    # Handle non-streaming requests (original behavior)
    tasks = [request.app.toolbox.get_response(chatRequest.messages) for _ in range(n)]
    msgs = await asyncio.gather(*tasks)
    outp = msgs[0]
    outp["choices"] = [msg["choices"][0] for msg in msgs]
    return outp


@OpenAIRouter.get("/chat")
async def chat(request: Request, msgs: List[ChatMessage]):
    toolbox = request.app.toolbox
    toolbox.logger.info("starting chat.")

    toolbox.logger.debug("tool sigs: " + str(toolbox.tool_signatures))

    msgs = [ChatMessage(**msg) for msg in msgs]

    response = await toolbox.get_response(msgs)

    toolbox.logger.info("Chat response: " + str(response))

    if error := response.get("error", False):
        raise HTTPException(status_code=500, detail=error)

    toolbox.logger.info("Chat response: " + str(response))

    resp = request.json
    return resp[-1]


@OpenAIRouter.get("/v1/models")
async def list_models():
    model_id = os.getenv("LLM_MODEL", "smoltalk")
    owned_by = os.getenv("MODEL_OWNER", "your-organization")
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": owned_by,
            }
        ],
    }
