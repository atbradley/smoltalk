import asyncio
import os
import time
from typing import List

from fastapi import APIRouter, HTTPException, Request

from .toolbox import ChatCompletionRequest, ChatMessage

OpenAIRouter = APIRouter()
starttime = int(time.time())


@OpenAIRouter.post("/v1/chat/completions")
async def create_chat_completion(request: Request, chatRequest: ChatCompletionRequest):
    n = chatRequest.n

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
