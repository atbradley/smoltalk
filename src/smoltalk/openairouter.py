from fastapi import APIRouter, Request
import time
import os
import asyncio
from .toolbox import ChatCompletionRequest, ChatCompletionResponse

OpenAIRouter = APIRouter()
starttime = int(time.time())

@OpenAIRouter.post("/v1/chat/completions")
async def create_chat_completion(request: Request, chatRequest:ChatCompletionRequest):
    # Your implementation logic here
    print(type(request), request)
    messages = chatRequest.messages
    n = chatRequest.n

    tasks = [request.app.toolbox.get_response(chatRequest.messages) for _ in range(n)]
    msgs = await asyncio.gather(*tasks)
    outp = msgs[0]
    outp.choices = [msg['choices'][0] for msg in msgs]
    return outp

# Optional but useful for compatibility
@OpenAIRouter.get("/v1/models")
async def list_models():
    model_id = os.getenv('LLM_MODEL', 'smoltalk')
    owned_by = os.getenv('MODEL_OWNER', 'your-organization')
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": owned_by
            }
        ]
    }
