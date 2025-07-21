# Smoltalk

Smoltalk is a small, simple "microframework" for creating agentic AI applications.

## Getting Started

1. Implement your tools as methods on a plain Python class. They can be either class or instance methods.
2. Import `smoltalk.Toolbox` and create a new `Toolbox` instance: 

```
from smoltalk import Toolbox

toolbox = Toolbox(My_Tools, root_url=LLM_BASE_URL, model=LLM_MODEL,
                  api_key=LLM_API_KEY, system_prompt=system_prompt, fail_on_tool_error=True)
```

3. Call `Toolbox.get_response` to retrieve a response from the agent. `get_response` accepts a list of messages identical to the `messages` parameter of [Open AI's `chat/completions` API call](https://platform.openai.com/docs/api-reference/chat/create):

```
response = await toolbox.get_response([
    {
        "role": "user",
        "content": "What is the current weather in Boston, MA today?"
    }
])
```

The response is the Open AI-compatible response from the API defined in the `Toolbox`'s `root_url` parameter. 