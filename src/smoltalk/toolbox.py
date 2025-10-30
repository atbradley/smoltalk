import inspect
import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, Union

from openai import OpenAI

from smoltalk.models import ChatMessage


class Toolbox:
    def __init__(
        self,
        tools: Union[Type[object], object],
        client: OpenAI,
        model: str,
        system_prompt: str|None = None,
        fail_on_tool_error: bool = False,
    ):
        self.logger = logging.getLogger(__name__)
        self.tools = tools
        self.client = client
        self.model = model
        self.fail_on_tool_error = fail_on_tool_error

        if not system_prompt:
            self.logger.warning("No system prompt provided. Was this deliberate?")

        self.system_prompt = system_prompt

        self.tool_signatures = self._generate_tool_signatures()

    def get_response_stream(
        self, messages: List[ChatMessage], auto_tool_call=True, fail_on_tool_error=None
    ):
        """
        Stream responses from OpenAI API. Yields chunks in the OpenAI streaming format.
        Note: This is a synchronous generator even though it's used in async contexts.
        The OpenAI client's streaming is synchronous, so this matches that behavior.
        
        Parameters
        ----------
        messages : List[ChatMessage]
            The conversation history
        auto_tool_call : bool
            Whether to automatically execute tool calls (not supported in streaming mode)
        fail_on_tool_error : bool
            Whether to fail on tool errors
            
        Yields
        ------
        dict
            Response chunks in OpenAI format
        """
        if fail_on_tool_error is None:
            fail_on_tool_error = self.fail_on_tool_error

        # Remove any existing system messages
        for n in range(len(messages)):
            if messages[n].role in ["system", "developer"]:
                messages.pop(n)
                break

        # Add system prompt if configured
        if self.system_prompt:
            messages.insert(0, ChatMessage(role="system", content=self.system_prompt))

        self.logger.debug("Getting a streaming response from the model")
        for m in messages:
            self.logger.debug("message: %s" % (m.dict(exclude_unset=True),))

        # Create streaming request
        start_time = time.perf_counter()
        end_time = start_time  # Initialize end_time
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[m.dict(exclude_unset=True) for m in messages],
            n=1,
            tools=self.tool_signatures if messages[-1].role != "tool" else None,
            tool_choice="auto" if messages[-1].role != "tool" else None,
            stream=True,
        )
        
        # Process streaming chunks
        for chunk in stream:
            end_time = time.perf_counter()
            self.logger.debug("Received chunk: %s" % str(chunk.dict()))
            
            # Convert to dict and yield
            yield chunk.dict()
        
        completed_time = time.ctime(end_time)
        self.logger.info(
            "Completed streaming response from %s at %s (after %6f seconds)"
            % (
                self.model,
                completed_time,
                end_time - start_time,
            )
        )



    async def get_response(
        self, messages: List[ChatMessage], auto_tool_call=True, fail_on_tool_error=None
    ):
        if fail_on_tool_error is None:
            fail_on_tool_error = self.fail_on_tool_error

        for n in range(len(messages)):
            if messages[n].role in [
                "system",
                "developer",
            ]:  # OpenAI calls this role "developer" now.
                messages.pop(n)
                break

        if self.system_prompt:
            messages.insert(0, ChatMessage(role="system", content=self.system_prompt))
        self.logger.debug("Getting a response from the model at %s" % (self.root_url,))
        for m in messages:
            self.logger.debug("message: %s" % (m.dict(exclude_unset=True),))
        request_body = {
            "model": self.model,
            "messages": [m.dict(exclude_unset=True) for m in messages],
            "n": 1,
        }
        if messages[-1].role != "tool":
            request_body["tools"] = self.tool_signatures
            request_body["tool_choice"] = "auto"

        self.logger.debug("request_body: %s" % (json.dumps(request_body),))
        start_time = time.perf_counter()

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[m.dict(exclude_unset=True) for m in messages],
            n=1,
            tools=self.tool_signatures if messages[-1].role != "tool" else None,
            tool_choice="auto" if messages[-1].role != "tool" else None,
        )
        end_time = time.perf_counter()
        completed_time = time.ctime(end_time)
        self.logger.info(
            "Received response from %s at %s (after %6f seconds)"
            % (
                self.model,
                completed_time,
                end_time - start_time,
            )
        )
        self.logger.debug("Response from model: %s" % str(json.dumps(resp.dict())))
        
        message = resp.choices[0].message
        messages.append(
            ChatMessage(
                role=message.role,
                content=message.content,
                tool_calls=getattr(message, "tool_calls", None),
            )
        )

        tool_calls = message.tool_calls
        if auto_tool_call and tool_calls is not None and len(tool_calls) > 0:
            # TODO: this should be async and call the tools in parallel.
            for tool_call in tool_calls:
                self.logger.debug("tool call: %s" % str(response))
                try:
                    response = await self._call_tool(tool_call)
                    self.logger.debug("tool response: %s" % str(response))
                    if fail_on_tool_error and (
                        type(response) is dict and response.get("error")
                    ):
                        self.logger.warning(
                            "Tool call failed with error: %s" % response.get("error")
                        )
                        return response
                # TODO: provide a more specific exception for tools to throw.
                except Exception as e:
                    self.logger.warning("Tool call failed with exception: %s" % str(e))
                    response = {"error": "Tool call failed with exception: %s" % str(e)}

                    if fail_on_tool_error:
                        return response

                messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(response),
                        tool_call_id=tool_call["id"],
                        name=tool_call["function"]["name"],
                    )
                )

            response = await self.get_response(messages)

        return response

    async def _call_tool(self, tool_call):
        start_time = time.perf_counter()
        self.logger.debug("_call_tool: %s" % (tool_call,))
        tool_name = tool_call["function"]["name"]
        tool_args = json.loads(tool_call["function"]["arguments"])
        self.logger.debug(
            "Calling tool '%s' with parameters '%s'" % (tool_name, tool_args)
        )
        tool = getattr(self.tools, tool_name)
        if inspect.iscoroutinefunction(tool):
            outp = await tool(**tool_args)
        else:
            outp = tool(**tool_args)
        end_time = time.perf_counter()
        completed_time = time.ctime(end_time)
        self.logger.info(
            "Tool %s returned at %s (after %6f seconds)"
            % (
                self.model,
                completed_time,
                end_time - start_time,
            )
        )
        return outp

    def _generate_tool_signatures(self):
        """
        Use litellm's function_to_dict to generate tool signatures for this toolbox.
        Called by get_response().
        """
        self.logger.debug("Generating tool signatures.")

        tools = [
            function_to_dict(func)
            for name, func in inspect.getmembers(
                self.tools, lambda x: inspect.isfunction(x) or inspect.ismethod(x)
            )
            if not name.startswith("_")
        ]
        return tools


def json_schema_type(python_type_name: str):
    """Converts standard python types to json schema types

    Parameters
    ----------
    python_type_name : str
        __name__ of type

    Returns
    -------
    str
        a standard JSON schema type, "string" if not recognized.
    """
    python_to_json_schema_types = {
        str.__name__: "string",
        int.__name__: "integer",
        float.__name__: "number",
        bool.__name__: "boolean",
        list.__name__: "array",
        dict.__name__: "object",
        "NoneType": "null",
    }

    return python_to_json_schema_types.get(python_type_name, "string")


def function_to_dict(input_function):  # noqa: C901
    """Using type hints and numpy-styled docstring,
    produce a dictionnary usable for OpenAI function calling

    Gleefully swiped from litellm. (https://github.com/BerriAI/litellm/blob/93273723cd04bd00e8bef7252e35fab184cfe910/litellm/utils.py#L4589)

    Parameters
    ----------
    input_function : function
        A function with a numpy-style docstring

    Returns
    -------
    dictionnary
        A dictionnary to add to the list passed to `functions` parameter of `litellm.completion`
    """
    # Get function name and docstring
    try:
        import inspect
        from ast import literal_eval

        from numpydoc.docscrape import NumpyDocString
    except Exception as e:
        raise e

    name = input_function.__name__
    docstring = inspect.getdoc(input_function)
    numpydoc = NumpyDocString(docstring)
    description = "\n".join([s.strip() for s in numpydoc["Summary"]])

    # Get function parameters and their types from annotations and docstring
    parameters = {}
    required_params = []
    param_info = inspect.signature(input_function).parameters

    for param_name, param in param_info.items():
        if hasattr(param, "annotation"):
            param_type = json_schema_type(param.annotation.__name__)
        else:
            param_type = None
        param_description = None
        param_enum = None

        # Try to extract param description from docstring using numpydoc
        for param_data in numpydoc["Parameters"]:
            if param_data.name == param_name:
                if hasattr(param_data, "type"):
                    # replace type from docstring rather than annotation
                    param_type = param_data.type
                    if "optional" in param_type:
                        param_type = param_type.split(",")[0]
                    elif "{" in param_type:
                        # may represent a set of acceptable values
                        # translating as enum for function calling
                        try:
                            # Vertex AI complained when this was a string.
                            param_enum = list(literal_eval(param_type))
                            param_type = "string"
                        except Exception:
                            pass
                    param_type = json_schema_type(param_type)
                param_description = "\n".join([s.strip() for s in param_data.desc])

        param_dict = {
            "type": param_type,
            "description": param_description,
            "enum": param_enum,
        }

        parameters[param_name] = dict(
            [(k, v) for k, v in param_dict.items() if isinstance(v, str)]
        )

        # Check if the parameter has no default value (i.e., it's required)
        if param.default == param.empty:
            required_params.append(param_name)

    # Create the dictionary
    result = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
            },
        },
    }

    # Add "required" key if there are required parameters
    if required_params:
        result["function"]["parameters"]["required"] = required_params

    return result
