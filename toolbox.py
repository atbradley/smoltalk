import inspect
import json
import logging
from typing import Type, Union
import os.path
import httpx

logger = logging.getLogger(__name__)


class Toolbox():
    def __init__(
        self, tools: Union[Type[object], object], root_url: str, model: str, api_key: str='no-key-needed', system_prompt: str = None
    ):
        self.tools = tools
        self.root_url = root_url
        self.model = model
        self.api_key = api_key

        here = os.path.dirname(__file__)

        if not system_prompt:            
            logger.warning("No system prompt provided. Was this deliberate?")

        self.system_prompt = system_prompt

    async def get_response(self, messages, auto_tool_call=True, fail_on_tool_error=True):
        if self.system_prompt:        
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        logger.debug("Getting a response from the model at %s" % (self.root_url,))
        print(self.root_url, f"{self.root_url}chat/completions",)
        request_body = {
            "model": self.model,
            "messages": messages,
            "tools": self._generate_tool_signatures(),
            "tool_choice": "auto",
            "n": 1,
        }
        print('request_body', request_body)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.root_url}chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_body,
                follow_redirects=True,
            )
        print(response)
        response = response.json()
        logger.debug("Response from model: %s" % (str(response),))
        print("Response from model: %s" % (str(response),))

        if auto_tool_call:
            # TODO: this should be async and call the tools in parallel.
            for tool_call in response['choices'][0]['message'].get('tool_calls', []):
                print(type(tool_call))
                response = await self._call_tool(tool_call)
                logger.debug("TOOL RESPONSE:", response)
                messages.append({"role": "assistant", "content": response})
                response = await self.get_response(messages)
                response = response.json()

        print(response)
        return response

    async def _call_tool(cls, tool_call):
        logger.debug("_call_tool: %s" % (tool_call,))
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        logger.debug("Calling tool '%s' with parameters '%s'" % (tool_name, tool_args))
        tool = getattr(cls.Tools, tool_name)
        if inspect.iscoroutinefunction(tool):
            return await tool(**tool_args)
        else:
            return tool(**tool_args)

    def _generate_tool_signatures(self):
        """
        Use litellm's function_to_dict to generate tool signatures for this toolbox.
        Called by get_response().
        """
        logger.debug("Generating tool signatures.")
        tools = [
            function_to_dict(func)
            for name, func in inspect.getmembers(
                self.tools, inspect.isfunction
            )
            if not name.startswith("_") #Better way to do this? Nested class, decorator?
        ]
        return tools


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
                            param_enum = str(list(literal_eval(param_type)))
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
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": parameters,
        },
    }

    # Add "required" key if there are required parameters
    if required_params:
        result["parameters"]["required"] = required_params

    return result