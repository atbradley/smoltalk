import inspect
import json
import logging
from typing import Type, Union
import os.path
import httpx

logger = logging.getLogger(__name__)


class Toolbox():
    def __init__(
        self, tools: Union[Type[object], object], root_url: str, model: str, api_key: str='no-key-needed', 
        system_prompt: str = None, fail_on_tool_error: bool = False
    ):
        self.tools = tools
        self.root_url = root_url
        self.model = model
        self.api_key = api_key
        self.fail_on_tool_error = fail_on_tool_error

        here = os.path.dirname(__file__)

        if not system_prompt:            
            logger.warning("No system prompt provided. Was this deliberate?")

        self.system_prompt = system_prompt

        self.tool_signatures = self._generate_tool_signatures()

    async def get_response(self, messages, auto_tool_call=True, fail_on_tool_error=None):
        if fail_on_tool_error is None:
            fail_on_tool_error = self.fail_on_tool_error
        if self.system_prompt:        
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        logger.debug("Getting a response from the model at %s" % (self.root_url,))
        request_body = {
            "model": self.model,
            "messages": messages,
            "tools": self.tool_signatures,
            "tool_choice": "auto",
            "n": 1,
        }
        logger.debug('request_body', json.dumps(request_body))
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.root_url}chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_body,
                follow_redirects=True,
                timeout=15,
            )
        response = response.json()
        print(response)
        logger.debug("Response from model: %s" % (str(response),))
        messages.append(response['choices'][0]['message'])
        
        if auto_tool_call and response['choices'][0]['message'].get('tool_calls', []):
            # TODO: this should be async and call the tools in parallel.
            for tool_call in response['choices'][0]['message'].get('tool_calls'):
                try:
                    response = await self._call_tool(tool_call)
                    logger.debug("tool response:", response)
                    if fail_on_tool_error and response.get('error'):
                        logger.warning("Tool call failed with error: %s" % (response['error'],))
                        return response
                #TODO: provide a more specific exception for tools to throw.
                except Exception as e:
                    logger.warning("Tool call failed with exception: %s" % (str(e),))
                    response = {"error": "Tool call failed with exception: %s" % (str(e),)}
                    if fail_on_tool_error:
                        return response
                messages.append({"role": "tool", "content": json.dumps(response), "tool_call_id": tool_call['id']})
                response = await self.get_response(messages)

        return response

    async def _call_tool(self, tool_call):
        logger.debug("_call_tool: %s" % (tool_call,))
        tool_name = tool_call['function']['name']
        tool_args = json.loads(tool_call['function']['arguments'])
        logger.debug("Calling tool '%s' with parameters '%s'" % (tool_name, tool_args))
        tool = getattr(self.tools, tool_name)
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
        "type": "function", 
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
            },
        }
    }

    # Add "required" key if there are required parameters
    if required_params:
        result["function"]["parameters"]["required"] = required_params

    return result
