import json
import re
from typing import List, Optional, Dict, Any, Type, Union

import httpx
import requests
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model, Field

from hippycampus.spec_parser import OperationObject, MediaTypeObject, SchemaObject, SchemaType, ParameterObject
from hippycampus.spec_parser import parse_yaml

__all__ = ["OpenAPIToolBuilder", 'load_tools_from_openapi', 'create_input_schema_from_json_schema']

REQUEST_LOCATIONS = ("path", "query", "header", "cookie")


def sanitize_tool_name(name: str) -> str:
    """
    Sanitize a tool name so that it can be used as a valid Python class name.
    Replace any non-alphanumeric characters with underscores and ensure the name
    doesn't start with a digit.
    """
    sanitized = re.sub(r'\W+', '_', name)
    if re.match(r'^\d', sanitized):
        sanitized = '_' + sanitized
    return sanitized


class OpenAPIToolBuilder:
    """
    A builder class to generate StructuredTool instances from an OpenAPI specification.
    """

    def __init__(self, spec_str: str):
        """
        Initialize the builder with the OpenAPI specification.
        :param spec_str: A string containing the OpenAPI specification in YAML or JSON format.
        """
        self.spec = parse_yaml(spec_str)

    def openapi_type_to_python_type(self, openapi_type: str, schema: Dict[str, Any] = None) -> Any:
        """
        Map OpenAPI types to Python types.

        Args:
            openapi_type: The OpenAPI type string
            schema: The complete schema object, needed for handling arrays with items
        """
        # Handle enum first if present in schema
        if schema and "enum" in schema and schema['enum']:
            from enum import Enum
            # Create a dynamic enum class
            enum_name = schema.get("title") or "DynamicEnum"
            enum_dict = {str(val).upper(): val for val in schema["enum"]}
            return Enum(enum_name, enum_dict)

        # If openapi_type is a list, filter out "null" (if present) and use the first remaining type.
        if isinstance(openapi_type, list):
            types = [t for t in openapi_type if t != "null"]
            # Fallback to "string" if no valid type is found.
            openapi_type = types[0] if types else "string"

        # Basic type mapping
        mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }

        # Special handling for arrays with items property
        if openapi_type == "array" and schema and "items" in schema:
            items = schema["items"]
            if isinstance(items, dict):
                if "type" in items:
                    # Check for enum in array items
                    if "enum" in items:
                        enum_type = self.openapi_type_to_python_type(items["type"], items)
                        return List[enum_type]
                    # Explicit type for array items
                    item_type = self.openapi_type_to_python_type(items["type"], items)
                    return List[item_type]
                elif "oneOf" in items:
                    # Handle oneOf for item types
                    return List[Any]

        return mapping.get(openapi_type, Any)

    def requires_authentication(self) -> bool:
        """
        Check if the OpenAPI spec indicates that authentication is required.
        """
        if self.spec.get("security"):
            return True
        if self.spec.get("components", {}).get("securitySchemes"):
            return True
        return False

    def _create_args_model(
            self,
            operation: OperationObject,
            model_name: str = "ToolArgs",
            placeholder: Optional[str] = "input",
    ) -> Type[BaseModel]:
        """
        Create a Pydantic model for the tool's input arguments based on the operation.
        It processes both parameters and JSON request body if present.
        If neither is defined (i.e. zero declared inputs), a placeholder field is added.
        """
        fields = {}
        has_inputs = False

        # Process parameters if present
        parameters = operation.parameters or []
        if parameters:
            for param in parameters:
                param_name = param.name
                # Skip parameters that don't have a valid name.
                if not param_name or not isinstance(param_name, str):
                    continue

                required = param.required or False
                schema = param.schema_ or SchemaObject(type=SchemaType.STRING)

                field_type = self.openapi_type_to_python_type(schema.dict().get("type", "string"), schema.dict())
                default = ... if required else None
                fields[param_name] = (field_type, default)
                has_inputs = True

        # Check for a JSON request body and add its fields
        if operation.requestBody:
            content = operation.requestBody.content or {}
            if "application/json" in content:
                schema = content["application/json"].schema_ or SchemaObject(properties={})
                # Process the schema properties
                properties = schema.properties or {}
                required_fields = schema.required or []

                for prop, prop_schema in properties.items():
                    # Skip if this property name conflicts with a parameter
                    if prop in fields:
                        continue

                    if isinstance(prop_schema, SchemaObject):
                        openapi_type = str(prop_schema.type)
                        py_type = self.openapi_type_to_python_type(openapi_type, prop_schema.dict())

                        # Handle oneOf case
                        if prop_schema.oneOf:
                            py_type = Any  # Use Any for oneOf types for simplicity
                    else:
                        # If prop_schema isn't a dict, infer its type directly
                        py_type = type(prop_schema)

                    if prop in required_fields:
                        fields[prop] = (py_type, ...)
                    else:
                        fields[prop] = (Optional[py_type], None)
                    has_inputs = True

        # If no inputs were found, add a placeholder field
        if not has_inputs and placeholder:
            fields[placeholder] = (Optional[str], None)

        return create_model(model_name, **fields)

    def _create_tool_class(
            self,
            tool_name: str,
            http_method: str,
            base_url: str,
            path: str,
            description: str,
            operation: OperationObject,
            token: str = None,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> Type[BaseTool]:
        """
        Dynamically create a subclass of BaseTool for a given OpenAPI operation.
        The generated tool's `_run` method makes an HTTP request using the specified method.
        This implementation automatically distributes keys from the flat JSON input into the appropriate
        parameter locations (path, query, header, cookie) based on the operation's parameter definitions.
        Any keys that don't match a parameter name are used as the request body.
        """
        sanitized_name = sanitize_tool_name(tool_name)
        parameters = operation.parameters or []

        # Build enhanced description
        enhanced_description = [description or f"Tool for {http_method.upper()} {path}"]

        # Join all description parts
        final_description = "\n".join(enhanced_description)

        # Create an on-the-fly Pydantic model for the tool's input arguments to give to the LLM.
        args_model = self._create_args_model(operation, model_name=f"{tool_name}Args", placeholder="input")

        def _run(inner_self, tool_input: Optional[str] = None, **kwargs) -> Any:
            # Synchronous version using requests
            # If keyword arguments are passed, use them as input.
            request_kwargs, url = self.convert_tool_params_request_params(parameters, http_method, token, base_url,
                                                                          path, tool_input, kwargs)
            response = requests.request(http_method.upper(), url, **request_kwargs)
            try:
                return json.dumps(response.json())
            except Exception:
                return response.text

        async def _arun(inner_self, tool_input: Optional[str] = None, **kwargs) -> Any:
            # Asynchronous version using httpx
            request_kwargs, url = self.convert_tool_params_request_params(parameters, http_method, token, base_url,
                                                                          path, tool_input, kwargs)
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.request(http_method.upper(), url, **request_kwargs)
                    return json.dumps(response.json())
                except Exception:
                    return str(response.text)

        def _parse_input(inner_self, tool_input: str, tool_call_id: Optional[str] = None) -> Dict[str, Any]:
            return self._parse_input_helper(tool_input, tool_name, sanitized_name)

        # Sanitize the tool name to ensure it's a valid Python identifier.
        sanitized_name = sanitize_tool_name(tool_name)
        tool_attrs: Dict[str, Any] = {
            "name": sanitized_name,
            "description": final_description,
            "_run": _run,
            "_arun": _arun,
            "_parse_input": _parse_input,  # our custom input parser
            "__module__": __name__,
            "args_schema": args_model,
            "__annotations__": {
                "name": str,
                "description": str,
                "args_schema": Type[BaseModel],
                "metadata": Optional[Dict[str, Any]],
            },
            "metadata": metadata,
        }

        return type(sanitized_name, (StructuredTool,), tool_attrs)

    def convert_tool_params_request_params(self, __parameters: List[ParameterObject], http_method: str, token: str,
                                           __base_url: str, __path: str, tool_input: Optional[str] = None, kwargs=None):
        # If keyword arguments are passed, use them as input.
        if kwargs:
            input_data = kwargs
        else:
            if tool_input is None:
                tool_input = ""
            try:
                input_data = json.loads(tool_input)
            except json.JSONDecodeError:
                input_data = {"body": tool_input}
            if not isinstance(input_data, dict):
                input_data = {"body": input_data}

        # Initialize dictionaries for each location.
        request_data: Dict[str, Dict[str, Any]] = {key: {} for key in REQUEST_LOCATIONS}

        # Keys not matching any parameter name will be used as body data.
        remaining_data: Dict[str, Any] = {}
        # Parameter name -> request location (query, path, etc.)
        param_locations: Dict[str, str] = {}

        for param in __parameters:
            if param.in_ in REQUEST_LOCATIONS:
                param_locations[param.name] = param.in_

        # Distribute input keys into parameter locations.
        for key, value in input_data.items():
            loc = param_locations[key] if key in param_locations else None
            if loc:
                request_data[loc][key] = value
            else:
                remaining_data[key] = value

        # Determine requestBody (if defined in the spec) or use remaining keys.
        body_data = remaining_data if remaining_data else None
        # Build the URL and substitute path parameters.
        url = __base_url.rstrip("/") + "/" + __path.lstrip("/")
        url = self._replace_url_path_variables(__parameters, request_data, url)

        # Prepare keyword arguments for the HTTP request.
        request_kwargs: Dict[str, Any] = {}
        for loc in REQUEST_LOCATIONS:
            if loc == "path":
                continue
            if request_data[loc]:
                request_kwargs[loc] = request_data[loc]

        if body_data is not None:
            request_kwargs["json"] = body_data
        if token is not None:
            # Check security schemes in the OpenAPI spec
            security_schemes = self.spec.components and self.spec.components.securitySchemes or {}
            # Default to Bearer if no security schemes are specified
            auth_type = "bearer"

            for scheme in security_schemes.values():
                if scheme.type == "http":
                    auth_type = (scheme.scheme or "bearer").lower()

            if auth_type == "basic":
                # For Basic Auth, use cromwellian@gmail.com as username
                import base64
                auth_string = base64.b64encode(
                    f"{token}".encode()
                ).decode()
                request_kwargs["headers"] = {"Authorization": f"Basic {auth_string}"}
            else:
                # Default to Bearer token
                request_kwargs["headers"] = {"Authorization": f"Bearer {token}"}
        return request_kwargs, url

    def _replace_url_path_variables(self, __parameters, request_data, url):
        for param in __parameters:
            if param.in_ == "path":
                pname = param.name
                if pname in request_data["path"]:
                    url = url.replace("{" + pname + "}", str(request_data["path"][pname]))
        return url

    @staticmethod
    def _parse_input_helper(tool_input, tool_name, sanitized_name):
        if isinstance(tool_input, str):
            try:
                parsed = json.loads(tool_input)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    return {"input": parsed}
            except Exception:
                try:
                    import ast
                    parsed = ast.literal_eval(tool_input)
                    if isinstance(parsed, dict):
                        return parsed
                    else:
                        return {"input": parsed}
                except Exception:
                    return {"input": tool_input}
        elif isinstance(tool_input, dict):
            return tool_input
        else:
            return {"input": tool_input}

    @staticmethod
    def extract_examples(content: Optional[Dict[str, MediaTypeObject]]) -> dict:
        """Extract and resolve examples from content object."""
        if not content or "application/json" not in content:
            return {}

        json_content = content["application/json"]
        examples = {}

        if json_content.examples is not None:
            for example_name, example_obj in json_content.examples.items():
                examples[example_name] = example_obj.value
        elif json_content.example is not None:
            examples["default"] = {
                "value": json_content.example
            }

        # Also check schema examples if no explicit examples found
        elif json_content.schema_ is not None:
            if json_content.schema_.example is not None:
                examples["default"] = {
                    "value": json_content.schema_["example"]
                }
            elif json_content.examples:
                examples["default"] = {
                    "value": json_content.examples['default']
                }

        return examples

    def build_tools(self, token: str = None) -> List[BaseTool]:
        """Build and return a list of tool instances corresponding to each operation in the OpenAPI spec."""
        tools: List[BaseTool] = []
        base_url = self.spec.servers[0].url if self.spec.servers is not None and len(self.spec.servers) > 0 else ""
        paths = self.spec.paths
        methods = ['get', 'put', 'delete', 'options', 'head', 'patch', 'trace']
        for path, path_item in paths.items():
            for method in methods:
                operation: Optional[OperationObject] = getattr(path_item, method, None)
                if not operation:
                    continue
                operation_id = self._extract_operationid_or_use_path(method, operation, path)
                summary = operation.summary or ""
                description = operation.description or summary

                metadata = {}
                metadata['externalDocs'] = self._external_external_docs(operation)
                metadata['requestExamples'] = self.extract_examples(
                    operation.requestBody and operation.requestBody.content)
                metadata['responseExamples'] = {status_code: self.extract_examples(response.content) for
                                                status_code, response in operation.responses}

                tool_class = self._create_tool_class(
                    operation_id, method, base_url, path,
                    description, operation, token, metadata
                )
                tools.append(tool_class())
        # Add an authentication tool if the spec indicates auth is required.
        # if self.requires_authentication() and token is None:
        #     tools.append(AuthTool())
        return tools

    def _external_external_docs(self, operation):
        return operation.externalDocs and operation.externalDocs.url or self.spec.externalDocs and self.spec.externalDocs.url

    @staticmethod
    def _extract_operationid_or_use_path(method, operation, path):
        return operation.operationId or f"{method}_{path.replace('/', '_')}"


loaded_tools: dict[Any, list[BaseTool]] = {}


def load_tools_from_openapi(openapi: str, token: str = None, url: str = None) -> list[BaseTool]:
    if not url:
        url = openapi

    if url in loaded_tools:
        return loaded_tools[url]

    with open(openapi, "r") as f:
        spec_content = f.read()
    obuilder = OpenAPIToolBuilder(spec_content)
    otools = obuilder.build_tools(token)
    loaded_tools[url] = otools
    return otools


class AuthTool(BaseTool):
    name: str = "AuthTool"
    description: str = "Tool to handle authentication and retrieve tokens via human-in-the-loop."

    def _run(self, tool_input: str) -> str:
        # For example purposes, prompt the user for an auth token.
        token = input("Please enter your authentication token: ")
        return token

    async def _arun(self, tool_input: str) -> str:
        return self._run(tool_input)


def create_input_schema_from_json_schema(json_schema):
    """Create a Pydantic model from a JSON schema."""
    if not json_schema:
        return None

    properties = json_schema.get('properties', {})
    required = json_schema.get('required', [])

    field_definitions = {}

    # Basic type mapping
    type_mapping = {
        'string': str,
        'integer': int,
        'number': float,
        'boolean': bool,
        'array': list,
        'object': dict,
        'null': None
    }

    def resolve_type(prop_schema):
        """Recursively resolve property type including complex schemas."""
        if not isinstance(prop_schema, dict):
            return Any

        # Handle oneOf, anyOf, allOf
        if 'oneOf' in prop_schema:
            return Union[tuple(resolve_type(s) for s in prop_schema['oneOf'])]
        if 'anyOf' in prop_schema:
            return Union[tuple(resolve_type(s) for s in prop_schema['anyOf'])]
        if 'allOf' in prop_schema:
            # For allOf, we should technically merge the schemas
            # For simplicity, we'll use Dict here
            return Dict[str, Any]

        # Handle references
        if '$ref' in prop_schema:
            # Should resolve reference here
            return Any  # Placeholder - ideally should resolve the reference

        prop_type = prop_schema.get('type')

        # Handle multiple types
        if isinstance(prop_type, list):
            types = [type_mapping.get(t) for t in prop_type if t != 'null']
            if not types:
                return Any
            if len(types) == 1:
                return types[0]
            return Union[tuple(t for t in types if t is not None)]

        if prop_type == 'array':
            items = prop_schema.get('items', {})
            if not items:
                return List[Any]
            item_type = resolve_type(items)
            return List[item_type]

        if prop_type == 'object':
            if 'properties' in prop_schema:
                # Create nested model
                nested_schema = create_input_schema_from_json_schema(prop_schema)
                return nested_schema
            return Dict[str, Any]

        # Handle string formats
        if prop_type == 'string':
            format_type = prop_schema.get('format')
            if format_type == 'date-time':
                from datetime import datetime
                return datetime
            if format_type == 'date':
                from datetime import date
                return date
            if format_type == 'time':
                from datetime import time
                return time
            if format_type == 'email':
                from pydantic import EmailStr
                return EmailStr
            if format_type == 'uri':
                from pydantic import AnyUrl
                return AnyUrl
            if format_type == 'binary':
                return bytes
            # Add other format types as needed

        # Handle enums
        if 'enum' in prop_schema:
            from enum import Enum
            enum_values = prop_schema['enum']
            return Enum(f'Enum_{hash(str(enum_values))}',
                        {str(v).upper(): v for v in enum_values})

        return type_mapping.get(prop_type, Any)

    for prop_name, prop_schema in properties.items():
        description = prop_schema.get('description', '')
        default = prop_schema.get('default', ...)

        python_type = resolve_type(prop_schema)

        # Handle required fields
        if prop_name in required:
            if default is ...:
                field_definitions[prop_name] = (python_type, Field(..., description=description))
            else:
                field_definitions[prop_name] = (python_type, Field(default=default, description=description))
        else:
            field_definitions[prop_name] = (
                Optional[python_type],
                Field(default=default if default is not ... else None, description=description)
            )

    # Create a dynamic model name
    model_name = json_schema.get('title', "DynamicInputSchema")

    # Create and return the model
    return create_model(model_name, **field_definitions)
