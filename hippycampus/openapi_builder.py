import json
import re
from typing import List, Optional, Dict, Any, Type, Union, Callable # Added Callable

import httpx
import requests # type: ignore
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model, Field

from hippycampus.spec_parser import (
    OpenAPIObject, 
    OperationObject, 
    MediaTypeObject, 
    SchemaObject, 
    SchemaType, 
    ParameterObject,
    PathItemObject # Added
)
from hippycampus.spec_parser import parse_yaml
from hippycampus.tool_auth.authentication import AbstractAuth # Assuming this exists for type hinting

__all__ = ["OpenAPIToolBuilder", 'load_tools_from_openapi', 'create_input_schema_from_json_schema']

REQUEST_LOCATIONS = ("path", "query", "header", "cookie")

# Type alias for JSON-like objects
JSONLike = Dict[str, Any]


def sanitize_tool_name(name: str) -> str:
    """
    Sanitize a tool name so that it can be used as a valid Python class name.
    Replace any non-alphanumeric characters with underscores and ensure the name
    doesn't start with a digit.
    """
    if not name: # Handle empty string case
        return "_"
    sanitized = re.sub(r'\W+', '_', name)
    if re.match(r'^\d', sanitized):
        sanitized = '_' + sanitized
    return sanitized


class OpenAPIToolBuilder:
    """
    A builder class to generate StructuredTool instances from an OpenAPI specification.
    """

    def __init__(self, spec: Union[str, Dict[str, Any]]):
        """
        Initialize the builder with the OpenAPI specification.
        :param spec: A string containing the OpenAPI specification in YAML or JSON format, or a pre-parsed dict.
        """
        if isinstance(spec, str):
            self.spec_obj: OpenAPIObject = parse_yaml(spec) 
        elif isinstance(spec, dict):
            # To ensure it's a valid OpenAPIObject, parse it (even if from dict)
            # This assumes parse_yaml can handle a dict by first dumping it to string if needed,
            # or that OpenAPIObject.model_validate can be used.
            # For simplicity, let's assume parse_yaml handles it or spec is always string for now.
            # If spec_parser.parse_yaml expects string, then:
            # self.spec_obj = parse_yaml(json.dumps(spec))
            # Or better, directly validate if spec_parser has such utility or use pydantic's model_validate
            self.spec_obj = OpenAPIObject.model_validate(spec)

        else:
            raise TypeError("spec must be a string (YAML/JSON) or a dictionary representing an OpenAPIObject.")

    @staticmethod
    def openapi_type_to_python_type(param_name: str, schema: SchemaObject, is_required: bool = False) -> tuple[Type, Field]: # type: ignore
        """
        Map OpenAPI SchemaObject to Python type and Pydantic Field.

        Args:
            param_name: Name of the parameter/property (for Enum naming).
            schema: The OpenAPI SchemaObject.
            is_required: Whether the field is required.

        Returns:
            A tuple containing the Python type and a Pydantic Field object.
        """
        openapi_type_val = schema.type
        schema_info = schema.model_dump(exclude_none=True) # Use Pydantic's dump for schema info

        is_nullable = schema.nullable or (isinstance(openapi_type_val, list) and "null" in openapi_type_val)
        
        processed_openapi_type: Union[str, SchemaType]
        if isinstance(openapi_type_val, list):
            types_no_null = [t for t in openapi_type_val if t != SchemaType.NULL and t != "null"]
            if not types_no_null:
                processed_openapi_type = SchemaType.STRING # Fallback, effectively Optional[Any] later
                is_nullable = True # If only "null" or empty list, it's nullable Any
            else:
                processed_openapi_type = types_no_null[0]
        else:
            processed_openapi_type = openapi_type_val or SchemaType.STRING # Default to string if type is missing

        # Handle enum
        if schema.enum:
            from enum import Enum as PyEnum # Alias to avoid conflict
            enum_title = schema.title or param_name
            enum_name = sanitize_tool_name(enum_title).capitalize() + "Enum"
            
            enum_members: Dict[str, Any] = {}
            for val in schema.enum:
                member_name = str(val)
                # Ensure member name is valid Python identifier
                if isinstance(val, str) and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', val):
                    member_name = val
                else:
                    member_name = f"MEMBER_{str(val).upper().replace('.', '_').replace('-', '_')}" # Sanitize further
                    member_name = re.sub(r'\W+', '_', member_name)
                    if member_name.startswith('_') or re.match(r'^\d', member_name.lstrip('_')):
                         member_name = f"ENUM_{member_name.lstrip('_')}" # Ensure valid start

                # Handle duplicate member names if values are different but sanitized names clash
                original_member_name = member_name
                counter = 1
                while member_name in enum_members and enum_members[member_name] != val:
                    member_name = f"{original_member_name}_{counter}"
                    counter += 1
                enum_members[member_name] = val

            try:
                created_enum = PyEnum(enum_name, enum_members)
            except TypeError: # Happens if all names are sanitized to MEMBER_X and values are distinct
                # Fallback for non-string enums where names might not be unique enough after sanitization
                enum_members_safe = {f"VALUE_{i}": v for i, v in enumerate(schema.enum)}
                created_enum = PyEnum(enum_name, enum_members_safe)


            python_type: Any = created_enum
        elif processed_openapi_type == SchemaType.ARRAY:
            items_schema = schema.items
            if items_schema and isinstance(items_schema, SchemaObject): # Ensure items is a SchemaObject
                item_py_type, _ = self.openapi_type_to_python_type(f"{param_name}_item", items_schema, True) # Item requiredness doesn't affect List type
                python_type = List[item_py_type] # type: ignore
            else:
                python_type = List[Any]
        elif processed_openapi_type == SchemaType.OBJECT:
            # For 'object' type, if properties are defined, it implies a nested model.
            # This function's role is to return the type for a single field.
            # The actual nested model creation is handled by create_input_schema_from_json_schema.
            # Here, we indicate it's a dict, or Any if no properties.
            python_type = Dict[str, Any] if schema.properties else Any
        else:
            # Basic type mapping
            mapping: Dict[Union[str, SchemaType], Type] = {
                SchemaType.STRING: str, SchemaType.INTEGER: int, SchemaType.NUMBER: float,
                SchemaType.BOOLEAN: bool,
                "string": str, "integer": int, "number": float, "boolean": bool, # str fallbacks
            }
            python_type = mapping.get(processed_openapi_type, Any)

        if is_nullable and python_type is not Any and not (hasattr(python_type, '__origin__') and python_type.__origin__ is Union):
             python_type = Optional[python_type] # type: ignore

        # Field information
        field_kwargs: Dict[str, Any] = {"description": schema.description}
        if schema.default is not None:
            field_kwargs["default"] = schema.default
        elif not is_required and not is_nullable: # Optional field without explicit default
            field_kwargs["default"] = None # Pydantic V2 needs explicit None for Optional fields
        elif is_required and schema.default is None : # Required but default is None (explicit null)
             field_kwargs["default"] = None # Allowed if type is Optional[X]
        elif is_required:
            field_kwargs["default"] = ... # Ellipsis for required fields

        return python_type, Field(**field_kwargs)


    def requires_authentication(self) -> bool:
        """
        Check if the OpenAPI spec indicates that global authentication is defined.
        """
        if self.spec_obj.security:
            return True
        if self.spec_obj.components and self.spec_obj.components.securitySchemes:
            return True
        return False

    def _create_args_model(
            self,
            operation_id: str,
            operation: OperationObject,
            placeholder_if_empty: Optional[str] = "input_payload",
    ) -> Type[BaseModel]:
        """
        Create a Pydantic model for the tool's input arguments.
        """
        fields: Dict[str, Any] = {}
        has_inputs = False
        # Sanitize operation_id for model name
        base_model_name = sanitize_tool_name(operation_id).capitalize()
        model_name = base_model_name + "Args"


        if operation.parameters:
            for param in operation.parameters:
                if not param.name or not isinstance(param.name, str):
                    continue

                param_schema_obj = param.schema_ or SchemaObject(type=SchemaType.STRING)
                
                field_name_sanitized = sanitize_tool_name(param.name)
                
                # Pass param_name for better Enum names if schema has no title
                python_type, pydantic_field = self.openapi_type_to_python_type(
                    param.name, param_schema_obj, param.required or False
                )
                
                # Add original param name to field info if sanitized
                if field_name_sanitized != param.name:
                    if pydantic_field.json_schema_extra is None:
                        pydantic_field.json_schema_extra = {}
                    if isinstance(pydantic_field.json_schema_extra, dict): # Type guard
                         pydantic_field.json_schema_extra['param_name'] = param.name


                fields[field_name_sanitized] = (python_type, pydantic_field)
                has_inputs = True

        if operation.requestBody:
            json_media_type = operation.requestBody.content.get("application/json")
            if json_media_type and json_media_type.schema_:
                body_schema_obj = json_media_type.schema_
                
                request_body_model_name = base_model_name + "RequestBody"
                
                schema_dict_for_body = body_schema_obj.model_dump(by_alias=True, exclude_none=True)
                if 'type' not in schema_dict_for_body and 'properties' in schema_dict_for_body:
                    schema_dict_for_body['type'] = 'object' # Default to object if properties exist
                
                if schema_dict_for_body.get('type') == 'object':
                    RequestBodyModel = create_input_schema_from_json_schema(request_body_model_name, schema_dict_for_body)
                    
                    field_description = operation.requestBody.description or "JSON request body"
                    field_info_args = {"description": field_description}
                    if operation.requestBody.required:
                        fields["request_body"] = (RequestBodyModel, Field(..., **field_info_args))
                    else:
                        fields["request_body"] = (Optional[RequestBodyModel], Field(default=None, **field_info_args))
                    has_inputs = True
                else: # Non-object request body (e.g. array, string)
                    # Create a simple model with a single field for the body
                    field_name = "body_payload" # Or derive from operation_id
                    # Use openapi_type_to_python_type for the body's schema
                    body_python_type, body_pydantic_field = self.openapi_type_to_python_type(
                        field_name, body_schema_obj, operation.requestBody.required or False
                    )
                    fields[field_name] = (body_python_type, body_pydantic_field)
                    has_inputs = True


        if not has_inputs and placeholder_if_empty:
            fields[placeholder_if_empty] = (Optional[str], Field(default=None, description="Placeholder for empty input"))
        
        # Ensure model name is valid if all inputs were skipped (e.g. invalid params)
        if not model_name: model_name = "DefaultToolArgs"

        return create_model(model_name, **fields)


    def _create_tool_class(
            self,
            tool_name: str, 
            http_method: str,
            base_url: str,
            path_template: str, 
            description: str,
            operation: OperationObject, 
            auth: Optional[AbstractAuth] = None, 
            metadata: Optional[Dict[str, Any]] = None,
    ) -> Type[BaseTool]:
        class_name_sanitized = sanitize_tool_name(tool_name).capitalize()
        
        args_model = self._create_args_model(
            operation_id=tool_name, 
            operation=operation,
            placeholder_if_empty=None 
        )

        final_description = description or f"Tool for {http_method.upper()} {path_template}"
        if operation.summary and operation.summary not in final_description: # Add summary if not already in description
            final_description = f"{operation.summary}. {final_description}"

        # Capture self for use in _run_sync and _arun_async
        builder_self = self

        def _run_sync(instance_self: BaseTool, **kwargs: Any) -> str:
            request_kwargs, constructed_url = builder_self._prepare_request_params(
                operation_params=operation.parameters or [],
                base_url=base_url,
                path_template=path_template,
                auth=auth,
                tool_input_data=kwargs 
            )
            try:
                response = requests.request(http_method.upper(), constructed_url, **request_kwargs)
                response.raise_for_status() 
                # Try to return JSON if possible, else text
                try:
                    return json.dumps(response.json()) 
                except json.JSONDecodeError:
                    return response.text
            except requests.exceptions.RequestException as e:
                return f"Request failed: {e}"


        async def _arun_async(instance_self: BaseTool, **kwargs: Any) -> str:
            request_kwargs, constructed_url = builder_self._prepare_request_params(
                operation_params=operation.parameters or [],
                base_url=base_url,
                path_template=path_template,
                auth=auth,
                tool_input_data=kwargs
            )
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.request(http_method.upper(), constructed_url, **request_kwargs)
                    response.raise_for_status()
                    try:
                        return json.dumps(response.json())
                    except json.JSONDecodeError:
                        return response.text
                except httpx.RequestError as e:
                    return f"Request failed: {e}"
                except httpx.HTTPStatusError as e:
                    return f"HTTP error {e.response.status_code}: {e.response.text}"


        tool_attrs: Dict[str, Any] = {
            "name": sanitize_tool_name(tool_name), 
            "description": final_description,
            "args_schema": args_model,
            "_run": _run_sync,
            "_arun": _arun_async,
            "__module__": builder_self.__class__.__module__, 
             "metadata": metadata, 
        }
        
        tool_class_name = class_name_sanitized + "Tool"
        # Ensure class name is valid if original tool_name was empty or all special chars
        if not tool_class_name.replace("Tool", ""): tool_class_name = "DefaultGeneratedTool"

        dynamic_tool_class = type(tool_class_name, (StructuredTool,), tool_attrs)
        return dynamic_tool_class 


    def _prepare_request_params(
        self,
        operation_params: List[ParameterObject],
        base_url: str,
        path_template: str,
        auth: Optional[AbstractAuth], 
        tool_input_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], str]: 
        return self._build_request_components(
            operation_params, base_url, path_template, auth, tool_input_data
        )

    # --- Refactored `convert_tool_params_request_params` and its helpers ---

    def _distribute_tool_input(
        self,
        tool_input_data: Dict[str, Any],
        operation_params: List[ParameterObject]
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[Any]]:
        headers: Dict[str, str] = {}
        query_params: Dict[str, Any] = {}
        path_params: Dict[str, Any] = {}
        cookie_params: Dict[str, str] = {}
        
        body_data = tool_input_data.pop("request_body", None) # If 'request_body' is a dedicated field in args_model

        # Check if there are other keys intended for the body, if body_data is None yet
        # This happens if the requestBody schema was not an object and its fields are at top level of args_model
        if body_data is None and "body_payload" in tool_input_data and not operation_params : # Simple body model
             body_data = tool_input_data.pop("body_payload")


        # Remaining tool_input_data are considered parameters or part of a flat request body
        param_names_in_spec = {sanitize_tool_name(p.name) for p in operation_params}
        
        # If no 'request_body' field and it's not a simple 'body_payload',
        # then remaining non-parameter fields might form an implicit request body.
        # This part can be tricky if parameter names clash with body field names.
        # The _create_args_model aims to avoid this by nesting body under 'request_body'.
        # If body_data is still None, and method is POST/PUT etc., remaining might be body.
        # For now, assume _create_args_model correctly structured 'request_body' or 'body_payload'.

        for op_param in operation_params:
            param_model_name = sanitize_tool_name(op_param.name) 
            param_http_name = op_param.name 

            if param_model_name in tool_input_data:
                param_val = tool_input_data.get(param_model_name) # Use get for safety, though key should exist if in model
                 # Use default from spec if input value is None and default is defined for the param
                if param_val is None and op_param.schema_ and op_param.schema_.default is not None:
                    param_val = op_param.schema_.default

                if param_val is not None: 
                    if op_param.in_ == "header":
                        headers[param_http_name] = str(param_val)
                    elif op_param.in_ == "query":
                        query_params[param_http_name] = param_val
                    elif op_param.in_ == "path":
                        path_params[param_http_name] = str(param_val)
                    elif op_param.in_ == "cookie":
                        cookie_params[param_http_name] = str(param_val)
        
        return headers, query_params, path_params, cookie_params, body_data

    def _build_url(self, base_url: str, path_template: str, path_params: Dict[str, Any]) -> str:
        url = base_url.rstrip("/") + "/" + path_template.lstrip("/")
        for p_name, p_val in path_params.items():
            url = url.replace(f"{{{p_name}}}", str(p_val)) 
        return url

    def _get_request_auth_headers(self, auth: Optional[AbstractAuth]) -> Dict[str, str]:
        if auth and hasattr(auth, 'get_auth_headers') and callable(auth.get_auth_headers):
            return auth.get_auth_headers()
        return {}

    def _prepare_request_kwargs(
        self,
        headers: Dict[str, str],
        query_params: Dict[str, Any],
        cookie_params: Dict[str, str], 
        body_data: Optional[Any],
        auth_headers: Dict[str, str]
    ) -> Dict[str, Any]:
        final_headers = headers.copy()
        final_headers.update(auth_headers) 

        request_kwargs: Dict[str, Any] = {}
        if final_headers: 
            request_kwargs["headers"] = final_headers
        if query_params:
            request_kwargs["params"] = query_params
        
        if body_data: # body_data here is already the Pydantic model instance or simple payload
            if isinstance(body_data, BaseModel):
                 request_kwargs["json"] = body_data.model_dump(by_alias=True, exclude_none=True) # Serialize Pydantic model
            else: # Simple payload (e.g. string, list for non-object requestBody)
                request_kwargs["json"] = body_data

            if "Content-Type" not in final_headers and "content-type" not in final_headers :
                final_headers["Content-Type"] = "application/json"
                request_kwargs["headers"] = final_headers 
        
        if cookie_params:
            request_kwargs.setdefault("headers", {})["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookie_params.items())

        return request_kwargs

    def _build_request_components(
        self,
        operation_params: List[ParameterObject],
        base_url: str,
        path_template: str,
        auth: Optional[AbstractAuth],
        tool_input_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], str]:
        headers, query_params, path_params, cookie_params, body_data = self._distribute_tool_input(
            tool_input_data, operation_params
        )
        
        constructed_url = self._build_url(base_url, path_template, path_params)
        
        auth_headers = self._get_request_auth_headers(auth)
        
        request_kwargs = self._prepare_request_kwargs(
            headers, query_params, cookie_params, body_data, auth_headers
        )
        
        return request_kwargs, constructed_url

    # --- End of Refactored `convert_tool_params_request_params` ---


    @staticmethod
    def _parse_input_helper(tool_input: Any, tool_name: str, sanitized_name: str) -> Dict[str, Any]: 
        if isinstance(tool_input, str):
            try:
                parsed = json.loads(tool_input)
                if isinstance(parsed, dict):
                    return parsed
                else: 
                    return {"input": parsed}
            except json.JSONDecodeError:
                return {"input": tool_input}
        elif isinstance(tool_input, dict): 
            return tool_input
        else: 
            return {"input": tool_input}

    @staticmethod
    def extract_examples(media_type_obj: Optional[MediaTypeObject]) -> Dict[str, Any]: 
        if not media_type_obj:
            return {}

        examples: Dict[str, Any] = {}

        if media_type_obj.examples: 
            for example_name, example_obj_or_ref in media_type_obj.examples.items():
                if hasattr(example_obj_or_ref, 'value'):
                    examples[example_name] = {"summary": getattr(example_obj_or_ref, 'summary', None), "value": example_obj_or_ref.value}
                else: 
                     examples[example_name] = {"value": example_obj_or_ref}
        elif media_type_obj.example is not None: 
            examples["default"] = {"value": media_type_obj.example}
        elif media_type_obj.schema_ and media_type_obj.schema_.example is not None: 
            examples["default"] = {"value": media_type_obj.schema_.example}
        
        return examples


    def build_tools(self, auth: Optional[AbstractAuth] = None) -> List[BaseTool]: 
        tools: List[BaseTool] = []
        base_url = self.spec_obj.servers[0].url if self.spec_obj.servers else ""
        
        if not self.spec_obj.paths:
            return tools

        for path_str, path_item_obj in self.spec_obj.paths.items(): 
            if not isinstance(path_item_obj, PathItemObject): continue 

            for http_method_name in ['get', 'put', 'post', 'delete', 'options', 'head', 'patch', 'trace']:
                operation_obj: Optional[OperationObject] = getattr(path_item_obj, http_method_name, None)
                if not operation_obj:
                    continue

                operation_id = self._extract_operationid_or_use_path(http_method_name, operation_obj, path_str)
                
                summary = operation_obj.summary or ""
                description = operation_obj.description or summary or f"API call to {operation_id}" 

                metadata: Dict[str, Any] = {}
                if operation_obj.externalDocs:
                    metadata['externalDocs'] = operation_obj.externalDocs.model_dump(exclude_none=True)
                
                if operation_obj.requestBody and operation_obj.requestBody.content:
                    json_media_type = operation_obj.requestBody.content.get("application/json")
                    if json_media_type:
                        metadata['requestBodyExamples'] = self.extract_examples(json_media_type)
                
                response_examples: Dict[str, Any] = {}
                if operation_obj.responses: # responses is ResponsesModel (RootModel[Dict[str, Union[ResponseObject, ReferenceObject]]])
                    for status_code, response_obj_or_ref in operation_obj.responses.items(): # Iterate through ResponsesModel
                        # Resolve reference if necessary (assuming resolver is run before builder or handled by spec_obj)
                        response_obj = response_obj_or_ref # Assuming resolved for now
                        if hasattr(response_obj, 'content') and response_obj.content:
                             json_media_type = response_obj.content.get("application/json") 
                             if json_media_type:
                                 extracted_ex = self.extract_examples(json_media_type)
                                 if extracted_ex:
                                     response_examples[status_code] = extracted_ex
                if response_examples:
                     metadata['responseExamples'] = response_examples


                tool_class = self._create_tool_class(
                    tool_name=operation_id, 
                    http_method=http_method_name,
                    base_url=base_url,
                    path_template=path_str,
                    description=description,
                    operation=operation_obj,
                    auth=auth, 
                    metadata=metadata
                )
                tools.append(tool_class()) 
        
        return tools

    def _external_external_docs(self, operation: OperationObject) -> Optional[str]: 
        if operation.externalDocs and operation.externalDocs.url:
            return operation.externalDocs.url
        if self.spec_obj.externalDocs and self.spec_obj.externalDocs.url: 
            return self.spec_obj.externalDocs.url
        return None

    @staticmethod
    def _extract_operationid_or_use_path(method: str, operation: OperationObject, path: str) -> str: 
        if operation.operationId:
            return operation.operationId
        sanitized_path = re.sub(r'[\W]+', '_', path.replace('{', '').replace('}', ''))
        # Ensure it doesn't start with a digit if path was e.g. /123/abc
        if sanitized_path.startswith('_') and len(sanitized_path) > 1 and sanitized_path[1].isdigit():
            sanitized_path = f"_{sanitized_path}" # Keep underscore if it was like /_123/
        elif sanitized_path and sanitized_path[0].isdigit():
             sanitized_path = f"_{sanitized_path}"
        return f"{method}{sanitized_path}"


loaded_tools: Dict[str, List[BaseTool]] = {} 


def load_tools_from_openapi(openapi_file_path: str, auth: Optional[AbstractAuth] = None, spec_url: Optional[str] = None) -> List[BaseTool]: 
    cache_key = spec_url or openapi_file_path

    if cache_key in loaded_tools:
        return loaded_tools[cache_key]

    try:
        with open(openapi_file_path, "r", encoding='utf-8') as f: # Specify encoding
            spec_content = f.read()
    except FileNotFoundError:
        raise ValueError(f"OpenAPI specification file not found: {openapi_file_path}")
    except Exception as e:
        raise ValueError(f"Error reading OpenAPI specification file: {e}")

    builder = OpenAPIToolBuilder(spec_content) 
    tools = builder.build_tools(auth=auth) 
    loaded_tools[cache_key] = tools
    return tools


class AuthTool(BaseTool): 
    name: str = "AuthTool"
    description: str = "Tool to handle authentication and retrieve tokens via human-in-the-loop."
    args_schema: Optional[Type[BaseModel]] = create_model("AuthToolArgs", tool_input=(Optional[str], Field(default=None, description="Optional input for auth specifics")))


    def _run(self, tool_input: Optional[str]=None) -> str: 
        prompt_message = "Please enter your authentication token"
        if tool_input:
            prompt_message += f" for {tool_input}"
        prompt_message += ": "
        
        try:
            token = input(prompt_message) 
        except EOFError: 
            return "Error: Cannot get authentication token. No interactive input available."
        return token

    async def _arun(self, tool_input: Optional[str]=None) -> str: 
        return self._run(tool_input=tool_input) 


def create_input_schema_from_json_schema(
    model_name: str, 
    json_schema: JSONLike
) -> Type[BaseModel]:
    if not isinstance(json_schema, dict): 
        raise ValueError("json_schema must be a dictionary.")

    # If it's an object type with no properties, create an empty model
    if json_schema.get("type") == "object" and not json_schema.get("properties"):
        return create_model(sanitize_tool_name(model_name).capitalize() or "DynamicEmptyInputSchema")
    
    # If it's not an object type but has properties, or is missing type and has properties, it's ambiguous.
    # We expect "type": "object" if "properties" are present for a structured model.
    if json_schema.get("type") != "object" and "properties" in json_schema:
         raise ValueError("Input schema must be an 'object' type if 'properties' are defined.")


    properties = json_schema.get('properties', {})
    required = json_schema.get('required', [])
    field_definitions: Dict[str, Any] = {}
    
    _type_mapping: Dict[str, Type] = {
        'string': str, 'integer': int, 'number': float,
        'boolean': bool, 'array': list, 'object': Dict[str, Any], 'null': type(None)
    }

    def _resolve_type(prop_name_for_enum: str, prop_schema_dict: JSONLike) -> Any:
        if not isinstance(prop_schema_dict, dict):
            return Any 

        # Handle oneOf, anyOf, allOf - simplified to Dict[str,Any] or Any
        # More advanced handling would create Union types or merged models.
        if 'oneOf' in prop_schema_dict or 'anyOf' in prop_schema_dict or 'allOf' in prop_schema_dict:
            # Could attempt to build a Union type if all sub-schemas are simple.
            # For now, keep it as Dict[str, Any] as per original openapi_type_to_python_type for these cases.
            return Dict[str, Any] 

        prop_type_val = prop_schema_dict.get('type')
        is_nullable = prop_schema_dict.get("nullable", False) # OpenAPI 3.0.x
        final_py_type: Any = Any

        if isinstance(prop_type_val, list):
            if 'null' in prop_type_val:
                is_nullable = True
            types_no_null = [t for t in prop_type_val if t != 'null']
            prop_type_val = types_no_null[0] if types_no_null else 'string' 

        if "enum" in prop_schema_dict:
            from enum import Enum as PyEnum # Local alias
            enum_values = prop_schema_dict['enum']
            enum_name_str = sanitize_tool_name(prop_schema_dict.get('title', model_name) + "_" + prop_name_for_enum).capitalize() + "Enum"
            
            enum_members: Dict[str, Any] = {}
            for i, val in enumerate(enum_values):
                member_name = str(val)
                if isinstance(val, str) and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', val) and not keyword.iskeyword(val):
                     member_name = val
                else: 
                     member_name = f"VALUE_{str(val).upper()}" # More robust naming
                     member_name = re.sub(r'\W+', '_', member_name)
                     if not member_name.isidentifier(): member_name = f"MEMBER_{i}"

                original_member_name = member_name
                counter = 1
                while member_name in enum_members and enum_members[member_name] != val: # Handle name clashes for different values
                    member_name = f"{original_member_name}_{counter}"
                    counter += 1
                enum_members[member_name] = val
            
            if not enum_members: # Handle empty enum case if it can occur
                final_py_type = Any
            else:
                try:
                    final_py_type = PyEnum(enum_name_str, enum_members)
                except TypeError: # If names are still not valid/unique after sanitization
                    final_py_type = Any # Fallback for complex enum cases
        
        elif prop_type_val == 'array':
            items_schema = prop_schema_dict.get('items', {}) # Default to empty dict if not present
            item_py_type = _resolve_type(f"{prop_name_for_enum}_item", items_schema) if items_schema else Any
            final_py_type = List[item_py_type] # type: ignore
        
        elif prop_type_val == 'object':
            if 'properties' in prop_schema_dict and prop_schema_dict['properties']: # Ensure properties is not empty
                nested_model_name = sanitize_tool_name(prop_schema_dict.get('title', model_name) + "_" + prop_name_for_enum).capitalize()
                if not nested_model_name: nested_model_name = "NestedInput" # Fallback name
                final_py_type = create_input_schema_from_json_schema(nested_model_name, prop_schema_dict)
            else: 
                final_py_type = Dict[str, Any]
        else: 
            final_py_type = _type_mapping.get(str(prop_type_val), Any)

        return Optional[final_py_type] if is_nullable and final_py_type is not Any else final_py_type


    for prop_name, prop_schema_dict_val in properties.items():
        if not isinstance(prop_schema_dict_val, dict): continue 

        description = prop_schema_dict_val.get('description', '')
        default_value = prop_schema_dict_val.get('default', ...) 
        
        python_type = _resolve_type(prop_name, prop_schema_dict_val)
        field_name_sanitized = sanitize_tool_name(prop_name)
        if not field_name_sanitized : field_name_sanitized = f"prop_{prop_name}" # Ensure valid field name
        
        field_args: Dict[str, Any] = {"description": description}
        # For Pydantic v2, alias should be set if the sanitized name is different
        # and the original name is a valid Python identifier (though Pydantic might handle non-ident aliases).
        # If original prop_name is not a valid Python identifier, it *must* be aliased.
        if field_name_sanitized != prop_name or not prop_name.isidentifier():
            field_args["alias"] = prop_name 
        
        is_field_optional = not (prop_name in required)
        is_type_already_optional = hasattr(python_type, '__origin__') and python_type.__origin__ is Union and type(None) in python_type.__args__


        if not is_field_optional: # Required field
            if default_value is ...: # No default provided in schema for a required field
                current_field = Field(**field_args, default=...)
            else: # Default provided in schema for a required field
                current_field = Field(**field_args, default=default_value)
        else: # Optional field
            effective_default = default_value if default_value is not ... else None
            current_field = Field(**field_args, default=effective_default)
            if not is_type_already_optional : # Make the type Optional[] if not already
                 python_type = Optional[python_type]
        
        field_definitions[field_name_sanitized] = (python_type, current_field)


    pydantic_model_name_str = sanitize_tool_name(model_name).capitalize()
    if not pydantic_model_name_str: pydantic_model_name_str = "DynamicInputSchema" 

    return create_model(pydantic_model_name_str, **field_definitions)

import keyword # For create_input_schema_from_json_schema enum member check
