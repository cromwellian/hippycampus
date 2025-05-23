import pytest
from typing import Any, Dict, List, Optional, Union, Type # Added Optional
from pydantic import BaseModel, Field, create_model, ValidationError as PydanticValidationError
from enum import Enum as PyEnum
import json
from unittest.mock import patch, MagicMock, ANY # Added ANY for mock assertions

from hippycampus.openapi_builder import (
    sanitize_tool_name,
    openapi_type_to_python_type, # This is a method of OpenAPIToolBuilder now
    create_input_schema_from_json_schema,
    OpenAPIToolBuilder,
    _parse_input_helper, 
    extract_examples,    
)
from hippycampus.spec_parser import ( 
    OpenAPIObject,
    InfoObject,
    PathsModel,
    PathItemObject,
    OperationObject,
    ParameterObject,
    ParameterLocation,
    RequestBodyObject,
    MediaTypeObject,
    SchemaObject,
    SchemaType,
    ResponseObject,
    ResponsesModel,
    ComponentsObject,
    SecuritySchemeObject,
    SecuritySchemeType,
    SecurityRequirementModel,
    ExampleObject, 
    parse_yaml, 
)
from hippycampus.tool_auth.authentication import AbstractAuth 

# --- Fixtures for OpenAPIObject components ---

@pytest.fixture
def minimal_openapi_spec_dict() -> Dict[str, Any]:
    """A very basic OpenAPI spec for testing, as dict."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Minimal API", "version": "1.0.0"},
        "servers": [{"url": "http://localhost:8000/api/v1"}],
        "paths": {
            "/test_get": {
                "get": {
                    "operationId": "get_test_operation",
                    "summary": "Test GET operation",
                    "description": "A simple GET request.",
                    "responses": {
                        "200": {"description": "Successful response"}
                    }
                }
            }
        }
    }

@pytest.fixture
def minimal_openapi_spec(minimal_openapi_spec_dict: Dict[str, Any]) -> OpenAPIObject:
    return OpenAPIObject.model_validate(minimal_openapi_spec_dict)


@pytest.fixture
def spec_with_params_and_body_dict() -> Dict[str, Any]: 
    """OpenAPI spec with various parameters and a request body, as a dict."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Params and Body API", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/items/{item_id}": {
                "post": {
                    "operationId": "create_item",
                    "summary": "Create an item",
                    "parameters": [
                        {"name": "item_id", "in": "path", "required": True, "schema": {"type": "integer", "description": "ID of the item"}},
                        {"name": "api-version", "in": "header", "schema": {"type": "string", "default": "v1"}}, 
                        {"name": "category", "in": "query", "schema": {"type": "string", "enum": ["electronics", "books"], "title": "Category"}}, # Added title for enum
                        {"name": "session_id", "in": "cookie", "schema": {"type": "string"}}
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "Name of the item"},
                                        "price": {"type": "number"},
                                        "is_active": {"type": "boolean", "default": True},
                                        "tags": {"type": "array", "items": {"type": "string"}}
                                    },
                                    "required": ["name", "price"]
                                }
                            }
                        }
                    },
                    "responses": {"201": {"description": "Item created"}}
                }
            }
        }
    }

@pytest.fixture
def spec_with_params_and_body(spec_with_params_and_body_dict: Dict[str, Any]) -> OpenAPIObject:
    return OpenAPIObject.model_validate(spec_with_params_and_body_dict)


@pytest.fixture
def spec_with_auth_api_key_dict() -> Dict[str, Any]:
    """OpenAPI spec with API key authentication (header), as dict."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "API Key Auth API", "version": "1.0.0"},
        "servers": [{"url": "http://localhost:8000"}],
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "name": "X-API-Key", "in": "header"}
            }
        },
        "paths": {
            "/secure_resource": {
                "get": {
                    "operationId": "get_secure_resource",
                    "summary": "Access a secure resource",
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {"200": {"description": "Success"}}
                }
            }
        }
    }

@pytest.fixture
def spec_with_auth_api_key(spec_with_auth_api_key_dict: Dict[str, Any]) -> OpenAPIObject:
    return OpenAPIObject.model_validate(spec_with_auth_api_key_dict)


@pytest.fixture
def spec_with_auth_bearer_dict() -> Dict[str, Any]:
    """OpenAPI spec with Bearer token authentication, as dict."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Bearer Auth API", "version": "1.0.0"},
        "servers": [{"url": "http://localhost:8000"}],
        "components": {
            "securitySchemes": {
                "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
            }
        },
        "paths": {
            "/bearer_resource": {
                "get": {
                    "operationId": "get_bearer_resource",
                    "summary": "Access bearer authenticated resource",
                    "security": [{"BearerAuth": []}],
                    "responses": {"200": {"description": "Success"}}
                }
            }
        }
    }
@pytest.fixture
def spec_with_auth_bearer(spec_with_auth_bearer_dict: Dict[str, Any]) -> OpenAPIObject:
    return OpenAPIObject.model_validate(spec_with_auth_bearer_dict)

    
@pytest.fixture
def spec_with_auth_basic_dict() -> Dict[str, Any]:
    """OpenAPI spec with Basic authentication, as dict."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Basic Auth API", "version": "1.0.0"},
        "servers": [{"url": "http://localhost:8000"}],
        "components": {
            "securitySchemes": {
                "BasicAuth": {"type": "http", "scheme": "basic"}
            }
        },
        "paths": {
            "/basic_resource": {
                "get": {
                    "operationId": "get_basic_resource",
                    "summary": "Access basic authenticated resource",
                    "security": [{"BasicAuth": []}],
                    "responses": {"200": {"description": "Success"}}
                }
            }
        }
    }

@pytest.fixture
def spec_with_auth_basic(spec_with_auth_basic_dict: Dict[str, Any]) -> OpenAPIObject:
    return OpenAPIObject.model_validate(spec_with_auth_basic_dict)


@pytest.fixture
def spec_with_special_op_id_dict() -> Dict[str, Any]:
    """OpenAPI spec with an operationId that needs sanitization, as dict."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Special Op ID API", "version": "1.0.0"},
        "servers": [{"url": "http://localhost:8000"}],
        "paths": {
            "/special_op": {
                "get": {
                    "operationId": "123-get data for user!",
                    "summary": "Operation with special characters in ID",
                    "responses": {"200": {"description": "Success"}}
                }
            }
        }
    }
@pytest.fixture
def spec_with_special_op_id(spec_with_special_op_id_dict: Dict[str, Any]) -> OpenAPIObject:
    return OpenAPIObject.model_validate(spec_with_special_op_id_dict)


# --- Unit Tests for standalone functions ---

class TestSanitizeToolName:
    @pytest.mark.parametrize("original_name, expected_name", [
        ("get user info", "get_user_info"),
        ("create-item", "create_item"),
        ("123process_data", "_123process_data"),
        ("!@#$%^&*()", "_"),
        (" leading_space", "leading_space"),
        ("trailing_space ", "trailing_space"),
        ("multiple   spaces", "multiple_spaces"),
        ("valid_name", "valid_name"),
        ("a", "a"),
        ("", "_") 
    ])
    def test_sanitize_tool_name(self, original_name: str, expected_name: str):
        assert sanitize_tool_name(original_name) == expected_name

class TestOpenAPITypeToPythonType: # Testing the method of OpenAPIToolBuilder
    dummy_builder = OpenAPIToolBuilder({"openapi": "3.0.0", "info": {"title": "dummy", "version": "1"}, "paths": {}})

    @pytest.mark.parametrize("openapi_type, schema_format, expected_python_type", [
        (SchemaType.STRING, None, str),
        (SchemaType.INTEGER, None, int),
        (SchemaType.NUMBER, None, float),
        (SchemaType.BOOLEAN, None, bool),
        (SchemaType.OBJECT, None, Dict[str, Any]),
        (SchemaType.ARRAY, None, List[Any]), 
        ("string", "date-time", str), 
        ("string", "email", str),
        ("string", "uuid", str),
        ("integer", "int32", int),
        ("integer", "int64", int),
        ("number", "float", float),
        ("number", "double", float),
    ])
    def test_basic_type_mappings(self, openapi_type: Union[SchemaType, str], schema_format: Optional[str], expected_python_type: Type[Any]):
        param_name = "test_param"
        schema = SchemaObject(type=openapi_type, format=schema_format)
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert python_type is expected_python_type

    def test_type_as_list_string_null(self):
        param_name = "test_param_nullable_str"
        schema = SchemaObject(type=[SchemaType.STRING, SchemaType.NULL])
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert python_type is Optional[str]

    def test_type_as_list_integer_null(self):
        param_name = "test_param_nullable_int"
        schema = SchemaObject(type=[SchemaType.INTEGER, "null"])  # type: ignore 
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert python_type is Optional[int]
    
    def test_type_as_list_object_null(self):
        param_name = "test_param_nullable_obj"
        schema = SchemaObject(type=[SchemaType.OBJECT, SchemaType.NULL])
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert python_type is Optional[Dict[str, Any]]

    def test_type_as_list_array_null(self):
        param_name = "test_param_nullable_array"
        schema = SchemaObject(type=[SchemaType.ARRAY, SchemaType.NULL])
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert python_type is Optional[List[Any]]
        
    def test_type_as_list_exclusive_null(self): 
        param_name = "test_param_nullable_via_prop"
        schema = SchemaObject(type=SchemaType.STRING, nullable=True)
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert python_type is Optional[str]

    def test_enum_creation(self):
        param_name = "color_enum_param"
        enum_values = ["red", "green", "blue"]
        schema = SchemaObject(type=SchemaType.STRING, enum=enum_values, title="ColorType") 
        python_type, field_info = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        
        assert issubclass(python_type, PyEnum)
        assert python_type.__name__ == "ColorTypeEnum" 
        for val in enum_values:
            assert hasattr(python_type, val.upper()) 
            assert getattr(python_type, val.upper()).value == val
        
        param_name_num = "status_code_enum"
        enum_num_values = [200, 404, 500]
        schema_num = SchemaObject(type=SchemaType.INTEGER, enum=enum_num_values, title="StatusCode")
        python_type_num, _ = self.dummy_builder.openapi_type_to_python_type(param_name_num, schema_num)
        assert issubclass(python_type_num, PyEnum) 
        assert python_type_num.__name__ == "StatusCodeEnum"
        assert python_type_num.MEMBER_200.value == 200  # type: ignore
        assert python_type_num.MEMBER_404.value == 404  # type: ignore
        assert python_type_num.MEMBER_500.value == 500  # type: ignore


    def test_array_with_basic_items(self):
        param_name = "string_array_param"
        schema = SchemaObject(type=SchemaType.ARRAY, items=SchemaObject(type=SchemaType.STRING))
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert python_type == List[str] 

    def test_array_with_enum_items(self):
        param_name = "enum_array_param"
        enum_values = ["active", "inactive"]
        items_schema = SchemaObject(type=SchemaType.STRING, enum=enum_values, title="ItemStatus")
        schema = SchemaObject(type=SchemaType.ARRAY, items=items_schema)
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        
        assert str(python_type).startswith("typing.List[") 
        item_type = python_type.__args__[0] # type: ignore 
        assert issubclass(item_type, PyEnum)
        assert item_type.__name__ == "ItemStatusEnum" 
        for val in enum_values:
            assert hasattr(item_type, val.upper())

    def test_array_with_one_of_items(self): # `oneOf` in items should ideally be `List[Union[...]]`
        param_name = "one_of_array_param"
        items_schema = SchemaObject(oneOf=[SchemaObject(type=SchemaType.STRING), SchemaObject(type=SchemaType.INTEGER)])
        schema = SchemaObject(type=SchemaType.ARRAY, items=items_schema)
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        # Current openapi_type_to_python_type maps oneOf/anyOf/allOf to Dict[str, Any]
        # So this results in List[Dict[str, Any]]
        assert python_type == List[Dict[str, Any]] 

    def test_default_type_for_unknown(self):
        param_name = "unknown_type_param"
        schema = SchemaObject(type="unknownCustomType")  # type: ignore
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert python_type is Any 

    def test_no_type_specified(self): 
        param_name = "no_type_param"
        schema = SchemaObject() 
        python_type, _ = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert python_type is Any

    def test_field_info_generation(self):
        param_name = "test_field"
        schema = SchemaObject(type=SchemaType.STRING, description="A test field", default="hello")
        _, field_info = self.dummy_builder.openapi_type_to_python_type(param_name, schema)
        assert field_info is not None
        assert field_info.description == "A test field"
        assert field_info.default == "hello"

        schema_no_default = SchemaObject(type=SchemaType.INTEGER, description="Required field")
        _, field_info_no_default = self.dummy_builder.openapi_type_to_python_type(
            param_name, schema_no_default, is_required=True
        )
        assert field_info_no_default.default is Ellipsis 
        assert field_info_no_default.description == "Required field"

        schema_nullable_with_default_none = SchemaObject(type=SchemaType.STRING, nullable=True, default=None)
        py_type, field_info_nullable = self.dummy_builder.openapi_type_to_python_type(
            param_name, schema_nullable_with_default_none, is_required=False
        )
        assert py_type is Optional[str]
        assert field_info_nullable.default is None 

class TestCreateInputSchemaFromJSONSchema:
    def test_simple_schema(self):
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        InputModel = create_input_schema_from_json_schema("UserInput", json_schema) 
        assert InputModel.__name__ == "UserInput"
        assert "name" in InputModel.model_fields
        assert "age" in InputModel.model_fields
        assert InputModel.model_fields["name"].description == "User name"
        assert InputModel.model_fields["name"].annotation is str 
        assert InputModel.model_fields["name"].is_required() 
        assert InputModel.model_fields["age"].annotation is Optional[int] 
        assert not InputModel.model_fields["age"].is_required()


        valid_data = InputModel(name="John Doe", age=30)
        assert valid_data.name == "John Doe"
        with pytest.raises(PydanticValidationError):
            InputModel(age=30) 

    def test_various_types(self):
        json_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "score": {"type": "number"},
                "is_active": {"type": "boolean"},
                "metadata": {"type": "object"}, 
                "tags": {"type": "array", "items": {"type": "string"}},
                "created_at": {"type": "string", "format": "date-time"}
            }
        }
        DataModel = create_input_schema_from_json_schema("DataModel", json_schema) 
        assert DataModel.model_fields["id"].annotation is Optional[str]
        assert DataModel.model_fields["score"].annotation is Optional[float]
        assert DataModel.model_fields["is_active"].annotation is Optional[bool]
        assert DataModel.model_fields["metadata"].annotation is Optional[Dict[str, Any]]
        assert DataModel.model_fields["tags"].annotation == Optional[List[str]] 
        assert DataModel.model_fields["created_at"].annotation is Optional[str]

    def test_nested_objects_and_arrays(self):
        json_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"username": {"type": "string"}, "email": {"type": "string"}},
                    "required": ["username"]
                },
                "history": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"event": {"type": "string"}, "timestamp": {"type": "string", "format": "date-time"}}
                    }
                }
            }
        }
        ComplexModel = create_input_schema_from_json_schema("ComplexModel", json_schema) 

        user_field = ComplexModel.model_fields["user"]
        assert user_field.annotation.__args__[0].__name__ == "Complexmodel_userNestedInput" # type: ignore # Check generated name
        user_nested_model = user_field.annotation.__args__[0] # type: ignore
        assert "username" in user_nested_model.model_fields
        assert user_nested_model.model_fields["username"].annotation is str
        assert user_nested_model.model_fields["email"].annotation is Optional[str]

        history_field = ComplexModel.model_fields["history"]
        history_item_model_name = history_field.annotation.__args__[0].__args__[0].__name__ # type: ignore
        assert history_item_model_name == "Complexmodel_history_itemNestedInput" # Check generated name
        
        instance = ComplexModel(user={"username": "test"}, history=[{"event": "login", "timestamp": "2023-01-01T00:00:00Z"}])
        assert instance.user is not None 
        assert instance.user.username == "test" # type: ignore
        assert instance.history is not None and len(instance.history) > 0
        assert instance.history[0].event == "login" # type: ignore

    def test_enum_and_default(self):
        json_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"], "default": "inactive", "title": "Status"},
                "count": {"type": "integer", "default": 0}
            }
        }
        EnumModel = create_input_schema_from_json_schema("EnumModel", json_schema) 
        status_field = EnumModel.model_fields["status"]
        # Check if the type is Optional[GeneratedEnum]
        assert status_field.annotation.__origin__ is Union # For Optional
        enum_type_in_optional = status_field.annotation.__args__[0]
        assert issubclass(enum_type_in_optional, PyEnum)
        assert enum_type_in_optional.__name__ == "Enummodel_statusEnum"
        assert status_field.default == "inactive"
        
        count_field = EnumModel.model_fields["count"]
        assert count_field.annotation is Optional[int] 
        assert count_field.default == 0

        instance = EnumModel() 
        assert instance.status is not None
        assert instance.status.value == "inactive"
        assert instance.count == 0

    def test_combiners_one_of_any_of_all_of(self):
        json_schema_one_of = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        OneOfModel = create_input_schema_from_json_schema("OneOfTest", {"type": "object", "properties": {"field": json_schema_one_of}}) 
        assert OneOfModel.model_fields["field"].annotation is Optional[Dict[str, Any]]

        json_schema_any_of = {"anyOf": [{"type": "string"}, {"type": "boolean"}]}
        AnyOfModel = create_input_schema_from_json_schema("AnyOfTest", {"type": "object", "properties": {"field": json_schema_any_of}}) 
        assert AnyOfModel.model_fields["field"].annotation is Optional[Dict[str, Any]]

        json_schema_all_of = {
            "allOf": [
                {"type": "object", "properties": {"name": {"type": "string"}}},
                {"type": "object", "properties": {"age": {"type": "integer"}}}
            ]
        }
        AllOfModel = create_input_schema_from_json_schema("AllOfTest", {"type": "object", "properties": {"field": json_schema_all_of}}) 
        assert AllOfModel.model_fields["field"].annotation is Optional[Dict[str, Any]]

    def test_empty_schema_or_no_type(self):
        EmptySchemaModel = create_input_schema_from_json_schema("EmptySchemaTest", {"type": "object"}) 
        assert not EmptySchemaModel.model_fields 
        EmptySchemaModel() 

        NoTypePropModel = create_input_schema_from_json_schema("NoTypePropTest", {"type": "object", "properties": {"field": {}}}) 
        assert NoTypePropModel.model_fields["field"].annotation is Optional[Any]

    def test_top_level_array_schema(self):
        json_schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        with pytest.raises(ValueError, match="Input schema must be an 'object' type if 'properties' are defined."): 
            create_input_schema_from_json_schema("TopLevelArrayTest", json_schema) 


class TestParseInputHelper:
    def test_valid_json_string(self):
        json_str = '{"key": "value", "number": 123}'
        expected_dict = {"key": "value", "number": 123}
        assert _parse_input_helper(json_str, "", "") == expected_dict 

    def test_non_json_string_returns_wrapped_input(self):
        non_json_str = "This is a plain string, not JSON."
        expected_output = {"input": non_json_str}
        assert _parse_input_helper(non_json_str, "", "") == expected_output

        almost_json_str = "{'key': 'value'}" 
        expected_output_almost = {"input": almost_json_str}
        assert _parse_input_helper(almost_json_str, "", "") == expected_output_almost
        
        tuple_str = "(1, 2, 3)"
        expected_output_tuple = {"input": tuple_str}
        assert _parse_input_helper(tuple_str, "", "") == expected_output_tuple


    def test_dictionary_input(self):
        dict_input = {"name": "test", "data": [1, 2]}
        assert _parse_input_helper(dict_input, "", "") == dict_input

    def test_integer_input(self):
        int_input = 12345
        expected_output = {"input": int_input}
        assert _parse_input_helper(int_input, "", "") == expected_output

    def test_list_input(self):
        list_input = ["a", "b", {"c": "d"}]
        expected_output = {"input": list_input}
        assert _parse_input_helper(list_input, "", "") == expected_output
        
    def test_none_input(self):
        assert _parse_input_helper(None, "", "") == {"input": None}

    def test_empty_string_input(self):
        assert _parse_input_helper("", "", "") == {"input": ""}

class TestExtractExamples:
    def test_with_media_type_examples(self):
        media_type = MediaTypeObject(
            examples={
                "ex1": ExampleObject(summary="Example 1", value={"name": "foo", "id": 1}),
                "ex2": ExampleObject(summary="Example 2", value={"name": "bar", "id": 2})
            }
        )
        examples = extract_examples(media_type)
        expected = {
            "ex1": {"summary": "Example 1", "value": {"name": "foo", "id": 1}},
            "ex2": {"summary": "Example 2", "value": {"name": "bar", "id": 2}}
        }
        assert json.loads(json.dumps(examples)) == expected


    def test_with_media_type_single_example(self):
        media_type = MediaTypeObject(example={"name": "baz", "id": 3})
        examples = extract_examples(media_type)
        expected = {"default": {"value": {"name": "baz", "id": 3}}}
        assert examples == expected

    def test_with_schema_level_example(self):
        media_type = MediaTypeObject(
            schema_=SchemaObject(example={"name": "schema_example", "id": 4})
        )
        examples = extract_examples(media_type)
        expected = {"default": {"value": {"name": "schema_example", "id": 4}}}
        assert examples == expected
        
    def test_with_schema_level_examples_multiple_direct(self):
        schema = SchemaObject(type=SchemaType.OBJECT, example={"prop": "value"}) # type: ignore
        media_type = MediaTypeObject(schema_=schema)
        examples = extract_examples(media_type)
        assert examples == {"default": {"value": {"prop": "value"}}}


    def test_no_examples_present(self):
        media_type = MediaTypeObject() 
        examples = extract_examples(media_type)
        assert not examples 

        media_type_with_empty_schema = MediaTypeObject(schema_=SchemaObject())
        examples_empty_schema = extract_examples(media_type_with_empty_schema)
        assert not examples_empty_schema

    def test_preference_order(self):
        media_type = MediaTypeObject(
            example={"name": "media_single_example", "id": 5}, 
            examples={
                "ex_media": ExampleObject(value={"name": "media_named_example", "id": 6})
            },
            schema_=SchemaObject(example={"name": "schema_level_ignored", "id": 7})
        )
        examples = extract_examples(media_type)
        expected = {"ex_media": {"summary": None, "value": {"name": "media_named_example", "id": 6}}} 
        assert json.loads(json.dumps(examples)) == expected


        media_type_only_single_ex = MediaTypeObject(
            example={"name": "media_single_example", "id": 5},
            schema_=SchemaObject(example={"name": "schema_level_ignored", "id": 7})
        )
        examples_single = extract_examples(media_type_only_single_ex)
        expected_single = {"default": {"value": {"name": "media_single_example", "id": 5}}}
        assert examples_single == expected_single
        
        media_type_only_schema_ex = MediaTypeObject(
             schema_=SchemaObject(example={"name": "schema_example", "id": 7})
        )
        examples_schema = extract_examples(media_type_only_schema_ex)
        expected_schema = {"default": {"value": {"name": "schema_example", "id": 7}}}
        assert examples_schema == expected_schema


# --- Unit Tests for OpenAPIToolBuilder methods (via helper or build_tools) ---

class TestOpenAPIToolBuilderUnitTests:
    
    def test_create_args_model_no_params_no_body(self, minimal_openapi_spec: OpenAPIObject):
        builder = OpenAPIToolBuilder(minimal_openapi_spec.model_dump(by_alias=True)) 
        operation = minimal_openapi_spec.paths.root["/test_get"].get # type: ignore
        assert operation is not None 
        ArgsModel = builder._create_args_model("GetTestOperation", operation, placeholder_if_empty=None) 
        
        assert ArgsModel.__name__ == "GetTestOperationArgs"
        assert not ArgsModel.model_fields 

        args_instance = ArgsModel()
        assert not args_instance.model_dump() 

    def test_create_args_model_only_parameters(self):
        spec_dict = {
            "openapi": "3.0.0", "info": {"title": "Test", "version": "1"},
            "paths": {
                "/users/{user_id}": {
                    "get": {
                        "operationId": "get_user",
                        "parameters": [
                            {"name": "user_id", "in": "path", "required": True, "schema": {"type": "integer", "description": "User ID"}},
                            {"name": "include_details", "in": "query", "schema": {"type": "boolean", "default": False}},
                            {"name": "X-Request-ID", "in": "header", "schema": {"type": "string", "format": "uuid"}}
                        ],
                        "responses": {"200": {"description": "ok"}}
                    }
                }
            }
        }
        spec = OpenAPIObject.model_validate(spec_dict)
        builder = OpenAPIToolBuilder(spec_dict)
        operation = spec.paths.root["/users/{user_id}"].get # type: ignore
        assert operation is not None
        ArgsModel = builder._create_args_model("GetUser", operation) 

        assert "user_id" in ArgsModel.model_fields
        assert ArgsModel.model_fields["user_id"].annotation is int
        assert ArgsModel.model_fields["user_id"].is_required()
        assert ArgsModel.model_fields["user_id"].description == "User ID"
        
        assert "include_details" in ArgsModel.model_fields
        assert ArgsModel.model_fields["include_details"].annotation is Optional[bool]
        assert ArgsModel.model_fields["include_details"].default is False 
        
        assert "X_Request_ID" in ArgsModel.model_fields 
        assert ArgsModel.model_fields["X_Request_ID"].annotation is Optional[str] 
        assert ArgsModel.model_fields["X_Request_ID"].json_schema_extra is not None
        assert ArgsModel.model_fields["X_Request_ID"].json_schema_extra.get("param_name") == "X-Request-ID" # type: ignore


    def test_create_args_model_only_request_body(self):
        spec_dict = {
            "openapi": "3.0.0", "info": {"title": "Test", "version": "1"},
            "paths": {
                "/submit": {
                    "post": {
                        "operationId": "submit_data",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "value": {"type": "number", "default": 0.0}
                                        },
                                        "required": ["name"]
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "ok"}}
                    }
                }
            }
        }
        spec = OpenAPIObject.model_validate(spec_dict)
        builder = OpenAPIToolBuilder(spec_dict)
        operation = spec.paths.root["/submit"].post # type: ignore
        assert operation is not None
        ArgsModel = builder._create_args_model("SubmitData", operation) 

        assert "request_body" in ArgsModel.model_fields
        RequestBodyModel = ArgsModel.model_fields["request_body"].annotation
        
        assert RequestBodyModel.__name__ == "SubmitDataRequestBody"
        assert "name" in RequestBodyModel.model_fields
        assert RequestBodyModel.model_fields["name"].annotation is str
        assert RequestBodyModel.model_fields["name"].is_required()
        
        assert "value" in RequestBodyModel.model_fields
        assert RequestBodyModel.model_fields["value"].annotation is Optional[float]
        assert RequestBodyModel.model_fields["value"].default == 0.0


    def test_create_args_model_mixed_params_and_body(self, spec_with_params_and_body: OpenAPIObject, spec_with_params_and_body_dict: Dict[str,Any]):
        builder = OpenAPIToolBuilder(spec_with_params_and_body_dict) 
        operation = spec_with_params_and_body.paths.root["/items/{item_id}"].post # type: ignore
        assert operation is not None
        ArgsModel = builder._create_args_model("CreateItem", operation) 

        assert "item_id" in ArgsModel.model_fields
        assert ArgsModel.model_fields["item_id"].annotation is int
        assert ArgsModel.model_fields["item_id"].is_required()
        assert ArgsModel.model_fields["item_id"].description == "ID of the item"

        assert "api_version" in ArgsModel.model_fields 
        assert ArgsModel.model_fields["api_version"].annotation is Optional[str]
        assert ArgsModel.model_fields["api_version"].default == "v1"
        assert ArgsModel.model_fields["api_version"].json_schema_extra is not None
        assert ArgsModel.model_fields["api_version"].json_schema_extra.get("param_name") == "api-version" # type: ignore

        assert "category" in ArgsModel.model_fields
        assert ArgsModel.model_fields["category"].annotation.__origin__ is Union # Optional[CategoryEnum] type: ignore
        assert issubclass(ArgsModel.model_fields["category"].annotation.__args__[0], PyEnum)  # type: ignore
        assert ArgsModel.model_fields["category"].annotation.__args__[0].__name__ == "CreateitemCategoryEnum" # type: ignore

        assert "session_id" in ArgsModel.model_fields
        assert ArgsModel.model_fields["session_id"].annotation is Optional[str]

        assert "request_body" in ArgsModel.model_fields
        RequestBodyModel = ArgsModel.model_fields["request_body"].annotation
        assert RequestBodyModel.__name__ == "CreateItemRequestBody"
        
        assert "name" in RequestBodyModel.model_fields
        assert RequestBodyModel.model_fields["name"].annotation is str
        assert RequestBodyModel.model_fields["name"].is_required()
        assert RequestBodyModel.model_fields["name"].description == "Name of the item"

        assert "price" in RequestBodyModel.model_fields
        assert RequestBodyModel.model_fields["price"].annotation is float 
        assert RequestBodyModel.model_fields["price"].is_required()

        assert "is_active" in RequestBodyModel.model_fields
        assert RequestBodyModel.model_fields["is_active"].annotation is Optional[bool]
        assert RequestBodyModel.model_fields["is_active"].default is True 
        
        assert "tags" in RequestBodyModel.model_fields
        assert RequestBodyModel.model_fields["tags"].annotation == Optional[List[str]]


# --- Integration-style Tests for OpenAPIToolBuilder.build_tools ---

class TestOpenAPIToolBuilderIntegration:
    
    @patch('hippycampus.openapi_builder.requests.request')
    def test_build_tools_simple_get_no_auth(self, mock_request: MagicMock, minimal_openapi_spec_dict: Dict[str,Any]):
        builder = OpenAPIToolBuilder(minimal_openapi_spec_dict)
        tools = builder.build_tools()

        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get_test_operation"
        assert tool.description == "A simple GET request." 
        assert tool.args_schema is not None
        assert not tool.args_schema.model_fields 

        mock_response = MagicMock()
        mock_response.status_code = 200
        # Ensure .json() method exists and returns a dict for successful json parsing
        mock_response.json.return_value = {"message": "success"}
        mock_response.text = '{"message": "success"}' # Fallback if .json() fails
        mock_request.return_value = mock_response

        tool_result = tool._run() 
        
        mock_request.assert_called_once_with(
            method="get",
            url="http://localhost:8000/api/v1/test_get", 
            headers={'Content-Type': 'application/json'}, 
            params={},
            json=None
        )
        assert tool_result == '{"message": "success"}' 


    @patch('hippycampus.openapi_builder.requests.request')
    def test_build_tools_post_with_params_body_no_auth(self, mock_request: MagicMock, spec_with_params_and_body_dict: Dict[str, Any]):
        builder = OpenAPIToolBuilder(spec_with_params_and_body_dict)
        tools = builder.build_tools()

        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "create_item"
        assert tool.args_schema is not None

        assert "item_id" in tool.args_schema.model_fields
        assert "api_version" in tool.args_schema.model_fields 
        assert "category" in tool.args_schema.model_fields
        assert "session_id" in tool.args_schema.model_fields
        assert "request_body" in tool.args_schema.model_fields

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1, "name": "Test Item"}
        mock_request.return_value = mock_response
        
        tool_input = {
            "item_id": 123,
            "category": "electronics",
            "api_version": "v2", 
            "request_body": {
                "name": "Test Item",
                "price": 99.99,
                "tags": ["new", "featured"]
            }
        }
        args_instance = tool.args_schema(**tool_input) # type: ignore
        tool_result = tool._run(**args_instance.model_dump())


        expected_headers = {
            'Content-Type': 'application/json',
            'api-version': 'v2' 
        }
        
        mock_request.assert_called_once_with(
            method="post",
            url="https://api.example.com/items/123",
            headers=expected_headers,
            params={"category": "electronics"},
            json={"name": "Test Item", "price": 99.99, "tags": ["new", "featured"], "is_active": True} 
        )
        assert tool_result == '{"id": 1, "name": "Test Item"}'


    @patch('hippycampus.openapi_builder.requests.request')
    def test_build_tools_api_key_auth(self, mock_request: MagicMock, spec_with_auth_api_key_dict: Dict[str, Any]):
        mock_auth_instance = MagicMock(spec=AbstractAuth)
        mock_auth_instance.get_auth_headers.return_value = {"X-API-Key": "test_api_key_value"}
        
        builder = OpenAPIToolBuilder(spec_with_auth_api_key_dict) 
        tools = builder.build_tools(auth=mock_auth_instance) 
        
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get_secure_resource"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "secure data"}
        mock_request.return_value = mock_response

        tool_result = tool._run()

        expected_headers = {
            'Content-Type': 'application/json',
            'X-API-Key': 'test_api_key_value' 
        }
        
        mock_request.assert_called_once_with(
            method="get",
            url="http://localhost:8000/secure_resource",
            headers=expected_headers,
            params={},
            json=None
        )
        assert tool_result == '{"data": "secure data"}'
        mock_auth_instance.get_auth_headers.assert_called_once() 


    @patch('hippycampus.openapi_builder.requests.request')
    def test_build_tools_bearer_auth(self, mock_request: MagicMock, spec_with_auth_bearer_dict: Dict[str, Any]):
        mock_auth_instance = MagicMock(spec=AbstractAuth)
        mock_auth_instance.get_auth_headers.return_value = {"Authorization": "Bearer test_bearer_token"}

        builder = OpenAPIToolBuilder(spec_with_auth_bearer_dict)
        tools = builder.build_tools(auth=mock_auth_instance)
        tool = tools[0]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "bearer data"}
        mock_request.return_value = mock_response

        tool._run()
        
        expected_headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test_bearer_token'
        }
        mock_request.assert_called_once_with(
            method="get",
            url="http://localhost:8000/bearer_resource",
            headers=expected_headers,
            params={},
            json=None
        )
        mock_auth_instance.get_auth_headers.assert_called_once()

    @patch('hippycampus.openapi_builder.requests.request')
    def test_build_tools_basic_auth(self, mock_request: MagicMock, spec_with_auth_basic_dict: Dict[str, Any]):
        mock_auth_instance = MagicMock(spec=AbstractAuth)
        mock_auth_instance.get_auth_headers.return_value = {"Authorization": "Basic dXNlcjpwYXNz"} 

        builder = OpenAPIToolBuilder(spec_with_auth_basic_dict)
        tools = builder.build_tools(auth=mock_auth_instance)
        tool = tools[0]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "basic data"}
        mock_request.return_value = mock_response

        tool._run()
        
        expected_headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Basic dXNlcjpwYXNz'
        }
        mock_request.assert_called_once_with(
            method="get",
            url="http://localhost:8000/basic_resource",
            headers=expected_headers, 
            params={},
            json=None
        )
        mock_auth_instance.get_auth_headers.assert_called_once()


    def test_build_tools_special_op_id_sanitization(self, spec_with_special_op_id_dict: Dict[str, Any]):
        builder = OpenAPIToolBuilder(spec_with_special_op_id_dict)
        tools = builder.build_tools()
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "_123_get_data_for_user_" 


# --- Placeholder for refactoring and security hardening notes/tests ---
class TestRefactoringAndSecurity:
    pass

if __name__ == "__main__":
    pytest.main()
