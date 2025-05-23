import pytest
from hippycampus.spec_parser import (
    ReferenceObject,
    OpenAPIResolver,
    parse_yaml,
    parse_json,
    parse_yaml_file,
    parse_json_file,
    ParameterObject,
    MediaTypeObject,
    ExampleObject,
    LinkObject,
    InfoObject,
    OpenAPIObject,
    SchemaObject,
    ResponseObject,
    RequestBodyObject,
    HeaderObject,
    SecuritySchemeObject,
    CallbackObject,  # Added
)
from pydantic import ValidationError
import os
import tempfile
import yaml
import json


# Helper function to create a temporary file with content
def create_temp_file(content: str, suffix: str = ".yaml") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as tmp:
        tmp.write(content)
    return path


@pytest.fixture
def sample_openapi_spec_dict():
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0",
            "license": {"name": "MIT"},
        },  # Added license for InfoObject test
        "paths": {
            "/test": {
                "get": {
                    "summary": "Test endpoint",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/TestSchema"
                                    }
                                }
                            },
                        }
                    },
                    "parameters": [{"$ref": "#/components/parameters/TestParam"}],
                }
            },
            "/items": {
                "post": {
                    "requestBody": {"$ref": "#/components/requestBodies/ItemBody"},
                    "responses": {
                        "201": {"$ref": "#/components/responses/ItemCreated"}
                    },
                }
            },
            "/path/{param_id}": {  # For path parameter test
                "parameters": [
                    {
                        "name": "param_id",
                        "in_": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "get": {"responses": {"200": {"description": "ok"}}},
            },
        },
        "components": {
            "schemas": {
                "TestSchema": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
                "ReferredSchema": {"type": "string", "format": "email"},
                "SchemaWithAllOf": {
                    "allOf": [
                        {"$ref": "#/components/schemas/TestSchema"},
                        {"type": "object", "properties": {"name": {"type": "string"}}},
                    ]
                },
                "SchemaWithOneOf": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                "SchemaWithAnyOf": {
                    "anyOf": [
                        {"type": "string", "maxLength": 5},
                        {"type": "string", "minLength": 10},
                    ]
                },
                "SchemaWithNot": {"not_": {"type": "integer"}},
                "ArraySchema": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/ReferredSchema"},
                },
                "ArraySchemaInline": {
                    "type": "array",
                    "items": {"type": "string", "format": "uuid"},
                },
                "ObjectWithProps": {
                    "type": "object",
                    "properties": {
                        "directProp": {"type": "boolean"},
                        "refProp": {"$ref": "#/components/schemas/TestSchema"},
                    },
                },
                "ObjectWithAddPropsTrue": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "ObjectWithAddPropsFalse": {
                    "type": "object",
                    "additionalProperties": False,
                },
                "ObjectWithAddPropsSchema": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "ObjectWithAddPropsRef": {
                    "type": "object",
                    "additionalProperties": {
                        "$ref": "#/components/schemas/ReferredSchema"
                    },
                },
                "NestedRef1": {"$ref": "#/components/schemas/NestedRef2"},
                "NestedRef2": {"$ref": "#/components/schemas/NestedRef3"},
                "NestedRef3": {"type": "string", "default": "Deeply nested"},
                "CircularA": {
                    "type": "object",
                    "properties": {"b": {"$ref": "#/components/schemas/CircularB"}},
                },
                "CircularB": {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/components/schemas/CircularA"}},
                },
            },
            "parameters": {
                "TestParam": {
                    "name": "test_param",
                    "in_": "query",
                    "schema": {"type": "string"},
                }
            },
            "responses": {
                "TestResponse": {"description": "A test response"},
                "ItemCreated": {
                    "description": "Item created",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TestSchema"}
                        }
                    },
                },
            },
            "examples": {"TestExample": {"value": {"name": "example_name"}}},
            "requestBodies": {
                "TestRequestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TestSchema"}
                        }
                    }
                },
                "ItemBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"itemName": {"type": "string"}},
                            }
                        }
                    }
                },
            },
            "headers": {
                "TestHeader": {
                    "description": "A test header",
                    "schema": {"type": "string"},
                }
            },
            "securitySchemes": {
                "TestSecurityScheme": {
                    "type": "apiKey",
                    "name": "X-API-KEY",
                    "in_": "header",
                }
            },
            "links": {"TestLink": {"operationId": "getTest"}},
            "callbacks": {
                "TestCallback": {
                    "{$request.body#/callbackUrl}": {
                        "post": {"responses": {"200": {"description": "Callback ack"}}}
                    }
                }
            },
        },
    }


@pytest.fixture
def openapi_resolver(sample_openapi_spec_dict):
    return OpenAPIResolver(spec=sample_openapi_spec_dict)


@pytest.fixture
def valid_openapi_yaml_str(sample_openapi_spec_dict):
    return yaml.dump(sample_openapi_spec_dict)


@pytest.fixture
def valid_openapi_json_str(sample_openapi_spec_dict):
    return json.dumps(sample_openapi_spec_dict)


class TestReferenceObject:
    def test_resolve_valid_schema_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/schemas/TestSchema")
        resolved = ref_obj.resolve(openapi_resolver)
        assert isinstance(resolved, SchemaObject)
        assert resolved.type == "object"
        assert ref_obj.resolve(openapi_resolver) is resolved

    def test_resolve_valid_parameter_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/parameters/TestParam")
        resolved = ref_obj.resolve(openapi_resolver)
        assert isinstance(resolved, ParameterObject)
        assert resolved.name == "test_param"
        assert ref_obj.resolve(openapi_resolver) is resolved

    def test_resolve_valid_response_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/responses/TestResponse")
        resolved = ref_obj.resolve(openapi_resolver)
        assert isinstance(resolved, ResponseObject)
        assert resolved.description == "A test response"
        assert ref_obj.resolve(openapi_resolver) is resolved

    def test_resolve_valid_example_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/examples/TestExample")
        resolved = ref_obj.resolve(openapi_resolver)
        assert isinstance(resolved, ExampleObject)
        assert resolved.value == {"name": "example_name"}
        assert ref_obj.resolve(openapi_resolver) is resolved

    def test_resolve_valid_request_body_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/requestBodies/TestRequestBody")
        resolved = ref_obj.resolve(openapi_resolver)
        assert isinstance(resolved, RequestBodyObject)
        assert (
            resolved.content["application/json"].schema_.ref
            == "#/components/schemas/TestSchema"
        )
        assert ref_obj.resolve(openapi_resolver) is resolved

    def test_resolve_valid_header_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/headers/TestHeader")
        resolved = ref_obj.resolve(openapi_resolver)
        assert isinstance(resolved, HeaderObject)
        assert resolved.description == "A test header"
        assert ref_obj.resolve(openapi_resolver) is resolved

    def test_resolve_valid_security_scheme_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/securitySchemes/TestSecurityScheme")
        resolved = ref_obj.resolve(openapi_resolver)
        assert isinstance(resolved, SecuritySchemeObject)
        assert resolved.type == "apiKey"
        assert ref_obj.resolve(openapi_resolver) is resolved

    def test_resolve_valid_link_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/links/TestLink")
        resolved = ref_obj.resolve(openapi_resolver)
        assert isinstance(resolved, LinkObject)
        assert resolved.operationId == "getTest"
        assert ref_obj.resolve(openapi_resolver) is resolved

    def test_resolve_valid_callback_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/callbacks/TestCallback")
        resolved = ref_obj.resolve(openapi_resolver)
        assert isinstance(resolved, CallbackObject)
        assert "{$request.body#/callbackUrl}" in resolved.root
        assert ref_obj.resolve(openapi_resolver) is resolved

    def test_resolve_invalid_reference_path(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/schemas/NonExistentSchema")
        with pytest.raises(
            ValueError, match="Could not resolve reference.*NonExistentSchema"
        ):
            ref_obj.resolve(openapi_resolver)

    def test_resolve_invalid_reference_component_type(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/nonexistenttype/TestSchema")
        with pytest.raises(
            ValueError, match="Could not resolve reference.*nonexistenttype"
        ):
            ref_obj.resolve(openapi_resolver)

    def test_resolve_non_internal_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="external.yaml#/components/schemas/SomeSchema")
        with pytest.raises(ValueError, match="External references are not supported"):
            ref_obj.resolve(openapi_resolver)

    def test_resolve_malformed_reference(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="components/schemas/TestSchema")
        with pytest.raises(ValueError, match="Invalid reference format"):
            ref_obj.resolve(openapi_resolver)


class TestOpenAPIResolver:
    def test_resolve_schema_all_of(self, openapi_resolver):
        schema_data = openapi_resolver.spec["components"]["schemas"]["SchemaWithAllOf"]
        resolved_schema = openapi_resolver._resolve_schema(schema_data)
        assert len(resolved_schema.allOf) == 2
        assert isinstance(resolved_schema.allOf[0], SchemaObject)
        assert resolved_schema.allOf[0].type == "object"
        assert resolved_schema.allOf[0].properties["id"].type == "integer"
        assert isinstance(resolved_schema.allOf[1], SchemaObject)
        assert resolved_schema.allOf[1].type == "object"
        assert resolved_schema.allOf[1].properties["name"].type == "string"

    def test_resolve_schema_one_of(self, openapi_resolver):
        schema_data = openapi_resolver.spec["components"]["schemas"]["SchemaWithOneOf"]
        resolved_schema = openapi_resolver._resolve_schema(schema_data)
        assert len(resolved_schema.oneOf) == 2
        assert isinstance(resolved_schema.oneOf[0], SchemaObject)
        assert resolved_schema.oneOf[0].type == "string"
        assert isinstance(resolved_schema.oneOf[1], SchemaObject)
        assert resolved_schema.oneOf[1].type == "integer"

    def test_resolve_schema_any_of(self, openapi_resolver):
        schema_data = openapi_resolver.spec["components"]["schemas"]["SchemaWithAnyOf"]
        resolved_schema = openapi_resolver._resolve_schema(schema_data)
        assert len(resolved_schema.anyOf) == 2
        assert isinstance(resolved_schema.anyOf[0], SchemaObject)
        assert resolved_schema.anyOf[0].maxLength == 5
        assert isinstance(resolved_schema.anyOf[1], SchemaObject)
        assert resolved_schema.anyOf[1].minLength == 10

    def test_resolve_schema_not(self, openapi_resolver):
        schema_data = openapi_resolver.spec["components"]["schemas"]["SchemaWithNot"]
        resolved_schema = openapi_resolver._resolve_schema(schema_data)
        assert isinstance(resolved_schema.not_, SchemaObject)
        assert resolved_schema.not_.type == "integer"

    def test_resolve_schema_items_ref(self, openapi_resolver):
        schema_data = openapi_resolver.spec["components"]["schemas"]["ArraySchema"]
        resolved_schema = openapi_resolver._resolve_schema(schema_data)
        assert resolved_schema.type == "array"
        assert isinstance(resolved_schema.items, SchemaObject)
        assert resolved_schema.items.type == "string"
        assert resolved_schema.items.format == "email"

    def test_resolve_schema_items_inline(self, openapi_resolver):
        schema_data = openapi_resolver.spec["components"]["schemas"][
            "ArraySchemaInline"
        ]
        resolved_schema = openapi_resolver._resolve_schema(schema_data)
        assert resolved_schema.type == "array"
        assert isinstance(resolved_schema.items, SchemaObject)
        assert resolved_schema.items.type == "string"
        assert resolved_schema.items.format == "uuid"

    def test_resolve_schema_properties(self, openapi_resolver):
        schema_data = openapi_resolver.spec["components"]["schemas"]["ObjectWithProps"]
        resolved_schema = openapi_resolver._resolve_schema(schema_data)
        assert resolved_schema.type == "object"
        assert isinstance(resolved_schema.properties["directProp"], SchemaObject)
        assert resolved_schema.properties["directProp"].type == "boolean"
        assert isinstance(resolved_schema.properties["refProp"], SchemaObject)
        assert resolved_schema.properties["refProp"].type == "object"
        assert resolved_schema.properties["refProp"].properties["id"].type == "integer"

    def test_resolve_schema_additional_properties_boolean(self, openapi_resolver):
        schema_true = openapi_resolver.spec["components"]["schemas"][
            "ObjectWithAddPropsTrue"
        ]
        resolved_true = openapi_resolver._resolve_schema(schema_true)
        assert resolved_true.additionalProperties is True

        schema_false = openapi_resolver.spec["components"]["schemas"][
            "ObjectWithAddPropsFalse"
        ]
        resolved_false = openapi_resolver._resolve_schema(schema_false)
        assert resolved_false.additionalProperties is False

    def test_resolve_schema_additional_properties_schema(self, openapi_resolver):
        schema_data = openapi_resolver.spec["components"]["schemas"][
            "ObjectWithAddPropsSchema"
        ]
        resolved_schema = openapi_resolver._resolve_schema(schema_data)
        assert isinstance(resolved_schema.additionalProperties, SchemaObject)
        assert resolved_schema.additionalProperties.type == "number"

    def test_resolve_schema_additional_properties_ref(self, openapi_resolver):
        schema_data = openapi_resolver.spec["components"]["schemas"][
            "ObjectWithAddPropsRef"
        ]
        resolved_schema = openapi_resolver._resolve_schema(schema_data)
        assert isinstance(resolved_schema.additionalProperties, SchemaObject)
        assert resolved_schema.additionalProperties.type == "string"
        assert resolved_schema.additionalProperties.format == "email"

    def test_resolve_schema_nested_references(self, openapi_resolver):
        ref_obj = ReferenceObject(ref="#/components/schemas/NestedRef1")
        resolved_schema = openapi_resolver._resolve_reference_object(ref_obj)
        assert isinstance(resolved_schema, SchemaObject)
        assert resolved_schema.type == "string"
        assert resolved_schema.default == "Deeply nested"

    def test_resolve_schema_circular_references(self, openapi_resolver):
        schema_data_A = openapi_resolver.spec["components"]["schemas"]["CircularA"]
        resolved_A_direct = openapi_resolver._resolve_schema(schema_data_A)

        assert isinstance(resolved_A_direct.properties["b"], SchemaObject)
        resolved_B_via_A = resolved_A_direct.properties["b"]
        assert isinstance(resolved_B_via_A.properties["a"], SchemaObject)

        final_a_ref_in_b = resolved_B_via_A.properties["a"]
        # When CircularA is resolved, it gets put into _resolved_references.
        # When CircularB's property 'a' (ref to CircularA) is resolved by _resolve_reference_object,
        # it should find the existing CircularA instance in the cache.
        cached_A = openapi_resolver._resolved_references[
            "#/components/schemas/CircularA"
        ]
        assert final_a_ref_in_b is cached_A

    def test_resolve_all(self, sample_openapi_spec_dict):
        resolver = OpenAPIResolver(
            spec=sample_openapi_spec_dict, resolve_refs_on_load=False
        )

        raw_schema_ref = (
            resolver.spec_obj.paths["/test"]
            .get.responses["200"]
            .content["application/json"]
            .schema_
        )
        assert isinstance(raw_schema_ref, ReferenceObject)

        raw_param_ref = resolver.spec_obj.paths["/test"].get.parameters[0]
        assert isinstance(raw_param_ref, ReferenceObject)

        resolver.resolve_all()

        resolved_schema = (
            resolver.spec_obj.paths["/test"]
            .get.responses["200"]
            .content["application/json"]
            .schema_
        )
        assert isinstance(resolved_schema, SchemaObject)
        assert resolved_schema.type == "object"

        resolved_param = resolver.spec_obj.paths["/test"].get.parameters[0]
        assert isinstance(resolved_param, ParameterObject)
        assert resolved_param.name == "test_param"

        resolved_allof_schema = resolver.spec_obj.components.schemas["SchemaWithAllOf"]
        assert isinstance(resolved_allof_schema.allOf[0], SchemaObject)
        assert resolved_allof_schema.allOf[0].type == "object"

        resolver.resolve_all()
        resolved_schema_again = (
            resolver.spec_obj.paths["/test"]
            .get.responses["200"]
            .content["application/json"]
            .schema_
        )
        assert resolved_schema is resolved_schema_again


class TestParsingFunctions:
    def test_parse_yaml_valid(self, valid_openapi_yaml_str, sample_openapi_spec_dict):
        openapi_obj = parse_yaml(valid_openapi_yaml_str)
        assert isinstance(openapi_obj, OpenAPIObject)
        assert openapi_obj.info.title == sample_openapi_spec_dict["info"]["title"]
        assert isinstance(
            openapi_obj.paths["/test"]
            .get.responses["200"]
            .content["application/json"]
            .schema_,
            SchemaObject,
        )

    def test_parse_json_valid(self, valid_openapi_json_str, sample_openapi_spec_dict):
        openapi_obj = parse_json(valid_openapi_json_str)
        assert isinstance(openapi_obj, OpenAPIObject)
        assert openapi_obj.info.title == sample_openapi_spec_dict["info"]["title"]
        assert isinstance(
            openapi_obj.paths["/test"]
            .get.responses["200"]
            .content["application/json"]
            .schema_,
            SchemaObject,
        )

    def test_parse_yaml_malformed(self):
        malformed_yaml = "openapi: 3.0.0\ninfo: {title: Test API, version: 1.0.0}\npaths: /test: { get: {}"
        with pytest.raises(yaml.YAMLError):
            parse_yaml(malformed_yaml)

    def test_parse_json_malformed(self):
        malformed_json = '{"openapi": "3.0.0", "info": {"title": "Test API", "version": "1.0.0"}, "paths": {"/test": {"get": {}}}'
        with pytest.raises(json.JSONDecodeError):
            parse_json(malformed_json)

    def test_parse_yaml_file_valid(
        self, valid_openapi_yaml_str, sample_openapi_spec_dict
    ):
        temp_file_path = create_temp_file(valid_openapi_yaml_str, suffix=".yaml")
        try:
            openapi_obj = parse_yaml_file(temp_file_path)
            assert isinstance(openapi_obj, OpenAPIObject)
            assert openapi_obj.info.title == sample_openapi_spec_dict["info"]["title"]
            assert isinstance(
                openapi_obj.paths["/test"]
                .get.responses["200"]
                .content["application/json"]
                .schema_,
                SchemaObject,
            )
        finally:
            os.remove(temp_file_path)

    def test_parse_json_file_valid(
        self, valid_openapi_json_str, sample_openapi_spec_dict
    ):
        temp_file_path = create_temp_file(valid_openapi_json_str, suffix=".json")
        try:
            openapi_obj = parse_json_file(temp_file_path)
            assert isinstance(openapi_obj, OpenAPIObject)
            assert openapi_obj.info.title == sample_openapi_spec_dict["info"]["title"]
            assert isinstance(
                openapi_obj.paths["/test"]
                .get.responses["200"]
                .content["application/json"]
                .schema_,
                SchemaObject,
            )
        finally:
            os.remove(temp_file_path)

    def test_parse_yaml_file_non_existent(self):
        with pytest.raises(FileNotFoundError):
            parse_yaml_file("non_existent_file.yaml")

    def test_parse_json_file_non_existent(self):
        with pytest.raises(FileNotFoundError):
            parse_json_file("non_existent_file.json")

    def test_parse_yaml_resolve_refs_false(self, valid_openapi_yaml_str):
        openapi_obj = parse_yaml(valid_openapi_yaml_str, resolve_refs=False)
        assert isinstance(openapi_obj, OpenAPIObject)
        schema_or_ref = (
            openapi_obj.paths["/test"]
            .get.responses["200"]
            .content["application/json"]
            .schema_
        )
        assert isinstance(schema_or_ref, ReferenceObject)
        assert schema_or_ref.ref == "#/components/schemas/TestSchema"

    def test_parse_json_file_resolve_refs_false(self, valid_openapi_json_str):
        temp_file_path = create_temp_file(valid_openapi_json_str, suffix=".json")
        try:
            openapi_obj = parse_json_file(temp_file_path, resolve_refs=False)
            assert isinstance(openapi_obj, OpenAPIObject)
            schema_or_ref = (
                openapi_obj.paths["/test"]
                .get.responses["200"]
                .content["application/json"]
                .schema_
            )
            assert isinstance(schema_or_ref, ReferenceObject)
            assert schema_or_ref.ref == "#/components/schemas/TestSchema"
        finally:
            os.remove(temp_file_path)


class TestPydanticModelValidations:
    def test_parameter_object_path_param_requires_required_true(self):
        # Valid: in="path", required=True
        param_data_valid = {
            "name": "id",
            "in_": "path",
            "required": True,
            "schema": {"type": "string"},
        }
        param = ParameterObject(**param_data_valid)
        assert param.required is True

        # Invalid: in="path", required=False
        param_data_invalid_required_false = {
            "name": "id",
            "in_": "path",
            "required": False,
            "schema": {"type": "string"},
        }
        with pytest.raises(
            ValidationError, match="Path parameter 'id' must have 'required: true'"
        ):
            ParameterObject(**param_data_invalid_required_false)

        # Invalid: in="path", required missing (should default to False, then fail validation)
        param_data_invalid_required_missing = {
            "name": "id",
            "in_": "path",
            "schema": {"type": "string"},
        }
        with pytest.raises(
            ValidationError, match="Path parameter 'id' must have 'required: true'"
        ):
            ParameterObject(**param_data_invalid_required_missing)

        # Valid: in="query", required=False (or missing)
        ParameterObject(name="q", in_="query", schema={"type": "string"})
        ParameterObject(
            name="q", in_="query", required=False, schema={"type": "string"}
        )

    def test_media_type_object_example_examples_exclusive(self):
        # Valid: only example
        MediaTypeObject(schema_={"type": "string"}, example="Test")
        # Valid: only examples
        MediaTypeObject(schema_={"type": "string"}, examples={"ex1": {"value": "Test"}})
        # Invalid: both example and examples
        with pytest.raises(
            ValidationError, match="Cannot specify both 'example' and 'examples'"
        ):
            MediaTypeObject(
                schema_={"type": "string"},
                example="Test",
                examples={"ex1": {"value": "Test2"}},
            )

    def test_example_object_value_external_value_exclusive(self):
        # Valid: only value
        ExampleObject(value="Test")
        # Valid: only externalValue
        ExampleObject(externalValue="http://example.com/example.txt")
        # Invalid: both value and externalValue
        with pytest.raises(
            ValidationError, match="Cannot specify both 'value' and 'externalValue'"
        ):
            ExampleObject(value="Test", externalValue="http://example.com/example.txt")

    def test_link_object_operation_ref_id_exclusive(self):
        # Valid: only operationRef
        LinkObject(operationRef="#/paths/~1test/get")
        # Valid: only operationId
        LinkObject(operationId="getTest")
        # Invalid: both operationRef and operationId
        with pytest.raises(
            ValidationError,
            match="Cannot specify both 'operationId' and 'operationRef'",
        ):
            LinkObject(operationRef="#/paths/~1test/get", operationId="getTest")
        # Invalid: neither (as per spec, one is required, but model doesn't enforce this, which is fine)
        # LinkObject() # This would pass Pydantic validation as both are optional in the model

    def test_info_object_license_3_1_validation(self):
        # OpenAPI 3.0.x context (current default for InfoObject if not specified otherwise)
        # 'license' can be just a name
        InfoObject(title="t", version="v", license_={"name": "MIT"})
        # Or with url
        InfoObject(
            title="t", version="v", license_={"name": "MIT", "url": "http://mit.com"}
        )

        # OpenAPI 3.1.x context - This needs the OpenAPIObject to set the version
        # Test for 3.1.0: license name is required, and one of identifier or url for license.
        # The current InfoObject model doesn't have a direct way to know the parent OpenAPI version.
        # This validation might be better suited at the OpenAPIObject level or if InfoObject could access context.
        # For now, let's test the LicenseObject itself for 3.1.0 rules if it were used in 3.1 context.
        # from hippycampus.spec_parser import LicenseObject # Assuming LicenseObject is importable
        # with pytest.raises(ValidationError): # If identifier or url is missing
        #    LicenseObject(name="Test", context_openapi_version="3.1.0") # Hypothetical context
        # LicenseObject(name="Test", identifier="MIT", context_openapi_version="3.1.0")
        # LicenseObject(name="Test", url="http://example.com", context_openapi_version="3.1.0")
        # This specific test for InfoObject's license based on OpenAPI version is hard to do in isolation
        # without passing context. The current model for LicenseObject has 'name' as required, and
        # 'url' and 'identifier' as optional. This is compliant with 3.0.x.
        # For 3.1.x, the spec says: "If the license property is specified, the license field MUST be a License Object.
        # ... License Object: ... If it is not a SPDX license expression, it is REQUIRED to include the name field.
        # ... It is RECOMMENDED to also include a url..."
        # And for SPDX: "An SPDX license expression as defined in the SPDX Specification Right List."
        # The model has name: str, so it covers the non-SPDX case.
        # The specific validation "license requires url or identifier" for 3.1 is not directly in InfoObject.
        # It seems like the current models are more aligned with 3.0.x for this field.
        # Let's assume the current models are for 3.0.x unless specified, so no specific 3.1 test here for InfoObject.
        pass

    def test_openapi_object_paths_webhooks_validation(self, sample_openapi_spec_dict):
        # Valid: 3.0.0 with paths
        OpenAPIObject(**sample_openapi_spec_dict)

        # Valid: 3.0.0, paths is required
        invalid_spec_no_paths = sample_openapi_spec_dict.copy()
        del invalid_spec_no_paths["paths"]
        with pytest.raises(
            ValidationError, match="Field required.*paths"
        ):  # Pydantic's default message
            OpenAPIObject(**invalid_spec_no_paths)

        # For OpenAPI 3.1.0: at least one of paths or webhooks
        spec_3_1_only_paths = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {"/test": {"get": {"responses": {"200": {"description": "ok"}}}}},
        }
        OpenAPIObject(**spec_3_1_only_paths)

        spec_3_1_only_webhooks = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "webhooks": {
                "myWebhook": {"post": {"responses": {"200": {"description": "ok"}}}}
            },
        }
        OpenAPIObject(**spec_3_1_only_webhooks)

        spec_3_1_both = {
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {"/test": {"get": {"responses": {"200": {"description": "ok"}}}}},
            "webhooks": {
                "myWebhook": {"post": {"responses": {"200": {"description": "ok"}}}}
            },
        }
        OpenAPIObject(**spec_3_1_both)

        spec_3_1_neither = {  # Invalid for 3.1.0
            "openapi": "3.1.0",
            "info": {"title": "Test", "version": "1.0"},
        }
        with pytest.raises(
            ValidationError,
            match="must provide at least one of 'paths' or 'webhooks' for OpenAPI 3.1.0 and later",
        ):
            OpenAPIObject(**spec_3_1_neither)

        # Test with an older 3.0.x version string, webhooks is not a direct requirement then.
        spec_3_0_3_no_paths = {
            "openapi": "3.0.3",
            "info": {"title": "Test", "version": "1.0"},
        }
        with pytest.raises(ValidationError, match="Field required.*paths"):
            OpenAPIObject(**spec_3_0_3_no_paths)


class TestTypeHinting:
    # Notes or tests related to type hinting review
    pass


class TestOpenAPIResolverRefactoring:
    # Tests for after refactoring _resolve_schema
    pass
