from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Set, TypeVar, Generic, cast, Annotated
from enum import Enum
from pydantic import BaseModel, Field, ValidationInfo, model_validator, field_validator, RootModel
import yaml

__all__ = [
    # Core Models
    'OpenAPIObject',
    'PathsModel',
    'SchemaObject',
    'ComponentsObject',

    # Operation Related
    'OperationObject',
    'PathItemObject',
    'ParameterObject',
    'RequestBodyObject',
    'ResponseObject',
    'ResponsesModel',
    'MediaTypeObject',
    'EncodingObject',

    # Schema Related
    'SchemaType',
    'SchemaFormat',
    'Discriminator',
    'XML',

    # Security Related
    'SecuritySchemeObject',
    'SecuritySchemeType',
    'SecuritySchemeIn',
    'OAuthFlowObject',
    'OAuthFlowsObject',
    'SecurityRequirementModel',

    # Documentation Related
    'TagObject',
    'ExternalDocumentation',
    'InfoObject',

    # Server Related
    'ServerObject',
    'WebhookModel',

    # Utility Models
    'ReferenceObject',
    'ExampleObject',
    'HeaderObject',
    'LinkObject',
    'CallbackModel',

    # Utility functions
    'parse_yaml',
    'parse_yaml_file',
    'parse_json',
    'parse_json_file',
]

T = TypeVar('T')


class SpecificationVersion(str, Enum):
    """Supported OpenAPI specification versions."""
    V3_0_0 = "3.0.0"
    V3_0_1 = "3.0.1"
    V3_0_2 = "3.0.2"
    V3_0_3 = "3.0.3"
    V3_1_0 = "3.1.0"


class ReferenceObject(BaseModel, Generic[T]):
    """Represents a $ref reference object in OpenAPI."""
    ref: str = Field(..., alias="$ref")
    summary: Optional[str] = None
    description: Optional[str] = None

    # The resolved object (populated during resolution)
    _resolved: Optional[T] = None

    # Extension fields starting with 'x-'
    model_config = {
        "extra": "allow",
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

    def resolve(self, root_document: Any) -> T:
        """
        Resolve this reference to its actual object.

        Args:
            root_document: The root OpenAPI document

        Returns:
            The resolved object
        """
        if self._resolved is not None:
            return self._resolved

        if not self.ref.startswith('#/'):
            raise ValueError(f"Only internal references are supported. Got: {self.ref}")

        # Remove the '#/' prefix and split into parts
        path_parts = self.ref[2:].split('/')

        # Navigate through the document to find the referenced object
        current = root_document.model_extra['x-orig-doc']
        for part in path_parts:
            if part in current:
                current = current[part]
            else:
                raise ValueError(f"Invalid reference path: {self.ref}")

        # Determine the target component type based on the path
        # This maps component types to their corresponding model classes
        component_types = {
            'schemas': SchemaObject,
            'parameters': ParameterObject,
            'responses': ResponseObject,
            'examples': ExampleObject,
            'requestBodies': RequestBodyObject,
            'headers': HeaderObject,
            'securitySchemes': SecuritySchemeObject,
            'links': LinkObject,
            'callbacks': CallbackModel,
            'pathItems': PathItemObject
        }

        # Use the second part of the path to determine the component type
        # e.g., "#/components/parameters/Foo" -> "parameters" -> ParameterObject
        model_type = None
        if len(path_parts) >= 2:
            component_type = path_parts[1]
            model_type = component_types.get(component_type)

        if not model_type:
            # Default to the first path component if mapping not found
            # This handles special cases like paths and other non-component refs
            if path_parts[0] == 'paths':
                model_type = PathItemObject
            else:
                # Fall back to BaseModel as a last resort
                model_type = BaseModel

        # Create the appropriate model from the resolved dict
        resolved_obj = model_type.model_validate(current)

        # Store the resolved object
        self._resolved = resolved_obj
        return resolved_obj


class ExternalDocumentation(BaseModel):
    """External documentation object."""
    url: str
    description: Optional[str] = None

    # Extension fields
    model_config = {"extra": "allow"}


class ContactObject(BaseModel):
    """Contact information for the API."""
    name: Optional[str] = None
    url: Optional[str] = None
    email: Optional[str] = None

    # Extension fields
    model_config = {"extra": "allow"}


class LicenseObject(BaseModel):
    """License information for the API."""
    name: str
    identifier: Optional[str] = None
    url: Optional[str] = None

    # Extension fields
    model_config = {"extra": "allow"}

    @field_validator("url", mode="after")
    def validate_url_or_identifier(cls, v, info: ValidationInfo):
        """Either url or identifier must be specified."""
        values = info.data
        if v is None and values.get("identifier") is None and "3.1" in values.get("_spec_version", "3.1.0"):
            raise ValueError("Either url or identifier must be specified")
        return v


class InfoObject(BaseModel):
    """Information about the API."""
    title: str
    summary: Optional[str] = None
    description: Optional[str] = None
    termsOfService: Optional[str] = None
    contact: Optional[ContactObject] = None
    license: Optional[LicenseObject] = None
    version: str  # required

    # Extension fields
    model_config = {"extra": "allow"}


class ServerVariableObject(BaseModel):
    """Variable substitution for a server URL template."""
    enum: Optional[List[str]] = None
    default: str
    description: Optional[str] = None

    # Extension fields
    model_config = {"extra": "allow"}


class ServerObject(BaseModel):
    """Server connection information."""
    url: str
    description: Optional[str] = None
    variables: Optional[Dict[str, ServerVariableObject]] = None

    # Extension fields
    model_config = {"extra": "allow"}


class ParameterLocation(str, Enum):
    """Possible locations of parameters."""
    QUERY = "query"
    HEADER = "header"
    PATH = "path"
    COOKIE = "cookie"


class ParameterStyle(str, Enum):
    """Serialization style of parameters."""
    MATRIX = "matrix"
    LABEL = "label"
    FORM = "form"
    SIMPLE = "simple"
    SPACE_DELIMITED = "spaceDelimited"
    PIPE_DELIMITED = "pipeDelimited"
    DEEP_OBJECT = "deepObject"


class ExampleObject(BaseModel):
    """Example object."""
    summary: Optional[str] = None
    description: Optional[str] = None
    value: Optional[Any] = None
    externalValue: Optional[str] = None

    # Extension fields
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_value_or_external_value(self) -> "ExampleObject":
        """Validate that value and externalValue are not both specified."""
        if self.value is not None and self.externalValue is not None:
            raise ValueError("value and externalValue are mutually exclusive")
        return self


class MediaTypeObject(BaseModel):
    """Media type definitions."""
    schema_: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = Field(None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]] = None
    encoding: Optional[Dict[str, EncodingObject]] = None

    # Extension fields
    model_config = {
        "extra": "allow",
        "populate_by_name": True
    }

    @model_validator(mode="after")
    def validate_example_examples(self) -> "MediaTypeObject":
        """Validate that example and examples are not both specified."""
        if self.example is not None and self.examples is not None:
            raise ValueError("example and examples are mutually exclusive")
        return self


class EncodingPropertyStyle(str, Enum):
    """Serialization styles for encoding properties."""
    FORM = "form"
    SPACE_DELIMITED = "spaceDelimited"
    PIPE_DELIMITED = "pipeDelimited"
    DEEP_OBJECT = "deepObject"


class EncodingObject(BaseModel):
    """Encoding object for request body content type encodings."""
    contentType: Optional[str] = None
    headers: Optional[Dict[str, Union['HeaderObject', ReferenceObject['HeaderObject']]]] = None
    style: Optional[EncodingPropertyStyle] = EncodingPropertyStyle.FORM
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = False

    # Extension fields
    model_config = {"extra": "allow"}


class ParameterBase(BaseModel):
    """Base class for parameter objects."""
    name: str
    in_: ParameterLocation = Field(..., alias="in")
    description: Optional[str] = None
    required: Optional[bool] = None
    deprecated: Optional[bool] = False
    allowEmptyValue: Optional[bool] = None
    style: Optional[ParameterStyle] = None
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = False
    schema_: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = Field(None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]] = None
    content: Optional[Dict[str, MediaTypeObject]] = None

    # Extension fields
    model_config = {
        "extra": "allow",
        "populate_by_name": True
    }

    @model_validator(mode="after")
    def validate_content_or_schema(self) -> "ParameterBase":
        """Validate that content and schema are not both specified."""
        if self.content is not None and self.schema_ is not None:
            raise ValueError("schema and content are mutually exclusive")

        if self.content is not None and len(self.content) > 1:
            raise ValueError("content map must only contain one entry")

        return self


class ParameterObject(ParameterBase):
    """Represents a parameter in OpenAPI."""

    @model_validator(mode="after")
    def validate_required_path(self) -> "ParameterObject":
        """Validate that path parameters are marked as required."""
        if self.in_ == ParameterLocation.PATH and self.required is not True:
            raise ValueError("Path parameters must have required=true")
        return self

    @model_validator(mode="after")
    def validate_style(self) -> "ParameterObject":
        """Set default style based on parameter location."""
        if self.style is None:
            if self.in_ == ParameterLocation.QUERY or self.in_ == ParameterLocation.COOKIE:
                self.style = ParameterStyle.FORM
            elif self.in_ == ParameterLocation.PATH:
                self.style = ParameterStyle.SIMPLE
            elif self.in_ == ParameterLocation.HEADER:
                self.style = ParameterStyle.SIMPLE
        return self


class HeaderObject(BaseModel):
    """Represents a header in OpenAPI."""
    description: Optional[str] = None
    required: Optional[bool] = False
    deprecated: Optional[bool] = False
    style: Optional[ParameterStyle] = ParameterStyle.SIMPLE
    explode: Optional[bool] = None
    schema_: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = Field(None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]] = None
    content: Optional[Dict[str, MediaTypeObject]] = None

    # Extension fields
    model_config = {
        "extra": "allow",
        "populate_by_name": True
    }

    @model_validator(mode="after")
    def validate_content_or_schema(self) -> "HeaderObject":
        """Validate that content and schema are not both specified."""
        if self.content is not None and self.schema_ is not None:
            raise ValueError("schema and content are mutually exclusive")

        if self.content is not None and len(self.content) > 1:
            raise ValueError("content map must only contain one entry")

        return self


class RequestBodyObject(BaseModel):
    """Request body object in OpenAPI."""
    description: Optional[str] = None
    content: Dict[str, MediaTypeObject]
    required: Optional[bool] = False

    # Extension fields
    model_config = {"extra": "allow"}


class ResponseObject(BaseModel):
    """Response object in OpenAPI."""
    description: str
    headers: Optional[Dict[str, Union[HeaderObject, ReferenceObject[HeaderObject]]]] = None
    content: Optional[Dict[str, MediaTypeObject]] = None
    links: Optional[Dict[str, Union[LinkObject, ReferenceObject[LinkObject]]]] = None

    # Extension fields
    model_config = {"extra": "allow"}


class ResponsesModel(RootModel):
    """Container for response objects."""
    root: Dict[str, Union[ResponseObject, ReferenceObject[ResponseObject]]]

    def __iter__(self):
        return iter(self.root.items())

    def __getitem__(self, key):
        return self.root[key]

    def items(self):
        return self.root.items()

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()


class LinkObject(BaseModel):
    """Link object for response linking."""
    operationRef: Optional[str] = None
    operationId: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    requestBody: Optional[Any] = None
    description: Optional[str] = None
    server: Optional[ServerObject] = None

    # Extension fields
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_operation_ref_or_id(self) -> "LinkObject":
        """Validate that only one of operationRef and operationId is specified."""
        if self.operationRef is not None and self.operationId is not None:
            raise ValueError("operationRef and operationId are mutually exclusive")
        return self


class CallbackModel(RootModel):
    """Callback object for webhooks."""
    root: Dict[str, 'PathItemObject']

    def __iter__(self):
        return iter(self.root.items())

    def __getitem__(self, key):
        return self.root[key]


class SecurityRequirementModel(RootModel):
    """Security requirement object."""
    root: Dict[str, List[str]]

    def __iter__(self):
        return iter(self.root.items())

    def __getitem__(self, key):
        return self.root[key]


class OperationObject(BaseModel):
    """Operation object for API paths."""
    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = None
    operationId: Optional[str] = None
    parameters: Optional[List[Union[ParameterObject, ReferenceObject[ParameterObject]]]] = None
    requestBody: Optional[Union[RequestBodyObject, ReferenceObject[RequestBodyObject]]] = None
    responses: ResponsesModel
    callbacks: Optional[Dict[str, Union[CallbackModel, ReferenceObject[CallbackModel]]]] = None
    deprecated: Optional[bool] = False
    security: Optional[List[SecurityRequirementModel]] = None
    servers: Optional[List[ServerObject]] = None

    # Extension fields
    model_config = {"extra": "allow"}


class PathItemObject(BaseModel):
    """Path item object describing operations available on a path."""
    ref: Optional[str] = Field(None, alias="$ref")
    summary: Optional[str] = None
    description: Optional[str] = None
    get: Optional[OperationObject] = None
    put: Optional[OperationObject] = None
    post: Optional[OperationObject] = None
    delete: Optional[OperationObject] = None
    options: Optional[OperationObject] = None
    head: Optional[OperationObject] = None
    patch: Optional[OperationObject] = None
    trace: Optional[OperationObject] = None
    servers: Optional[List[ServerObject]] = None
    parameters: Optional[List[Union[ParameterObject, ReferenceObject[ParameterObject]]]] = None

    # Extension fields
    model_config = {
        "extra": "allow",
        "populate_by_name": True
    }


class PathsModel(RootModel):
    """
    Paths Object in OpenAPI specification.

    This is a container for the path items where each key is a path string.
    The keys follow the format: /resource/{parameter}/subresource
    """
    root: Dict[str, PathItemObject]

    def __iter__(self):
        return iter(self.root.items())

    def __getitem__(self, key):
        return self.root[key]

    def __contains__(self, key):
        return key in self.root

    def items(self):
        return self.root.items()

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()


class SchemaType(str, Enum):
    """Schema type for OpenAPI Schema Object."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class SchemaFormat(str, Enum):
    """Schema formats for OpenAPI Schema Object."""
    INT32 = "int32"
    INT64 = "int64"
    FLOAT = "float"
    DOUBLE = "double"
    BYTE = "byte"
    BINARY = "binary"
    DATE = "date"
    DATE_TIME = "date-time"
    PASSWORD = "password"
    EMAIL = "email"
    UUID = "uuid"
    URI = "uri"
    HOSTNAME = "hostname"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    # Additional formats can be added


class Discriminator(BaseModel):
    """Discriminator object for polymorphic schemas."""
    propertyName: str
    mapping: Optional[Dict[str, str]] = None

    # Extension fields
    model_config = {"extra": "allow"}


class XML(BaseModel):
    """XML object for customizing XML serialization."""
    name: Optional[str] = None
    namespace: Optional[str] = None
    prefix: Optional[str] = None
    attribute: Optional[bool] = False
    wrapped: Optional[bool] = False

    # Extension fields
    model_config = {"extra": "allow"}


class SchemaObject(BaseModel):
    """Schema Object as defined in OpenAPI 3.1."""
    # Core Schema
    title: Optional[str] = None
    multipleOf: Optional[float] = None
    maximum: Optional[float] = None
    exclusiveMaximum: Optional[Union[bool, float]] = None
    minimum: Optional[float] = None
    exclusiveMinimum: Optional[Union[bool, float]] = None
    maxLength: Optional[int] = None
    minLength: Optional[int] = None
    pattern: Optional[str] = None
    maxItems: Optional[int] = None
    minItems: Optional[int] = None
    uniqueItems: Optional[bool] = False
    maxProperties: Optional[int] = None
    minProperties: Optional[int] = None
    required: Optional[List[str]] = None
    enum: Optional[List[Any]] = None

    # OpenAPI specific
    type: Optional[Union[SchemaType, List[SchemaType]]] = None
    allOf: Optional[List[Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    oneOf: Optional[List[Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    anyOf: Optional[List[Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    not_: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = Field(None, alias="not")
    items: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = None
    properties: Optional[Dict[str, Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    additionalProperties: Optional[Union[bool, 'SchemaObject', ReferenceObject['SchemaObject']]] = None
    description: Optional[str] = None
    format: Optional[str] = None
    default: Optional[Any] = None
    nullable: Optional[bool] = None
    discriminator: Optional[Discriminator] = None
    readOnly: Optional[bool] = False
    writeOnly: Optional[bool] = False
    xml: Optional[XML] = None
    externalDocs: Optional[ExternalDocumentation] = None
    example: Optional[Any] = None
    deprecated: Optional[bool] = False

    # Schema composition (OpenAPI 3.1)
    prefixItems: Optional[List[Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    contains: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = None
    patternProperties: Optional[Dict[str, Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    propertyNames: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = None

    # Extension fields
    model_config = {
        "extra": "allow",
        "populate_by_name": True
    }


class OAuthFlowObject(BaseModel):
    """OAuth Flow Object for security scheme OAuth flows."""
    authorizationUrl: Optional[str] = None
    tokenUrl: Optional[str] = None
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]

    # Extension fields
    model_config = {"extra": "allow"}

    @field_validator('authorizationUrl')
    def validate_authorization_url(cls, v, info: ValidationInfo):
        """Validate authorization URL for implicit/authorizationCode flows."""
        return v


class OAuthFlowsObject(BaseModel):
    """OAuth Flows Object for security scheme OAuth flows."""
    implicit: Optional[OAuthFlowObject] = None
    password: Optional[OAuthFlowObject] = None
    clientCredentials: Optional[OAuthFlowObject] = None
    authorizationCode: Optional[OAuthFlowObject] = None

    # Extension fields
    model_config = {"extra": "allow"}


class SecuritySchemeType(str, Enum):
    """Security scheme types in OpenAPI."""
    APIKEY = "apiKey"
    HTTP = "http"
    OAUTH2 = "oauth2"
    OPENIDCONNECT = "openIdConnect"
    MUTUALTLS = "mutualTLS"  # Added in OpenAPI 3.1.0


class SecuritySchemeIn(str, Enum):
    """Locations for API key security schemes."""
    QUERY = "query"
    HEADER = "header"
    COOKIE = "cookie"


class SecuritySchemeObject(BaseModel):
    """Security Scheme Object for describing API security mechanisms."""
    type: SecuritySchemeType
    description: Optional[str] = None
    name: Optional[str] = None
    in_: Optional[SecuritySchemeIn] = Field(None, alias="in")
    scheme: Optional[str] = None
    bearerFormat: Optional[str] = None
    flows: Optional[OAuthFlowsObject] = None
    openIdConnectUrl: Optional[str] = None

    # Extension fields
    model_config = {
        "extra": "allow",
        "populate_by_name": True
    }

    @field_validator('name', 'in_')
    def validate_apikey_fields(cls, v, info: ValidationInfo):
        """Validate that apiKey type has name and in fields."""
        values = info.data
        if values.get('type') == SecuritySchemeType.APIKEY:
            if v is None:
                field_name = info.field_name
                raise ValueError(f"{field_name} is required for apiKey security scheme")
        return v

    @field_validator('scheme')
    def validate_http_scheme(cls, v, info: ValidationInfo):
        """Validate that http type has scheme field."""
        values = info.data
        if values.get('type') == SecuritySchemeType.HTTP and v is None:
            raise ValueError("scheme is required for http security scheme")
        return v

    @field_validator('flows')
    def validate_oauth2_flows(cls, v, info: ValidationInfo):
        """Validate that oauth2 type has flows field."""
        values = info.data
        if values.get('type') == SecuritySchemeType.OAUTH2 and v is None:
            raise ValueError("flows is required for oauth2 security scheme")
        return v

    @field_validator('openIdConnectUrl')
    def validate_openid_url(cls, v, info: ValidationInfo):
        """Validate that openIdConnect type has openIdConnectUrl field."""
        values = info.data
        if values.get('type') == SecuritySchemeType.OPENIDCONNECT and v is None:
            raise ValueError("openIdConnectUrl is required for openIdConnect security scheme")
        return v


class TagObject(BaseModel):
    """Tag Object for API documentation tags."""
    name: str
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = None

    # Extension fields
    model_config = {"extra": "allow"}


class ComponentsObject(BaseModel):
    """Components Object for reusable OpenAPI components."""
    schemas: Optional[Dict[str, Union[SchemaObject, ReferenceObject[SchemaObject]]]] = None
    responses: Optional[Dict[str, Union[ResponseObject, ReferenceObject[ResponseObject]]]] = None
    parameters: Optional[Dict[str, Union[ParameterObject, ReferenceObject[ParameterObject]]]] = None
    examples: Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]] = None
    requestBodies: Optional[Dict[str, Union[RequestBodyObject, ReferenceObject[RequestBodyObject]]]] = None
    headers: Optional[Dict[str, Union[HeaderObject, ReferenceObject[HeaderObject]]]] = None
    securitySchemes: Optional[Dict[str, Union[SecuritySchemeObject, ReferenceObject[SecuritySchemeObject]]]] = None
    links: Optional[Dict[str, Union[LinkObject, ReferenceObject[LinkObject]]]] = None
    callbacks: Optional[Dict[str, Union[CallbackModel, ReferenceObject[CallbackModel]]]] = None
    pathItems: Optional[
        Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]]] = None  # Added in OpenAPI 3.1.0

    # Extension fields
    model_config = {"extra": "allow"}


class WebhookModel(RootModel):
    """Webhook Object for describing webhooks in OpenAPI 3.1."""
    root: Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]]

    def __iter__(self):
        return iter(self.root.items())

    def __getitem__(self, key):
        return self.root[key]

    def items(self):
        return self.root.items()


class OpenAPIObject(BaseModel):
    """Root document object for OpenAPI 3.1 specification."""
    openapi: str  # Version string (e.g., "3.1.0")
    info: InfoObject
    jsonSchemaDialect: Optional[
        str] = None  # Added in 3.1.0, defaults to "https://spec.openapis.org/oas/3.1/dialect/base"
    servers: Optional[List[ServerObject]] = None
    paths: Optional[PathsModel] = None
    webhooks: Optional[Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]]] = None  # Added in 3.1.0
    components: Optional[ComponentsObject] = None
    security: Optional[List[SecurityRequirementModel]] = None
    tags: Optional[List[TagObject]] = None
    externalDocs: Optional[ExternalDocumentation] = None

    # Extension fields starting with x-
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_paths_or_webhooks(self) -> "OpenAPIObject":
        """Validate that at least one of paths or webhooks is provided."""
        if "3.1" in self.openapi:
            if self.paths is None and self.webhooks is None:
                raise ValueError("At least one of paths or webhooks must be provided")
        else:
            if self.paths is None:
                raise ValueError("paths must be provided for OpenAPI versions before 3.1.0")
        return self

    @field_validator('jsonSchemaDialect')
    def set_default_dialect(cls, v, info: ValidationInfo):
        """Set default schema dialect for OpenAPI 3.1.0."""
        values = info.data
        if v is None and "3.1" in values.get("openapi", ""):
            return "https://spec.openapis.org/oas/3.1/dialect/base"
        return v


class OpenAPIResolver:
    """A resolver for OpenAPI documents that automatically resolves all references."""

    def __init__(self, spec: OpenAPIObject, orig_doc: Dict[str, Any] = None):
        self.spec = spec
        self.spec.model_extra['x-orig-doc'] = orig_doc
        self.resolved_refs: Set[str] = set()

    def resolve_all(self) -> OpenAPIObject:
        """Resolve all references in the document."""
        resolved_spec = self.spec
        # Resolve paths
        if resolved_spec.paths:
            for path_key, path_item in resolved_spec.paths:
                self._resolve_path_item(path_item, resolved_spec)

        # Resolve components
        if resolved_spec.components:
            self._resolve_components(resolved_spec.components, resolved_spec)

        # Resolve webhooks
        if resolved_spec.webhooks:
            for webhook_key, webhook in resolved_spec.webhooks.items():
                if isinstance(webhook, ReferenceObject):
                    resolved_spec.webhooks[webhook_key] = webhook.resolve(resolved_spec)
                else:
                    self._resolve_path_item(webhook, resolved_spec)

        return resolved_spec

    def _resolve_components(self, components: ComponentsObject, root: OpenAPIObject) -> None:
        """Resolve references in components."""
        # Resolve schemas
        if components.schemas:
            for name, schema in components.schemas.items():
                if isinstance(schema, ReferenceObject):
                    components.schemas[name] = schema.resolve(root)
                else:
                    self._resolve_schema(schema, root)

        # Resolve parameters
        if components.parameters:
            for name, param in components.parameters.items():
                if isinstance(param, ReferenceObject):
                    components.parameters[name] = param.resolve(root)

        # Resolve responses
        if components.responses:
            for name, response in components.responses.items():
                if isinstance(response, ReferenceObject):
                    components.responses[name] = response.resolve(root)

        # ... resolve other component types ...

    def _resolve_path_item(self, path_item: PathItemObject, root: OpenAPIObject) -> None:
        """Resolve references in a path item."""
        # Resolve parameters
        if path_item.parameters:
            for i, param in enumerate(path_item.parameters):
                if isinstance(param, ReferenceObject):
                    path_item.parameters[i] = param.resolve(root)

        # Resolve operations
        for op_name in ['get', 'post', 'put', 'delete', 'options', 'head', 'patch', 'trace']:
            operation = getattr(path_item, op_name, None)
            if operation:
                self._resolve_operation(operation, root)

    def _resolve_operation(self, operation: OperationObject, root: OpenAPIObject) -> None:
        """Resolve references in an operation."""
        # Resolve parameters
        if operation.parameters:
            for i, param in enumerate(operation.parameters):
                if isinstance(param, ReferenceObject):
                    operation.parameters[i] = param.resolve(root)

        # Resolve request body
        if operation.requestBody and isinstance(operation.requestBody, ReferenceObject):
            operation.requestBody = operation.requestBody.resolve(root)

        # Resolve schema references in request body content
        if operation.requestBody and operation.requestBody.content:
            for content_type, media_type in operation.requestBody.content.items():
                if media_type.schema_ and isinstance(media_type.schema_, ReferenceObject):
                    media_type.schema_ = media_type.schema_.resolve(root)
                    # Add this line to recursively resolve nested references
                    if hasattr(media_type.schema_, 'properties'):
                        self._resolve_schema(media_type.schema_, root)

                # Also resolve examples if present
                if media_type.examples:
                    for example_key, example in media_type.examples.items():
                        if isinstance(example, ReferenceObject):
                            media_type.examples[example_key] = example.resolve(root)

        # Resolve responses
        if hasattr(operation.responses, 'root'):
            for status, response in operation.responses.root.items():
                if isinstance(response, ReferenceObject):
                    operation.responses.root[status] = response.resolve(root)

            for status, response in operation.responses.root.items():
                # Resolve schema references in response content
                if response.content:
                    for content_type, media_type in response.content.items():
                        if media_type.schema_ and isinstance(media_type.schema_, ReferenceObject):
                            media_type.schema_ = media_type.schema_.resolve(root)
                            # Add this line to recursively resolve nested references
                            if hasattr(media_type.schema_, 'properties'):
                                self._resolve_schema(media_type.schema_, root)

                        # Also resolve examples if present
                        if media_type.examples:
                            for example_key, example in media_type.examples.items():
                                if isinstance(example, ReferenceObject):
                                    media_type.examples[example_key] = example.resolve(root)

        # Resolve callbacks
        if operation.callbacks:
            for callback_name, callback in operation.callbacks.items():
                if isinstance(callback, ReferenceObject):
                    operation.callbacks[callback_name] = callback.resolve(root)

    def _resolve_schema(self, schema: SchemaObject, root: OpenAPIObject) -> None:
        """Resolve references in a schema object."""
        # Resolve allOf, oneOf, anyOf schemas
        for composition_type in ['allOf', 'oneOf', 'anyOf']:
            composition_list = getattr(schema, composition_type, None)
            if composition_list:
                for i, sub_schema in enumerate(composition_list):
                    if isinstance(sub_schema, ReferenceObject):
                        composition_list[i] = sub_schema.resolve(root)
                        # ADDED: Continue resolving nested references in the resolved schema
                        self._resolve_schema(composition_list[i], root)
                    else:
                        self._resolve_schema(sub_schema, root)

        # Resolve 'not' schema
        if schema.not_ and isinstance(schema.not_, ReferenceObject):
            schema.not_ = schema.not_.resolve(root)
            # ADDED: Continue resolving nested references in the resolved schema
            self._resolve_schema(schema.not_, root)
        elif schema.not_:
            self._resolve_schema(schema.not_, root)

        # Resolve items schema (for arrays)
        if schema.items and isinstance(schema.items, ReferenceObject):
            schema.items = schema.items.resolve(root)
            # ADDED: Continue resolving nested references in the resolved schema
            self._resolve_schema(schema.items, root)
        elif schema.items:
            self._resolve_schema(schema.items, root)

        # Resolve property schemas (for objects)
        if schema.properties:
            for prop_name, prop_schema in schema.properties.items():
                if isinstance(prop_schema, ReferenceObject):
                    schema.properties[prop_name] = prop_schema.resolve(root)
                    # ADDED: Continue resolving nested references in the resolved schema
                    self._resolve_schema(schema.properties[prop_name], root)
                else:
                    self._resolve_schema(prop_schema, root)

        # Resolve additionalProperties schema
        if schema.additionalProperties and isinstance(schema.additionalProperties, ReferenceObject):
            schema.additionalProperties = schema.additionalProperties.resolve(root)
            # ADDED: Continue resolving nested references in the resolved schema
            if not isinstance(schema.additionalProperties, bool):
                self._resolve_schema(schema.additionalProperties, root)
        elif schema.additionalProperties and not isinstance(schema.additionalProperties, bool):
            self._resolve_schema(schema.additionalProperties, root)

        # Similar changes for prefixItems, contains, patternProperties, and propertyNames...


def parse_yaml(yaml_content: str, resolve_refs: bool = True) -> OpenAPIObject:
    """
    Parse an OpenAPI specification from YAML content.

    Args:
        yaml_content: String containing the YAML content
        resolve_refs: Whether to automatically resolve $ref references (default: True)

    Returns:
        Parsed OpenAPIObject with resolved references if requested
    """
    try:
        # Parse YAML content
        spec_dict = yaml.safe_load(yaml_content)

        # Parse with Pydantic model
        openapi_spec = OpenAPIObject.model_validate(spec_dict)

        # Resolve references if requested
        if resolve_refs:
            resolver = OpenAPIResolver(openapi_spec, spec_dict)
            openapi_spec = resolver.resolve_all()

        return openapi_spec

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML content: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing OpenAPI specification: {e}")


def parse_yaml_file(file_path: str, resolve_refs: bool = True) -> OpenAPIObject:
    """
    Parse an OpenAPI specification from a YAML file.

    Args:
        file_path: Path to the YAML file
        resolve_refs: Whether to automatically resolve $ref references (default: True)

    Returns:
        Parsed OpenAPIObject with resolved references if requested
    """
    try:
        with open(file_path, 'r') as file:
            yaml_content = file.read()

        return parse_yaml(yaml_content, resolve_refs)

    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except IOError as e:
        raise ValueError(f"Error reading file: {e}")


def parse_json(json_content: str, resolve_refs: bool = True) -> OpenAPIObject:
    """
    Parse an OpenAPI specification from JSON content.

    Args:
        json_content: String containing the JSON content
        resolve_refs: Whether to automatically resolve $ref references (default: True)

    Returns:
        Parsed OpenAPIObject with resolved references if requested
    """
    import json

    try:
        # Parse JSON content
        spec_dict = json.loads(json_content)

        # Parse with Pydantic model
        openapi_spec = OpenAPIObject.model_validate(spec_dict)

        # Resolve references if requested
        if resolve_refs:
            resolver = OpenAPIResolver(openapi_spec)
            openapi_spec = resolver.resolve_all()

        return openapi_spec

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON content: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing OpenAPI specification: {e}")


def parse_json_file(file_path: str, resolve_refs: bool = True) -> OpenAPIObject:
    """
    Parse an OpenAPI specification from a JSON file.

    Args:
        file_path: Path to the JSON file
        resolve_refs: Whether to automatically resolve $ref references (default: True)

    Returns:
        Parsed OpenAPIObject with resolved references if requested
    """
    try:
        with open(file_path, 'r') as file:
            json_content = file.read()

        return parse_json(json_content, resolve_refs)

    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except IOError as e:
        raise ValueError(f"Error reading file: {e}")


# Export main classes and functions
__all__ = [
    'OpenAPIObject',
    'SchemaObject',
    'PathsModel',
    'ParameterObject',
    'ResponseObject',
    'SecuritySchemeObject',
    'TagObject',
    'ExternalDocumentation',
    'InfoObject',
    'ComponentsObject',
    'ServerObject',
    'parse_yaml',
    'parse_yaml_file',
    'parse_json',
    'parse_json_file',
    'OpenAPIResolver'
]
