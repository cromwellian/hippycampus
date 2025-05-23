# -*- coding: utf-8 -*-
"""
Pydantic models for parsing OpenAPI 3.0 and 3.1 specifications.

This module provides a comprehensive set of Pydantic models that correspond to the
objects defined in the OpenAPI Specification (OAS). It supports parsing both
YAML and JSON formatted OpenAPI documents and includes a resolver for handling
internal `$ref` references.

The models are designed to be compliant with both OAS 3.0.x and 3.1.x where possible,
with some fields or behaviors adapting based on the specified OpenAPI version in the
document. Key features include:
- Validation of OpenAPI structure according to the specification.
- Easy parsing from YAML/JSON strings or files.
- Resolution of internal JSON pointers (`$ref`).
- Support for common OpenAPI extensions (fields starting with 'x-').

Typical usage involves parsing an OpenAPI document string or file into an
`OpenAPIObject` instance, which then provides access to the entire specification
tree as Pydantic models.
"""
from __future__ import annotations

import json # Added for OpenAPIResolver._resolve_reference_str_to_model (parsing potentially nested JSON in string fields)
from typing import Any, Dict, List, Optional, Union, Set, TypeVar, Generic, cast, Annotated
from enum import Enum
from pydantic import BaseModel, Field, ValidationInfo, model_validator, field_validator, RootModel, ValidationError
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
    'ContactObject', 
    'LicenseObject', 

    # Server Related
    'ServerObject',
    'ServerVariableObject', 

    # Utility Models / RootModels that wrap dicts
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
    'OpenAPIResolver', 
]

# Generic Type Variable for ReferenceObject
ReferencedType = TypeVar('ReferencedType', bound=BaseModel)

_current_resolver_instance: Optional['OpenAPIResolver'] = None

def get_current_resolver() -> Optional['OpenAPIResolver']:
    """
    Gets the current globally set OpenAPIResolver instance.

    Note:
        This global resolver is a fallback. Explicitly passing the resolver
        to methods that need it is the preferred approach.

    Returns:
        Optional[OpenAPIResolver]: The current global resolver instance, or None.
    """
    return _current_resolver_instance

def set_current_resolver(resolver: Optional['OpenAPIResolver']) -> None:
    """
    Sets the current global OpenAPIResolver instance.

    Note:
        This global resolver is a fallback. Explicitly passing the resolver
        to methods that need it is the preferred approach.

    Args:
        resolver: The OpenAPIResolver instance to set globally, or None to clear it.
    """
    global _current_resolver_instance
    _current_resolver_instance = resolver


class ReferenceObject(BaseModel, Generic[ReferencedType]):
    """
    Represents a JSON Reference (`$ref`) object as defined in the OpenAPI Specification.

    This object is used to create a reference to another component within the OpenAPI
    document or an external document. The `resolve` method can be used to obtain the
    actual referenced Pydantic model instance.

    Attributes:
        ref (str): The reference string (JSON Pointer).
        summary (Optional[str]): A short summary of the referenced component.
                                 Relevant for OpenAPI 3.1.0 when $ref is used alongside other keywords.
        description (Optional[str]): A description of the referenced component.
                                     Relevant for OpenAPI 3.1.0.
    """
    ref: str = Field(..., alias="$ref", serialization_alias='$ref') 
    summary: Optional[str] = None
    description: Optional[str] = None

    _resolved_obj: Optional[ReferencedType] = Field(None, exclude=True, repr=False)


    model_config = {
        "extra": "allow", 
        "populate_by_name": True,
        "arbitrary_types_allowed": True 
    }

    def resolve(self, resolver: 'OpenAPIResolver') -> ReferencedType:
        """
        Resolves this reference to its actual Pydantic model instance using the provided resolver.

        This method utilizes the given `OpenAPIResolver` to find the component pointed to by the
        `$ref` string, parse it into the appropriate Pydantic model, and return it.
        The resolved object is cached internally within this `ReferenceObject` instance
        for subsequent calls.

        Args:
            resolver: The `OpenAPIResolver` instance responsible for looking up and
                      parsing the referenced component from the specification document.

        Returns:
            The resolved Pydantic model instance, cast to the generic type `ReferencedType`.

        Raises:
            ValueError: If the reference string is invalid, the component cannot be found,
                        or the resolved component's type is incompatible with `ReferencedType`.
        """
        if self._resolved_obj is not None:
            return self._resolved_obj

        resolved_component = resolver._resolve_reference_str_to_model(self.ref)
        
        # Basic type check (can be enhanced if ReferencedType is not a TypeVar at runtime)
        # For now, relies on resolver to return correctly typed model based on ref path.
        # Consider using `get_args(self.__orig_class__)[0]` if more specific runtime check is needed.
        self._resolved_obj = cast(ReferencedType, resolved_component)
        return self._resolved_obj


class ExternalDocumentation(BaseModel):
    """
    Allows referencing an external resource for extended documentation.

    Attributes:
        url (str): REQUIRED. The URL for the target documentation.
        description (Optional[str]): A short description of the target documentation.
                                     CommonMark syntax MAY be used for rich text representation.
    """
    url: str
    description: Optional[str] = None
    model_config = {"extra": "allow"}

class ContactObject(BaseModel):
    """
    Contact information for the exposed API.

    Attributes:
        name (Optional[str]): The identifying name of the contact person/organization.
        url (Optional[str]): The URL pointing to the contact information. MUST be in the format of a URL.
        email (Optional[str]): The email address of the contact person/organization.
                               MUST be in the format of an email address.
    """
    name: Optional[str] = None
    url: Optional[str] = None
    email: Optional[str] = None
    model_config = {"extra": "allow"}

class LicenseObject(BaseModel):
    """
    License information for the exposed API.

    Attributes:
        name (str): REQUIRED. The license name used for the API.
        identifier (Optional[str]): An SPDX license expression for the API. The `identifier` field
                                    is mutually exclusive of the `url` field. (OpenAPI 3.1.0)
        url (Optional[str]): A URL to the license used for the API. MUST be in the format of a URL.
                             The `url` field is mutually exclusive of the `identifier` field.
    """
    name: str
    identifier: Optional[str] = None 
    url: Optional[str] = None
    model_config = {"extra": "allow"}


class InfoObject(BaseModel):
    """
    The object provides metadata about the API.
    The metadata MAY be used by the clients if needed, and MAY be presented in editing or documentation generation tools for convenience.

    Attributes:
        title (str): REQUIRED. The title of the API.
        summary (Optional[str]): A short summary of the API. (OpenAPI 3.1.0)
        description (Optional[str]): A description of the API. CommonMark syntax MAY be used for rich text representation.
        termsOfService (Optional[str]): A URL to the Terms of Service for the API. MUST be in the format of a URL.
        contact (Optional[ContactObject]): The contact information for the exposed API.
        license (Optional[LicenseObject]): The license information for the exposed API.
        version (str): REQUIRED. The version of the OpenAPI document (which is distinct from the OpenAPI Specification version or the API implementation version).
    """
    title: str
    summary: Optional[str] = None 
    description: Optional[str] = None
    termsOfService: Optional[str] = Field(None, alias="termsOfService")
    contact: Optional[ContactObject] = None
    license: Optional[LicenseObject] = None
    version: str
    model_config = {"extra": "allow"}

class ServerVariableObject(BaseModel):
    """
    An object representing a Server Variable for server URL template substitution.

    Attributes:
        enum (Optional[List[str]]): An enumeration of string values to be used if the substitution options are from a limited set.
                                    The array SHOULD NOT be empty.
        default (str): REQUIRED. The default value to use for substitution, which SHALL be sent if an alternate value is not supplied.
                       Note this behavior is different than the Schema Object's treatment of default values, because in those cases schema validation SHOULD NOT fail for unknown fixed fields.
        description (Optional[str]): An optional description for the server variable. CommonMark syntax MAY be used for rich text representation.
    """
    enum: Optional[List[str]] = None
    default: str
    description: Optional[str] = None
    model_config = {"extra": "allow"}

class ServerObject(BaseModel):
    """
    An object representing a Server.

    Attributes:
        url (str): REQUIRED. A URL to the target host. This URL supports Server Variables and MAY be relative,
                   to indicate that the host location is relative to the location where the OpenAPI document is being served.
                   Variable substitutions will be made when a variable is named in {brackets}.
        description (Optional[str]): An optional description for the server. CommonMark syntax MAY be used for rich text representation.
        variables (Optional[Dict[str, ServerVariableObject]]): A map between a variable name and its value. The value is used for substitution in the server's URL template.
    """
    url: str
    description: Optional[str] = None
    variables: Optional[Dict[str, ServerVariableObject]] = None
    model_config = {"extra": "allow"}

class ExampleObject(BaseModel):
    """
    Represents an example of a particular data type or schema.

    In all cases, the example value is expected to be compatible with the type schema
    of its associated value. Tooling implementations MAY choose to validate compatibility
    automatically, and reject the example value if it is incompatible.

    Attributes:
        summary (Optional[str]): Short description for the example.
        description (Optional[str]): Long description for the example. CommonMark syntax MAY be used.
        value (Optional[Any]): Embedded literal example. The `value` field and `externalValue` field are mutually exclusive.
        externalValue (Optional[str]): A URL that points to the literal example. This provides the capability to reference examples that cannot easily be included in JSON or YAML documents.
                                       The `value` field and `externalValue` field are mutually exclusive.
    """
    summary: Optional[str] = None
    description: Optional[str] = None
    value: Optional[Any] = None
    externalValue: Optional[str] = Field(None, alias="externalValue")
    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def check_value_or_external_value(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'value' in data and 'externalValue' in data:
                raise ValueError("Cannot specify both 'value' and 'externalValue' in ExampleObject")
        return data

class EncodingObject(BaseModel):
    """
    A single encoding definition applied to a single schema property.

    Attributes:
        contentType (Optional[str]): The Content-Type for encoding a specific property.
                                     Default value depends on the property type: for `object` - `application/json`;
                                     for `array` - the default is defined based on the `style` and `explode` keywords;
                                     for all other types - the Content-Type is `application/octet-stream`.
        headers (Optional[Dict[str, Union['HeaderObject', ReferenceObject['HeaderObject']]]]):
            A map allowing additional information to be provided as headers.
            Each header provides the header name and schema for the header value.
        style (Optional[str]): Describes how a specific property value will be serialized depending on its type.
                               See `Parameter Object` for details on the `style` keyword. The default value is `form`.
        explode (Optional[bool]): When this is true, property values of type `array` or `object` generate separate parameters
                                  for each value of the array or key-value pair of the map.
                                  For other types of properties this property has no effect.
                                  When `style` is `form`, the default value is `true`. For all other styles, the default value is `false`.
        allowReserved (Optional[bool]): Determines whether the parameter value SHOULD allow reserved characters,
                                        as defined by RFC3986 `:/?#[]@!$&'()*+,;=` to be included without percent-encoding.
                                        The default value is `false`.
    """
    contentType: Optional[str] = Field(None, alias="contentType")
    headers: Optional[Dict[str, Union['HeaderObject', ReferenceObject['HeaderObject']]]] = None
    style: Optional[str] = None 
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = Field(None, alias="allowReserved")
    model_config = {"extra": "allow"}

class MediaTypeObject(BaseModel):
    """
    Each Media Type Object provides schema and examples for the media type identified by its key.

    Attributes:
        schema_ (Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]]): The schema defining the type used for the request body.
        example (Optional[Any]): Example of the media type. The example object SHOULD be in the correct format as specified by the media type.
                                 The `example` field is mutually exclusive of the `examples` field.
        examples (Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]]):
            Examples of the media type. Each example object SHOULD correspond to the media type and schema definitions.
            The `examples` field is mutually exclusive of the `example` field.
        encoding (Optional[Dict[str, EncodingObject]]): A map between a property name and its encoding information.
                                                        The key, being the property name, MUST exist in the schema as a property.
    """
    schema_: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = Field(None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]] = None
    encoding: Optional[Dict[str, EncodingObject]] = None
    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def check_example_and_examples(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'example' in data and 'examples' in data:
                raise ValueError("Cannot specify both 'example' and 'examples' in MediaTypeObject")
        return data

class ParameterLocation(str, Enum):
    """Enumeration of possible locations for a parameter."""
    QUERY = "query"
    HEADER = "header"
    PATH = "path"
    COOKIE = "cookie"

class ParameterObject(BaseModel):
    """
    Describes a single operation parameter.
    A unique parameter is defined by a combination of a name and location.

    Attributes:
        name (str): REQUIRED. The name of the parameter. Parameter names are case sensitive.
        in_ (ParameterLocation): REQUIRED. The location of the parameter.
        description (Optional[str]): A brief description of the parameter. CommonMark syntax MAY be used.
        required (Optional[bool]): Determines whether this parameter is mandatory. If the parameter location is "path",
                                   this property is REQUIRED and its value MUST be true. Otherwise, the property MAY be included
                                   and its default value is false.
        deprecated (Optional[bool]): Specifies that a parameter is deprecated and SHOULD be transitioned out of usage. Default is false.
        allowEmptyValue (Optional[bool]): Sets the ability to pass empty-valued parameters. This is valid only for query parameters
                                         and allows sending a parameter with an empty value. Default value is false.
                                         If style is used, and if behavior is n/a (cannot be serialized), the value of allowEmptyValue SHALL be ignored.
        style (Optional[str]): Describes how the parameter value will be serialized depending on the type of the parameter value.
                               Default values (based on parameter location): for query - form; for path - simple; for header - simple; for cookie - form.
        explode (Optional[bool]): When this is true, parameter values of type array or object generate separate parameters for each value
                                  of the array or key-value pair of the map. For other types of parameters this property has no effect.
                                  When style is form, the default value is true. For all other styles, the default value is false.
        allowReserved (Optional[bool]): Determines whether the parameter value SHOULD allow reserved characters, as defined by RFC3986. Default value is false.
        schema_ (Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]]): The schema defining the type used for the parameter. Mutually exclusive with `content`.
        example (Optional[Any]): Example of the parameter's value.
        examples (Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]]): Examples of the parameter's value.
        content (Optional[Dict[str, MediaTypeObject]]): A map containing the media type and its schema for the parameter.
                                                        The map MUST only contain one entry. Mutually exclusive with `schema`.
    """
    name: str
    in_: ParameterLocation = Field(..., alias="in") 
    description: Optional[str] = None
    required: Optional[bool] = None 
    deprecated: Optional[bool] = False
    allowEmptyValue: Optional[bool] = Field(None, alias="allowEmptyValue") 

    style: Optional[str] = None 
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = Field(None, alias="allowReserved")
    schema_: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = Field(None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]] = None
    
    content: Optional[Dict[str, MediaTypeObject]] = None 
    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="after") 
    def check_path_param_required(self) -> 'ParameterObject':
        if self.in_ == ParameterLocation.PATH and self.required is not True:
            raise ValueError(f"Path parameter '{self.name}' must have 'required: true'")
        return self
    
    @model_validator(mode="before")
    @classmethod
    def check_schema_or_content(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'schema' in data and 'content' in data:
                raise ValueError("ParameterObject cannot have both 'schema' and 'content'")
            # Relaxing the single entry check for 'content' as per previous decision.
            # if 'content' in data and len(data['content']) > 1:
            # raise ValueError("ParameterObject 'content' map must only contain one entry")
        return data

class HeaderObject(BaseModel): 
    """
    Describes a single header parameter (part of ResponseObject or EncodingObject).
    Follows the structure of the Parameter Object with the following changes:
    1. `name` MUST NOT be specified, it is given in the corresponding headers map.
    2. `in_` MUST NOT be specified, it is implicitly in "header".
    3. All traits that are affected by the location MUST be applicable to a location of "header" (e.g. `style`).

    Attributes:
        description (Optional[str]): A brief description of the header. CommonMark syntax MAY be used.
        required (Optional[bool]): Determines whether this header is mandatory. Default value is false.
        deprecated (Optional[bool]): Specifies that a header is deprecated. Default is false.
        allowEmptyValue (Optional[bool]): Ignored. The value of this keyword is ignored when passed in a header.
        style (Optional[str]): Default value is "simple".
        explode (Optional[bool]): Default value is false.
        allowReserved (Optional[bool]): Default value is false.
        schema_ (Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]]): Schema for the header's value.
        example (Optional[Any]): Example of the header's value.
        examples (Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]]): Examples of header values.
        content (Optional[Dict[str, MediaTypeObject]]): Media type and schema for complex headers. Mutually exclusive with `schema`.
    """
    description: Optional[str] = None
    required: Optional[bool] = False
    deprecated: Optional[bool] = False
    allowEmptyValue: Optional[bool] = Field(None, alias="allowEmptyValue")
    
    style: Optional[str] = Field("simple") 
    explode: Optional[bool] = None 
    allowReserved: Optional[bool] = Field(None, alias="allowReserved")
    schema_: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = Field(None, alias="schema")
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]] = None
    
    content: Optional[Dict[str, MediaTypeObject]] = None
    model_config = {"extra": "allow", "populate_by_name": True}
    
    @model_validator(mode="before")
    @classmethod
    def check_schema_or_content(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'schema' in data and 'content' in data:
                raise ValueError("HeaderObject cannot have both 'schema' and 'content'")
        return data


class RequestBodyObject(BaseModel):
    """
    Describes a single request body.

    Attributes:
        description (Optional[str]): A brief description of the request body. CommonMark syntax MAY be used.
        content (Dict[str, MediaTypeObject]): REQUIRED. The content of the request body.
                                              The key is a media type or media type range and the value describes it.
                                              For requests that match multiple keys, only the most specific key is applicable. e.g. text/plain overrides text/*
        required (Optional[bool]): Determines if the request body is required in the request. Defaults to false.
    """
    description: Optional[str] = None
    content: Dict[str, MediaTypeObject] 
    required: Optional[bool] = False
    model_config = {"extra": "allow"}

class LinkObject(BaseModel):
    """
    The Link object represents a possible design-time link for a response.
    The presence of a link does not guarantee the caller's ability to successfully invoke it, rather it provides a known relationship and traversal mechanism between responses and other operations.

    Attributes:
        operationRef (Optional[str]): A relative or absolute URI reference to an OAS operation.
                                      This field is mutually exclusive of the `operationId` field, and MUST point to an Operation Object.
        operationId (Optional[str]): The name of an existing, resolvable OAS operation, as defined by a unique `operationId`.
                                     This field is mutually exclusive of the `operationRef` field.
        parameters (Optional[Dict[str, Any]]): A map representing parameters to pass to an operation as specified with `operationId` or `operationRef`.
                                               The key is the parameter name to be used, whereas the value can be a constant or an expression to be evaluated at runtime.
        requestBody (Optional[Any]): A literal value or {expression} to use as a request body when calling the target operation.
        description (Optional[str]): A description of the link. CommonMark syntax MAY be used for rich text representation.
        server (Optional[ServerObject]): A server object to be used by the target operation.
    """
    operationRef: Optional[str] = Field(None, alias="operationRef") 
    operationId: Optional[str] = Field(None, alias="operationId")
    parameters: Optional[Dict[str, Any]] = None 
    requestBody: Optional[Any] = Field(None, alias="requestBody") 
    description: Optional[str] = None
    server: Optional[ServerObject] = None
    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def check_operation_ref_id_exclusive(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'operationRef' in data and 'operationId' in data:
                raise ValueError("LinkObject: Cannot specify both 'operationId' and 'operationRef'")
        return data

class ResponseObject(BaseModel):
    """
    Describes a single response from an API Operation, including design-time, static links to operations based on the response.

    Attributes:
        description (str): REQUIRED. A description of the response. CommonMark syntax MAY be used for rich text representation.
        headers (Optional[Dict[str, Union[HeaderObject, ReferenceObject[HeaderObject]]]]):
            Maps a header name to its definition. RFC7230 states header names are case insensitive.
            If a response header is defined with the name "Content-Type", it SHALL be ignored.
        content (Optional[Dict[str, MediaTypeObject]]): A map containing descriptions of potential response payloads.
                                                       The key is a media type or media type range and the value describes it.
        links (Optional[Dict[str, Union[LinkObject, ReferenceObject[LinkObject]]]]):
            A map of operations links that can be followed from the response. The key of the map is a short name for the link,
            following the naming constraints of the names for Component Objects.
    """
    description: str
    headers: Optional[Dict[str, Union[HeaderObject, ReferenceObject[HeaderObject]]]] = None
    content: Optional[Dict[str, MediaTypeObject]] = None 
    links: Optional[Dict[str, Union[LinkObject, ReferenceObject[LinkObject]]]] = None
    model_config = {"extra": "allow"}

class ResponsesModel(RootModel[Dict[str, Union[ResponseObject, ReferenceObject[ResponseObject]]]]):
    """
    A container for the expected responses of an operation.
    The container maps a HTTP response code to the expected response.
    The documentation is not necessarily exhaustive. Only basic responses that are defined
    explicitly SHOULD be declared in this map.
    """
    root: Dict[str, Union[ResponseObject, ReferenceObject[ResponseObject]]]

    def __iter__(self): # type: ignore
        return iter(self.root)

    def __getitem__(self, item: str) -> Union[ResponseObject, ReferenceObject[ResponseObject]]: # type: ignore
        return self.root[item]
    
    def items(self): # type: ignore
        return self.root.items()


class CallbackModel(RootModel[Dict[str, Union['PathItemObject', ReferenceObject['PathItemObject']]]]):
    """
    A map of possible out-of-band callbacks related to the parent operation.
    Each value in the map is a Path Item Object that describes a set of requests that may be initiated by the API provider
    and the expected responses. The key value used to identify the callback object is an expression, evaluated at runtime,
    that identifies a URL to send requests to (a callback URL).
    """
    root: Dict[str, Union['PathItemObject', ReferenceObject['PathItemObject']]]
    
    def __iter__(self): # type: ignore
        return iter(self.root)

    def __getitem__(self, item: str) -> Union['PathItemObject', ReferenceObject['PathItemObject']]: # type: ignore
        return self.root[item]

    def items(self): # type: ignore
        return self.root.items()


class SecurityRequirementModel(RootModel[Dict[str, List[str]]]):
    """
    Lists the security schemes to execute an operation.
    The name used for each property MUST correspond to a security scheme declared
    in the Security Schemes under the Components Object.

    If the security scheme is of type "oauth2" or "openIdConnect", then the value is a list of scope names required for the execution.
    For other security scheme types, the array MUST be empty.
    """
    root: Dict[str, List[str]] 

    def __iter__(self): # type: ignore
        return iter(self.root)

    def __getitem__(self, item: str) -> List[str]: # type: ignore
        return self.root[item]
    
    def items(self): # type: ignore
        return self.root.items()


class OperationObject(BaseModel):
    """
    Describes a single API operation on a path.

    Attributes:
        tags (Optional[List[str]]): A list of tags for API documentation control. Tags can be used for logical grouping of operations by resources or any other qualifier.
        summary (Optional[str]): A short summary of what the operation does.
        description (Optional[str]): A verbose explanation of the operation behavior. CommonMark syntax MAY be used.
        externalDocs (Optional[ExternalDocumentation]): Additional external documentation for this operation.
        operationId (Optional[str]): Unique string used to identify the operation. The id MUST be unique among all operations described in the API.
                                     The operationId value is case-sensitive. Tools and libraries MAY use the operationId to uniquely identify an operation,
                                     therefore, it is RECOMMENDED to follow common programming naming conventions.
        parameters (Optional[List[Union[ParameterObject, ReferenceObject[ParameterObject]]]]): A list of parameters that are applicable for this operation.
        requestBody (Optional[Union[RequestBodyObject, ReferenceObject[RequestBodyObject]]]): The request body applicable for this operation.
        responses (Optional[ResponsesModel]): REQUIRED (in OAS, but optional here for flexibility before validation). The list of possible responses as they are returned from executing this operation.
        callbacks (Optional[Dict[str, Union[CallbackModel, ReferenceObject[CallbackModel]]]]): A map of possible out-of-band callbacks related to the parent operation.
        deprecated (Optional[bool]): Declares this operation to be deprecated. Consumers SHOULD refrain from usage of the declared operation. Default value is false.
        security (Optional[List[SecurityRequirementModel]]): A declaration of which security mechanisms can be used for this operation.
        servers (Optional[List[ServerObject]]): An alternative server array to service this operation. If an alternative server object is specified at the Path Item Object or Root level, it will be overridden by this value.
    """
    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = Field(None, alias="externalDocs")
    operationId: Optional[str] = Field(None, alias="operationId")
    parameters: Optional[List[Union[ParameterObject, ReferenceObject[ParameterObject]]]] = None
    requestBody: Optional[Union[RequestBodyObject, ReferenceObject[RequestBodyObject]]] = Field(None, alias="requestBody")
    responses: Optional[ResponsesModel] = Field(default_factory=lambda: ResponsesModel(root={})) # Default to empty responses
    callbacks: Optional[Dict[str, Union[CallbackModel, ReferenceObject[CallbackModel]]]] = None
    deprecated: Optional[bool] = False
    security: Optional[List[SecurityRequirementModel]] = None 
    servers: Optional[List[ServerObject]] = None
    model_config = {"extra": "allow", "populate_by_name": True}

class PathItemObject(BaseModel):
    """
    Describes the operations available on a single path.
    A Path Item MAY be empty, due to ACL constraints. The path itself is still exposed to the documentation viewer
    but they will not know which operations and parameters are available.

    Attributes:
        ref (Optional[str]): Allows for an external definition of this path item.
        summary (Optional[str]): An optional, string summary, intended to apply to all operations in this path.
        description (Optional[str]): An optional, string description, intended to apply to all operations in this path. CommonMark syntax MAY be used.
        get (Optional[OperationObject]): A definition of a GET operation on this path.
        put (Optional[OperationObject]): A definition of a PUT operation on this path.
        post (Optional[OperationObject]): A definition of a POST operation on this path.
        delete (Optional[OperationObject]): A definition of a DELETE operation on this path.
        options (Optional[OperationObject]): A definition of an OPTIONS operation on this path.
        head (Optional[OperationObject]): A definition of a HEAD operation on this path.
        patch (Optional[OperationObject]): A definition of a PATCH operation on this path.
        trace (Optional[OperationObject]): A definition of a TRACE operation on this path.
        servers (Optional[List[ServerObject]]): An alternative server array to service all operations in this path.
        parameters (Optional[List[Union[ParameterObject, ReferenceObject[ParameterObject]]]]): A list of parameters that are applicable for all the operations described under this path.
                                                                                           These parameters can be overridden at the operation level, but cannot be removed there.
    """
    ref: Optional[str] = Field(None, alias="$ref", serialization_alias='$ref') 
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
    model_config = {"extra": "allow", "populate_by_name": True}

class PathsModel(RootModel[Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]]]):
    """
    Holds the relative paths to the individual endpoints and their operations.
    The path is appended to the URL from the Server Object in order to construct the full URL.
    The Paths MAY be empty, due to ACL constraints.
    """
    root: Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]]

    def __iter__(self): # type: ignore
        return iter(self.root)

    def __getitem__(self, item: str) -> Union[PathItemObject, ReferenceObject[PathItemObject]]: # type: ignore
        return self.root[item]
    
    def items(self): # type: ignore
        return self.root.items()


class SchemaType(str, Enum):
    """
    Enumeration of valid data types for the `type` field in a Schema Object,
    as defined by the OpenAPI Specification and JSON Schema.
    """
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null" 

class Discriminator(BaseModel):
    """
    When request bodies or response payloads may be one of a number of different schemas,
    a `discriminator` object can be used to aid in serialization, deserialization, and validation.
    The discriminator is a specific object in a schema which is used to inform the consumer of the specification
    of an alternative schema based on the value associated with it.

    Attributes:
        propertyName (str): REQUIRED. The name of the property in the payload that will hold the discriminator value.
        mapping (Optional[Dict[str, str]]): An object to hold mappings between payload values and schema names or references.
                                           Keys are the discriminator values, values are the schema names or $ref strings.
    """
    propertyName: str = Field(..., alias="propertyName")
    mapping: Optional[Dict[str, str]] = None
    model_config = {"extra": "allow", "populate_by_name": True}

class XML(BaseModel):
    """
    A metadata object that allows for more fine-tuned XML model definitions.
    When using arrays, XML element names are not inferred (for singular/plural forms) and the `name` property SHOULD be used to add that information.
    See examples for expected behavior.

    Attributes:
        name (Optional[str]): Replaces the name of the element/attribute used for the described schema property.
                              When defined within `items`, it will affect the name of the individual XML elements within the list.
                              When defined alongside `type` being `array` (outside the `items`), it will affect the wrapping XML element name.
        namespace (Optional[str]): The URI of the namespace definition. Value MUST be in the form of an absolute URI.
        prefix (Optional[str]): The prefix to be used for the `name`.
        attribute (Optional[bool]): Declares whether the property definition translates to an attribute instead of an element. Default value is `false`.
        wrapped (Optional[bool]): MAY be used only for an array definition. Signifies whether the array is wrapped (for example, `<books><book/><book/></books>`)
                                or unwrapped (`<book/><book/>`). Default value is `false`.
                                The definition of `wrapped` true HAS NO EFFECT if `name` is ` schéma`.
    """
    name: Optional[str] = None
    namespace: Optional[str] = None
    prefix: Optional[str] = None
    attribute: Optional[bool] = False
    wrapped: Optional[bool] = False
    model_config = {"extra": "allow"}

class SchemaObject(BaseModel):
    """
    The Schema Object allows the definition of input and output data types.
    These types can be objects, but also primitives and arrays. This object is a superset of the JSON Schema Specification Draft 2020-12.
    For more information about the properties, see JSON Schema Core and JSON Schema Validation.
    Unless stated otherwise, the property definitions follow the JSON Schema.

    OpenAPI specific keywords: `nullable`, `discriminator`, `readOnly`, `writeOnly`, `xml`, `externalDocs`, `example`, `deprecated`.
    """
    # JSON Schema Core and Validation (subset chosen for common use)
    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    examples: Optional[List[Any]] = None # JSON Schema `examples` (list of any)
    
    multipleOf: Optional[float] = Field(None, gt=0)
    maximum: Optional[float] = None
    exclusiveMaximum: Optional[Union[float, bool]] = None 
    minimum: Optional[float] = None
    exclusiveMinimum: Optional[Union[float, bool]] = None 
    
    maxLength: Optional[int] = Field(None, ge=0)
    minLength: Optional[int] = Field(None, ge=0, default=0)
    pattern: Optional[str] = None 
    
    maxItems: Optional[int] = Field(None, ge=0)
    minItems: Optional[int] = Field(None, ge=0, default=0)
    uniqueItems: Optional[bool] = False
    
    maxProperties: Optional[int] = Field(None, ge=0)
    minProperties: Optional[int] = Field(None, ge=0, default=0)
    
    required: Optional[List[str]] = None 
    enum: Optional[List[Any]] = None 

    type: Optional[Union[SchemaType, List[SchemaType]]] = None 
    
    allOf: Optional[List[Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    oneOf: Optional[List[Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    anyOf: Optional[List[Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    not_: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = Field(None, alias="not")
    
    properties: Optional[Dict[str, Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None
    additionalProperties: Optional[Union[bool, 'SchemaObject', ReferenceObject['SchemaObject']]] = Field(True) 
    propertyNames: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = None 

    items: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = None 
    prefixItems: Optional[List[Union['SchemaObject', ReferenceObject['SchemaObject']]]] = None 
    contains: Optional[Union['SchemaObject', ReferenceObject['SchemaObject']]] = None 

    # OpenAPI Specific Vocabulary
    nullable: Optional[bool] = Field(None) 
    discriminator: Optional[Discriminator] = None
    readOnly: Optional[bool] = False
    writeOnly: Optional[bool] = False
    xml: Optional[XML] = None
    externalDocs: Optional[ExternalDocumentation] = Field(None, alias="externalDocs")
    example: Optional[Any] = None # OpenAPI singular example
    deprecated: Optional[bool] = False
    
    model_config = {"extra": "allow", "populate_by_name": True}


class OAuthFlowObject(BaseModel):
    """
    Configuration details for a supported OAuth Flow.

    Attributes:
        authorizationUrl (Optional[str]): REQUIRED for `implicit` and `authorizationCode` flows.
                                          The authorization URL to be used for this flow.
        tokenUrl (Optional[str]): REQUIRED for `password`, `clientCredentials` and `authorizationCode` flows.
                                  The token URL to be used for this flow.
        refreshUrl (Optional[str]): The URL to be used for obtaining refresh tokens.
        scopes (Dict[str, str]): REQUIRED. The available scopes for the OAuth2 security scheme.
                                 A map between the scope name and a short description for it.
                                 The map MAY be empty.
    """
    authorizationUrl: Optional[str] = Field(None, alias="authorizationUrl") 
    tokenUrl: Optional[str] = Field(None, alias="tokenUrl") 
    refreshUrl: Optional[str] = Field(None, alias="refreshUrl")
    scopes: Dict[str, str] 
    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="after")
    def check_urls_for_flow_type(self, info: ValidationInfo) -> 'OAuthFlowObject':
        # This validation is context-dependent (based on which flow type this object defines).
        # It's better handled in OAuthFlowsObject or by the consumer.
        # For example, if this instance is `OAuthFlowsObject.implicit`, then authorizationUrl is required.
        # `info.context` could potentially be used if the parent model passed it down.
        return self


class OAuthFlowsObject(BaseModel):
    """
    Allows configuration of the supported OAuth Flows.

    Attributes:
        implicit (Optional[OAuthFlowObject]): Configuration for the OAuth Implicit flow.
        password (Optional[OAuthFlowObject]): Configuration for the OAuth Resource Owner Password flow.
        clientCredentials (Optional[OAuthFlowObject]): Configuration for the OAuth Client Credentials flow.
        authorizationCode (Optional[OAuthFlowObject]): Configuration for the OAuth Authorization Code flow.
    """
    implicit: Optional[OAuthFlowObject] = None
    password: Optional[OAuthFlowObject] = None
    clientCredentials: Optional[OAuthFlowObject] = Field(None, alias="clientCredentials")
    authorizationCode: Optional[OAuthFlowObject] = Field(None, alias="authorizationCode")
    model_config = {"extra": "allow", "populate_by_name": True}


class SecuritySchemeType(str, Enum):
    """Enumeration of supported security scheme types."""
    APIKEY = "apiKey"
    HTTP = "http"
    MUTUALTLS = "mutualTLS" 
    OAUTH2 = "oauth2"
    OPENIDCONNECT = "openIdConnect"

class SecuritySchemeIn(str, Enum):
    """Enumeration of locations for an API key."""
    QUERY = "query"
    HEADER = "header"
    COOKIE = "cookie"

class SecuritySchemeObject(BaseModel):
    """
    Defines a security scheme that can be used by the operations.
    Supported schemes are HTTP authentication, an API key (either in a header, a cookie parameter or as a query parameter),
    OAuth2's common flows (implicit, password, client credentials and authorization code) as defined in RFC6749,
    and OpenID Connect Discovery. mutualTLS RFC8705 can also be defined.

    Attributes:
        type (SecuritySchemeType): REQUIRED. The type of the security scheme.
        description (Optional[str]): A description for the security scheme. CommonMark syntax MAY be used.
        name (Optional[str]): REQUIRED for `apiKey`. The name of the header, query or cookie parameter to be used.
        in_ (Optional[SecuritySchemeIn]): REQUIRED for `apiKey`. The location of the API key.
        scheme (Optional[str]): REQUIRED for `http`. The name of the HTTP Authorization scheme to be used in the Authorization header as defined in RFC7235.
                                The values used SHOULD be registered in the IANA Authentication Scheme registry.
        bearerFormat (Optional[str]): A hint to the client to identify how the bearer token is formatted.
                                      Bearer tokens are usually generated by an authorization server, so this information is primarily for documentation purposes.
        flows (Optional[OAuthFlowsObject]): REQUIRED for `oauth2`. An object containing configuration information for the supported OAuth Flows.
        openIdConnectUrl (Optional[str]): REQUIRED for `openIdConnect`. OpenID Connect URL to discover OAuth2 configuration values. This MUST be in the form of a URL.
    """
    type: SecuritySchemeType
    description: Optional[str] = None
    
    name: Optional[str] = None 
    in_: Optional[SecuritySchemeIn] = Field(None, alias="in") 
    
    scheme: Optional[str] = None 
    bearerFormat: Optional[str] = Field(None, alias="bearerFormat") 
    
    flows: Optional[OAuthFlowsObject] = None 
    
    openIdConnectUrl: Optional[str] = Field(None, alias="openIdConnectUrl") 
    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="after")
    def check_required_fields_by_type(self) -> 'SecuritySchemeObject':
        if self.type == SecuritySchemeType.APIKEY:
            if not self.name: raise ValueError("'name' is required for apiKey security scheme")
            if not self.in_: raise ValueError("'in' is required for apiKey security scheme")
        elif self.type == SecuritySchemeType.HTTP:
            if not self.scheme: raise ValueError("'scheme' is required for http security scheme")
        elif self.type == SecuritySchemeType.OAUTH2:
            if not self.flows: raise ValueError("'flows' is required for oauth2 security scheme")
        elif self.type == SecuritySchemeType.OPENIDCONNECT:
            if not self.openIdConnectUrl: raise ValueError("'openIdConnectUrl' is required for openIdConnect security scheme")
        return self

class TagObject(BaseModel):
    """
    Adds metadata to a single tag that is used by the Operation Object.
    It is not mandatory to have a Tag Object per tag defined in the Operation Object instances.

    Attributes:
        name (str): REQUIRED. The name of the tag.
        description (Optional[str]): A description for the tag. CommonMark syntax MAY be used.
        externalDocs (Optional[ExternalDocumentation]): Additional external documentation for this tag.
    """
    name: str
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = Field(None, alias="externalDocs")
    model_config = {"extra": "allow", "populate_by_name": True}


class ComponentsObject(BaseModel):
    """
    Holds a set of reusable objects for different aspects of the OAS.
    All objects defined within the components object will have no effect on the API unless they are explicitly referenced from properties outside the components object.

    Attributes:
        schemas (Optional[Dict[str, Union[SchemaObject, ReferenceObject[SchemaObject]]]]): An object to hold reusable Schema Objects.
        responses (Optional[Dict[str, Union[ResponseObject, ReferenceObject[ResponseObject]]]]): An object to hold reusable Response Objects.
        parameters (Optional[Dict[str, Union[ParameterObject, ReferenceObject[ParameterObject]]]]): An object to hold reusable Parameter Objects.
        examples (Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]]): An object to hold reusable Example Objects.
        requestBodies (Optional[Dict[str, Union[RequestBodyObject, ReferenceObject[RequestBodyObject]]]]): An object to hold reusable Request Body Objects.
        headers (Optional[Dict[str, Union[HeaderObject, ReferenceObject[HeaderObject]]]]): An object to hold reusable Header Objects.
        securitySchemes (Optional[Dict[str, Union[SecuritySchemeObject, ReferenceObject[SecuritySchemeObject]]]]): An object to hold reusable Security Scheme Objects.
        links (Optional[Dict[str, Union[LinkObject, ReferenceObject[LinkObject]]]]): An object to hold reusable Link Objects.
        callbacks (Optional[Dict[str, Union[CallbackModel, ReferenceObject[CallbackModel]]]]): An object to hold reusable Callback Objects.
        pathItems (Optional[Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]]]): An object to hold reusable Path Item Objects. (OpenAPI 3.1.0)
    """
    schemas: Optional[Dict[str, Union[SchemaObject, ReferenceObject[SchemaObject]]]] = None
    responses: Optional[Dict[str, Union[ResponseObject, ReferenceObject[ResponseObject]]]] = None
    parameters: Optional[Dict[str, Union[ParameterObject, ReferenceObject[ParameterObject]]]] = None
    examples: Optional[Dict[str, Union[ExampleObject, ReferenceObject[ExampleObject]]]] = None
    requestBodies: Optional[Dict[str, Union[RequestBodyObject, ReferenceObject[RequestBodyObject]]]] = Field(None, alias="requestBodies")
    headers: Optional[Dict[str, Union[HeaderObject, ReferenceObject[HeaderObject]]]] = None
    securitySchemes: Optional[Dict[str, Union[SecuritySchemeObject, ReferenceObject[SecuritySchemeObject]]]] = Field(None, alias="securitySchemes")
    links: Optional[Dict[str, Union[LinkObject, ReferenceObject[LinkObject]]]] = None
    callbacks: Optional[Dict[str, Union[CallbackModel, ReferenceObject[CallbackModel]]]] = None
    pathItems: Optional[Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]]] = Field(None, alias="pathItems") 
    model_config = {"extra": "allow", "populate_by_name": True}


class OpenAPIObject(BaseModel):
    """
    This is the root object of the OpenAPI document.
    It contains the version of the OpenAPI specification, information about the API,
    server connectivity, paths, components, security requirements, tags, and external documentation.
    """
    openapi: str 
    info: InfoObject
    jsonSchemaDialect: Optional[str] = Field(None, alias="jsonSchemaDialect") 
    servers: Optional[List[ServerObject]] = None
    paths: Optional[PathsModel] = None 
    webhooks: Optional[Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]]] = None 
    components: Optional[ComponentsObject] = None
    security: Optional[List[SecurityRequirementModel]] = None 
    tags: Optional[List[TagObject]] = None
    externalDocs: Optional[ExternalDocumentation] = Field(None, alias="externalDocs")
    model_config = {"extra": "allow", "populate_by_name": True}

    @model_validator(mode="after")
    def check_paths_or_webhooks_for_3_1(self) -> 'OpenAPIObject':
        if self.openapi.startswith("3.1"):
            if self.paths is None and self.webhooks is None:
                raise ValueError("For OpenAPI 3.1.x, at least one of 'paths' or 'webhooks' MUST be present.")
        elif self.paths is None: 
            raise ValueError("'paths' is required for OpenAPI versions prior to 3.1.0")
        return self

    @field_validator('jsonSchemaDialect', mode='before')
    @classmethod
    def default_json_schema_dialect(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None and info.data.get('openapi', '').startswith("3.1"):
            return "https://spec.openapis.org/oas/v3.1.0/dialect/base"
        return v


# --- OpenAPI Resolver ---
class OpenAPIResolver:
    """
    Resolves all `$ref` references within a parsed OpenAPI document.

    This class navigates an `OpenAPIObject` (which has already been parsed from raw data
    into Pydantic models, including `ReferenceObject` instances) and replaces these
    `ReferenceObject` instances with their fully resolved Pydantic model counterparts.
    It uses the original raw dictionary of the specification for lookups and caches
    resolved objects to handle circular references and improve performance.

    Attributes:
        spec_obj (OpenAPIObject): The root Pydantic model of the parsed OpenAPI specification.
                                  This object will be mutated by the resolution process.
        _raw_spec_dict (Dict[str, Any]): The original Python dictionary from which `spec_obj` was created.
                                         Used to look up the content of `$ref` pointers.
        _resolved_references (Dict[str, BaseModel]): A cache for already resolved reference strings
                                                     and their corresponding Pydantic model instances.
        _currently_resolving (Set[str]): A set of reference strings currently in the process of resolution,
                                         used for detecting circular dependencies.
    """
    def __init__(self, spec_obj: OpenAPIObject, raw_spec_dict: Dict[str, Any]):
        self.spec_obj: OpenAPIObject = spec_obj
        self._raw_spec_dict: Dict[str, Any] = raw_spec_dict 
        self._resolved_references: Dict[str, BaseModel] = {} 
        self._currently_resolving: Set[str] = set() 

        # Fallback global resolver instance. Explicit passing is preferred.
        set_current_resolver(self)


    def _get_target_dict_from_ref(self, ref_str: str) -> Any:
        """
        Retrieves the raw dictionary or value from the original specification
        based on a JSON Pointer reference string.

        Args:
            ref_str: The JSON Pointer string (e.g., `#/components/schemas/MySchema`).

        Returns:
            The raw Python data (dict, list, primitive) found at the reference path.

        Raises:
            ValueError: If the reference string format is invalid, or if the path
                        does not exist within the specification document.
        """
        if not ref_str.startswith('#/'):
            raise ValueError(f"Resolver supports only internal JSON Pointer references (e.g., '#/components/schemas/MySchema'). Got: {ref_str}")
        
        path_parts = ref_str[2:].split('/')
        current_val = self._raw_spec_dict
        for part in path_parts:
            unescaped_part = part.replace('~1', '/').replace('~0', '~')
            if isinstance(current_val, dict):
                if unescaped_part not in current_val:
                    raise ValueError(f"Invalid reference path: '{ref_str}'. Part '{unescaped_part}' not found in dict.")
                current_val = current_val[unescaped_part]
            elif isinstance(current_val, list):
                try:
                    idx = int(unescaped_part)
                    if not (0 <= idx < len(current_val)):
                         raise ValueError(f"Invalid reference path: '{ref_str}'. Index '{idx}' out of bounds for list.")
                    current_val = current_val[idx]
                except ValueError: 
                    raise ValueError(f"Invalid reference path: '{ref_str}'. Part '{unescaped_part}' is not a valid list index.")
            else: 
                raise ValueError(f"Invalid reference path: '{ref_str}'. Cannot traverse part '{unescaped_part}' in non-dict/list element.")
        return current_val

    _REF_TYPE_MAP: Dict[str, Type[BaseModel]] = {
        "schemas": SchemaObject,
        "responses": ResponseObject,
        "parameters": ParameterObject,
        "examples": ExampleObject,
        "requestBodies": RequestBodyObject,
        "headers": HeaderObject,
        "securitySchemes": SecuritySchemeObject,
        "links": LinkObject,
        "callbacks": CallbackModel, 
        "pathItems": PathItemObject, 
    }

    def _get_model_type_for_ref(self, ref_str: str) -> Type[BaseModel]:
        """
        Determines the expected Pydantic model type for a given JSON Pointer reference string.

        This method inspects the reference path (e.g., `#/components/schemas/...`) to infer
        the type of component being referenced (e.g., `SchemaObject`, `ParameterObject`).

        Args:
            ref_str: The JSON Pointer string.

        Returns:
            The Pydantic model class corresponding to the component type.

        Raises:
            ValueError: If the reference string format is unrecognized or points to an
                        unknown component category.
        """
        path_parts = ref_str[2:].split('/')
        if len(path_parts) < 2: 
            raise ValueError(f"Reference '{ref_str}' does not appear to be a standard component reference.")

        component_category = path_parts[1] 
        
        model_type = self._REF_TYPE_MAP.get(component_category)
        if model_type:
            return model_type
        
        if path_parts[0] == "paths" and len(path_parts) > 1: 
            return PathItemObject

        raise ValueError(f"Unknown component category '{component_category}' in reference '{ref_str}'. Cannot determine model type.")


    def _resolve_reference_str_to_model(self, ref_str: str) -> BaseModel:
        """
        Resolves a reference string to its corresponding Pydantic model instance.

        This is the core resolution logic. It handles fetching the raw data for the reference,
        determining the target Pydantic model type, recursively resolving any nested references
        within the target data, and then validating/parsing that data into a model instance.
        It uses caching to avoid re-resolving the same reference and detects circular dependencies.

        Args:
            ref_str: The JSON Pointer string to resolve.

        Returns:
            A Pydantic `BaseModel` instance representing the resolved component.

        Raises:
            ValueError: If the reference is circular (for non-schema components), invalid,
                        or if parsing/validation into the Pydantic model fails.
        """
        if ref_str in self._resolved_references:
            return self._resolved_references[ref_str]

        if ref_str in self._currently_resolving:
             raise ValueError(f"Circular reference detected for: {ref_str}. Direct circular references for non-schema components are problematic.")


        self._currently_resolving.add(ref_str)
        
        try:
            target_dict = self._get_target_dict_from_ref(ref_str)
            model_type = self._get_model_type_for_ref(ref_str)
            resolved_target_dict = self._recursively_resolve_refs_in_dict_or_list(target_dict)
            
            previous_resolver = get_current_resolver()
            set_current_resolver(self) # Ensure nested ReferenceObjects can find this resolver
            
            try:
                if issubclass(model_type, RootModel): 
                    instance = model_type(root=resolved_target_dict)
                else:
                    instance = model_type.model_validate(resolved_target_dict)
            finally:
                set_current_resolver(previous_resolver) 

            self._resolved_references[ref_str] = instance
            return instance
        finally:
            self._currently_resolving.remove(ref_str)


    def _recursively_resolve_refs_in_dict_or_list(self, data: Any) -> Any:
        """
        Recursively traverses a dictionary or list, resolving any `$ref` strings.

        If an item in the data is a dictionary representing a reference (e.g., `{"$ref": "..."}`),
        it's replaced by the resolved Pydantic model instance. Otherwise, traversal continues.

        Args:
            data: The dictionary or list to traverse.

        Returns:
            The data structure with `$ref`s resolved to their model instances.
        """
        if isinstance(data, dict):
            if '$ref' in data and len(data) == 1 and isinstance(data['$ref'], str) : # Strictly a reference object
                return self._resolve_reference_str_to_model(data['$ref'])
            else:
                new_dict = {}
                for key, value in data.items():
                    new_dict[key] = self._recursively_resolve_refs_in_dict_or_list(value)
                return new_dict
        elif isinstance(data, list):
            new_list = []
            for item in data:
                new_list.append(self._recursively_resolve_refs_in_dict_or_list(item))
            return new_list
        else:
            return data 

    def resolve_all(self) -> OpenAPIObject:
        """
        Resolves all internal `$ref` references throughout the entire `OpenAPIObject`.

        This method traverses the main sections of the OpenAPI document (paths, components, webhooks)
        and resolves any `ReferenceObject` instances found within them or their children.
        The resolution mutates the `spec_obj` in place.

        Returns:
            The mutated `OpenAPIObject` with references resolved.
        """
        if self.spec_obj.paths:
            resolved_paths_root: Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]] = {}
            for path_key, path_item_or_ref in self.spec_obj.paths.root.items(): 
                resolved_paths_root[path_key] = self._resolve_field_if_ref(path_item_or_ref, PathItemObject) # type: ignore
            self.spec_obj.paths = PathsModel(root=resolved_paths_root) 

        if self.spec_obj.components:
            self._resolve_components_obj(self.spec_obj.components)
            
        if self.spec_obj.webhooks: 
            resolved_webhooks: Dict[str, Union[PathItemObject, ReferenceObject[PathItemObject]]] = {}
            for key, item_or_ref in self.spec_obj.webhooks.items():
                 resolved_webhooks[key] = self._resolve_field_if_ref(item_or_ref, PathItemObject) # type: ignore
            self.spec_obj.webhooks = resolved_webhooks

        return self.spec_obj

    def _resolve_field_if_ref(self, field_value: Any, expected_type: Type[ReferencedType]) -> Union[ReferencedType, Any]: # type: ignore
        """
        Helper to resolve a field if it's a `ReferenceObject`.

        If the field is a `ReferenceObject`, it's resolved. If it's already an instance
        of the `expected_type` (and a `BaseModel`), its internal fields are then recursively
        resolved. Otherwise, the field value is returned as is.

        Args:
            field_value: The value of the field to potentially resolve.
            expected_type: The Pydantic model type expected after resolution or if already a model.

        Returns:
            The resolved Pydantic model, the processed model, or the original value.
        """
        if isinstance(field_value, ReferenceObject):
            return field_value.resolve(self) 
        elif isinstance(field_value, expected_type) and isinstance(field_value, BaseModel): 
            if isinstance(field_value, PathItemObject):
                self._resolve_path_item_obj(field_value)
            elif isinstance(field_value, OperationObject):
                self._resolve_operation_obj(field_value)
            elif isinstance(field_value, SchemaObject):
                self._resolve_schema_obj(field_value)
            # Extend with other specific model types that have nested resolvable fields
            return field_value
        return field_value 


    def _resolve_components_obj(self, components: ComponentsObject) -> None:
        """Recursively resolves references within a `ComponentsObject`."""
        component_map_fields = {
            "schemas": SchemaObject, "responses": ResponseObject, "parameters": ParameterObject,
            "examples": ExampleObject, "requestBodies": RequestBodyObject, "headers": HeaderObject,
            "securitySchemes": SecuritySchemeObject, "links": LinkObject, "callbacks": CallbackModel,
            "pathItems": PathItemObject
        }
        for field_name_str, ExpectedComponentType in component_map_fields.items():
            # Ensure field_name_str is a valid attribute name for ComponentsObject
            field_name = cast(str, field_name_str) 
            component_map = getattr(components, field_name, None)
            if component_map: # component_map is Dict[str, Union[ComponentType, ReferenceObject[ComponentType]]]
                resolved_map: Dict[str, Any] = {}
                for key, comp_or_ref in component_map.items():
                    resolved_map[key] = self._resolve_field_if_ref(comp_or_ref, ExpectedComponentType) # type: ignore
                setattr(components, field_name, resolved_map)


    def _resolve_path_item_obj(self, path_item: PathItemObject) -> None:
        """Recursively resolves references within a `PathItemObject`."""
        if path_item.parameters:
            resolved_params: List[Union[ParameterObject, ReferenceObject[ParameterObject]]] = []
            for p_or_r in path_item.parameters:
                resolved_params.append(self._resolve_field_if_ref(p_or_r, ParameterObject)) # type: ignore
            path_item.parameters = resolved_params
        
        for op_name in ['get', 'put', 'post', 'delete', 'options', 'head', 'patch', 'trace']:
            op_obj = getattr(path_item, op_name, None)
            if op_obj: 
                setattr(path_item, op_name, self._resolve_field_if_ref(op_obj, OperationObject))


    def _resolve_operation_obj(self, operation: OperationObject) -> None:
        """Recursively resolves references within an `OperationObject`."""
        if operation.parameters:
            resolved_params: List[Union[ParameterObject, ReferenceObject[ParameterObject]]] = []
            for p_or_r in operation.parameters:
                resolved_params.append(self._resolve_field_if_ref(p_or_r, ParameterObject)) # type: ignore
            operation.parameters = resolved_params
        
        if operation.requestBody:
            operation.requestBody = self._resolve_field_if_ref(operation.requestBody, RequestBodyObject) # type: ignore
            if isinstance(operation.requestBody, RequestBodyObject): 
                for media_type_obj in operation.requestBody.content.values():
                    if media_type_obj.schema_:
                        media_type_obj.schema_ = self._resolve_field_if_ref(media_type_obj.schema_, SchemaObject) # type: ignore
                    if media_type_obj.examples:
                        for ex_key, ex_or_ref in media_type_obj.examples.items():
                            media_type_obj.examples[ex_key] = self._resolve_field_if_ref(ex_or_ref, ExampleObject) # type: ignore
        
        if operation.responses and isinstance(operation.responses, ResponsesModel): 
            resolved_responses_root: Dict[str, Any] = {}
            for code, resp_or_ref in operation.responses.items(): 
                resolved_resp = self._resolve_field_if_ref(resp_or_ref, ResponseObject) # Corrected: self._resolve_field_if_ref
                if isinstance(resolved_resp, ResponseObject):
                    if resolved_resp.headers:
                        for h_key, h_or_r in resolved_resp.headers.items():
                            resolved_resp.headers[h_key] = self._resolve_field_if_ref(h_or_r, HeaderObject) # type: ignore
                    if resolved_resp.content:
                        for media_type_obj in resolved_resp.content.values():
                            if media_type_obj.schema_:
                                media_type_obj.schema_ = self._resolve_field_if_ref(media_type_obj.schema_, SchemaObject) # type: ignore
                            if media_type_obj.examples:
                                for ex_key, ex_or_ref in media_type_obj.examples.items():
                                     media_type_obj.examples[ex_key] = self._resolve_field_if_ref(ex_or_ref, ExampleObject) # type: ignore
                    if resolved_resp.links:
                        for l_key, l_or_r in resolved_resp.links.items():
                            resolved_resp.links[l_key] = self._resolve_field_if_ref(l_or_r, LinkObject) # type: ignore
                resolved_responses_root[code] = resolved_resp
            operation.responses = ResponsesModel(root=resolved_responses_root)

        if operation.callbacks:
            resolved_callbacks: Dict[str, Any] = {}
            for cb_name, cb_or_ref in operation.callbacks.items():
                resolved_cb = self._resolve_field_if_ref(cb_or_ref, CallbackModel) # type: ignore
                if isinstance(resolved_cb, CallbackModel):
                    resolved_cb_root: Dict[str, Any] = {}
                    for key_expr, pi_or_ref in resolved_cb.root.items():
                        resolved_cb_root[key_expr] = self._resolve_field_if_ref(pi_or_ref, PathItemObject)
                    resolved_cb.root = resolved_cb_root
                resolved_callbacks[cb_name] = resolved_cb
            operation.callbacks = resolved_callbacks

    def _resolve_schema_obj(self, schema: SchemaObject) -> None:
        """Recursively resolves references within a `SchemaObject`'s fields."""
        def _resolve_sub_schema(sub_schema_or_ref: Optional[Union[SchemaObject, ReferenceObject[SchemaObject]]]) -> Optional[SchemaObject]: # Added SchemaObject to ReferenceObject Union
            if isinstance(sub_schema_or_ref, ReferenceObject):
                resolved = sub_schema_or_ref.resolve(self)
                if isinstance(resolved, SchemaObject): 
                    self._resolve_schema_obj(resolved) 
                    return resolved
                return cast(SchemaObject, resolved) 
            elif isinstance(sub_schema_or_ref, SchemaObject):
                self._resolve_schema_obj(sub_schema_or_ref) 
                return sub_schema_or_ref
            return None 

        for field_name_str in ['not_', 'items', 'contains', 'propertyNames']:
            field_name = cast(str, field_name_str)
            current_val = getattr(schema, field_name, None)
            if current_val:
                setattr(schema, field_name, _resolve_sub_schema(current_val)) # type: ignore

        for field_name_str in ['allOf', 'oneOf', 'anyOf', 'prefixItems']:
            field_name = cast(str, field_name_str)
            current_list = getattr(schema, field_name, None)
            if current_list:
                new_list = [_resolve_sub_schema(item) for item in current_list if item is not None]
                setattr(schema, field_name, [item for item in new_list if item is not None])


        for field_name_str in ['properties', 'patternProperties']:
            field_name = cast(str, field_name_str)
            current_dict = getattr(schema, field_name, None)
            if current_dict:
                new_dict = {k: _resolve_sub_schema(v) for k, v in current_dict.items() if v is not None}
                setattr(schema, field_name, {k:v for k,v in new_dict.items() if v is not None})
        
        if isinstance(schema.additionalProperties, (SchemaObject, ReferenceObject)):
            schema.additionalProperties = _resolve_sub_schema(cast(Union[SchemaObject, ReferenceObject[SchemaObject]], schema.additionalProperties))


# --- Parsing Functions ---
def _parse_content(content_str: str, content_type: str = "yaml", resolve_refs: bool = True) -> OpenAPIObject:
    """
    Helper to parse YAML or JSON content string into an OpenAPIObject.

    Args:
        content_str: The string content to parse.
        content_type: The type of content ('yaml' or 'json'). Defaults to 'yaml'.
        resolve_refs: Whether to resolve internal $ref references. Defaults to True.

    Returns:
        An `OpenAPIObject` instance.

    Raises:
        ValueError: If content_type is unsupported, content is invalid,
                    or OpenAPI specification validation fails.
    """
    try:
        if content_type == "yaml":
            raw_spec_dict = yaml.safe_load(content_str)
        elif content_type == "json":
            raw_spec_dict = json.loads(content_str)
        else:
            raise ValueError("Unsupported content type. Must be 'yaml' or 'json'.")

        if not isinstance(raw_spec_dict, dict):
            raise ValueError(f"Parsed content is not a dictionary (actual type: {type(raw_spec_dict)}).")

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML content: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON content: {e}")

    try:
        spec_obj = OpenAPIObject.model_validate(raw_spec_dict)
        
        if resolve_refs:
            resolver = OpenAPIResolver(spec_obj=spec_obj, raw_spec_dict=raw_spec_dict)
            spec_obj = resolver.resolve_all() 
        
        return spec_obj
    except ValidationError as e: 
        raise ValueError(f"OpenAPI specification validation error: {e}")


def parse_yaml(yaml_content: str, resolve_refs: bool = True) -> OpenAPIObject:
    """
    Parses YAML content string into an OpenAPIObject.

    Args:
        yaml_content: The YAML string content.
        resolve_refs: Whether to resolve internal $ref references. Defaults to True.
    Returns: An `OpenAPIObject` instance.
    """
    return _parse_content(yaml_content, "yaml", resolve_refs)

def parse_json(json_content: str, resolve_refs: bool = True) -> OpenAPIObject:
    """
    Parses JSON content string into an OpenAPIObject.

    Args:
        json_content: The JSON string content.
        resolve_refs: Whether to resolve internal $ref references. Defaults to True.
    Returns: An `OpenAPIObject` instance.
    """
    return _parse_content(json_content, "json", resolve_refs)

def parse_yaml_file(file_path: str, resolve_refs: bool = True) -> OpenAPIObject:
    """
    Parses a YAML file into an OpenAPIObject.

    Args:
        file_path: Path to the YAML file.
        resolve_refs: Whether to resolve internal $ref references. Defaults to True.
    Returns: An `OpenAPIObject` instance.
    Raises: FileNotFoundError or ValueError for read/parse errors.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return parse_yaml(content, resolve_refs)
    except FileNotFoundError:
        raise 
    except Exception as e: 
        raise ValueError(f"Error reading YAML file '{file_path}': {e}")

def parse_json_file(file_path: str, resolve_refs: bool = True) -> OpenAPIObject:
    """
    Parses a JSON file into an OpenAPIObject.

    Args:
        file_path: Path to the JSON file.
        resolve_refs: Whether to resolve internal $ref references. Defaults to True.
    Returns: An `OpenAPIObject` instance.
    Raises: FileNotFoundError or ValueError for read/parse errors.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return parse_json(content, resolve_refs)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Error reading JSON file '{file_path}': {e}")

# Rebuild models to resolve forward references after all class definitions
PathItemObject.model_rebuild()
CallbackModel.model_rebuild()
SchemaObject.model_rebuild()
ComponentsObject.model_rebuild()
MediaTypeObject.model_rebuild() 
ParameterObject.model_rebuild()
HeaderObject.model_rebuild()
ResponseObject.model_rebuild()
OperationObject.model_rebuild()
