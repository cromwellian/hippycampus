from typing import Optional

from langflow.custom import Component
from langflow.inputs import SecretStrInput
from langflow.io import MessageTextInput, Output
from pydantic import BaseModel


class OpenApiSpec(BaseModel):
    url: str
    token: Optional[str] = None


class OpenApiService(Component):
    openapi_definition = OpenApiSpec
    display_name = "OpenApi Service"
    description = "Specifies an OpenApi exposed service for the Hippycampus MCP Server."
    documentation: str = "http://docs.langflow.org/components/custom"
    icon = "code"
    name = "OpenApiService"

    inputs = [
        MessageTextInput(
            name="url",
            display_name="OpenAPI YAML Url",
            info="Yaml Url to load tools from",
            value="https://raw.githubusercontent.com/APIs-guru/unofficial_openapi_specs/master/xkcd.com/1.0.0/openapi.yaml",
            tool_mode=True,
        ),
        SecretStrInput(
            name="token",
            display_name="Optional API Token",
            required=False,
            info="Optional API Token to use for authentication",
        ),
    ]

    outputs = [
        Output(display_name="OpenApi Definition", name="openapi_definition", method="build_output"),
    ]

    async def build_output(self) -> OpenApiSpec:
        return OpenApiSpec(url=self.url, token=self.token)
