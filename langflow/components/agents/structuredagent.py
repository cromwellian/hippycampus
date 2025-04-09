from langchain import hub
from langchain_core.tools import StructuredTool

from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_DYNAMIC_UPDATE_FIELDS,
    MODEL_PROVIDERS_DICT,
)
from langflow.base.models.model_utils import get_model_name
from langflow.components.helpers import CurrentDateComponent
from langflow.components.helpers.memory import MemoryComponent
from langflow.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from langflow.custom.custom_component.component import _get_component_toolkit
from langflow.custom.utils import update_component_build_config
from langflow.field_typing import Tool
from langflow.io import BoolInput, DropdownInput, MultilineInput, Output
from langflow.logging import logger
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message

from hippycampus.langchain_util import fixed_create_structured_chat_agent


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


class StructuredAgentComponent(ToolCallingAgentComponent):
    display_name: str = "Hippycampus Structured Agent"
    description: str = "Define the agent's instructions, then enter a task to complete using tools."
    icon = "bot"
    beta = False
    name = "Hippycampus Structured Agent"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model that the agent will use to generate responses.",
            options=[*sorted(MODEL_PROVIDERS_DICT.keys()), "Custom"],
            value="OpenAI",
            real_time_refresh=True,
            input_types=[],
        ),
        *MODEL_PROVIDERS_DICT["OpenAI"]["inputs"],
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
            advanced=False,
        ),
        *LCToolsAgentComponent._base_inputs,
        *memory_inputs,
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
    ]
    outputs = [Output(name="response", display_name="Response", method="message_response")]

    async def message_response(self) -> Message:
        try:
            llm_model, display_name = self.get_llm()
            if llm_model is None:
                msg = "No language model selected"
                raise ValueError(msg)
            self.model_name = get_model_name(llm_model, display_name=display_name)
        except Exception as e:
            # Log the error for debugging purposes
            logger.error(f"Error retrieving language model: {e}")
            raise

        try:
            self.chat_history = await self.get_memory_data()
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            raise

        if self.add_current_date_tool:
            try:
                if not isinstance(self.tools, list):  # type: ignore[has-type]
                    self.tools = []
                # Convert CurrentDateComponent to a StructuredTool
                current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)
                if isinstance(current_date_tool, StructuredTool):
                    self.tools.append(current_date_tool)
                else:
                    msg = "CurrentDateComponent must be converted to a StructuredTool"
                    raise TypeError(msg)
            except Exception as e:
                logger.error(f"Error adding current date tool: {e}")
                raise

        if not self.tools:
            msg = "Tools are required to run the agent."
            logger.error(msg)
            raise ValueError(msg)

        try:
            self.set(
                llm=llm_model,
                tools=self.tools,
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )
            agent = self.create_agent_runnable()
        except Exception as e:
            logger.error(f"Error setting up the agent: {e}")
            raise

        return await self.run_agent(agent)

    async def get_memory_data(self):
        memory_kwargs = {
            component_input.name: getattr(self, f"{component_input.name}") for component_input in self.memory_inputs
        }
        # filter out empty values
        memory_kwargs = {k: v for k, v in memory_kwargs.items() if v}

        return await MemoryComponent(**self.get_base_args()).set(**memory_kwargs).retrieve_messages()

    def get_llm(self):
        if isinstance(self.agent_llm, str):
            try:
                provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
                if provider_info:
                    component_class = provider_info.get("component_class")
                    display_name = component_class.display_name
                    inputs = provider_info.get("inputs")
                    prefix = provider_info.get("prefix", "")
                    return (
                        self._build_llm_model(component_class, inputs, prefix),
                        display_name,
                    )
            except Exception as e:
                msg = f"Error building {self.agent_llm} language model"
                raise ValueError(msg) from e
        return self.agent_llm, None

    def _build_llm_model(self, component, inputs, prefix=""):
        model_kwargs = {input_.name: getattr(self, f"{prefix}{input_.name}") for input_ in inputs}
        return component.set(**model_kwargs).build_model()

    def set_component_params(self, component):
        provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
        if provider_info:
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix")
            model_kwargs = {input_.name: getattr(self, f"{prefix}{input_.name}") for input_ in inputs}

            return component.set(**model_kwargs)
        return component

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config."""
        for field in fields:
            build_config.pop(field, None)

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """Update input types for all fields in build_config."""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
        return build_config

    async def update_build_config(
            self, build_config: dotdict, field_value: str, field_name: str | None = None
    ) -> dotdict:
        # Iterate over all providers in the MODEL_PROVIDERS_DICT
        # Existing logic for updating build_config
        if field_name in ("agent_llm",):
            build_config["agent_llm"]["value"] = field_value
            provider_info = MODEL_PROVIDERS_DICT.get(field_value)
            if provider_info:
                component_class = provider_info.get("component_class")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call the component class's update_build_config method
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )

            provider_configs: dict[str, tuple[dict, list[dict]]] = {
                provider: (
                    MODEL_PROVIDERS_DICT[provider]["fields"],
                    [
                        MODEL_PROVIDERS_DICT[other_provider]["fields"]
                        for other_provider in MODEL_PROVIDERS_DICT
                        if other_provider != provider
                    ],
                )
                for provider in MODEL_PROVIDERS_DICT
            }
            if field_value in provider_configs:
                fields_to_add, fields_to_delete = provider_configs[field_value]

                # Delete fields from other providers
                for fields in fields_to_delete:
                    self.delete_fields(build_config, fields)

                # Add provider-specific fields
                if field_value == "OpenAI" and not any(field in build_config for field in fields_to_add):
                    build_config.update(fields_to_add)
                else:
                    build_config.update(fields_to_add)
                # Reset input types for agent_llm
                build_config["agent_llm"]["input_types"] = []
            elif field_value == "Custom":
                # Delete all provider fields
                self.delete_fields(build_config, ALL_PROVIDER_FIELDS)
                # Update with custom component
                custom_component = DropdownInput(
                    name="agent_llm",
                    display_name="Language Model",
                    options=[*sorted(MODEL_PROVIDERS_DICT.keys()), "Custom"],
                    value="Custom",
                    real_time_refresh=True,
                    input_types=["LanguageModel"],
                )
                build_config.update({"agent_llm": custom_component.to_dict()})
            # Update input types for all fields
            build_config = self.update_input_types(build_config)

            # Validate required keys
            default_keys = [
                "code",
                "_type",
                "agent_llm",
                "tools",
                "input_value",
                "add_current_date_tool",
                "system_prompt",
                "agent_description",
                "max_iterations",
                "handle_parsing_errors",
                "verbose",
            ]
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        if (
                isinstance(self.agent_llm, str)
                and self.agent_llm in MODEL_PROVIDERS_DICT
                and field_name in MODEL_DYNAMIC_UPDATE_FIELDS
        ):
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if provider_info:
                component_class = provider_info.get("component_class")
                component_class = self.set_component_params(component_class)
                prefix = provider_info.get("prefix")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call each component class's update_build_config method
                    # remove the prefix from the field_name
                    if isinstance(field_name, str) and isinstance(prefix, str):
                        field_name = field_name.replace(prefix, "")
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    def create_agent_runnable(self):
        prompt = hub.pull("hwchase17/structured-chat-agent")
        prompt.messages[0].prompt = PromptTemplate.from_template("""Respond to the human as helpfully and accurately as possible. You have access to the following tools:
        

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Before using a tool for the first time, you MUST fetch additional documentation using fetch_documentation_for_tool and 
providing the tool name you are attempting to use, this will return the response schema as well as examples if 
available. Remember, the document examples from fetch_documentation_for_tool just show you the structure of the response, not the actual response, 
only use the actual tool response for extracting answers. If you are unsure of a parameter, such as repo or owner,
 ask the user. Creating or updating or commiting files in a GitHub repository requires using the repos_create_or_update_file_contents. Do not perform base64 encoding or decoding,
 the tool will handle that for you. If you need to update an existing field, you need to retrieve it first to obtain it's sha field, and then
  supply the sha when using subsequent calls to repos_create_or_update_file_contents.
  
Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. 
Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation\n""")
        self.validate_tool_names()
        try:
            return fixed_create_structured_chat_agent(self.llm, self.tools or [], prompt)
        except NotImplementedError as e:
            message = f"{self.display_name} does not support tool calling. Please try using a compatible model."
            raise NotImplementedError(message) from e

    async def to_toolkit(self) -> list[Tool]:
        component_toolkit = _get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        # TODO: Agent Description Depreciated Feature to be removed
        description = f"{agent_description}{tools_names}"
        tools = component_toolkit(component=self).get_tools(
            tool_name=self.get_tool_name(), tool_description=description, callbacks=self.get_langchain_callbacks()
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)
        return tools
