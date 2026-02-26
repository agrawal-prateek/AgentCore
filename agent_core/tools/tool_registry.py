from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel, ConfigDict, Field


class ToolExecutionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    async def run(self, args: BaseModel) -> ToolExecutionPayload:
        raise NotImplementedError


class ToolRegistry:
    """Registry for tools with typed argument validation."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def exists(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool '{name}' is not registered") from exc

    def validate_arguments(self, tool_name: str, raw_args: dict[str, Any]) -> BaseModel:
        tool = self.get(tool_name)
        return tool.args_schema.model_validate(raw_args)
