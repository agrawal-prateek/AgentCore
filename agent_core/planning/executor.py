from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_core.llm.llm_adapter import LLMAdapter, LLMRequest


class ExecutorProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    expected_outcome: str = Field(min_length=1)


class Executor(ABC):
    @abstractmethod
    async def propose(self, context_payload: dict[str, Any], objective: str) -> ExecutorProposal:
        raise NotImplementedError


class LLMExecutor(Executor):
    """Executor proposes tool calls only; no direct state mutations."""

    def __init__(self, *, adapter: LLMAdapter, model: str, system_prompt: str) -> None:
        self._adapter = adapter
        self._model = model
        self._system_prompt = system_prompt

    async def propose(self, context_payload: dict[str, Any], objective: str) -> ExecutorProposal:
        request = LLMRequest(
            model=self._model,
            system_prompt=self._system_prompt,
            user_payload={
                "task": "executor",
                "objective": objective,
                "context": context_payload,
                "output_schema": ExecutorProposal.model_json_schema(),
            },
        )
        response = await self._adapter.complete(request)
        return ExecutorProposal.model_validate(response.content)
