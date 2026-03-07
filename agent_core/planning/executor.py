from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_core.llm.llm_adapter import LLMAdapter, LLMRequest, LLMTraceContext


class ExecutorProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    expected_outcome: str = Field(min_length=1)


class Executor(ABC):
    last_trace_id: str | None = None

    @abstractmethod
    async def propose(
        self,
        context_payload: dict[str, Any],
        objective: str,
        *,
        trace_context: LLMTraceContext | None = None,
    ) -> ExecutorProposal:
        raise NotImplementedError


class LLMExecutor(Executor):
    """Executor proposes tool calls only; no direct state mutations."""

    def __init__(self, *, adapter: LLMAdapter, model: str, system_prompt: str, temperature: float = 0.0) -> None:
        self._adapter = adapter
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self.last_trace_id: str | None = None

    async def propose(
        self,
        context_payload: dict[str, Any],
        objective: str,
        *,
        trace_context: LLMTraceContext | None = None,
    ) -> ExecutorProposal:
        agent_metadata = None
        if trace_context is not None:
            agent_metadata = {
                "id": trace_context.agent_id,
                "role": trace_context.agent_role,
                "iteration": trace_context.iteration,
            }
        request = LLMRequest(
            model=self._model,
            system_prompt=self._system_prompt,
            user_payload={
                "task": "executor",
                "objective": objective,
                "context": context_payload,
                "agent": agent_metadata,
                "output_schema": ExecutorProposal.model_json_schema(),
            },
            temperature=self._temperature,
            trace_context=trace_context,
        )
        self.last_trace_id = None
        response = await self._adapter.complete(request)
        self.last_trace_id = response.trace_id
        return ExecutorProposal.model_validate(response.content)
