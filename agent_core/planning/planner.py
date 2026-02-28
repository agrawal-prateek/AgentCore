from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_core.llm.llm_adapter import LLMAdapter, LLMRequest, LLMTraceContext
from agent_core.state.agent_tree import AgentSpawnRequest


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    next_objective: str = Field(min_length=1)
    target_branch_id: str = Field(min_length=1)
    phase_transition: str | None = None
    reasoning_summary: str = Field(min_length=1)
    termination_flag: bool = False
    spawn_children: tuple[AgentSpawnRequest, ...] = Field(default_factory=tuple)


class Planner(ABC):
    last_trace_id: str | None = None

    @abstractmethod
    async def plan(
        self,
        context_payload: dict[str, Any],
        *,
        trace_context: LLMTraceContext | None = None,
    ) -> PlannerOutput:
        raise NotImplementedError


class LLMPlanner(Planner):
    """Planner uses a dedicated model and strict output schema."""

    def __init__(self, *, adapter: LLMAdapter, model: str, system_prompt: str) -> None:
        self._adapter = adapter
        self._model = model
        self._system_prompt = system_prompt
        self.last_trace_id: str | None = None

    async def plan(
        self,
        context_payload: dict[str, Any],
        *,
        trace_context: LLMTraceContext | None = None,
    ) -> PlannerOutput:
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
                "task": "planner",
                "context": context_payload,
                "agent": agent_metadata,
                "output_schema": PlannerOutput.model_json_schema(),
            },
            trace_context=trace_context,
        )
        self.last_trace_id = None
        response = await self._adapter.complete(request)
        self.last_trace_id = response.trace_id
        return PlannerOutput.model_validate(response.content)
