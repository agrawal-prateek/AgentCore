from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_core.llm.llm_adapter import LLMAdapter, LLMRequest


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    next_objective: str = Field(min_length=1)
    target_branch_id: str = Field(min_length=1)
    phase_transition: str | None = None
    reasoning_summary: str = Field(min_length=1)
    termination_flag: bool = False


class Planner(ABC):
    @abstractmethod
    async def plan(self, context_payload: dict[str, Any]) -> PlannerOutput:
        raise NotImplementedError


class LLMPlanner(Planner):
    """Planner uses a dedicated model and strict output schema."""

    def __init__(self, *, adapter: LLMAdapter, model: str, system_prompt: str) -> None:
        self._adapter = adapter
        self._model = model
        self._system_prompt = system_prompt

    async def plan(self, context_payload: dict[str, Any]) -> PlannerOutput:
        request = LLMRequest(
            model=self._model,
            system_prompt=self._system_prompt,
            user_payload={
                "task": "planner",
                "context": context_payload,
                "output_schema": PlannerOutput.model_json_schema(),
            },
        )
        response = await self._adapter.complete(request)
        return PlannerOutput.model_validate(response.content)
