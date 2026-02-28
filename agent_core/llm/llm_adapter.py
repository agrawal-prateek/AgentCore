from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LLMTraceContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    investigation_id: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    agent_id: str = Field(min_length=1)
    agent_role: str = Field(min_length=1)
    task: str = Field(min_length=1)


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    system_prompt: str
    user_payload: dict[str, Any]
    seed: int | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    trace_context: LLMTraceContext | None = None


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: dict[str, Any]
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    raw_text: str = ""
    provider_payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class LLMAdapter(ABC):
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
