from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LLMTraceRequestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    investigation_id: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    agent_id: str = Field(min_length=1)
    agent_role: str = Field(min_length=1)
    task: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    system_prompt: str
    user_payload: dict[str, Any]
    raw_request: dict[str, Any]


class LLMTraceResponseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    raw_text: str
    parsed_content: dict[str, Any] | None = None
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    provider_payload: dict[str, Any] = Field(default_factory=dict)


class LLMTraceFailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    raw_text: str | None = None
    error_message: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    provider_payload: dict[str, Any] = Field(default_factory=dict)


class LLMTraceSinkPort(ABC):
    @abstractmethod
    async def record_request(self, record: LLMTraceRequestRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    async def record_response(self, record: LLMTraceResponseRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def record_failure(self, record: LLMTraceFailureRecord) -> None:
        raise NotImplementedError


class NoopLLMTraceSink(LLMTraceSinkPort):
    async def record_request(self, record: LLMTraceRequestRecord) -> str:
        return (
            f"trace:{record.investigation_id}:"
            f"{record.iteration}:{record.agent_id}:{record.task}"
        )

    async def record_response(self, record: LLMTraceResponseRecord) -> None:
        return None

    async def record_failure(self, record: LLMTraceFailureRecord) -> None:
        return None
