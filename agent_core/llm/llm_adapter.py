from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    system_prompt: str
    user_payload: dict[str, Any]
    seed: int | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: dict[str, Any]
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class LLMAdapter(ABC):
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
