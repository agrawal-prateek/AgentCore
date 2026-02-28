from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class PhaseDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    allowed_next_phases: tuple[str, ...] = Field(default_factory=tuple)
    allowed_tools: tuple[str, ...] = Field(default_factory=tuple)


class CompletionSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    should_complete: bool
    reason: str = Field(default="")


class ProfileInterface(ABC):
    """Profile contract for domain-specific behavior without engine coupling."""

    @abstractmethod
    def phases(self) -> Mapping[str, PhaseDefinition]:
        raise NotImplementedError

    @abstractmethod
    def initial_phase(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def planner_prompt_template(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def executor_prompt_template(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def completion_criteria(self, *, phase: str, top_confidence: float) -> CompletionSignal:
        raise NotImplementedError

    @abstractmethod
    def domain_constraints(self) -> Sequence[str]:
        raise NotImplementedError
