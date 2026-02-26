from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class HypothesisStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    REFUTED = "refuted"
    STALE = "stale"


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    status: HypothesisStatus = Field(default=HypothesisStatus.CANDIDATE)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    refuting_evidence_ids: list[str] = Field(default_factory=list)
    last_updated_iteration: int = Field(default=0, ge=0)

    def recalculate_confidence(self, current_iteration: int) -> None:
        """Deterministic confidence recalculation using evidence counts and staleness."""
        support = min(3, len(self.supporting_evidence_ids)) * 0.12
        refute = min(3, len(self.refuting_evidence_ids)) * 0.18
        stale_penalty = 0.0
        if current_iteration - self.last_updated_iteration >= 5:
            stale_penalty = 0.08

        score = 0.5 + support - refute - stale_penalty
        self.confidence_score = max(0.0, min(1.0, round(score, 4)))

        if self.confidence_score >= 0.8:
            self.status = HypothesisStatus.VALIDATED
        elif self.confidence_score <= 0.2:
            self.status = HypothesisStatus.REFUTED
        elif stale_penalty > 0.0:
            self.status = HypothesisStatus.STALE
        else:
            self.status = HypothesisStatus.CANDIDATE
