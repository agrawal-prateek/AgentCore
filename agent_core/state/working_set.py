from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WorkingSet(BaseModel):
    """Strict LLM-visible short-term context."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    current_phase: str = Field(min_length=1)
    active_stack_node_id: str = Field(min_length=1)
    parent_summaries: list[str] = Field(default_factory=list)
    top_hypotheses: list[str] = Field(default_factory=list)
    top_evidence_summaries: list[str] = Field(default_factory=list)
    current_tool_result_summary: str = Field(default="")
