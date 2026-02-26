from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


def estimate_tokens(text: str) -> int:
    """Deterministic rough token estimate for budget enforcement."""
    if not text:
        return 0
    words = text.split()
    # Conservative estimate to keep hard bounds under provider variance.
    return max(1, int(len(words) * 1.35))


class TokenBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(ge=1)
    reserved_response_tokens: int = Field(default=512, ge=0)

    @property
    def available_context_tokens(self) -> int:
        available = self.max_tokens - self.reserved_response_tokens
        return max(0, available)

    def fits(self, token_count: int) -> bool:
        return token_count <= self.available_context_tokens
