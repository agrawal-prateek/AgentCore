"""Tests for LLM summarization policy."""

from __future__ import annotations

from typing import Any

import pytest

from agent_core.context.llm_summarization_policy import LLMSummarizationPolicy
from agent_core.llm.llm_adapter import LLMAdapter, LLMRequest, LLMResponse


class FakeLLMAdapter(LLMAdapter):
    """Fake adapter that returns a configurable summary."""

    def __init__(self, summary: str = "Found 3 code files in payment service.") -> None:
        self._summary = summary

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content={"summary": self._summary},
            prompt_tokens=100,
            completion_tokens=20,
            raw_text=f'{{"summary": "{self._summary}"}}',
        )


class FailingLLMAdapter(LLMAdapter):
    """Fake adapter that always fails."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError("LLM is down")


@pytest.mark.asyncio
async def test_llm_summarization_produces_summary() -> None:
    adapter = FakeLLMAdapter(summary="Payment handler processes transactions via Stripe API.")
    policy = LLMSummarizationPolicy(adapter=adapter, model="test-model", max_chars=600)

    payload = {
        "count": 5,
        "snippets": [
            {"file_path": f"src/payment/handler_{i}.py", "content": f"def process_payment_{i}(amount, currency): validate(amount); charge(currency); log_transaction(id={i}); return receipt" * 3}
            for i in range(5)
        ],
    }
    result = await policy.summarize_async(payload)

    assert "Payment handler" in result.summary
    assert result.compression_score > 0.0


@pytest.mark.asyncio
async def test_llm_summarization_falls_back_on_failure() -> None:
    adapter = FailingLLMAdapter()
    policy = LLMSummarizationPolicy(adapter=adapter, model="test-model", max_chars=600)

    payload = {
        "count": 5,
        "snippets": [{"file_path": "src/payment.py", "content": "code..." * 20}],
    }
    result = await policy.summarize_async(payload)

    # Should get a deterministic fallback summary, not crash
    assert len(result.summary) > 0
    assert "payment.py" in result.summary  # Structured extraction should work


@pytest.mark.asyncio
async def test_llm_summarization_skips_small_payloads() -> None:
    adapter = FakeLLMAdapter(summary="This should NOT be used")
    policy = LLMSummarizationPolicy(adapter=adapter, model="test-model", max_chars=300)

    # Small payload -> deterministic, no LLM call
    payload = {"status": "ok", "count": 1}
    result = await policy.summarize_async(payload)

    assert "This should NOT be used" not in result.summary


def test_sync_summarize_uses_deterministic() -> None:
    adapter = FakeLLMAdapter()
    policy = LLMSummarizationPolicy(adapter=adapter, model="test-model", max_chars=300)

    payload = {"status": "ok", "count": 1}
    result = policy.summarize(payload)

    assert len(result.summary) > 0
