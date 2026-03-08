from __future__ import annotations

import logging
from typing import Any

from agent_core.context.summarization_policy import (
    DeterministicSummarizationPolicy,
    SummarizationPolicy,
    ToolExecutionSummary,
)
from agent_core.context.token_budget import estimate_tokens
from agent_core.llm.llm_adapter import LLMAdapter, LLMRequest, LLMTraceContext

logger = logging.getLogger("agent_core.summarizer")

_SUMMARIZER_PROMPT = """You are a tool-output summarizer for an autonomous investigation agent.
Given a raw tool output, produce a concise summary that preserves:
- File paths and function/class names
- Key findings, patterns, or anomalies
- Numeric counts and identifiers (trace IDs, error codes)
- Relationship to the investigation goal

Output ONLY a JSON object: {{"summary": "<your concise summary>"}}
Keep the summary under {max_chars} characters. You MUST include relevant code snippets, database schemas, and critical log lines if they directly answer the agent's objective."""


class LLMSummarizationPolicy(SummarizationPolicy):
    """LLM-based summarization that falls back to deterministic extraction on failure."""

    def __init__(
        self,
        *,
        adapter: LLMAdapter,
        model: str,
        max_chars: int = 600,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._max_chars = max_chars
        self._fallback = DeterministicSummarizationPolicy(max_chars=max_chars)
        self._system_prompt = _SUMMARIZER_PROMPT.format(max_chars=max_chars)

    def summarize(self, payload: dict[str, Any], context: str | None = None, trace_context: LLMTraceContext | None = None) -> ToolExecutionSummary:
        """Synchronous interface — uses deterministic fallback only."""
        return self._fallback.summarize(payload, context, trace_context)

    async def summarize_async(self, payload: dict[str, Any], context: str | None = None, trace_context: LLMTraceContext | None = None) -> ToolExecutionSummary:
        """Async LLM summarization with deterministic fallback."""
        # Small payloads don't need LLM summarization
        raw_text = str(payload)
        if estimate_tokens(raw_text) < 100:
            return self._fallback.summarize(payload, context, trace_context)

        try:
            user_payload = {"task": "summarizer", "tool_output": payload}
            if context:
                user_payload["investigation_objective"] = context

            request = LLMRequest(
                model=self._model,
                system_prompt=self._system_prompt,
                user_payload=user_payload,
                temperature=0.0,
                trace_context=trace_context,
            )
            response = await self._adapter.complete(request)
            summary_text = str(response.content.get("summary", ""))
            if not summary_text or len(summary_text) < 5:
                return self._fallback.summarize(payload, context, trace_context)

            summary_text = summary_text[: self._max_chars].strip()
            source_tokens = max(1, estimate_tokens(raw_text))
            summary_tokens = max(1, estimate_tokens(summary_text))
            compression = min(1.0, round(summary_tokens / source_tokens, 4))

            return ToolExecutionSummary(
                summary=summary_text,
                entities=self._extract_entities(summary_text),
                compression_score=compression,
                trace_id=response.trace_id,
            )
        except Exception as exc:
            logger.warning("LLM summarization failed, falling back to deterministic: %s", exc)
            return self._fallback.summarize(payload, context, trace_context)

    @staticmethod
    def _extract_entities(text: str) -> tuple[str, ...]:
        import re

        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", text)
        return tuple(dict.fromkeys(candidates[:8]))
