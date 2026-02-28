from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_core.llm.llm_adapter import LLMTraceContext

from pydantic import BaseModel, ConfigDict, Field

from agent_core.context.token_budget import estimate_tokens


class ToolExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    entities: tuple[str, ...] = Field(default_factory=tuple)
    compression_score: float = Field(ge=0.0, le=1.0)


class SummarizationPolicy(ABC):
    def summarize(self, payload: dict[str, Any], context: str | None = None, trace_context: 'LLMTraceContext' | None = None) -> ToolExecutionSummary:
        raise NotImplementedError


class DeterministicSummarizationPolicy(SummarizationPolicy):
    """Produces a fixed-shape summary with structured extraction for known tool outputs."""

    def __init__(self, max_chars: int = 320) -> None:
        self._max_chars = max_chars

    def summarize(self, payload: dict[str, Any], context: str | None = None, trace_context: 'LLMTraceContext' | None = None) -> ToolExecutionSummary:
        # Try structured extraction first for known tool output shapes
        structured = self._try_structured_extract(payload)
        if structured:
            trimmed = structured[: self._max_chars].strip()
        else:
            flattened = "; ".join(f"{k}={self._to_scalar(v)}" for k, v in sorted(payload.items()))
            trimmed = flattened[: self._max_chars].strip() or "empty-tool-output"

        entity_candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", trimmed)
        entities = tuple(dict.fromkeys(entity_candidates[:8]))

        source_tokens = estimate_tokens(str(payload))
        summary_tokens = max(1, estimate_tokens(trimmed))
        compression = min(1.0, round(summary_tokens / source_tokens, 4)) if source_tokens > 0 else 1.0
        return ToolExecutionSummary(summary=trimmed, entities=entities, compression_score=compression)

    def _try_structured_extract(self, payload: dict[str, Any]) -> str | None:
        """Extract meaningful content from known tool output shapes."""

        # Code search results: extract file paths and content previews
        snippets = payload.get("snippets")
        if isinstance(snippets, list) and snippets:
            count = payload.get("count", len(snippets))
            parts = [f"found {count} code results:"]
            for snippet in snippets[:4]:
                if isinstance(snippet, dict):
                    fp = snippet.get("file_path", "")
                    func = snippet.get("function_name", "")
                    cls = snippet.get("class_name", "")
                    content = str(snippet.get("content", ""))[:self._max_chars]
                    location = fp
                    if func:
                        location += f"::{func}"
                    elif cls:
                        location += f"::{cls}"
                    parts.append(f"  [{location}] {content}")
            return "\n".join(parts)

        # Grep results: extract match locations
        matches = payload.get("matches")
        if isinstance(matches, list) and matches:
            count = payload.get("count", len(matches))
            parts = [f"found {count} grep matches:"]
            for match in matches[:5]:
                if isinstance(match, dict):
                    fp = match.get("file_path", "")
                    content = str(match.get("content", ""))[:self._max_chars]
                    parts.append(f"  [{fp}] {content}")
            return "\n".join(parts)

        # File content: extract preview
        content_preview = payload.get("content_preview")
        if content_preview is not None:
            fp = payload.get("file_path", "unknown")
            found = payload.get("found", True)
            if not found:
                return f"file not found: {fp}"
            preview = str(content_preview)[:self._max_chars]
            return f"file [{fp}]: {preview}"

        # Log search results: extract hit count and samples
        hits = payload.get("hits")
        if isinstance(hits, list):
            count = payload.get("count", len(hits))
            parts = [f"found {count} log entries:"]
            for hit in hits[:3]:
                if isinstance(hit, dict):
                    msg = str(hit.get("message", hit.get("_source", {}).get("message", "")))[:self._max_chars]
                    ts = hit.get("timestamp", hit.get("@timestamp", ""))
                    parts.append(f"  [{ts}] {msg}")
            return "\n".join(parts)

        # Database/Mongo results: extract row/doc count and sample
        rows = payload.get("rows") or payload.get("documents")
        if isinstance(rows, list):
            count = payload.get("row_count", payload.get("document_count", len(rows)))
            db = payload.get("database", "")
            table = payload.get("collection", "")
            parts = [f"query returned {count} rows from {db}.{table}:" if table else f"query returned {count} rows from {db}:"]
            for row in rows[:2]:
                parts.append(f"  {str(row)[:self._max_chars]}")
            return "\n".join(parts)

        # Conclusion
        root_cause = payload.get("root_cause")
        if root_cause:
            confidence = payload.get("confidence", "?")
            return f"conclusion: {root_cause} (confidence={confidence})"

        return None

    @staticmethod
    def _to_scalar(value: Any) -> str:
        if isinstance(value, (int, float, bool, str)):
            return str(value)
        if value is None:
            return "null"
        if isinstance(value, dict):
            return f"dict(len={len(value)})"
        if isinstance(value, list):
            if value and len(value) <= 3:
                previews = [str(v)[:40] for v in value]
                return f"[{', '.join(previews)}]"
            return f"list(len={len(value)})"
        return type(value).__name__
