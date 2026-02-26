from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_core.config.agent_config import AgentConfig
from agent_core.context.token_budget import TokenBudget, estimate_tokens
from agent_core.state.agent_state import AgentState
from agent_core.state.memory_layers import MidTermMemory


class BuiltContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    payload: dict[str, object]
    token_count: int = Field(ge=0)
    trimmed: bool = False


class ContextBuilder:
    """Constructs bounded context in priority order with deterministic trimming."""

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._budget = TokenBudget(max_tokens=config.token_budget)

    def build(
        self,
        *,
        state: AgentState,
        memory: MidTermMemory,
        phase_rules: str,
        latest_tool_result_summary: str,
    ) -> BuiltContext:
        active_id = memory.stack_tree.active_node_id
        if active_id is None:
            raise ValueError("StackTree has no active node")
        active_node = memory.stack_tree.nodes[active_id]

        parent_summaries = memory.stack_tree.ancestry_summaries(
            active_id,
            cap=self._config.context_parent_summary_cap,
        )
        compressed_parent_summaries = [s[:120] for s in parent_summaries]

        top_hypotheses = [
            f"{h.id}:{h.status}:{h.confidence_score:.2f}:{h.description[:120]}"
            for h in memory.top_hypotheses(self._config.context_hypothesis_cap)
        ]

        evidence_nodes = memory.evidence_graph.top_relevant(self._config.context_evidence_cap)
        evidence_summaries = [
            f"{e.id}:{e.type}:{e.relevance_score:.2f}:{e.summary[:140]}"
            for e in evidence_nodes
        ]

        sections: list[tuple[str, object, int, int]] = [
            ("goal", state.goal, estimate_tokens(state.goal), 1),
            ("phase_rules", phase_rules, estimate_tokens(phase_rules), 2),
            (
                "active_branch",
                {
                    "id": active_node.id,
                    "objective": active_node.objective,
                    "summary": active_node.summary[:180],
                    "depth": active_node.depth,
                },
                estimate_tokens(active_node.objective + " " + active_node.summary),
                3,
            ),
            ("parent_summaries", compressed_parent_summaries, estimate_tokens(" ".join(compressed_parent_summaries)), 4),
            ("top_hypotheses", top_hypotheses, estimate_tokens(" ".join(top_hypotheses)), 5),
            ("top_evidence", evidence_summaries, estimate_tokens(" ".join(evidence_summaries)), 6),
            (
                "latest_tool_result_summary",
                latest_tool_result_summary[:220],
                estimate_tokens(latest_tool_result_summary),
                7,
            ),
        ]

        payload: dict[str, object] = {}
        total_tokens = 0
        trimmed = False

        for key, value, token_count, _priority in sorted(sections, key=lambda s: s[3]):
            if total_tokens + token_count <= self._budget.available_context_tokens:
                payload[key] = value
                total_tokens += token_count
                continue

            trimmed = True
            # Dynamic trimming for list-heavy sections keeps highest-priority items.
            if isinstance(value, list) and value:
                kept: list[str] = []
                for item in value:
                    item_tokens = estimate_tokens(item)
                    if total_tokens + item_tokens > self._budget.available_context_tokens:
                        break
                    kept.append(item)
                    total_tokens += item_tokens
                payload[key] = kept
            elif isinstance(value, str) and value:
                remaining = self._budget.available_context_tokens - total_tokens
                if remaining <= 0:
                    payload[key] = ""
                else:
                    by_words = value.split()
                    capped_words = max(1, int(remaining / 1.35))
                    clipped = " ".join(by_words[:capped_words])
                    payload[key] = clipped
                    total_tokens += estimate_tokens(clipped)
            else:
                payload[key] = value

        return BuiltContext(payload=payload, token_count=total_tokens, trimmed=trimmed)
