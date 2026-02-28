from __future__ import annotations

from typing import Any

import pytest

from agent_core.llm.llm_adapter import LLMAdapter, LLMRequest, LLMResponse, LLMTraceContext
from agent_core.planning.executor import LLMExecutor
from agent_core.planning.planner import LLMPlanner


class CaptureAdapter(LLMAdapter):
    def __init__(self, response_content: dict[str, Any], trace_id: str) -> None:
        self.response_content = response_content
        self.trace_id = trace_id
        self.last_request: LLMRequest | None = None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            content=self.response_content,
            prompt_tokens=12,
            completion_tokens=8,
            raw_text="{}",
            provider_payload={},
            trace_id=self.trace_id,
        )


@pytest.mark.asyncio
async def test_llm_planner_propagates_trace_context() -> None:
    adapter = CaptureAdapter(
        response_content={
            "next_objective": "inspect logs",
            "target_branch_id": "branch:1",
            "phase_transition": None,
            "reasoning_summary": "plan",
            "termination_flag": False,
            "spawn_children": [],
        },
        trace_id="trace-planner-1",
    )
    planner = LLMPlanner(adapter=adapter, model="model-a", system_prompt="sys")
    context = LLMTraceContext(
        investigation_id="inv-1",
        iteration=2,
        agent_id="agent-1",
        agent_role="orchestrator",
        task="planner",
    )
    await planner.plan({"goal": "debug"}, trace_context=context)

    assert adapter.last_request is not None
    assert adapter.last_request.trace_context == context
    assert planner.last_trace_id == "trace-planner-1"


@pytest.mark.asyncio
async def test_llm_executor_propagates_trace_context() -> None:
    adapter = CaptureAdapter(
        response_content={
            "tool_name": "view_file",
            "arguments": {"file_path": "app.py"},
            "expected_outcome": "inspect content",
        },
        trace_id="trace-executor-1",
    )
    executor = LLMExecutor(adapter=adapter, model="model-b", system_prompt="sys")
    context = LLMTraceContext(
        investigation_id="inv-1",
        iteration=3,
        agent_id="agent-2",
        agent_role="specialist",
        task="executor",
    )
    await executor.propose({"goal": "debug"}, "view file", trace_context=context)

    assert adapter.last_request is not None
    assert adapter.last_request.trace_context == context
    assert executor.last_trace_id == "trace-executor-1"
