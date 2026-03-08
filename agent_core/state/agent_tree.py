from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class AgentNodeStatus(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class AgentSpawnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: str = Field(min_length=1)
    role: str = Field(default="specialist", min_length=1)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    child_id: str | None = None


class AgentNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    parent_agent_id: str | None = None
    role: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    status: AgentNodeStatus = AgentNodeStatus.OPEN
    depth: int = Field(ge=0)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    created_seq: int = Field(ge=0)
    spawned_iteration: int = Field(ge=0)
    last_active_iteration: int = Field(default=0, ge=0)
    closed_iteration: int | None = None
    findings_summary: str = ""
    findings_confidence: float = 0.0


class AgentTree(BaseModel):
    """Bounded agent hierarchy for deterministic multi-agent scheduling."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    max_active_agents: int = Field(ge=1)
    max_total_agents: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    nodes: dict[str, AgentNode] = Field(default_factory=dict)
    root_agent_id: str | None = None
    _seq: int = PrivateAttr(default=0)

    def get_child_reports(self, parent_agent_id: str) -> list[AgentNode]:
        """Return closed children of the given parent that have findings."""
        return [
            node for node in self.nodes.values()
            if node.parent_agent_id == parent_agent_id
            and node.status in (AgentNodeStatus.COMPLETED, AgentNodeStatus.FAILED)
            and node.findings_summary
        ]

    def ensure_root(self, *, agent_id: str, objective: str, role: str = "orchestrator") -> AgentNode:
        existing = self.nodes.get(agent_id)
        if existing is not None:
            return existing
        if self.root_agent_id is not None:
            raise ValueError("AgentTree root already exists")
        node = AgentNode(
            id=agent_id,
            role=role,
            objective=objective,
            depth=0,
            priority=0.5,
            created_seq=self._next_seq(),
            spawned_iteration=0,
        )
        self.nodes[agent_id] = node
        self.root_agent_id = agent_id
        return node

    @property
    def open_count(self) -> int:
        return sum(1 for node in self.nodes.values() if node.status == AgentNodeStatus.OPEN)

    @property
    def total_count(self) -> int:
        return len(self.nodes)

    def spawn_child(
        self,
        *,
        parent_agent_id: str,
        request: AgentSpawnRequest,
        iteration: int,
    ) -> AgentNode | None:
        parent = self.nodes.get(parent_agent_id)
        if parent is None:
            raise ValueError(f"Unknown parent agent '{parent_agent_id}'")
        if parent.depth + 1 > self.max_depth:
            return None
        if self.total_count >= self.max_total_agents:
            return None
        if self.open_count >= self.max_active_agents:
            return None

        child_id = request.child_id or f"{parent_agent_id}:child:{self._seq + 1}"
        if child_id in self.nodes:
            return self.nodes[child_id]

        node = AgentNode(
            id=child_id,
            parent_agent_id=parent_agent_id,
            role=request.role,
            objective=request.objective,
            depth=parent.depth + 1,
            priority=request.priority,
            created_seq=self._next_seq(),
            spawned_iteration=iteration,
        )
        self.nodes[child_id] = node
        return node

    def select_next_agent(self, *, iteration: int) -> AgentNode:
        candidates = [node for node in self.nodes.values() if node.status == AgentNodeStatus.OPEN]
        if not candidates:
            raise ValueError("No open agents available")

        # Exclude parents that still have open children — they delegated
        # work and should wait until children report back.
        parents_with_open_children: set[str] = set()
        for node in candidates:
            if node.parent_agent_id is not None and node.parent_agent_id in self.nodes:
                parent = self.nodes[node.parent_agent_id]
                if parent.status == AgentNodeStatus.OPEN:
                    parents_with_open_children.add(parent.id)

        eligible = [n for n in candidates if n.id not in parents_with_open_children]
        if not eligible:
            # All open agents are parents waiting for children — shouldn't
            # happen in practice, but fall back to the full candidate list.
            eligible = candidates

        # Round-robin with priority tiebreaker:
        # 1. Least recently active first (ensures all agents get turns)
        # 2. Higher priority wins ties
        # 3. Deeper agents before shallower (children before parent)
        # 4. Earlier creation order, then ID for determinism
        ordered = sorted(
            eligible,
            key=lambda node: (node.last_active_iteration, -node.priority, -node.depth, node.created_seq, node.id),
        )
        selected = ordered[0]
        selected.last_active_iteration = iteration
        return selected

    def mark_closed(self, *, agent_id: str, status: AgentNodeStatus, iteration: int) -> None:
        node = self.nodes.get(agent_id)
        if node is None:
            raise ValueError(f"Unknown agent '{agent_id}'")
        node.status = status
        node.closed_iteration = iteration

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq
