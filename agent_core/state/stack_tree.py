from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StackNodeStatus(str, Enum):
    OPEN = "open"
    EXHAUSTED = "exhausted"
    VALIDATED = "validated"
    ABANDONED = "abandoned"


class StackNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    parent_id: str | None = None
    depth: int = Field(ge=0)
    branch_score: float = Field(default=0.0)
    status: StackNodeStatus = Field(default=StackNodeStatus.OPEN)
    summary: str = Field(default="")
    created_at: datetime
    closed_at: datetime | None = None


class StackTree(BaseModel):
    """Bounded DFS stack tree with collapse controls."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    max_depth: int = Field(ge=1)
    nodes: dict[str, StackNode] = Field(default_factory=dict)
    root_id: str | None = None
    active_node_id: str | None = None

    def add_node(self, *, node_id: str, objective: str, parent_id: str | None, branch_score: float = 0.0) -> StackNode:
        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        if parent_id is None:
            depth = 0
            if self.root_id is not None:
                raise ValueError("Root node already exists")
        else:
            parent = self.nodes.get(parent_id)
            if parent is None:
                raise ValueError(f"Unknown parent_id '{parent_id}'")
            depth = parent.depth + 1

        if depth > self.max_depth:
            raise ValueError(f"Max stack depth exceeded: {depth} > {self.max_depth}")

        node = StackNode(
            id=node_id,
            objective=objective,
            parent_id=parent_id,
            depth=depth,
            branch_score=branch_score,
            created_at=datetime.now(tz=UTC),
        )
        self.nodes[node_id] = node
        if self.root_id is None:
            self.root_id = node_id
        self.active_node_id = node_id
        return node

    def set_active(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node_id '{node_id}'")
        self.active_node_id = node_id

    def close_node(self, node_id: str, status: StackNodeStatus, summary: str = "") -> None:
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Unknown node_id '{node_id}'")
        node.status = status
        node.summary = summary
        node.closed_at = datetime.now(tz=UTC)

    def collapse_branch(self, node_id: str, *, preserve_ancestor: bool = True) -> str | None:
        """Collapse a branch by abandoning descendants and returning chosen fallback node."""
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Unknown node_id '{node_id}'")

        descendants = self._descendants_of(node_id)
        for descendant_id in descendants:
            descendant = self.nodes[descendant_id]
            if descendant.status == StackNodeStatus.OPEN:
                descendant.status = StackNodeStatus.ABANDONED
                descendant.closed_at = datetime.now(tz=UTC)

        fallback = node.parent_id if preserve_ancestor else None
        if fallback is not None:
            self.active_node_id = fallback
            if self.nodes[fallback].status == StackNodeStatus.OPEN:
                return fallback

        open_nodes = [n for n in self.nodes.values() if n.status == StackNodeStatus.OPEN]
        if open_nodes:
            best = sorted(open_nodes, key=lambda n: (n.depth, -n.branch_score))[0]
            self.active_node_id = best.id
            return best.id

        self.active_node_id = self.root_id
        return self.active_node_id

    def ancestry_summaries(self, node_id: str, cap: int) -> list[str]:
        summaries: list[str] = []
        cursor = self.nodes.get(node_id)
        while cursor and cursor.parent_id is not None and len(summaries) < cap:
            parent = self.nodes[cursor.parent_id]
            if parent.summary:
                summaries.append(parent.summary[:180])
            cursor = parent
        return summaries

    def _descendants_of(self, node_id: str) -> list[str]:
        descendants: list[str] = []
        stack = [node_id]
        while stack:
            current = stack.pop()
            children = [n.id for n in self.nodes.values() if n.parent_id == current]
            descendants.extend(children)
            stack.extend(children)
        return [d for d in descendants if d != node_id]
