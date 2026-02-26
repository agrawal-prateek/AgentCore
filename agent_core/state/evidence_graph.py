from __future__ import annotations

from hashlib import sha256
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class EvidenceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: str = Field(pattern=r"^(log|code|metric|db|inference|summary)$")
    source_reference: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    raw_pointer: str = Field(min_length=1)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    created_iteration: int = Field(ge=0)

    def dedup_hash(self) -> str:
        normalized = "|".join(
            [self.type, self.source_reference.strip().lower(), self.summary.strip().lower()]
        )
        return sha256(normalized.encode("utf-8")).hexdigest()


class EvidenceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src_id: str
    dst_id: str
    relation: str = Field(pattern=r"^(supports|contradicts|derived_from|correlates_with)$")


class EvidenceGraph(BaseModel):
    """Bounded evidence graph with deduplication and deterministic pruning."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    max_nodes: int = Field(ge=1)
    nodes: dict[str, EvidenceNode] = Field(default_factory=dict)
    edges: list[EvidenceEdge] = Field(default_factory=list)
    _hash_index: dict[str, str] = PrivateAttr(default_factory=dict)

    def add_or_merge_node(self, node: EvidenceNode, *, current_iteration: int) -> str:
        node_hash = node.dedup_hash()
        existing_id = self._hash_index.get(node_hash)
        if existing_id is not None:
            existing = self.nodes[existing_id]
            existing.relevance_score = max(existing.relevance_score, node.relevance_score)
            existing.weight = min(1.0, round((existing.weight + node.weight) / 2 + 0.05, 4))
            existing.created_iteration = max(existing.created_iteration, current_iteration)
            return existing_id

        self.nodes[node.id] = node
        self._hash_index[node_hash] = node.id
        self._enforce_limits(current_iteration=current_iteration)
        return node.id

    def add_edge(self, edge: EvidenceEdge) -> None:
        if edge.src_id not in self.nodes or edge.dst_id not in self.nodes:
            raise ValueError("Both edge endpoints must exist in graph")
        if edge in self.edges:
            return
        self.edges.append(edge)

    def top_relevant(self, limit: int) -> list[EvidenceNode]:
        ordered = sorted(
            self.nodes.values(),
            key=lambda n: (n.relevance_score, n.weight, n.created_iteration),
            reverse=True,
        )
        return ordered[:limit]

    def prune_low_weight_stale(self, *, current_iteration: int, stale_after: int = 4) -> int:
        to_prune = [
            n
            for n in self.nodes.values()
            if (current_iteration - n.created_iteration >= stale_after)
        ]
        removed = 0
        for node in sorted(to_prune, key=lambda n: (n.weight, n.relevance_score, n.created_iteration)):
            if len(self.nodes) <= self.max_nodes:
                break
            self._remove_node(node.id)
            removed += 1
        return removed

    def _enforce_limits(self, *, current_iteration: int) -> None:
        if len(self.nodes) <= self.max_nodes:
            return
        self.prune_low_weight_stale(current_iteration=current_iteration, stale_after=0)
        while len(self.nodes) > self.max_nodes:
            candidate = min(self.nodes.values(), key=lambda n: (n.weight, n.relevance_score, n.created_iteration))
            self._remove_node(candidate.id)

    def _remove_node(self, node_id: str) -> None:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        node_hash = node.dedup_hash()
        if self._hash_index.get(node_hash) == node_id:
            del self._hash_index[node_hash]
        self.edges = [e for e in self.edges if e.src_id != node_id and e.dst_id != node_id]

    def reindex(self) -> None:
        self._hash_index = {node.dedup_hash(): node.id for node in self.nodes.values()}

    @property
    def count(self) -> int:
        return len(self.nodes)

    def node_ids(self) -> Iterable[str]:
        return self.nodes.keys()
