from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    payload: dict[str, Any]


class StorageBackend(ABC):
    """Long-term storage abstraction."""

    @abstractmethod
    async def save_snapshot(self, snapshot: StateSnapshot) -> None:
        raise NotImplementedError

    @abstractmethod
    async def load_latest_snapshot(self, agent_id: str) -> StateSnapshot | None:
        raise NotImplementedError


class InMemoryStorageBackend(StorageBackend):
    def __init__(self) -> None:
        self._snapshots: dict[str, list[StateSnapshot]] = {}

    async def save_snapshot(self, snapshot: StateSnapshot) -> None:
        self._snapshots.setdefault(snapshot.agent_id, []).append(snapshot)

    async def load_latest_snapshot(self, agent_id: str) -> StateSnapshot | None:
        history = self._snapshots.get(agent_id, [])
        return history[-1] if history else None
