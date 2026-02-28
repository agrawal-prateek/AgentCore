from __future__ import annotations

from abc import ABC, abstractmethod


class SimilarityPort(ABC):
    """Port for computing semantic similarity between two text strings.

    AgentCore defines the interface; domain applications provide implementations
    (e.g. using embedding models, rerankers, etc.).
    """

    @abstractmethod
    async def similarity(self, text_a: str, text_b: str) -> float:
        """Return similarity score in [0.0, 1.0] between two texts."""
        raise NotImplementedError
