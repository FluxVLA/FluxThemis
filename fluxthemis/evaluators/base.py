from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from fluxthemis.structures import EpisodeResult, TaskSpec


class BaseEvaluator(ABC):
    """Lifecycle hooks for metrics and evaluation artifacts."""

    def setup(self) -> None:
        """Allocate evaluator resources."""

    def on_run_start(self, tasks: Sequence[TaskSpec]) -> None:
        """Reset run-level metric state."""

    @abstractmethod
    def on_episode_end(self, result: EpisodeResult) -> None:
        """Consume a completed episode."""

    @abstractmethod
    def summarize(self) -> dict:
        """Return a JSON-serializable run summary."""

    def close(self) -> None:
        """Release evaluator resources."""
