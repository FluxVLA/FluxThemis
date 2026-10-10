from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from fluxthemis.structures import ActionChunk, TaskSpec


class BasePolicyClient(ABC):
    """Policy boundary used by simulator workers.

    Implementations must ensure that preprocessing, inference and action
    denormalization happen locally or in their remote policy service. Returned
    actions must already be in environment units.
    """

    def setup(self) -> None:
        """Allocate model or network resources."""

    def reset(self, task: TaskSpec, episode_index: int,
              policy_seed: int) -> None:
        """Reset episode-local model state."""

    @abstractmethod
    def set_episode_id(self, episode_id: str) -> None:
        """Set the exact id used by subsequent episode inference requests."""

    @abstractmethod
    def predict_action(self, observation: Any,
                       context: Mapping[str, Any]) -> ActionChunk:
        """Return float32 actions with shape ``[T, A]``."""

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""
