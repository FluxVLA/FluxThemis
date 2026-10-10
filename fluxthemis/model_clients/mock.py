from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fluxthemis.engines.utils import POLICY_CLIENTS
from fluxthemis.structures import ActionChunk

from .base import BasePolicyClient


@POLICY_CLIENTS.register_module()
class MockPolicyClient(BasePolicyClient):
    """Deterministic action source used by the smoke configuration."""

    def __init__(self,
                 action_dim: int = 7,
                 chunk_size: int = 4,
                 fill_value: float = 0.0) -> None:
        if action_dim < 1 or chunk_size < 1:
            raise ValueError('action_dim and chunk_size must be positive')
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.fill_value = float(fill_value)
        self.call_count = 0
        self.episode_id = None

    def reset(self, task, episode_index: int, policy_seed: int) -> None:
        self.call_count = 0
        self.episode_id = None

    def set_episode_id(self, episode_id: str) -> None:
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('episode_id must be a non-empty string')
        self.episode_id = episode_id

    def predict_action(self, observation: Any,
                       context: Mapping[str, Any]) -> ActionChunk:
        self.call_count += 1
        actions = np.full(
            (self.chunk_size, self.action_dim),
            self.fill_value,
            dtype=np.float32,
        )
        metadata = {'mock': True}
        if self.episode_id is not None:
            metadata['episode_id'] = self.episode_id
        return ActionChunk(actions=actions, metadata=metadata)
