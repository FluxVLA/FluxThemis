from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class TaskSpec:
    """A task exposed by an evaluation environment."""

    task_id: str
    description: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task_id = str(self.task_id)
        if not task_id:
            raise ValueError('task_id cannot be empty')
        if not isinstance(self.metadata, Mapping):
            raise TypeError('task metadata must be a mapping')
        self.task_id = task_id
        self.description = str(self.description)
        self.metadata = dict(self.metadata)

    @classmethod
    def from_value(cls, value: Any) -> 'TaskSpec':
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(task_id=value)
        if isinstance(value, Mapping):
            data = dict(value)
            if 'id' in data and 'task_id' not in data:
                data['task_id'] = data.pop('id')
            return cls(**data)
        raise TypeError(f'Cannot create TaskSpec from {type(value).__name__}')


@dataclass
class StepResult:
    """One environment transition with explicit termination semantics."""

    observation: Any
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    success: bool | None = None
    info: Mapping[str, Any] = field(default_factory=dict)
    score: float | None = None


@dataclass
class ActionChunk:
    """Environment-unit actions with canonical shape ``[T, A]``."""

    actions: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Serializable result of one task episode."""

    task_id: str
    episode_index: int
    policy_seed: int
    environment_seed: int | None
    environment_case_index: int | None
    success: bool
    episode_return: float
    steps: int
    duration_s: float
    termination_reason: str
    model_calls: int
    info: Mapping[str, Any] = field(default_factory=dict)
    prediction_metadata: list[Mapping[str, Any]] = field(default_factory=list)
    score: float | None = None
