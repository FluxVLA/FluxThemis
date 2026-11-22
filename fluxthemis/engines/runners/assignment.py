"""Worker assignments and environment execution capabilities."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

from fluxthemis.structures import TaskSpec, positive_integer

if TYPE_CHECKING:
    from fluxthemis.environments.base import BaseEnvironment
    from fluxthemis.model_clients.base import BasePolicyClient

    from .episode import EpisodeJob


@dataclass(frozen=True)
class EpisodeAssignment:
    """An ordered list of distinct episodes assigned to one worker."""

    jobs: list[EpisodeJob]
    assignment_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'jobs', list(self.jobs))
        if not self.jobs:
            raise ValueError('An assignment must contain at least one episode')
        if len({job.episode_id for job in self.jobs}) != len(self.jobs):
            raise ValueError('Assignment contains duplicate episode IDs')
        if not isinstance(self.assignment_id, str) or not self.assignment_id:
            raise ValueError('Assignment ID must be a nonempty string')


@dataclass(frozen=True)
class ExecutionPolicy:
    """Validate capacity and the capabilities declared by an environment."""

    capacity: int = 1
    requires_single_task: bool = False
    task_capacities: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive_integer(self.capacity, 'capacity')
        if type(self.requires_single_task) is not bool:
            raise TypeError('requires_single_task must be a boolean')
        task_capacities = dict(self.task_capacities)
        for capacity in task_capacities.values():
            positive_integer(capacity, 'task capacity')
            if capacity > self.capacity:
                raise ValueError('Task capacity exceeds worker capacity')
        object.__setattr__(
            self, 'task_capacities', {
                task_id: capacity
                for task_id, capacity in task_capacities.items()
                if capacity < self.capacity
            })

    @classmethod
    def from_environment(cls, environment: BaseEnvironment) -> ExecutionPolicy:
        """Read capabilities after the environment is set up."""
        task_capacities = {}
        for value in environment.tasks():
            task = TaskSpec.from_value(value)
            task_capacities[task.task_id] = environment.task_capacity(task)
        return cls(environment.num_envs, environment.requires_single_task,
                   task_capacities)

    def validate_capacity(self, capacity: int,
                          task_capacities: Mapping[str, int]) -> None:
        if type(capacity) is not int or capacity != self.capacity:
            raise ValueError('Environment capacity changed after restart')
        if task_capacities != self.task_capacities:
            raise ValueError(
                'Environment task capacities changed after restart')

    def task_capacity(self, task: TaskSpec) -> int:
        """Read the environment's capacity for a task, or its default."""
        return self.task_capacities.get(task.task_id, self.capacity)

    def validate_assignment(
            self,
            assignment: EpisodeAssignment,
            policy_clients: Sequence[BasePolicyClient] | None = None) -> None:
        if len(assignment.jobs) > self.capacity:
            raise ValueError('Assignment exceeds worker capacity')
        if self.requires_single_task and any(
                job.task != assignment.jobs[0].task
                for job in assignment.jobs):
            raise ValueError('Environment requires one task per assignment')
        if any(
                len(assignment.jobs) > self.task_capacity(job.task)
                for job in assignment.jobs):
            raise ValueError('Assignment exceeds task slot limit')
        if policy_clients is not None:
            if len(policy_clients) != self.capacity:
                raise ValueError('Environment and policy client sizes differ')
            if len({id(client)
                    for client in policy_clients}) != len(policy_clients):
                raise ValueError(
                    'Each environment slot needs its own policy client')
