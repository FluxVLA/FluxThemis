"""Worker assignments and the supported execution/recovery policy."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .episode import EpisodeJob


@dataclass(frozen=True)
class EpisodeAssignment:
    """An ordered group of distinct episodes assigned to one worker."""

    jobs: tuple[EpisodeJob, ...]
    assignment_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError('An assignment must contain at least one episode')
        if len({job.episode_id for job in self.jobs}) != len(self.jobs):
            raise ValueError('Assignment contains duplicate episode IDs')
        if not isinstance(self.assignment_id, str) or not self.assignment_id:
            raise ValueError('Assignment ID must be a nonempty string')


@dataclass(frozen=True)
class ExecutionPolicy:
    """Capacity and execution restrictions shared by workers and scheduling."""

    batch_size: int = 1

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('episode_batch_size must be a positive integer')

    def validate_options(self,
                         *,
                         save_videos: bool = False,
                         strict_worker_assignment: bool = False) -> None:
        if self.batch_size != 1:
            if save_videos:
                raise ValueError('Batch execution does not yet support video')
            if strict_worker_assignment:
                raise ValueError(
                    'Batch execution requires dynamic scheduling; '
                    'fixed model-worker pairing is scalar-only')

    def validate_assignment(self, assignment: EpisodeAssignment) -> None:
        if len(assignment.jobs) > self.batch_size:
            raise ValueError('Assignment exceeds worker capacity')
        if any(job.task != assignment.jobs[0].task for job in assignment.jobs):
            raise ValueError(
                'This execution mode requires one task per assignment')
