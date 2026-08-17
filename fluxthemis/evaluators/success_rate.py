from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean

from fluxthemis.engines.utils import EVALUATORS
from fluxthemis.structures import EpisodeResult, TaskSpec

from .base import BaseEvaluator


@EVALUATORS.register_module()
class SuccessRateEvaluator(BaseEvaluator):
    """Aggregate overall and per-task success, return and episode length."""

    supports_parallel_rollouts = True

    def __init__(self,
                 group_by_metadata: Sequence[str] | str | None = None) -> None:
        self.group_by_metadata = self._normalize_group_fields(
            group_by_metadata)
        self.results: list[EpisodeResult] = []
        self._task_metadata = {}

    def on_run_start(self, tasks: Sequence[TaskSpec]) -> None:
        self.results.clear()
        self._task_metadata = {
            task.task_id: dict(task.metadata)
            for task in tasks
        }
        if self.group_by_metadata:
            known_fields = {
                str(key)
                for task in tasks
                for key in task.metadata
            }
            unknown_fields = sorted(set(self.group_by_metadata) - known_fields)
            if unknown_fields:
                raise ValueError(
                    f'Unknown group_by_metadata fields {unknown_fields}; '
                    f'available fields: {sorted(known_fields)}')

    def on_episode_end(self, result: EpisodeResult) -> None:
        self.results.append(result)

    def summarize(self) -> dict:
        grouped = defaultdict(list)
        for result in self.results:
            grouped[result.task_id].append(result)
        summary = {
            'overall': self._aggregate(self.results),
            'tasks': {
                task_id: self._aggregate(results)
                for task_id, results in sorted(grouped.items())
            },
        }
        if self.group_by_metadata:
            summary['groups'] = {
                field: self._group_by_field(field)
                for field in self.group_by_metadata
            }
        return summary

    def _group_by_field(self, field: str) -> dict[str, dict]:
        grouped = defaultdict(list)
        for result in self.results:
            value = self._task_metadata[result.task_id].get(field)
            grouped[self._group_label(value)].append(result)
        return {
            label: self._aggregate(results)
            for label, results in sorted(grouped.items())
        }

    @staticmethod
    def _group_label(value) -> str:
        if value is None:
            return 'null'
        if isinstance(value, bool):
            return 'true' if value else 'false'
        return str(value)

    @staticmethod
    def _normalize_group_fields(
            fields: Sequence[str] | str | None) -> tuple[str, ...]:
        if fields is None:
            return ()
        if isinstance(fields, str):
            fields = (fields, )
        elif not isinstance(fields, Sequence):
            raise TypeError('group_by_metadata must be a sequence of strings')
        normalized = tuple(fields)
        if not normalized:
            raise ValueError('group_by_metadata cannot be empty')
        if any(not isinstance(field, str) or not field
               for field in normalized):
            raise ValueError(
                'group_by_metadata fields must be non-empty strings')
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                f'group_by_metadata contains duplicates: {normalized}')
        return normalized

    @staticmethod
    def _aggregate(results: list[EpisodeResult]) -> dict:
        count = len(results)
        successes = sum(result.success for result in results)
        if count == 0:
            return {
                'episodes': 0,
                'successes': 0,
                'success_rate': 0.0,
                'mean_return': 0.0,
                'mean_steps': 0.0,
                'mean_duration_s': 0.0,
            }
        return {
            'episodes': count,
            'successes': successes,
            'success_rate': successes / count,
            'mean_return': fmean(result.episode_return for result in results),
            'mean_steps': fmean(result.steps for result in results),
            'mean_duration_s': fmean(result.duration_s for result in results),
        }
