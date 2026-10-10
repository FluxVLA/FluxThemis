from __future__ import annotations

from collections import defaultdict
from statistics import fmean

from fluxthemis.engines.utils import EVALUATORS
from fluxthemis.structures import EpisodeResult

from .base import BaseEvaluator


@EVALUATORS.register_module()
class SuccessRateEvaluator(BaseEvaluator):
    """Aggregate overall and per-task success, return and episode length."""

    def __init__(self) -> None:
        self.results: list[EpisodeResult] = []

    def on_run_start(self, tasks) -> None:
        self.results.clear()

    def on_episode_end(self, result: EpisodeResult) -> None:
        self.results.append(result)

    def summarize(self) -> dict:
        grouped = defaultdict(list)
        for result in self.results:
            grouped[result.task_id].append(result)
        return {
            'overall': self._aggregate(self.results),
            'tasks': {
                task_id: self._aggregate(results)
                for task_id, results in sorted(grouped.items())
            },
        }

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
        metrics = {
            'episodes': count,
            'successes': successes,
            'success_rate': successes / count,
            'mean_return': fmean(result.episode_return for result in results),
            'mean_steps': fmean(result.steps for result in results),
            'mean_duration_s': fmean(result.duration_s for result in results),
        }
        scores = [
            result.score for result in results if result.score is not None
        ]
        if scores:
            metrics['scored_episodes'] = len(scores)
            metrics['mean_score'] = fmean(scores)
        return metrics
