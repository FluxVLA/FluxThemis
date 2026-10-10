from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fluxthemis.evaluators.base import BaseEvaluator
from fluxthemis.reporters.base import BaseEvaluationReporter
from fluxthemis.structures import EpisodeResult, TaskSpec

from .evaluation import EvaluationSettings, EvaluationStatus, utc_now
from .evaluation_events import (build_episode_end_event, build_run_end_event,
                                build_run_start_event, build_summary)


@dataclass
class _EvaluationState:
    status: EvaluationStatus = EvaluationStatus.CREATED
    selected_tasks: tuple[TaskSpec, ...] = ()
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float = 0.0
    completed_episodes: int = 0
    total_episodes: int = 0


class EvaluationRun:
    """Track metrics and reporting for one coordinator run."""

    def __init__(self, settings: EvaluationSettings, evaluator: BaseEvaluator,
                 reporter: BaseEvaluationReporter) -> None:
        self.settings = settings
        self._state = _EvaluationState()
        self.evaluator = evaluator
        self.reporter = reporter
        self._started_monotonic: float | None = None

    @property
    def run_session_id(self) -> str:
        return self.reporter.run_session_id

    @property
    def report_run_dir(self) -> str | None:
        return self.reporter.evaluation_report_run_dir

    def __enter__(self) -> EvaluationRun:
        self.reporter.setup()
        try:
            self.evaluator.setup()
        except Exception:
            self.reporter.close()
            raise
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        for component in (self.evaluator, self.reporter):
            try:
                component.close()
            except Exception:
                pass

    def start(self, selected_tasks: Sequence[TaskSpec],
              total_episodes: int) -> None:
        if self._state.status is not EvaluationStatus.CREATED:
            raise RuntimeError(f'cannot start evaluation while status is '
                               f'{self._state.status.value!r}')
        self._state.selected_tasks = tuple(selected_tasks)
        self._state.total_episodes = total_episodes
        self._state.started_at = utc_now()
        self._state.status = EvaluationStatus.RUNNING
        self._started_monotonic = time.perf_counter()
        self.evaluator.on_run_start(self._state.selected_tasks)
        self.reporter.report_evaluation(
            'run_start', build_run_start_event(self.settings, self._state))

    def on_episode_start(self, payload: Mapping[str, Any]) -> None:
        self._require_running()
        self.reporter.report_evaluation('episode_start', payload)

    def on_episode_end(self, result: EpisodeResult, episode_id: str) -> None:
        self._require_running()
        self.evaluator.on_episode_end(result)
        self._state.completed_episodes += 1
        self.reporter.report_evaluation(
            'episode_end',
            build_episode_end_event(result, self._state, episode_id),
        )

    def finish(self, last_result: EpisodeResult | None) -> dict[str, Any]:
        self._require_running()
        metrics = self.evaluator.summarize()
        self._stop(EvaluationStatus.FINISHED)
        summary = build_summary(
            self.settings,
            self._state,
            metrics,
            self.report_run_dir,
        )
        self.reporter.report_evaluation(
            'run_end',
            build_run_end_event(
                self.settings,
                self._state,
                metrics,
                last_result=last_result,
                summary=summary,
            ),
        )
        return summary

    def fail(self, status: EvaluationStatus,
             current_episode: Mapping[str, Any] | None,
             last_result: EpisodeResult | None, error: Mapping[str,
                                                               Any]) -> None:
        if self._state.status is not EvaluationStatus.RUNNING:
            return
        self._stop(status)
        metrics, metrics_error = self._summarize_safely()
        try:
            self.reporter.report_evaluation(
                'run_end',
                build_run_end_event(
                    self.settings,
                    self._state,
                    metrics,
                    current_episode=current_episode,
                    last_result=last_result,
                    error=error,
                    metrics_error=metrics_error,
                ),
            )
        except Exception as report_exc:
            warnings.warn(
                'Could not report terminal evaluation state to FluxVLA: '
                f'{report_exc}',
                RuntimeWarning,
                stacklevel=2,
            )

    def _stop(self, status: EvaluationStatus) -> None:
        self._state.status = status
        self._state.finished_at = utc_now()
        if self._started_monotonic is not None:
            self._state.duration_s = max(
                0.0,
                time.perf_counter() - self._started_monotonic)

    def _summarize_safely(
            self) -> tuple[dict[str, Any], dict[str, str]
                           | None]:
        try:
            return self.evaluator.summarize(), None
        except Exception as exc:
            return {}, {
                'error_type': type(exc).__name__,
                'message': str(exc),
            }

    def _require_running(self) -> None:
        if self._state.status is not EvaluationStatus.RUNNING:
            raise RuntimeError(
                f'evaluation is not running: {self._state.status.value!r}')
