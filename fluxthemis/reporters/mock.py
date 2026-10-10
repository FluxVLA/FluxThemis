from __future__ import annotations

from typing import Any, Mapping

from fluxthemis.engines.utils import EVALUATION_REPORTERS

from .base import BaseEvaluationReporter


@EVALUATION_REPORTERS.register_module()
class MockEvaluationReporter(BaseEvaluationReporter):
    """In-memory lifecycle reporter used by smoke tests and unit tests."""

    def __init__(self,
                 run_session_id: str = 'mock-evaluation-session') -> None:
        if not isinstance(run_session_id, str) or not run_session_id:
            raise ValueError('run_session_id must be a non-empty string')
        self._run_session_id = run_session_id
        self.events: list[tuple[str, dict[str, Any]]] = []

    @property
    def run_session_id(self) -> str:
        return self._run_session_id

    def report_evaluation(self, event_type: str, payload: Mapping[str,
                                                                  Any]) -> str:
        self.events.append((event_type, dict(payload)))
        return ''
