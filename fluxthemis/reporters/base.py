from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseEvaluationReporter(ABC):
    """Publishes lifecycle events to the authoritative report service."""

    def setup(self) -> None:
        """Allocate network resources and start one reporting session."""

    @property
    @abstractmethod
    def run_session_id(self) -> str:
        """Opaque identifier for the current reporting session."""

    @property
    def evaluation_report_run_dir(self) -> str | None:
        """Authoritative artifact directory acknowledged by the service."""
        return None

    @abstractmethod
    def report_evaluation(self, event_type: str, payload: Mapping[str,
                                                                  Any]) -> str:
        """Report one lifecycle event and return its authoritative run dir."""

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""
