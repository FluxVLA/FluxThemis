from __future__ import annotations

import threading
import warnings
from abc import abstractmethod
from typing import Any, Sequence

import numpy as np

from fluxthemis.structures import StepResult, TaskSpec

from ..base import BaseEnvironment


class BaseRealRobotEnvironment(BaseEnvironment):
    """Safety-oriented base for robot adapters.

    Actuation stays disabled unless ``allow_actuation=True`` is explicitly
    present in the runtime config. Concrete adapters remain responsible for
    hardware-specific watchdogs and emergency-stop integration.
    """

    def __init__(self,
                 allow_actuation: bool = False,
                 action_low: Sequence[float] | None = None,
                 action_high: Sequence[float] | None = None,
                 require_action_bounds: bool = True) -> None:
        if not isinstance(allow_actuation, bool):
            raise TypeError('allow_actuation must be a boolean')
        if not isinstance(require_action_bounds, bool):
            raise TypeError('require_action_bounds must be a boolean')
        self.allow_actuation = allow_actuation
        self.require_action_bounds = require_action_bounds
        self.action_low = (None if action_low is None else np.asarray(
            action_low, dtype=np.float32))
        self.action_high = (None if action_high is None else np.asarray(
            action_high, dtype=np.float32))
        if ((self.action_low is None) != (self.action_high is None)):
            raise ValueError('action_low and action_high must be set together')
        if (self.action_low is not None
                and self.action_low.shape != self.action_high.shape):
            raise ValueError('action bounds must have identical shapes')
        if self.action_low is not None:
            if self.action_low.ndim != 1 or self.action_low.size == 0:
                raise ValueError('action bounds must be non-empty vectors')
            if (not np.isfinite(self.action_low).all()
                    or not np.isfinite(self.action_high).all()):
                raise ValueError('action bounds must be finite')
            if (self.action_low > self.action_high).any():
                raise ValueError('action_low cannot exceed action_high')
        self._emergency_stopped = False
        self._connected = False
        self._control_lock = threading.RLock()

    def setup(self) -> None:
        with self._control_lock:
            if not self.allow_actuation:
                raise RuntimeError(
                    'Real-robot actuation is disabled. Set '
                    'allow_actuation=True only after checking the robot '
                    'workspace and emergency stop.')
            if self.require_action_bounds and self.action_low is None:
                raise RuntimeError(
                    'Real-robot actuation requires explicit action_low and '
                    'action_high bounds. Set require_action_bounds=False only '
                    'when the concrete adapter enforces equivalent limits.')
            try:
                self.connect()
            except Exception:
                try:
                    self.disconnect()
                except Exception as cleanup_error:  # pragma: no cover
                    warnings.warn(
                        f'Robot disconnect after failed connect also failed: '
                        f'{cleanup_error}',
                        RuntimeWarning,
                        stacklevel=2,
                    )
                raise
            self._connected = True

    def step(self, action: Any) -> StepResult:
        with self._control_lock:
            if not self._connected:
                raise RuntimeError('Robot environment is not connected')
            if self._emergency_stopped:
                raise RuntimeError('Robot environment is emergency-stopped')
            action_array = np.asarray(action, dtype=np.float32)
            if action_array.ndim != 1:
                raise ValueError('Robot action must be one-dimensional, '
                                 f'got {action_array.shape}')
            if not np.isfinite(action_array).all():
                raise ValueError('Robot action contains NaN or infinity')
            if self.action_low is not None:
                if action_array.shape != self.action_low.shape:
                    raise ValueError(
                        f'Expected action shape {self.action_low.shape}, got '
                        f'{action_array.shape}')
                if ((action_array < self.action_low).any()
                        or (action_array > self.action_high).any()):
                    raise ValueError('Robot action exceeds configured bounds')
            return self.execute_action(action_array)

    def emergency_stop(self) -> None:
        with self._control_lock:
            self._emergency_stopped = True
            self.on_emergency_stop()

    def close(self) -> None:
        with self._control_lock:
            if self._connected:
                self.disconnect()
            self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """Connect to the robot and start hardware watchdogs."""

    @abstractmethod
    def execute_action(self, action: np.ndarray) -> StepResult:
        """Send one validated command and return the next observation."""

    @abstractmethod
    def on_emergency_stop(self) -> None:
        """Stop motion using the hardware-specific safe path."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release hardware resources."""

    @abstractmethod
    def tasks(self) -> Sequence[TaskSpec]:
        raise NotImplementedError
