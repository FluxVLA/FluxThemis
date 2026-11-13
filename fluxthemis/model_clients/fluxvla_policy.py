"""FluxVLA policy client used by simulator workers."""

from __future__ import annotations

import time
import uuid
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from fluxthemis.engines.utils import POLICY_CLIENTS
from fluxthemis.structures import ActionChunk, TaskSpec
from fluxthemis.transports.fluxvla_zmq import (CONNECTION_TRANSPORT_FIELDS,
                                               FluxVLAZMQTransport,
                                               connection_transport,
                                               nonempty_string, string_value)
from fluxthemis.transports.zmq_protocol import (decode_frame, encode_frame,
                                                encode_header)

from .base import BasePolicyClient

_POLICY_TRANSPORT_FIELDS = CONNECTION_TRANSPORT_FIELDS | frozenset({
    'image_keys',
    'state_keys',
    'unnorm_key',
})


def _select_history_steps(available, current_step, num_frames, stride):
    """Anchor, then the closest earlier observation at each requested step."""
    steps = sorted(set(available) | {0})
    requested = [
        current_step - stride * j for j in reversed(range(num_frames))
    ]
    return [0] + [
        steps[max(0,
                  bisect_right(steps, max(0, t)) - 1)] for t in requested
    ]


@POLICY_CLIENTS.register_module()
class FluxVLAPolicyClient(BasePolicyClient):
    """Synchronous policy client backed by the FluxVLA model service.

    ``history=None`` sends only the current observation. A history mapping
    enables an episode anchor and ``num_frames`` images spaced ``stride``
    simulator steps apart. With history enabled, ``replan_steps`` must match
    the number of actions executed between predictions.
    """

    def __init__(self,
                 transport: Mapping[str, Any],
                 *,
                 history: Mapping[str, Any] | None = None,
                 replan_steps: int = 24) -> None:
        if not isinstance(transport, Mapping):
            raise TypeError('transport must be a mapping')
        values = dict(transport)
        unknown_fields = sorted(set(values) - _POLICY_TRANSPORT_FIELDS)
        if unknown_fields:
            raise ValueError(f'Unknown policy transport fields: '
                             f'{unknown_fields}')
        if 'image_keys' not in values:
            raise ValueError('transport.image_keys is required')
        if 'state_keys' not in values:
            raise ValueError('transport.state_keys is required')

        self.transport = FluxVLAZMQTransport(connection_transport(values))
        self.image_keys = self._normalize_keys(values['image_keys'],
                                               'transport.image_keys')
        self.state_keys = self._normalize_keys(values['state_keys'],
                                               'transport.state_keys')
        overlapping_keys = sorted(set(self.image_keys) & set(self.state_keys))
        if overlapping_keys:
            raise ValueError(
                f'image_keys and state_keys overlap: {overlapping_keys}')
        self.unnorm_key = string_value(
            values.get('unnorm_key', ''), 'transport.unnorm_key')

        self._task: TaskSpec | None = None
        self._episode_index: int | None = None
        self._episode_id: str | None = None
        self._policy_seed: int | None = None
        self._prediction_sequence = 0
        self.last_profile: dict[str, float] = {}
        self._history_enabled = history is not None
        if self._history_enabled:
            if not isinstance(history, Mapping):
                raise TypeError('history must be a mapping or None')
            history = dict(history)
            self.num_history_frames = int(history.pop('num_frames', 8))
            self.history_stride = int(history.pop('stride', 25))
            include_anchor = history.pop('include_anchor', True)
            if history or not include_anchor:
                raise ValueError(
                    'History requires an anchor and num_frames/stride only')
            self.replan_steps = int(replan_steps)
            if min(self.num_history_frames, self.history_stride,
                   self.replan_steps) <= 0:
                raise ValueError(
                    'History and replan dimensions must be positive')
        self._clear_history()

    def _clear_history(self) -> None:
        self._frames = OrderedDict()
        self._anchor = None
        self._step = 0
        self._replan_index = 0

    def setup(self) -> None:
        self.transport.setup()

    def reset(self, task: TaskSpec, episode_index: int,
              policy_seed: int) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError('task must be a TaskSpec')
        if isinstance(episode_index,
                      bool) or not isinstance(episode_index, Integral):
            raise TypeError('episode_index must be an integer')
        if episode_index < 0:
            raise ValueError('episode_index must be non-negative')
        if isinstance(policy_seed,
                      bool) or not isinstance(policy_seed, Integral):
            raise TypeError('policy_seed must be an integer')
        self._task = task
        self._episode_index = int(episode_index)
        self._episode_id = f'{task.task_id}:{int(episode_index)}'
        self._policy_seed = int(policy_seed)
        self._prediction_sequence = 0
        self._clear_history()

    def _snapshot(self, observation: Mapping[str,
                                             Any]) -> dict[str, np.ndarray]:
        return {
            key: self._validate_image(observation[key], key).copy()
            for key in self.image_keys
        }

    def observe(self, observation: Any) -> None:
        if not self._history_enabled:
            return
        if self._anchor is None:
            raise RuntimeError(
                'The reset observation must be predicted before observe()')
        frame = self._snapshot(observation)
        self._step += 1
        self._frames[self._step] = frame
        cutoff = self._step - (self.num_history_frames -
                               1) * self.history_stride
        while self._frames and next(iter(self._frames)) < cutoff:
            self._frames.popitem(last=False)

    def set_episode_id(self, episode_id: str) -> None:
        episode_id = nonempty_string(episode_id, 'episode_id')
        if self._task is None or self._episode_index is None:
            raise RuntimeError(
                'reset() must be called before set_episode_id()')
        self._episode_id = episode_id
        self._prediction_sequence = 0

    def predict_action(self, observation: Any,
                       context: Mapping[str, Any]) -> ActionChunk:
        if not isinstance(observation, Mapping):
            raise TypeError('FluxVLA ZMQ observations must be mappings')
        if not isinstance(context, Mapping):
            raise TypeError('FluxVLA ZMQ context must be a mapping')
        self._ensure_ready()
        if self._history_enabled:
            if self._step != self.replan_steps * self._replan_index:
                raise RuntimeError(
                    f'History alignment mismatch: step={self._step}, '
                    f'replan={self._replan_index}, horizon={self.replan_steps}'
                )
            if self._anchor is None:
                self._anchor = self._snapshot(observation)

        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        self._prediction_sequence += 1
        header, payload = self._build_request(request_id, observation, context)
        response_header, response_payload = self.transport.request(
            header, payload)
        roundtrip_s = time.perf_counter() - started
        actions, server_infer_s = self._validate_response(
            response_header, response_payload, request_id)

        self.last_profile = {
            'roundtrip_s': roundtrip_s,
            'server_infer_s': server_infer_s,
            'network_and_queue_s': max(roundtrip_s - server_infer_s, 0.0),
        }
        result = ActionChunk(
            actions=actions,
            metadata={
                'profile': dict(self.last_profile),
                'request_id': request_id,
                'episode_id': self._episode_id,
                'sequence': self._prediction_sequence,
                'policy_seed': self._policy_seed,
                'denormalized': True,
            },
        )
        if self._history_enabled:
            self._replan_index += 1
        return result

    def close(self) -> None:
        try:
            self.transport.close()
            self._task = None
            self._episode_index = None
            self._episode_id = None
            self._policy_seed = None
            self._prediction_sequence = 0
        finally:
            self._clear_history()

    def _ensure_ready(self) -> None:
        if any(
                value is None for value in (self._task, self._episode_id,
                                            self._policy_seed)):
            raise RuntimeError(
                'FluxVLAPolicyClient.reset() must be called before '
                'predict_action()')

    def _build_request(self, request_id: str, observation: Mapping[str, Any],
                       context: Mapping[str, Any]) -> tuple[bytes, bytes]:
        images = {}
        for key in self.image_keys:
            if key not in observation:
                raise KeyError(f'Observation is missing image key {key!r}')
            images[key] = self._validate_image(observation[key], key)
        state = {}
        for key in self.state_keys:
            if key not in observation:
                raise KeyError(f'Observation is missing state key {key!r}')
            state[key] = self._validate_state(observation[key], key)
        prompt = self._resolve_prompt(context)
        unnorm_key = string_value(
            context.get('unnorm_key', self.unnorm_key), 'context.unnorm_key')
        payload = dict(images)
        payload.update(state)
        payload['task_description'] = prompt
        if self._history_enabled:
            steps = _select_history_steps(self._frames, self._step,
                                          self.num_history_frames,
                                          self.history_stride)
            for key in self.image_keys:
                payload[key + '_history'] = np.stack([
                    (self._anchor if step == 0 else self._frames[step])[key]
                    for step in steps
                ])
            payload['history_steps'] = np.asarray(steps, dtype=np.int64)
            payload['replan_index'] = self._replan_index
        header = encode_header(
            'predict_action',
            deployment_id=self.transport.server_deployment_id,
            request_id=request_id,
            policy_seed=self._policy_seed,
            episode_id=self._episode_id,
            sequence=self._prediction_sequence,
            deadline_ms=int(self.transport.request_timeout_s * 1000),
            unnorm_key=unnorm_key,
        )
        return header, encode_frame(payload)

    def _validate_response(self, header: Mapping[str, Any], payload: bytes,
                           request_id: str) -> tuple[np.ndarray, float]:
        self.transport.validate_response(header, request_id)
        if header.get('type') != 'predict_response':
            raise RuntimeError('FluxVLA returned an invalid prediction type')
        body = decode_frame(payload)
        if not isinstance(body, Mapping) or body.get('actions') is None:
            raise RuntimeError('FluxVLA predict response missing actions')
        _validate_action_type(body['actions'])
        actions = np.asarray(body['actions'])
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
            raise RuntimeError(
                f'FluxVLA actions must have shape [T, A], got {actions.shape}')
        if not np.isfinite(actions).all():
            raise RuntimeError('FluxVLA returned NaN or infinite actions')
        with np.errstate(over='ignore'):
            actions = actions.astype(np.float32, copy=False)
        if not np.isfinite(actions).all():
            raise RuntimeError('FluxVLA actions are outside the FP32 range')
        return actions, float(header.get('inference_time_s', 0.0))

    @staticmethod
    def _resolve_prompt(context: Mapping[str, Any]) -> str:
        prompt = context.get('task_description', '')
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError(
                'Missing task_description in policy client context')
        return prompt

    @staticmethod
    def _normalize_keys(value: Any, name: str) -> tuple[str, ...]:
        if isinstance(value,
                      (str, bytes)) or not isinstance(value, (list, tuple)):
            raise TypeError(f'{name} must be a sequence of key names')
        keys = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise TypeError(f'{name} entries must be non-empty strings')
            keys.append(item.strip())
        keys = tuple(keys)
        if not keys:
            raise ValueError(f'{name} cannot be empty')
        if len(set(keys)) != len(keys):
            raise ValueError(f'{name} cannot contain duplicates')
        return keys

    @staticmethod
    def _validate_image(value: Any, name: str) -> np.ndarray:
        image = np.asarray(value)
        if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8):
            raise TypeError(
                f'Observation image {name!r} must be uint8 [H, W, 3], got '
                f'{image.dtype} {image.shape}')
        return image

    @staticmethod
    def _validate_state(value: Any, name: str) -> np.ndarray:
        state = np.asarray(value, dtype=np.float64)
        if state.ndim != 1 or state.size == 0:
            raise ValueError(
                f'Observation state {name!r} must be a nonempty 1-D array')
        if not np.isfinite(state).all():
            raise ValueError(
                f'Observation state {name!r} contains NaN or infinity')
        return state


def _validate_action_type(value: Any) -> None:
    # A mixed Python sequence can lose its bool elements during coercion.
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_action_type(item)
    elif np.asarray(value).dtype.kind not in 'fiu':
        raise TypeError('FluxVLA actions must contain real numbers')


__all__ = ['FluxVLAPolicyClient']
