"""FluxVLA policy client used by simulator workers."""

from __future__ import annotations

import time
import uuid
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


@POLICY_CLIENTS.register_module()
class FluxVLAPolicyClient(BasePolicyClient):
    """Synchronous policy client backed by the FluxVLA model service."""

    def __init__(self, transport: Mapping[str, Any]) -> None:
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
        return ActionChunk(
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

    def close(self) -> None:
        self.transport.close()
        self._task = None
        self._episode_index = None
        self._episode_id = None
        self._policy_seed = None
        self._prediction_sequence = 0

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
        header = encode_header(
            'predict_action',
            deployment_id=self.transport.server_deployment_id,
            request_id=request_id,
            policy_seed=self._policy_seed,
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
        actions = np.asarray(body['actions'], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
            raise RuntimeError(
                f'FluxVLA actions must have shape [T, A], got {actions.shape}')
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


__all__ = ['FluxVLAPolicyClient']
