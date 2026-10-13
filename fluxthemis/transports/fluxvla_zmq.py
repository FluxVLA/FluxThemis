"""Shared synchronous ZMQ transport for FluxVLA service clients."""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Mapping
from typing import Any

from .zmq_protocol import (decode_frame, decode_header, encode_frame,
                           encode_header)

DEFAULT_ENDPOINT = 'tcp://127.0.0.1:15555'
DEFAULT_REQUEST_TIMEOUT_S = 120.0
CONNECTION_TRANSPORT_FIELDS = frozenset({
    'endpoint',
    'connect_timeout_s',
    'request_timeout_s',
})
_SERVER_PROBE_TIMEOUT_S = 1.0
_SERVER_WAIT_LOG_INTERVAL_S = 10.0


class FluxVLAZMQTransport:
    """Own one REQ socket and FluxVLA connection state."""

    def __init__(self, transport: Mapping[str, Any]) -> None:
        if not isinstance(transport, Mapping):
            raise TypeError('transport must be a mapping')
        values = dict(transport)
        unknown_fields = sorted(set(values) - CONNECTION_TRANSPORT_FIELDS)
        if unknown_fields:
            raise ValueError(
                f'Unknown ZMQ connection fields: {unknown_fields}')

        self.endpoint = endpoint_value(
            values.get('endpoint', DEFAULT_ENDPOINT), 'transport.endpoint')
        self.connect_timeout_s = optional_positive_number(
            values.get('connect_timeout_s'), 'transport.connect_timeout_s')
        self.request_timeout_s = positive_number(
            values.get('request_timeout_s', DEFAULT_REQUEST_TIMEOUT_S),
            'transport.request_timeout_s')

        self._context = None
        self._socket = None
        self._server_deployment_id: str | None = None

    @property
    def server_deployment_id(self) -> str:
        if self._server_deployment_id is None:
            raise RuntimeError(
                'FluxVLA ZMQ server deployment has not been resolved')
        return self._server_deployment_id

    def setup(self, *, require_evaluation_reporting: bool = False) -> None:
        if self._socket is not None:
            return
        import zmq
        self._context = zmq.Context()
        self._socket = self._connect_socket(zmq)
        try:
            self._wait_for_server_ready(require_evaluation_reporting)
        except BaseException:
            self.close()
            raise

    def request(
            self,
            header: bytes,
            payload: bytes,
            timeout_s: float | None = None) -> tuple[dict[str, Any], bytes]:
        import zmq
        self._require_ready()
        timeout_s = self.request_timeout_s if timeout_s is None else timeout_s
        socket = self._socket
        socket.setsockopt(zmq.RCVTIMEO, max(1, int(timeout_s * 1000)))
        try:
            socket.send_multipart([header, payload])
            frames = socket.recv_multipart()
            if len(frames) != 2:
                raise RuntimeError(
                    f'FluxVLA returned {len(frames)} frames; expected 2')
            return decode_header(frames[0]), frames[1]
        except zmq.Again as exc:
            if self._socket is socket and self._context is not None:
                socket.close(linger=0)
                self._socket = self._connect_socket(zmq)
            raise TimeoutError(
                f'FluxVLA ZMQ server {self.endpoint} did not respond within '
                f'{timeout_s:g}s') from exc

    @staticmethod
    def validate_response(header: Mapping[str, Any], request_id: str) -> None:
        if str(header.get('request_id', '')) != request_id:
            raise RuntimeError('FluxVLA ZMQ request_id mismatch')
        if not bool(header.get('ok', False)):
            error_type = str(header.get('error_type') or 'server_error')
            message = str(header.get('error') or 'unknown error')
            raise RuntimeError(f'FluxVLA {error_type}: {message}')

    def close(self) -> None:
        socket = self._socket
        context = self._context
        self._socket = None
        self._context = None
        self._server_deployment_id = None
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()

    def _connect_socket(self, zmq: Any) -> Any:
        if self.endpoint is None:
            raise RuntimeError('FluxVLA endpoint has not been resolved')
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        return socket

    def _require_ready(self) -> None:
        if self._socket is None or self._context is None:
            raise RuntimeError('FluxVLAZMQTransport.setup() was not called')

    def _wait_for_server_ready(self,
                               require_evaluation_reporting: bool) -> None:
        started = time.monotonic()
        next_log_at = 0.0
        while True:
            if (self.connect_timeout_s is not None
                    and time.monotonic() - started >= self.connect_timeout_s):
                raise TimeoutError(
                    f'FluxVLA server at {self.endpoint} did not become ready '
                    f'within {self.connect_timeout_s:g}s')
            request_id = uuid.uuid4().hex
            try:
                response_header, response_payload = self.request(
                    encode_header('status', request_id=request_id),
                    encode_frame({}), _SERVER_PROBE_TIMEOUT_S)
                self.validate_response(response_header, request_id)
                status = decode_frame(response_payload)
                if not isinstance(status, Mapping):
                    raise RuntimeError('FluxVLA status response is invalid')
                server_deployment_id = nonempty_string(
                    status.get('deployment_id'), 'status.deployment_id')
                if (self._server_deployment_id is not None and
                        server_deployment_id != self._server_deployment_id):
                    raise RuntimeError('FluxVLA server deployment changed '
                                       'while the client was connected')
                self._server_deployment_id = server_deployment_id
                if status.get('server_mode') != 'stateless':
                    raise RuntimeError('FluxVLA server is not stateless')
                if (require_evaluation_reporting and
                        not bool(status.get('evaluation_reporting', False))):
                    raise RuntimeError(
                        'FluxVLA server does not enable evaluation reporting')
                if status.get('status') == 'ready':
                    return
            except TimeoutError:
                pass
            now = time.monotonic()
            if now >= next_log_at:
                print(
                    '[FluxThemis] waiting for FluxVLA ZMQ server at '
                    f'{self.endpoint}...',
                    file=sys.stderr,
                    flush=True,
                )
                next_log_at = now + _SERVER_WAIT_LOG_INTERVAL_S
            time.sleep(0.2)


def connection_transport(values: Mapping[str, Any]) -> dict[str, Any]:
    """Select connection-only fields from a combined transport mapping."""
    return {
        key: values[key]
        for key in CONNECTION_TRANSPORT_FIELDS if key in values
    }


def nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def string_value(value: Any, name: str) -> str:
    if value is None:
        return ''
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string')
    return value


def positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be a positive number')
    value = float(value)
    if value <= 0:
        raise ValueError(f'{name} must be a positive number')
    return value


def optional_positive_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return positive_number(value, name)


def endpoint_value(value: Any, name: str) -> str:
    value = nonempty_string(value, name)
    if not value.startswith(('tcp://', 'ipc://', 'inproc://')):
        raise ValueError(f'{name} must be a complete ZMQ endpoint')
    return value


__all__ = [
    'CONNECTION_TRANSPORT_FIELDS',
    'DEFAULT_ENDPOINT',
    'DEFAULT_REQUEST_TIMEOUT_S',
    'FluxVLAZMQTransport',
    'connection_transport',
]
