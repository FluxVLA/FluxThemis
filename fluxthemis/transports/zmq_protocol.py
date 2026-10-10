"""Versioned multipart protocol for the FluxVLA ZMQ transport.

The encoding matches FluxVLA's ``MsgSerializer`` (numpy arrays encoded as
``{'__ndarray__': True, 'data': <npy bytes>}``) so both repositories share
one wire format.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1


def encode_frame(data: Any) -> bytes:
    import msgpack
    return msgpack.packb(data, default=_encode, use_bin_type=True)


def decode_frame(data: bytes) -> Any:
    import msgpack
    return msgpack.unpackb(
        data, object_hook=_decode, raw=False, strict_map_key=False)


def encode_header(message_type: str, **values: Any) -> bytes:
    if not isinstance(message_type, str) or not message_type:
        raise ValueError('message_type must be a non-empty string')
    return encode_frame({
        'version': PROTOCOL_VERSION,
        'type': message_type,
        **values,
    })


def decode_header(frame: bytes) -> dict[str, Any]:
    value = decode_frame(frame)
    if not isinstance(value, dict):
        raise TypeError('ZMQ message header must be a mapping')
    if value.get('version') != PROTOCOL_VERSION:
        raise ValueError(
            f'Unsupported ZMQ protocol version {value.get("version")!r}')
    message_type = value.get('type')
    if not isinstance(message_type, str) or not message_type:
        raise ValueError('ZMQ message header.type must be a non-empty string')
    return value


def empty_payload() -> bytes:
    return encode_frame({})


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {'__ndarray__': True, 'data': buf.getvalue()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'Cannot serialize {type(obj).__name__}')


def _decode(obj: dict[str, Any]) -> Any:
    if obj.get('__ndarray__') is True:
        return np.load(io.BytesIO(obj['data']), allow_pickle=False)
    return obj


__all__ = [
    'PROTOCOL_VERSION',
    'decode_frame',
    'decode_header',
    'empty_payload',
    'encode_frame',
    'encode_header',
]
