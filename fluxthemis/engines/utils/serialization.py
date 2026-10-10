from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert common numeric/container objects into JSON-safe values."""
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Cannot serialize NaN or infinite floats')
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, 'item'):
        try:
            converted = value.item()
        except (TypeError, ValueError):
            pass
        else:
            if converted is not value:
                return to_jsonable(converted)
    if hasattr(value, 'tolist'):
        try:
            converted = value.tolist()
        except (TypeError, ValueError):
            pass
        else:
            if converted is not value:
                return to_jsonable(converted)
    raise TypeError(f'Cannot serialize {type(value).__name__} as JSON')
