from pathlib import Path

import numpy as np
import pytest

from fluxthemis.engines.utils import to_jsonable


def test_to_jsonable_recursively_converts_supported_values():
    value = {
        'array': np.asarray([1.0, 2.0], dtype=np.float32),
        'scalar': np.int64(3),
        'path': Path('/tmp/result.json'),
    }

    assert to_jsonable(value) == {
        'array': [1.0, 2.0],
        'scalar': 3,
        'path': '/tmp/result.json',
    }


def test_to_jsonable_rejects_invalid_values():
    with pytest.raises(ValueError, match='NaN or infinite'):
        to_jsonable(float('nan'))
    with pytest.raises(TypeError, match='object'):
        to_jsonable(object())
