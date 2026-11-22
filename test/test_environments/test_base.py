import pytest

from fluxthemis.environments import MockEnvironment


@pytest.mark.parametrize('method', ['reset_batch', 'step_batch'])
@pytest.mark.parametrize('length', [0, 2])
def test_single_environment_adapter_requires_one_slot(method, length):
    environment = MockEnvironment()
    with pytest.raises(
            ValueError,
            match='^SingleEnvironmentAdapter expects exactly one slot$'):
        getattr(environment, method)([None] * length)
    assert getattr(environment, method)([None]) == [None]
