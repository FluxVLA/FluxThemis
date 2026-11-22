import pytest

from fluxthemis.environments import MockVectorEnvironment


@pytest.mark.parametrize('capacity', [1, 3])
@pytest.mark.parametrize('slot', [-1, 'capacity', True, 0.0, '0', None])
@pytest.mark.parametrize('method', ['render', 'finalize_episode'])
def test_mock_vector_environment_checks_slot_bounds(capacity, slot, method):
    environment = MockVectorEnvironment(capacity)
    call = getattr(environment, method)
    kwargs = (
        dict(success=False, termination_reason='max_steps')
        if method == 'finalize_episode' else {})
    invalid_slot = capacity if slot == 'capacity' else slot
    with pytest.raises(ValueError, match='slot must be an integer in'):
        call(invalid_slot, **kwargs)
    for valid_slot in range(capacity):
        expected = {} if method == 'finalize_episode' else None
        assert call(valid_slot, **kwargs) == expected
