import pytest

from fluxthemis.environments import EnvironmentCaseRejected
from fluxthemis.environments.sim.robodojo import RoboDojoEnvironment
from fluxthemis.structures import TaskSpec


class UnStableError(RuntimeError):
    pass


class _UnstableNativeEnvironment:

    def __init__(self):
        self.reset_calls = []
        self.close_calls = 0

    def reset(self, seed):
        self.reset_calls.append(seed)
        raise UnStableError('unstable scene')

    def close(self):
        self.close_calls += 1


def test_robodojo_translates_unstable_layout_to_case_rejection():
    task = TaskSpec('task')
    native = _UnstableNativeEnvironment()
    environment = object.__new__(RoboDojoEnvironment)
    environment._env = native
    environment._task = task
    environment._task_ids = {task.task_id}
    environment._episode_active = False
    environment._episode_task = None
    environment._episode_index = None
    environment._episode_layout_index = None
    environment._physx_monitor = None
    environment._video_finalized = True

    with pytest.raises(
            EnvironmentCaseRejected, match='RoboDojo layout 17 is unstable'):
        environment.reset(
            task,
            episode_index=0,
            environment_seed=41,
            environment_case_index=17,
        )

    assert native.reset_calls == [[17]]
    assert native.close_calls == 1


class _BatchNativeEnvironment:

    def __init__(self):
        self.reset_calls = []
        self.action_calls = []
        self.close_calls = 0
        self.reward_calls = 0
        self.unstable_envs = set()

    def reset(self, seed):
        self.reset_calls.append(seed)
        self.end_flag = [value is None for value in seed]
        self.success = [True] * len(seed)
        self.take_action_cnt = [0] * len(seed)

    def close(self):
        self.close_calls += 1

    def run_reward(self):
        self.reward_calls += 1

    def get_obs_batch(self, env_idx_list=None):
        indices = range(len(
            self.end_flag)) if env_idx_list is None else env_idx_list
        return [{'slot': slot} for slot in indices]

    def take_action_batch(self, actions, env_idx_list):
        self.action_calls.append((actions, env_idx_list))
        for slot in env_idx_list:
            self.take_action_cnt[slot] += 1
        if 0 in env_idx_list:
            self.end_flag[0] = True


@pytest.fixture
def batch_environment(monkeypatch, tmp_path):
    from fluxthemis.environments.base import EnvironmentReset
    from fluxthemis.environments.sim import robodojo

    env = RoboDojoEnvironment(
        task_name='put_bottles_into_dustbin',
        num_envs=3,
        robodojo_root=tmp_path)
    native = _BatchNativeEnvironment()
    env._env = native
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)
    monkeypatch.setattr(env, '_action_to_dict',
                        lambda action: {'action': action})
    requests = [
        EnvironmentReset(env.tasks()[0], i, 17 + i, 10 + i) for i in range(3)
    ]
    return env, native, requests


def test_robodojo_batch_routes_active_slots_and_resets_layouts(
        batch_environment):
    env, native, requests = batch_environment
    observations = env.reset_batch(requests)
    assert native.reset_calls == [[10, 11, 12]]
    assert [info['environment_slot'] for _, info in observations] == [0, 1, 2]
    assert [info['environment_seed']
            for _, info in observations] == [17, 18, 19]
    first = env.step_batch([1, 2, 3])
    assert first[0].terminated and first[0].success
    assert not first[1].terminated
    second = env.step_batch([None, 4, 5])
    assert second[0] is None
    assert second[2].observation == {'slot': 2}
    assert second[2].info['layout_index'] == 12
    assert native.action_calls[-1] == ([{'action': 4}, {'action': 5}], [1, 2])
    with pytest.raises(ValueError, match='reactivate'):
        env.step_batch([1, 4, 5])
    env.reset_batch(requests)
    assert native.close_calls == 1
    assert native.reward_calls == 2
    assert env.step_batch([1, 2, 3])[0].terminated


def test_robodojo_batch_rejects_partial_and_unstable_reset(batch_environment):
    env, native, requests = batch_environment
    with pytest.raises(ValueError, match='one entry per slot'):
        env.reset_batch(requests[:2])
    assert not native.reset_calls
    native.unstable_envs = {1}
    with pytest.raises(EnvironmentCaseRejected, match='slots') as error:
        env.reset_batch(requests)
    assert error.value.slots == (1, )


def test_robodojo_batch_scores_are_slot_specific(batch_environment):
    from types import SimpleNamespace

    env, native, requests = batch_environment
    native.get_score = lambda: None
    native.reward_manager = SimpleNamespace(get_score=lambda: [100, 25, 75])
    env.reset_batch(requests)
    transitions = env.step_batch([1, 2, 3])
    assert [item.score for item in transitions] == [1.0, 0.25, 0.75]


def test_robodojo_tail_batch_masks_unused_native_lanes(batch_environment):
    env, native, requests = batch_environment
    observations = env.reset_batch([requests[0], None, requests[2]])
    assert observations[1] is None
    assert observations[2][1]['layout_index'] == 12
    assert native.reset_calls == [[10, None, 12]]
    transitions = env.step_batch([1, None, 2])
    assert transitions[1] is None
    assert native.action_calls[-1][1] == [0, 2]


def test_robodojo_inactive_lane_cannot_reject_tail_batch(batch_environment):
    env, native, requests = batch_environment
    native.unstable_envs = {1}
    assert env.reset_batch([requests[0], None, None])[0] is not None
    assert env.step_batch([1, None, None])[1:] == [None, None]


def test_runtime_invalid_layout_freezes_only_its_lane(batch_environment,
                                                      monkeypatch):
    env, native, requests = batch_environment
    env.reset_batch(requests)
    take_action = native.take_action_batch

    def invalidate(actions, env_idx_list):
        take_action(actions, env_idx_list)
        native.unstable_envs.add(1)

    monkeypatch.setattr(native, 'take_action_batch', invalidate)
    transitions = env.step_batch([1, 2, 3])
    assert transitions[0].success
    assert transitions[1].case_rejected
    assert transitions[1].success is None
    assert transitions[2].observation == {'slot': 2}
    assert not transitions[2].case_rejected
    transitions = env.step_batch([None, None, 4])
    assert transitions[:2] == [None, None]
    assert native.take_action_cnt == [1, 1, 2]
    assert native.action_calls[-1][1] == [2]


def test_partial_native_reset_failure_identifies_slot_and_cleans_up(
        batch_environment, monkeypatch):
    env, native, requests = batch_environment
    reset = native.reset

    def unstable(seed):
        reset(seed)
        native.end_flag[1] = True
        raise UnStableError('partial reset')

    monkeypatch.setattr(native, 'reset', unstable)
    with pytest.raises(EnvironmentCaseRejected) as error:
        env.reset_batch(requests)
    assert error.value.slots == (1, )
    monkeypatch.setattr(native, 'reset', reset)
    env.reset_batch(requests)
    assert native.close_calls == 1
