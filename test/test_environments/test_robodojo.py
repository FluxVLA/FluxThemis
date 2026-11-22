import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluxthemis.engines.runners.episode import build_context
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings)
from fluxthemis.engines.runners.resume import build_bundle, manifest
from fluxthemis.environments import EnvironmentResetRequest
from fluxthemis.environments.sim import robodojo
from fluxthemis.environments.sim.robodojo import RoboDojoEnvironment
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.reporters import MockEvaluationReporter
from fluxthemis.structures import (CaseRejection, EnvironmentReset,
                                   EpisodeInterruption, StepResult)


class UnStableError(RuntimeError):
    pass


class PhysXBrokenError(RuntimeError):

    def __init__(self, slots):
        self.broken_envs = slots


class _BatchNativeEnvironment:

    def __init__(self):
        self.reset_calls = []
        self.action_calls = []
        self.observation_calls = []
        self.close_calls = 0
        self.reward_calls = 0
        self.unstable_envs = set()
        self.abort_calls = []
        self.video_calls = []
        self.failure_at = None
        self.failure_type = PhysXBrokenError
        self.failure_slot = 0
        self.initialized = False
        self.pending_controls = []
        self.step_reward_calls = 0

    def reset(self, seed):
        self.reset_calls.append(seed)
        self.end_flag = [value is None for value in seed]
        self.success = [True] * len(seed)
        self.take_action_cnt = [0] * len(seed)
        self.pending_controls = [[] for _ in seed]
        self.initialized = False
        self._check_failure('reset')
        self.initialized = True

    def close(self):
        self.close_calls += 1
        self.pending_controls = []
        self.initialized = False
        self.unstable_envs.clear()

    def _check_failure(self, stage):
        if self.failure_at != stage:
            return
        self.failure_at = None
        if self.failure_type is UnStableError:
            self.unstable_envs.add(self.failure_slot)
            self.end_flag[self.failure_slot] = True
        raise self.failure_type({self.failure_slot})

    def run_reward(self):
        assert self.initialized
        self.reward_calls += 1

    def get_obs_batch(self, env_idx_list):
        assert self.initialized
        self.observation_calls.append(list(env_idx_list))
        return [{'slot': slot} for slot in env_idx_list]

    def take_action_batch(self, actions, env_idx_list):
        assert self.initialized
        self._check_failure('step_start')
        self.action_calls.append((actions, env_idx_list))
        for action, slot in zip(actions, env_idx_list):
            self.take_action_cnt[slot] += 1
            self.pending_controls[slot].extend([action] * 2)
            self.pending_controls[slot].pop(0)
        self._check_failure('step_mid')
        for slot in env_idx_list:
            self.pending_controls[slot].clear()
        self.step_reward_calls += 1
        if 0 in env_idx_list:
            self.end_flag[0] = True

    def _abort_video_writers(self, env_idx_list):
        self.abort_calls.append(list(env_idx_list))

    def save_video(self, slot, path, tag):
        self.video_calls.append((slot, tag))
        Path(path.replace('.mp4', f'_camera_{tag}.mp4')).write_bytes(b'video')


@pytest.fixture
def runtime_environment(monkeypatch, tmp_path):
    """Build real adapter configs and switch scenes without launching Kit."""
    monkeypatch.setenv('ROBODOJO_RUN_ID', 'test-slots')
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)

    def build(capacity=8, clutter_env_limit=None):
        sim_cfg = {'scene': {}}
        if clutter_env_limit is not None:
            sim_cfg['scene']['clutter_env_limit'] = clutter_env_limit
        files = {
            'arx_x5.yml': {
                'config':
                dict(sim='sim', scene='scene', camera='camera', robot='robot'),
                'observation': {},
            },
            'sim.yml': sim_cfg,
        }
        resolved, configs, natives = [], [], []
        loaded_paths = []

        def load_yaml(path):
            loaded_paths.append(Path(path).name)
            return deepcopy(files.get(Path(path).name, {}))

        def resolve(task_name, num_envs, sim):
            resolved.append((task_name, num_envs, deepcopy(sim)))
            return (min(num_envs, sim['scene'].get('clutter_env_limit', 5))
                    if task_name.endswith('_random') else num_envs)

        def update(config, path, value, force_add):
            keys = path.split('.')
            for key in keys[:-1]:
                config = config.setdefault(key, {})
            config[keys[-1]] = value

        def create(config, app):
            configs.append(deepcopy(config))
            native = _BatchNativeEnvironment()
            native.save_dir = tmp_path
            natives.append(native)
            return native

        runtime = SimpleNamespace(
            ROOT_DIR=str(tmp_path),
            ENV_CONFIG_PATH=str(tmp_path / 'env_cfg'),
            BENCHMARK='RoboDojo',
            load_yaml=load_yaml,
            loaded_paths=loaded_paths,
            OmegaConf=SimpleNamespace(create=deepcopy, update=update),
            process_randomization=lambda config: config,
            process_config=lambda config, task_name: (config, 25),
            resolve_random_task_num_envs=resolve)
        env = RoboDojoEnvironment(
            task_name=['stack_bowls', 'stack_bowls_random'],
            num_envs=capacity,
            robodojo_root=tmp_path,
            save_videos=True,
            video_dir=tmp_path)
        env._configure_task_slots(runtime)
        env._runtime = runtime
        env._post = SimpleNamespace(
            task_registry=SimpleNamespace(task_config_path=lambda directory,
                                          task: f'{directory}/{task}.yml'),
            create_eval_env=create)
        env._simulation_app = SimpleNamespace(close=lambda: None)
        monkeypatch.setattr(env, '_sync_runtime_paths', lambda: None)
        monkeypatch.setattr(env, '_action_to_dict',
                            lambda action: {'action': action})
        env._activate_task(env.tasks()[0])
        return env, natives, configs, resolved

    return build


@pytest.mark.parametrize('capacity', [1, 5, 8])
@pytest.mark.parametrize('clutter_env_limit', [None, 3, 12])
def test_native_task_capacity_uses_runtime_resolver(runtime_environment,
                                                    capacity,
                                                    clutter_env_limit):
    env, natives, configs, resolved = runtime_environment(
        capacity, clutter_env_limit)
    base, random = env.tasks()
    limit = min(capacity,
                5 if clutter_env_limit is None else clutter_env_limit)
    assert 'max_slots' not in base.metadata
    assert 'max_slots' not in random.metadata
    assert env.task_capacity(base) == capacity
    assert env.task_capacity(random) == limit
    env._activate_task(random)
    assert env.num_envs == capacity
    assert env._native_num_envs == limit
    for config, count in zip(configs, (capacity, limit)):
        assert config['sim']['scene']['num_envs'] == count
        assert config['eval_cfg']['num_envs'] == count
        assert config['sim']['seed'] == [0] * count
    assert [(task, count)
            for task, count, _ in resolved] == [(base.task_id, capacity),
                                                (random.task_id, capacity)]
    assert env._runtime.loaded_paths.count('sim.yml') == 1
    assert all(sim['scene'].get('clutter_env_limit') == clutter_env_limit
               for _, _, sim in resolved)
    env.close()


@pytest.mark.parametrize('capacity', [1, 8])
def test_task_metadata_context_and_resume_manifest_ignore_capacity(
        runtime_environment, tmp_path, capacity):
    env, _, _, _ = runtime_environment(capacity)
    metadata = [
        dict(
            benchmark='robodojo',
            task_name=name,
            task_index=index,
            dimension='generalization',
            variant=variant,
            base_task='stack_bowls')
        for index, (name, variant) in enumerate((('stack_bowls', 'base'),
                                                 ('stack_bowls_random',
                                                  'random')))
    ]
    assert [task.metadata for task in env.tasks()] == metadata
    for task in env.tasks():
        context = build_context(task, 0, 7, {})
        assert context['task_metadata'] == task.metadata
        assert 'max_slots' not in json.dumps(context)

    # Pre-capability run_start format: only task identity/benchmark metadata.
    old_start = dict(
        schema_version='1.0',
        run_name='original',
        started_at='2026-01-01T00:00:00+00:00',
        base_seed=7,
        episodes_per_task=1,
        episodes_per_task_by_task={
            'stack_bowls': 1,
            'stack_bowls_random': 1
        },
        max_episode_steps=1000,
        execute_horizon=None,
        stop_on_success=True,
        task_ids=['stack_bowls', 'stack_bowls_random'],
        total_tasks=2,
        total_episodes=2,
        parallel_workers=1,
        strict_worker_assignment=False,
        tasks=[
            dict(
                task_id=m['task_name'],
                task_index=index,
                description='',
                metadata=m) for index, m in enumerate(metadata)
        ])
    journal = dict(
        event_type='run_start',
        sequence=1,
        run_session_id='original',
        recorded_at=old_start['started_at'],
        payload=old_start)
    (tmp_path / 'events.jsonl').write_text(json.dumps(journal) + '\n')
    bundle_path = tmp_path / 'resume.json'
    bundle_path.write_text(json.dumps(build_bundle(tmp_path)))
    with EvaluationRun(
            EvaluationSettings(resume_path=str(bundle_path)),
            SuccessRateEvaluator(), MockEvaluationReporter()) as run:
        run.start(env.tasks(), 2)
        assert manifest(run.start_payload) == manifest(old_start)
        assert run.start_payload['restored_episodes'] == 0
    env.close()


@pytest.mark.parametrize('invalid_slot', [5, 6, 7])
def test_random_task_requires_empty_slots_beyond_native_scenes(
        runtime_environment, invalid_slot):
    env, natives, _, _ = runtime_environment()
    task = env.tasks()[1]
    requests = [
        EnvironmentResetRequest(task, slot, slot, slot) for slot in range(5)
    ] + [None] * 3
    invalid = list(requests)
    invalid[invalid_slot] = EnvironmentResetRequest(task, invalid_slot, 0, 99)
    with pytest.raises(ValueError, match='outside the native batch'):
        env.reset_batch(invalid)
    native = natives[-1]
    assert not native.reset_calls
    resets = env.reset_batch(requests)
    assert native.reset_calls == [list(range(5))]
    assert len(resets) == 8 and resets[5:] == [None] * 3
    assert [reset.observation['slot']
            for reset in resets[:5]] == list(range(5))
    actions = [1] * 5 + [None] * 3
    invalid = list(actions)
    invalid[invalid_slot] = 1
    with pytest.raises(ValueError, match='outside the native batch'):
        env.step_batch(invalid)
    assert not native.action_calls
    transitions = env.step_batch(actions)
    assert len(transitions) == 8 and transitions[5:] == [None] * 3
    assert native.action_calls[-1][1] == list(range(5))
    assert len(native.end_flag) == 5
    with pytest.raises(ValueError, match='outside the native batch'):
        env.finalize_episode(
            invalid_slot, success=False, termination_reason='terminated')
    env.close()
    assert native.video_calls == [(slot, 'truncated') for slot in range(5)]
    assert env._video_finalized_slots == set(range(5))


def test_switching_tasks_resizes_native_state_and_preserves_video_ownership(
        runtime_environment):
    env, natives, _, _ = runtime_environment()
    base, random = env.tasks()
    for task, count in ((base, 8), (random, 5), (base, 8)):
        requests = [
            EnvironmentResetRequest(task, slot, slot, slot)
            for slot in range(count)
        ] + [None] * (8 - count)
        resets = env.reset_batch(requests)
        assert len(resets) == env.num_envs == 8
        assert len(env._batch_requests) == count
        assert env._batch_inactive == set()
        assert env._video_finalized_slots == set()
        env.finalize_episode(
            count - 1, success=False, termination_reason='case_rejected')
        assert natives[-1].abort_calls == [[count - 1]]
        assert env._batch_inactive == {count - 1}
    env.close()
    assert [native.video_calls
            for native in natives] == [[(slot, 'truncated')
                                        for slot in range(count - 1)]
                                       for count in (8, 5, 8)]


@pytest.fixture(params=[1, 3])
def batch_environment(monkeypatch, tmp_path, request):
    capacity = request.param
    env = RoboDojoEnvironment(
        task_name='put_bottles_into_dustbin',
        num_envs=capacity,
        robodojo_root=tmp_path)
    native = _BatchNativeEnvironment()
    env._env = native
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)
    monkeypatch.setattr(env, '_action_to_dict',
                        lambda action: {'action': action})
    requests = [
        EnvironmentResetRequest(env.tasks()[0], slot, 17 + slot, 10 + slot)
        for slot in range(capacity)
    ]
    return env, native, requests


def test_routes_active_slots_and_resets_layouts(batch_environment):
    env, native, requests = batch_environment
    capacity = len(requests)
    resets = env.reset_batch(requests)
    assert native.reset_calls == [list(range(10, 10 + capacity))]
    assert [reset.info['environment_slot']
            for reset in resets] == list(range(capacity))
    assert [reset.info['environment_seed']
            for reset in resets] == list(range(17, 17 + capacity))
    for slot, reset in enumerate(resets):
        assert reset.info == dict(
            task_id=requests[slot].task.task_id,
            episode_index=slot,
            environment_seed=17 + slot,
            environment_case_index=10 + slot,
            layout_index=10 + slot,
            environment_slot=slot)
    assert all(reset.observation['predicates']['environment_success'] is False
               for reset in resets)
    first = env.step_batch(list(range(1, capacity + 1)))
    assert first[0].terminated and first[0].success
    assert all(not result.terminated for result in first[1:])
    for slot, result in enumerate(first):
        assert result.info == dict(
            task_id=requests[slot].task.task_id,
            step=1,
            score=1.0 if slot == 0 else 0.0,
            environment_case_index=10 + slot,
            layout_index=10 + slot,
            environment_slot=slot)
    second = env.step_batch([None] + list(range(2, capacity + 1)))
    assert second[0] is None
    assert [result.info['layout_index']
            for result in second[1:]] == list(range(11, 10 + capacity))
    assert native.take_action_cnt == [1] + [2] * (capacity - 1)
    assert native.observation_calls[0] == list(range(capacity))
    for actions, slots in native.action_calls:
        assert len(actions) == len(slots)
    with pytest.raises(ValueError, match='reactivate'):
        env.step_batch([1] + [None] * (capacity - 1))
    env.reset_batch(requests)
    assert native.close_calls == 1
    assert native.reward_calls == 2
    assert env.step_batch(list(range(1, capacity + 1)))[0].terminated


@pytest.mark.parametrize('marker', ['unstable_envs', 'end_flag'])
def test_rejects_only_unstable_reset_slot(batch_environment, monkeypatch,
                                          marker):
    env, native, requests = batch_environment
    with pytest.raises(ValueError, match='one entry per slot'):
        env.reset_batch(requests[:-1])
    assert not native.reset_calls
    rejected_slot = len(requests) - 1
    reset = native.reset

    def unstable(seed):
        reset(seed)
        if marker == 'end_flag':
            native.end_flag[rejected_slot] = True
        else:
            native.unstable_envs.add(rejected_slot)

    monkeypatch.setattr(native, 'reset', unstable)
    resets = env.reset_batch(requests)
    assert [type(reset) for reset in resets
            ] == ([EnvironmentReset] * rejected_slot + [CaseRejection])
    assert resets[rejected_slot].info[
        'environment_case_index'] == 10 + rejected_slot
    assert native.abort_calls == [[rejected_slot]]
    transitions = env.step_batch([1] * rejected_slot + [None])
    assert transitions[rejected_slot] is None
    assert all(result is not None for result in transitions[:rejected_slot])


def test_unidentified_native_reset_failure_returns_slot_rejections(
        batch_environment, monkeypatch):
    env, native, requests = batch_environment

    def unstable(seed):
        native.reset_calls.append(seed)
        raise UnStableError('unstable scene')

    monkeypatch.setattr(native, 'reset', unstable)
    resets = env.reset_batch(requests)
    assert all(isinstance(reset, CaseRejection) for reset in resets)
    assert len(resets) == len(requests)
    assert not native.observation_calls


def test_scores_are_slot_specific(batch_environment):
    env, native, requests = batch_environment
    scores = [100, 25, 75][:len(requests)]
    native.get_score = lambda: None
    native.reward_manager = SimpleNamespace(get_score=lambda: scores)
    env.reset_batch(requests)
    transitions = env.step_batch([1] * len(requests))
    assert [item.score
            for item in transitions] == [score / 100 for score in scores]


def test_partial_assignment_masks_unused_native_slots(batch_environment):
    env, native, requests = batch_environment
    inactive = [None] * (len(requests) - 1)
    native.unstable_envs = set(range(1, len(requests)))
    resets = env.reset_batch([requests[0], *inactive])
    assert isinstance(resets[0], EnvironmentReset)
    assert resets[1:] == inactive
    assert native.reset_calls == [[10, *inactive]]
    transitions = env.step_batch([1, *inactive])
    assert transitions[1:] == inactive
    assert native.action_calls[-1][1] == [0]


def test_all_unused_reset_slots_do_not_start_native_episodes(
        batch_environment):
    env, native, requests = batch_environment
    inactive = [None] * len(requests)
    assert env.reset_batch(inactive) == inactive
    assert not native.reset_calls


def test_runtime_invalid_layout_freezes_only_its_lane(batch_environment,
                                                      monkeypatch):
    env, native, requests = batch_environment
    capacity = len(requests)
    rejected_slot = capacity - 1
    env.reset_batch(requests)
    take_action = native.take_action_batch

    def invalidate(actions, env_idx_list):
        take_action(actions, env_idx_list)
        native.unstable_envs.add(rejected_slot)

    monkeypatch.setattr(native, 'take_action_batch', invalidate)
    transitions = env.step_batch([1] * capacity)
    assert isinstance(transitions[rejected_slot], CaseRejection)
    assert all(
        isinstance(result, StepResult)
        for result in transitions[:rejected_slot])
    assert native.abort_calls == [[rejected_slot]]
    actions = [None] * capacity
    for slot in range(1, rejected_slot):
        actions[slot] = 2
    transitions = env.step_batch(actions)
    assert transitions[0] is None and transitions[rejected_slot] is None
    assert all(transitions[slot] is not None
               for slot in range(1, rejected_slot))
    assert native.take_action_cnt == [
        1 + int(1 <= slot < rejected_slot) for slot in range(capacity)
    ]


@pytest.mark.parametrize('phase', ['reset', 'step'])
def test_observation_precheck_keeps_healthy_slots_observable(
        batch_environment, monkeypatch, phase):
    env, native, requests = batch_environment
    rejected_slot = len(requests) - 1
    observe = native.get_obs_batch
    broken = set()
    env._physx_monitor = SimpleNamespace(
        reset=broken.clear, get_broken_envs=lambda: set(broken))

    def check_then_observe(env_idx_list):
        broken.add(rejected_slot)
        # Native checks all live lanes before rendering, without changing
        # control/reward state. Masking the bad lane makes capture safe.
        live = {slot for slot in broken if not native.end_flag[slot]}
        if live:
            raise PhysXBrokenError(live)
        return observe(env_idx_list)

    if phase == 'step':
        env.reset_batch(requests)
    monkeypatch.setattr(native, 'get_obs_batch', check_then_observe)
    results = (
        env.reset_batch(requests)
        if phase == 'reset' else env.step_batch([1] * len(requests)))
    rejected = results[rejected_slot]
    assert isinstance(rejected, CaseRejection)
    assert all(not isinstance(result, EpisodeInterruption)
               for result in results)
    assert native.abort_calls == [[rejected_slot]]
    transitions = env.step_batch(
        [None if flag else 1 for flag in native.end_flag])
    assert transitions[rejected_slot] is None
    assert all(
        isinstance(result, StepResult) for result in transitions
        if result is not None)


@pytest.mark.parametrize('failure_at', ['reset', 'step_start', 'step_mid'])
@pytest.mark.parametrize('failure_type', [PhysXBrokenError, UnStableError])
def test_interrupted_native_call_discards_all_unfinished_slots(
        batch_environment, failure_at, failure_type):
    env, native, requests = batch_environment
    capacity = len(requests)
    if failure_at != 'reset':
        env.reset_batch(requests)
    native.failure_at = failure_at
    native.failure_type = failure_type
    native.failure_slot = capacity - 1
    observations_before = list(native.observation_calls)
    rewards_before = native.reward_calls
    results = (
        env.reset_batch(requests)
        if failure_at == 'reset' else env.step_batch([1] * capacity))
    assert len(results) == capacity
    assert all(
        isinstance(result, EpisodeInterruption) for result in results[:-1])
    rejected = results[-1]
    assert isinstance(rejected, CaseRejection)
    for slot, result in enumerate(results):
        info = dict(environment_case_index=10 + slot, environment_slot=slot)
        if failure_at == 'reset':
            info.update(
                task_id=requests[slot].task.task_id,
                episode_index=slot,
                environment_seed=17 + slot,
                layout_index=10 + slot,
                reason=('RoboDojo initial layout is unstable' if slot
                        == capacity - 1 else 'RoboDojo reset was interrupted'))
        else:
            info['reason'] = (
                'RoboDojo marked this sample unstable' if slot == capacity -
                1 else 'RoboDojo did not finish executing the batch')
        assert result.info == info
    assert native.observation_calls == observations_before
    assert native.reward_calls == rewards_before
    assert native.step_reward_calls == 0
    assert native.initialized == (failure_at != 'reset')
    assert native.take_action_cnt == [int(failure_at == 'step_mid')] * capacity
    assert [len(queue) for queue in native.pending_controls
            ] == [int(failure_at == 'step_mid')] * capacity
    assert native.abort_calls == [[slot] for slot in range(capacity)]
    assert env._batch_inactive == set(range(capacity))
    assert env._video_finalized_slots == set(range(capacity))
    assert env.step_batch([None] * capacity) == [None] * capacity
    with pytest.raises(ValueError, match='reactivate'):
        env.step_batch([1] * capacity)

    # The next attempt must rebuild the scene, including control queues and
    # robot/reward initialization, before any actions are accepted.
    resets = env.reset_batch(requests)
    assert native.close_calls == 1
    assert native.initialized
    assert native.pending_controls == [[] for _ in requests]
    assert native.take_action_cnt == [0] * capacity
    assert all(isinstance(reset, EnvironmentReset) for reset in resets)
    assert not native.video_calls
    assert all(
        isinstance(result, StepResult)
        for result in env.step_batch([1] * capacity))


def test_native_interruption_preserves_finished_and_unavailable_slots(
        runtime_environment):
    env, natives, _, _ = runtime_environment()
    task = env.tasks()[1]
    requests = [
        EnvironmentResetRequest(task, slot, slot, slot) for slot in range(5)
    ] + [None] * 3
    env.reset_batch(requests)
    native = natives[-1]
    results = env.step_batch([1] * 5 + [None] * 3)
    assert results[0].terminated and results[0].success
    artifacts = env.finalize_episode(
        0, success=True, termination_reason='success')
    native.failure_at = 'step_mid'
    native.failure_slot = 4
    results = env.step_batch([None] + [1] * 4 + [None] * 3)
    assert results[0] is None
    assert all(
        isinstance(result, EpisodeInterruption) for result in results[1:4])
    assert isinstance(results[4], CaseRejection)
    assert results[5:] == [None] * 3
    assert native.abort_calls == [[slot] for slot in range(1, 5)]
    assert native.video_calls == [(0, 'success')]
    assert env._video_finalized_slots == set(range(5))
    assert all(
        Path(path).read_bytes() == b'video'
        for path in artifacts['video_paths'])
    retries = [None, *requests[1:5], None, None, None]
    env.reset_batch(retries)
    assert native.close_calls == 1
    assert native.pending_controls == [[] for _ in range(5)]
    assert native.reset_calls[-1] == [None, 1, 2, 3, 4]
    assert native.video_calls == [(0, 'success')]


def test_rejection_during_initial_observation_is_slot_specific(
        batch_environment, monkeypatch):
    env, native, requests = batch_environment
    get_observations = native.get_obs_batch
    rejected_slot = len(requests) - 1

    def unstable(env_idx_list):
        observations = get_observations(env_idx_list)
        native.unstable_envs.add(rejected_slot)
        return observations

    monkeypatch.setattr(native, 'get_obs_batch', unstable)
    resets = env.reset_batch(requests)
    assert [type(reset) for reset in resets
            ] == ([EnvironmentReset] * rejected_slot + [CaseRejection])


def test_policy_stop_freezes_its_slot_without_changing_native_success(
        batch_environment):
    env, native, requests = batch_environment
    slot = len(requests) - 1
    env.reset_batch(requests)
    before = (list(native.end_flag), list(native.success))
    assert env.finalize_episode(
        slot, success=False, termination_reason='policy_completed') == {}
    assert (native.end_flag, native.success) == before
    transitions = env.step_batch([1] * slot + [None])
    assert transitions[slot] is None
    assert native.take_action_cnt == [1] * slot + [0]
    with pytest.raises(ValueError, match='reactivate'):
        env.step_batch([None] * slot + [1])


@pytest.mark.parametrize('capacity', [1, 3])
def test_completion_diagnostics_use_each_native_slot(monkeypatch, tmp_path,
                                                     capacity):
    env = RoboDojoEnvironment(
        task_name='stack_blocks',
        num_envs=capacity,
        completion_diagnostics=True,
        robodojo_root=tmp_path)
    native = _BatchNativeEnvironment()
    native._stacked_xy_thresholds = lambda: 0.01
    values = {'stacked': [1, 0, 1], 'home': [0, 1, 1]}
    native.reward_manager = SimpleNamespace(
        is_stacked=lambda **kwargs: 'stacked',
        all_robot_back_to_origin=lambda: 'home',
        call_func_parser=lambda check, slot: values[check][slot])
    env._env = native
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)
    monkeypatch.setattr(env, '_action_to_dict', lambda action: action)
    requests = [
        EnvironmentResetRequest(env.tasks()[0], slot, slot, slot)
        for slot in range(capacity)
    ]
    resets = env.reset_batch(requests)
    for slot, reset in enumerate(resets):
        assert reset.observation['predicates']['blocks_stacked'] == bool(
            values['stacked'][slot])
        assert reset.observation['predicates']['arms_at_origin'] == bool(
            values['home'][slot])
        assert reset.info['completion_predicates'] == reset.observation[
            'predicates']
    transitions = env.step_batch([1] * capacity)
    assert [
        result.observation['predicates']['environment_success']
        for result in transitions
    ] == [True] + [False] * (
        capacity - 1)


@pytest.mark.parametrize('shared_video_dir', [False, True])
def test_native_video_finalizes_and_aborts_each_slot(batch_environment,
                                                     tmp_path,
                                                     shared_video_dir):
    env, native, requests = batch_environment
    env.save_videos = True
    env._video_dir = tmp_path / 'videos'
    env._video_dir.mkdir()
    native.save_dir = tmp_path / ('videos' if shared_video_dir else 'native')
    Path(native.save_dir).mkdir(exist_ok=True)
    env.reset_batch(requests)
    for slot, request in enumerate(requests):
        info = env.finalize_episode(
            slot, success=False, termination_reason='policy_completed')
        assert native.video_calls[-1] == (slot, 'agent_done')
        path, = info['video_paths']
        assert Path(path).parent == env._video_dir
        assert f'episode_{request.episode_index:07d}' in Path(path).name
        assert Path(path).read_bytes() == b'video'
        assert env.finalize_episode(
            slot, success=False, termination_reason='policy_completed') == {}
    env.reset_batch(requests)
    rejected_slot = len(requests) - 1
    assert env.finalize_episode(
        rejected_slot, success=False,
        termination_reason='case_rejected') == {}
    assert native.abort_calls == [[rejected_slot]]
    env.close()
    assert [slot for slot, _ in native.video_calls] == (
        list(range(len(requests))) + list(range(rejected_slot)))
