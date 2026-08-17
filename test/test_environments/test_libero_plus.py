from types import SimpleNamespace

import pytest

from fluxthemis.environments.sim import libero_plus as libero_plus_module
from fluxthemis.environments.sim.libero_plus import (LIBERO_PLUS_SUITE_TASKS,
                                                     LiberoPlusEnvironment,
                                                     _LiberoPlusRuntime,
                                                     _validate_classification)


class _RandomizationError(Exception):
    pass


class _FakeSimulator:

    def __init__(self):
        self.close_calls = 0

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return {}

    def set_init_state(self, state):
        return {'state': state}

    def step(self, action):
        return {'state': action}, 0.0, False, {}

    def close(self):
        self.close_calls += 1


class _FakeEnvs:

    def __init__(self, failures=0, error_type=_RandomizationError):
        self.failures = failures
        self.error_type = error_type
        self.calls = 0
        self.instances = []

    def OffScreenRenderEnv(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error_type('placement failed')
        simulator = _FakeSimulator()
        simulator.kwargs = kwargs
        self.instances.append(simulator)
        return simulator


class _FakeSuite:

    def __init__(self):
        self.values = [
            SimpleNamespace(
                name='camera_task',
                language='camera instruction',
                problem_folder='libero_10',
                bddl_file='camera_task.bddl',
            ),
            SimpleNamespace(
                name='language_task',
                language='language instruction',
                problem_folder='libero_10',
                bddl_file='language_task.bddl',
            ),
        ]
        self.n_tasks = len(self.values)

    def get_task(self, task_index):
        return self.values[task_index]

    def get_task_init_states(self, task_index):
        return (f'state-{task_index}', )


def _runtime(envs):
    suite = _FakeSuite()
    benchmark = SimpleNamespace(get_benchmark_dict=lambda:
                                {'libero_10': lambda task_order_index: suite})
    libero = SimpleNamespace(
        get_libero_path=lambda key: '/fake/libero-plus/bddl')
    classification = {
        'libero_10': (
            {
                'id': 1,
                'name': 'camera_task',
                'category': 'Camera Viewpoints',
                'difficulty_level': 3,
            },
            {
                'id': 2,
                'name': 'language_task',
                'category': 'Language Instructions',
                'difficulty_level': None,
            },
        ),
    }
    return _LiberoPlusRuntime(
        benchmark=benchmark,
        libero=libero,
        envs=envs,
        classification=classification,
        randomization_error=_RandomizationError,
    )


def test_official_suite_task_counts_are_fixed():
    assert LIBERO_PLUS_SUITE_TASKS == {
        'libero_spatial': 2402,
        'libero_object': 2518,
        'libero_goal': 2591,
        'libero_10': 2519,
    }


def test_setup_attaches_zero_based_ids_and_official_classification(
        monkeypatch):
    monkeypatch.setitem(LIBERO_PLUS_SUITE_TASKS, 'libero_10', 2)
    monkeypatch.setattr(libero_plus_module, '_load_libero_plus_runtime',
                        lambda: _runtime(_FakeEnvs()))
    environment = LiberoPlusEnvironment(task_suite_name='libero_10')

    environment.setup()
    tasks = environment.tasks()

    assert [task.task_id for task in tasks] == ['0', '1']
    assert tasks[0].metadata == {
        'benchmark': 'libero',
        'benchmark_variant': 'libero_plus',
        'task_suite_name': 'libero_10',
        'task_index': 0,
        'task_name': 'camera_task',
        'recommended_max_episode_steps': 520,
        'classification_id': 1,
        'category': 'Camera Viewpoints',
        'difficulty_level': 3,
    }
    assert tasks[1].metadata['classification_id'] == 2
    assert tasks[1].metadata['difficulty_level'] is None
    assert environment._env is None


def test_environment_creation_retries_only_randomization_errors(monkeypatch):
    monkeypatch.setitem(LIBERO_PLUS_SUITE_TASKS, 'libero_10', 2)
    envs = _FakeEnvs(failures=2)
    monkeypatch.setattr(libero_plus_module, '_load_libero_plus_runtime',
                        lambda: _runtime(envs))
    environment = LiberoPlusEnvironment(
        task_suite_name='libero_10',
        max_env_creation_attempts=3,
        num_steps_wait=0,
    )
    environment.setup()

    environment.reset(environment.tasks()[0], episode_index=0, seed=7)

    assert envs.calls == 3
    assert len(envs.instances) == 1
    environment.close()
    assert envs.instances[0].close_calls == 1


def test_environment_creation_does_not_retry_other_errors(monkeypatch):
    monkeypatch.setitem(LIBERO_PLUS_SUITE_TASKS, 'libero_10', 2)
    envs = _FakeEnvs(failures=1, error_type=RuntimeError)
    monkeypatch.setattr(libero_plus_module, '_load_libero_plus_runtime',
                        lambda: _runtime(envs))
    environment = LiberoPlusEnvironment(task_suite_name='libero_10')
    environment.setup()

    with pytest.raises(RuntimeError, match='placement failed'):
        environment.reset(environment.tasks()[0], episode_index=0, seed=7)

    assert envs.calls == 1


def test_randomization_error_is_reraised_after_bounded_attempts(monkeypatch):
    monkeypatch.setitem(LIBERO_PLUS_SUITE_TASKS, 'libero_10', 2)
    envs = _FakeEnvs(failures=3)
    monkeypatch.setattr(libero_plus_module, '_load_libero_plus_runtime',
                        lambda: _runtime(envs))
    environment = LiberoPlusEnvironment(
        task_suite_name='libero_10', max_env_creation_attempts=2)
    environment.setup()

    with pytest.raises(_RandomizationError, match='placement failed'):
        environment.reset(environment.tasks()[0], episode_index=0, seed=7)

    assert envs.calls == 2


def test_task_order_is_fixed_and_suite_is_restricted():
    with pytest.raises(ValueError, match='task_order_index=0'):
        LiberoPlusEnvironment(task_suite_name='libero_10', task_order_index=1)
    with pytest.raises(ValueError, match='Unsupported LIBERO-Plus'):
        LiberoPlusEnvironment(task_suite_name='libero_90')


def test_setup_reports_actual_task_count_and_closes_runtime(monkeypatch):
    monkeypatch.setitem(LIBERO_PLUS_SUITE_TASKS, 'libero_10', 3)
    monkeypatch.setattr(libero_plus_module, '_load_libero_plus_runtime',
                        lambda: _runtime(_FakeEnvs()))
    environment = LiberoPlusEnvironment(task_suite_name='libero_10')

    with pytest.raises(ValueError, match=r'exposes 2 tasks; expected 3'):
        environment.setup()

    assert environment._runtime is None
    assert environment._suite is None
    assert environment._tasks == ()


def test_classification_validation_requires_all_official_rows(monkeypatch):
    monkeypatch.setattr(
        libero_plus_module, 'LIBERO_PLUS_SUITE_TASKS', {
            'libero_spatial': 1,
            'libero_object': 1,
            'libero_goal': 1,
            'libero_10': 1,
        })
    row = {
        'id': 1,
        'name': 'task',
        'category': 'Camera Viewpoints',
        'difficulty_level': None,
    }

    classification = _validate_classification(
        {suite: [row]
         for suite in libero_plus_module.LIBERO_PLUS_SUITE_TASKS})

    assert set(classification) == set(
        libero_plus_module.LIBERO_PLUS_SUITE_TASKS)


def test_runtime_import_is_lazy_and_has_actionable_error(monkeypatch):
    monkeypatch.setattr(libero_plus_module, 'configure_mujoco_egl_defaults',
                        lambda: None)

    def missing_import(name):
        raise ModuleNotFoundError(f'No module named {name!r}', name=name)

    monkeypatch.setattr(libero_plus_module.importlib, 'import_module',
                        missing_import)
    environment = LiberoPlusEnvironment(task_suite_name='libero_10')

    with pytest.raises(
            ModuleNotFoundError,
            match=r'bash scripts/install_libero_plus\.sh') as error:
        environment.setup()

    assert error.value.__cause__.name == 'robosuite'
