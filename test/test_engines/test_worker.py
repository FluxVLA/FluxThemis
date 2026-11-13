"""Bounded CPU checks for supervisor shutdown and restart semantics."""

from threading import Event
from unittest.mock import Mock

import pytest

from fluxthemis.engines.runners import worker


class _Process:

    def __init__(self, kwargs, outcome):
        self.kwargs = kwargs
        self.exitcode, self.stopped = outcome

    def start(self):
        self.kwargs['ready_event'].set()
        if self.stopped:
            self.kwargs['stop_received'].set()

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


@pytest.mark.parametrize('outcomes', [
    [(0, True)],
    [(0, False), (0, True)],
    [(1, True), (0, True)],
    [(-14, False), (0, False), (0, True)],
])
def test_supervisor_requires_shutdown_ack_and_fresh_generation_flags(
        monkeypatch, outcomes):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(worker.time, 'sleep', lambda _: None)
    processes = []

    def process(**values):
        child = _Process(values['kwargs'], outcomes[len(processes)])
        processes.append(child)
        return child

    context = Mock(Event=Event, Process=process)
    monkeypatch.setattr(worker.multiprocessing, 'get_context',
                        lambda _: context)
    supervisor = worker.SimulatorWorkerSupervisor(
        'config.py',
        'tcp://127.0.0.1:15558',
        'worker-0',
        'deployment',
        '-1',
        episode_timeout_s=1)
    supervisor.run()

    assert len(processes) == len(outcomes)
    assert [p.kwargs['generation']
            for p in processes] == list(range(len(outcomes)))
    assert len({id(p.kwargs['stop_received'])
                for p in processes}) == len(outcomes)
    assert len({id(p.kwargs['ready_event'])
                for p in processes}) == len(outcomes)


def test_zero_restart_limit_does_not_launch_replacement(monkeypatch):
    processes = []

    def process(**values):
        child = _Process(values['kwargs'], (1, False))
        processes.append(child)
        return child

    monkeypatch.setattr(worker.multiprocessing, 'get_context',
                        lambda _: Mock(Event=Event, Process=process))
    supervisor = worker.SimulatorWorkerSupervisor(
        'config.py',
        'tcp://127.0.0.1:15558',
        'worker-0',
        'deployment',
        '-1',
        episode_timeout_s=1,
        restart_on_failure=False)
    with pytest.raises(RuntimeError, match='restarting is disabled'):
        supervisor.run()
    assert len(processes) == 1


def test_terminate_escalates_only_when_process_stays_alive():
    process = Mock()
    process.is_alive.side_effect = [True, True]
    worker._terminate_process(process)
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.join.call_args_list == [((), {'timeout': 10.0}), ((), {})]


@pytest.mark.parametrize('field,value', [
    ('deployment_id', 'wrong'),
    ('worker_id', 'wrong'),
    ('generation', 0),
])
def test_worker_rejects_commands_for_other_identity(field, value):
    child = worker._SimulatorWorker(
        config_path='config.py',
        rollout_endpoint='tcp://127.0.0.1:15558',
        worker_id='worker-0',
        deployment_id='deployment',
        generation=1,
        device='-1',
        episode_timeout_s=1,
        config_overrides={},
        ready_event=Event(),
        stop_received=Event())
    header = dict(child._message_identity)
    header[field] = value
    with pytest.raises(RuntimeError, match='mismatch'):
        child._validate_header(header)


@pytest.mark.parametrize('startup_timeout,expected_generations,expected_time',
                         [
                             (None, 2, 180),
                             (120, 2, 120),
                             (300, 1, 250),
                             (900, 1, 250),
                         ])
def test_worker_startup_deadline_uses_configured_budget(
        monkeypatch, startup_timeout, expected_generations, expected_time):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '')
    now = [0.0]
    monkeypatch.setattr(worker.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(worker.time, 'sleep', lambda _: None)

    class ReadyEvent(Event):

        def wait(self, timeout=None):
            now[0] += timeout
            if now[0] >= 250:
                self.set()
            return self.is_set()

    class ColdProcess:

        def __init__(self, values):
            self.values = values
            self.exitcode = None
            self.alive = True

        def start(self):
            pass

        def is_alive(self):
            if self.values['ready_event'].is_set():
                self.values['stop_received'].set()
                self.exitcode = 0
                self.alive = False
            return self.alive

        def terminate(self):
            self.alive = False
            self.exitcode = -15

        def join(self, timeout=None):
            pass

    processes = []

    def process(**values):
        child = (
            ColdProcess(values['kwargs']) if not processes else _Process(
                values['kwargs'], (0, True)))
        processes.append(child)
        return child

    context = Mock(Event=ReadyEvent, Process=process)
    monkeypatch.setattr(worker.multiprocessing, 'get_context',
                        lambda _: context)
    options = {} if startup_timeout is None else {
        'startup_timeout_s': startup_timeout
    }
    supervisor = worker.SimulatorWorkerSupervisor(
        'config.py',
        'tcp://127.0.0.1:15558',
        'worker-0',
        'deployment',
        '-1',
        episode_timeout_s=1,
        **options)
    supervisor.run()
    assert len(processes) == expected_generations
    assert now[0] == expected_time


@pytest.mark.parametrize('value', [0, -1, True, '900', None])
def test_worker_rejects_invalid_startup_timeout(value):
    with pytest.raises(ValueError, match='startup_timeout_s'):
        worker.SimulatorWorkerSupervisor(
            'config.py',
            'tcp://127.0.0.1:15558',
            'worker-0',
            'deployment',
            '-1',
            episode_timeout_s=1,
            startup_timeout_s=value)


@pytest.mark.parametrize('startup_timeout', [45.0, 900.0])
def test_launcher_forwards_startup_timeout(monkeypatch, startup_timeout):
    from fluxthemis.engines.runners import launcher

    monkeypatch.setenv('FLUXTHEMIS_DEPLOYMENT_ID', 'startup-budget-test')
    supervisor = Mock()
    monkeypatch.setattr(launcher, 'SimulatorWorkerSupervisor', supervisor)
    launcher.run_distributed_evaluation_rank(
        config_path='config.py',
        device='-1',
        rollout_bind='tcp://127.0.0.1:15558',
        rollout_endpoint='tcp://127.0.0.1:15558',
        rank=1,
        world_size=2,
        startup_timeout_s=startup_timeout)
    assert supervisor.call_args.kwargs['startup_timeout_s'] == startup_timeout
    supervisor.return_value.run.assert_called_once_with()


@pytest.mark.parametrize('state', [
    'init', 'idle', 'episode', 'timeout', 'exit_zero', 'exit_nonzero',
    'shutdown'
])
def test_real_child_zero_restarts_and_reaping(monkeypatch, tmp_path, state):
    import multiprocessing
    import os
    import time
    context = multiprocessing.get_context('fork')
    monkeypatch.setattr(worker.multiprocessing, 'get_context',
                        lambda _: context)
    record = tmp_path / 'children'

    def child(**kwargs):
        record.write_text(f"{os.getpid()}:{kwargs['generation']}")
        if state not in ('init', 'timeout'):
            kwargs['ready_event'].set()
        if state == 'shutdown':
            kwargs['stop_received'].set()
            return
        if state == 'timeout':
            time.sleep(10)
        else:
            os._exit(0 if state == 'exit_zero' else 1)

    monkeypatch.setattr(worker, '_run_simulator_worker', child)
    supervisor = worker.SimulatorWorkerSupervisor(
        'config.py',
        'tcp://127.0.0.1:15558',
        'worker',
        'deployment',
        '-1',
        episode_timeout_s=1,
        startup_timeout_s=0.1,
        restart_on_failure=False)
    if state == 'shutdown':
        supervisor.run()
    else:
        with pytest.raises(RuntimeError, match='restarting is disabled'):
            supervisor.run()
    pid, generation = record.read_text().split(':')
    assert generation == '0'
    assert not os.path.exists(f'/proc/{pid}')
    assert all(p.pid != int(pid) for p in multiprocessing.active_children())


@pytest.mark.parametrize('limit,expected', [(0, False), (3, True), (7, True)])
def test_launcher_passes_only_restart_boolean(monkeypatch, limit, expected):
    from fluxthemis.engines.runners import launcher
    supervisor = Mock()
    monkeypatch.setattr(launcher, 'SimulatorWorkerSupervisor', supervisor)
    launcher.run_distributed_evaluation_rank(
        config_path='config.py',
        device='-1',
        rollout_bind='tcp://127.0.0.1:15558',
        rollout_endpoint='tcp://127.0.0.1:15558',
        rank=1,
        world_size=2,
        simulator_restart_limit=limit)
    assert supervisor.call_args.kwargs['restart_on_failure'] is expected
    assert 'restart_limit' not in supervisor.call_args.kwargs


@pytest.mark.parametrize('failure', ['initialization', 'idle', 'episode'])
def test_real_launcher_coordinator_and_worker_exit_bounded(
        monkeypatch, tmp_path, failure):
    import multiprocessing
    import socket

    from fluxthemis.engines.runners import config as config_module
    from fluxthemis.engines.runners import launcher
    from fluxthemis.environments.mock import MockEnvironment
    from fluxthemis.model_clients.mock import MockPolicyClient
    from fluxthemis.reporters.mock import MockEvaluationReporter
    context = multiprocessing.get_context('fork')
    monkeypatch.setattr(worker.multiprocessing, 'get_context',
                        lambda _: context)
    previous = {p.pid for p in multiprocessing.active_children()}

    class Reporter(MockEvaluationReporter):

        @property
        def evaluation_report_run_dir(self):
            return str(tmp_path / 'reported')

    monkeypatch.setattr(config_module, 'build_evaluation_reporter_from_cfg',
                        lambda _: Reporter())
    environment = MockEnvironment(tasks=['A'], success_after=2)
    if failure == 'initialization':

        def broken_setup():
            raise RuntimeError('synthetic initialization failure')

        environment.setup = broken_setup
    monkeypatch.setattr(worker, 'build_environment_from_cfg',
                        lambda _: environment)
    monkeypatch.setattr(worker, 'build_policy_client_from_cfg',
                        lambda _: MockPolicyClient(action_dim=8, chunk_size=1))
    if failure == 'idle':

        def broken_receive(self, socket):
            raise RuntimeError('synthetic idle failure')

        monkeypatch.setattr(worker._SimulatorWorker, '_receive_job',
                            broken_receive)
    elif failure == 'episode':

        def broken_execute(self, **kwargs):
            raise RuntimeError('synthetic episode failure')

        monkeypatch.setattr(worker._SimulatorWorker, '_execute_job',
                            broken_execute)
    config = tmp_path / 'config.py'
    config.write_text('themis=dict(transport=dict(), evaluation=dict(\n'
                      'environment=dict(type="MockEnvironment"),\n'
                      'policy_client=dict(type="MockPolicyClient"),\n'
                      'reporter=dict(type="MockEvaluationReporter"),\n'
                      'evaluator=dict(type="SuccessRateEvaluator"),\n'
                      'episodes_per_task=1, max_episode_steps=3))\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        endpoint = f'tcp://127.0.0.1:{sock.getsockname()[1]}'
    with pytest.raises(RuntimeError, match='restarting is disabled'):
        launcher.run_distributed_evaluation_rank(
            config_path=str(config),
            device='-1',
            rollout_bind=endpoint,
            rollout_endpoint=endpoint,
            rank=0,
            world_size=1,
            startup_timeout_s=3,
            episode_timeout_s=1,
            simulator_restart_limit=0,
            deployment_id='real-process-test')
    assert {p.pid for p in multiprocessing.active_children()} == previous
