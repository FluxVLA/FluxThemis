"""Exercise slot execution and worker capability messages over real ZMQ."""

import json
import threading
from contextlib import nullcontext
from pathlib import Path
from test.test_engines.helpers import RouterSocket, make_coordinator, make_jobs
from threading import Event
from unittest.mock import Mock

import pytest

from fluxthemis.engines.runners import worker
from fluxthemis.engines.runners.coordinator import \
    DistributedEvaluationCoordinator
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings)
from fluxthemis.engines.runners.resume import validate_journal
from fluxthemis.engines.runners.worker import _SimulatorWorker
from fluxthemis.environments import MockVectorEnvironment
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.reporters import MockEvaluationReporter
from fluxthemis.structures import CaseRejection, EpisodeInterruption, TaskSpec
from fluxthemis.transports.zmq_protocol import (decode_frame, decode_header,
                                                encode_header)


class _ExecutionEnvironment(MockVectorEnvironment):

    def __init__(self, directory, reject_phase, requires_single_task,
                 task_capacities, **kwargs):
        super().__init__(**kwargs)
        self.directory = Path(directory)
        self.reject_phase = reject_phase
        self.requires_single_task = requires_single_task
        self.task_capacities = task_capacities
        self.read_task_capacities = set()
        for index, environment in enumerate(self.environments):
            environment.success_after = index * 2 + 1

    def task_capacity(self, task):
        assert all(environment.is_setup for environment in self.environments)
        assert task.task_id not in self.read_task_capacities
        self.read_task_capacities.add(task.task_id)
        return self.task_capacities.get(task.task_id, self.num_envs)

    def reset_batch(self, requests):
        self.requests = requests
        with (self.directory / 'resets.jsonl').open('a') as stream:
            stream.write(
                json.dumps([
                    None if request is None else
                    [request.episode_index, request.environment_case_index]
                    for request in requests
                ]) + '\n')
        results = super().reset_batch(requests)
        if self._interrupts('reset'):
            return self._discard_batch(requests)
        for slot, (request, result) in enumerate(zip(requests, results)):
            if request is not None and self._reject_at(request) == 'reset':
                results[slot] = CaseRejection({
                    **result.info, 'reason':
                    'bad initial layout'
                })
        return results

    def step_batch(self, actions):
        if self._interrupts('step'):
            return self._discard_batch(actions)
        results = super().step_batch(actions)
        for slot, (request, result) in enumerate(zip(self.requests, results)):
            if result is not None and self._reject_at(request) == 'step':
                results[slot] = CaseRejection({
                    **result.info, 'reason':
                    'broken physics'
                })
        return results

    def _interrupts(self, phase):
        return self.reject_phase == f'interrupt_{phase}' and any(
            request is not None and request.environment_case_index == 0
            for request in self.requests)

    def _discard_batch(self, active):
        results = []
        for request, value in zip(self.requests, active):
            if value is None:
                results.append(None)
            elif request.environment_case_index == 0:
                results.append(CaseRejection())
            else:
                results.append(EpisodeInterruption({'reason': 'partial call'}))
        return results

    def _reject_at(self, request):
        if request.environment_case_index not in (0, 5, 6):
            return None
        if self.reject_phase == 'mixed':
            return 'reset' if request.environment_case_index == 5 else 'step'
        return self.reject_phase


def _mock_worker_process(**kwargs):
    """Supply CPU components while retaining the actual worker process loop."""
    config = json.loads(Path(kwargs['config_path']).read_text())
    evaluation = dict(
        environment=dict(
            type=_ExecutionEnvironment,
            directory=config['directory'],
            reject_phase=config['reject_phase'],
            requires_single_task=config['requires_single_task'],
            task_capacities={'mock': min(config['capacity'], 5)},
            num_envs=config['capacity']),
        policy_client=dict(type='MockPolicyClient', chunk_size=1),
        max_episode_steps=10)
    worker.load_distributed_evaluation_config = lambda *a, **kw: evaluation
    worker._run_simulator_worker(**kwargs)


@pytest.mark.parametrize('capacity', [1, 2, 3, 8])
@pytest.mark.parametrize(
    'reject_phase',
    [None, 'reset', 'step', 'mixed', 'interrupt_reset', 'interrupt_step'])
@pytest.mark.parametrize('requires_single_task', [False, True])
def test_supervised_slot_execution(tmp_path, monkeypatch, capacity,
                                   reject_phase, requires_single_task):
    endpoint = f'ipc://{tmp_path}/workers.sock'
    config = tmp_path / 'config.json'
    config.write_text(
        json.dumps(
            dict(
                directory=str(tmp_path),
                capacity=capacity,
                reject_phase=reject_phase,
                requires_single_task=requires_single_task)))
    monkeypatch.setattr(worker, '_run_simulator_worker', _mock_worker_process)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '0')
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    supervisor = worker.SimulatorWorkerSupervisor(
        str(config),
        endpoint,
        'simulator-worker-0',
        'deployment',
        '0',
        10,
        restart_on_failure=False)
    errors = []

    def supervise():
        try:
            supervisor.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=supervise, daemon=True)
    coordinator = DistributedEvaluationCoordinator(
        endpoint,
        1,
        startup_timeout_s=10,
        episode_timeout_s=10,
        simulator_restart_limit=0,
        deployment_id='deployment')
    monkeypatch.setattr(coordinator, '_install_signal_handlers', lambda: None)
    reporter = MockEvaluationReporter()
    with EvaluationRun(
            EvaluationSettings(episodes_per_task=5), SuccessRateEvaluator(),
            reporter) as run:
        thread.start()
        try:
            summary = coordinator.run(run)
            assert summary['run']['completed_episodes'] == 5
        finally:
            thread.join(timeout=15)
    assert not thread.is_alive()
    assert not errors
    slot = coordinator._worker_slots['simulator-worker-0']
    assert slot.capacity == capacity
    assert slot.requires_single_task == requires_single_task
    assert slot.execution_policy.task_capacities == ({
        'mock': 5
    } if capacity > 5 else {})
    assert all(task.metadata == {} for task in slot.task_manifest)
    ends = [
        payload for kind, payload in reporter.events if kind == 'episode_end'
    ]
    assert len({end['episode_id'] for end in ends}) == 5
    interrupts = reject_phase in ('interrupt_reset', 'interrupt_step')
    expected_cases = ([1, 2, 3, 4, 5] if interrupts else
                      [1, 2, 3, 4, 7] if reject_phase else list(range(5)))
    assert sorted(end['environment_case_index']
                  for end in ends) == expected_cases
    assert sorted(end['policy_seed'] for end in ends) == list(range(7, 12))
    assert all(end['success'] for end in ends)
    assert coordinator._scheduling_state.restart_counts_by_episode_id == {}
    resets = [
        json.loads(line)
        for line in (tmp_path / 'resets.jsonl').read_text().splitlines()
    ]
    cases = [
        request[1] for batch in resets for request in batch
        if request is not None
    ]
    if interrupts:
        # Only the faulty case gets a replacement. Peers repeat their exact
        # case in the same process, ahead of jobs that were still pending.
        assert sorted(cases) == sorted(
            list(range(6)) + list(range(1, min(capacity, 5))))
        for case in range(1, min(capacity, 5)):
            attempts = [
                request for batch in resets for request in batch
                if request is not None and request[1] == case
            ]
            assert attempts[0] == attempts[1]
        assert slot.generation == 0
        starts = [
            payload for kind, payload in reporter.events
            if kind == 'episode_start'
        ]
        assert len(starts) == 5 + int(reject_phase == 'interrupt_step')
    else:
        assert sorted(cases) == list(range(8 if reject_phase else 5))
    rows = [
        dict(
            event_type=kind,
            payload=body,
            sequence=index,
            run_session_id=reporter.run_session_id)
        for index, (kind, body) in enumerate(reporter.events, 1)
    ]
    _, _, completed = validate_journal(rows)
    assert len(completed) == 5
    if reject_phase in ('step', 'mixed'):
        replacement = next(
            row for row in rows
            if 'replaces_environment_case_index' in row['payload'])
        replacement['payload'].pop('replaces_environment_case_index')
        with pytest.raises(ValueError, match='Duplicate episode_start'):
            validate_journal(rows)


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

        monkeypatch.setattr(worker._SimulatorWorker, '_receive_assignment',
                            broken_receive)
    elif failure == 'episode':

        def broken_execute(self, **kwargs):
            raise RuntimeError('synthetic episode failure')

        monkeypatch.setattr(worker._SimulatorWorker, '_execute_assignment',
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


@pytest.mark.parametrize('capacity', [1, 2, 10])
@pytest.mark.parametrize('requires_single_task', [False, True])
def test_worker_coordinator_roundtrip_waits_for_assignment_end(
        monkeypatch, capacity, requires_single_task):
    # Exercise the actual wire codec, worker execution and coordinator events
    # without launching a simulator or connecting a model service.
    monkeypatch.setattr('fluxthemis.engines.runners.worker._episode_timeout',
                        lambda timeout: nullcontext())
    count = 25 if capacity == 10 else 3
    jobs = make_jobs(count, ('a', 'b'))
    coordinator, slot = make_coordinator(capacity, jobs, requires_single_task)
    socket = RouterSocket()
    worker = _SimulatorWorker(
        config_path='',
        rollout_endpoint='',
        worker_id=slot.worker_id,
        deployment_id='deployment',
        generation=0,
        device='0',
        episode_timeout_s=10,
        config_overrides={},
        ready_event=None,
        stop_received=None)
    environment = MockVectorEnvironment(capacity, success_after=1)
    environment.setup()
    run = EvaluationRun(
        EvaluationSettings(episodes_per_task=count), SuccessRateEvaluator(),
        MockEvaluationReporter())
    with run:
        run.start([TaskSpec('a'), TaskSpec('b')], len(jobs))
        batch_sizes = []
        while coordinator._scheduling_state.pending_jobs:
            coordinator._dispatch_jobs()
            socket.incoming = coordinator._worker_router.sent[-1][1:]
            assignment = worker._receive_assignment(socket)
            assert 0 < len(assignment.jobs) <= capacity
            batch_sizes.append(len(assignment.jobs))
            if requires_single_task:
                assert len({job.task.task_id for job in assignment.jobs}) == 1
            socket.sent.clear()
            deadline = slot.episode_deadline
            worker._execute_assignment(
                socket=socket,
                environment=environment,
                execution_policy=slot.execution_policy,
                policy_clients=[MockPolicyClient() for _ in range(capacity)],
                evaluation_cfg={'max_episode_steps': 10},
                assignment=assignment,
                save_videos=False,
                video_dir=None,
                video_fps=20)
            assert decode_header(
                socket.sent[-1][0])['type'] == 'assignment_end'
            for frames in socket.sent[:-1]:
                coordinator._worker_router.incoming = [
                    slot.routing_id, *frames
                ]
                coordinator._receive_worker_message(run)
                assert not slot.is_idle
                assert slot.episode_deadline == deadline
            assert coordinator._active_episode_progress() is None
            coordinator._worker_router.incoming = [
                slot.routing_id, *socket.sent[-1]
            ]
            coordinator._receive_worker_message(run)
            assert slot.is_idle
        expected_sizes = []
        group_lengths = [count] * 2 if requires_single_task else [count * 2]
        for length in group_lengths:
            expected_sizes.extend([capacity] * (length // capacity))
            expected_sizes.extend([length % capacity] *
                                  bool(length % capacity))
        assert batch_sizes == expected_sizes
        assert coordinator._scheduling_state.is_complete
        results = coordinator._scheduling_state.ordered_results()
        assert [r.policy_seed
                for r in results] == list(range(7, 7 + len(jobs)))
        assert run.finish(
            results[-1])['run']['completed_episodes'] == len(jobs)


def test_assignment_identity_is_checked_at_worker_boundary():
    worker = _SimulatorWorker(
        config_path='',
        rollout_endpoint='',
        worker_id='worker',
        deployment_id='deployment',
        generation=0,
        device='0',
        episode_timeout_s=10,
        config_overrides={},
        ready_event=None,
        stop_received=None)
    coordinator, slot = make_coordinator(1, make_jobs(1))
    coordinator._dispatch_jobs()
    socket = RouterSocket()
    socket.incoming = coordinator._worker_router.sent[-1][1:]
    header = decode_header(socket.incoming[0])
    body = decode_frame(socket.incoming[1])
    assert body['jobs'][0]['episode_id'] == 'run:1'
    header.update(worker_id='worker', assignment_id='wrong')
    socket.incoming[0] = encode_header(header.pop('type'), **header)
    with pytest.raises(RuntimeError, match='assignment_id mismatch'):
        worker._receive_assignment(socket)
