"""Exercise process supervision and coordinator decisions over real ZMQ."""

import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from fluxthemis.engines.runners import worker
from fluxthemis.engines.runners.coordinator import \
    DistributedEvaluationCoordinator
from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_episode)
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings,
                                                   EvaluationStatus, utc_now)
from fluxthemis.engines.runners.resume import build_bundle
from fluxthemis.environments import (EnvironmentCaseRejected, MockEnvironment,
                                     MockVectorEnvironment)
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.model_clients import MockPolicyClient
from fluxthemis.reporters import MockEvaluationReporter


class _CrashEnvironment(MockVectorEnvironment):

    def __init__(self, directory, mode, exitcode, **kwargs):
        super().__init__(**kwargs)
        self.directory = Path(directory)
        self.mode = mode
        self.exitcode = exitcode
        if self.num_envs > 1:
            self.environments[1].success_after = 3

    def reset_batch(self, requests):
        self.episode_index = requests[0].episode_index
        with (self.directory / 'resets.jsonl').open('a') as stream:
            stream.write(
                json.dumps([
                    None if r is
                    None else [r.episode_index, r.environment_case_index]
                    for r in requests
                ]) + '\n')
        if self.mode == 'batch_rejection':
            slots = [
                i for i, r in enumerate(requests)
                if r is not None and r.environment_case_index == 0
            ]
            if slots:
                raise EnvironmentCaseRejected(
                    'bad initial layout', slots=slots)
        return super().reset_batch(requests)

    def step_batch(self, actions):
        if self.mode == 'scalar_retry':
            marker = self.directory / f'crashed-{self.episode_index}'
            if not marker.exists():
                marker.touch()
                os._exit(9)
        elif ((self.mode == 'batch_failure' and actions[0] is None)
              or (self.mode == 'batch_exhausted' and
                  (actions[0] is None or
                   (self.directory / 'reported').exists()))):
            # Crash only after the coordinator acknowledges the first result.
            deadline = time.monotonic() + 5
            while not (self.directory / 'reported').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('First episode was not reported')
                time.sleep(0.01)
            os._exit(self.exitcode)
        return super().step_batch(actions)


def _mock_worker_process(**kwargs):
    """Supply CPU components while retaining the real worker process loop."""
    config = json.loads(Path(kwargs['config_path']).read_text())
    directory = Path(config['directory'])
    with (directory / 'generations').open('a') as stream:
        stream.write(f'{kwargs["generation"]}\n')
    evaluation = dict(
        episode_batch_size=config['capacity'],
        environment=dict(
            type=_CrashEnvironment,
            directory=str(directory),
            mode=config['mode'],
            exitcode=config['exitcode'],
            num_envs=config['capacity'],
            success_after=1),
        policy_client=dict(type='MockPolicyClient', chunk_size=1),
        max_episode_steps=5)
    worker.load_distributed_evaluation_config = lambda *a, **kw: evaluation
    # Spawn imports this module afresh, so the production target is unchanged
    # here even though the parent substitutes this wrapper as its spawn target.
    worker._run_simulator_worker(**kwargs)


class _Reporter(MockEvaluationReporter):

    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def report_evaluation(self, event_type, payload):
        result = super().report_evaluation(event_type, payload)
        if event_type == 'episode_end':
            (self.directory / 'reported').touch()
        return result


@pytest.mark.parametrize('mode,capacity,episodes,generations,exitcode', [
    ('batch_complete', 2, 5, [0], 9),
    ('batch_failure', 2, 2, [0, 1], 9),
    ('batch_failure', 2, 2, [0, 1], 0),
    ('batch_exhausted', 2, 2, [0, 1], 9),
    ('batch_rejection', 2, 3, [0], 9),
    ('batch_resume', 2, 3, [0], 9),
    ('scalar_retry', 1, 4, [0, 1, 2, 3, 4], 9),
])
def test_supervised_evaluation(tmp_path, monkeypatch, mode, capacity, episodes,
                               generations, exitcode):
    endpoint = f'ipc://{tmp_path}/workers.sock'
    config = tmp_path / 'config.json'
    config.write_text(
        json.dumps(
            dict(
                directory=str(tmp_path),
                mode=mode,
                capacity=capacity,
                exitcode=exitcode)))
    monkeypatch.setattr(worker, '_run_simulator_worker', _mock_worker_process)
    monkeypatch.setattr(worker, '_SIMULATOR_STARTUP_TIMEOUT_S', 10)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '0')
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    supervisor = worker.SimulatorWorkerSupervisor(
        str(config), endpoint, 'simulator-worker-0', 'deployment', '0', 10)
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
        simulator_restart_limit=1,
        deployment_id='deployment')
    monkeypatch.setattr(coordinator, '_install_signal_handlers', lambda: None)
    reporter = _Reporter(tmp_path)
    resume_path = None
    if mode == 'batch_resume':
        resume_path = str(_batch_resume_input(tmp_path))
    run = EvaluationRun(
        EvaluationSettings(
            episodes_per_task=episodes,
            episode_batch_size=capacity,
            resume_path=resume_path), SuccessRateEvaluator(), reporter)
    thread.start()
    try:
        with run:
            if mode == 'batch_exhausted':
                with pytest.raises(RuntimeError, match='restart limit'):
                    coordinator.run(run)
            else:
                summary = coordinator.run(run)
                assert summary['run']['completed_episodes'] == episodes
    finally:
        thread.join(timeout=15)
    assert not thread.is_alive()
    assert list(map(int,
                    (tmp_path /
                     'generations').read_text().splitlines())) == generations
    end = reporter.events[-1]
    assert end[0] == 'run_end'
    if mode == 'batch_exhausted':
        assert len(errors) == 1
        assert f'stopped after exit code {exitcode}' in str(errors[0])
        assert end[1]['status'] == 'failed'
        assert end[1]['completed_episodes'] == 1
    else:
        assert not errors
        assert end[1]['status'] == 'finished'
    ends = [
        payload for kind, payload in reporter.events if kind == 'episode_end'
    ]
    assert len({end['episode_id'] for end in ends}) == len(ends)
    if mode == 'batch_failure':
        resets = [
            json.loads(line)
            for line in (tmp_path / 'resets.jsonl').read_text().splitlines()
        ]
        assert resets == [[[0, 0], [1, 1]], [[1, 1], None]]
        assert len(ends) == 2
    if mode == 'batch_resume':
        resets = [
            json.loads(line)
            for line in (tmp_path / 'resets.jsonl').read_text().splitlines()
        ]
        assert resets == [[[1, 5], [2, 2]]]
        assert sorted(end['episode_index'] for end in ends) == [0, 1, 2]
    if mode == 'batch_rejection':
        assert sorted(end['environment_case_index']
                      for end in ends) == [1, 2, 3]
        assert {
            end['episode_index']: end['policy_seed']
            for end in ends
        } == {
            0: 7,
            1: 8,
            2: 9
        }
    if mode == 'scalar_retry':
        state = coordinator._scheduling_state
        assert list(state.restart_counts_by_episode_id.values()) == [1] * 4


def test_startup_failure_stops_before_exceeding_restart_budget(
        tmp_path, monkeypatch):
    # A missing config fails before readiness; use the same real supervisor
    # handshake to check the independent startup retry budget.
    endpoint = f'ipc://{tmp_path}/workers.sock'
    coordinator = DistributedEvaluationCoordinator(
        endpoint,
        1,
        startup_timeout_s=10,
        simulator_restart_limit=0,
        deployment_id='deployment')
    monkeypatch.setattr(coordinator, '_install_signal_handlers', lambda: None)
    monkeypatch.setattr(worker, '_SIMULATOR_STARTUP_TIMEOUT_S', 10)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '0')
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    errors = []

    def supervise():
        try:
            worker.SimulatorWorkerSupervisor(
                str(tmp_path / 'missing.py'), endpoint, 'simulator-worker-0',
                'deployment', '0', 10).run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=supervise, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match='startup restart limit'):
            coordinator.run(None)
    finally:
        thread.join(timeout=15)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert coordinator._worker_slots['simulator-worker-0'].generation == 0


def _batch_resume_input(directory):
    reporter = MockEvaluationReporter(run_session_id='previous-session')
    task = MockEnvironment().tasks()[0]
    jobs = build_episode_jobs([task], 3, 7, reporter.run_session_id)
    with EvaluationRun(
            EvaluationSettings(episodes_per_task=3, episode_batch_size=2),
            SuccessRateEvaluator(), reporter) as run:
        run.start([task], 3)
        env = MockEnvironment(success_after=1)
        env.setup()
        result = execute_episode(
            env,
            MockPolicyClient(),
            jobs[0],
            5,
            None,
            True,
            on_episode_started=lambda: run.on_episode_start(jobs[
                0].episode_start_payload(utc_now())))
        run.on_episode_end(result, jobs[0].episode_id)
        # A replacement layout was already selected and started before the
        # interruption. Resume must preserve it along with canonical seeds.
        pending = replace(jobs[1], environment_case_index=5)
        run.on_episode_start(pending.episode_start_payload(utc_now()))
        run.fail(EvaluationStatus.INTERRUPTED, None, result,
                 dict(error_type='KeyboardInterrupt', message='interrupted'))
    journal = directory / 'events.jsonl'
    journal.write_text(''.join(
        json.dumps(
            dict(
                event_type=kind,
                payload=body,
                sequence=index,
                run_session_id=reporter.run_session_id,
                recorded_at=utc_now())) + '\n'
        for index, (kind, body) in enumerate(reporter.events, 1)))
    bundle = build_bundle(directory)
    assert [job['started'] for job in bundle['jobs']] == [True, True, False]
    path = directory / 'resume.json'
    path.write_text(json.dumps(bundle))
    return path


def test_interrupted_resume_preserves_inherited_results_and_initial_cases(
        tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    bundle = _batch_resume_input(original)
    reporter = MockEvaluationReporter(run_session_id='resumed-session')
    with EvaluationRun(
            EvaluationSettings(
                episodes_per_task=3,
                episode_batch_size=2,
                resume_path=str(bundle)), SuccessRateEvaluator(),
            reporter) as run:
        run.start(MockEnvironment().tasks(), 3)
        # Interrupt before even replaying the first completed episode.
        run.fail(EvaluationStatus.INTERRUPTED, None, None,
                 dict(error_type='KeyboardInterrupt', message='interrupted'))
    resumed = tmp_path / 'resumed'
    resumed.mkdir()
    (resumed / 'resume_input.json').write_text(bundle.read_text())
    (resumed / 'events.jsonl').write_text(''.join(
        json.dumps(
            dict(
                event_type=kind,
                payload=body,
                sequence=index,
                run_session_id=reporter.run_session_id,
                recorded_at=utc_now())) + '\n'
        for index, (kind, body) in enumerate(reporter.events, 1)))
    recovered = build_bundle(resumed)
    assert len(recovered['completed']) == 1
    assert [job['environment_case_index']
            for job in recovered['jobs']] == [0, 5, 2]
    assert [job['started'] for job in recovered['jobs']] == [True, True, False]
