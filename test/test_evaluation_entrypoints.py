"""Check the public launcher and recovery path outside package-only tests."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from mmengine import Config

from fluxthemis.engines.runners.assignment import (EpisodeAssignment,
                                                   ExecutionPolicy)
from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_assignment)
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings,
                                                   EvaluationStatus, utc_now)
from fluxthemis.environments import MockVectorEnvironment
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.model_clients import MockPolicyClient
from fluxthemis.reporters import MockEvaluationReporter
from fluxthemis.structures import TaskSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def resume(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return importlib.import_module('evaluation.resume')


def _config(capacity):
    evaluation = dict(
        environment=dict(type='MockVectorEnvironment', num_envs=capacity),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(type='SuccessRateEvaluator'),
        episodes_per_task=2,
        max_episode_steps=5)
    return Config(
        dict(
            themis=dict(
                evaluation=evaluation,
                transport=dict(image_keys=['image'], state_keys=['state']))))


@pytest.mark.parametrize('dry_run', [False, True])
def test_launcher_accepts_environment_capacity_configuration(
        tmp_path, dry_run):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    command = [sys.executable, str(ROOT / 'scripts/run_evaluation.py')]
    if dry_run:
        path = tmp_path / 'evaluation.py'
        _config(2).dump(str(path))
        pointer = tmp_path / 'result_pointer.json'
        command += [
            '--dry-run', '--config',
            str(path), '--gpu-ids', '0', '--server-endpoint',
            'tcp://127.0.0.1:15555', '--output',
            str(tmp_path / 'output'), '--result-pointer',
            str(pointer)
        ]
    else:
        command += ['--help']
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    if dry_run:
        output = Path(json.loads(pointer.read_text())['directory'])
        frozen = Config.fromfile(str(output / 'config_snapshot.py'))
        assert frozen.themis.evaluation.environment.num_envs == 2
        assert not (output / 'simulator.log').exists()


@pytest.mark.parametrize('config_format', ['py', 'yaml'])
@pytest.mark.parametrize('timeout_s', [None, 7200.5])
def test_launcher_forwards_assignment_timeout(tmp_path, monkeypatch,
                                              config_format, timeout_s):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    launcher = importlib.import_module('run_evaluation')
    distributed = importlib.import_module('distributed_eval')
    config = tmp_path / 'evaluation.py'
    _config(8).dump(str(config))
    if config_format == 'yaml':
        config = tmp_path / 'launch.yaml'
        config.write_text('simulator:\n  config: evaluation.py\n')
    pointer = tmp_path / 'result_pointer.json'
    command = [
        'bash',
        str(ROOT / 'scripts/run_evaluation.sh'), '--config',
        str(config), '--dry-run', '--gpu-ids', '0', '--output',
        str(tmp_path / 'output'), '--result-pointer',
        str(pointer)
    ]
    if timeout_s is not None:
        command += ['--episode-timeout-s', str(timeout_s)]
    env = dict(
        os.environ, PYTHONPATH=str(ROOT), FLUXTHEMIS_PYTHON=sys.executable)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    output = Path(json.loads(pointer.read_text())['directory'])
    plan = json.loads((output / 'plan.json').read_text())
    assert plan['local']['episode_timeout_s'] == (3600 if timeout_s is None
                                                  else timeout_s)
    arguments = launcher.simulator_arguments(plan)
    assert arguments.count('--episode-timeout-s') == 1
    index = arguments.index('--episode-timeout-s')
    timeout_arguments = arguments[index:index + 2]
    parsed = distributed.parse_args(
        ['--config', plan['local']['frozen_config'], *timeout_arguments])
    assert parsed.episode_timeout_s == (3600.0
                                        if timeout_s is None else timeout_s)
    assert not (output / 'simulator.log').exists()


@pytest.mark.parametrize('capacity', [None, 1, 3])
def test_smoke_episode_count_uses_environment_capacity(monkeypatch, capacity):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    launcher = importlib.import_module('evaluation.config')
    options = {} if capacity is None else {
        'themis.evaluation.environment.num_envs': capacity
    }
    plan = launcher.load_config(
        ROOT / 'configs/robodojo/arx_x5.py',
        mode='smoke',
        overrides=dict(gpu_ids='0,1', workers_per_gpu=2, cfg_options=options))
    assert plan['simulator_workers'] == 4
    assert plan['local']['episodes'] == 4 * (capacity or 1)
    evaluation = plan['config'].themis.evaluation
    assert evaluation.episodes_per_task == plan['local']['episodes']
    assert evaluation.episodes_per_task_overrides[
        'put_bottles_into_dustbin'] == plan['local']['episodes']


@pytest.mark.parametrize('capacity', [1, 2])
def test_prepare_resume_preserves_slot_results(tmp_path, capacity, resume):
    source = tmp_path / 'stopped_run'
    source.mkdir()
    cfg = _config(capacity)
    snapshot = source / 'config_snapshot.py'
    cfg.dump(str(snapshot))
    local = dict(task='task', episodes=2)
    (source / 'plan.json').write_text(json.dumps(dict(local=local)))
    (source / 'recovery_manifest.json').write_text(
        json.dumps(
            dict(schema_version=2, config_sha256=resume.file_hash(snapshot))))
    reporter = MockEvaluationReporter()
    tasks = [TaskSpec('task')]
    jobs = build_episode_jobs(tasks, 2, 7, reporter.run_session_id)
    with EvaluationRun(
            EvaluationSettings(episodes_per_task=2), SuccessRateEvaluator(),
            reporter) as run:
        run.start(tasks, 2)
        environment = MockVectorEnvironment(capacity, success_after=1)
        environment.setup()
        result = execute_assignment(
            environment, [MockPolicyClient() for _ in range(capacity)],
            EpisodeAssignment([jobs[0]]),
            5,
            None,
            True,
            execution_policy=ExecutionPolicy.from_environment(environment),
            on_episode_started=lambda job: run.on_episode_start(
                job.episode_start_payload(utc_now())))[0]
        run.on_episode_end(result, jobs[0].episode_id)
        run.fail(EvaluationStatus.INTERRUPTED, None, result,
                 dict(error_type='KeyboardInterrupt', message='interrupted'))
    rows = [
        dict(
            event_type=kind,
            payload=body,
            sequence=index,
            run_session_id=reporter.run_session_id,
            recorded_at=utc_now())
        for index, (kind, body) in enumerate(reporter.events, 1)
    ]
    (source / 'events.jsonl').write_text(''.join(
        json.dumps(row) + '\n' for row in rows))
    settings = dict(
        config=cfg,
        local=local,
        simulator_workers=1,
        output_root=str(tmp_path))
    plan, lock = resume.prepare_resume(source, settings)
    try:
        bundle = resume.recovery.read_bundle(
            Path(plan['output']) / 'resume_input.json')
        assert len(bundle['completed']) == 1
        assert bundle['completed'][0]['end']['policy_seed'] == 7
        assert [job['policy_seed'] for job in bundle['jobs']] == [7, 8]
        assert [job['environment_case_index']
                for job in bundle['jobs']] == [0, 1]
    finally:
        lock.close()
