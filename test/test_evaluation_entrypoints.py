"""Check the public launcher and recovery path outside package-only tests."""

import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from mmengine import Config

from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_episode)
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings,
                                                   EvaluationStatus, utc_now)
from fluxthemis.environments import MockEnvironment
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.model_clients import MockPolicyClient
from fluxthemis.reporters import MockEvaluationReporter
from fluxthemis.structures import TaskSpec

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_retains_explicit_runtime_case_replacement(tmp_path):
    from fluxthemis.engines.runners.resume import (build_bundle,
                                                   validate_journal)

    reporter = MockEvaluationReporter()
    tasks = [TaskSpec('task')]
    job = build_episode_jobs(tasks, 1, 7, reporter.run_session_id)[0]
    replacement = replace(job, environment_case_index=1)
    with EvaluationRun(EvaluationSettings(), SuccessRateEvaluator(),
                       reporter) as run:
        run.start(tasks, 1)
        run.on_episode_start(job.episode_start_payload(utc_now()))
        payload = replacement.episode_start_payload(utc_now())
        payload['replaces_environment_case_index'] = 0
        run.on_episode_start(payload)
        environment = MockEnvironment(success_after=1)
        environment.setup()
        result = execute_episode(environment, MockPolicyClient(), replacement,
                                 5, None, True)
        run.on_episode_end(result, replacement.episode_id)
        run.finish(result)
    rows = [
        dict(
            event_type=kind,
            payload=body,
            sequence=index,
            run_session_id=reporter.run_session_id,
            recorded_at=utc_now())
        for index, (kind, body) in enumerate(reporter.events, 1)
    ]
    journal = tmp_path / 'events.jsonl'
    journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    bundle = build_bundle(tmp_path)
    assert len(bundle['completed']) == 1
    assert bundle['completed'][0]['start']['environment_case_index'] == 1
    assert bundle['jobs'][0]['environment_case_index'] == 1
    rows[2]['payload'].pop('replaces_environment_case_index')
    with pytest.raises(ValueError, match='Duplicate episode_start'):
        validate_journal(rows)


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
    if capacity != 1:
        evaluation['episode_batch_size'] = capacity
    return Config(
        dict(
            themis=dict(
                evaluation=evaluation,
                transport=dict(image_keys=['image'], state_keys=['state']))))


@pytest.mark.parametrize('stale_package', [False, True])
@pytest.mark.parametrize('dry_run', [False, True])
def test_launcher_imports_current_checkout(tmp_path, stale_package, dry_run):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    if stale_package:
        package = tmp_path / 'old_install' / 'fluxthemis'
        package.mkdir(parents=True)
        (package / '__init__.py'
         ).write_text('raise RuntimeError("Imported another checkout")\n')
        env['PYTHONPATH'] = str(package.parent)
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
        assert frozen.themis.evaluation.episode_batch_size == 2
        assert not (output / 'simulator.log').exists()


@pytest.mark.parametrize('capacity', [1, 2])
def test_prepare_resume_preserves_scalar_and_batch_results(
        tmp_path, capacity, resume):
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
            EvaluationSettings(
                episodes_per_task=2, episode_batch_size=capacity),
            SuccessRateEvaluator(), reporter) as run:
        run.start(tasks, 2)
        environment = MockEnvironment(success_after=1)
        environment.setup()
        result = execute_episode(
            environment,
            MockPolicyClient(),
            jobs[0],
            5,
            None,
            True,
            on_episode_started=lambda: run.on_episode_start(jobs[
                0].episode_start_payload(utc_now())))
        run.on_episode_end(result, jobs[0].episode_id)
        run.fail(EvaluationStatus.INTERRUPTED, None, result,
                 dict(error_type='KeyboardInterrupt', message='interrupted'))
    if capacity == 1:
        # Existing journals predate the optional batch setting.
        reporter.events[0][1].pop('episode_batch_size')
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
