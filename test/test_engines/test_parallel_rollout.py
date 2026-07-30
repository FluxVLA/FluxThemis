import copy
import multiprocessing
import queue
import time

import pytest

from fluxthemis.engines import build_runner_from_cfg
from fluxthemis.engines.runners.parallel_rollout import (
    build_context, build_episode_jobs, episode_worker_index,
    make_inference_only_model_config, rollout_worker_main)
from fluxthemis.evaluators.base import BaseEvaluator
from fluxthemis.model_clients.mock import MockModelClient
from fluxthemis.structures import EpisodeResult, TaskSpec


class _ParallelReportingMockModelClient(MockModelClient):

    def __init__(self, transport, action_dim=7, chunk_size=1):
        super().__init__(action_dim=action_dim, chunk_size=chunk_size)
        self.report_service_name = transport.get('report_service_name')
        self.events = []
        self._run_id = None
        self._run_dir = None

    def setup(self):
        if self.evaluation_reporting_enabled:
            self._run_id = 'parallel-test-run'

    @property
    def evaluation_reporting_enabled(self):
        return bool(self.report_service_name)

    @property
    def evaluation_report_run_id(self):
        return self._run_id

    @property
    def evaluation_report_run_dir(self):
        return self._run_dir

    def report_evaluation(self, event_type, payload):
        self.events.append((event_type, copy.deepcopy(payload)))
        self._run_dir = '/fluxvla/work_dirs/parallel-test-run'
        return self._run_dir


class _StepHookEvaluator(BaseEvaluator):

    def on_episode_end(self, result):
        del result

    def summarize(self):
        return {}


def _get_event(event_queue, expected_type, timeout=10.0):
    event = event_queue.get(timeout=timeout)
    assert event['type'] == expected_type, event
    return event


def test_episode_jobs_are_task_major_and_globally_unique():
    tasks = [TaskSpec('a'), TaskSpec('b')]

    jobs = build_episode_jobs(
        tasks, episodes_per_task=2, seed=7, report_run_id='run-id')

    assert [job.episode_id for job in jobs] == [
        'run-id:1',
        'run-id:2',
        'run-id:3',
        'run-id:4',
    ]
    assert [(job.task.task_id, job.episode_index, job.seed)
            for job in jobs] == [
                ('a', 0, 7),
                ('a', 1, 8),
                ('b', 0, 9),
                ('b', 1, 10),
            ]
    assert [episode_worker_index(job, 2) for job in jobs] == [0, 1, 0, 1]


def test_inference_worker_config_drops_reporting_without_mutating_source():
    source = {
        'type': 'FluxVLAROSModelClient',
        'transport': {
            'service_name': '/predict',
            'report_service_name': '/report',
        },
    }

    worker = make_inference_only_model_config(source)

    assert worker['transport'] == {'service_name': '/predict'}
    assert source['transport']['report_service_name'] == '/report'


def test_context_prefers_dynamic_environment_task_description():
    task = TaskSpec('0', description='stable environment id')

    context = build_context(
        task,
        episode_index=2,
        seed=9,
        info={'task_description': 'unlocked_waist: move the bottle'},
    )

    assert context['task_description'] == \
        'unlocked_waist: move the bottle'


def test_spawned_worker_waits_for_parent_lifecycle_acknowledgements():
    context = multiprocessing.get_context('spawn')
    job_queue = context.Queue()
    control_queue = context.Queue()
    event_queue = context.Queue()
    process = context.Process(
        target=rollout_worker_main,
        args=(
            0,
            {
                'type': 'MockEnvironment',
                'tasks': ['task'],
                'success_after': 1,
            },
            {
                'type': 'MockModelClient',
                'action_dim': 7,
                'chunk_size': 1,
            },
            job_queue,
            control_queue,
            event_queue,
            '7',
            3,
            1,
            True,
            5.0,
        ),
    )
    process.start()
    try:
        ready = _get_event(event_queue, 'ready')
        assert ready['cuda_visible_devices'] == '7'
        assert ready['mujoco_egl_device_id'] == '7'
        jobs = build_episode_jobs(
            [TaskSpec('task')],
            episodes_per_task=2,
            seed=11,
            report_run_id='spawn-run',
        )
        job_queue.put(jobs[0])
        job_queue.put(jobs[1])
        job_queue.put(None)

        first_start = _get_event(event_queue, 'episode_start')
        assert first_start['job'].episode_id == 'spawn-run:1'
        with pytest.raises(queue.Empty):
            event_queue.get(timeout=0.2)
        control_queue.put({
            'type': 'episode_start_ack',
            'episode_id': 'spawn-run:1',
        })
        first_end = _get_event(event_queue, 'episode_end')
        assert isinstance(first_end['result'], EpisodeResult)
        assert first_end['result'].prediction_metadata[0][
            'episode_id'] == 'spawn-run:1'

        with pytest.raises(queue.Empty):
            event_queue.get(timeout=0.2)
        control_queue.put({
            'type': 'episode_end_ack',
            'episode_id': 'spawn-run:1',
        })
        second_start = _get_event(event_queue, 'episode_start')
        assert second_start['job'].episode_id == 'spawn-run:2'
        control_queue.put({
            'type': 'episode_start_ack',
            'episode_id': 'spawn-run:2',
        })
        second_end = _get_event(event_queue, 'episode_end')
        assert second_end['result'].prediction_metadata[0][
            'episode_id'] == 'spawn-run:2'
        control_queue.put({
            'type': 'episode_end_ack',
            'episode_id': 'spawn-run:2',
        })
        _get_event(event_queue, 'stopped')
        process.join(timeout=10.0)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)


def test_eval_runner_coordinates_spawned_workers_and_parent_reporting(
        tmp_path):
    runner = build_runner_from_cfg(
        dict(
            type='EvalRunner',
            environment=dict(
                type='MockEnvironment',
                tasks=['task_a', 'task_b'],
                success_after=1,
            ),
            model_client=dict(
                type=_ParallelReportingMockModelClient,
                transport={'report_service_name': '/report'},
            ),
            evaluator=dict(type='SuccessRateEvaluator'),
            seed=7,
            episodes_per_task=2,
            max_episode_steps=3,
            execute_horizon=1,
            work_dir=str(tmp_path),
            run_name='parallel',
            parallel_workers=2,
            simulator_gpu_ids=['2', '3'],
            parallel_startup_timeout_s=15.0,
            parallel_event_timeout_s=15.0,
            parallel_ack_timeout_s=15.0,
        ))
    try:
        runner.run_setup()
        summary = runner.run()
    finally:
        runner.cleanup()

    assert summary['run']['completed_episodes'] == 4
    assert summary['run']['parallel_workers'] == 2
    assert summary['metrics']['overall']['successes'] == 4
    assert [(result.task_id, result.episode_index)
            for result in runner.evaluator.results] == [
                ('task_a', 0),
                ('task_a', 1),
                ('task_b', 0),
                ('task_b', 1),
            ]
    assert not runner.run_dir.exists()
    events = runner.model_client.events
    assert events[0][0] == 'run_start'
    assert events[-1][0] == 'run_end'
    starts = [payload for kind, payload in events if kind == 'episode_start']
    ends = [payload for kind, payload in events if kind == 'episode_end']
    assert {payload['episode_id']
            for payload in starts} == {
                'parallel-test-run:1',
                'parallel-test-run:2',
                'parallel-test-run:3',
                'parallel-test-run:4',
            }
    assert {payload['episode_id']
            for payload in ends
            } == {payload['episode_id']
                  for payload in starts}
    assert events[0][1]['parallel_workers'] == 2
    assert events[0][1]['simulator_gpu_ids'] == ['2', '3']


def test_parallel_runner_rejects_a_client_without_remote_reporting(tmp_path):
    with pytest.raises(ValueError, match='requires FluxVLA evaluation'):
        build_runner_from_cfg(
            dict(
                type='EvalRunner',
                environment=dict(type='MockEnvironment'),
                model_client=dict(type='MockModelClient'),
                evaluator=dict(type='SuccessRateEvaluator'),
                work_dir=str(tmp_path),
                parallel_workers=2,
            ))


def test_parallel_runner_rejects_evaluator_without_merge_capability(tmp_path):
    with pytest.raises(ValueError, match='supports_parallel_rollouts=True'):
        build_runner_from_cfg(
            dict(
                type='EvalRunner',
                environment=dict(type='MockEnvironment'),
                model_client=dict(
                    type=_ParallelReportingMockModelClient,
                    transport={'report_service_name': '/report'},
                ),
                evaluator=dict(type=_StepHookEvaluator),
                work_dir=str(tmp_path),
                parallel_workers=2,
            ))


def test_parallel_failure_cleanup_does_not_join_unconsumed_job_queues(
        tmp_path):
    runner = build_runner_from_cfg(
        dict(
            type='EvalRunner',
            environment=dict(
                type='MockEnvironment',
                tasks=['task'],
                success_after=1,
            ),
            model_client=dict(
                type=_ParallelReportingMockModelClient,
                transport={'report_service_name': '/report'},
            ),
            evaluator=dict(type='SuccessRateEvaluator'),
            episodes_per_task=500,
            max_episode_steps=1,
            work_dir=str(tmp_path),
            parallel_workers=2,
            parallel_startup_timeout_s=15.0,
            parallel_event_timeout_s=15.0,
        ))
    runner.run_setup()
    original_report = runner.model_client.report_evaluation
    worker_processes = list(runner._parallel_processes)
    terminal_workers_alive = []

    def fail_first_episode(event_type, payload):
        if event_type == 'episode_start':
            raise RuntimeError('reporter rejected episode')
        if event_type == 'run_end':
            terminal_workers_alive.append(
                [process.is_alive() for process in worker_processes])
        return original_report(event_type, payload)

    runner.model_client.report_evaluation = fail_first_episode
    try:
        with pytest.raises(RuntimeError, match='reporter rejected episode'):
            runner.run()
    finally:
        started = time.monotonic()
        runner.cleanup()
        cleanup_s = time.monotonic() - started

    assert cleanup_s < 6.0
    assert terminal_workers_alive == [[False, False]]
