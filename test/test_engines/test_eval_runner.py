import copy
import json
from pathlib import Path

import pytest

import fluxthemis.engines.runners.eval_runner as eval_runner_module
from fluxthemis.engines import build_runner_from_cfg
from fluxthemis.model_clients.mock import MockModelClient


class _ReportingMockModelClient(MockModelClient):

    def __init__(self):
        super().__init__(action_dim=7, chunk_size=4)
        self.events = []
        self._run_dir = None

    @property
    def evaluation_reporting_enabled(self):
        return True

    @property
    def evaluation_report_run_id(self):
        return 'reporting-session'

    @property
    def evaluation_report_run_dir(self):
        return self._run_dir

    def report_evaluation(self, event_type, payload):
        self.events.append((event_type, copy.deepcopy(payload)))
        self._run_dir = '/srv/fluxvla/evaluations/authoritative-run'
        return self._run_dir


def _runner_cfg(tmp_path, **overrides):
    cfg = dict(
        type='EvalRunner',
        environment=dict(
            type='MockEnvironment',
            tasks=['task_a', 'task_b'],
            success_after=3,
        ),
        model_client=dict(type='MockModelClient', action_dim=7, chunk_size=4),
        evaluator=dict(type='SuccessRateEvaluator'),
        episodes_per_task=2,
        max_episode_steps=10,
        execute_horizon=2,
        work_dir=str(tmp_path),
        run_name='test_run',
    )
    cfg.update(overrides)
    return cfg


def test_eval_runner_executes_chunks_and_writes_artifacts(tmp_path):
    runner = build_runner_from_cfg(_runner_cfg(tmp_path))
    try:
        runner.run_setup()
        summary = runner.run()
    finally:
        runner.cleanup()

    assert summary['run']['completed_episodes'] == 4
    assert summary['metrics']['overall']['episodes'] == 4
    assert summary['metrics']['overall']['successes'] == 4
    assert summary['metrics']['overall']['success_rate'] == 1.0
    assert set(summary['metrics']['tasks']) == {'task_a', 'task_b'}

    with runner.episodes_path.open(encoding='utf-8') as file:
        episodes = [json.loads(line) for line in file]
    assert len(episodes) == 4
    assert all(episode['steps'] == 3 for episode in episodes)
    assert all(episode['model_calls'] == 2 for episode in episodes)
    assert all(
        len(episode['prediction_metadata']) == 2 for episode in episodes)
    assert all(episode['termination_reason'] == 'success'
               for episode in episodes)
    assert runner.summary_path.is_file()
    assert summary['artifacts']['progress'] == str(runner.progress_path)
    progress = json.loads(runner.progress_path.read_text(encoding='utf-8'))
    assert progress['status'] == 'finished'
    assert progress['completed_episodes'] == 4
    assert progress['total_episodes'] == 4
    assert progress['eta_s'] == 0.0
    assert not runner.progress_path.with_suffix('.json.tmp').exists()
    assert not runner.environment.is_setup


def test_remote_reporting_owns_logs_and_artifacts(tmp_path, capsys):
    model_client = _ReportingMockModelClient()
    runner = build_runner_from_cfg(
        _runner_cfg(
            tmp_path,
            environment=dict(
                type='MockEnvironment',
                tasks=[
                    dict(
                        task_id='task_a',
                        description='place the cup',
                        metadata={'suite': 'mock'}),
                    dict(
                        task_id='task_b',
                        description='close the drawer',
                        metadata={'difficulty': 2}),
                ],
                success_after=1,
            ),
            model_client=model_client,
            episodes_per_task=1,
            run_name='remote',
        ))
    runner.set_resolved_config('/configs/acceptance.py',
                               'themis = dict(enabled=True)\n')
    try:
        runner.run_setup()
        summary = runner.run()
    finally:
        runner.cleanup()

    assert not runner.run_dir.exists()
    assert summary['artifacts'] == {
        'remote_run_dir': '/srv/fluxvla/evaluations/authoritative-run'
    }
    assert '[FluxThemis]' not in capsys.readouterr().err
    assert [event_type for event_type, _ in model_client.events] == [
        'run_start',
        'episode_start',
        'episode_end',
        'episode_start',
        'episode_end',
        'run_end',
    ]

    run_start = model_client.events[0][1]
    assert run_start['run_name'] == 'remote'
    assert run_start['seed'] == 7
    assert run_start['episodes_per_task'] == 1
    assert run_start['max_episode_steps'] == 10
    assert run_start['execute_horizon'] == 2
    assert run_start['total_tasks'] == 2
    assert run_start['total_episodes'] == 2
    assert run_start['full_suite'] is True
    assert run_start['tasks'] == [
        {
            'task_id': 'task_a',
            'task_index': 0,
            'description': 'place the cup',
            'metadata': {
                'suite': 'mock'
            },
        },
        {
            'task_id': 'task_b',
            'task_index': 1,
            'description': 'close the drawer',
            'metadata': {
                'difficulty': 2
            },
        },
    ]
    assert run_start['resolved_config'] == {
        'source': '/configs/acceptance.py',
    }

    episode_start = model_client.events[1][1]
    assert episode_start['task_id'] == 'task_a'
    assert episode_start['task_index'] == 0
    assert episode_start['description'] == 'place the cup'
    assert episode_start['episode_index'] == 0
    assert episode_start['seed'] == 7
    episode_end = model_client.events[2][1]
    assert episode_end['task_id'] == 'task_a'
    assert episode_end['task_index'] == 0
    assert episode_end['description'] == 'place the cup'
    assert episode_end['success'] is True
    assert episode_end['steps'] == 1
    assert episode_end['model_calls'] == 1
    assert episode_start['episode_id'] == 'reporting-session:1'
    assert episode_end['episode_id'] == 'reporting-session:1'
    assert episode_end['prediction_metadata'] == [{
        'mock':
        True,
        'episode_id':
        'reporting-session:1',
    }]

    run_end = model_client.events[-1][1]
    assert run_end['status'] == 'finished'
    assert run_end['completed_episodes'] == 2
    assert run_end['metrics']['overall']['successes'] == 2
    assert run_end['summary']['artifacts'] == summary['artifacts']


def test_remote_reporting_emits_failure_without_local_artifacts(
        tmp_path, monkeypatch, capsys):
    model_client = _ReportingMockModelClient()
    runner = build_runner_from_cfg(
        _runner_cfg(
            tmp_path,
            model_client=model_client,
            episodes_per_task=1,
            run_name='remote-failure',
        ))
    runner.run_setup()

    def fail_episode(*args, **kwargs):
        raise RuntimeError('environment disconnected')

    def fail_summary():
        raise RuntimeError('summary unavailable')

    monkeypatch.setattr(runner, '_run_episode', fail_episode)
    monkeypatch.setattr(runner.evaluator, 'summarize', fail_summary)
    try:
        with pytest.raises(RuntimeError, match='environment disconnected'):
            runner.run()
    finally:
        runner.cleanup()

    assert not runner.run_dir.exists()
    assert '[FluxThemis]' not in capsys.readouterr().err
    assert [event_type for event_type, _ in model_client.events
            ] == ['run_start', 'episode_start', 'run_end']
    run_end = model_client.events[-1][1]
    assert run_end['status'] == 'failed'
    assert run_end['completed_episodes'] == 0
    assert run_end['error'] == {
        'error_type': 'RuntimeError',
        'message': 'environment disconnected',
    }
    assert run_end['metrics'] == {}
    assert run_end['metrics_error']['message'] == 'summary unavailable'


def test_remote_reporting_emits_interruption_without_local_artifacts(
        tmp_path, monkeypatch, capsys):
    model_client = _ReportingMockModelClient()
    runner = build_runner_from_cfg(
        _runner_cfg(
            tmp_path,
            model_client=model_client,
            episodes_per_task=1,
            run_name='remote-interruption',
        ))
    runner.run_setup()

    def interrupt_episode(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, '_run_episode', interrupt_episode)
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run()
    finally:
        runner.cleanup()

    assert not runner.run_dir.exists()
    assert '[FluxThemis]' not in capsys.readouterr().err
    assert model_client.events[-1][0] == 'run_end'
    assert model_client.events[-1][1]['status'] == 'interrupted'
    assert model_client.events[-1][1]['error']['error_type'] == (
        'KeyboardInterrupt')


def test_progress_tracks_episode_start_and_durable_completion(
        tmp_path, monkeypatch, capsys):
    runner = build_runner_from_cfg(
        _runner_cfg(tmp_path, run_name='progress_lifecycle'))
    progress_writes = []
    completed_writes = []
    observed_starts = []
    highest_completed = 0

    original_write_json_atomic = eval_runner_module.write_json_atomic

    def recording_write_json_atomic(path, value):
        nonlocal highest_completed
        if Path(path) == runner.progress_path:
            snapshot = copy.deepcopy(value)
            progress_writes.append(snapshot)
            completed = snapshot['completed_episodes']
            if completed > highest_completed:
                with runner.episodes_path.open(encoding='utf-8') as file:
                    durable_episodes = [line for line in file if line.strip()]
                assert len(durable_episodes) == completed
                assert len(runner.evaluator.results) == completed
                completed_writes.append(snapshot)
                highest_completed = completed
        original_write_json_atomic(path, value)

    monkeypatch.setattr(eval_runner_module, 'write_json_atomic',
                        recording_write_json_atomic)

    runner.run_setup()
    original_run_episode = runner._run_episode

    def inspect_progress_at_episode_start(task, episode_index, seed):
        observed_starts.append(
            json.loads(runner.progress_path.read_text(encoding='utf-8')))
        return original_run_episode(task, episode_index, seed)

    monkeypatch.setattr(runner, '_run_episode',
                        inspect_progress_at_episode_start)
    try:
        runner.run()
    finally:
        runner.cleanup()

    assert progress_writes[0]['status'] == 'running'
    assert progress_writes[0]['completed_episodes'] == 0
    assert progress_writes[0]['current_episode'] is None

    assert [item['completed_episodes']
            for item in observed_starts] == [0, 1, 2, 3]
    assert [item['current_episode']['task_id'] for item in observed_starts
            ] == ['task_a', 'task_a', 'task_b', 'task_b']
    assert [
        item['current_episode']['episode_index'] for item in observed_starts
    ] == [0, 1, 0, 1]
    assert [item['current_episode']['seed']
            for item in observed_starts] == [7, 8, 9, 10]
    assert all(item['status'] == 'running' for item in observed_starts)
    assert [item['completed_episodes']
            for item in completed_writes] == [1, 2, 3, 4]

    progress = json.loads(runner.progress_path.read_text(encoding='utf-8'))
    assert progress['status'] == 'finished'
    assert progress['completed_episodes'] == 4
    assert progress['eta_s'] == 0.0

    terminal_events = [
        'episode_start' if 'episode_start' in line else 'episode_done'
        for line in capsys.readouterr().err.splitlines()
        if 'episode_start' in line or 'episode_done' in line
    ]
    assert terminal_events == [
        'episode_start',
        'episode_done',
        'episode_start',
        'episode_done',
        'episode_start',
        'episode_done',
        'episode_start',
        'episode_done',
    ]


def test_failed_episode_does_not_increment_completed_progress(
        tmp_path, monkeypatch, capsys):
    runner = build_runner_from_cfg(_runner_cfg(tmp_path, run_name='failed'))
    runner.run_setup()
    original_on_episode_end = runner.evaluator.on_episode_end
    hook_calls = 0

    def fail_second_episode(result):
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 2:
            raise RuntimeError('evaluator failed')
        original_on_episode_end(result)

    monkeypatch.setattr(runner.evaluator, 'on_episode_end',
                        fail_second_episode)
    try:
        with pytest.raises(RuntimeError, match='evaluator failed'):
            runner.run()
    finally:
        runner.cleanup()

    progress = json.loads(runner.progress_path.read_text(encoding='utf-8'))
    failure = json.loads(runner.failure_path.read_text(encoding='utf-8'))
    with runner.episodes_path.open(encoding='utf-8') as file:
        durable_episodes = [line for line in file if line.strip()]

    assert progress['status'] == 'failed'
    assert progress['completed_episodes'] == 1
    assert progress['error']['error_type'] == 'RuntimeError'
    assert failure['completed_episodes'] == 1
    assert len(durable_episodes) == 1
    terminal_events = [
        'episode_start' if 'episode_start' in line else 'episode_done'
        for line in capsys.readouterr().err.splitlines()
        if 'episode_start' in line or 'episode_done' in line
    ]
    assert terminal_events == [
        'episode_start', 'episode_done', 'episode_start'
    ]


def test_keyboard_interrupt_preserves_last_completed_episode(
        tmp_path, monkeypatch, capsys):
    runner = build_runner_from_cfg(
        _runner_cfg(tmp_path, run_name='interrupted'))
    runner.run_setup()
    original_run_episode = runner._run_episode
    episode_calls = 0

    def interrupt_second_episode(task, episode_index, seed):
        nonlocal episode_calls
        episode_calls += 1
        if episode_calls == 2:
            raise KeyboardInterrupt
        return original_run_episode(task, episode_index, seed)

    monkeypatch.setattr(runner, '_run_episode', interrupt_second_episode)
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run()
    finally:
        runner.cleanup()

    progress = json.loads(runner.progress_path.read_text(encoding='utf-8'))
    with runner.episodes_path.open(encoding='utf-8') as file:
        durable_episodes = [line for line in file if line.strip()]

    assert progress['status'] == 'interrupted'
    assert progress['completed_episodes'] == 1
    assert len(durable_episodes) == 1
    assert not runner.failure_path.exists()
    terminal_events = [
        'episode_start' if 'episode_start' in line else 'episode_done'
        for line in capsys.readouterr().err.splitlines()
        if 'episode_start' in line or 'episode_done' in line
    ]
    assert terminal_events == [
        'episode_start', 'episode_done', 'episode_start'
    ]


def test_truncation_is_not_implicitly_counted_as_success(tmp_path):
    cfg = _runner_cfg(
        tmp_path,
        environment=dict(
            type='MockEnvironment',
            tasks=['task'],
            success_after=None,
            max_internal_steps=2,
        ),
        episodes_per_task=1,
        run_name='truncated',
    )
    runner = build_runner_from_cfg(cfg)
    try:
        runner.run_setup()
        summary = runner.run()
    finally:
        runner.cleanup()

    assert summary['metrics']['overall']['successes'] == 0
    with runner.episodes_path.open(encoding='utf-8') as file:
        episode = json.loads(file.readline())
    assert episode['termination_reason'] == 'truncated'
    assert episode['success'] is False


def test_task_filter_preserves_requested_order(tmp_path):
    runner = build_runner_from_cfg(
        _runner_cfg(
            tmp_path,
            task_ids=['task_b'],
            episodes_per_task=1,
            run_name='filtered',
        ))
    try:
        runner.run_setup()
        summary = runner.run()
    finally:
        runner.cleanup()

    assert summary['run']['tasks'] == ['task_b']
    assert set(summary['metrics']['tasks']) == {'task_b'}


def test_metadata_filters_use_or_within_fields_and_and_across_fields(tmp_path):
    tasks = [
        dict(
            task_id='camera-2',
            metadata={
                'category': 'Camera Viewpoints',
                'difficulty_level': 2,
            }),
        dict(
            task_id='camera-3',
            metadata={
                'category': 'Camera Viewpoints',
                'difficulty_level': 3,
            }),
        dict(
            task_id='light-3',
            metadata={
                'category': 'Light Conditions',
                'difficulty_level': 3,
            }),
    ]
    runner = build_runner_from_cfg(
        _runner_cfg(
            tmp_path,
            environment=dict(
                type='MockEnvironment', tasks=tasks, success_after=1),
            task_metadata_filters=dict(
                category=['Camera Viewpoints', 'Light Conditions'],
                difficulty_level=[3],
            ),
            episodes_per_task=1,
            run_name='metadata-filtered',
        ))
    try:
        runner.run_setup()
        summary = runner.run()
    finally:
        runner.cleanup()

    assert summary['run']['tasks'] == ['camera-3', 'light-3']


def test_task_ids_and_metadata_filters_are_intersected(tmp_path):
    runner = build_runner_from_cfg(
        _runner_cfg(
            tmp_path,
            environment=dict(
                type='MockEnvironment',
                tasks=[
                    dict(task_id='a', metadata={'difficulty_level': None}),
                    dict(task_id='b', metadata={'difficulty_level': 3}),
                    dict(task_id='c', metadata={'difficulty_level': None}),
                ],
                success_after=1,
            ),
            task_ids=['c', 'b'],
            task_metadata_filters={'difficulty_level': [None]},
            episodes_per_task=1,
            run_name='intersected',
        ))
    try:
        runner.run_setup()
    finally:
        runner.cleanup()

    assert [task.task_id for task in runner.tasks_to_run] == ['c']


def test_metadata_filter_rejects_unknown_field_and_empty_result(tmp_path):
    for run_name, filters, message in (
        ('unknown-field', {
            'missing': [1]
        }, 'Unknown task metadata'),
        ('empty-result', {
            'difficulty': [9]
        }, 'selected no tasks'),
    ):
        runner = build_runner_from_cfg(
            _runner_cfg(
                tmp_path,
                environment=dict(
                    type='MockEnvironment',
                    tasks=[dict(task_id='a', metadata={'difficulty': 1})],
                ),
                task_metadata_filters=filters,
                run_name=run_name,
            ))
        with pytest.raises(ValueError, match=message):
            runner.run_setup()
        runner.cleanup()


def test_metadata_filter_marks_reported_run_as_partial(tmp_path):
    model_client = _ReportingMockModelClient()
    runner = build_runner_from_cfg(
        _runner_cfg(
            tmp_path,
            environment=dict(
                type='MockEnvironment',
                tasks=[dict(task_id='a', metadata={'category': 'Camera'})],
                success_after=1,
            ),
            model_client=model_client,
            task_metadata_filters={'category': ['Camera']},
            episodes_per_task=1,
            run_name='metadata-report',
        ))
    try:
        runner.run_setup()
        runner.run()
    finally:
        runner.cleanup()

    assert model_client.events[0][1]['full_suite'] is False
    assert model_client.events[0][1]['task_metadata_filters'] == {
        'category': ['Camera']
    }
    assert model_client.events[-1][1]['full_suite'] is False


@pytest.mark.parametrize('run_name', ['.', '..', '../outside', '/tmp/outside'])
def test_run_name_cannot_escape_work_dir(tmp_path, run_name):
    with pytest.raises(ValueError):
        build_runner_from_cfg(
            _runner_cfg(tmp_path, run_name=run_name, overwrite=True))


def test_existing_resolved_config_is_not_overwritten(tmp_path):
    run_dir = tmp_path / 'existing'
    run_dir.mkdir()
    resolved_config = run_dir / 'resolved_config.py'
    resolved_config.write_text('sentinel = True\n', encoding='utf-8')
    runner = build_runner_from_cfg(_runner_cfg(tmp_path, run_name='existing'))

    with pytest.raises(FileExistsError):
        runner.run_setup()
    runner.cleanup()
    assert resolved_config.read_text(encoding='utf-8') == 'sentinel = True\n'


def test_existing_progress_is_not_overwritten(tmp_path):
    run_dir = tmp_path / 'existing_progress'
    run_dir.mkdir()
    progress_path = run_dir / 'progress.json'
    progress_path.write_text('{"sentinel": true}\n', encoding='utf-8')
    runner = build_runner_from_cfg(
        _runner_cfg(tmp_path, run_name='existing_progress'))

    with pytest.raises(FileExistsError):
        runner.run_setup()
    runner.cleanup()
    assert progress_path.read_text(encoding='utf-8') == '{"sentinel": true}\n'


def test_eval_runner_requires_absolute_work_dir(tmp_path):
    with pytest.raises(ValueError, match='absolute path'):
        build_runner_from_cfg(_runner_cfg(tmp_path, work_dir='work_dirs'))


def test_eval_runner_requires_injected_work_dir(tmp_path):
    cfg = _runner_cfg(tmp_path)
    cfg.pop('work_dir')

    with pytest.raises(ValueError, match='FluxVLA configuration'):
        build_runner_from_cfg(cfg)
