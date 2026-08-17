import json

import pytest
from mmengine import Config

from fluxthemis.cli.eval import (ensure_run_artifacts_available, load_config,
                                 main, parse_args)


def _write_fluxvla_config(tmp_path, name, contents):
    fluxvla_root = tmp_path / 'FluxVLA'
    package_dir = fluxvla_root / 'fluxvla'
    config_dir = fluxvla_root / 'configs' / 'tests'
    package_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / '__init__.py').write_text('', encoding='utf-8')
    config_path = config_dir / name
    config_path.write_text(contents, encoding='utf-8')
    return config_path, fluxvla_root


@pytest.fixture
def fluxvla_config(tmp_path):
    config_path, _ = _write_fluxvla_config(
        tmp_path,
        'fluxvla_mock.py',
        """
model = dict(type='FluxVLAModelOwnedByServer')
eval = dict(type='FluxVLAEvalOwnedByServer')
custom_imports = dict(
    imports=['fluxvla_model_import_owned_by_server'],
    allow_failed_imports=False,
)
themis = dict(
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='MockEnvironment',
            tasks=[dict(task_id='pick', description='pick the object')],
            success_after=2,
        ),
        model_client=dict(
            type='MockModelClient', action_dim=7, chunk_size=2),
        evaluator=dict(type='SuccessRateEvaluator'),
        episodes_per_task=2,
        max_episode_steps=4,
        work_dir='unused',
    ),
)
""",
    )
    return config_path


def test_cli_uses_fluxvla_themis_section_and_saves_full_config(
        fluxvla_config, tmp_path, capsys):
    fluxvla_root = fluxvla_config.parents[2]
    exit_code = main([
        '--config',
        str(fluxvla_config),
        '--work-dir',
        'cli-results',
        '--run-name',
        'cli',
        '--cfg-options',
        'themis.runner.episodes_per_task=1',
    ])

    assert exit_code == 0
    work_dir = fluxvla_root / 'work_dirs' / 'cli-results'
    run_dir = work_dir / 'cli'
    assert (run_dir / 'summary.json').is_file()
    progress_path = run_dir / 'progress.json'
    assert progress_path.is_file()
    assert progress_path.is_relative_to(fluxvla_root / 'work_dirs')
    progress = json.loads(progress_path.read_text(encoding='utf-8'))
    assert progress['status'] == 'finished'
    summary = json.loads(
        (run_dir / 'summary.json').read_text(encoding='utf-8'))
    assert summary['artifacts']['progress'] == str(progress_path)
    resolved_path = run_dir / 'resolved_config.py'
    assert resolved_path.is_file()
    resolved = Config.fromfile(resolved_path, import_custom_modules=False)
    assert resolved.model.type == 'FluxVLAModelOwnedByServer'
    assert resolved.eval.type == 'FluxVLAEvalOwnedByServer'
    assert resolved.themis.runner.work_dir == str(work_dir)
    assert resolved.themis.runner.run_name == 'cli'
    assert resolved.themis.runner.episodes_per_task == 1
    assert 'Summary written to' in capsys.readouterr().out


def test_cli_dry_run_only_builds_components(fluxvla_config, capsys):
    exit_code = main([
        '--config',
        str(fluxvla_config),
        '--dry-run',
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert 'MockEnvironment' in output
    assert 'MockModelClient' in output
    assert not (fluxvla_config.parents[2] / 'work_dirs').exists()


def test_cli_adapts_standard_libero_config_for_libero_plus_zero_shot(
        tmp_path, capsys):
    config_path, _ = _write_fluxvla_config(
        tmp_path,
        'fluxvla_libero_10.py',
        """
themis = dict(
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='LiberoEnvironment',
            task_suite_name='libero_10',
            task_order_index=0,
        ),
        model_client=dict(type='MockModelClient', action_dim=7),
        evaluator=dict(type='SuccessRateEvaluator'),
        episodes_per_task=50,
        max_episode_steps=520,
        execute_horizon=10,
        work_dir='unused',
    ),
)
""",
    )
    arguments = [
        '--config',
        str(config_path),
        '--dry-run',
        '--cfg-options',
        'themis.runner.environment.type=LiberoPlusEnvironment',
        'themis.runner.task_ids=["0"]',
        'themis.runner.episodes_per_task=1',
    ]

    cfg, runner_cfg = load_config(parse_args(arguments))
    assert cfg.themis.runner.environment.type == 'LiberoPlusEnvironment'
    assert runner_cfg.environment.type == 'LiberoPlusEnvironment'
    assert runner_cfg.task_ids == [0]
    assert runner_cfg.episodes_per_task == 1

    assert main(arguments) == 0
    assert 'LiberoPlusEnvironment' in capsys.readouterr().out


def test_cli_applies_parallel_worker_and_simulator_gpu_overrides(
        fluxvla_config):
    cfg, runner_cfg = load_config(
        parse_args([
            '--config',
            str(fluxvla_config),
            '--parallel-workers',
            '3',
            '--simulator-gpus',
            '2, 4',
        ]))

    assert runner_cfg.parallel_workers == 3
    assert runner_cfg.simulator_gpu_ids == ['2', '4']
    assert cfg.themis.runner.parallel_workers == 3
    assert cfg.themis.runner.simulator_gpu_ids == ['2', '4']


def test_run_artifact_preflight_rejects_existing_explicit_run(tmp_path):
    run_dir = tmp_path / 'existing'
    run_dir.mkdir()
    (run_dir / 'episodes.jsonl').write_text('{}\n', encoding='utf-8')
    (run_dir / 'resolved_config.py').write_text('', encoding='utf-8')
    runner_cfg = {
        'work_dir': str(tmp_path),
        'run_name': 'existing',
    }

    with pytest.raises(FileExistsError) as error:
        ensure_run_artifacts_available(runner_cfg)

    message = str(error.value)
    assert str(run_dir) in message
    assert 'episodes.jsonl' in message
    assert 'resolved_config.py' in message
    assert 'Use a new --run-name' in message
    assert 'themis.runner.overwrite=True' in message


@pytest.mark.parametrize(
    'runner_cfg,dry_run',
    [
        ({
            'run_name': None
        }, False),
        ({
            'run_name': 'existing',
            'overwrite': True
        }, False),
        ({
            'run_name': 'existing'
        }, True),
    ],
)
def test_run_artifact_preflight_allows_non_destructive_modes(
        tmp_path, runner_cfg, dry_run):
    run_dir = tmp_path / 'existing'
    run_dir.mkdir(exist_ok=True)
    (run_dir / 'episodes.jsonl').write_text('{}\n', encoding='utf-8')
    runner_cfg = {'work_dir': str(tmp_path), **runner_cfg}

    ensure_run_artifacts_available(runner_cfg, dry_run=dry_run)


def test_run_artifact_preflight_defers_to_remote_reporter(tmp_path):
    run_dir = tmp_path / 'existing'
    run_dir.mkdir()
    (run_dir / 'summary.json').write_text('{}\n', encoding='utf-8')
    runner_cfg = {
        'work_dir': str(tmp_path),
        'run_name': 'existing',
        'model_client': {
            'type': 'FluxVLAROSModelClient',
            'transport': {
                'report_service_name': '/fluxvla/report_evaluation'
            },
        },
    }

    ensure_run_artifacts_available(runner_cfg)


def test_cli_remote_mode_prints_only_authoritative_handoff(
        fluxvla_config, monkeypatch, capsys):

    class FakeRemoteRunner:

        local_artifacts_enabled = False
        authoritative_run_dir = '/srv/fluxvla/evaluations/run-1'

        def __init__(self):
            self.resolved_config = None
            self.setup_called = False
            self.cleanup_called = False

        def set_resolved_config(self, source, text):
            self.resolved_config = (source, text)

        def run_setup(self):
            self.setup_called = True

        @staticmethod
        def run():
            return {'metrics': {'remote-only-secret': 42}}

        def cleanup(self):
            self.cleanup_called = True

    runner = FakeRemoteRunner()
    monkeypatch.setattr('fluxthemis.cli.eval.build_runner_from_cfg',
                        lambda cfg: runner)

    exit_code = main(['--config', str(fluxvla_config)])

    assert exit_code == 0
    assert runner.setup_called is True
    assert runner.cleanup_called is True
    assert runner.resolved_config[0] == str(fluxvla_config.resolve())
    assert 'themis' in runner.resolved_config[1]
    output = capsys.readouterr().out
    assert output.strip() == (
        'Evaluation reported to /srv/fluxvla/evaluations/run-1')
    assert 'remote-only-secret' not in output


def test_load_config_resolves_configured_work_dir_from_fluxvla_root(
        fluxvla_config):
    cfg, runner_cfg = load_config(
        parse_args(['--config', str(fluxvla_config)]))
    expected = fluxvla_config.parents[2] / 'work_dirs' / 'unused'

    assert runner_cfg.work_dir == str(expected)
    assert cfg.themis.runner.work_dir == str(expected)


def test_load_config_work_dir_is_independent_of_current_directory(
        fluxvla_config, tmp_path, monkeypatch):
    caller_dir = tmp_path / 'caller'
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)

    _, runner_cfg = load_config(parse_args(['--config', str(fluxvla_config)]))

    assert runner_cfg.work_dir == str(fluxvla_config.parents[2] / 'work_dirs' /
                                      'unused')


def test_load_config_requires_themis_section(tmp_path):
    config_path, _ = _write_fluxvla_config(
        tmp_path,
        'fluxvla_without_themis.py',
        "model = dict(type='Model')\n",
    )

    with pytest.raises(KeyError, match='must define a `themis` section'):
        load_config(parse_args(['--config', str(config_path)]))


def test_load_config_injects_one_shared_ros_transport(tmp_path):
    config_path, fluxvla_root = _write_fluxvla_config(
        tmp_path,
        'fluxvla_ros.py',
        """
themis = dict(
    transport=dict(
        service_name='/fluxvla/predict_action',
        image_keys=['camera'],
        state_keys=['state'],
        unnorm_key='robot',
    ),
    runner=dict(
        type='EvalRunner',
        environment=dict(type='MockEnvironment'),
        model_client=dict(type='FluxVLAROSModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
    ),
    ros_server=dict(dataset_section='eval'),
)
""",
    )

    cfg, runner_cfg = load_config(parse_args(['--config', str(config_path)]))

    assert runner_cfg.model_client.transport == cfg.themis.transport
    assert 'transport' not in cfg.themis.runner.model_client
    assert runner_cfg.work_dir == str(fluxvla_root / 'work_dirs' /
                                      'fluxthemis')


def test_load_config_injects_transport_into_ros2_client(tmp_path):
    config_path, _ = _write_fluxvla_config(
        tmp_path,
        'fluxvla_ros2.py',
        """
themis = dict(
    transport=dict(
        image_keys=['camera'],
        state_keys=['state'],
    ),
    runner=dict(
        type='EvalRunner',
        environment=dict(type='MockEnvironment'),
        model_client=dict(type='FluxVLAROS2ModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
    ),
    ros_server=dict(ros_version=2),
)
""",
    )

    cfg, runner_cfg = load_config(parse_args(['--config', str(config_path)]))

    assert runner_cfg.model_client.transport == cfg.themis.transport
    assert 'transport' not in cfg.themis.runner.model_client


@pytest.mark.parametrize(
    'client_type,server_version',
    [
        ('FluxVLAROSModelClient', 2),
        ('FluxVLAROS2ModelClient', 1),
    ],
)
def test_load_config_rejects_ros_client_server_version_mismatch(
        tmp_path, client_type, server_version):
    config_path, _ = _write_fluxvla_config(
        tmp_path,
        'fluxvla_ros_mismatch.py',
        f"""
themis = dict(
    transport=dict(image_keys=['camera'], state_keys=['state']),
    runner=dict(
        type='EvalRunner',
        environment=dict(type='MockEnvironment'),
        model_client=dict(type={client_type!r}),
        evaluator=dict(type='SuccessRateEvaluator'),
    ),
    ros_server=dict(ros_version={server_version}),
)
""",
    )

    with pytest.raises(ValueError, match='version mismatch'):
        load_config(parse_args(['--config', str(config_path)]))


@pytest.mark.parametrize('work_dir', ['../outside', 'work_dirs/../outside'])
def test_load_config_rejects_relative_work_dir_outside_fluxvla_work_dirs(
        fluxvla_config, work_dir):
    with pytest.raises(ValueError, match='must stay inside'):
        load_config(
            parse_args([
                '--config',
                str(fluxvla_config),
                '--work-dir',
                work_dir,
            ]))


def test_load_config_rejects_absolute_work_dir_outside_fluxvla(
        fluxvla_config, tmp_path):
    with pytest.raises(ValueError, match='must stay inside'):
        load_config(
            parse_args([
                '--config',
                str(fluxvla_config),
                '--work-dir',
                str(tmp_path / 'outside'),
            ]))


def test_load_config_accepts_fluxvla_work_dirs_root(fluxvla_config):
    work_dirs = fluxvla_config.parents[2] / 'work_dirs'
    _, runner_cfg = load_config(
        parse_args([
            '--config',
            str(fluxvla_config),
            '--work-dir',
            str(work_dirs),
        ]))

    assert runner_cfg.work_dir == str(work_dirs)


def test_load_config_rejects_work_dir_symlink_escape(fluxvla_config, tmp_path):
    work_dirs = fluxvla_config.parents[2] / 'work_dirs'
    work_dirs.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (work_dirs / 'escape').symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match='must stay inside'):
        load_config(
            parse_args([
                '--config',
                str(fluxvla_config),
                '--work-dir',
                'work_dirs/escape/results',
            ]))


def test_load_config_rejects_config_outside_fluxvla_configs(tmp_path):
    config_path = tmp_path / 'standalone.py'
    config_path.write_text(
        "themis = dict(runner=dict(type='EvalRunner'))\n",
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='FluxVLA `configs`'):
        load_config(parse_args(['--config', str(config_path)]))


def test_load_config_rejects_config_symlink_outside_fluxvla(tmp_path):
    _, fluxvla_root = _write_fluxvla_config(
        tmp_path,
        'placeholder.py',
        "themis = dict(runner=dict(type='EvalRunner'))\n",
    )
    external = tmp_path / 'external.py'
    external.write_text(
        "themis = dict(runner=dict(type='EvalRunner'))\n",
        encoding='utf-8',
    )
    linked_config = fluxvla_root / 'configs' / 'linked.py'
    linked_config.symlink_to(external)

    with pytest.raises(ValueError, match='FluxVLA `configs`'):
        load_config(parse_args(['--config', str(linked_config)]))
