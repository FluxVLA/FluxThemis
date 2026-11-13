"""CPU launch/preflight checks; no model or simulator is started."""

import importlib
import json
from pathlib import Path

import pytest
import yaml
from mmengine import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setenv('ROBOLAB_ROOT', '/tmp/robolab-fixture')
    return importlib.import_module('run_evaluation')


def scalar_config(tmp_path):
    config = Config.fromfile(ROOT / 'configs/robolab/droid.py')
    evaluation = config.themis.evaluation
    evaluation.environment.num_envs = 1
    evaluation.episodes_per_task = 1
    path = tmp_path / 'scalar.py'
    config.dump(str(path))
    return path


def write_launch(tmp_path, **simulator):
    path = tmp_path / 'launch.yaml'
    data = dict(
        simulator=dict(
            config=str(scalar_config(tmp_path)), gpu_ids=[0], **simulator),
        run=dict(output_root=str(tmp_path / 'outputs')))
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize('task', ['BananaInBowlTask', 'MustardInRightBinTask'])
def test_yaml_task_selects_registration_and_quota(launch, tmp_path, task):
    settings = launch.load_config(
        write_launch(tmp_path, task=task, episodes=2))
    evaluation = settings['config'].themis.evaluation
    assert evaluation.environment.task_name == task
    assert evaluation.task_ids is None
    assert evaluation.episodes_per_task == 2
    assert evaluation.episodes_per_task_overrides == {task: 2}


def test_full_restores_official_cli_scope(launch, tmp_path):
    settings = launch.load_config(
        write_launch(tmp_path, task='BananaInBowlTask', episodes=2),
        mode='full')
    evaluation = settings['config'].themis.evaluation
    assert evaluation.environment.task_name == 'all'
    assert evaluation.task_ids is None
    assert evaluation.episodes_per_task == 1
    assert not evaluation.get('episodes_per_task_overrides')
    assert evaluation.environment.num_envs == 1
    assert evaluation.environment.simulator_seed == 0


def test_smoke_scales_to_workers(launch, tmp_path):
    settings = launch.load_config(
        write_launch(tmp_path, task='MustardInRightBinTask', episodes=9),
        mode='smoke',
        overrides=dict(gpu_ids='6', workers_per_gpu=2))
    evaluation = settings['config'].themis.evaluation
    assert evaluation.environment.task_name == 'BananaInBowlTask'
    assert evaluation.task_ids is None
    assert evaluation.episodes_per_task_overrides == {'BananaInBowlTask': 2}


def test_python_task_list_preserves_order(launch, tmp_path):
    tasks = ['MustardInRightBinTask', 'BananaInBowlTask']
    settings = launch.load_config(
        scalar_config(tmp_path),
        overrides=dict(
            cfg_options={
                'themis.evaluation.environment.task_name': tasks,
                'themis.evaluation.episodes_per_task': 3,
            }))
    evaluation = settings['config'].themis.evaluation
    assert evaluation.environment.task_name == tasks
    assert evaluation.episodes_per_task == 3


def test_root_precedence_and_child_environment(launch, tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOLAB_ROOT', str(tmp_path / 'fallback'))
    monkeypatch.setenv('ROBO_TEST_ROOT', str(tmp_path / 'installed'))
    config = write_launch(tmp_path, robolab_root='${ROBO_TEST_ROOT}/../actual')
    settings = launch.load_config(config)
    root = str(tmp_path / 'actual')
    assert settings['local']['robolab_root'] == root
    assert settings[
        'config'].themis.evaluation.environment.robolab_root == root
    runtime = launch.ProcessRunner(settings)
    assert runtime.env['ROBOLAB_ROOT'] == root
    assert runtime.python == settings['local']['python']
    settings = launch.load_config(write_launch(tmp_path))
    assert settings['local']['robolab_root'] == str(tmp_path / 'fallback')
    settings = launch.load_config(
        write_launch(tmp_path),
        overrides=dict(
            cfg_options={
                'themis.evaluation.environment.robolab_root': '../python_root'
            }))
    assert settings['local']['robolab_root'] == str(tmp_path.parent /
                                                    'python_root')


@pytest.mark.parametrize('values, message', [
    ({
        'robolab_root': '$UNSET_ROBOLAB_PATH'
    }, 'unset environment'),
    ({
        'episode_timeout_s': 0
    }, 'episode_timeout_s'),
    ({
        'simulator_restart_limit': -1
    }, 'simulator_restart_limit'),
    ({
        'task': ['BananaInBowlTask']
    }, 'simulator.task'),
    ({
        'unknown': True
    }, 'unknown fields'),
])
def test_invalid_launch_fields(launch, tmp_path, values, message):
    with pytest.raises(ValueError, match=message):
        launch.load_config(write_launch(tmp_path, **values))


def test_dry_run_freezes_output_and_launch_limits(launch, tmp_path):
    config = write_launch(
        tmp_path,
        robolab_root='../RoboLab',
        task='MustardInRightBinTask',
        episodes=2,
        episode_timeout_s=300,
        simulator_restart_limit=0)
    assert launch.main(
        ['--config', str(config), '--dry-run', '--gpu-ids', '6']) == 0
    output, = (tmp_path / 'outputs').iterdir()
    plan = json.loads((output / 'plan.json').read_text())
    cfg = Config.fromfile(str(output / 'config_snapshot.py'))
    assert cfg.themis.evaluation.environment.output_dir == str(output /
                                                               'native')
    args = launch.simulator_arguments(plan)
    assert args[args.index('--episode-timeout-s') + 1] == '300'
    assert args[args.index('--simulator-restart-limit') + 1] == '0'
    assert 'themis.evaluation.environment.robolab_root=' + str(
        tmp_path.parent / 'RoboLab') in args
    assert not (output / 'preflight.log').exists()
    assert not (output / 'native').exists()


@pytest.mark.parametrize('benchmark, smoke_task', [
    ('robodojo', 'put_bottles_into_dustbin'),
    ('robotwin', None),
])
def test_existing_remote_defaults_unchanged(launch, benchmark, smoke_task):
    settings = launch.load_config(ROOT / f'configs/{benchmark}/remote.yaml')
    assert settings['local']['gpus'] == '0'
    assert settings['local']['workers'] == 1
    assert settings['local']['episodes'] == 1
    assert settings['local']['episode_timeout_s'] == 3600
    assert settings['local']['simulator_restart_limit'] == 3
    if smoke_task:
        assert settings['local']['task'] == smoke_task


def test_robolab_root_cannot_affect_other_benchmarks(launch, tmp_path):
    path = write_launch(tmp_path, robolab_root='../RoboLab')
    data = yaml.safe_load(path.read_text())
    data['simulator']['config'] = str(ROOT / 'configs/robodojo/arx_x5.py')
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='only to RoboLab'):
        launch.load_config(path)


def test_invalid_interpreter_fails_before_subprocess(launch, tmp_path):
    settings = launch.load_config(
        write_launch(tmp_path, python='./missing-python'))
    with pytest.raises(ValueError, match='Python is not executable'):
        launch.ProcessRunner(settings).connect()


def test_robolab_resume_is_unsupported(launch, tmp_path, capsys):
    path = write_launch(tmp_path)
    assert launch.run_config(path, dry_run=True, resume=tmp_path / 'old') == 1
    assert 'RoboLab resume is unsupported' in capsys.readouterr().out
    assert not (tmp_path / 'outputs').exists()


@pytest.mark.parametrize('visible, assigned, expected', [
    ('6', '6', 0),
    ('GPU-example', 'GPU-example', 0),
    ('2,6', '6', 1),
    (None, '6', 6),
])
def test_worker_cuda_mapping(monkeypatch, visible, assigned, expected):
    from fluxthemis.environments.sim.robolab import RoboLabEnvironment
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    env = RoboLabEnvironment(task_name='all', render_device_id=6)
    env.configure_simulator_device(assigned)
    assert env.device_id == expected
    assert env.render_device_id == 6


@pytest.mark.parametrize('visible, assigned', [
    ('6', '0'),
    ('', '6'),
    ('6', 'GPU-other'),
    (None, 'GPU-example'),
    ('6,6', '6'),
    (None, '-1'),
])
def test_worker_rejects_invisible_device(monkeypatch, visible, assigned):
    from fluxthemis.environments.sim.robolab import RoboLabEnvironment
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(ValueError, match='GPU|CUDA_VISIBLE_DEVICES'):
        RoboLabEnvironment(
            task_name='all').configure_simulator_device(assigned)


@pytest.fixture
def preflight(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return importlib.import_module('preflight.robolab')


def write_file(root, name, text='asset'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def checkout(tmp_path, preflight, monkeypatch):
    monkeypatch.setattr(preflight.importlib.metadata, 'version',
                        lambda _: '2.2.0')
    for name in preflight.REQUIRED_SOURCES:
        write_file(tmp_path, name, '# source\n')
    for name in [
            'assets/robots/franka_robotiq_2f_85_flattened.usd',
            'assets/fixtures/franka_table.usd',
            'assets/backgrounds/default/home_office.exr',
            'assets/objects/object_catalog.json'
    ]:
        write_file(tmp_path, name)
    write_file(tmp_path, 'assets/scenes/test.usda',
               '#usda 1.0\n def Xform "test" {}\n')
    for task in ['BananaInBowlTask', 'MustardInRightBinTask']:
        write_file(
            tmp_path, f'robolab/tasks/benchmark/{task}.py',
            f'class {task}(Task):\n'
            '    scene = import_scene("test.usda")\n')
    return tmp_path


def test_discover_real_source_without_importing(preflight, checkout):
    # Executing these fixtures would raise NameError (no Task/Isaac imports).
    result = preflight.check_environment(checkout, 'all')
    assert result['status'] == 'ok'
    assert set(result['task_sources']) == {
        'BananaInBowlTask', 'MustardInRightBinTask'
    }
    result = preflight.check_environment(
        checkout, ['MustardInRightBinTask', 'BananaInBowlTask'])
    assert list(result['task_sources']) == [
        'MustardInRightBinTask', 'BananaInBowlTask'
    ]


def test_unknown_or_invalid_tasks_rejected(preflight, checkout):
    selections = ['MissingTask', [], ['all', 'BananaInBowlTask']]
    selections.append(['BananaInBowlTask'] * 2)
    for selection in selections:
        with pytest.raises(ValueError, match='task'):
            preflight.check_environment(checkout, selection)


@pytest.mark.parametrize('name', [
    'robolab/core/environments/runtime.py',
    'assets/robots/franka_robotiq_2f_85_flattened.usd',
    'assets/fixtures/franka_table.usd',
    'assets/backgrounds/default/home_office.exr',
    'assets/scenes/test.usda',
    'assets/objects/object_catalog.json',
])
def test_missing_required_input_rejected(preflight, checkout, name):
    (checkout / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        preflight.check_environment(checkout, 'BananaInBowlTask')


def test_empty_asset_and_missing_scene_reference(preflight, checkout):
    asset = checkout / 'assets/fixtures/franka_table.usd'
    asset.write_text('')
    with pytest.raises(ValueError, match='Empty'):
        preflight.check_environment(checkout)
    asset.write_text('asset')
    (checkout / 'assets/scenes/test.usda').write_text(
        '#usda 1.0\n def Xform "x" '
        '(references = @../objects/missing.usd@) {}\n')
    with pytest.raises(FileNotFoundError, match='missing.usd'):
        preflight.check_environment(checkout)


def test_missing_simulator_dependency_rejected(preflight, checkout,
                                               monkeypatch):

    def missing(name):
        if name == 'isaacsim':
            raise importlib.metadata.PackageNotFoundError(name)
        return '2.2.0'

    monkeypatch.setattr(preflight.importlib.metadata, 'version', missing)
    with pytest.raises(
            importlib.metadata.PackageNotFoundError, match='isaacsim'):
        preflight.check_environment(checkout)


def test_lfs_pointer_is_not_an_installed_asset(preflight, checkout):
    (checkout / 'assets/fixtures/franka_table.usd').write_text(
        'version https://git-lfs.github.com/spec/v1\n'
        'oid sha256:example\nsize 123\n')
    with pytest.raises(ValueError, match='Unmaterialized.*LFS'):
        preflight.check_environment(checkout)


def test_removed_usd_payload_is_not_a_required_asset(preflight, checkout):
    (checkout / 'assets/scenes/test.usda').write_text(
        '#usda 1.0\n def Xform "bin" (\n'
        '    delete payload = @../fixtures/old.usd@\n'
        '    reorder references = [\n        @missing.usd@\n    ]\n'
        '    prepend payload = @https://example.com/bin.usd@\n) {}\n')
    result = preflight.check_environment(checkout)
    assert result['unchecked_assets'] == ['https://example.com/bin.usd']


def test_dynamic_scene_cannot_silently_pass(preflight, checkout):
    write_file(
        checkout, 'robolab/tasks/benchmark/BananaInBowlTask.py',
        'class BananaInBowlTask(Task):\n'
        '    scene = import_scene(select_scene())\n')
    with pytest.raises(ValueError, match='literal scene'):
        preflight.check_environment(checkout, 'BananaInBowlTask')


def test_environment_dispatch_uses_robolab_checker(preflight, checkout,
                                                   monkeypatch):
    from mmengine import Config
    environment = importlib.import_module('preflight.environment')
    monkeypatch.delenv('ROBOLAB_ROOT', raising=False)
    path = checkout / 'config.py'
    Config(
        dict(
            themis=dict(
                evaluation=dict(
                    environment=dict(
                        type='RoboLabEnvironment',
                        robolab_root=str(checkout),
                        task_name='MustardInRightBinTask'))))).dump(str(path))
    result = environment.check_environment(str(path))
    assert list(result['task_sources']) == ['MustardInRightBinTask']
    monkeypatch.setenv('ROBOLAB_ROOT', str(checkout / 'missing'))
    with pytest.raises(FileNotFoundError):
        environment.check_environment(str(path))


@pytest.fixture
def installer(tmp_path, monkeypatch):
    """Use local Git and fake package tools; no downloads or Isaac imports."""
    import os
    import shutil
    import subprocess
    import sys

    git = shutil.which('git')
    source = tmp_path / 'source repo'
    source.mkdir()

    def git_run(*args):
        return subprocess.check_output(
            [git, '-C', str(source), *args], text=True).strip()

    git_run('init', '-q')
    for name in ('pyproject.toml', 'uv.lock'):
        (source / name).write_text('# fixture\n')
    git_run('add', '.')
    git_run('-c', 'user.name=Test', '-c', 'user.email=test@example.com', '-c',
            'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', 'commit',
            '-qm', 'fixture')
    ref = git_run('rev-parse', 'HEAD')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stub = bin_dir / 'tool'
    stub.write_text(f'#!{sys.executable}\n' + '''
import json
import os
from pathlib import Path
import subprocess
import sys

name, args = Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ['INSTALL_LOG'], 'a') as log:
    log.write(json.dumps([name, args]) + '\\n')
stage = ''
if name == 'git':
    if 'lfs' not in args:
        sys.exit(subprocess.call([os.environ['REAL_GIT'], *args]))
    stage = 'assets' if 'pull' in args else 'git-lfs'
elif name == 'uv':
    stage = args[0] if args[0] == 'sync' else args[1]
elif name == 'python':
    code = args[1]
    stage = 'preflight' if 'check_environment' in code else 'metadata'
if stage == os.environ.get('INSTALL_FAIL'):
    sys.exit(17)
if name == 'uv' and stage == 'sync':
    python = Path(os.environ['UV_PROJECT_ENVIRONMENT']) / 'bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(os.environ['INSTALL_STUB'])
if name == 'python' and stage == 'metadata':
    import importlib.metadata
    sys.version_info = (3, 11, 0)
    if os.environ.get('INSTALL_BAD_PYTHON'):
        sys.version_info = (3, 12, 0)
    def version(package):
        if os.environ.get('INSTALL_BAD_STACK'):
            return '0.0.0'
        return {'isaaclab': '2.2.0', 'isaacsim': os.environ.get(
            'INSTALL_ISAACSIM_VERSION', '5.0.0')}[package]
    importlib.metadata.version = version
    exec(code)
''')
    stub.chmod(0o755)
    for name in ('git', 'uv'):
        (bin_dir / name).symlink_to(stub)
    log = tmp_path / 'tools.jsonl'
    for key, value in {
            'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
            'REAL_GIT': git,
            'INSTALL_LOG': str(log),
            'INSTALL_STUB': str(stub),
            'ROBOLAB_REPO': str(source),
            'ROBOLAB_REF': ref,
    }.items():
        monkeypatch.setenv(key, value)

    def run(*args):
        return subprocess.run([
            'bash',
            str(ROOT / 'scripts/install_robolab.sh'), '--robolab-root',
            str(tmp_path / 'new checkout'), *map(str, args)
        ],
                              capture_output=True,
                              text=True,
                              timeout=30)

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()
                ] if log.exists() else []

    return run, calls, source, stub


def test_installer_pinned_clone_and_asset_scope(installer, tmp_path):
    run, calls, source, _ = installer
    env = tmp_path / 'simulator env'
    result = run('--env-path', env)
    assert result.returncode == 0, result.stderr
    assert (env / 'bin/python').is_symlink()
    assert any(name == 'uv' and args[:2] == ['pip', 'install']
               and 'websockets==12.0' in args for name, args in calls())
    assert (tmp_path /
            'new checkout/uv.lock').read_bytes() == (source /
                                                     'uv.lock').read_bytes()
    assert any(
        name == 'git'
        and args[-4:] == ['lfs', 'pull', '--include=assets/**', '--exclude=']
        for name, args in calls())
    assert any(name == 'python' and 'check_environment' in args[1]
               for name, args in calls())


@pytest.mark.parametrize('version,accepted', [
    ('5.0.0', True),
    ('5.0.0.0', True),
    ('5.0.0.0+vendor', True),
    ('5.0.0.1', False),
    ('5.0.1.0', False),
    ('5.1.0.0', False),
    ('5.0.0rc1', False),
    ('5.0.0.post1', False),
])
def test_installer_accepts_only_isaac50_release(installer, tmp_path,
                                                monkeypatch, version,
                                                accepted):
    run, calls, _, stub = installer
    python = tmp_path / 'python'
    python.symlink_to(stub)
    monkeypatch.setenv('INSTALL_ISAACSIM_VERSION', version)
    result = run('--python', python, '--skip-assets')
    assert (result.returncode == 0) is accepted, result.stderr
    assert any(name == 'uv' for name, _ in calls()) is accepted


def test_installer_prepared_python_keeps_venv_path(installer, tmp_path):
    run, calls, _, stub = installer
    python = tmp_path / 'prepared env/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(stub)
    result = run('--python', python, '--skip-assets')
    assert result.returncode == 0, result.stderr
    uv = [args for name, args in calls() if name == 'uv']
    assert uv[0][0:4] == ['pip', 'install', '--python', str(python)]
    assert not any('sync' in args or 'lfs' in args for _, args in calls())
    assert not any('check_environment' in ' '.join(args)
                   for _, args in calls())


@pytest.mark.parametrize('change', ['dirty', 'revision'])
def test_installer_preserves_existing_checkout(installer, monkeypatch, change):
    run, calls, source, _ = installer
    if change == 'dirty':
        (source / 'uv.lock').write_text('# user edit\n')
    else:
        import subprocess
        subprocess.run([
            'git', '-C',
            str(source), '-c', 'user.name=Test', '-c',
            'user.email=test@example.com', '-c', 'commit.gpgsign=false', '-c',
            'core.hooksPath=/dev/null', 'commit', '--allow-empty', '-qm',
            'different revision'
        ],
                       check=True)
    before = (source / 'uv.lock').read_bytes()
    result = run('--robolab-root', source, '--skip-assets')
    assert result.returncode != 0
    assert (source / 'uv.lock').read_bytes() == before
    assert not any(name == 'uv' for name, _ in calls())
    assert not any('checkout' in args or 'reset' in args
                   for _, args in calls())


@pytest.mark.parametrize(
    'stage', ['sync', 'metadata', 'assets', 'install', 'check', 'preflight'])
def test_installer_failure_stops_pipeline(installer, monkeypatch, stage):
    run, _, _, _ = installer
    monkeypatch.setenv('INSTALL_FAIL', stage)
    result = run()
    assert result.returncode != 0
    assert 'installation complete' not in result.stdout


@pytest.mark.parametrize('kind', ['PYTHON', 'STACK'])
def test_installer_rejects_incompatible_prepared_env(installer, tmp_path,
                                                     monkeypatch, kind):
    run, calls, _, stub = installer
    python = tmp_path / 'python'
    python.symlink_to(stub)
    monkeypatch.setenv('INSTALL_BAD_' + kind, '1')
    result = run('--python', python, '--skip-assets')
    assert result.returncode != 0
    assert not any(name == 'uv' for name, _ in calls())


def test_installer_plan_and_invalid_options_do_not_write(installer, tmp_path):
    run, calls, _, _ = installer
    assert run('--help').returncode == 0
    assert run('--dry-run').returncode == 0
    assert not (tmp_path / 'new checkout').exists()
    assert not calls()
    for args in [('--unknown', ), ('--python', ),
                 ('--python', '/x', '--env-path', '/y')]:
        assert run(*args).returncode != 0
    assert not calls()


def test_proc_snapshot_tolerates_exit_but_preserves_permission_errors(
        launch, monkeypatch):
    from evaluation.runtime import ProcessTree
    monkeypatch.setattr('os.listdir', lambda path: ['123', 'not-a-pid'])

    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, 'read_text', missing)
    assert ProcessTree.snapshot() == {}

    def denied(path):
        raise PermissionError(path)

    monkeypatch.setattr(Path, 'read_text', denied)
    with pytest.raises(PermissionError):
        ProcessTree.snapshot()


def test_proc_members_does_not_adopt_reused_pid(launch, monkeypatch):
    from evaluation.runtime import ProcessTree
    monkeypatch.setattr(ProcessTree, 'snapshot',
                        staticmethod(lambda: {10: (1, 10, 100)}))
    tree = ProcessTree(10)
    monkeypatch.setattr(
        ProcessTree, 'snapshot',
        staticmethod(lambda: {
            10: (1, 10, 200),
            11: (10, 10, 300)
        }))
    assert tree.members() == {}


def test_direct_distributed_dry_run_reads_actual_evaluation_mapping(launch):
    entry = importlib.import_module('distributed_eval')
    assert entry.main([
        '--config',
        str(ROOT / 'configs/robolab/droid.py'), '--simulator-restart-limit',
        '0', '--dry-run'
    ]) == 0


@pytest.mark.parametrize('extra', [
    ['--simulator-restart-limit', '1'],
    ['--resume-from', '/unused/recovery.json'],
    ['--cfg-options', 'themis.evaluation.environment.num_envs=True'],
    ['--cfg-options', 'themis.evaluation.environment.num_envs=2'],
])
def test_direct_distributed_rejects_robolab_incompatible_overrides(
        launch, extra):
    entry = importlib.import_module('distributed_eval')
    with pytest.raises(ValueError, match='RoboLab'):
        entry.main([
            '--config',
            str(ROOT / 'configs/robolab/droid.py'),
            '--simulator-restart-limit', '0', '--dry-run', *extra
        ])


def test_existing_launch_plans_keep_shared_retry_and_timeout_defaults(
        launch, tmp_path):
    settings = launch.load_config(write_launch(tmp_path))
    local = dict(settings['local'])
    local.pop('episode_timeout_s')
    local.pop('simulator_restart_limit')
    local['frozen_config'] = str(tmp_path / 'snapshot.py')
    local['python'] = '/usr/bin/python3'
    plan = dict(
        local=local,
        run='existing-plan',
        output=str(tmp_path),
        wait_s=1800,
        endpoint='tcp://127.0.0.1:15555')
    arguments = launch.simulator_arguments(plan)
    assert arguments[arguments.index('--episode-timeout-s') + 1] == '3600'
    assert arguments[arguments.index('--simulator-restart-limit') + 1] == '3'


@pytest.mark.parametrize('after_snapshot', [False, True])
def test_proc_permission_failure_reaps_owned_process_and_preserves_error(
        launch, tmp_path, monkeypatch, after_snapshot):
    import subprocess
    import sys

    from evaluation.runtime import ProcessRunner, ProcessTree

    runner = ProcessRunner(
        dict(
            local=dict(python=sys.executable, port=15558),
            output=str(tmp_path)))
    processes = []
    popen = subprocess.Popen
    read_text = Path.read_text

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    def denied(path, *args, **kwargs):
        if path.parts[:2] == ('/', 'proc'):
            raise PermissionError('restricted process table')
        return read_text(path, *args, **kwargs)

    if after_snapshot:
        # Deny one unrelated entry only after the initial complete snapshot.
        snapshot = ProcessTree.snapshot
        completed = False

        def late_denial(permission_errors=None):
            nonlocal completed
            if completed:
                if permission_errors is None:
                    raise PermissionError('restricted process table')
                permission_errors.append(
                    PermissionError('restricted process table'))
            completed = True
            return snapshot()

        monkeypatch.setattr(ProcessTree, 'snapshot', staticmethod(late_denial))
    else:
        monkeypatch.setattr(Path, 'read_text', denied)
    monkeypatch.setattr(subprocess, 'Popen', capture)
    try:
        with pytest.raises(PermissionError, match='restricted process table'):
            runner.run_logged(
                [sys.executable, '-c', 'import time; time.sleep(10)'],
                tmp_path / 'permission.log', None, 2)
        assert len(processes) == 1
        assert processes[0].poll() is not None
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
