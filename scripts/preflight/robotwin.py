"""Inspect RoboTwin inputs and check GPUs in isolated subprocesses."""

import importlib.metadata
import sys
from pathlib import Path

from robotwin.oidn import fingerprint as oidn_fingerprint

from fluxthemis.environments.sim.robotwin import RoboTwinEnvironment


def check_gpus(local, runtime, progress, output):
    for gpu in local['gpus'].split(','):
        logfile = output / f'oidn_gpu_{gpu}.log'
        command = [
            'env', f'CUDA_VISIBLE_DEVICES={gpu}', local['python'], '-m',
            'robotwin.oidn', 'check'
        ]
        if runtime.run_logged(command, logfile, progress, 120):
            raise RuntimeError(f'OIDN CUDA preflight failed; see {logfile}')
        progress.show(f'GPU {gpu} passed OIDN CUDA denoising check')


def check_environment(root, task_suite='all'):
    root = Path(root).resolve(strict=True)
    adapter = RoboTwinEnvironment(robotwin_root=root, task_suite=task_suite)
    tasks = adapter.tasks()
    required = [
        'envs/_base_task.py', 'task_config/_eval_step_limit.yml',
        'task_config/_embodiment_config.yml', 'task_config/_camera_config.yml',
        'description/utils/generate_episode_instructions.py',
        'description/objects_description',
        'script/update_embodiment_config_path.py', 'assets/background_texture',
        'assets/objects', 'assets/embodiments'
    ]
    required += [f'envs/{t.metadata["robotwin_task"]}.py' for t in tasks]
    required += [
        f'description/task_instruction/{t.metadata["robotwin_task"]}.json'
        for t in tasks
    ]
    required += [
        f'task_config/{t.metadata["robotwin_task_config"]}.yml' for t in tasks
    ]
    missing = sorted({name for name in required if not (root / name).exists()})
    if missing:
        raise ValueError(f'Incomplete RoboTwin checkout {root}: {missing}')
    for name in ('background_texture', 'objects', 'embodiments'):
        if not any(p.is_file() for p in (root / 'assets' / name).rglob('*')):
            raise ValueError(f'Empty RoboTwin assets: {name}')
    templates = list((root / 'assets/embodiments').rglob('*_tmp.yml'))
    if not templates:
        raise ValueError('No robot path templates in assets/embodiments')
    for template in templates:
        expected = template.read_text().replace('${ASSETS_PATH}', str(root))
        expected = expected.replace('$ASSETS_PATH', str(root))
        target = template.with_name(template.name.replace('_tmp.yml', '.yml'))
        if not target.is_file() or target.read_text() != expected:
            raise ValueError(f'Robot paths need preparation: {target}; run '
                             'script/update_embodiment_config_path.py in '
                             f'{root} before evaluation')
    # Parse the actual camera/embodiment configs used by setup, without setup.
    for setting in sorted({t.metadata['robotwin_setting'] for t in tasks}):
        adapter._build_args(setting)
    packages = {
        name: importlib.metadata.version(name)
        for name in ('torch', 'sapien', 'mplib', 'warp-lang', 'nvidia-curobo',
                     'numpy', 'scipy', 'toppra', 'open3d')
    }
    return dict(
        status='ok',
        python=sys.version,
        packages=packages,
        oidn=oidn_fingerprint(),
        asset_roots=dict(assets=str(root / 'assets')),
        coverage='Python and simulator package versions; '
        'OIDN loader/libraries; assets hashed by recovery; '
        'not repository sources, CUDA driver or all installed binary contents')
