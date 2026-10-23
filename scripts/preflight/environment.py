"""Inspect simulator dependencies, assets and layouts without starting CUDA."""

import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def check_environment(config_path):
    """Inspect the simulator environment without CUDA."""
    from mmengine import Config
    cfg = Config.fromfile(config_path, import_custom_modules=False)
    environment = cfg.themis.evaluation.environment
    for module in ('zmq', 'msgpack', 'mmengine', 'gymnasium', 'fluxthemis'):
        importlib.import_module(module)
    kind = environment['type']
    assets = {}
    if kind == 'RoboDojoEnvironment':
        from .robodojo import check_environment as check_robodojo
        root = os.environ.get('ROBODOJO_ROOT') or environment['robodojo_root']
        identity = check_robodojo(root)
        assets['assets'] = str(Path(root) / 'Assets')
    elif kind == 'RoboTwinEnvironment':
        from .robotwin import check_environment as check_robotwin
        root = os.environ.get('ROBOTWIN_ROOT') or environment['robotwin_root']
        identity = check_robotwin(root, environment.get('task_suite', 'all'))
        assets = identity.pop('asset_roots')
    else:
        packages = sorted((d.metadata['Name'], d.version)
                          for d in importlib.metadata.distributions()
                          if d.metadata['Name'])
        if kind == 'LiberoEnvironment':
            from libero.libero import get_libero_path
            assets = {
                name: get_libero_path(name)
                for name in ('assets', 'bddl_files', 'init_states')
            }
            for path in assets.values():
                if not Path(path).is_dir():
                    raise ValueError(f'LIBERO input directory missing: {path}')
        identity = dict(
            status='ok',
            python=sys.version,
            packages=packages,
            coverage='Python/package versions and declared simulator assets')
    identity['asset_roots'] = assets
    return identity


def check_layouts(config):
    from fluxthemis.engines.runners import config as runner_config
    from fluxthemis.engines.utils import build_environment_from_cfg
    cfg = runner_config.load_distributed_evaluation_config(config)
    if (cfg['environment']['type'] == 'RoboDojoEnvironment'
            and os.environ.get('ROBODOJO_ROOT')):
        cfg['environment']['robodojo_root'] = os.environ['ROBODOJO_ROOT']
    environment = build_environment_from_cfg(cfg['environment'])
    if hasattr(environment, 'validate_layouts'):
        counts = environment.validate_layouts(
            cfg.get('episodes_per_task', 1),
            cfg.get('episodes_per_task_overrides'))
        print(json.dumps(dict(status='ok', layout_counts=counts)))


if __name__ == '__main__':
    print(json.dumps(check_environment(sys.argv[1])))
