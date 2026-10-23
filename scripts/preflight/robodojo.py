"""RoboDojo-specific source, package and generated robot-path checks."""

import importlib
import importlib.metadata
import sys
from pathlib import Path


def check_environment(root):
    """Inspect packages, source and asset paths without starting Isaac Sim."""
    root = Path(root).resolve(strict=True)
    for module in ('zmq', 'msgpack', 'mmengine', 'gymnasium', 'fluxthemis'):
        importlib.import_module(module)
    packages = {
        name: importlib.metadata.version(name)
        for name in ('torch', 'isaacsim', 'isaaclab', 'numpy', 'mmengine',
                     'pyzmq', 'msgpack', 'gymnasium')
    }
    required = ('env/global_configs.py', 'env_cfg/arx_x5.yml',
                'src/eval_client/eval_env.py',
                'src/eval_client/physx_warning_monitor.py',
                'task/RoboDojo/task_registry.py', 'Assets/Robots',
                'Assets/Object', 'Assets/Material', 'Assets/Eval_Layout')
    missing = [
        str(root / name) for name in required if not (root / name).exists()
    ]
    if missing:
        raise ValueError('Missing RoboDojo files: ' + ', '.join(missing))
    templates = list((root / 'Assets/Robots').rglob('*_tmp.yml'))
    if not templates:
        raise ValueError('No robot path templates in Assets/Robots')
    for template in templates:
        expected = template.read_text().replace('${ASSETS_PATH}', str(root))
        expected = expected.replace('$ASSETS_PATH', str(root))
        target = template.with_name(template.name.replace('_tmp.yml', '.yml'))
        if not target.is_file() or target.read_text() != expected:
            raise ValueError(f'Robot paths need preparation: {target}; run '
                             'utils/update_embodiment_config_path.py in '
                             f'{root} before evaluation')
    return dict(
        status='ok',
        python=sys.version,
        packages=packages,
        coverage='Python and simulator package versions; '
        'not repository sources, CUDA driver or installed binary contents')
