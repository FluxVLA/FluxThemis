"""Resolve simulation launch settings without starting models or simulators."""

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/robodojo/remote.yaml'


def mapping(value, name, defaults):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f'{name}: unknown fields {sorted(map(str, unknown))}')
    return {**defaults, **value}


def integer(value, name, minimum=1, maximum=None):
    if (type(value) is not int or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f'{name} must be an integer >= {minimum}' +
                         (f' and <= {maximum}' if maximum else ''))
    return value


def path_value(value, name, base, resolve=True):
    if (not isinstance(value, str) or not value.strip()
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f'{name} must be a nonempty path')
    for match in re.finditer(r'\$(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_]*))',
                             value):
        variable = match.group(1) or match.group(2)
        if variable not in os.environ:
            raise ValueError(f'{name}: unset environment variable {variable}')
    path = Path(os.path.expandvars(value)).expanduser()
    path = path if path.is_absolute() else base / path
    return str(path.resolve()) if resolve else os.path.abspath(path)


def select_gpus(count=None, explicit=None, environ=None):
    environ = os.environ if environ is None else environ
    if count is not None:
        integer(count, 'gpus')
    allocated = environ.get('CUDA_VISIBLE_DEVICES', 'all')
    if explicit is not None:
        devices = explicit.split(',')
        if count is not None and len(devices) != count:
            raise ValueError('--gpus and --gpu-ids counts differ')
    else:
        count = count or 1
        devices = ([str(i) for i in range(count)]
                   if allocated == 'all' else allocated.split(',')[:count])
        if len(devices) != count:
            raise ValueError('GPU count exceeds CUDA_VISIBLE_DEVICES')
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('GPU IDs must be nonempty and unique')
    if any(not (d.isdecimal() or re.fullmatch(r'GPU-[A-Za-z0-9-]+', d))
           for d in devices):
        raise ValueError('GPU IDs must be indices or GPU UUIDs')
    if allocated != 'all' and any(
            d not in allocated.split(',') for d in devices):
        raise ValueError('Requested GPU is outside CUDA_VISIBLE_DEVICES')
    return devices


def read_yaml(path):
    import yaml

    class UniqueLoader(yaml.SafeLoader):

        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str) or key in result:
                    raise ValueError('YAML field names must be unique strings')
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    return yaml.load(path.read_text(), Loader=UniqueLoader)


def load_config(path, *, mode=None, overrides=None):
    """Accept launch YAML or a Python config containing themis.evaluation."""
    if mode not in (None, 'smoke', 'full'):
        raise ValueError('mode must be smoke, full or None')
    overrides = overrides or {}
    path = Path(path).expanduser().resolve(strict=True)
    data = {} if path.suffix == '.py' else read_yaml(path)
    data = mapping(data, 'launch',
                   dict(version=1, simulator={}, model={}, run={}, resume={}))
    integer(data['version'], 'version', maximum=1)
    sim = mapping(
        data['simulator'], 'simulator',
        dict(
            python=sys.executable,
            config=str(path) if path.suffix == '.py' else None,
            robodojo_root=None,
            robotwin_root=None,
            assets=None,
            gpu_ids=None,
            workers_per_gpu=1,
            task=None,
            episodes=None,
            cache=None,
            coordinator_port=15558,
            scheduling=None,
            save_videos=None))
    model = mapping(
        data['model'], 'model',
        dict(
            host=None,
            port=15555,
            wait_seconds=1800,
            result_dir=None,
            expected_workers=None))
    run = mapping(
        data['run'], 'run',
        dict(output_root=str(ROOT / 'work_dirs/evaluation'), max_minutes=0))
    resume = mapping(data['resume'], 'resume', dict(from_dir=None))
    for field in ('workers_per_gpu', 'scheduling'):
        if overrides.get(field) is not None:
            sim[field] = overrides[field]
    if overrides.get('output') is not None:
        run['output_root'] = str(
            Path(overrides['output']).expanduser().resolve())
    if overrides.get('save_videos') is not None:
        sim['save_videos'] = overrides['save_videos']
    config_path = path_value(sim['config'], 'simulator.config', path.parent)
    cfg = Config.fromfile(config_path, import_custom_modules=False)
    if overrides.get('cfg_options'):
        for key in overrides['cfg_options']:
            if not key.startswith(('themis.evaluation.', 'themis.transport.')):
                raise ValueError('Evaluation overrides must target '
                                 'themis.evaluation or themis.transport')
        cfg.merge_from_dict(overrides['cfg_options'])
    if 'themis' not in cfg or not all(
            k in cfg.themis for k in ('evaluation', 'transport')):
        raise ValueError(
            'Config requires themis.evaluation and themis.transport')
    # Model construction, checkpoints and preprocessing stay in the model task.
    cfg = Config(
        dict(themis={
            k: v
            for k, v in cfg.themis.items() if k != 'server'
        }))
    evaluation = cfg.themis.evaluation
    environment = evaluation.environment
    kind = environment['type']
    robotwin_tasks = None
    if kind == 'RoboTwinEnvironment':
        from fluxthemis.environments.sim.robotwin import RoboTwinEnvironment
        robotwin_tasks = [
            t.task_id for t in RoboTwinEnvironment(**{
                k: v
                for k, v in environment.items() if k != 'type'
            }).tasks()
        ]
    ids = sim['gpu_ids']
    if ids is not None:
        if (not isinstance(ids, list) or not ids or any(
                type(i) not in (int, str) or (type(i) is int and i < 0)
                for i in ids)):
            raise ValueError(
                'simulator.gpu_ids must be a list of GPU indices/UUIDs')
        ids = ','.join(map(str, ids))
    if overrides.get('gpus') is not None and overrides.get('gpu_ids') is None:
        ids = None
    explicit = overrides.get('gpu_ids')
    devices = select_gpus(
        overrides.get('gpus'), explicit if explicit is not None else ids)
    workers = integer(sim['workers_per_gpu'], 'simulator.workers_per_gpu')
    total_workers = len(devices) * workers
    scheduling = sim['scheduling']
    if scheduling is None:
        strict = evaluation.get('strict_worker_assignment', False)
        if type(strict) is not bool:
            raise ValueError('strict_worker_assignment must be boolean')
        scheduling = 'fixed' if strict else 'dynamic'
    if scheduling not in ('dynamic', 'fixed'):
        raise ValueError('scheduling must be dynamic or fixed')
    evaluation.strict_worker_assignment = scheduling == 'fixed'
    task, episodes = sim['task'], sim['episodes']
    if mode == 'smoke':
        smoke_tasks = {
            'RoboDojoEnvironment': 'put_bottles_into_dustbin',
            'LiberoEnvironment': '0'
        }
        if robotwin_tasks:
            smoke_tasks[kind] = robotwin_tasks[0]
        if kind not in smoke_tasks:
            raise ValueError(
                'Set a smoke task explicitly for this environment')
        task = smoke_tasks[kind]
        episodes = total_workers
    elif mode == 'full':
        task, episodes = 'all', None
    if task is not None:
        if robotwin_tasks is not None:
            valid_task = task == 'all' or task in robotwin_tasks
        else:
            valid_task = isinstance(task, str) and re.fullmatch(
                r'[A-Za-z0-9_]+', task)
        if not valid_task:
            raise ValueError('simulator.task must be a task ID or all')
        if kind == 'RoboDojoEnvironment':
            environment.task_name = task
            evaluation.task_ids = None
        else:
            evaluation.task_ids = None if task == 'all' else [task]
    if episodes is not None:
        integer(episodes, 'simulator.episodes')
        if task in (None, 'all'):
            raise ValueError('An episode override requires one simulator.task')
        evaluation.episodes_per_task = episodes
        # ConfigDict.setdefault can return the original default dict instead
        # of its stored copy. Assign explicitly so new overrides are retained.
        evaluation.episodes_per_task_overrides = {
            **evaluation.get('episodes_per_task_overrides', {}), task: episodes
        }
    if robotwin_tasks is not None and evaluation.get('task_ids'):
        selected = evaluation.task_ids
        selected = ([t.strip() for t in selected.split(',')
                     if t.strip()] if isinstance(selected, str) else selected)
        if any(t not in robotwin_tasks for t in selected):
            raise ValueError('RoboTwin task_ids must belong to task_suite')
    robodojo_root = robotwin_root = assets = None
    if kind == 'RoboDojoEnvironment':
        value = sim['robodojo_root'] or environment.get(
            'robodojo_root') or os.environ.get('ROBODOJO_ROOT') or str(
                ROOT.parent / 'RoboDojo')
        robodojo_root = path_value(value, 'simulator.robodojo_root',
                                   path.parent)
        environment.robodojo_root = robodojo_root
        assets = path_value(
            sim['assets'] or str(Path(robodojo_root) / 'Assets'),
            'simulator.assets', path.parent)
    elif sim['robodojo_root'] is not None or sim['assets'] is not None:
        raise ValueError(
            'robodojo_root/assets launch fields apply only to RoboDojo')
    if kind == 'RoboTwinEnvironment':
        value = sim['robotwin_root'] or environment.get(
            'robotwin_root') or os.environ.get('ROBOTWIN_ROOT') or str(
                ROOT.parent / 'RoboTwin')
        robotwin_root = path_value(value, 'simulator.robotwin_root',
                                   path.parent)
        environment.robotwin_root = robotwin_root
    elif sim['robotwin_root'] is not None:
        raise ValueError('robotwin_root launch field applies only to RoboTwin')
    coord = integer(
        sim['coordinator_port'], 'simulator.coordinator_port', maximum=65534)
    endpoint = overrides.get('server_endpoint')
    if endpoint is None and model['host'] is not None:
        host = model['host']
        if not isinstance(host, str) or not re.fullmatch(
                r'[A-Za-z0-9_.-]+', host):
            raise ValueError('model.host must be a hostname or IPv4 address')
        port = integer(model['port'], 'model.port', maximum=65535)
        endpoint = f'tcp://{host}:{port}'
    endpoint = endpoint or cfg.themis.transport.get('endpoint',
                                                    'tcp://127.0.0.1:15555')
    parsed = urlsplit(endpoint)
    if (parsed.scheme != 'tcp' or not parsed.hostname or parsed.port is None
            or parsed.path or parsed.query or parsed.fragment
            or parsed.username is not None or parsed.port < 1):
        raise ValueError('Model endpoint must be tcp://host:port')
    if parsed.hostname in ('0.0.0.0', '*'):
        raise ValueError('Model endpoint must use a reachable host address')
    if (parsed.hostname == 'localhost' or
            parsed.hostname.startswith('127.')) and parsed.port in (coord,
                                                                    coord + 1):
        raise ValueError('Model port conflicts with simulator/torchrun port')
    cfg.themis.transport.endpoint = endpoint
    integer(model['wait_seconds'], 'model.wait_seconds')
    cfg.themis.transport.connect_timeout_s = model['wait_seconds']
    if model['expected_workers'] is not None:
        integer(model['expected_workers'], 'model.expected_workers')
    for section, key in ((model, 'result_dir'), (resume, 'from_dir')):
        if section[key] is not None:
            section[key] = path_value(section[key], key, path.parent)
    minutes = integer(run['max_minutes'], 'run.max_minutes', minimum=0)
    if sim['save_videos'] is not None and type(sim['save_videos']) is not bool:
        raise ValueError('simulator.save_videos must be boolean')
    save_videos = (
        sim['save_videos'] if sim['save_videos'] is not None else
        evaluation.get('save_videos', False))
    evaluation.save_videos = save_videos
    local = dict(
        python=path_value(
            sim['python'], 'simulator.python', path.parent, resolve=False),
        config=config_path,
        robodojo_root=robodojo_root,
        robotwin_root=robotwin_root,
        assets=assets,
        cache=path_value(sim['cache'], 'simulator.cache', path.parent)
        if sim['cache'] else None,
        gpus=','.join(devices),
        workers=workers,
        task=task,
        episodes=episodes,
        port=coord,
        scheduling=scheduling,
        environment_type=kind,
        save_videos=save_videos)
    return dict(
        source=str(path),
        source_text=path.read_text(),
        config=cfg,
        local=local,
        model=model,
        endpoint=endpoint,
        wait_s=model['wait_seconds'],
        limit_s=minutes * 60 if minutes else None,
        resume=resume,
        simulator_workers=total_workers,
        output_root=path_value(run['output_root'], 'run.output_root',
                               path.parent))
