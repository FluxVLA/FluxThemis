"""Inspect the DROID checkout and packages without importing RoboLab/Isaac.

The static task/scene checks target the benchmark layout used by the adapter.
They do not execute registrations, open USD stages or validate rendering.
"""

import ast
import importlib.metadata
import re
import sys
from pathlib import Path

REQUIRED_SOURCES = (
    'robolab/constants.py',
    'robolab/core/task/task.py',
    'robolab/core/environments/factory.py',
    'robolab/core/environments/runtime.py',
    'robolab/core/environments/scene_fixture.py',
    'robolab/core/scenes/utils.py',
    'robolab/robots/droid.py',
    'robolab/variations/backgrounds.py',
    'robolab/registrations/droid/auto_env_registrations_jointpos.py',
)


def _require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing RoboLab input: {path}')
    if not path.stat().st_size:
        raise ValueError(f'Empty RoboLab input: {path}')
    with path.open('rb') as stream:
        if stream.read(128).startswith(
                b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Unmaterialized RoboLab LFS asset: {path}')
    return path


def _task_sources(root, task_name):
    tasks = {}
    directory = root / 'robolab/tasks/benchmark'
    for path in sorted(directory.rglob('*.py')):
        if path.name == '__init__.py' or 'generated' in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == 'Task'
                for base in node.bases)
        ]
        if len(classes) != 1:
            raise ValueError(f'Expected one static Task subclass in {path}')
        name = classes[0].name
        if name in tasks:
            raise ValueError(f'Duplicate RoboLab task: {name}')
        tasks[name] = (path, tree)
    if not tasks:
        raise ValueError(f'No RoboLab benchmark tasks found in {directory}')
    names = [task_name] if isinstance(task_name, str) else list(task_name)
    if (not names or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
            or ('all' in names and len(names) != 1)):
        raise ValueError('Invalid RoboLab task_name selection')
    if names == ['all']:
        names = list(tasks)
    missing = set(names) - tasks.keys()
    if missing:
        raise ValueError(f'Unknown RoboLab benchmark tasks: {sorted(missing)}')
    return {name: tasks[name] for name in names}


def _find_asset(directory, name):
    path = Path(name)
    if path.is_absolute():
        return _require_file(path)
    matches = [
        p for p in directory.rglob(path.name)
        if p.as_posix().endswith('/' + path.as_posix()) and not any(
            part.startswith('_') or part in ('not_used', 'tmp')
            for part in p.relative_to(directory).parts[:-1])
    ]
    if len(matches) != 1:
        raise ValueError(f'Expected one RoboLab asset {name} in {directory}; '
                         f'found {len(matches)}')
    return _require_file(matches[0])


def _scene_assets(root, tasks):
    scenes = set()
    for name, (source, tree) in tasks.items():
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ('import_scene',
                                 'import_scene_and_contact_object_list')
        ]
        if not calls:
            raise ValueError(f'No static scene call for RoboLab task {name}')
        for call in calls:
            if (not call.args or not isinstance(call.args[0], ast.Constant)
                    or not isinstance(call.args[0].value, str)
                    or not call.args[0].value):
                raise ValueError(f'Expected literal scene path in {source}')
            scenes.add(_find_asset(root / 'assets/scenes', call.args[0].value))
    unchecked = set()
    for scene in scenes:
        with scene.open('rb') as stream:
            ascii_usd = stream.read(8).startswith(b'#usda')
        if not ascii_usd:
            unchecked.add(str(scene))
            continue
        # Check authored file references only. Kit resolver names/URLs and
        # transitive USD/MDL dependencies still require runtime validation.
        # USD delete/reorder list edits do not introduce file dependencies.
        text = re.sub(
            r'(?m)^\s*(?:delete|reorder)\s+(?:payload|references)\s*='
            r'\s*(?:\[[^\]]*\]|[^\n]*)', '', scene.read_text())
        for reference in re.findall(r'@([^@]+)@', text):
            if '://' in reference or ('/' not in reference
                                      and reference.endswith('.mdl')):
                unchecked.add(reference)
            else:
                _require_file((scene.parent / reference).resolve())
    return sorted(map(str, scenes)), sorted(unchecked)


def check_environment(root, task_name='all'):
    root = Path(root).expanduser().resolve(strict=True)
    for name in REQUIRED_SOURCES:
        _require_file(root / name)
    tasks = _task_sources(root, task_name)
    packages = {
        name: importlib.metadata.version(name)
        for name in ('isaaclab', 'isaacsim', 'torch', 'numpy', 'scipy', 'h5py',
                     'opencv-python')
    }
    # RoboLab's DROID definition uses the IsaacLab 3 asset for XYZW runtimes.
    robot = ('franka_robotiq_2f_85_isaac60.usd'
             if int(packages['isaaclab'].split('.')[0]) >= 3 else
             'franka_robotiq_2f_85_flattened.usd')
    for name in (f'assets/robots/{robot}', 'assets/fixtures/franka_table.usd',
                 'assets/objects/object_catalog.json'):
        _require_file(root / name)
    background = _find_asset(root / 'assets/backgrounds', 'home_office.exr')
    scenes, unchecked = _scene_assets(root, tasks)
    return dict(
        status='ok',
        python=sys.version,
        packages=packages,
        task_sources={
            name: str(source)
            for name, (source, _) in tasks.items()
        },
        scenes=scenes,
        background=str(background),
        unchecked_assets=unchecked,
        asset_roots=dict(assets=str(root / 'assets')),
        coverage='Static benchmark task classes and scene paths; DROID, '
        'fixture, background and direct ASCII USD file references; package '
        'metadata. Not runtime registration, binary/transitive USD or MDL '
        'dependencies, CUDA/Vulkan, simulation or resume equivalence.')
