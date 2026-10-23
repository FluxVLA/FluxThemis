"""Evaluation recovery preparation and input identity checks (no GPU work)."""

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'episode_recovery', ROOT / 'fluxthemis/engines/runners/resume.py')
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def file_hash(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def asset_identity(root):
    """Hash asset content; ignore task-private generated robot paths."""
    root = Path(root)
    entries = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        template = path.with_name(path.stem + '_tmp' + path.suffix)
        generated = (
            rel.parts[0] == 'Robots'
            or (rel.parts[0] == 'embodiments' and path.suffix == '.yml'))
        if generated and template.is_file():
            continue
        stat = path.stat()
        entries.append([str(rel), stat.st_size, file_hash(path)])
    if not entries:
        raise ValueError('Assets directory is empty')
    return recovery.digest(entries)


def input_assets(plan, runtime):
    local = plan['local']
    if local.get('assets'):
        return asset_identity(local['assets'])
    roots = runtime.identity().get('asset_roots', {})
    return recovery.digest({
        name: asset_identity(path)
        for name, path in roots.items()
    })


def evaluation_settings(cfg):
    """Compare evaluation settings, allowing host and output relocation."""
    value = copy.deepcopy(cfg.themis.to_dict())
    value.pop('server', None)
    evaluation = value['evaluation']
    evaluation.setdefault('strict_worker_assignment', False)
    evaluation.setdefault('task_ids', None)
    evaluation['environment'].pop('robodojo_root', None)
    evaluation['environment'].pop('robotwin_root', None)
    for key in ('run_name', 'video_dir', 'save_videos', 'resume_path'):
        evaluation.pop(key, None)
    evaluation['reporter'].pop('event_log', None)
    for key in ('endpoint', 'connect_timeout_s', 'request_timeout_s'):
        value['transport'].pop(key, None)
    return value


class RunLock:

    def __init__(self, directory):
        self.stream = (Path(directory) / 'run.lock').open('a+')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError(
                'This evaluation directory is still in use') from None

    def close(self):
        self.stream.close()


def assert_stopped(directory):
    name = Path(directory).name.encode()
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if path.parent.name == str(os.getpid()):
            continue
        try:
            command = path.read_bytes()
        except OSError:
            continue
        if b'--run-name\0' + name + b'\0' in command:
            raise RuntimeError('The source simulator is still running; '
                               'resume only after it stops')


def freeze_inputs(plan, runtime):
    output = Path(plan['output'])
    path = output / 'recovery_manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    source = Path(plan['local'].get('frozen_config', plan['local']['config']))
    snapshot = output / 'config_snapshot.py'
    if source.resolve() != snapshot.resolve():
        # Flatten inherited configs while preserving model-independent options.
        from mmengine import Config
        Config.fromfile(
            str(source), import_custom_modules=False).dump(str(snapshot))
    manifest = dict(
        schema_version=2,
        captured_at=time.time(),
        config_sha256=file_hash(snapshot),
        assets_identity=input_assets(plan, runtime),
        native_identity=runtime.identity(),
        model_identity=None)
    recovery.atomic_json(path, manifest)
    plan['local']['frozen_config'] = str(snapshot)
    return manifest


def latest_model_identity(plan):
    """Model identity is independent of optional live resource samples."""
    directory = plan.get('model_result_dir')
    if directory is None:
        return None
    path = Path(directory) / 'model_identity.json'
    value = json.loads(path.read_text())
    if value.get('deployment_id') != plan['deployment']:
        raise ValueError('Model identity does not match the connected service')
    if not isinstance(value.get('fingerprint'),
                      str) or not value['fingerprint']:
        raise ValueError('Model identity requires a nonempty fingerprint')
    return value


def verify_model(plan):
    path = Path(plan['output']) / 'recovery_manifest.json'
    manifest = json.loads(path.read_text())
    current = latest_model_identity(plan)
    expected = manifest.get('model_identity')
    if plan.get('resume'):
        if expected is None or current is None:
            raise ValueError('Resume requires original and current model '
                             'fingerprints via model.result_dir')
        if expected['fingerprint'] != current['fingerprint']:
            raise ValueError('Resume rejected: model inputs differ')
    manifest['model_identity'] = current
    recovery.atomic_json(path, manifest)


def prepare_resume(source, settings):
    source = Path(source).expanduser().resolve(strict=True)
    assert_stopped(source)
    lock = RunLock(source)
    try:
        original = json.loads((source / 'plan.json').read_text())
        bundle = recovery.build_bundle(source)
        events = recovery.read_journal(source / 'events.jsonl')
        if (len(bundle['completed']) == bundle['start']['total_episodes']
                and events and events[-1]['event_type'] == 'run_end'
                and events[-1]['payload']['status'] == 'finished'):
            print(
                'All episodes already completed; no new task needed.',
                flush=True)
            lock.close()
            return None, None
        manifest = json.loads((source / 'recovery_manifest.json').read_text())
        if manifest['schema_version'] != 2:
            raise ValueError('Resume requires a current frozen input manifest')
        for field in ('task', 'episodes'):
            if original['local'][field] != settings['local'][field]:
                raise ValueError(f'Resume cannot change simulator.{field}')
        snapshot = source / 'config_snapshot.py'
        if file_hash(snapshot) != manifest['config_sha256']:
            raise ValueError('Frozen simulator config changed')
        from mmengine import Config
        frozen = Config.fromfile(str(snapshot), import_custom_modules=False)
        if evaluation_settings(frozen) != evaluation_settings(
                settings['config']):
            raise ValueError('Resume cannot change evaluation settings')
        name = 'evaluation_resume_' + time.strftime(
            '%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
        output = Path(settings['output_root']) / name
        output.mkdir(parents=True)
        plan = copy.deepcopy(original)
        plan['local'] = copy.deepcopy(settings['local'])
        plan['simulator_workers'] = settings['simulator_workers']
        for key in ('deployment', 'model_workers'):
            plan.pop(key, None)
        plan.update(
            run=name,
            output=str(output),
            schema_version=4,
            resume=dict(
                source=str(source),
                completed=len(bundle['completed']),
                total=bundle['start']['total_episodes'],
                elapsed_execution_s=bundle['elapsed_execution_s']),
            lineage=[*original.get('lineage', []),
                     str(source)])
        shutil.copyfile(snapshot, output / 'config_snapshot.py')
        plan['local']['frozen_config'] = str(output / 'config_snapshot.py')
        recovery.atomic_json(output / 'recovery_manifest.json', manifest)
        recovery.atomic_json(output / 'resume_input.json', bundle)
        recovery.atomic_json(output / 'plan.json', plan)
        count = len(bundle['completed'])
        total = bundle['start']['total_episodes']
        print(
            f'[RESUME] preserved={count}/{total}; pending={total - count}; '
            f'output={output}',
            flush=True)
        return plan, lock
    except BaseException:
        lock.close()
        raise


def verify_environment(plan, runtime):
    manifest = json.loads(
        (Path(plan['output']) / 'recovery_manifest.json').read_text())
    expected = manifest['native_identity']
    current = runtime.identity()
    # Legacy fingerprints include repository sources and cannot distinguish
    # code updates from dependency changes. Compare recorded runtime fields.
    for field in ('python', 'packages', 'oidn'):
        if field in expected and expected[field] != current.get(field):
            raise ValueError(
                f'Simulator {field} changed since the original evaluation')
    if input_assets(plan, runtime) != manifest['assets_identity']:
        raise ValueError('Assets/layouts changed since the source evaluation')
