"""RoboTwin native dependency installation and startup failure contracts."""

import hashlib
import importlib
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def installer(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('robotwin.oidn')


@pytest.fixture
def oidn_installation(tmp_path, monkeypatch, installer):
    root = tmp_path / 'sapien'
    libraries = root / 'oidn_library'
    libraries.mkdir(parents=True)
    (root / '_oidn_tricks.py').write_text('\n'.join(
        'oidn_library/' + name.replace('2.3.3', '2.0.1')
        for name in installer.LIBRARIES[:2]))
    for name in installer.LIBRARIES:
        (libraries / name.replace('2.3.3', '2.0.1')).write_bytes(b'old')
    archive = tmp_path / 'oidn.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for name in installer.LIBRARIES:
            content = name.encode()
            member = tarfile.TarInfo(f'oidn-2.3.3.x86_64.linux/lib/{name}')
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(installer, 'SHA256',
                        hashlib.sha256(archive.read_bytes()).hexdigest())
    return archive, root


def test_oidn_install_replaces_library_set_and_is_repeatable(
        oidn_installation, installer):
    archive, root = oidn_installation
    cache = root / '__pycache__' / '_oidn_tricks.cpython-310.pyc'
    cache.parent.mkdir()
    cache.write_bytes(b'old loader bytecode')
    for _ in range(2):
        installer.install(archive, root)
        libraries = root / 'oidn_library'
        assert not list(libraries.glob('*2.0.1*'))
        assert '2.0.1' not in (root / '_oidn_tricks.py').read_text()
        assert not cache.exists()
        for name in installer.LIBRARIES:
            assert (libraries / name).read_bytes() == name.encode()
        assert (libraries / 'libOpenImageDenoise.so.2').resolve() == (
            libraries / installer.LIBRARIES[0])


@pytest.mark.parametrize('failure', ['checksum', 'loader', 'incomplete'])
def test_bad_oidn_input_preserves_installation(oidn_installation, failure,
                                               monkeypatch, installer):
    archive, root = oidn_installation
    if failure == 'checksum':
        archive.write_bytes(b'corrupt download')
    elif failure == 'loader':
        (root / '_oidn_tricks.py').write_text('custom loader')
    else:
        with tarfile.open(archive, 'w:gz'):
            pass
        monkeypatch.setattr(installer, 'SHA256',
                            hashlib.sha256(archive.read_bytes()).hexdigest())
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises((ValueError, KeyError)):
        installer.install(archive, root)
    assert {
        str(p): p.read_bytes()
        for p in root.rglob('*') if p.is_file()
    } == before


def test_oidn_fingerprint_tracks_symlink_targets(oidn_installation,
                                                 monkeypatch, installer):
    archive, root = oidn_installation
    monkeypatch.setattr(
        installer.importlib.metadata, 'distribution',
        lambda name: SimpleNamespace(locate_file=lambda relative: root))
    before = installer.fingerprint()
    installer.install(archive, root)
    installed = installer.fingerprint()
    assert before != installed
    library = root / 'oidn_library' / installer.LIBRARIES[-1]
    target = root / library.name
    library.rename(target)
    library.symlink_to(target)
    assert installer.fingerprint() == installed
    target.write_bytes(b'changed binary with same version')
    assert installer.fingerprint() != installed


@pytest.mark.parametrize('failed_gpu', [None, '2'])
def test_preflight_checks_selected_gpus_and_stops_on_failure(
        tmp_path, monkeypatch, failed_gpu):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / 'scripts'))
    launcher = importlib.import_module('run_evaluation')
    monkeypatch.setattr(launcher, 'simulator_arguments', lambda plan: [])
    checked = []

    def run_logged(command, logfile, progress, timeout):
        if 'robotwin.oidn' in command:
            assert command[-1] == 'check'
            gpu = command[1].split('=', 1)[1]
            checked.append(gpu)
            return int(gpu == failed_gpu)
        return 0

    runtime = SimpleNamespace(connect=lambda: None, run_logged=run_logged)
    progress = SimpleNamespace(show=lambda *a, **kw: None)
    plan = dict(
        output=str(tmp_path),
        local=dict(
            environment_type='RoboTwinEnvironment',
            gpus='2,GPU-test-uuid',
            python='/usr/bin/python',
            frozen_config='config.py'))
    if failed_gpu:
        with pytest.raises(RuntimeError, match='oidn_gpu_2.log'):
            launcher.preflight(plan, runtime, progress)
        assert checked == ['2']
    else:
        launcher.preflight(plan, runtime, progress)
        assert checked == ['2', 'GPU-test-uuid']
