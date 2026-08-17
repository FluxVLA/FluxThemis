import hashlib
import importlib.util
import stat
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / 'scripts' / 'libero_plus_assets.py'
SPEC = importlib.util.spec_from_file_location('libero_plus_assets',
                                              MODULE_PATH)
assets_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets_module)


def _write_archive(path, unsafe_member=None, symlink_member=None):
    prefix = f'root/{assets_module.LIBERO_PLUS_ASSET_MARKER}'
    with zipfile.ZipFile(path, 'w') as archive:
        for directory in assets_module.LIBERO_PLUS_REQUIRED_DIRECTORIES:
            archive.writestr(f'{prefix}{directory}/marker.txt', 'ok')
        for filename in assets_module.LIBERO_PLUS_REQUIRED_FILES:
            archive.writestr(f'{prefix}{filename}', 'ok')
        if unsafe_member is not None:
            archive.writestr(f'{prefix}{unsafe_member}', 'unsafe')
        if symlink_member is not None:
            info = zipfile.ZipInfo(f'{prefix}{symlink_member}')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, '/tmp/outside')


def test_archive_hash_and_size_are_both_validated(tmp_path):
    archive = tmp_path / 'assets.zip'
    archive.write_bytes(b'official-archive-placeholder')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    assets_module.validate_archive(
        archive, expected_sha256=digest, expected_size=archive.stat().st_size)

    with pytest.raises(ValueError, match='size'):
        assets_module.validate_archive(
            archive, expected_sha256=digest, expected_size=1)
    with pytest.raises(ValueError, match='checksum'):
        assets_module.validate_archive(
            archive,
            expected_sha256='0' * 64,
            expected_size=archive.stat().st_size,
        )


def test_safe_extraction_validates_and_atomically_replaces_destination(
        tmp_path):
    archive = tmp_path / 'assets.zip'
    _write_archive(archive)
    destination = tmp_path / 'assets'
    destination.mkdir()
    (destination / 'old-sentinel').write_text('old', encoding='utf-8')

    assets_module.extract_assets(archive, destination)

    assets_module.validate_assets(destination)
    assert not (destination / 'old-sentinel').exists()
    assert not destination.with_name('.assets.previous').exists()
    assert not list(tmp_path.glob('.libero-plus-assets.*'))


@pytest.mark.parametrize('unsafe_member', ['../outside', 'safe/../../outside'])
def test_extraction_rejects_path_traversal_without_touching_destination(
        tmp_path, unsafe_member):
    archive = tmp_path / 'assets.zip'
    _write_archive(archive, unsafe_member=unsafe_member)
    destination = tmp_path / 'assets'
    destination.mkdir()
    sentinel = destination / 'sentinel'
    sentinel.write_text('preserved', encoding='utf-8')

    with pytest.raises(ValueError, match='Unsafe archive member'):
        assets_module.extract_assets(archive, destination)

    assert sentinel.read_text(encoding='utf-8') == 'preserved'
    assert not (tmp_path / 'outside').exists()


def test_extraction_rejects_symlinks(tmp_path):
    archive = tmp_path / 'assets.zip'
    _write_archive(archive, symlink_member='unsafe-link')

    with pytest.raises(ValueError, match='cannot be a symlink'):
        assets_module.extract_assets(archive, tmp_path / 'assets')


def test_config_is_isolated_and_written_atomically(tmp_path):
    checkout = tmp_path / 'checkout'
    classification = (
        checkout / 'libero_plus' / 'libero' / 'benchmark' /
        'task_classification.json')
    classification.parent.mkdir(parents=True)
    classification.write_text('{}', encoding='utf-8')
    config_dir = tmp_path / 'config' / 'libero-plus'

    config_path = assets_module.write_libero_config(checkout, config_dir)

    config = config_path.read_text(encoding='utf-8')
    assert str(checkout / 'libero_plus' / 'libero' / 'assets') in config
    assert config_path == config_dir / 'config.yaml'
    assert not list(config_dir.glob('.config.yaml.*'))
