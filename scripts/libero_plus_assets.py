#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

LIBERO_PLUS_ASSET_REPO = 'Sylvest/LIBERO-plus'
LIBERO_PLUS_ASSET_REVISION = \
    'dd2bd61b7d9a6fef1abc52d606e983b41886a149'
LIBERO_PLUS_ASSET_ARCHIVE = 'assets.zip'
LIBERO_PLUS_ASSET_SIZE = 6_395_849_578
LIBERO_PLUS_ASSET_SHA256 = (
    '96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf')
LIBERO_PLUS_ASSET_MARKER = 'LIBERO-plus-0/assets/'
LIBERO_PLUS_REQUIRED_DIRECTORIES = (
    'articulated_objects',
    'new_objects',
    'scenes',
    'stable_hope_objects',
    'stable_scanned_objects',
    'textures',
    'turbosquid_objects',
)
LIBERO_PLUS_REQUIRED_FILES = (
    'serving_region.xml',
    'wall_frames.stl',
    'wall.xml',
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(archive: Path,
                     expected_sha256: str = LIBERO_PLUS_ASSET_SHA256,
                     expected_size: int = LIBERO_PLUS_ASSET_SIZE) -> None:
    archive = archive.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(
            f'LIBERO-Plus asset archive does not exist: {archive}')
    actual_size = archive.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f'LIBERO-Plus asset archive has size {actual_size}; expected '
            f'{expected_size}')
    actual_sha256 = file_sha256(archive)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f'LIBERO-Plus asset checksum is {actual_sha256}; expected '
            f'{expected_sha256}')


def validate_assets(assets_dir: Path) -> None:
    assets_dir = assets_dir.expanduser().resolve()
    missing = [
        path for path in LIBERO_PLUS_REQUIRED_DIRECTORIES
        if not (assets_dir / path).is_dir()
    ]
    missing.extend(
        path for path in LIBERO_PLUS_REQUIRED_FILES
        if not (assets_dir / path).is_file())
    if missing:
        formatted = '\n'.join(f'  - {path}' for path in missing)
        raise FileNotFoundError(
            f'LIBERO-Plus assets are incomplete under {assets_dir}:\n'
            f'{formatted}')


def download_archive(cache_dir: Path, endpoint: str) -> Path:
    try:
        hf_hub_download = importlib.import_module(
            'huggingface_hub').hf_hub_download
    except ImportError as exc:
        raise ModuleNotFoundError(
            'huggingface-hub is required to download LIBERO-Plus assets') \
            from exc
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            repo_id=LIBERO_PLUS_ASSET_REPO,
            repo_type='dataset',
            filename=LIBERO_PLUS_ASSET_ARCHIVE,
            revision=LIBERO_PLUS_ASSET_REVISION,
            endpoint=endpoint,
            local_dir=str(cache_dir),
        ))
    validate_archive(archive)
    return archive


def extract_assets(archive: Path, destination: Path) -> None:
    """Safely extract into a temporary directory and atomically publish."""
    archive = archive.expanduser().resolve()
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix='.libero-plus-assets.', dir=destination.parent))
    backup = destination.with_name(f'.{destination.name}.previous')
    try:
        with zipfile.ZipFile(archive) as zip_file:
            prefix = _archive_prefix(zip_file)
            extracted_files = 0
            for member in zip_file.infolist():
                if member.is_dir() or not member.filename.startswith(prefix):
                    continue
                relative = _safe_relative_path(member, prefix)
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, target.open(
                        'xb') as output:
                    shutil.copyfileobj(source, output)
                extracted_files += 1
        if extracted_files == 0:
            raise ValueError('LIBERO-Plus asset archive contains no files')
        validate_assets(staging)

        if backup.exists() or backup.is_symlink():
            raise FileExistsError(
                f'Cannot publish assets while backup path exists: {backup}')
        destination_existed = destination.exists() or destination.is_symlink()
        if destination_existed:
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if destination_existed and not destination.exists():
                backup.rename(destination)
            raise
        if destination_existed:
            _remove_path(backup)
    except BaseException:
        _remove_path(staging)
        raise


def write_libero_config(checkout_root: Path, config_dir: Path) -> Path:
    checkout_root = checkout_root.expanduser().resolve()
    benchmark_root = checkout_root / 'libero_plus' / 'libero'
    classification = (
        benchmark_root / 'benchmark' / 'task_classification.json')
    if not classification.is_file():
        raise ValueError(f'Invalid LIBERO-Plus checkout root: {checkout_root}')
    paths = {
        'benchmark_root': benchmark_root,
        'bddl_files': benchmark_root / 'bddl_files',
        'init_states': benchmark_root / 'init_files',
        'datasets': checkout_root / 'libero_plus' / 'datasets',
        'assets': benchmark_root / 'assets',
    }
    config_dir = config_dir.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / 'config.yaml'
    lines = [
        f'{key}: {json.dumps(str(value.resolve()))}'
        for key, value in paths.items()
    ]
    _write_text_atomic(config_path, '\n'.join(lines) + '\n')
    return config_path


def _archive_prefix(zip_file: zipfile.ZipFile) -> str:
    prefixes = set()
    for member in zip_file.infolist():
        name = member.filename
        marker_index = name.find(LIBERO_PLUS_ASSET_MARKER)
        if marker_index >= 0:
            prefixes.add(name[:marker_index + len(LIBERO_PLUS_ASSET_MARKER)])
    if len(prefixes) != 1:
        raise ValueError(
            f'Unexpected LIBERO-Plus archive layout: {sorted(prefixes)}')
    return prefixes.pop()


def _safe_relative_path(member: zipfile.ZipInfo, prefix: str) -> PurePosixPath:
    name = member.filename
    if '\x00' in name:
        raise ValueError('Archive member contains a NUL byte')
    relative = PurePosixPath(name[len(prefix):])
    if (not relative.parts or relative.is_absolute()
            or any(part in {'', '.', '..'} for part in relative.parts)):
        raise ValueError(f'Unsafe archive member path: {name!r}')
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f'Archive member cannot be a symlink: {name!r}')
    return relative


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _write_text_atomic(path: Path, value: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.{path.name}.', dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            output.write(value)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate or download official LIBERO-Plus assets.')
    parser.add_argument('--assets-dir', type=Path)
    parser.add_argument('--checkout-root', type=Path)
    parser.add_argument('--config-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument(
        '--endpoint',
        default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--write-config-only', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_config_only:
        if args.checkout_root is None or args.config_dir is None:
            raise ValueError(
                '--write-config-only requires --checkout-root and '
                '--config-dir')
        config_path = write_libero_config(args.checkout_root, args.config_dir)
        print(f'Wrote LIBERO-Plus config to {config_path}')
        return 0
    if args.assets_dir is None:
        raise ValueError('--assets-dir is required')
    assets_dir = args.assets_dir.expanduser().resolve()
    if args.validate_only:
        validate_assets(assets_dir)
    else:
        if args.archive is not None:
            archive = args.archive.expanduser().resolve()
            validate_archive(archive)
        else:
            cache_dir = args.cache_dir or assets_dir.parent / 'downloads'
            archive = download_archive(cache_dir, args.endpoint)
        extract_assets(archive, assets_dir)
    if args.checkout_root is not None:
        if args.config_dir is None:
            raise ValueError('--config-dir is required with --checkout-root')
        write_libero_config(args.checkout_root, args.config_dir)
    print(f'Validated LIBERO-Plus assets under {assets_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
