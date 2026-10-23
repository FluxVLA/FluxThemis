"""Install and validate SAPIEN's pinned OIDN runtime for RoboTwin."""

import argparse
import ctypes as ct
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = '2.3.3'
ARCHIVE = f'oidn-{VERSION}.x86_64.linux.tar.gz'
URL = ('https://github.com/RenderKit/oidn/releases/download/'
       f'v{VERSION}/{ARCHIVE}')
SHA256 = '3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f'
LIBRARIES = tuple(f'libOpenImageDenoise{suffix}.so.{VERSION}'
                  for suffix in ('', '_core', '_device_cuda'))

VERSION_NUMBER = sum(
    int(v) * scale for v, scale in zip(VERSION.split('.'), (10000, 100, 1)))


def install(archive, sapien_root):
    """Validate the archive and loader before replacing any installed files."""
    archive, sapien_root = Path(archive), Path(sapien_root)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('OIDN archive SHA256 mismatch')
    loader = sapien_root / '_oidn_tricks.py'
    source = loader.read_text()
    patched = source.replace('2.0.1', VERSION)
    if any(f'oidn_library/{name}' not in patched for name in LIBRARIES[:2]):
        raise ValueError('Unexpected SAPIEN OIDN loader; no files changed')
    libraries = sapien_root / 'oidn_library'
    with tempfile.TemporaryDirectory(dir=sapien_root) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive) as bundle:
            prefix = f'oidn-{VERSION}.x86_64.linux/'
            members = [
                bundle.getmember(prefix + 'lib/' + name) for name in LIBRARIES
            ]
            members.extend(
                m for m in bundle.getmembers()
                if m.name.startswith(prefix +
                                     'doc/') and m.name.endswith('.txt'))
            for member in members:
                name = Path(member.name).name
                if not member.isfile():
                    raise ValueError(f'Expected a regular OIDN file: {name}')
                with bundle.extractfile(member) as src:
                    with (staging / name).open('wb') as dst:
                        shutil.copyfileobj(src, dst)
        # Remove old versions and symlinks together: a stale CUDA plugin can
        # otherwise load a second core library into the same process.
        for path in libraries.glob('*OpenImageDenoise*'):
            path.unlink()
        for path in staging.iterdir():
            path.replace(libraries / path.name)
        for name in ('libOpenImageDenoise.so', 'libOpenImageDenoise.so.2'):
            (libraries / name).symlink_to(LIBRARIES[0])
        loader.write_text(patched)
        for cache in (sapien_root / '__pycache__').glob('_oidn_tricks.*.pyc'):
            cache.unlink()
    print(f'SAPIEN OIDN main/core/CUDA runtime installed: {VERSION}')


def fingerprint():
    """Hash the loader and libraries without importing SAPIEN or CUDA."""
    root = Path(
        importlib.metadata.distribution('sapien').locate_file('sapien'))
    paths = [root / '_oidn_tricks.py']
    paths.extend(sorted((root / 'oidn_library').glob('*OpenImageDenoise*')))
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def check_cuda():
    import numpy as np
    from sapien import _oidn_tricks

    library = _oidn_tricks.oidn_dll
    ptr, size = ct.c_void_p, ct.c_size_t
    signatures = {
        'oidnNewDevice': (ptr, [ct.c_int]),
        'oidnCommitDevice': (None, [ptr]),
        'oidnGetDeviceInt': (ct.c_int, [ptr, ct.c_char_p]),
        'oidnGetDeviceError': (ct.c_int, [ptr, ct.POINTER(ct.c_char_p)]),
        'oidnReleaseDevice': (None, [ptr]),
        'oidnNewBuffer': (ptr, [ptr, size]),
        'oidnWriteBuffer': (None, [ptr, size, size, ptr]),
        'oidnReadBuffer': (None, [ptr, size, size, ptr]),
        'oidnReleaseBuffer': (None, [ptr]),
        'oidnNewFilter': (ptr, [ptr, ct.c_char_p]),
        'oidnSetFilterImage':
        (None, [ptr, ct.c_char_p, ptr, ct.c_int, size, size, size, size,
                size]),
        'oidnCommitFilter': (None, [ptr]),
        'oidnExecuteFilter': (None, [ptr]),
        'oidnReleaseFilter': (None, [ptr]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(library, name)
        function.restype, function.argtypes = result, arguments

    def check_error(device):
        message = ct.c_char_p()
        error = library.oidnGetDeviceError(device, ct.byref(message))
        if error:
            raise RuntimeError(f'OIDN error {error}: {message.value!r}')

    device = library.oidnNewDevice(3)  # OIDN_DEVICE_TYPE_CUDA; never auto/CPU.
    buffers, filter_handle = [], None
    try:
        check_error(device)
        if not device:
            raise RuntimeError('OIDN returned no CUDA device')
        library.oidnCommitDevice(device)
        check_error(device)
        version = library.oidnGetDeviceInt(device, b'version')
        check_error(device)
        if version != VERSION_NUMBER:
            raise RuntimeError(
                f'OIDN {VERSION} required; loaded version {version}')
        noisy = np.random.default_rng(0).normal(.5, .05,
                                                (64, 64, 3)).astype('f4')
        denoised = np.empty_like(noisy)
        for _ in range(2):
            buffers.append(library.oidnNewBuffer(device, noisy.nbytes))
            check_error(device)
            if not buffers[-1]:
                raise RuntimeError('OIDN buffer allocation failed')
        library.oidnWriteBuffer(buffers[0], 0, noisy.nbytes, noisy.ctypes.data)
        check_error(device)
        filter_handle = library.oidnNewFilter(device, b'RT')
        check_error(device)
        if not filter_handle:
            raise RuntimeError('OIDN filter creation failed')
        for name, buffer in zip((b'color', b'output'), buffers):
            # OIDN_FORMAT_FLOAT3: packed RGB with default pixel/row strides.
            library.oidnSetFilterImage(filter_handle, name, buffer, 3, 64, 64,
                                       0, 0, 0)
            check_error(device)
        library.oidnCommitFilter(filter_handle)
        check_error(device)
        library.oidnExecuteFilter(filter_handle)
        check_error(device)
        library.oidnReadBuffer(buffers[1], 0, denoised.nbytes,
                               denoised.ctypes.data)
        check_error(device)
        if (not np.isfinite(denoised).all()
                or abs(float(denoised.mean()) - .5) > .05
                or not 0 < float(denoised.std()) < float(noisy.std())):
            raise RuntimeError(
                'OIDN CUDA filter did not denoise the test image')
        loaded = sorted({
            line.split()[-1]
            for line in Path('/proc/self/maps').read_text().splitlines()
            if 'libOpenImageDenoise' in line
        })
        if ({Path(p).name
             for p in loaded} != set(LIBRARIES)
                or len(loaded) != len(LIBRARIES)):
            raise RuntimeError(f'Mixed or incomplete OIDN libraries: {loaded}')
        return dict(
            status='ok',
            version=version,
            device='CUDA',
            loaded=loaded,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            input_std=float(noisy.std()),
            output_std=float(denoised.std()),
            filter_executed=True)
    finally:
        if filter_handle:
            library.oidnReleaseFilter(filter_handle)
        for buffer in buffers:
            if buffer:
                library.oidnReleaseBuffer(buffer)
        if device:
            library.oidnReleaseDevice(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    install_parser = commands.add_parser(
        'install', help='Install the pinned runtime')
    commands.add_parser(
        'check', help='Execute CUDA denoising on the visible GPU')
    install_parser.add_argument(
        '--archive',
        type=Path,
        help='Use a local official archive (SHA256 verified)')
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check_cuda()))
        return
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('The pinned OIDN archive requires Linux x86-64')
    # Finding the package does not import SAPIEN or preload its old libraries.
    spec = importlib.util.find_spec('sapien')
    if spec is None:
        parser.error(
            'Install sapien==3.0.0b1 in this Python environment first')
    sapien_root = Path(spec.origin).parent
    if args.archive:
        install(args.archive, sapien_root)
    else:
        with tempfile.TemporaryDirectory(prefix='robotwin-oidn-') as temporary:
            archive = Path(temporary) / ARCHIVE
            with urllib.request.urlopen(URL, timeout=120) as response:
                with archive.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
            install(archive, sapien_root)


if __name__ == '__main__':
    main()
