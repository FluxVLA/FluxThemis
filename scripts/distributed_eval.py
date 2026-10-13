#!/usr/bin/env python
"""Launch a single- or multi-node FluxThemis simulator task."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from mmengine import DictAction

from fluxthemis.transports.fluxvla_zmq import DEFAULT_ENDPOINT

_LOCAL_GPU_COUNT_ENV_VARS = ('NPROC_PER_NODE', 'MLP_WORKER_GPU')
DEFAULT_ROLLOUT_BIND = 'tcp://0.0.0.0:15558'
DEFAULT_STARTUP_TIMEOUT_S = 900.0
DEFAULT_EPISODE_TIMEOUT_S = 3600.0
DEFAULT_SIMULATOR_RESTART_LIMIT = 3


def _torchrun_environment() -> tuple[int, int, int, str]:
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size < 1:
        raise ValueError('WORLD_SIZE must be positive')
    if rank < 0 or rank >= world_size:
        raise ValueError('RANK must be in [0, WORLD_SIZE)')
    if local_rank < 0:
        raise ValueError('LOCAL_RANK must be non-negative')
    return (rank, world_size, local_rank,
            os.environ.get('MASTER_ADDR', '127.0.0.1'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Launch distributed FluxThemis evaluation over ZMQ.')
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--server-endpoint',
        default=None,
        help=f'Public FluxVLA endpoint (local default: {DEFAULT_ENDPOINT}).')
    parser.add_argument(
        '--simulator-gpus',
        default=None,
        help='Override auto-detected local simulator GPUs, for example 0,1.')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--video-dir', default=None)
    parser.add_argument('--save-videos', action='store_true')
    networking = parser.add_argument_group('advanced networking')
    networking.add_argument(
        '--rollout-bind',
        default=DEFAULT_ROLLOUT_BIND,
        help=f'Internal simulator-worker bind address (default: '
        f'{DEFAULT_ROLLOUT_BIND}).')
    networking.add_argument(
        '--rollout-endpoint',
        default=None,
        help='Address simulator workers connect to; defaults to the rollout '
        'port on MASTER_ADDR.')
    reliability = parser.add_argument_group('advanced reliability')
    reliability.add_argument(
        '--startup-timeout-s', type=float, default=DEFAULT_STARTUP_TIMEOUT_S)
    reliability.add_argument(
        '--episode-timeout-s', type=float, default=DEFAULT_EPISODE_TIMEOUT_S)
    reliability.add_argument(
        '--simulator-restart-limit',
        type=int,
        default=DEFAULT_SIMULATOR_RESTART_LIMIT,
        help='Maximum fresh-process retries for one episode or startup '
        'sequence (default: 3).')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default=None)
    return parser.parse_args(argv)


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(',') if item.strip())
    if not devices:
        raise ValueError('--simulator-gpus cannot be empty')
    if len(devices) != len(set(devices)):
        raise ValueError('--simulator-gpus cannot contain duplicates')
    return devices


def _connect_endpoint(bind: str, host: str) -> str:
    parsed = urlsplit(bind)
    if parsed.scheme != 'tcp' or parsed.port is None:
        raise ValueError('distributed endpoints must use tcp://host:port')
    return f'tcp://{host}:{parsed.port}'


def _discover_devices(
        environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Return local CUDA visibility tokens for simulator workers."""
    environ = os.environ if environ is None else environ
    worker_count = _local_gpu_count(environ)
    visible = environ.get('CUDA_VISIBLE_DEVICES')
    devices = None
    if visible is not None:
        stripped = visible.strip()
        if not stripped or stripped == '-1':
            devices = ()
        elif stripped.lower() != 'all':
            devices = _parse_devices(stripped)

    if devices is None:
        if worker_count is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    'Cannot discover simulator GPUs without torch; set '
                    'CUDA_VISIBLE_DEVICES, NPROC_PER_NODE, or '
                    'MLP_WORKER_GPU') from exc
            worker_count = torch.cuda.device_count()
        devices = tuple(str(index) for index in range(worker_count))

    if worker_count is not None:
        if worker_count > len(devices):
            raise ValueError(
                f'Requested {worker_count} local simulator workers, but only '
                f'{len(devices)} CUDA devices are visible')
        devices = devices[:worker_count]
    if not devices:
        raise RuntimeError('No CUDA devices are visible; pass '
                           '--simulator-gpus to select devices explicitly')
    return devices


def _local_gpu_count(environ: Mapping[str, str]) -> int | None:
    for name in _LOCAL_GPU_COUNT_ENV_VARS:
        value = environ.get(name)
        if value is None or not value.strip():
            continue
        try:
            count = int(value)
        except ValueError as exc:
            raise ValueError(f'{name} must be a positive integer') from exc
        if count <= 0:
            raise ValueError(f'{name} must be a positive integer')
        return count
    return None


def main(argv=None) -> int:
    from fluxthemis.engines.runners.launcher import \
        run_distributed_evaluation_rank

    args = parse_args(argv)
    rank, world_size, local_rank, master_addr = _torchrun_environment()
    config_path = str(Path(args.config).expanduser().resolve(strict=True))
    overrides = dict(args.cfg_options or {})
    if args.server_endpoint is not None:
        overrides['themis.transport.endpoint'] = args.server_endpoint
    if args.run_name is not None:
        overrides['themis.evaluation.run_name'] = args.run_name
    if args.video_dir is not None:
        overrides['themis.evaluation.video_dir'] = args.video_dir
    if args.save_videos:
        overrides['themis.evaluation.save_videos'] = True
    devices = (
        _parse_devices(args.simulator_gpus)
        if args.simulator_gpus is not None else _discover_devices())
    if local_rank >= len(devices):
        raise ValueError(
            f'LOCAL_RANK={local_rank} has no simulator GPU in {devices}')
    device = devices[local_rank]
    print(
        f'[FluxThemis] rank={rank} starting simulator worker on GPU {device}',
        flush=True)
    rollout_bind = args.rollout_bind
    rollout_endpoint = (
        args.rollout_endpoint or _connect_endpoint(rollout_bind, master_addr))
    run_distributed_evaluation_rank(
        config_path=config_path,
        device=device,
        rollout_bind=rollout_bind,
        rollout_endpoint=rollout_endpoint,
        config_overrides=overrides,
        startup_timeout_s=args.startup_timeout_s,
        episode_timeout_s=args.episode_timeout_s,
        simulator_restart_limit=args.simulator_restart_limit,
        rank=rank,
        world_size=world_size,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
