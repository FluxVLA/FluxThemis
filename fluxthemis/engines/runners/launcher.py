"""Distributed evaluation launcher and process topology."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import build_distributed_evaluation_run
from .coordinator import DistributedEvaluationCoordinator
from .episode import configure_worker_lifetime
from .worker import SimulatorWorkerSupervisor


def run_distributed_evaluation_rank(*,
                                    config_path: str,
                                    device: str,
                                    rollout_bind: str,
                                    rollout_endpoint: str,
                                    rank: int,
                                    world_size: int,
                                    config_overrides: Mapping[str, Any]
                                    | None = None,
                                    startup_timeout_s: float = 900.0,
                                    episode_timeout_s: float = 3600.0,
                                    simulator_restart_limit: int = 3,
                                    deployment_id: str | None = None) -> None:
    """Run one simulator supervisor and the rank-zero coordinator."""
    if rank < 0 or rank >= world_size:
        raise ValueError('rank must be in [0, world_size)')
    deployment_id = (
        deployment_id or os.environ.get('FLUXTHEMIS_DEPLOYMENT_ID')
        or os.environ.get('TORCHELASTIC_RUN_ID'))
    if not deployment_id or deployment_id == 'none':
        stable = (f'{os.environ.get("MASTER_ADDR", "127.0.0.1")}:'
                  f'{os.environ.get("MASTER_PORT", "29500")}:'
                  f'{Path(config_path).resolve()}')
        deployment_id = hashlib.sha256(stable.encode()).hexdigest()[:32]

    coordinator_process = None
    coordinator_monitor = None
    if rank == 0:
        context = multiprocessing.get_context('spawn')
        coordinator_process = context.Process(
            target=_run_coordinator_process,
            kwargs={
                'config_path': config_path,
                'config_overrides': dict(config_overrides or {}),
                'rollout_bind': rollout_bind,
                'expected_workers': world_size,
                'startup_timeout_s': startup_timeout_s,
                'episode_timeout_s': episode_timeout_s,
                'simulator_restart_limit': simulator_restart_limit,
                'deployment_id': deployment_id,
                'parent_pid': os.getpid(),
            },
            name='fluxthemis-coordinator',
        )
        coordinator_process.start()
        coordinator_monitor = threading.Thread(
            target=_monitor_coordinator_process,
            args=(coordinator_process, os.getpid()),
            daemon=True,
        )
        coordinator_monitor.start()

    try:
        SimulatorWorkerSupervisor(
            config_path=config_path,
            rollout_endpoint=rollout_endpoint,
            worker_id=f'simulator-worker-{rank}',
            deployment_id=deployment_id,
            device=device,
            episode_timeout_s=episode_timeout_s,
            config_overrides=config_overrides,
        ).run()
    except BaseException:
        if coordinator_process is not None:
            if coordinator_process.is_alive():
                coordinator_process.terminate()
            coordinator_monitor.join()
        raise
    else:
        if coordinator_monitor is not None:
            coordinator_monitor.join()


def _monitor_coordinator_process(process: multiprocessing.Process,
                                 parent_pid: int) -> None:
    process.join()
    if process.exitcode not in {0, -signal.SIGTERM}:
        print(
            f'[FluxThemis] coordinator exited with code {process.exitcode}',
            file=sys.stderr,
            flush=True,
        )
        os.kill(parent_pid, signal.SIGTERM)


def _run_coordinator_process(*, config_path: str,
                             config_overrides: Mapping[str,
                                                       Any], rollout_bind: str,
                             expected_workers: int, startup_timeout_s: float,
                             episode_timeout_s: float,
                             simulator_restart_limit: int, deployment_id: str,
                             parent_pid: int) -> None:
    configure_worker_lifetime(parent_pid)
    evaluation_run = build_distributed_evaluation_run(
        config_path=config_path,
        config_overrides=config_overrides,
    )
    coordinator = DistributedEvaluationCoordinator(
        rollout_bind=rollout_bind,
        expected_workers=expected_workers,
        startup_timeout_s=startup_timeout_s,
        episode_timeout_s=episode_timeout_s,
        simulator_restart_limit=simulator_restart_limit,
        deployment_id=deployment_id,
    )
    with evaluation_run:
        summary = coordinator.run(evaluation_run)
    print(json.dumps(summary['metrics'], indent=2), flush=True)
    print(
        f'Evaluation reported to {evaluation_run.report_run_dir}',
        flush=True,
    )


__all__ = ['run_distributed_evaluation_rank']
