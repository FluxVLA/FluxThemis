#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from fluxthemis.environments.sim.libero import (LIBERO_DUMMY_ACTION,
                                                LIBERO_MAX_STEPS,
                                                LiberoEnvironment,
                                                _load_libero_runtime)

IMAGE_KEYS = ('agentview_image', 'robot0_eye_in_hand_image')
STATE_KEYS = (
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Verify the pinned FluxThemis LIBERO runtime.')
    parser.add_argument(
        '--suite', choices=sorted(LIBERO_MAX_STEPS), default='libero_10')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--episode-index', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--num-steps-wait', type=int, default=10)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--imports-only', action='store_true')
    return parser.parse_args()


def describe_array(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {'shape': list(array.shape), 'dtype': str(array.dtype)}


def verify_observation(observation: dict[str, Any], resolution: int) -> None:
    for key in IMAGE_KEYS:
        if key not in observation:
            raise KeyError(f'LIBERO observation is missing image {key!r}')
        image = np.asarray(observation[key])
        if image.shape != (resolution, resolution, 3):
            raise ValueError(
                f'{key} must have shape {(resolution, resolution, 3)}, '
                f'got {image.shape}')
        if image.dtype != np.uint8:
            raise TypeError(f'{key} must be uint8, got {image.dtype}')
    for key in STATE_KEYS:
        if key not in observation:
            raise KeyError(f'LIBERO observation is missing state {key!r}')
        state = np.asarray(observation[key])
        if state.ndim != 1 or state.size == 0:
            raise ValueError(f'{key} must be a non-empty vector')
        if not np.isfinite(state).all():
            raise ValueError(f'{key} contains NaN or infinity')


def runtime_summary() -> dict[str, Any]:
    runtime = _load_libero_runtime()
    import robosuite
    import torch

    libero_root = Path(inspect.getfile(runtime.libero)).resolve().parent
    paths = {
        name: Path(runtime.libero.get_libero_path(name)).resolve()
        for name in ('assets', 'bddl_files', 'init_states')
    }
    for name, path in paths.items():
        if not path.is_dir():
            raise FileNotFoundError(
                f'LIBERO {name} directory is missing: {path}')
        try:
            path.relative_to(libero_root)
        except ValueError as exc:
            raise RuntimeError(
                f'LIBERO {name} resolves outside the installed checkout: '
                f'{path} (checkout: {libero_root})') from exc
    return {
        'libero_root': str(libero_root),
        'libero_paths': {
            key: str(value)
            for key, value in paths.items()
        },
        'robosuite_version': robosuite.__version__,
        'torch_version': torch.__version__,
    }


def main() -> int:
    args = parse_args()
    summary = runtime_summary()
    if args.imports_only:
        summary['smoke'] = 'imports'
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.task_id < 0:
        raise ValueError('--task-id must be non-negative')
    if args.steps < 0:
        raise ValueError('--steps must be non-negative')

    environment = LiberoEnvironment(
        task_suite_name=args.suite,
        resolution=args.resolution,
        num_steps_wait=args.num_steps_wait,
    )
    try:
        environment.setup()
        tasks = environment.tasks()
        if args.task_id >= len(tasks):
            raise ValueError(
                f'--task-id {args.task_id} is outside [0, {len(tasks)})')
        task = tasks[args.task_id]
        observation, reset_info = environment.reset(
            task=task,
            episode_index=args.episode_index,
            environment_seed=7 + args.episode_index,
            environment_case_index=args.episode_index,
        )
        verify_observation(observation, args.resolution)

        transition = None
        for _ in range(args.steps):
            transition = environment.step(LIBERO_DUMMY_ACTION)
            verify_observation(transition.observation, args.resolution)
            if transition.terminated or transition.truncated:
                break

        summary.update({
            'smoke':
            'libero',
            'suite':
            args.suite,
            'task_id':
            task.task_id,
            'task':
            task.description,
            'task_count':
            len(tasks),
            'recommended_max_episode_steps':
            environment.recommended_max_episode_steps,
            'reset_info':
            reset_info,
            'observation': {
                key: describe_array(observation[key])
                for key in (*IMAGE_KEYS, *STATE_KEYS)
            },
            'step_success':
            None if transition is None else transition.success,
        })
    finally:
        environment.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
