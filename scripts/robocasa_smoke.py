#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from fluxthemis.environments.sim.robocasa import (ROBOCASA_ACTION_DIM,
                                                  ROBOCASA_PROMPT_KEY,
                                                  RoboCasaEnvironment,
                                                  _load_robocasa_runtime)

DEFAULT_ENV_ID = ('gr1_unified/'
                  'PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env')
IMAGE_KEY = 'video.ego_view_bg_crop_pad_res256_freq20'
STATE_KEYS = {
    'state.left_arm': 7,
    'state.left_hand': 6,
    'state.right_arm': 7,
    'state.right_hand': 6,
    'state.waist': 3,
}
REQUIRED_ASSET_PATHS = (
    'fixtures',
    'textures',
    'generative_textures',
    'objects/objaverse',
    'objects/sketchfab',
    'objects/lightwheel',
    'fixtures/toasters/basic_popup_2/visuals/model_0.obj',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Verify the pinned FluxThemis RoboCasa runtime.')
    parser.add_argument('--env-id', default=DEFAULT_ENV_ID)
    parser.add_argument('--episode-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument(
        '--action-order', choices=('fluxvla', 'n15'), default='fluxvla')
    parser.add_argument('--imports-only', action='store_true')
    return parser.parse_args()


def describe_array(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {'shape': list(array.shape), 'dtype': str(array.dtype)}


def verify_observation(observation: dict[str, Any]) -> None:
    if IMAGE_KEY not in observation:
        raise KeyError(f'RoboCasa observation is missing image {IMAGE_KEY!r}')
    image = np.asarray(observation[IMAGE_KEY])
    if image.shape != (256, 256, 3):
        raise ValueError(
            f'{IMAGE_KEY} must have shape (256, 256, 3), got {image.shape}')
    if image.dtype != np.uint8:
        raise TypeError(f'{IMAGE_KEY} must be uint8, got {image.dtype}')
    for key, size in STATE_KEYS.items():
        if key not in observation:
            raise KeyError(f'RoboCasa observation is missing state {key!r}')
        state = np.asarray(observation[key])
        if state.shape != (size, ):
            raise ValueError(
                f'{key} must have shape {(size,)}, got {state.shape}')
        if not np.isfinite(state).all():
            raise ValueError(f'{key} contains NaN or infinity')
    prompt = observation.get(ROBOCASA_PROMPT_KEY)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f'{ROBOCASA_PROMPT_KEY} must be a non-empty string')


def runtime_summary() -> dict[str, Any]:
    runtime = _load_robocasa_runtime()
    import robosuite
    import torch

    robocasa_root = Path(inspect.getfile(runtime.robocasa)).resolve().parent
    assets_root = robocasa_root / 'models' / 'assets'
    missing = [
        relative for relative in REQUIRED_ASSET_PATHS
        if not (assets_root / relative).exists()
    ]
    if missing:
        raise FileNotFoundError('RoboCasa assets are incomplete under '
                                f'{assets_root}: {", ".join(missing)}')
    registered = sorted(spec.id
                        for spec in runtime.gymnasium.registry.values()
                        if spec.id.startswith('gr1_unified/'))
    if DEFAULT_ENV_ID not in registered:
        raise RuntimeError(f'{DEFAULT_ENV_ID!r} is not registered')
    return {
        'robocasa_root': str(robocasa_root),
        'asset_root': str(assets_root.resolve()),
        'registered_gr1_unified_tasks': len(registered),
        'robosuite_version': robosuite.__version__,
        'torch_version': torch.__version__,
    }


def main() -> int:
    args = parse_args()
    if args.steps < 0:
        raise ValueError('--steps must be non-negative')
    if args.episode_index < 0:
        raise ValueError('--episode-index must be non-negative')

    summary = runtime_summary()
    if args.imports_only:
        summary['smoke'] = 'imports'
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    environment = RoboCasaEnvironment(
        task_list=[args.env_id], action_order=args.action_order)
    try:
        environment.setup()
        task = environment.tasks()[0]
        observation, reset_info = environment.reset(
            task=task,
            episode_index=args.episode_index,
            seed=args.seed,
        )
        verify_observation(observation)
        transition = None
        for _ in range(args.steps):
            transition = environment.step(
                np.zeros(ROBOCASA_ACTION_DIM, dtype=np.float32))
            verify_observation(transition.observation)
            if transition.terminated or transition.truncated:
                break

        summary.update({
            'smoke':
            'robocasa',
            'env_id':
            args.env_id,
            'action_order':
            args.action_order,
            'episode_seed':
            reset_info['episode_seed'],
            'prompt':
            reset_info['task_description'],
            'observation': {
                key: describe_array(observation[key])
                for key in (IMAGE_KEY, *STATE_KEYS)
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
