"""Configuration loading for distributed evaluation."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from mmengine import Config
from mmengine.utils import import_modules_from_strings

from fluxthemis.engines.utils import (build_evaluation_reporter_from_cfg,
                                      build_evaluator_from_cfg)

from ...transports.fluxvla_zmq import connection_transport
from .evaluation import EvaluationRun, EvaluationSettings

_EVALUATION_SETTING_NAMES = {
    field.name
    for field in fields(EvaluationSettings)
}


def load_distributed_evaluation_config(
        config_path: str,
        config_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load evaluation config and inject its canonical ZMQ settings."""
    path = Path(config_path).expanduser().resolve(strict=True)
    cfg = Config.fromfile(str(path), import_custom_modules=False)
    if config_overrides:
        cfg.merge_from_dict(dict(config_overrides))
    if 'themis' not in cfg or not isinstance(cfg.themis, Mapping):
        raise KeyError('FluxVLA config must define a `themis` mapping')
    custom_imports = cfg.themis.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    if ('evaluation' not in cfg.themis
            or not isinstance(cfg.themis.evaluation, Mapping)):
        raise KeyError('FluxVLA config must define `themis.evaluation`')
    evaluation_cfg = copy.deepcopy(dict(cfg.themis.evaluation))
    transport = cfg.themis.get('transport')
    if not isinstance(transport, Mapping):
        raise KeyError('FluxVLA config must define `themis.transport`')
    policy_client_cfg = evaluation_cfg.get('policy_client')
    if not isinstance(policy_client_cfg, Mapping):
        raise TypeError('themis.evaluation.policy_client must be a mapping')
    policy_client_cfg = copy.deepcopy(dict(policy_client_cfg))
    policy_client_cfg['transport'] = copy.deepcopy(dict(transport))
    evaluation_cfg['policy_client'] = policy_client_cfg

    reporter_cfg = evaluation_cfg.get('reporter')
    if not isinstance(reporter_cfg, Mapping):
        raise TypeError('themis.evaluation.reporter must be a mapping')
    reporter_cfg = copy.deepcopy(dict(reporter_cfg))
    reporter_cfg['transport'] = connection_transport(transport)
    evaluation_cfg['reporter'] = reporter_cfg
    return evaluation_cfg


def distributed_video_settings(
        evaluation_cfg: Mapping[str, Any]) -> tuple[bool, str | None, int]:
    save_videos = evaluation_cfg.get('save_videos', False)
    if not isinstance(save_videos, bool):
        raise TypeError('themis.evaluation.save_videos must be a boolean')
    video_fps = _positive_int(
        evaluation_cfg.get('video_fps', 20), 'themis.evaluation.video_fps')
    if not save_videos:
        return False, None, video_fps
    value = evaluation_cfg.get('video_dir')
    if value is None:
        raise ValueError(
            'themis.evaluation.video_dir is required when saving videos')
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError('themis.evaluation.video_dir must be a path or None')
    if not str(value).strip():
        raise ValueError('themis.evaluation.video_dir cannot be empty')
    return True, str(Path(value).expanduser().resolve()), video_fps


def build_distributed_evaluation_run(
        config_path: str,
        config_overrides: Mapping[str, Any] | None = None) -> EvaluationRun:
    evaluation_cfg = load_distributed_evaluation_config(
        config_path, config_overrides=config_overrides)
    settings_cfg = {
        name: copy.deepcopy(evaluation_cfg[name])
        for name in _EVALUATION_SETTING_NAMES if name in evaluation_cfg
    }
    reporter = build_evaluation_reporter_from_cfg(
        copy.deepcopy(evaluation_cfg['reporter']))
    evaluator = build_evaluator_from_cfg(
        copy.deepcopy(evaluation_cfg['evaluator']))
    return EvaluationRun(
        settings=EvaluationSettings(**settings_cfg),
        evaluator=evaluator,
        reporter=reporter,
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


__all__ = [
    'build_distributed_evaluation_run',
    'distributed_video_settings',
    'load_distributed_evaluation_config',
]
