from __future__ import annotations

from typing import Any, Mapping

from mmengine.config import Config, ConfigDict
from mmengine.registry import Registry

from .root import (ENVIRONMENTS, EVALUATION_REPORTERS, EVALUATORS,
                   POLICY_CLIENTS)

ConfigLike = Mapping[str, Any] | ConfigDict | Config


def _build(cfg: ConfigLike, registry: Registry, default_args=None) -> Any:
    if cfg is None:
        raise TypeError(f'cfg for registry {registry.name!r} cannot be None')
    return registry.build(cfg, default_args=default_args)


def build_environment_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, ENVIRONMENTS, default_args)


def build_evaluator_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, EVALUATORS, default_args)


def build_policy_client_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, POLICY_CLIENTS, default_args)


def build_evaluation_reporter_from_cfg(cfg: ConfigLike,
                                       default_args=None) -> Any:
    return _build(cfg, EVALUATION_REPORTERS, default_args)
