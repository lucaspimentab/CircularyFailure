"""Configuration utilities."""
from __future__ import annotations

from pathlib import Path
import os
import yaml


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_env(obj):
    if isinstance(obj, str):
        for k, v in os.environ.items():
            obj = obj.replace(f"${{{k}}}", v)
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    return obj


def load_config(config_dir: Path, dataset: str) -> dict:
    base = _load_yaml(config_dir / 'default.yaml')
    ds = _load_yaml(config_dir / f"{dataset}.yaml")
    return _resolve_env(_merge(base, ds))
