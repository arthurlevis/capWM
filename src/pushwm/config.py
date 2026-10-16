"""Configuration loading. One YAML file describes the physical setup so the real
box, the simulated box and the dataset all share the same numbers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None, **overrides: Any) -> SimpleNamespace:
    """Load the YAML config. ``overrides`` are dotted keys, e.g. ``box__size_x=0.3``."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f)
    for key, value in overrides.items():
        section, name = key.split("__", 1)
        raw[section][name] = value
    cfg = _to_ns(raw)
    cfg.path = str(path)
    return cfg
