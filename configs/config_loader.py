"""
Hierarchical YAML config loader.
Loads default.yaml first, then merges model-specific overrides.
"""

import os
import yaml
import copy
from pathlib import Path

CONFIG_DIR = Path(__file__).parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(model_config_name: str) -> dict:
    """
    Load configuration by merging default.yaml with a model-specific YAML.

    Args:
        model_config_name: e.g. 'resnet18' or 'resnet18.yaml'
    Returns:
        Merged config dict.
    """
    # Load defaults
    default_path = CONFIG_DIR / "default.yaml"
    with open(default_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model-specific overrides
    if not model_config_name.endswith(".yaml"):
        model_config_name += ".yaml"
    model_path = CONFIG_DIR / model_config_name
    if model_path.exists():
        with open(model_path, "r") as f:
            overrides = yaml.safe_load(f)
        if overrides:
            config = _deep_merge(config, overrides)

    return config


class Config:
    """Dot-access wrapper around a config dict."""

    def __init__(self, d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config({self.to_dict()})"
