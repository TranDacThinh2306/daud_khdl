"""
config_loader.py - Configuration loader
==========================================
Load YAML configuration files.
"""

import os
import logging
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Dict[str, Dict] = {}


def load_config(
    config_path: str = "src/config/config.yaml",
    override: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file
        override: Optional dictionary to override config values

    Returns:
        Configuration dictionary
    """
    if config_path in _CONFIG_CACHE:
        config = _CONFIG_CACHE[config_path].copy()
    else:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        _CONFIG_CACHE[config_path] = config.copy()
        logger.info(f"Loaded config from {config_path}")

    if override:
        config = _deep_merge(config, override)

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_model_config() -> Dict:
    """Load model-specific configuration."""
    return load_config("src/config/model_config.yaml")


def load_xai_config() -> Dict:
    """Load XAI-specific configuration."""
    return load_config("src/config/xai_config.yaml")
