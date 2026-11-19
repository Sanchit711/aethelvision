"""Centralized Configuration Manager for Aethel Vision Framework.

Loads, merges, and validates YAML configuration specs for datasets, models,
training orchestrators, deployment targets, and experiments.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from aethel.utils.exceptions import ConfigurationException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.configs")


class ConfigDict(dict):
    """Dictionary subclass supporting attribute-style dot notation access."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = ConfigDict(value) if isinstance(value, dict) else value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")


class ConfigManager:
    """Centralized manager for parsing and validating YAML configuration files."""

    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> ConfigDict:
        """Load a YAML file into a ConfigDict with attribute-style access.

        Args:
            file_path: Path to target .yaml file.

        Returns:
            ConfigDict object.
        """
        path = Path(file_path)
        if not path.exists():
            raise ConfigurationException(
                message=f"Configuration file not found: {path}",
                details=f"Absolute path checked: {path.resolve()}",
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
            logger.debug(f"Successfully loaded configuration from {path}")
            return ConfigDict(content)
        except Exception as e:
            raise ConfigurationException(
                message=f"Failed to parse YAML configuration at {path}",
                details=str(e),
            )

    @staticmethod
    def merge_configs(base_config: ConfigDict, override_config: ConfigDict) -> ConfigDict:
        """Recursively merge override_config into base_config."""
        merged = ConfigDict(base_config)
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigManager.merge_configs(merged[key], ConfigDict(value))
            else:
                merged[key] = value
        return merged


__all__ = ["ConfigDict", "ConfigManager"]
