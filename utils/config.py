"""
Configuration loader for the Financial Risk Intelligence Copilot.

Loads YAML configuration with environment variable overrides.
Implements singleton pattern to avoid repeated file I/O.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Project root (two levels up from utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class Config:
    """
    Singleton configuration manager.
    
    Loads config/config.yaml and merges with environment variables.
    Environment variables take precedence over YAML values.
    
    Usage:
        config = Config()
        model_type = config.get("ml_engine.model.type")
    """

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}"
            )
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot-notation.

        Args:
            key_path: Dot-separated path, e.g. 'ml_engine.model.type'
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        # Check environment variable override (dots replaced with underscores, uppercased)
        env_key = key_path.replace(".", "_").upper()
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return self._cast_env_value(env_val)

        # Traverse YAML config
        keys = key_path.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Top-level section name, e.g. 'ml_engine'.

        Returns:
            Dictionary of section values.
        """
        return self._config.get(section, {})

    @staticmethod
    def _cast_env_value(value: str) -> Any:
        """Attempt to cast environment variable string to appropriate type."""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @property
    def project_root(self) -> Path:
        """Return the project root path."""
        return PROJECT_ROOT

    def resolve_path(self, relative_path: str) -> Path:
        """
        Resolve a relative path against the project root.

        Args:
            relative_path: Path relative to project root.

        Returns:
            Absolute Path object.
        """
        return PROJECT_ROOT / relative_path
