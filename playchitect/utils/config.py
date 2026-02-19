"""
User configuration management.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Config:
    """User configuration manager."""

    DEFAULT_CONFIG_PATH = Path.home() / ".config" / "playchitect" / "config.yaml"

    DEFAULT_CONFIG = {
        "test_music_path": "/mnt/1tb_ssd/Media/Music/Trying Before You Buying/dark 4",
        "default_output_dir": None,
        "default_target_tracks": 25,
        "default_target_duration": None,
        "cache_dir": "~/.cache/playchitect",
        "log_level": "INFO",
    }

    def __init__(self, config_path: Path | None = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config file (default: ~/.config/playchitect/config.yaml)
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from file, or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self.config = self.DEFAULT_CONFIG.copy()
        else:
            logger.info("No config file found, using defaults")
            self.config = self.DEFAULT_CONFIG.copy()

    def save(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value

    def get_test_music_path(self) -> Path | None:
        """
        Get test music path from config.

        Returns:
            Path to test music directory, or None
        """
        path_str = self.get("test_music_path")
        if path_str:
            return Path(path_str).expanduser()
        return None

    def get_cache_dir(self) -> Path:
        """
        Get cache directory.

        Returns:
            Path to cache directory
        """
        cache_str = self.get("cache_dir", "~/.cache/playchitect")
        cache_path = Path(cache_str).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """
    Get global config instance.

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config
