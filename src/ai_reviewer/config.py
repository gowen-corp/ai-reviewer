"""Configuration loading and management for AI Reviewer.

This module handles hierarchical configuration loading with the following
priority order (highest to lowest):
1. CLI arguments
2. ./ai-reviewer.toml (local config)
3. ~/.config/ai-reviewer/config.toml (user config)
4. ~/.config/ai-reviewer/credentials.toml (credentials)
5. Built-in defaults

All configuration is validated using Pydantic models.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

from .models import (
    AuthConfig,
    Config,
    ModelConfig,
    PathsConfig,
    ReviewConfig,
    get_api_key_from_env,
    get_model_from_env,
    get_xdg_paths,
)


# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 3


def load_toml_file(path: Path) -> dict[str, Any]:
    """Load and parse a TOML file.
    
    Args:
        path: Path to the TOML file.
        
    Returns:
        Parsed TOML content as a dictionary.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        tomllib.TOMLDecodeError: If the file contains invalid TOML.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Later configurations override earlier ones (left-to-right priority).
    Performs deep merge for nested dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge.
        
    Returns:
        Merged configuration dictionary.
    """
    result: dict[str, Any] = {}
    
    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result


def load_credentials(path: Path) -> dict[str, Any]:
    """Load credentials from a TOML file.
    
    Args:
        path: Path to the credentials file.
        
    Returns:
        Dictionary with 'auth' section containing credentials.
    """
    if not path.exists():
        return {}
    
    try:
        data = load_toml_file(path)
        # Wrap in 'auth' section if credentials are at top level
        if "auth" in data:
            return {"auth": data["auth"]}
        return {"auth": data}
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def load_config_file(path: Path) -> dict[str, Any]:
    """Load configuration from a TOML file.
    
    Args:
        path: Path to the configuration file.
        
    Returns:
        Configuration dictionary, or empty dict if file doesn't exist.
    """
    if not path.exists():
        return {}
    
    try:
        return load_toml_file(path)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def build_config(
    repo_path: Path | None = None,
    model_name: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> Config:
    """Build complete configuration from all sources.
    
    Configuration priority (highest to lowest):
    1. CLI arguments (model_name, verbose, dry_run)
    2. Local config (./ai-reviewer.toml)
    3. User config (~/.config/ai-reviewer/config.toml)
    4. Credentials (~/.config/ai-reviewer/credentials.toml)
    5. Environment variables
    6. Built-in defaults
    
    Args:
        repo_path: Path to repository (for local config lookup).
        model_name: Model name from CLI argument.
        verbose: Verbose flag from CLI.
        dry_run: Dry-run flag from CLI.
        
    Returns:
        Validated Config object.
        
    Raises:
        SystemExit: If API key is missing (exit code 3).
    """
    # Get XDG paths
    paths = get_xdg_paths()
    
    # Load configurations from all sources
    credentials = load_credentials(paths.credentials_file)
    user_config = load_config_file(paths.config_file)
    
    # Local config in current directory or repo_path
    local_config_path = Path("ai-reviewer.toml")
    if repo_path and not local_config_path.exists():
        local_config_path = repo_path / "ai-reviewer.toml"
    local_config = load_config_file(local_config_path)
    
    # Build base configuration from files (lowest to highest priority)
    file_config = merge_configs(credentials, user_config, local_config)
    
    # Apply environment variables
    env_config: dict[str, Any] = {}
    env_api_key = get_api_key_from_env()
    env_model = get_model_from_env()
    
    if env_api_key:
        env_config["auth"] = {"api_key": env_api_key}
    if env_model:
        env_config["model"] = {"name": env_model}
    
    # Apply CLI arguments (highest priority)
    cli_config: dict[str, Any] = {}
    if model_name:
        cli_config["model"] = {"name": model_name}
    if verbose:
        cli_config["verbose"] = True
    if dry_run:
        cli_config["dry_run"] = True
    
    # Merge all configurations
    merged = merge_configs(file_config, env_config, cli_config)
    
    # Ensure paths are included
    merged["paths"] = {
        "config_dir": str(paths.config_dir),
        "data_dir": str(paths.data_dir),
        "cache_dir": str(paths.cache_dir),
    }
    
    # Validate and create Config object
    try:
        config = Config.model_validate(merged)
    except Exception as e:
        print(f"Configuration validation error: {e}", file=sys.stderr)
        sys.exit(EXIT_CONFIG_ERROR)
    
    # Verify API key is present
    if not config.auth.api_key:
        print(
            "Error: API key is required but not found.\n"
            "Set one of the following:\n"
            "  - AI_REVIEWER_API_KEY environment variable\n"
            "  - OPENROUTER_API_KEY environment variable\n"
            "  - api_key in ~/.config/ai-reviewer/credentials.toml\n"
            "  - api_key in ./ai-reviewer.toml",
            file=sys.stderr,
        )
        sys.exit(EXIT_CONFIG_ERROR)
    
    return config


def ensure_directories(config: Config) -> None:
    """Ensure all required directories exist.
    
    Creates directories if they don't exist.
    
    Args:
        config: Configuration object with paths.
    """
    for dir_path in [config.paths.config_dir, config.paths.data_dir, config.paths.cache_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)


def print_config_summary(config: Config) -> None:
    """Print configuration summary (for verbose mode).
    
    Note: API key is masked for security.
    
    Args:
        config: Configuration object to summarize.
    """
    import sys
    
    masked_key = "***" + config.auth.api_key[-4:] if len(config.auth.api_key) > 4 else "***"
    
    print(f"Provider: {config.auth.provider}", file=sys.stderr)
    print(f"API Key: {masked_key}", file=sys.stderr)
    print(f"Base URL: {config.auth.base_url}", file=sys.stderr)
    print(f"Model: {config.model.name}", file=sys.stderr)
    print(f"Temperature: {config.model.temperature}", file=sys.stderr)
    print(f"Config Dir: {config.paths.config_dir}", file=sys.stderr)
    print(f"Verbose: {config.verbose}", file=sys.stderr)
    print(f"Dry Run: {config.dry_run}", file=sys.stderr)
