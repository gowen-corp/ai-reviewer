"""Pydantic models for AI Reviewer configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AuthConfig(BaseModel):
    """Authentication configuration for LLM providers.
    
    Attributes:
        provider: LLM provider name (e.g., 'openrouter', 'ollama', 'vllm').
        api_key: API key for authentication. Never logged or exposed.
        base_url: Base URL for the API endpoint.
    """
    
    provider: Literal["openrouter", "ollama", "vllm", "openai"] = Field(
        default="openrouter",
        description="LLM provider name",
    )
    api_key: str = Field(
        ...,
        description="API key for authentication",
        min_length=1,
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the API endpoint",
    )
    
    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate and normalize base URL."""
        return v.rstrip("/")


class ModelConfig(BaseModel):
    """Model-specific configuration.
    
    Attributes:
        name: Model identifier (e.g., 'qwen/qwen3.5-flash').
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.
    """
    
    name: str = Field(
        default="qwen/qwen3.5-flash",
        description="Model identifier",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens in response",
    )
    timeout: int = Field(
        default=60,
        ge=1,
        description="Request timeout in seconds",
    )


class ReviewConfig(BaseModel):
    """Code review configuration.
    
    Attributes:
        ignore_patterns: Glob patterns for files to ignore.
        max_file_size: Maximum file size in bytes to process.
        languages: List of programming languages to review.
    """
    
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "*.pyc", "__pycache__/*", ".git/*", "*.min.js", 
            "node_modules/*", "venv/*", ".venv/*",
        ],
        description="Glob patterns for files to ignore",
    )
    max_file_size: int = Field(
        default=1_000_000,  # 1MB
        ge=1,
        description="Maximum file size in bytes to process",
    )
    languages: list[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "go", "rust"],
        description="Programming languages to review",
    )


class PathsConfig(BaseModel):
    """XDG-compliant path configuration.
    
    Attributes:
        config_dir: Directory for configuration files.
        data_dir: Directory for data files.
        cache_dir: Directory for cache files.
    """
    
    config_dir: Path = Field(..., description="Directory for configuration files")
    data_dir: Path = Field(..., description="Directory for data files")
    cache_dir: Path = Field(..., description="Directory for cache files")
    
    @property
    def config_file(self) -> Path:
        """Path to main configuration file."""
        return self.config_dir / "config.toml"
    
    @property
    def credentials_file(self) -> Path:
        """Path to credentials file."""
        return self.config_dir / "credentials.toml"


class HooksConfig(BaseModel):
    """Hook configuration for pre/post processing.
    
    Attributes:
        pre_process: Command to run before review starts.
        post_process: Command to run after review completes.
    """
    
    pre_process: str | None = Field(
        default=None,
        description="Command to run before review starts",
    )
    post_process: str | None = Field(
        default=None,
        description="Command to run after review completes",
    )


class Config(BaseModel):
    """Main configuration model combining all sections.
    
    This is the root configuration object that combines authentication,
    model settings, review options, and paths.
    
    Attributes:
        auth: Authentication configuration.
        model: Model-specific configuration.
        review: Code review configuration.
        paths: XDG-compliant paths.
        hooks: Hook configuration for pre/post processing.
        verbose: Enable verbose output.
        dry_run: Perform dry run without changes.
    """
    
    auth: AuthConfig = Field(default_factory=AuthConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    paths: PathsConfig
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    verbose: bool = Field(default=False, description="Enable verbose output")
    dry_run: bool = Field(default=False, description="Perform dry run without changes")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "ignore"  # Ignore unknown fields in TOML


def get_xdg_paths(app_name: str = "ai-reviewer") -> PathsConfig:
    """Get XDG-compliant paths for the application.
    
    Args:
        app_name: Application name for directory structure.
        
    Returns:
        PathsConfig with resolved XDG directories.
    """
    home = Path.home()
    
    # XDG Base Directory Specification
    xdg_config_home = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    xdg_data_home = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    xdg_cache_home = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    
    return PathsConfig(
        config_dir=xdg_config_home / app_name,
        data_dir=xdg_data_home / app_name,
        cache_dir=xdg_cache_home / app_name,
    )


def get_api_key_from_env() -> str | None:
    """Retrieve API key from environment variables.
    
    Checks multiple environment variable names in priority order:
    1. AI_REVIEWER_API_KEY
    2. OPENROUTER_API_KEY
    
    Returns:
        API key if found, None otherwise.
    """
    for env_var in ["AI_REVIEWER_API_KEY", "OPENROUTER_API_KEY"]:
        if key := os.environ.get(env_var):
            return key
    return None


def get_model_from_env() -> str | None:
    """Retrieve model name from environment variables.
    
    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get("AI_REVIEWER_MODEL")
