"""Hook system for AI Reviewer.

This module provides pre/post-processing hooks that can be configured
in ai-reviewer.toml. Hooks are external scripts/commands that run at
specific points in the review pipeline.

Hook configuration in ai-reviewer.toml:
    [hooks]
    pre_process = "/path/to/pre_hook.sh"
    post_process = "/path/to/post_hook.py"

Hooks receive environment variables:
- AI_REPORT_PATH: Path to the JSON report file (post_process only)
- AI_REVIEW_MODE: Current review mode
- AI_MODEL: Model identifier
- AI_TARGET_HASH: Hash of reviewed content
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .models import Config


logger = logging.getLogger(__name__)


class HookError(Exception):
    """Exception raised when a hook fails."""

    pass


class HookResult:
    """Result of a hook execution.

    Attributes:
        success: Whether the hook completed successfully.
        return_code: Process return code.
        stdout: Standard output from the hook.
        stderr: Standard error from the hook.
        message: Human-readable result message.
    """

    def __init__(
        self,
        success: bool,
        return_code: int = 0,
        stdout: str = "",
        stderr: str = "",
        message: str = "",
    ) -> None:
        """Initialize hook result.

        Args:
            success: Whether hook succeeded.
            return_code: Process return code.
            stdout: Standard output.
            stderr: Standard error.
            message: Result message.
        """
        self.success = success
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.message = message or f"Hook exited with code {return_code}"

    def __repr__(self) -> str:
        """String representation."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"HookResult({status}, code={self.return_code})"


def run_hook(
    command: str,
    env_vars: dict[str, str] | None = None,
    timeout: int = 300,
    verbose: bool = False,
) -> HookResult:
    """Execute a hook command.

    Args:
        command: Shell command to execute.
        env_vars: Additional environment variables.
        timeout: Maximum execution time in seconds (default 300).
        verbose: Enable verbose logging.

    Returns:
        HookResult with execution details.

    Raises:
        HookError: If command is not found or execution fails critically.
    """
    # For shell commands, we use shell=True so we don't need to check executable
    # Only validate if it's a simple command (no shell operators)
    cmd_parts = command.split()
    if cmd_parts and not any(c in command for c in ['|', '&&', '||', ';', '>', '<', '$']):
        # Simple command - check if executable exists
        executable = cmd_parts[0]
        # Skip check for shell builtins
        if executable not in ('exit', 'cd', 'export', 'source', '.', ':', 'true', 'false'):
            if not shutil.which(executable):
                raise HookError(f"Hook command not found: {executable}")

    # Prepare environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    logger.info("Running hook: %s", command)
    if verbose:
        logger.debug("Hook environment: %s", {k: v for k, v in env.items() if not k.endswith("_KEY")})

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )

        success = result.returncode == 0

        hook_result = HookResult(
            success=success,
            return_code=result.returncode,
            stdout=result.stdout[:10000] if result.stdout else "",  # Limit output
            stderr=result.stderr[:10000] if result.stderr else "",
            message=f"Hook completed with exit code {result.returncode}",
        )

        if success:
            logger.info("Hook succeeded: %s", command)
        else:
            logger.warning("Hook failed (%d): %s", result.returncode, command)
            if verbose and result.stderr:
                logger.debug("Hook stderr: %s", result.stderr[:500])

        return hook_result

    except subprocess.TimeoutExpired as e:
        logger.error("Hook timed out after %d seconds: %s", timeout, command)
        raise HookError(f"Hook timeout after {timeout}s: {command}") from e
    except OSError as e:
        logger.error("OS error running hook: %s", e)
        raise HookError(f"Failed to execute hook: {e}") from e


def run_pre_process_hook(
    config: Config,
    target_hash: str,
    verbose: bool = False,
) -> HookResult | None:
    """Run pre-process hook if configured.

    This hook runs before the review pipeline starts. It can be used
    for setup tasks, validation, or preprocessing.

    Args:
        config: Configuration object.
        target_hash: Hash of the review target.
        verbose: Enable verbose logging.

    Returns:
        HookResult if hook was run, None if no hook configured.

    Raises:
        HookError: If hook execution fails.
    """
    # Check if pre_process hook is configured
    # Note: This requires hooks config in Config model (Phase 7 addition)
    hooks_config = getattr(config, "hooks", None)
    if hooks_config is None:
        return None

    pre_command = getattr(hooks_config, "pre_process", None)
    if not pre_command:
        return None

    # Set up environment variables
    env_vars = {
        "AI_REVIEW_MODE": config.model.name,
        "AI_MODEL": config.model.name,
        "AI_TARGET_HASH": target_hash,
    }

    return run_hook(pre_command, env_vars=env_vars, verbose=verbose)


def run_post_process_hook(
    config: Config,
    report_path: Path,
    target_hash: str,
    verbose: bool = False,
) -> HookResult | None:
    """Run post-process hook if configured.

    This hook runs after the review completes. It receives the path
    to the JSON report via AI_REPORT_PATH environment variable.

    Args:
        config: Configuration object.
        report_path: Path to the JSON report file.
        target_hash: Hash of the review target.
        verbose: Enable verbose logging.

    Returns:
        HookResult if hook was run, None if no hook configured.

    Raises:
        HookError: If hook execution fails.
    """
    # Check if post_process hook is configured
    hooks_config = getattr(config, "hooks", None)
    if hooks_config is None:
        return None

    post_command = getattr(hooks_config, "post_process", None)
    if not post_command:
        return None

    # Verify report file exists
    if not report_path.exists():
        logger.warning("Report file does not exist: %s", report_path)
        return None

    # Set up environment variables
    env_vars = {
        "AI_REPORT_PATH": str(report_path),
        "AI_REVIEW_MODE": config.model.name,
        "AI_MODEL": config.model.name,
        "AI_TARGET_HASH": target_hash,
    }

    return run_hook(post_command, env_vars=env_vars, verbose=verbose)


def create_temp_report(data: dict[str, Any]) -> Path:
    """Create a temporary JSON report file.

    Args:
        data: Report data to write.

    Returns:
        Path to the temporary file.
    """
    import json

    fd, path = tempfile.mkstemp(suffix=".json", prefix="ai_reviewer_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        os.close(fd)
        raise

    return Path(path)


def cleanup_temp_file(path: Path) -> None:
    """Clean up a temporary file.

    Args:
        path: Path to the file to remove.
    """
    try:
        if path.exists():
            path.unlink()
            logger.debug("Cleaned up temp file: %s", path)
    except OSError as e:
        logger.warning("Failed to clean up temp file %s: %s", path, e)
