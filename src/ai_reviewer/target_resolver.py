"""Target resolution for AI Reviewer.

This module handles flexible parsing of review targets including:
- File paths, directories, glob patterns
- stdin (when piped)
- Git diff mode (--diff-only)
- PR mode (--pr <url_or_number>)

All I/O operations are async-compatible.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import urlparse


class TargetSourceType(Enum):
    """Source type for review targets."""

    FILES = auto()
    DIRECTORY = auto()
    GLOB = auto()
    STDIN = auto()
    GIT_DIFF = auto()
    PR = auto()


@dataclass
class ReviewTarget:
    """Represents a review target with resolved files.

    Attributes:
        files: List of file paths to review.
        source_type: Type of source (files, directory, glob, stdin, git, PR).
        content_hash: SHA256 hash of combined file contents.
        stdin_content: Content from stdin if source_type is STDIN.
        pr_url: Original PR URL if source_type is PR.
        pr_number: Extracted PR number if source_type is PR.
        repo_name: Repository name extracted from PR URL.
    """

    files: list[Path] = field(default_factory=list)
    source_type: TargetSourceType = TargetSourceType.FILES
    content_hash: str = ""
    stdin_content: str | None = None
    pr_url: str | None = None
    pr_number: int | None = None
    repo_name: str | None = None

    def __post_init__(self) -> None:
        """Compute content hash after initialization."""
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA256 hash of file contents.

        Returns:
            Hex-encoded SHA256 hash of combined file contents.
        """
        hasher = hashlib.sha256()

        if self.source_type == TargetSourceType.STDIN and self.stdin_content:
            hasher.update(self.stdin_content.encode("utf-8"))
        else:
            # Sort files for consistent hashing
            for file_path in sorted(self.files):
                if file_path.exists() and file_path.is_file():
                    try:
                        hasher.update(file_path.read_bytes())
                    except (IOError, OSError):
                        # Skip unreadable files but continue
                        pass

        return hasher.hexdigest()


@dataclass
class PRInfo:
    """Parsed PR information.

    Attributes:
        url: Original PR URL.
        number: PR number.
        repo_name: Repository name (owner/repo format).
        platform: Platform (github, gitlab, etc.).
    """

    url: str
    number: int
    repo_name: str
    platform: str


def parse_pr_url(url: str) -> PRInfo:
    """Parse and validate a PR URL.

    Supports GitHub-style URLs:
    - https://github.com/owner/repo/pull/123
    - https://gitlab.com/owner/repo/merge_requests/123

    Args:
        url: PR URL to parse.

    Returns:
        PRInfo with parsed details.

    Raises:
        ValueError: If URL is invalid or cannot be parsed.
    """
    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL scheme: {url}")

    path_parts = parsed.path.strip("/").split("/")

    if len(path_parts) < 4:
        raise ValueError(f"Invalid PR URL path: {url}")

    # Detect platform and extract PR number
    platform = ""
    pr_number: int | None = None
    repo_name = ""

    if "github.com" in parsed.netloc:
        platform = "github"
        # Format: /owner/repo/pull/123 (5 parts after split)
        if len(path_parts) >= 4 and path_parts[2] == "pull":
            repo_name = f"{path_parts[0]}/{path_parts[1]}"
            try:
                pr_number = int(path_parts[3])
            except ValueError:
                raise ValueError(f"Invalid PR number in URL: {url}")
        else:
            raise ValueError(f"Invalid GitHub PR URL format: {url}")

    elif "gitlab.com" in parsed.netloc:
        platform = "gitlab"
        # Format: /owner/repo/merge_requests/123 (5 parts after split)
        if len(path_parts) >= 4 and path_parts[2] == "merge_requests":
            repo_name = f"{path_parts[0]}/{path_parts[1]}"
            try:
                pr_number = int(path_parts[3])
            except ValueError:
                raise ValueError(f"Invalid MR number in URL: {url}")
        else:
            raise ValueError(f"Invalid GitLab MR URL format: {url}")
    else:
        # Generic fallback - try to extract number from last segment
        platform = "unknown"
        repo_name = f"{path_parts[0]}/{path_parts[1]}" if len(path_parts) >= 2 else "unknown/repo"
        try:
            pr_number = int(path_parts[-1])
        except ValueError:
            raise ValueError(f"Cannot extract PR number from URL: {url}")

    if pr_number is None:
        raise ValueError(f"Could not extract PR number from URL: {url}")

    return PRInfo(
        url=url,
        number=pr_number,
        repo_name=repo_name,
        platform=platform,
    )


def parse_pr_argument(arg: str) -> PRInfo:
    """Parse PR argument which can be a URL or just a number.

    Args:
        arg: PR URL or PR number string.

    Returns:
        PRInfo with parsed details.

    Raises:
        ValueError: If argument is invalid.
    """
    # Check if it's just a number
    if arg.isdigit():
        return PRInfo(
            url="",
            number=int(arg),
            repo_name="",
            platform="unknown",
        )

    # Otherwise, parse as URL
    return parse_pr_url(arg)


async def read_stdin() -> str:
    """Read content from stdin asynchronously.

    Returns:
        Content read from stdin.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sys.stdin.read)


def resolve_glob(pattern: str, base_path: Path | None = None) -> list[Path]:
    """Resolve a glob pattern to matching file paths.

    Args:
        pattern: Glob pattern (e.g., "*.py", "src/**/*.py").
        base_path: Base directory for relative patterns. Defaults to cwd.

    Returns:
        List of matching file paths.
    """
    if base_path is None:
        base_path = Path.cwd()

    # Handle absolute patterns
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        search_base = pattern_path.parent
        search_pattern = pattern_path.name
    else:
        search_base = base_path
        search_pattern = pattern

    # Use rglob for ** patterns, glob otherwise
    if "**" in pattern:
        matches = list(search_base.rglob(search_pattern))
    else:
        matches = list(search_base.glob(pattern))

    # Filter to only files (not directories)
    return [m for m in matches if m.is_file()]


def resolve_path(path_str: str) -> ReviewTarget:
    """Resolve a single path argument to a ReviewTarget.

    Handles:
    - Single file paths
    - Directory paths (all files recursively)
    - Glob patterns

    Args:
        path_str: Path string to resolve.

    Returns:
        ReviewTarget with resolved files.

    Raises:
        SystemExit: If path is inaccessible.
    """
    path = Path(path_str)

    # Check for glob patterns
    if any(c in path_str for c in ("*", "?", "[")):
        matches = resolve_glob(path_str)
        if not matches:
            print(f"Warning: No files match pattern '{path_str}'", file=sys.stderr)
        return ReviewTarget(
            files=matches,
            source_type=TargetSourceType.GLOB,
        )

    # Handle non-existent paths
    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    # Handle directories
    if path.is_dir():
        files = [f for f in path.rglob("*") if f.is_file()]
        return ReviewTarget(
            files=files,
            source_type=TargetSourceType.DIRECTORY,
        )

    # Handle single files
    if path.is_file():
        return ReviewTarget(
            files=[path],
            source_type=TargetSourceType.FILES,
        )

    # Unknown path type
    print(f"Error: Cannot resolve path: {path}", file=sys.stderr)
    sys.exit(1)


async def resolve_targets(
    paths: list[str] | None = None,
    diff_only: bool = False,
    pr_arg: str | None = None,
) -> ReviewTarget:
    """Resolve all input sources to a unified ReviewTarget.

    This is the main entry point for target resolution. It handles:
    - Multiple file/directory/glob arguments
    - stdin (if no paths provided and stdin is not a TTY)
    - Git diff mode
    - PR mode

    Args:
        paths: List of path strings (files, dirs, globs).
        diff_only: If True, use git diff to determine files.
        pr_arg: PR URL or number for PR mode.

    Returns:
        Unified ReviewTarget object.

    Raises:
        SystemExit: If resolution fails.
    """
    # PR mode takes precedence
    if pr_arg is not None:
        try:
            pr_info = parse_pr_argument(pr_arg)
        except ValueError as e:
            print(f"Error: Invalid PR argument: {e}", file=sys.stderr)
            sys.exit(1)

        # For now, just return metadata - actual file fetching is future phase
        return ReviewTarget(
            files=[],
            source_type=TargetSourceType.PR,
            pr_url=pr_info.url,
            pr_number=pr_info.number,
            repo_name=pr_info.repo_name,
        )

    # Git diff mode
    if diff_only:
        return await _resolve_git_diff()

    # Stdin mode (only if no paths and stdin is not a TTY)
    has_paths = paths is not None and len(paths) > 0
    if not has_paths and not sys.stdin.isatty():
        stdin_content = await read_stdin()
        return ReviewTarget(
            files=[],
            source_type=TargetSourceType.STDIN,
            stdin_content=stdin_content,
        )

    # Path-based resolution
    if has_paths:
        all_files: list[Path] = []
        for path_str in paths:
            target = resolve_path(path_str)
            all_files.extend(target.files)

        if not all_files:
            print("Error: No files to review", file=sys.stderr)
            sys.exit(1)

        return ReviewTarget(
            files=all_files,
            source_type=TargetSourceType.FILES,
        )

    # No input specified
    print("Error: No input specified. Provide paths, pipe to stdin, or use --diff-only/--pr", file=sys.stderr)
    sys.exit(1)


async def _resolve_git_diff() -> ReviewTarget:
    """Resolve files from git diff.

    Uses git diff --cached or git diff origin/main...HEAD to find changed files.

    Returns:
        ReviewTarget with files from git diff.

    Raises:
        SystemExit: If git command fails or not in a git repo.
    """
    try:
        # First, check if we're in a git repository
        check_process = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "--git-dir",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await check_process.communicate()

        if check_process.returncode != 0:
            print("Error: Not in a git repository", file=sys.stderr)
            sys.exit(1)

        # Try staged changes first
        diff_process = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMRT",  # Added, Copied, Modified, Renamed, Type-changed
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await diff_process.communicate()

        if diff_process.returncode != 0:
            # Fallback to comparing with origin/main
            diff_process = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "origin/main...HEAD",
                "--name-only",
                "--diff-filter=ACMRT",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await diff_process.communicate()

            if diff_process.returncode != 0:
                print("Error: Failed to get git diff", file=sys.stderr)
                print(stderr.decode().strip(), file=sys.stderr)
                sys.exit(1)

        # Parse file list
        files_str = stdout.decode().strip()
        if not files_str:
            print("Warning: No changed files found in git diff", file=sys.stderr)
            return ReviewTarget(
                files=[],
                source_type=TargetSourceType.GIT_DIFF,
            )

        files = [Path(f) for f in files_str.split("\n") if f.strip()]
        # Filter to existing files only
        files = [f for f in files if f.exists() and f.is_file()]

        return ReviewTarget(
            files=files,
            source_type=TargetSourceType.GIT_DIFF,
        )

    except FileNotFoundError:
        print("Error: git command not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Git operation failed: {e}", file=sys.stderr)
        sys.exit(1)


def resolve_targets_sync(
    paths: list[str] | None = None,
    diff_only: bool = False,
    pr_arg: str | None = None,
) -> ReviewTarget:
    """Synchronous wrapper for resolve_targets.

    Convenience function for synchronous contexts.

    Args:
        paths: List of path strings (files, dirs, globs).
        diff_only: If True, use git diff to determine files.
        pr_arg: PR URL or number for PR mode.

    Returns:
        Unified ReviewTarget object.
    """
    return asyncio.run(resolve_targets(paths, diff_only, pr_arg))
