"""Pre-check module for running local linters before LLM review.

This module runs static analysis tools (ruff, bandit) on files before
sending them to the LLM. Results are cached by sha256(file_content + tool_version).

If syntax errors or critical issues are found, the file is marked as BLOCKED
and the LLM is not called.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PrecheckStatus(Enum):
    """Status of precheck results."""

    PASS = auto()       # No issues found
    WARN = auto()       # Non-critical issues found
    BLOCK = auto()      # Critical issues (syntax error, secrets, etc.)


@dataclass
class PrecheckIssue:
    """Represents a single precheck issue."""

    severity: str  # critical, major, minor
    category: str  # syntax, security, style, error
    code: str      # Issue code (e.g., E999, B101)
    message: str
    line_number: int | None = None
    column: int | None = None
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity,
            "category": self.category,
            "code": self.code,
            "message": self.message,
            "line_number": self.line_number,
            "column": self.column,
            "file_path": self.file_path,
        }


@dataclass
class PrecheckResult:
    """Result of running precheck on a file."""

    file_path: Path
    status: PrecheckStatus
    issues: list[PrecheckIssue] = field(default_factory=list)
    cached: bool = False
    ruff_output: str = ""
    bandit_output: str = ""

    @property
    def has_syntax_error(self) -> bool:
        """Check if any issues are syntax errors."""
        return any(
            i.category == "syntax" or i.code.startswith("E9")
            for i in self.issues
        )

    @property
    def has_security_issue(self) -> bool:
        """Check if any issues are security-related."""
        return any(i.category == "security" for i in self.issues)

    @property
    def has_critical_issues(self) -> bool:
        """Check if any issues are critical."""
        return any(i.severity == "critical" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "status": self.status.name,
            "has_syntax_error": self.has_syntax_error,
            "has_security_issue": self.has_security_issue,
            "issues_count": len(self.issues),
            "cached": self.cached,
            "issues": [i.to_dict() for i in self.issues],
        }


def compute_cache_key(content: str, tool_name: str, tool_version: str) -> str:
    """Compute cache key for precheck results.

    Args:
        content: File content.
        tool_name: Name of the tool (ruff, bandit).
        tool_version: Version string of the tool.

    Returns:
        SHA256 hash of content + tool version.
    """
    combined = f"{content}:{tool_name}:{tool_version}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def get_tool_version(tool_name: str) -> str:
    """Get version string for a tool.

    Args:
        tool_name: Name of the tool (ruff, bandit).

    Returns:
        Version string, or "unknown" if tool not found.
    """
    try:
        result = subprocess.run(
            [tool_name, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        # Extract version from output (e.g., "ruff 0.1.6" -> "0.1.6")
        output = result.stdout.strip()
        parts = output.split()
        if len(parts) >= 2:
            return parts[-1]
        return output
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def run_ruff_check(file_path: Path, content: str) -> tuple[list[PrecheckIssue], str]:
    """Run ruff check on a file.

    Args:
        file_path: Path to the file.
        content: File content.

    Returns:
        Tuple of (list of issues, raw output).
    """
    # Ruff select codes: E9 (syntax errors), F63 (invalid syntax), F7 (parser errors), F82 (undefined names)
    cmd = [
        "ruff",
        "check",
        "--select=E9,F63,F7,F82",
        "--output-format=json",
        "--force-exclude",
        str(file_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        raw_output = result.stdout

        if not raw_output.strip():
            return [], ""

        # Parse JSON output
        try:
            issues_data = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning("Failed to parse ruff JSON output: %s", raw_output[:200])
            return [], raw_output

        issues = []
        for item in issues_data:
            # Determine severity based on code
            code = item.get("code", "")
            if code.startswith("E9") or code.startswith("F63") or code.startswith("F7"):
                severity = "critical"
                category = "syntax"
            else:
                severity = "major"
                category = "error"

            issues.append(PrecheckIssue(
                severity=severity,
                category=category,
                code=code,
                message=item.get("message", ""),
                line_number=item.get("location", {}).get("row"),
                column=item.get("location", {}).get("column"),
                file_path=str(file_path),
            ))

        return issues, raw_output

    except subprocess.TimeoutExpired:
        logger.warning("Ruff check timed out for %s", file_path)
        return [], ""
    except FileNotFoundError:
        logger.warning("Ruff not found, skipping ruff check")
        return [], ""
    except Exception as e:
        logger.warning("Ruff check failed for %s: %s", file_path, e)
        return [], ""


def run_bandit_check(file_path: Path, content: str) -> tuple[list[PrecheckIssue], str]:
    """Run bandit security check on a file.

    Args:
        file_path: Path to the file.
        content: File content.

    Returns:
        Tuple of (list of issues, raw output).
    """
    # Bandit with low confidence level (-ll) and JSON output
    cmd = [
        "bandit",
        "-f", "json",
        "-ll",  # Low confidence level
        str(file_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        raw_output = result.stdout

        if not raw_output.strip():
            return [], ""

        # Parse JSON output
        try:
            issues_data = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning("Failed to parse bandit JSON output: %s", raw_output[:200])
            return [], raw_output

        # Bandit output structure: {"results": [...], "metrics": {...}}
        results = issues_data.get("results", [])

        issues = []
        for item in results:
            # Bandit severity: HIGH -> critical, MEDIUM -> major, LOW -> minor
            confidence = item.get("confidence", "LOW").upper()
            if confidence == "HIGH":
                severity = "critical"
            elif confidence == "MEDIUM":
                severity = "major"
            else:
                severity = "minor"

            issues.append(PrecheckIssue(
                severity=severity,
                category="security",
                code=item.get("test_id", ""),
                message=item.get("issue_text", ""),
                line_number=item.get("line_number"),
                column=None,
                file_path=str(file_path),
            ))

        return issues, raw_output

    except subprocess.TimeoutExpired:
        logger.warning("Bandit check timed out for %s", file_path)
        return [], ""
    except FileNotFoundError:
        logger.warning("Bandit not found, skipping bandit check")
        return [], ""
    except Exception as e:
        logger.warning("Bandit check failed for %s: %s", file_path, e)
        return [], ""


# In-memory cache for precheck results
_precheck_cache: dict[str, PrecheckResult] = {}


def run_precheck(
    file_path: Path,
    content: str,
    use_cache: bool = True,
) -> PrecheckResult:
    """Run all precheck tools on a file.

    Args:
        file_path: Path to the file.
        content: File content.
        use_cache: Whether to use cached results.

    Returns:
        PrecheckResult with all issues found.
    """
    # Check in-memory cache first
    cache_key = compute_cache_key(
        content,
        "precheck",
        f"ruff:{get_tool_version('ruff')}:bandit:{get_tool_version('bandit')}",
    )

    if use_cache and cache_key in _precheck_cache:
        result = _precheck_cache[cache_key]
        result.cached = True
        logger.debug("Using cached precheck result for %s", file_path)
        return result

    issues: list[PrecheckIssue] = []
    ruff_output = ""
    bandit_output = ""

    # Run ruff check
    ruff_issues, ruff_output = run_ruff_check(file_path, content)
    issues.extend(ruff_issues)

    # Run bandit check
    bandit_issues, bandit_output = run_bandit_check(file_path, content)
    issues.extend(bandit_issues)

    # Determine overall status
    if any(i.severity == "critical" for i in issues):
        status = PrecheckStatus.BLOCK
    elif any(i.severity == "major" for i in issues):
        status = PrecheckStatus.WARN
    else:
        status = PrecheckStatus.PASS

    result = PrecheckResult(
        file_path=file_path,
        status=status,
        issues=issues,
        cached=False,
        ruff_output=ruff_output,
        bandit_output=bandit_output,
    )

    # Store in cache
    if use_cache:
        _precheck_cache[cache_key] = result

    return result


def clear_cache() -> None:
    """Clear the precheck cache."""
    _precheck_cache.clear()
    logger.debug("Precheck cache cleared")
