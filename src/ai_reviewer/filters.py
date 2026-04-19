"""Filters module for detecting and sanitizing secrets/PII in code.

This module provides regex-based scanning for common secret patterns
like AWS keys, private keys, passwords, API tokens, etc.

When --sanitize is enabled, detected secrets are replaced with [REDACTED]
before sending code to the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class SecretType(Enum):
    """Types of detected secrets."""

    AWS_KEY = auto()
    PRIVATE_KEY = auto()
    PASSWORD = auto()
    API_TOKEN = auto()
    GENERIC_SECRET = auto()
    GIT_MERGE_CONFLICT = auto()


@dataclass
class SecretMatch:
    """Represents a detected secret match."""

    secret_type: SecretType
    pattern_name: str
    matched_text: str
    line_number: int
    column: int | None = None
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "secret_type": self.secret_type.name,
            "pattern_name": self.pattern_name,
            "matched_text": self.matched_text[:20] + "..." if len(self.matched_text) > 20 else self.matched_text,
            "line_number": self.line_number,
            "column": self.column,
            "file_path": self.file_path,
        }


@dataclass
class FilterResult:
    """Result of filtering/scanning content for secrets."""

    file_path: Path | None
    has_secrets: bool
    secrets: list[SecretMatch] = field(default_factory=list)
    sanitized_content: str | None = None
    original_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path) if self.file_path else None,
            "has_secrets": self.has_secrets,
            "secrets_count": len(self.secrets),
            "sanitized": self.sanitized_content is not None,
            "secrets": [s.to_dict() for s in self.secrets],
        }


# Regex patterns for secret detection
# Order matters: more specific patterns first
SECRET_PATTERNS: list[tuple[str, re.Pattern, SecretType]] = [
    # AWS Access Key ID (starts with AKIA)
    (
        "AWS_ACCESS_KEY",
        re.compile(r'AKIA[0-9A-Z]{16}'),
        SecretType.AWS_KEY,
    ),
    # AWS Secret Access Key (40 characters, base64-like)
    (
        "AWS_SECRET_KEY",
        re.compile(r'(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)\s*[=:]\s*["\']?([A-Za-z0-9/+=]{40})["\']?'),
        SecretType.AWS_KEY,
    ),
    # OpenRouter API key pattern
    (
        "OPENROUTER_KEY",
        re.compile(r'sk-or-[a-zA-Z0-9]{32,}'),
        SecretType.API_TOKEN,
    ),
    # Generic API key patterns
    (
        "API_KEY_ASSIGNMENT",
        re.compile(r'(?:api[_-]?key|apikey)\s*[=:]\s*["\']([^"\']{8,})["\']', re.IGNORECASE),
        SecretType.API_TOKEN,
    ),
    # Private key header
    (
        "PRIVATE_KEY_HEADER",
        re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----'),
        SecretType.PRIVATE_KEY,
    ),
    # Password assignments
    (
        "PASSWORD_ASSIGNMENT",
        re.compile(r'(?:password|passwd|pwd)\s*[=:]\s*["\']([^"\']{4,})["\']', re.IGNORECASE),
        SecretType.PASSWORD,
    ),
    # Generic secret/token assignments
    (
        "SECRET_ASSIGNMENT",
        re.compile(r'(?:secret|token|auth)\s*[=:]\s*["\']([^"\']{8,})["\']', re.IGNORECASE),
        SecretType.GENERIC_SECRET,
    ),
    # Git merge conflict markers
    (
        "GIT_MERGE_CONFLICT",
        re.compile(r'^<{7}\s+\w|^={7}$|^>{7}\s+\w', re.MULTILINE),
        SecretType.GIT_MERGE_CONFLICT,
    ),
]

# Replacement string for sanitized secrets
REDACTED_PLACEHOLDER = "[REDACTED]"


def scan_line_for_secrets(line: str, line_number: int, file_path: str | None = None) -> list[SecretMatch]:
    """Scan a single line for secret patterns.

    Args:
        line: Line content to scan.
        line_number: Line number (1-indexed).
        file_path: Optional file path for reporting.

    Returns:
        List of SecretMatch objects for detected secrets.
    """
    matches = []

    for pattern_name, regex, secret_type in SECRET_PATTERNS:
        for match in regex.finditer(line):
            matches.append(SecretMatch(
                secret_type=secret_type,
                pattern_name=pattern_name,
                matched_text=match.group(0),
                line_number=line_number,
                column=match.start(),
                file_path=file_path,
            ))

    return matches


def scan_content_for_secrets(
    content: str,
    file_path: Path | None = None,
) -> FilterResult:
    """Scan content for secret patterns.

    Args:
        content: Content to scan.
        file_path: Optional file path for reporting.

    Returns:
        FilterResult with all detected secrets.
    """
    all_secrets: list[SecretMatch] = []
    lines = content.splitlines(keepends=False)

    for line_num, line in enumerate(lines, start=1):
        secrets = scan_line_for_secrets(line, line_num, str(file_path) if file_path else None)
        all_secrets.extend(secrets)

    return FilterResult(
        file_path=file_path,
        has_secrets=len(all_secrets) > 0,
        secrets=all_secrets,
        original_content=content,
    )


def sanitize_content(content: str, secrets: list[SecretMatch]) -> str:
    """Sanitize content by replacing detected secrets with [REDACTED].

    Args:
        content: Original content.
        secrets: List of detected secrets.

    Returns:
        Sanitized content with secrets replaced.
    """
    if not secrets:
        return content

    # Sort secrets by position (reverse order to replace from end to start)
    # This preserves positions for earlier replacements
    sorted_secrets = sorted(secrets, key=lambda s: (s.line_number, s.column or 0), reverse=True)

    lines = content.splitlines(keepends=True)

    for secret in sorted_secrets:
        line_idx = secret.line_number - 1
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            # Replace the matched text in the line
            # Handle case where column is known
            if secret.column is not None:
                start = secret.column
                end = start + len(secret.matched_text)
                if end <= len(line):
                    lines[line_idx] = line[:start] + REDACTED_PLACEHOLDER + line[end:]
            else:
                # Fallback: replace first occurrence of matched text in the line
                lines[line_idx] = line.replace(secret.matched_text, REDACTED_PLACEHOLDER, 1)

    return "".join(lines)


def filter_and_sanitize(
    content: str,
    file_path: Path | None = None,
    do_sanitize: bool = False,
) -> FilterResult:
    """Scan content for secrets and optionally sanitize.

    Args:
        content: Content to scan and potentially sanitize.
        file_path: Optional file path for reporting.
        do_sanitize: Whether to sanitize detected secrets.

    Returns:
        FilterResult with secrets and optionally sanitized content.
    """
    result = scan_content_for_secrets(content, file_path)

    if do_sanitize and result.has_secrets:
        result.sanitized_content = sanitize_content(content, result.secrets)

    return result


def has_blocking_secrets(secrets: list[SecretMatch]) -> bool:
    """Check if any secrets should block LLM review.

    Blocking secrets include:
    - Git merge conflicts
    - Actual credentials (AWS keys, private keys)
    - Passwords

    Args:
        secrets: List of detected secrets.

    Returns:
        True if any blocking secrets are found.
    """
    blocking_types = {
        SecretType.GIT_MERGE_CONFLICT,
        SecretType.AWS_KEY,
        SecretType.PRIVATE_KEY,
        SecretType.PASSWORD,
    }

    return any(s.secret_type in blocking_types for s in secrets)


def get_blocking_reason(secrets: list[SecretMatch]) -> str | None:
    """Get human-readable reason for blocking.

    Args:
        secrets: List of detected secrets.

    Returns:
        Reason string if blocking secrets found, None otherwise.
    """
    if not secrets:
        return None

    types_found = {s.secret_type for s in secrets}

    reasons = []
    if SecretType.GIT_MERGE_CONFLICT in types_found:
        reasons.append("Git merge conflict markers detected")
    if SecretType.AWS_KEY in types_found:
        reasons.append("AWS credentials detected")
    if SecretType.PRIVATE_KEY in types_found:
        reasons.append("Private key detected")
    if SecretType.PASSWORD in types_found:
        reasons.append("Password/credential detected")

    return "; ".join(reasons) if reasons else None


def filter_file(
    file_path: Path,
    do_sanitize: bool = False,
) -> FilterResult:
    """Scan a file for secrets and optionally sanitize.

    Args:
        file_path: Path to the file.
        do_sanitize: Whether to sanitize detected secrets.

    Returns:
        FilterResult with secrets and optionally sanitized content.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (IOError, OSError, UnicodeDecodeError) as e:
        return FilterResult(
            file_path=file_path,
            has_secrets=False,
            secrets=[],
            original_content="",
        )

    return filter_and_sanitize(content, file_path, do_sanitize)
