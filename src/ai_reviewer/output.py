"""Output formatting for AI Reviewer results.

This module handles formatting review results in different formats
(markdown for humans, JSON for machines/CI) and manages exit codes.

Exit codes:
    0 = PASS - No issues found
    1 = WARNINGS - Non-blocking issues found
    2 = BLOCKED - Critical issues found, review blocked
    3 = ERROR - System/configuration error
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .pipeline import ReviewResult, Verdict, Finding


# Exit code constants
EXIT_PASS = 0
EXIT_WARNINGS = 1
EXIT_BLOCKED = 2
EXIT_ERROR = 3


@dataclass
class FormattedOutput:
    """Container for formatted review output."""

    content: str
    format_type: str  # "markdown" or "json"


def get_exit_code(results: list[ReviewResult]) -> int:
    """Determine exit code based on review results.

    Args:
        results: List of review results.

    Returns:
        Exit code (0=PASS, 1=WARNINGS, 2=BLOCKED).
    """
    if not results:
        return EXIT_PASS

    has_blocked = any(r.blocked or r.overall_verdict == Verdict.BLOCK for r in results)
    has_warnings = any(
        r.overall_verdict == Verdict.WARN or 
        any(f.severity in ("critical", "major") for f in r.all_findings)
        for r in results
    )

    if has_blocked:
        return EXIT_BLOCKED
    elif has_warnings:
        return EXIT_WARNINGS
    else:
        return EXIT_PASS


def _format_finding_markdown(finding: Finding, index: int) -> str:
    """Format a single finding as markdown.

    Args:
        finding: Finding to format.
        index: Finding index for numbering.

    Returns:
        Markdown-formatted string.
    """
    severity_emoji = {
        "critical": "🔴",
        "major": "🟠",
        "minor": "🟡",
    }.get(finding.severity, "⚪")

    lines = [
        f"#### {severity_emoji} #{index}: {finding.category.upper()}",
        "",
        f"**Severity:** {finding.severity}",
        f"**Description:** {finding.description}",
    ]

    if finding.line_number is not None:
        lines.append(f"**Line:** {finding.line_number}")

    if finding.file_path:
        lines.append(f"**File:** `{finding.file_path}`")

    if finding.suggestion:
        lines.append("")
        lines.append(f"**Suggestion:** {finding.suggestion}")

    return "\n".join(lines)


def _format_result_markdown(result: ReviewResult) -> str:
    """Format a single review result as markdown.

    Args:
        result: Review result to format.

    Returns:
        Markdown-formatted string.
    """
    verdict_emoji = {
        Verdict.PASS: "✅",
        Verdict.WARN: "⚠️",
        Verdict.BLOCK: "🚫",
    }.get(result.overall_verdict, "❓")

    lines = [
        f"### {verdict_emoji} Review Result: `{result.target_hash if hasattr(result.target, 'target_hash') else 'unknown'}`",
        "",
        f"**Verdict:** {result.overall_verdict.value}",
        f"**Mode:** {result.mode.name}",
    ]

    if result.blocked:
        lines.append(f"**Blocked at stage:** {result.blocked_at_stage or 'N/A'}")

    lines.append("")
    lines.append(f"**Summary:** {result.summary}")
    lines.append("")

    if result.all_findings:
        lines.append(f"**Findings:** {len(result.all_findings)} total")
        lines.append(f"- Critical: {result.critical_count}")
        lines.append(f"- Major: {result.major_count}")
        lines.append(f"- Minor: {result.minor_count}")
        lines.append("")

        for i, finding in enumerate(result.all_findings, 1):
            lines.append(_format_finding_markdown(finding, i))
            lines.append("")
    else:
        lines.append("**No findings.**")
        lines.append("")

    if result.stage_results:
        lines.append("**Stage Results:**")
        lines.append("")
        for sr in result.stage_results:
            stage_verdict = {
                Verdict.PASS: "✅",
                Verdict.WARN: "⚠️",
                Verdict.BLOCK: "🚫",
            }.get(sr.verdict, "❓")
            lines.append(f"- {sr.stage_name}: {stage_verdict} ({len(sr.findings)} findings)")
        lines.append("")

    return "\n".join(lines)


def format_markdown(results: list[ReviewResult], dry_run: bool = False) -> str:
    """Format review results as markdown.

    Args:
        results: List of review results.
        dry_run: Whether this is a dry run.

    Returns:
        Complete markdown document.
    """
    lines = [
        "# AI Code Review Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Dry Run:** {'Yes' if dry_run else 'No'}",
        "",
        "---",
        "",
    ]

    if not results:
        lines.append("*No files were reviewed.*")
    else:
        for result in results:
            lines.append(_format_result_markdown(result))
            lines.append("---")
            lines.append("")

    # Summary section
    total_findings = sum(len(r.all_findings) for r in results)
    total_critical = sum(r.critical_count for r in results)
    total_major = sum(r.major_count for r in results)
    total_minor = sum(r.minor_count for r in results)

    lines.extend([
        "## Summary",
        "",
        f"**Files Reviewed:** {len(results)}",
        f"**Total Findings:** {total_findings}",
        f"- Critical: {total_critical}",
        f"- Major: {total_major}",
        f"- Minor: {total_minor}",
        "",
    ])

    return "\n".join(lines)


def format_json(results: list[ReviewResult], dry_run: bool = False) -> str:
    """Format review results as JSON.

    Args:
        results: List of review results.
        dry_run: Whether this is a dry run.

    Returns:
        JSON string.
    """
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "results": [r.to_dict() for r in results],
        "summary": {
            "files_reviewed": len(results),
            "total_findings": sum(len(r.all_findings) for r in results),
            "critical_count": sum(r.critical_count for r in results),
            "major_count": sum(r.major_count for r in results),
            "minor_count": sum(r.minor_count for r in results),
        },
    }

    return json.dumps(output, indent=2)


def print_results(
    results: list[ReviewResult],
    format_type: str = "markdown",
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """Print review results to stdout.

    Args:
        results: List of review results.
        format_type: Output format ("markdown" or "json").
        verbose: Enable verbose output.
        dry_run: Whether this is a dry run.
    """
    console = Console(file=sys.stdout)

    if format_type == "json":
        output = format_json(results, dry_run=dry_run)
        console.print(output)
    else:  # markdown
        output = format_markdown(results, dry_run=dry_run)
        if verbose:
            # Render markdown with rich formatting
            md = Markdown(output)
            console.print(md)
        else:
            # Plain text output
            console.print(output)


def print_dry_run_plan(
    files: list[str],
    model: str,
    mode: str,
    estimated_tokens: int = 0,
    estimated_cost: float = 0.0,
) -> None:
    """Print dry run plan without executing review.

    Args:
        files: List of files to be reviewed.
        model: Model identifier.
        mode: Review mode.
        estimated_tokens: Estimated token count.
        estimated_cost: Estimated cost in USD.
    """
    console = Console(file=sys.stderr)

    console.print("\n[bold blue]=== DRY RUN PLAN ===[/bold blue]\n")

    console.print(f"[bold]Mode:[/bold] {mode}")
    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Files to review:[/bold] {len(files)}")

    if files:
        console.print("\n[bold]File list:[/bold]")
        for f in files[:10]:  # Show first 10
            console.print(f"  - {f}")
        if len(files) > 10:
            console.print(f"  ... and {len(files) - 10} more")

    console.print(f"\n[bold]Estimated tokens:[/bold] {estimated_tokens:,}")
    console.print(f"[bold]Estimated cost:[/bold] ${estimated_cost:.4f}")

    console.print("\n[yellow]No API calls will be made in dry-run mode.[/yellow]\n")
