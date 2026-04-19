"""History management for AI Reviewer.

This module handles recording and retrieving review history in JSONL format.
History is stored in XDG data directory: ~/.local/share/ai-reviewer/history/

Each history entry contains:
- ts: ISO 8601 timestamp
- mode: Review mode (light, standard, deep)
- model: Model identifier
- target_hash: SHA256 hash of reviewed content
- tokens: Token usage (input, output, cached)
- cost_usd: Cost in USD
- verdict: Review verdict (PASS, WARN, BLOCK)
- findings_count: Number of findings
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .pipeline import ReviewResult, Verdict
from .models import PathsConfig


logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """Single history record."""

    ts: str  # ISO 8601 timestamp
    mode: str
    model: str
    target_hash: str
    verdict: str
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0
    findings_count: int = 0
    critical_count: int = 0
    major_count: int = 0
    minor_count: int = 0
    blocked: bool = False
    blocked_at_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        """Create from dictionary."""
        return cls(**data)


class HistoryManager:
    """Manages review history storage and retrieval.

    This class provides methods to append new history entries and
    read existing entries from the JSONL history file.

    Attributes:
        history_dir: Directory for history files.
        history_file: Path to the main history JSONL file.
    """

    def __init__(self, paths: PathsConfig) -> None:
        """Initialize the history manager.

        Args:
            paths: XDG paths configuration.
        """
        self.history_dir = paths.data_dir / "history"
        self.history_file = self.history_dir / "reviews.jsonl"

    def ensure_history_dir(self) -> None:
        """Ensure history directory exists."""
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def append_entry(
        self,
        result: ReviewResult,
        model: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        tokens_cached: int = 0,
        cost_usd: float = 0.0,
    ) -> HistoryEntry:
        """Append a new history entry.

        Args:
            result: Review result to record.
            model: Model identifier used.
            tokens_input: Input tokens used.
            tokens_output: Output tokens generated.
            tokens_cached: Cached tokens (if any).
            cost_usd: Cost in USD.

        Returns:
            The created HistoryEntry.
        """
        entry = HistoryEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            mode=result.mode.name.lower(),
            model=model,
            target_hash=result.target.content_hash if hasattr(result.target, 'content_hash') and result.target.content_hash else "unknown",
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_cached=tokens_cached,
            cost_usd=cost_usd,
            verdict=result.overall_verdict.value,
            findings_count=len(result.all_findings),
            critical_count=result.critical_count,
            major_count=result.major_count,
            minor_count=result.minor_count,
            blocked=result.blocked,
            blocked_at_stage=result.blocked_at_stage,
        )

        self._write_entry(entry)
        return entry

    def _write_entry(self, entry: HistoryEntry) -> None:
        """Write a single entry to the history file.

        Args:
            entry: History entry to write.
        """
        self.ensure_history_dir()

        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except (IOError, OSError) as e:
            logger.warning("Failed to write history entry: %s", e)

    def read_entries(self, limit: int | None = None) -> list[HistoryEntry]:
        """Read history entries from file.

        Args:
            limit: Maximum number of entries to return (None for all).

        Returns:
            List of HistoryEntry objects, newest first.
        """
        if not self.history_file.exists():
            return []

        entries: list[HistoryEntry] = []

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entries.append(HistoryEntry.from_dict(data))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("Invalid history entry: %s", e)
                        continue
        except (IOError, OSError) as e:
            logger.warning("Failed to read history file: %s", e)
            return []

        # Return newest first
        entries.reverse()

        if limit is not None:
            return entries[:limit]

        return entries

    def get_last(self, count: int = 5) -> list[HistoryEntry]:
        """Get the last N history entries.

        Args:
            count: Number of entries to retrieve.

        Returns:
            List of the most recent HistoryEntry objects.
        """
        return self.read_entries(limit=count)

    def clear_history(self) -> None:
        """Clear all history entries."""
        if self.history_file.exists():
            try:
                self.history_file.unlink()
                logger.info("History cleared")
            except OSError as e:
                logger.warning("Failed to clear history: %s", e)

    def get_statistics(self) -> dict[str, Any]:
        """Get aggregate statistics from history.

        Returns:
            Dictionary with statistics (total reviews, avg cost, etc.).
        """
        entries = self.read_entries()

        if not entries:
            return {
                "total_reviews": 0,
                "total_cost_usd": 0.0,
                "total_tokens": 0,
                "pass_count": 0,
                "warn_count": 0,
                "block_count": 0,
            }

        total_cost = sum(e.cost_usd for e in entries)
        total_tokens = sum(e.tokens_input + e.tokens_output for e in entries)
        pass_count = sum(1 for e in entries if e.verdict == "PASS")
        warn_count = sum(1 for e in entries if e.verdict == "WARN")
        block_count = sum(1 for e in entries if e.verdict == "BLOCK")

        return {
            "total_reviews": len(entries),
            "total_cost_usd": total_cost,
            "avg_cost_usd": total_cost / len(entries) if entries else 0.0,
            "total_tokens": total_tokens,
            "pass_count": pass_count,
            "warn_count": warn_count,
            "block_count": block_count,
        }


def print_history_table(entries: list[HistoryEntry]) -> None:
    """Print history entries as a formatted table.

    Args:
        entries: List of history entries to display.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if not entries:
        console.print("[yellow]No history entries found.[/yellow]")
        return

    table = Table(title="Review History")

    table.add_column("Timestamp", style="cyan")
    table.add_column("Mode", style="magenta")
    table.add_column("Model", style="blue")
    table.add_column("Verdict", style="green")
    table.add_column("Findings", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost ($)", justify="right")

    for entry in entries:
        verdict_style = {
            "PASS": "green",
            "WARN": "yellow",
            "BLOCK": "red",
        }.get(entry.verdict, "white")

        table.add_row(
            entry.ts[:19],  # Show date and time without microseconds
            entry.mode,
            entry.model[:30] + "..." if len(entry.model) > 30 else entry.model,
            f"[{verdict_style}]{entry.verdict}[/{verdict_style}]",
            str(entry.findings_count),
            str(entry.tokens_input + entry.tokens_output),
            f"{entry.cost_usd:.4f}",
        )

    console.print(table)
