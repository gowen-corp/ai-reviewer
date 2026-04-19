"""LLM response caching for AI Reviewer.

This module provides caching of LLM responses to avoid redundant API calls
for identical requests. Cache keys are based on SHA256 hashes of:
- Target content hash
- Model identifier
- Prompt version
- Pipeline stage name

Cache is stored in XDG cache directory: ~/.cache/ai-reviewer/llm_cache/
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import PathsConfig


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry for an LLM response."""

    key: str  # SHA256 cache key
    created_at: str  # ISO 8601 timestamp
    model: str
    stage: str
    prompt_hash: str
    response_content: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return cls(**data)


class LLMLruCache:
    """LRU cache for LLM responses.

    This class manages caching of LLM responses to reduce API calls
    and costs. Cache entries are stored as individual JSON files
    keyed by their SHA256 hash.

    Attributes:
        cache_dir: Directory for cache files.
        max_entries: Maximum number of entries to keep (for cleanup).
    """

    def __init__(self, paths: PathsConfig, max_entries: int = 1000) -> None:
        """Initialize the LLM cache.

        Args:
            paths: XDG paths configuration.
            max_entries: Maximum entries before cleanup (default 1000).
        """
        self.cache_dir = paths.cache_dir / "llm_cache"
        self.max_entries = max_entries

    def ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get path to cache file for a given key.

        Args:
            key: Cache key (SHA256 hash).

        Returns:
            Path to cache file.
        """
        # Use first 2 chars as subdirectory for better filesystem performance
        subdir = self.cache_dir / key[:2]
        return subdir / f"{key}.json"

    def compute_cache_key(
        self,
        target_hash: str,
        model: str,
        prompt_version: str,
        stage: str,
    ) -> str:
        """Compute SHA256 cache key for a request.

        Args:
            target_hash: Hash of the target content.
            model: Model identifier.
            prompt_version: Version string for the prompt template.
            stage: Pipeline stage name.

        Returns:
            SHA256 hex digest of the combined inputs.
        """
        key_data = f"{target_hash}|{model}|{prompt_version}|{stage}"
        return hashlib.sha256(key_data.encode("utf-8")).hexdigest()

    def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cached response.

        Args:
            key: Cache key (SHA256 hash).

        Returns:
            CacheEntry if found and valid, None otherwise.
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                entry = CacheEntry.from_dict(data)
                logger.debug("Cache hit for key %s", key[:16])
                return entry
        except (json.JSONDecodeError, IOError, OSError, TypeError) as e:
            logger.warning("Failed to read cache entry %s: %s", key[:16], e)
            # Remove corrupted entry
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None

    def set(
        self,
        key: str,
        model: str,
        stage: str,
        prompt_hash: str,
        response_content: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
    ) -> CacheEntry:
        """Store a response in the cache.

        Args:
            key: Cache key (SHA256 hash).
            model: Model identifier used.
            stage: Pipeline stage name.
            prompt_hash: Hash of the prompt content.
            response_content: LLM response content.
            tokens_input: Input tokens used.
            tokens_output: Output tokens generated.
            cost_usd: Cost in USD.

        Returns:
            The created CacheEntry.
        """
        self.ensure_cache_dir()

        entry = CacheEntry(
            key=key,
            created_at=datetime.now(timezone.utc).isoformat(),
            model=model,
            stage=stage,
            prompt_hash=prompt_hash,
            response_content=response_content,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
        )

        cache_path = self._get_cache_path(key)

        try:
            # Ensure subdirectory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, indent=2)
            logger.debug("Cache miss: stored entry for key %s", key[:16])
        except (IOError, OSError) as e:
            logger.warning("Failed to write cache entry: %s", e)

        return entry

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key.

        Returns:
            True if entry exists, False otherwise.
        """
        return self.get(key) is not None

    def delete(self, key: str) -> bool:
        """Delete a specific cache entry.

        Args:
            key: Cache key to delete.

        Returns:
            True if deleted, False if not found.
        """
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.debug("Deleted cache entry %s", key[:16])
                return True
            except OSError as e:
                logger.warning("Failed to delete cache entry: %s", e)
                return False

        return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries deleted.
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        try:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for cache_file in subdir.glob("*.json"):
                        try:
                            cache_file.unlink()
                            count += 1
                        except OSError:
                            pass
                    # Remove empty subdirectory
                    try:
                        subdir.rmdir()
                    except OSError:
                        pass
        except (IOError, OSError) as e:
            logger.warning("Failed to clear cache: %s", e)

        return count

    def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """Remove cache entries older than specified age.

        Args:
            max_age_days: Maximum age in days (default 30).

        Returns:
            Number of entries deleted.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        count = 0

        if not self.cache_dir.exists():
            return 0

        try:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for cache_file in subdir.glob("*.json"):
                        try:
                            with open(cache_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                created_at = datetime.fromisoformat(
                                    data["created_at"].replace("Z", "+00:00")
                                )
                                if created_at < cutoff:
                                    cache_file.unlink()
                                    count += 1
                        except (json.JSONDecodeError, IOError, OSError, KeyError):
                            # Delete corrupted entries
                            try:
                                cache_file.unlink()
                                count += 1
                            except OSError:
                                pass
        except (IOError, OSError) as e:
            logger.warning("Failed to cleanup cache: %s", e)

        return count

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        if not self.cache_dir.exists():
            return {
                "total_entries": 0,
                "total_size_bytes": 0,
                "oldest_entry": None,
                "newest_entry": None,
            }

        total_entries = 0
        total_size = 0
        oldest: str | None = None
        newest: str | None = None

        try:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for cache_file in subdir.glob("*.json"):
                        total_entries += 1
                        total_size += cache_file.stat().st_size

                        try:
                            with open(cache_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                ts = data.get("created_at")
                                if ts:
                                    if oldest is None or ts < oldest:
                                        oldest = ts
                                    if newest is None or ts > newest:
                                        newest = ts
                        except (json.JSONDecodeError, IOError):
                            pass
        except (IOError, OSError):
            pass

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "oldest_entry": oldest,
            "newest_entry": newest,
        }
