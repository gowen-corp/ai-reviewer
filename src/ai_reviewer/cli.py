#!/usr/bin/env python3
"""CLI entry point for ai_reviewer."""

import argparse
import sys
from pathlib import Path

from .config import build_config, ensure_directories, print_config_summary
from .target_resolver import resolve_targets_sync, ReviewTarget


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="ai_reviewer",
        description="AI-powered code review CLI utility",
        epilog="For more information, visit the documentation.",
    )

    # Positional argument for paths (files, directories, globs)
    parser.add_argument(
        "paths",
        nargs="*",
        type=str,
        default=None,
        help="Paths to review: files, directories, or glob patterns (e.g., 'src/**/*.py')",
    )

    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Path to the repository to review (default: current directory)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (e.g., 'qwen/qwen3.5-flash')",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["light", "review", "fix", "analyze"],
        default="review",
        help="Operation mode: light, review, fix, or analyze (default: review)",
    )

    parser.add_argument(
        "--diff-only",
        action="store_true",
        dest="diff_only",
        help="Review only changed files from git diff (staged or origin/main...HEAD)",
    )

    parser.add_argument(
        "--pr",
        type=str,
        default=None,
        metavar="URL_OR_NUMBER",
        help="Pull/Merge Request URL or number to review (e.g., 'https://github.com/owner/repo/pull/123' or '123')",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without making any changes",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()

    # If no arguments provided, show help and exit
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    # Resolve review targets
    try:
        target: ReviewTarget | None = resolve_targets_sync(
            paths=args.paths if args.paths else None,
            diff_only=args.diff_only,
            pr_arg=args.pr,
        )
    except SystemExit:
        # Re-raise SystemExit from target_resolver
        raise

    # Build configuration from all sources
    config = build_config(
        repo_path=args.repo or Path.cwd(),
        model_name=args.model,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    # Ensure directories exist
    ensure_directories(config)

    # Print configuration in verbose mode
    if args.verbose:
        print(f"Repository: {args.repo or Path.cwd()}", file=sys.stderr)
        print(f"Mode: {args.mode}", file=sys.stderr)
        print(f"Target source type: {target.source_type.name}", file=sys.stderr)
        if target.source_type.name == "STDIN":
            print(f"Stdin content length: {len(target.stdin_content or '')}", file=sys.stderr)
        elif target.source_type.name == "PR":
            print(f"PR URL: {target.pr_url}", file=sys.stderr)
            print(f"PR number: {target.pr_number}", file=sys.stderr)
            print(f"Repo name: {target.repo_name}", file=sys.stderr)
        else:
            print(f"Files to review: {len(target.files)}", file=sys.stderr)
            for f in target.files[:5]:  # Show first 5 files
                print(f"  - {f}", file=sys.stderr)
            if len(target.files) > 5:
                print(f"  ... and {len(target.files) - 5} more", file=sys.stderr)
        print_config_summary(config)

    # TODO: Implement actual review logic in subsequent phases
    if args.verbose:
        print("Review functionality will be implemented in subsequent phases.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
