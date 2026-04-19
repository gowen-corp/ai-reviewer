#!/usr/bin/env python3
"""CLI entry point for ai_reviewer."""

import argparse
import asyncio
import sys
from pathlib import Path

from .config import build_config, ensure_directories, print_config_summary
from .target_resolver import resolve_targets_sync, ReviewTarget
from .output import (
    print_results,
    print_dry_run_plan,
    get_exit_code,
    EXIT_PASS,
    EXIT_WARNINGS,
    EXIT_BLOCKED,
    EXIT_ERROR,
)
from .history import HistoryManager, print_history_table
from .models import Config


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

    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        dest="output_format",
        help="Output format: markdown (human-readable) or json (machine/CI)",
    )

    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable writing review history",
    )

    # Subcommand for viewing history (optional, not required for normal operation)
    # Note: We use a separate approach to avoid conflicts with positional 'paths' argument
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    history_parser = subparsers.add_parser("history", help="View review history")
    history_parser.add_argument(
        "--last",
        type=int,
        default=5,
        help="Number of recent entries to show (default: 5)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0=PASS, 1=WARNINGS, 2=BLOCKED, 3=ERROR).
    """
    # If no arguments provided, show help and exit
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser = create_parser()
        parser.print_help()
        return EXIT_PASS

    # Check for history command as first argument before full parsing
    # This avoids conflicts with positional 'paths' argument
    if argv[0] == "history":
        parser = create_parser()
        args = parser.parse_args(argv)
        return _handle_history_command(args.last)

    # Normal parsing for review mode
    parser = create_parser()
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
    except Exception as e:
        print(f"Error resolving targets: {e}", file=sys.stderr)
        return EXIT_ERROR

    # Build configuration from all sources
    try:
        config = build_config(
            repo_path=args.repo or Path.cwd(),
            model_name=args.model,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
    except SystemExit:
        # Re-raise SystemExit (e.g., missing API key)
        raise
    except Exception as e:
        print(f"Error building configuration: {e}", file=sys.stderr)
        return EXIT_ERROR

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

    # Handle dry-run mode
    if args.dry_run:
        return _handle_dry_run(target, config, args)

    # Run the actual review
    return asyncio.run(_run_review(target, config, args))


def _handle_history_command(last: int) -> int:
    """Handle the 'history' subcommand.

    Args:
        last: Number of recent entries to show.

    Returns:
        Exit code.
    """
    from .models import get_xdg_paths

    paths = get_xdg_paths()
    history_manager = HistoryManager(paths)
    entries = history_manager.get_last(last)
    print_history_table(entries)
    return EXIT_PASS


def _handle_dry_run(target: ReviewTarget, config: Config, args: argparse.Namespace) -> int:
    """Handle dry-run mode.

    Args:
        target: Review target.
        config: Configuration.
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    # Get file list
    files = [str(f) for f in target.files] if hasattr(target, 'files') else []

    # Print dry run plan
    print_dry_run_plan(
        files=files,
        model=config.model.name,
        mode=args.mode,
        estimated_tokens=0,  # Would need token estimation logic
        estimated_cost=0.0,
    )

    return EXIT_PASS


async def _run_review(
    target: ReviewTarget,
    config: Config,
    args: argparse.Namespace,
) -> int:
    """Run the actual code review.

    Args:
        target: Review target.
        config: Configuration.
        args: Parsed arguments.

    Returns:
        Exit code based on review results.
    """
    from .llm_client import LLMClient
    from .pipeline import ReviewPipeline, ReviewMode, run_review
    from .token_tracker import TokenTracker

    # Initialize LLM client
    llm_client = LLMClient(
        api_key=config.auth.api_key,
        base_url=config.auth.base_url,
        model=config.model.name,
        timeout=config.model.timeout,
    )

    # Initialize token tracker
    token_tracker = TokenTracker()

    # Map mode string to ReviewMode
    mode_map = {
        "light": ReviewMode.LIGHT,
        "review": ReviewMode.STANDARD,
        "fix": ReviewMode.STANDARD,
        "analyze": ReviewMode.DEEP,
    }
    review_mode = mode_map.get(args.mode.lower(), ReviewMode.STANDARD)

    try:
        # Run review pipeline
        results = await run_review(
            target=target,
            llm_client=llm_client,
            mode=args.mode,
            fail_fast_threshold=1,
        )

        # Get token usage from tracker
        tokens_input = token_tracker.input_tokens
        tokens_output = token_tracker.output_tokens
        tokens_cached = token_tracker.cached_tokens
        cost_usd = token_tracker.total_cost

        # Print results
        print_results(
            results=results,
            format_type=args.output_format,
            verbose=args.verbose,
            dry_run=False,
        )

        # Record history (unless disabled)
        if not args.no_history and results:
            history_manager = HistoryManager(config.paths)
            for result in results:
                history_manager.append_entry(
                    result=result,
                    model=config.model.name,
                    tokens_input=tokens_input // max(len(results), 1),
                    tokens_output=tokens_output // max(len(results), 1),
                    tokens_cached=tokens_cached // max(len(results), 1),
                    cost_usd=cost_usd / max(len(results), 1),
                )

        # Determine exit code
        return get_exit_code(results)

    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        else:
            print(f"Review failed: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
