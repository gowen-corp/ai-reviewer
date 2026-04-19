#!/usr/bin/env python3
"""CLI entry point for ai_reviewer."""

import argparse
import sys
from pathlib import Path


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
    
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path to the repository to review (default: current directory)",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["review", "fix", "analyze"],
        default="review",
        help="Operation mode: review, fix, or analyze (default: review)",
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
    
    # Log configuration in verbose mode
    if args.verbose:
        print(f"Repository: {args.repo}", file=sys.stderr)
        print(f"Mode: {args.mode}", file=sys.stderr)
        print(f"Dry run: {args.dry_run}", file=sys.stderr)
    
    # TODO: Implement actual review logic in subsequent phases
    if args.verbose:
        print("Review functionality will be implemented in subsequent phases.", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
