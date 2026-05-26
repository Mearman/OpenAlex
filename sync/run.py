#!/usr/bin/env python3
"""Orchestrate OpenAlex snapshot data management.

Usage:
    python sync/run.py sync [--entity ENTITY] [--workers N]
    python sync/run.py extract [--entity ENTITY] [--workers N]
    python sync/run.py commit [--message MSG]
    python sync/run.py push
    python sync/run.py full [--entity ENTITY] [--workers N]

All commands run from the repo root (openalex-snapshot/).
PYTHONPATH must include the repo root for `from sync.xxx import` to work.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT)] + list(args),
        capture_output=True, text=True, check=check,
    )


def cmd_sync(args) -> None:
    """Sync .gz files from OpenAlex S3."""
    from sync.download import run_sync

    run_sync(
        dest=REPO_ROOT,
        entity=args.entity,
        workers=args.workers or 8,
        dry_run=args.dry_run,
        delete=not args.no_delete,
    )


def cmd_extract(args) -> None:
    """Extract parquet from snapshot JSONL."""
    from sync.extract import main as extract_main

    extract_main(
        entity=args.entity,
        force=args.force,
        workers=args.workers,
        slice_index=args.slice_index,
        slice_total=args.slice_total,
    )
    # Metadata is updated automatically by extract.py per-entity.
    # If extracting a single entity, the per-entity update already ran.
    # If extracting all entities, each one updated individually.
    # No additional step needed.


def cmd_commit(args) -> None:
    """Stage and commit changes by file type."""
    message = args.message or f"update: {time.strftime('%Y-%m-%d %H:%M')}"

    # Stage .gz files (snapshot)
    _git("add", "*.gz")
    r = _git("diff", "--cached", "--name-only")
    gz_files = [f for f in r.stdout.strip().split("\n") if f]
    if gz_files:
        _git("commit", "-m", f"snapshot: {message}")

    # Stage .parquet files (extracted)
    _git("add", "*.parquet")
    r = _git("diff", "--cached", "--name-only")
    pq_files = [f for f in r.stdout.strip().split("\n") if f]
    if pq_files:
        _git("commit", "-m", f"parquet: {message}")

    # Stage metadata (.gitattributes, .gitignore, sync/, README)
    _git("add", ".gitattributes", ".gitignore", "sync/", "README.md")
    r = _git("diff", "--cached", "--name-only")
    meta_files = [f for f in r.stdout.strip().split("\n") if f]
    if meta_files:
        _git("commit", "-m", f"meta: {message}")

    if not gz_files and not pq_files and not meta_files:
        log("Nothing to commit")


def cmd_push(args) -> None:
    """Push to remote."""
    _git("push", check=False)


def cmd_full(args) -> None:
    """Full pipeline: sync → extract → commit → push."""
    cmd_sync(args)
    cmd_extract(args)
    cmd_commit(args)
    cmd_push(args)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="OpenAlex snapshot data management",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sync
    p_sync = subparsers.add_parser("sync", help="Sync .gz from S3")
    p_sync.add_argument("--entity", type=str, default=None)
    p_sync.add_argument("--workers", type=int, default=8)
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.add_argument("--no-delete", action="store_true")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract parquet")
    p_extract.add_argument("--entity", type=str, default=None)
    p_extract.add_argument("--workers", type=int, default=None)
    p_extract.add_argument("--slice-index", type=int, default=None)
    p_extract.add_argument("--slice-total", type=int, default=None)
    p_extract.add_argument("--force", action="store_true")

    # commit
    p_commit = subparsers.add_parser("commit", help="Git add + commit")
    p_commit.add_argument("--message", "-m", type=str, default=None)

    # push
    subparsers.add_parser("push", help="Git push")

    # full
    p_full = subparsers.add_parser("full", help="sync → extract → commit → push")
    p_full.add_argument("--entity", type=str, default=None)
    p_full.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()

    handlers = {
        "sync": cmd_sync,
        "extract": cmd_extract,
        "commit": cmd_commit,
        "push": cmd_push,
        "full": cmd_full,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
