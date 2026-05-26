#!/usr/bin/env python3
"""Orchestrate OpenAlex snapshot data management.

Usage:
    python -m sync sync [--entity ENTITY] [--workers N]
    python -m sync extract [--entity ENTITY] [--workers N]
    python -m sync upload [--batch-size N] [--max-retries N] [--repo-id REPO]
    python -m sync commit [--message MSG]
    python -m sync push
    python -m sync full [--entity ENTITY] [--workers N]

All commands run from the repo root (parent of openalex-snapshot/).
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

from sync.common import SYNC_ROOT


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(SYNC_ROOT)] + list(args),
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


def _find_untracked_parquets() -> list[tuple[int, str]]:
    """Return untracked parquet files sorted by size (smallest first)."""
    all_parquets: set[str] = set()
    for p in SYNC_ROOT.rglob("*.parquet"):
        if not p.name.startswith("._"):
            all_parquets.add(str(p.relative_to(SYNC_ROOT)))

    r = _git("ls-files")
    tracked = {f for f in r.stdout.strip().split("\n") if f.endswith(".parquet")}

    untracked = all_parquets - tracked
    if not untracked:
        return []

    with_size: list[tuple[int, str]] = []
    for rel in untracked:
        full = SYNC_ROOT / rel
        try:
            sz = full.stat().st_size
        except OSError:
            sz = 0
        with_size.append((sz, rel))
    with_size.sort()
    return with_size


def _sync_git_refs() -> None:
    """Fetch remote and reset local index to match (leaves working tree intact)."""
    log("Fetching remote...")
    _git("fetch", "origin")
    # --mixed updates the index to match origin/main's tree without
    # touching the working tree. LFS clean filter means git compares
    # the working tree files (through the pointer) against the index
    # and sees them as matching.
    _git("reset", "--mixed", "origin/main")
    log("Git refs synced")


def cmd_upload(args) -> None:
    """Upload untracked parquet files to HuggingFace via the REST API."""
    from huggingface_hub import HfApi, upload_large_folder

    repo_id = args.repo_id
    num_workers = args.workers

    # Find which parquet files are untracked locally (for reporting)
    untracked = _find_untracked_parquets()
    if not untracked:
        log("All parquet files already tracked locally")
        return

    total = len(untracked)
    log(f"{total} untracked parquet files, uploading via HF API with {num_workers} workers")

    # upload_large_folder handles hashing, pre-upload, committing, and
    # resumption automatically. It uploads ONLY files not already on the
    # server, so re-running after an interruption is safe.
    upload_large_folder(
        repo_id=repo_id,
        folder_path=str(SYNC_ROOT),
        repo_type="dataset",
        allow_patterns=["*.parquet"],
        num_workers=num_workers,
    )

    # Bring local git refs in line with what the server now has
    _sync_git_refs()
    log("ALL DONE")


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

    # upload
    p_upload = subparsers.add_parser("upload", help="Upload untracked parquet to HF via REST API")
    p_upload.add_argument("--workers", type=int, default=8, help="Parallel upload workers (default: 8)")
    p_upload.add_argument("--repo-id", type=str, default="Mearman/OpenAlex", help="HF dataset repo ID")

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
        "upload": cmd_upload,
        "full": cmd_full,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
