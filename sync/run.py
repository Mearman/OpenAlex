#!/usr/bin/env python3
"""Orchestrate OpenAlex snapshot data management.

Usage:
    python -m sync sync [--entity ENTITY] [--workers N]
    python -m sync extract [--entity ENTITY] [--workers N]
    python -m sync upload [--batch-size N] [--max-retries N] [--repo-id REPO]
    python -m sync commit [--message MSG]
    python -m sync push
    python -m sync verify [--entity ENTITY] [--workers N]
    python -m sync full [--entity ENTITY] [--workers N] [--verify]

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


def _validate_slice(slice_index: int | None, slice_total: int | None) -> None:
    """Validate that --slice-index and --slice-total form a coherent pair.

    Either both are provided (and 0 <= index < total), or neither is.
    """
    if slice_index is None and slice_total is None:
        return
    if slice_index is None or slice_total is None:
        raise SystemExit(
            "--slice-index and --slice-total must be provided together"
        )
    if slice_total <= 0:
        raise SystemExit("--slice-total must be a positive integer")
    if not (0 <= slice_index < slice_total):
        raise SystemExit(
            f"--slice-index ({slice_index}) must satisfy "
            f"0 <= slice-index < slice-total ({slice_total})"
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
    _validate_slice(args.slice_index, args.slice_total)

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


# Files deleted per HF commit when pruning obsolete remote parquets.
_HF_DELETE_CHUNK = 100


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
    """Reconcile the HuggingFace dataset's parquet files with the local set.

    Makes the remote mirror the canonical local extraction exactly:
      1. Upload new and changed parquet files. ``upload_large_folder`` hashes
         each file and uploads only those the server lacks or whose content
         differs, so a shard that went from empty to populated (e.g.
         ``external_ids``) is re-uploaded, and a renamed table lands at its new
         path.
      2. Prune remote parquet files absent from the local set — phantom
         directories, renamed tables, dropped relationships — so obsolete files
         from a previous (buggy) extraction don't linger on the dataset.

    Pass ``--no-prune`` to upload additively without deleting anything.
    """
    from huggingface_hub import CommitOperationDelete, HfApi, upload_large_folder

    repo_id = args.repo_id
    api = HfApi()

    # Canonical local parquet set (paths relative to the repo root, posix form).
    local: set[str] = {
        p.relative_to(SYNC_ROOT).as_posix()
        for p in SYNC_ROOT.rglob("*.parquet")
        if not p.name.startswith("._")
    }
    log(f"{len(local)} local parquet files; uploading new/changed with {args.workers} workers")

    upload_large_folder(
        repo_id=repo_id,
        folder_path=str(SYNC_ROOT),
        repo_type="dataset",
        allow_patterns=["*.parquet"],
        ignore_patterns=["._*"],  # Apple Double files
        num_workers=args.workers,
    )

    # Prune remote parquets that no longer exist locally so HF mirrors local.
    if not args.no_prune:
        remote = [
            f for f in api.list_repo_files(repo_id, repo_type="dataset")
            if f.endswith(".parquet")
        ]
        obsolete = sorted(set(remote) - local)
        if obsolete:
            log(f"Pruning {len(obsolete)} obsolete remote parquet files")
            for i in range(0, len(obsolete), _HF_DELETE_CHUNK):
                batch = obsolete[i:i + _HF_DELETE_CHUNK]
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=[CommitOperationDelete(path_in_repo=f) for f in batch],
                    commit_message=f"reconcile: prune {len(batch)} obsolete parquet files",
                )
            log(f"Pruned {len(obsolete)} obsolete remote files")
        else:
            log("No obsolete remote parquet files to prune")

    # Bring local git refs in line with what the server now has
    _sync_git_refs()
    log("ALL DONE")


def cmd_verify(args) -> None:
    """Deep self-heal: content-verify sources and parquet shards, re-fetching corruption.

    Re-downloads sources that fail gzip decompression (content-verified, not
    size-only) and prunes stale ones, then re-extracts parquet shards that no
    longer open cleanly. Corruption that survives both passes fails loudly
    inside the called modules rather than being silently dropped.
    """
    from sync.download import run_sync
    from sync.extract import main as extract_main

    log("=== VERIFY START: deep self-heal (content-verified sources + parquet) ===")
    run_sync(
        dest=REPO_ROOT,
        entity=args.entity,
        workers=args.workers or 8,
        dry_run=False,
        delete=True,
        verify_content=True,
    )
    extract_main(
        entity=args.entity,
        force=False,
        workers=args.workers,
        verify=True,
    )
    log("=== VERIFY END: deep self-heal complete ===")


def cmd_full(args) -> None:
    """Full pipeline: sync → extract → [verify] → commit → push."""
    cmd_sync(args)
    cmd_extract(args)
    if args.verify:
        cmd_verify(args)
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
    p_extract.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes. Default: auto-sized from RAM and CPU count.",
    )
    p_extract.add_argument(
        "--slice-index", type=int, default=None,
        help=(
            "Process only the slice_index-th of slice_total chunks "
            "(modulo source-file index). 0-based. Use together with --slice-total "
            "to split work across machines."
        ),
    )
    p_extract.add_argument(
        "--slice-total", type=int, default=None,
        help="Total number of slices for distributed processing. Use with --slice-index.",
    )
    p_extract.add_argument("--force", action="store_true")

    # commit
    p_commit = subparsers.add_parser("commit", help="Git add + commit")
    p_commit.add_argument("--message", "-m", type=str, default=None)

    # push
    subparsers.add_parser("push", help="Git push")

    # upload
    p_upload = subparsers.add_parser(
        "upload", help="Reconcile HF parquet files with the local set (upload new/changed, prune obsolete)"
    )
    p_upload.add_argument("--workers", type=int, default=8, help="Parallel upload workers (default: 8)")
    p_upload.add_argument("--repo-id", type=str, default="Mearman/OpenAlex", help="HF dataset repo ID")
    p_upload.add_argument(
        "--no-prune", action="store_true",
        help="Upload additively without deleting remote parquet files absent locally",
    )

    # verify
    p_verify = subparsers.add_parser(
        "verify", help="Deep self-heal: content-verify sources + parquet shards"
    )
    p_verify.add_argument("--entity", type=str, default=None)
    p_verify.add_argument("--workers", type=int, default=None)

    # full
    p_full = subparsers.add_parser("full", help="sync → extract → [verify] → commit → push")
    p_full.add_argument("--entity", type=str, default=None)
    p_full.add_argument("--workers", type=int, default=None)
    p_full.add_argument(
        "--verify", action="store_true",
        help="Deep self-heal between extract and commit (content-verify sources + parquet).",
    )
    # cmd_full delegates to cmd_sync and cmd_extract, which read these flags;
    # declare them here (with the same defaults as the standalone subcommands)
    # so the shared args namespace is complete.
    p_full.add_argument("--dry-run", action="store_true")
    p_full.add_argument("--no-delete", action="store_true")
    p_full.add_argument("--force", action="store_true")
    p_full.add_argument("--slice-index", type=int, default=None)
    p_full.add_argument("--slice-total", type=int, default=None)

    args = parser.parse_args()

    handlers = {
        "sync": cmd_sync,
        "extract": cmd_extract,
        "commit": cmd_commit,
        "push": cmd_push,
        "upload": cmd_upload,
        "verify": cmd_verify,
        "full": cmd_full,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
