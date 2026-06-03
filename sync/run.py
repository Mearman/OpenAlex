#!/usr/bin/env python3
"""Sync the OpenAlex snapshot — one idempotent command, no subcommands.

    python -m sync [--entity E] [--workers N] [--verify] [--force]
                   [--no-upload] [--no-prune] [--dry-run] [--no-delete]
                   [--repo-id REPO] [-m MSG] [--slice-index I --slice-total T]

Runs the full pipeline end to end: download sources from S3 → extract Parquet
tables → commit → push → reconcile the HuggingFace dataset (upload new/changed,
prune obsolete). Every stage is idempotent, so re-running converges the local
tree, the git remote, and the HF dataset to the canonical state and resumes
where it left off. Runs from the repo root (parent of openalex-snapshot/).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
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
# Seconds between background upload sweeps while extraction is still running.
_UPLOAD_SWEEP_SECONDS = 300


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


def cmd_upload(args, *, workers: int | None = None) -> None:
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
    ``workers`` overrides the parallel-upload count (e.g. the resource governor
    grants the final pass the whole machine once extraction has finished).
    """
    _hf_upload_pass(args.repo_id, workers or args.workers or 8)
    if not args.no_prune:
        _hf_prune(args.repo_id)
    # Bring local git refs in line with what the server now has
    _sync_git_refs()
    log("ALL DONE")


def _hf_upload_pass(repo_id: str, num_workers: int) -> None:
    """Upload new and changed parquet files (additive). ``upload_large_folder``
    hashes each file and uploads only those the server lacks or whose content
    differs, so re-running is cheap and resumable — safe to call repeatedly
    while extraction is still producing shards."""
    from huggingface_hub import upload_large_folder

    upload_large_folder(
        repo_id=repo_id,
        folder_path=str(SYNC_ROOT),
        repo_type="dataset",
        allow_patterns=["*.parquet"],
        ignore_patterns=["._*"],  # Apple Double files
        num_workers=num_workers,
    )


def _hf_prune(repo_id: str) -> None:
    """Delete remote parquet files absent from the local set so HF mirrors
    local. Destructive — run only once the local extraction is complete."""
    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi()
    local = {
        p.relative_to(SYNC_ROOT).as_posix()
        for p in SYNC_ROOT.rglob("*.parquet")
        if not p.name.startswith("._")
    }
    remote = [
        f for f in api.list_repo_files(repo_id, repo_type="dataset")
        if f.endswith(".parquet")
    ]
    obsolete = sorted(set(remote) - local)
    if not obsolete:
        log("No obsolete remote parquet files to prune")
        return
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


def _background_uploader(repo_id: str, num_workers: int, stop: threading.Event) -> None:
    """Repeatedly upload completed parquet shards (additive only) until
    signalled, so the HF upload overlaps the long extraction instead of running
    as a serial tail. Pruning and git-ref sync are deliberately excluded — they
    run once at the end against the final local set. A failed sweep (e.g. an HF
    rate limit) is logged and retried on the next pass; it never aborts the run.
    """
    while not stop.is_set():
        try:
            _hf_upload_pass(repo_id, num_workers)
        except Exception as exc:  # noqa: BLE001 — a sweep failure must not kill extraction
            log(f"background upload sweep failed (will retry): {exc}")
        stop.wait(_UPLOAD_SWEEP_SECONDS)


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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    """Single entrypoint: download from S3 → extract Parquet → commit → push →
    reconcile HuggingFace. Every stage is idempotent, so ``python -m sync``
    converges the local tree, the git remote, and the HF dataset to the
    canonical state; re-running is safe and resumes where it left off.
    """
    parser = argparse.ArgumentParser(
        prog="sync",
        description=(
            "Sync the OpenAlex snapshot: download sources from S3, extract "
            "Parquet tables, commit/push, and reconcile the HuggingFace dataset. "
            "Idempotent — re-run to converge."
        ),
    )
    parser.add_argument("--entity", type=str, default=None, help="Limit to one entity (default: all)")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Extract worker processes (default: auto-sized from RAM/CPU). Download and upload use 8 when unset.",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract all units regardless of provenance")
    parser.add_argument(
        "--verify", action="store_true",
        help="Deep self-heal: content-verify sources and Parquet shards (re-fetch/re-extract corruption) before commit",
    )
    parser.add_argument("--dry-run", action="store_true", help="Download: report actions without fetching")
    parser.add_argument("--no-delete", action="store_true", help="Download: keep local files absent from S3")
    parser.add_argument("--no-prune", action="store_true", help="Upload: do not delete remote Parquet files absent locally")
    parser.add_argument("--no-upload", action="store_true", help="Skip the HuggingFace upload/reconcile stage")
    parser.add_argument("--repo-id", type=str, default="Mearman/OpenAlex", help="HuggingFace dataset repo ID")
    parser.add_argument("--message", "-m", type=str, default=None, help="Commit message")
    parser.add_argument(
        "--slice-index", type=int, default=None,
        help="Process the slice_index-th of slice_total chunks (0-based); with --slice-total splits work across machines",
    )
    parser.add_argument(
        "--slice-total", type=int, default=None,
        help="Total slices for distributed processing; use with --slice-index",
    )
    args = parser.parse_args()

    _validate_slice(args.slice_index, args.slice_total)

    # ── Resource governor ────────────────────────────────────────────────
    # Detect CPU/RAM once and split worker budgets across the concurrent
    # stages so they never oversubscribe the machine. Extraction is CPU+RAM
    # bound; the overlapping upload is network-bound and needs only a few
    # cores, so it gets a small slice during overlap and the whole machine for
    # the final solo pass. An explicit --workers overrides the extraction count
    # (the upload slice then fills whatever cores that leaves).
    from sync.extract import _auto_workers

    cpu = os.cpu_count() or 4
    if args.no_upload:
        upload_overlap = 0
        upload_solo = 0
    else:
        upload_overlap = max(2, cpu // 4)  # network-bound: a few cores suffice
        upload_solo = cpu                  # whole machine once extraction ends
    if args.workers:
        extract_workers = args.workers
        if not args.no_upload:
            upload_overlap = max(2, cpu - args.workers)
    else:
        extract_workers = _auto_workers(reserve=upload_overlap)
    args.workers = extract_workers  # extraction reads args.workers
    log(
        f"resource plan: {cpu} CPUs -> extract={extract_workers}, "
        f"upload(overlap)={upload_overlap}, upload(final)={upload_solo}"
    )

    log("=== download (S3 → sources) ===")
    cmd_sync(args)

    # Overlap the HF upload with extraction: a background thread uploads
    # completed shards (additive only) while extraction runs, so the upload
    # hides under the long extract instead of being a serial tail. Pruning and
    # git-ref sync run once at the end, against the final local set.
    stop_upload: threading.Event | None = None
    uploader: threading.Thread | None = None
    if not args.no_upload:
        stop_upload = threading.Event()
        uploader = threading.Thread(
            target=_background_uploader,
            args=(args.repo_id, upload_overlap, stop_upload),
            name="hf-uploader",
            daemon=True,
        )
        uploader.start()
        log(f"=== background HF upload started ({upload_overlap} workers, overlapping extraction) ===")

    log("=== extract (sources → Parquet) ===")
    cmd_extract(args)

    # Stop overlapping uploads before the git stages and the final reconcile so
    # nothing races on git refs or runs two upload_large_folder passes at once.
    if uploader is not None and stop_upload is not None:
        stop_upload.set()
        uploader.join()

    if args.verify:
        log("=== verify (deep self-heal) ===")
        cmd_verify(args)
    log("=== commit ===")
    cmd_commit(args)
    log("=== push ===")
    cmd_push(args)
    if not args.no_upload:
        log(f"=== upload (final reconcile: catch-up + prune, {upload_solo} workers) ===")
        cmd_upload(args, workers=upload_solo)
    log("=== sync complete ===")


if __name__ == "__main__":
    main()
