"""Phase 1: Sync OpenAlex snapshot from S3 (anonymous, parallel, resumable).

Downloads new/changed files and removes local files not present in S3,
keeping the local snapshot identical to the remote.  OpenAlex partitions
are mutable — records move between ``updated_date`` partitions — so stale
files must be pruned to prevent incorrect entity counts.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import time
from pathlib import Path

from sync.common import format_size

log = logging.getLogger("openalex-sync")

S3_BUCKET = "openalex"
S3_PREFIX = "data/"
S3_EXCLUDE_PREFIXES = ["legacy-data/"]
DOWNLOAD_DEST = Path(
    os.environ.get(
        "OPENALEX_DOWNLOAD_DEST",
        "/scratch/SCWF00070/b.abs217/openalex-snapshot",
    )
)

# boto3 globals -- set on first call to _init_boto3()
from typing import Any

_boto3: Any = None
_botocore_UNSIGNED: Any = None
_botocore_Config: Any = None


def _init_boto3():
    """Lazy-import boto3 and botocore (only needed for download phase)."""
    global _boto3, _botocore_UNSIGNED, _botocore_Config
    if _boto3 is not None:
        return
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        _boto3 = boto3
        _botocore_UNSIGNED = UNSIGNED
        _botocore_Config = Config
    except ImportError:
        log.error("boto3 is required for the download phase: pip install boto3")
        sys.exit(1)


def _get_s3_client():
    """Create an anonymous S3 client."""
    _init_boto3()
    assert _boto3 is not None and _botocore_Config is not None and _botocore_UNSIGNED is not None
    return _boto3.client(
        "s3",
        config=_botocore_Config(
            signature_version=_botocore_UNSIGNED,
            max_pool_connections=50,
        ),
        region_name="us-east-1",
    )


def _list_s3_objects(s3, prefix: str = S3_PREFIX) -> list[dict]:
    """List all objects under prefix, excluding legacy-data/."""
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if any(key.startswith(ex) for ex in S3_EXCLUDE_PREFIXES):
                continue
            objects.append({"key": key, "size": obj["Size"]})
    return objects


def _local_path_for_key(key: str, dest_dir: Path) -> Path:
    """Map an S3 key to a local path.

    S3 stores files as ``part_XXXX.gz`` (gzip-compressed JSON Lines).
    We save them as ``part_XXXX.jsonl.gz`` so the HuggingFace dataset viewer
    detects the inner format correctly.
    """
    local_key = key
    if local_key.endswith(".gz") and not local_key.endswith(".jsonl.gz"):
        local_key = local_key[:-3] + ".jsonl.gz"
    return dest_dir / local_key


def _download_s3_file(s3, key: str, dest_dir: Path, dry_run: bool = False) -> dict:
    """Download a single file, skipping if it already exists with correct size."""
    local_path = _local_path_for_key(key, dest_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        local_size = local_path.stat().st_size
        remote_size = s3.head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"]
        if local_size == remote_size:
            return {"key": key, "status": "skipped", "size": remote_size}

    if dry_run:
        return {"key": key, "status": "dry_run", "size": 0}

    s3.download_file(S3_BUCKET, key, str(local_path))
    size = local_path.stat().st_size
    return {"key": key, "status": "downloaded", "size": size}


def _list_stale_files(dest: Path, remote_keys: set[str], prefix: str) -> list[Path]:
    """Find local .gz files under dest/prefix that are not in remote_keys.

    Only considers .gz files — parquet directories and metadata are not
    snapshot files and must not be deleted.
    """
    local_root = dest / prefix
    if not local_root.is_dir():
        return []

    stale = []
    for f in local_root.rglob("*.jsonl.gz"):
        if not f.is_file():
            continue
        # Reconstruct the original S3 key (strip .jsonl.gz → .gz)
        rel = str(f.relative_to(dest))
        s3_key = rel[:-8] + ".gz" if rel.endswith(".jsonl.gz") else rel
        if s3_key not in remote_keys:
            stale.append(f)
    # Also clean up any old-style .gz files (from before the rename)
    for f in local_root.rglob("*.gz"):
        if f.name.endswith(".jsonl.gz"):
            continue
        if not f.is_file():
            continue
        stale.append(f)
    return stale


def run_sync(
    dest: Path | None = None,
    entity: str | None = None,
    slice_index: int | None = None,
    slice_total: int | None = None,
    workers: int = 8,
    dry_run: bool = False,
    delete: bool = True,
):
    """Sync OpenAlex snapshot from S3: download new/changed files, optionally
    delete local files not present remotely.

    OpenAlex partitions are mutable — records move between ``updated_date``
    partitions — so stale files must be pruned to prevent incorrect entity
    counts.  ``delete=True`` (the default) performs this pruning automatically.
    """
    _init_boto3()
    dest = dest or DOWNLOAD_DEST

    s3 = _get_s3_client()
    prefix = S3_PREFIX
    if entity:
        prefix = f"{S3_PREFIX}{entity}/"

    log.info("Listing objects in s3://%s/%s ...", S3_BUCKET, prefix)
    objects = _list_s3_objects(s3, prefix)

    # Apply slicing for parallel job arrays
    if slice_index is not None and slice_total is not None:
        objects = [
            obj for i, obj in enumerate(objects) if i % slice_total == slice_index
        ]
        log.info(
            "Slice %d/%d: %d files assigned",
            slice_index, slice_total, len(objects),
        )

    # Sort smallest-first: small entities sync quickly while large partitions
    # fill in the background.  Without this, all workers get stuck on big files.
    objects.sort(key=lambda o: o["size"])

    total_size = sum(o["size"] for o in objects)
    log.info("Found %d files (%s) to process (smallest-first)", len(objects), format_size(total_size))

    if dry_run:
        for obj in objects[:20]:
            log.info("  %s (%s)", obj["key"], format_size(obj["size"]))
        if len(objects) > 20:
            log.info("  ... and %d more files", len(objects) - 20)

        if delete:
            remote_keys = {o["key"] for o in objects}
            stale = _list_stale_files(dest, remote_keys, prefix)
            if stale:
                log.info("Would delete %d stale files:", len(stale))
                for s in stale[:20]:
                    log.info("  %s", s.relative_to(dest))
                if len(stale) > 20:
                    log.info("  ... and %d more", len(stale) - 20)
            else:
                log.info("No stale files to delete")
        return

    dest.mkdir(parents=True, exist_ok=True)
    log.info("Syncing to %s with %d workers", dest, workers)

    # Phase 1: Download new/changed files
    start = time.time()
    downloaded = 0
    skipped = 0
    downloaded_bytes = 0
    errors = []

    def worker(obj):
        thread_s3 = _get_s3_client()
        return _download_s3_file(thread_s3, obj["key"], dest)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, obj): obj for obj in objects}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                result = future.result()
                if result["status"] == "downloaded":
                    downloaded += 1
                    downloaded_bytes += result["size"]
                elif result["status"] == "skipped":
                    skipped += 1
                if i % 100 == 0:
                    elapsed = time.time() - start
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining_files = len(objects) - i
                    eta_secs = remaining_files / rate if rate > 0 else 0
                    eta_mins = eta_secs / 60
                    log.info(
                        "Progress: %d/%d files | %d downloaded | %d skipped | %s | %.0fs elapsed | ETA %.0f min",
                        i, len(objects), downloaded, skipped,
                        format_size(downloaded_bytes), elapsed, eta_mins,
                    )
            except Exception as e:
                obj = futures[future]
                log.error("Failed to download %s: %s", obj["key"], e)
                errors.append(obj["key"])

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("Download phase completed in %.0f seconds (%.1f minutes)", elapsed, elapsed / 60)
    log.info("Downloaded: %d files (%s)", downloaded, format_size(downloaded_bytes))
    log.info("Skipped (already exist): %d files", skipped)

    # Phase 2: Delete stale local files
    if delete:
        remote_keys = {o["key"] for o in objects}
        stale = _list_stale_files(dest, remote_keys, prefix)
        if stale:
            log.info("Deleting %d stale local files ...", len(stale))
            for path in stale:
                log.info("  Deleting: %s", path.relative_to(dest))
                try:
                    path.unlink()
                except FileNotFoundError:
                    # macOS Apple Double files (._*) may vanish between
                    # listing and deletion — not a real data integrity issue.
                    pass
            # Clean up empty directories left behind
            local_root = dest / prefix
            for d in sorted(local_root.rglob("*"), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            log.info("Deleted %d stale files", len(stale))
        else:
            log.info("No stale files to delete")

    if errors:
        log.error("Failed: %d files", len(errors))
        for key in errors:
            log.error("  %s", key)
        sys.exit(1)


# Backward-compatible alias
run_download = run_sync
