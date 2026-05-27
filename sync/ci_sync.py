#!/usr/bin/env python3
"""Detect and sync new OpenAlex snapshot shards from S3 to HuggingFace.

Compares S3 manifests (or S3 directory listings for manifest-less entities)
against the current HF dataset to find new shards, then downloads, renames,
extracts to parquet, and uploads each one.

Designed for CI: processes one shard at a time to keep disk bounded, with
per-shard error isolation, retries, timeouts, and structured result logging.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from sync.common import ENTITY_TYPES_BUILD_ORDER

S3_BUCKET = "openalex"
HF_REPO_ID = "Mearman/OpenAlex"

# Per-shard timeout in seconds. Works shards can be large (~700MB compressed)
# so extraction may take a while. 10 minutes is generous.
SHARD_TIMEOUT = 600

# Maximum consecutive failures before aborting the sync run.
MAX_CONSECUTIVE_FAILURES = 5

log = logging.getLogger("openalex-sync")


# ── S3 helpers ───────────────────────────────────────────────────────────


def _s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _fetch_manifest(entity: str) -> dict[str, dict]:
    """Fetch S3 manifest for an entity. Returns {s3_key: meta}."""
    s3 = _s3_client()
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"data/{entity}/manifest")
        manifest = json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        return {}
    entries = {}
    for entry in manifest.get("entries", []):
        url: str = entry.get("url", "")
        if url.startswith(f"s3://{S3_BUCKET}/"):
            key = url[len(f"s3://{S3_BUCKET}/"):]
            entries[key] = entry.get("meta", {})
    return entries


def _list_s3_shards(entity: str) -> dict[str, dict]:
    """List .gz files on S3 for a manifest-less entity.

    Used as a fallback for entities without manifests (e.g. awards).
    Returns {s3_key: {}} with empty meta since we have no metadata.
    """
    s3 = _s3_client()
    prefix = f"data/{entity}/"
    entries: dict[str, dict] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            if key.endswith(".gz") and not key.endswith(".jsonl.gz"):
                entries[key] = {}
    return entries


# ── HF helpers ───────────────────────────────────────────────────────────


def _hf_entity_state(entity: str) -> tuple[set[str], set[str]] | None:
    """List all files on HF for an entity in a single pass.

    Returns (source_files, parquet_shard_keys) or None if listing fails.
    - source_files: set of .jsonl.gz paths
    - parquet_shard_keys: set of shard keys derived from .parquet filenames
    """
    from huggingface_hub import HfApi
    api = HfApi()

    try:
        source_files: set[str] = set()
        parquet_keys: set[str] = set()
        for item in api.list_repo_tree(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            path_in_repo=f"data/{entity}",
            recursive=True,
        ):
            if item.path.endswith(".jsonl.gz"):
                source_files.add(item.path)
            elif item.path.endswith(".parquet"):
                filename = item.path.split("/")[-1]
                key = filename.rsplit(".", 1)[0]
                parquet_keys.add(key)
        return source_files, parquet_keys
    except Exception:
        pass

    # Fallback: shallow + per-partition approach for large entities.
    try:
        partition_dirs: set[str] = set()
        for item in api.list_repo_tree(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            path_in_repo=f"data/{entity}",
            recursive=False,
        ):
            if item.path.startswith(f"data/{entity}/updated_date="):
                partition_dirs.add(item.path)

        source_files = set()
        parquet_keys = set()
        for part_dir in partition_dirs:
            for item in api.list_repo_tree(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                path_in_repo=part_dir,
                recursive=True,
            ):
                if item.path.endswith(".jsonl.gz"):
                    source_files.add(item.path)
                elif item.path.endswith(".parquet"):
                    filename = item.path.split("/")[-1]
                    key = filename.rsplit(".", 1)[0]
                    parquet_keys.add(key)
        return source_files, parquet_keys
    except Exception:
        return None


def _s3_key_to_shard_key(s3_key: str) -> str:
    """Derive a shard key from an S3 key.

    s3 key:   data/works/updated_date=2024-01-13/part_0000.gz
    shard key: works__updated_date=2024-01-13__part_0000
    """
    parts = s3_key.split("/")
    entity = parts[1]  # e.g. "works"
    partition = parts[2]  # e.g. "updated_date=2024-01-13"
    filename = parts[-1]  # e.g. "part_0000.gz"
    stem = filename.rsplit(".", 1)[0]  # e.g. "part_0000"
    return f"{entity}__{partition}__{stem}"


# ── Path conversion ──────────────────────────────────────────────────────


def _s3_key_to_hf_path(s3_key: str) -> str:
    """Convert S3 key to HF path with .jsonl.gz extension.

    s3 key:   data/works/updated_date=2024-01-13/part_0000.gz
    hf path:  data/works/updated_date=2024-01-13/part_0000.jsonl.gz
    """
    if s3_key.endswith(".gz") and not s3_key.endswith(".jsonl.gz"):
        return s3_key[:-3] + ".jsonl.gz"
    return s3_key


# ── Shard operations ────────────────────────────────────────────────────


def _download_shard(s3_key: str, dest: Path) -> None:
    """Download a single shard from S3."""
    s3 = _s3_client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(S3_BUCKET, s3_key, str(dest))


def _extract_shard(source_path: Path, entity: str, output_dir: Path) -> list[Path]:
    """Extract a single source shard to parquet. Returns list of parquet files."""
    from sync.extract import convert_relationships

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        mock_data = tmp_dir / "data" / entity
        mock_data.mkdir(parents=True)

        partition_dir = mock_data / source_path.parent.name
        partition_dir.mkdir(exist_ok=True)
        target = partition_dir / source_path.name

        shutil.copy2(source_path, target)

        import sync.common as common
        original_snapshot = common.SNAPSHOT_DIR
        common.SNAPSHOT_DIR = tmp_dir / "data"

        try:
            convert_relationships(entity, force=True, workers=1)
        finally:
            common.SNAPSHOT_DIR = original_snapshot

        parquet_files = list((tmp_dir / "data").rglob("*.parquet"))
        result = []
        for pq in parquet_files:
            dest = output_dir / pq.relative_to(tmp_dir / "data")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pq), str(dest))
            result.append(dest)

        return result


def _prepare_shard(entity: str, s3_key: str, staging_dir: Path) -> list:
    """Download and extract a shard into staging. Returns CommitOperationAdd list."""
    from huggingface_hub import CommitOperationAdd

    hf_path = _s3_key_to_hf_path(s3_key)

    # Download
    local_gz = staging_dir / Path(s3_key).name
    _download_shard(s3_key, local_gz)

    # Rename to .jsonl.gz
    jsonl_gz = staging_dir / (local_gz.stem + ".jsonl.gz")
    shutil.move(str(local_gz), str(jsonl_gz))

    # Extract to parquet
    parquet_files = _extract_shard(jsonl_gz, entity, staging_dir / "parquet")

    # Build upload operations — files must exist on disk until the commit is made
    operations = [
        CommitOperationAdd(
            path_in_repo=hf_path,
            path_or_fileobj=str(jsonl_gz),
        )
    ]
    for pq in parquet_files:
        pq_rel = pq.relative_to(staging_dir / "parquet")
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"data/{pq_rel}",
                path_or_fileobj=str(pq),
            )
        )

    return operations


# ── Detection ────────────────────────────────────────────────────────────


def detect_new_shards(entity_filter: str | None = None) -> dict[str, list[str]]:
    """Compare S3 manifests against HF to find shards needing sync.

    A shard needs syncing if either:
    1. Its source .jsonl.gz file is not on HF, OR
    2. Its source file is on HF but its parquet extractions are missing

    For entities without manifests (e.g. awards), falls back to S3
    directory listing.

    Args:
        entity_filter: If set, only detect for this entity. Otherwise all.

    Returns {entity: [s3_key, ...]} for each entity with gaps.
    """
    from huggingface_hub import HfApi
    api = HfApi()

    entities = (
        [entity_filter] if entity_filter else ENTITY_TYPES_BUILD_ORDER
    )
    new_shards: dict[str, list[str]] = {}

    for entity in entities:
        # Try manifest first, fall back to S3 directory listing
        manifest = _fetch_manifest(entity)
        if not manifest:
            manifest = _list_s3_shards(entity)
        if not manifest:
            continue

        state = _hf_entity_state(entity)

        if state is not None:
            hf_source_files, parquet_keys = state

            # First pass: find missing source files
            missing_source = []
            has_source = []
            for s3_key in manifest:
                hf_path = _s3_key_to_hf_path(s3_key)
                if hf_path not in hf_source_files:
                    missing_source.append(s3_key)
                else:
                    has_source.append(s3_key)

            # Second pass: check if existing source files have parquet extractions
            missing_parquet = []
            for s3_key in has_source:
                shard_key = _s3_key_to_shard_key(s3_key)
                if shard_key not in parquet_keys:
                    missing_parquet.append(s3_key)

            new_for_entity = missing_source + missing_parquet
        else:
            # Tree listing failed — check each manifest entry individually.
            # Only check for missing source files; can't verify parquet
            # completeness without a tree listing.
            new_for_entity = []
            for s3_key in manifest:
                hf_path = _s3_key_to_hf_path(s3_key)
                if not api.file_exists(HF_REPO_ID, hf_path, repo_type="dataset"):
                    new_for_entity.append(s3_key)

        if new_for_entity:
            new_shards[entity] = new_for_entity

    return new_shards


# ── Sync ─────────────────────────────────────────────────────────────────

# SyncResult is written as JSONL at the end of a run for CI artifact upload.
SyncResult = dict  # {"entity": str, "s3_key": str, "status": str, "files": int, "error": str|None, "seconds": float}


# HF free-tier rate limit: 128 commits per hour. Stay well under with batches.
COMMIT_BATCH_SIZE = 50
# Maximum retries for a batch commit (handles 429 rate limits).
COMMIT_MAX_RETRIES = 3


def sync_shards(
    new_shards: dict[str, list[str]] | None = None,
    entity_filter: str | None = None,
) -> None:
    """Download, extract, and upload new shards to HuggingFace.

    Batches multiple shards into a single commit to stay under the HF
    rate limit (128 commits/hour on the free tier). Within each batch,
    shards are downloaded and extracted one at a time to keep disk bounded.
    All files from the batch are committed together, then the staging
    directory is cleaned up.

    Isolates failures per shard — a bad shard is skipped, not retried
    within the batch. Failed shards are logged for the next run.
    Aborts after MAX_CONSECUTIVE_FAILURES consecutive unrecoverable failures.

    Writes a sync_results.jsonl file with per-shard outcomes.
    """
    from huggingface_hub import HfApi

    if new_shards is None:
        new_shards = detect_new_shards(entity_filter=entity_filter)

    if not new_shards:
        print("No new shards to sync")
        return

    # Flatten to ordered list of (entity, s3_key) pairs
    queue: list[tuple[str, str]] = []
    for entity in ENTITY_TYPES_BUILD_ORDER:
        if entity in new_shards:
            for s3_key in new_shards[entity]:
                queue.append((entity, s3_key))

    total = len(queue)
    print(f"Syncing {total} new shards in batches of {COMMIT_BATCH_SIZE}")

    api = HfApi()
    results: list[SyncResult] = []
    succeeded = 0
    failed = 0
    consecutive_failures = 0
    processed = 0

    for batch_start in range(0, total, COMMIT_BATCH_SIZE):
        batch = queue[batch_start:batch_start + COMMIT_BATCH_SIZE]
        batch_ops: list = []  # Accumulated CommitOperationAdd
        batch_staging_dirs: list[Path] = []
        batch_results: list[SyncResult] = []

        for entity, s3_key in batch:
            processed += 1
            result: SyncResult = {
                "entity": entity,
                "s3_key": s3_key,
                "status": "pending",
                "files": 0,
                "error": None,
                "seconds": 0.0,
            }

            t0 = time.monotonic()
            try:
                # Each shard gets its own temp dir — cleaned up after commit
                staging = Path(tempfile.mkdtemp(prefix="openalex-sync-"))
                ops = _prepare_shard(entity, s3_key, staging)

                elapsed = time.monotonic() - t0
                result["status"] = "ok"
                result["files"] = len(ops)
                result["seconds"] = round(elapsed, 1)
                succeeded += 1
                consecutive_failures = 0

                batch_ops.extend(ops)
                batch_staging_dirs.append(staging)
                batch_results.append(result)

                print(
                    f"  [{processed}/{total}] prepared {s3_key} "
                    f"({len(ops)} files, {elapsed:.0f}s)"
                )

            except Exception as exc:
                elapsed = time.monotonic() - t0
                result["status"] = "error"
                result["error"] = str(exc)
                result["seconds"] = round(elapsed, 1)
                failed += 1
                consecutive_failures += 1
                batch_results.append(result)

                print(
                    f"  [{processed}/{total}] FAILED {s3_key}: {exc}"
                )

        # Commit the batch with retries for rate limiting
        if batch_ops:
            committed = False
            for attempt in range(COMMIT_MAX_RETRIES):
                try:
                    api.create_commit(
                        repo_id=HF_REPO_ID,
                        repo_type="dataset",
                        operations=batch_ops,
                        commit_message=f"feat: sync {len(batch_ops)} files from {len(batch)} shards",
                    )
                    print(f"  Committed batch of {len(batch_ops)} files")
                    committed = True
                    consecutive_failures = 0
                    break
                except Exception as exc:
                    exc_str = str(exc)
                    if "429" in exc_str or "rate limit" in exc_str.lower():
                        # Parse retry-after from the error message
                        import re
                        match = re.search(r"Retry after (\d+) seconds", exc_str)
                        wait = int(match.group(1)) if match else 300
                        # Cap the wait at 5 minutes for safety
                        wait = min(wait, 300)
                        if attempt < COMMIT_MAX_RETRIES - 1:
                            print(
                                f"  Rate limited, waiting {wait}s "
                                f"(attempt {attempt + 1}/{COMMIT_MAX_RETRIES})"
                            )
                            time.sleep(wait)
                        else:
                            print(f"  Batch commit FAILED after {COMMIT_MAX_RETRIES} retries: {exc}")
                    else:
                        print(f"  Batch commit FAILED: {exc}")
                        break  # Non-rate-limit errors are not retried

            if not committed:
                # Commit failure marks all shards in batch as failed
                failed += len(batch_results)  # Count each shard as failed
                consecutive_failures += 1
                for r in batch_results:
                    if r["status"] == "ok":
                        r["status"] = "error"
                        r["error"] = "commit failed: rate limited"
                        succeeded -= 1

        results.extend(batch_results)

        # Clean up staging dirs — files are now on the server
        for staging in batch_staging_dirs:
            shutil.rmtree(staging, ignore_errors=True)

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(
                f"Aborting: {MAX_CONSECUTIVE_FAILURES} consecutive "
                f"unrecoverable failures"
            )
            break

        # Rate limit margin: with batch_size=50 and ~30 shards/min,
        # we do ~1 commit every 2min → 30/hr, well under the 128/hr limit.

    # Write results file for CI artifact
    results_path = Path("sync_results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Done: {succeeded} succeeded, {failed} failed out of {total}")
    if failed > 0:
        sys.exit(1)


# ── CLI ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync new OpenAlex shards to HuggingFace"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    detect_parser = sub.add_parser("detect", help="Show new shards without syncing")
    detect_parser.add_argument(
        "--entity", type=str, default=None,
        help="Only detect for this entity (default: all)",
    )

    sync_parser = sub.add_parser("sync", help="Download, extract, and upload new shards")
    sync_parser.add_argument(
        "--entity", type=str, default=None,
        help="Only sync this entity (default: all)",
    )

    args = parser.parse_args()

    if args.command == "detect":
        new = detect_new_shards(entity_filter=args.entity)
        if not new:
            print("No new shards")
        else:
            total = sum(len(v) for v in new.values())
            print(f"{total} new shards:")
            for entity, keys in new.items():
                print(f"  {entity}: {len(keys)} new")
                for k in keys[:5]:
                    print(f"    {k}")
                if len(keys) > 5:
                    print(f"    ... and {len(keys) - 5} more")

    elif args.command == "sync":
        sync_shards(entity_filter=args.entity)
