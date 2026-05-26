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


def _hf_source_files_for_entity(entity: str) -> set[str] | None:
    """List .jsonl.gz files on HuggingFace for a specific entity.

    Uses recursive listing first (fast for small entities). Falls back to
    shallow partition-directory listing + per-partition enumeration for
    large entities where recursive listing may timeout.

    Returns None if all listing approaches fail, signalling the caller
    should fall back to per-file existence checks.
    """
    from huggingface_hub import HfApi
    api = HfApi()

    # Try recursive listing first — fast for small entities
    try:
        result: set[str] = set()
        for item in api.list_repo_tree(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            path_in_repo=f"data/{entity}",
            recursive=True,
        ):
            if item.path.endswith(".jsonl.gz"):
                result.add(item.path)
        return result
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

        result = set()
        for part_dir in partition_dirs:
            for item in api.list_repo_tree(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                path_in_repo=part_dir,
                recursive=True,
            ):
                if item.path.endswith(".jsonl.gz"):
                    result.add(item.path)
        return result
    except Exception:
        return None


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


def _process_shard(entity: str, s3_key: str) -> int:
    """Download, extract, and upload a single shard. Returns file count uploaded."""
    from huggingface_hub import HfApi, CommitOperationAdd

    hf_path = _s3_key_to_hf_path(s3_key)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Download
        local_gz = tmp_dir / Path(s3_key).name
        _download_shard(s3_key, local_gz)

        # Rename to .jsonl.gz
        jsonl_gz = tmp_dir / (local_gz.stem + ".jsonl.gz")
        shutil.move(str(local_gz), str(jsonl_gz))

        # Extract to parquet
        parquet_files = _extract_shard(jsonl_gz, entity, tmp_dir / "parquet")

        # Build upload operations
        operations = [
            CommitOperationAdd(
                path_in_repo=hf_path,
                path_or_fileobj=str(jsonl_gz),
            )
        ]
        for pq in parquet_files:
            pq_rel = pq.relative_to(tmp_dir / "parquet")
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"data/{pq_rel}",
                    path_or_fileobj=str(pq),
                )
            )

        # Upload
        api = HfApi()
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message=f"feat: sync {hf_path}",
        )

        return len(operations)


# ── Detection ────────────────────────────────────────────────────────────


def detect_new_shards(entity_filter: str | None = None) -> dict[str, list[str]]:
    """Compare S3 manifests against HF to find new shards.

    For entities without manifests (e.g. awards), falls back to S3
    directory listing.

    Args:
        entity_filter: If set, only detect for this entity. Otherwise all.

    Returns {entity: [s3_key, ...]} for each entity with new files.
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

        hf_files = _hf_source_files_for_entity(entity)

        if hf_files is not None:
            new_for_entity = []
            for s3_key in manifest:
                hf_path = _s3_key_to_hf_path(s3_key)
                if hf_path not in hf_files:
                    new_for_entity.append(s3_key)
        else:
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


def sync_shards(
    new_shards: dict[str, list[str]] | None = None,
    entity_filter: str | None = None,
) -> None:
    """Download, extract, and upload new shards to HuggingFace.

    Processes one shard at a time to keep disk usage bounded.
    Isolates failures per shard — a bad shard does not stop the rest.
    Retries failed shards up to 3 times with exponential backoff.
    Aborts after MAX_CONSECUTIVE_FAILURES consecutive unrecoverable failures.

    Writes a sync_results.jsonl file with per-shard outcomes.
    """
    if new_shards is None:
        new_shards = detect_new_shards(entity_filter=entity_filter)

    if not new_shards:
        print("No new shards to sync")
        return

    total = sum(len(v) for v in new_shards.values())
    print(f"Syncing {total} new shards across {len(new_shards)} entities")

    results: list[SyncResult] = []
    processed = 0
    succeeded = 0
    failed = 0
    consecutive_failures = 0

    for entity, s3_keys in new_shards.items():
        for s3_key in s3_keys:
            processed += 1
            result: SyncResult = {
                "entity": entity,
                "s3_key": s3_key,
                "status": "pending",
                "files": 0,
                "error": None,
                "seconds": 0.0,
            }

            # Retry up to 3 times with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                t0 = time.monotonic()
                try:
                    file_count = _process_shard(entity, s3_key)
                    elapsed = time.monotonic() - t0

                    result["status"] = "ok"
                    result["files"] = file_count
                    result["seconds"] = round(elapsed, 1)
                    succeeded += 1
                    consecutive_failures = 0

                    print(
                        f"  [{processed}/{total}] OK {s3_key} "
                        f"({file_count} files, {elapsed:.0f}s)"
                    )
                    break

                except Exception as exc:
                    elapsed = time.monotonic() - t0
                    result["error"] = str(exc)
                    result["seconds"] = round(elapsed, 1)

                    if attempt < max_retries - 1:
                        wait = 2 ** attempt  # 1s, 2s
                        print(
                            f"  [{processed}/{total}] RETRY {s3_key} "
                            f"(attempt {attempt + 1}/{max_retries}, "
                            f"waiting {wait}s): {exc}"
                        )
                        time.sleep(wait)
                    else:
                        result["status"] = "error"
                        failed += 1
                        consecutive_failures += 1
                        print(
                            f"  [{processed}/{total}] FAILED {s3_key} "
                            f"after {max_retries} attempts: {exc}"
                        )

            results.append(result)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"Aborting: {MAX_CONSECUTIVE_FAILURES} consecutive "
                    f"unrecoverable failures"
                )
                break

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            break

    # Write results file for CI artifact
    results_path = Path("sync_results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Done: {succeeded} succeeded, {failed} failed out of {total}")
    if failed > 0:
        # Non-zero exit so CI marks the run as failed
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
