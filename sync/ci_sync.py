#!/usr/bin/env python3
"""Detect and sync new OpenAlex snapshot shards from S3 to HuggingFace.

Compares S3 manifests (or S3 directory listings for manifest-less entities)
against the current HF dataset to find new shards, then downloads, renames,
extracts to parquet, and uploads each one.

Designed for CI: processes one shard at a time to keep disk bounded, with
per-shard error isolation, retries, timeouts, and structured result logging.

Supports dynamic GitHub Actions matrix splitting by entity and batch
for parallel sync.
"""
from __future__ import annotations

import os
os.environ.setdefault("TQDM_DISABLE", "1")

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

# Shards per matrix entry. Larger = fewer jobs but longer per job.
SHARDS_PER_BATCH = 200

log = logging.getLogger("openalex-sync")


# ── Entity relationship types ────────────────────────────────────────────
# Mirror of extract.py _ENTITY_DISPATCH relationship type counts.
# Used for matrix generation — relationship splitting is only applied to
# entities with multiple relationship types.

_ENTITY_REL_COUNTS: dict[str, int] = {
    "works": 18,
    "authors": 9,
    "sources": 9,
    "institutions": 10,
    "publishers": 6,
    "funders": 4,
    "concepts": 4,
    "topics": 5,
    "subfields": 4,
    "fields": 3,
    "domains": 2,
    "sdgs": 1,
    "awards": 3,
}


def _entity_rel_types(entity: str) -> list[str] | None:
    """Get relationship type names for an entity.

    Returns None if the entity doesn't support relationship extraction.
    Returns a list of relationship subdirectory names (as they appear on HF).
    """
    from sync.extract import _ENTITY_DISPATCH, nested_rt_path
    if entity not in _ENTITY_DISPATCH:
        return None
    rel_types, _ = _ENTITY_DISPATCH[entity]
    return sorted(nested_rt_path(rt) for rt in rel_types)


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
    """List .gz files on S3 for a manifest-less entity."""
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


def _parquet_path_key(path: str) -> str:
    """Derive a unique key from a parquet path including relationship type.

    data/works/abstracts/works__updated_date=2024-01-13__part_0000.parquet
      → abstracts/works__updated_date=2024-01-13__part_0000
    """
    parts = path.split("/")
    filename = parts[-1].rsplit(".", 1)[0]
    return f"{parts[-2]}/{filename}"


def _hf_entity_state(entity: str) -> tuple[set[str], set[str]] | None:
    """List all files on HF for an entity in a single pass.

    Returns (source_files, parquet_rel_keys) or None if listing fails.
    """
    from huggingface_hub import HfApi
    api = HfApi()

    try:
        source_files: set[str] = set()
        parquet_rel_keys: set[str] = set()
        for item in api.list_repo_tree(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            path_in_repo=f"data/{entity}",
            recursive=True,
        ):
            if item.path.endswith(".jsonl.gz"):
                source_files.add(item.path)
            elif item.path.endswith(".parquet"):
                parquet_rel_keys.add(_parquet_path_key(item.path))
        return source_files, parquet_rel_keys
    except Exception as e:
        log.warning("Recursive listing failed for %s: %s", entity, e)

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
        parquet_rel_keys = set()
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
                    parquet_rel_keys.add(_parquet_path_key(item.path))
        return source_files, parquet_rel_keys
    except Exception as e:
        log.warning("Per-partition listing failed for %s: %s", entity, e)
        return None


def _s3_key_to_shard_key(s3_key: str) -> str:
    """Derive a shard key from an S3 key.

    data/works/updated_date=2024-01-13/part_0000.gz
      → works__updated_date=2024-01-13__part_0000
    """
    parts = s3_key.split("/")
    entity = parts[1]
    partition = parts[2]
    stem = parts[-1].rsplit(".", 1)[0]
    return f"{entity}__{partition}__{stem}"


# ── Path conversion ──────────────────────────────────────────────────────


def _s3_key_to_hf_path(s3_key: str) -> str:
    """Convert S3 key to HF path with .jsonl.gz extension."""
    if s3_key.endswith(".gz") and not s3_key.endswith(".jsonl.gz"):
        return s3_key[:-3] + ".jsonl.gz"
    return s3_key


# ── Shard operations ────────────────────────────────────────────────────


def _download_shard(s3_key: str, dest: Path) -> None:
    """Download a single shard from S3."""
    s3 = _s3_client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(S3_BUCKET, s3_key, str(dest))


def _extract_shard(
    source_path: Path,
    entity: str,
    output_dir: Path,
) -> list[Path]:
    """Extract a single source shard to parquet.

    Returns list of parquet file paths written.
    """
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
            convert_relationships(
                entity, force=True, workers=2,
            )
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


def _prepare_shard(
    entity: str,
    s3_key: str,
    staging_dir: Path,
) -> list:
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
    parquet_files = _extract_shard(
        jsonl_gz, entity, staging_dir / "parquet",
    )

    # Build upload operations
    operations: list = []

    # Source file
    operations.append(
        CommitOperationAdd(
            path_in_repo=hf_path,
            path_or_fileobj=str(jsonl_gz),
        )
    )

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
    2. Its source file is on HF but its parquet extractions are incomplete

    Returns {entity: [s3_key, ...]} for each entity with gaps.
    """
    from huggingface_hub import HfApi
    api = HfApi()

    entities = (
        [entity_filter] if entity_filter else ENTITY_TYPES_BUILD_ORDER
    )
    new_shards: dict[str, list[str]] = {}

    for entity in entities:
        manifest = _fetch_manifest(entity)
        if not manifest:
            manifest = _list_s3_shards(entity)
        if not manifest:
            log.info("No manifest for %s", entity)
            continue

        state = _hf_entity_state(entity)
        log.info("Entity %s: manifest=%d, state=%s", entity, len(manifest),
                 "None" if state is None else f"(src={len(state[0])}, pq={len(state[1])})")

        if state is not None:
            hf_source_files, parquet_rel_keys = state

            all_rel_types = {k.split("/", 1)[0] for k in parquet_rel_keys}
            expected_count = len(all_rel_types)

            shard_rels: dict[str, set[str]] = {}
            for rk in parquet_rel_keys:
                rel, shard_key = rk.split("/", 1)
                shard_rels.setdefault(shard_key, set()).add(rel)

            missing_source = []
            has_source = []
            for s3_key in manifest:
                hf_path = _s3_key_to_hf_path(s3_key)
                if hf_path not in hf_source_files:
                    missing_source.append(s3_key)
                else:
                    has_source.append(s3_key)

            missing_parquet = []
            for s3_key in has_source:
                shard_key = _s3_key_to_shard_key(s3_key)
                present_rels = shard_rels.get(shard_key, set())
                if len(present_rels) < expected_count:
                    missing_parquet.append(s3_key)

            new_for_entity = missing_source + missing_parquet
        else:
            new_for_entity = []
            for s3_key in manifest:
                hf_path = _s3_key_to_hf_path(s3_key)
                if not api.file_exists(HF_REPO_ID, hf_path, repo_type="dataset"):
                    new_for_entity.append(s3_key)

        if new_for_entity:
            new_shards[entity] = new_for_entity

    return new_shards


# ── Matrix generation ────────────────────────────────────────────────────


def prepare_matrix(
    entity_filter: str | None = None,
    shards_per_batch: int = SHARDS_PER_BATCH,
) -> list[dict]:
    """Generate a GitHub Actions matrix from detect results.

    Each matrix entry is a (entity, batch_index) pair. The sync job handles
    ALL relationship types for its assigned batch of shards — no relationship
    splitting. This keeps the matrix small (~25 entries for a full sync)
    and avoids duplicate source file uploads.

    Returns list of matrix entries (dicts with string values).
    """
    new_shards = detect_new_shards(entity_filter=entity_filter)

    if not new_shards:
        return []

    # Write detect results for sync jobs to reuse (avoids re-running detect)
    detect_path = os.environ.get("DETECT_RESULTS_PATH", "detect_results.json")
    with open(detect_path, "w") as f:
        json.dump(new_shards, f)
    print(f"Detect results written to {detect_path}")

    matrix: list[dict] = []

    for entity, s3_keys in new_shards.items():
        n_batches = -(-len(s3_keys) // shards_per_batch)  # ceil div
        for i in range(n_batches):
            start = i * shards_per_batch
            batch = s3_keys[start:start + shards_per_batch]
            matrix.append({
                "entity": entity,
                "batch_index": str(i),
                "shard_count": str(len(batch)),
                "total_shards": str(len(s3_keys)),
                "label": f"{entity}/{i}",
            })

    return matrix


# ── Sync ─────────────────────────────────────────────────────────────────

SyncResult = dict  # {"entity": str, "s3_key": str, "status": str, ...}

COMMIT_BATCH_SIZE = 50
COMMIT_MAX_RETRIES = 3


def sync_shards(
    new_shards: dict[str, list[str]] | None = None,
    entity_filter: str | None = None,
    batch_index: int | None = None,
    shards_per_batch: int = SHARDS_PER_BATCH,
) -> None:
    """Download, extract, and upload new shards to HuggingFace.

    When batch_index is provided, processes only the specified batch of
    shards for the given entity (for matrix parallelism). Each shard gets
    its full extraction (all relationship types).

    Args:
        new_shards: Pre-computed detect results. If None, runs detect.
        entity_filter: Only process this entity.
        batch_index: Only process this batch of shards (0-indexed).
            If None, processes all shards.
        shards_per_batch: Number of shards per batch (must match matrix).
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

    # Apply batch slicing
    if batch_index is not None:
        batch_start = batch_index * shards_per_batch
        queue = queue[batch_start:batch_start + shards_per_batch]
        if not queue:
            print(f"Batch {batch_index} is empty, nothing to do")
            return

    total = len(queue)
    print(f"Syncing {total} shards")

    api = HfApi()
    results: list[SyncResult] = []
    succeeded = 0
    failed = 0
    consecutive_failures = 0
    processed = 0

    for batch_start in range(0, total, COMMIT_BATCH_SIZE):
        batch = queue[batch_start:batch_start + COMMIT_BATCH_SIZE]
        batch_ops: list = []
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

                print(f"  [{processed}/{total}] FAILED {s3_key}: {exc}")

        # Commit the batch with retries
        if batch_ops:
            committed = False
            for attempt in range(COMMIT_MAX_RETRIES):
                try:
                    api.create_commit(
                        repo_id=HF_REPO_ID,
                        repo_type="dataset",
                        operations=batch_ops,
                        commit_message=(
                            f"feat: sync {len(batch_ops)} files "
                            f"({entity}, batch {batch_index or 0})"
                        ),
                    )
                    print(f"  Committed batch of {len(batch_ops)} files")
                    committed = True
                    consecutive_failures = 0
                    break
                except Exception as exc:
                    exc_str = str(exc)
                    if "429" in exc_str or "rate limit" in exc_str.lower():
                        import re
                        match = re.search(r"Retry after (\d+) seconds", exc_str)
                        wait = int(match.group(1)) if match else 300
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
                        break

            if not committed:
                failed += len(batch_results)
                consecutive_failures += 1
                for r in batch_results:
                    if r["status"] == "ok":
                        r["status"] = "error"
                        r["error"] = "commit failed: rate limited"
                        succeeded -= 1

        results.extend(batch_results)

        for staging in batch_staging_dirs:
            shutil.rmtree(staging, ignore_errors=True)

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"Aborting: {MAX_CONSECUTIVE_FAILURES} consecutive failures")
            break

    # Write results
    results_path = Path("sync_results.jsonl")
    with open(results_path, "a") as f:
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
    detect_parser.add_argument("--entity", type=str, default=None)

    matrix_parser = sub.add_parser("prepare-matrix", help="Generate GitHub Actions matrix JSON")
    matrix_parser.add_argument("--entity", type=str, default=None)
    matrix_parser.add_argument("--shards-per-batch", type=int, default=SHARDS_PER_BATCH)

    sync_parser = sub.add_parser("sync", help="Download, extract, and upload new shards")
    sync_parser.add_argument("--entity", type=str, default=None)
    sync_parser.add_argument(
        "--batch-index", type=int, default=None,
        help="Only process this batch (0-indexed)",
    )
    sync_parser.add_argument(
        "--detect-file", type=str, default=None,
        help="Load detect results from this file instead of re-running detect",
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

    elif args.command == "prepare-matrix":
        matrix = prepare_matrix(
            entity_filter=args.entity,
            shards_per_batch=args.shards_per_batch,
        )
        matrix_json = json.dumps({"include": matrix}, separators=(',', ':'))

        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                if matrix:
                    f.write("has_new=true\n")
                    f.write(f"matrix={matrix_json}\n")
                else:
                    f.write("has_new=false\n")
                    f.write("matrix={\"include\":[]}\n")
        else:
            print(matrix_json)
        print(f"Matrix: {len(matrix)} entries")

    elif args.command == "sync":
        new_shards = None
        if args.detect_file:
            with open(args.detect_file) as f:
                new_shards = json.load(f)
        sync_shards(
            new_shards=new_shards,
            entity_filter=args.entity,
            batch_index=args.batch_index,
        )
