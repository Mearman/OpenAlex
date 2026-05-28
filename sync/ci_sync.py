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

# Maximum number of shards per sync job.
# Since per-shard upload keeps disk bounded, this controls parallelism and
# matrix size, not disk usage. Higher = fewer jobs but longer per job.
# Works (18 rel types) at 22 shards/job = 86 jobs for a full sync — too many.
# At 100 shards/job = 19 works jobs, much more reasonable.
MAX_SHARDS_PER_JOB = 100

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


def _shards_per_job_for_entity(entity: str, total_shards: int) -> int:
    """Compute batch size weighted by relationship type count.

    Works has 18 relationship types, producing more parquets per shard
    than smaller entities. This ensures jobs have roughly equal output
    volume. Since per-shard upload keeps disk bounded, the batch size
    primarily controls parallelism and matrix size.
    """
    rel_count = _ENTITY_REL_COUNTS.get(entity, 1)
    # Scale inversely with rel count: works gets fewer shards per job
    # to keep individual job duration reasonable.
    base = MAX_SHARDS_PER_JOB
    scaled = max(1, base * 10 // (rel_count + 9))  # works: ~35, authors: ~52, sdgs: 100
    return min(scaled, total_shards)


def _size_aware_batch(
    keys: list[str],
    sizes: dict[str, int],
    max_batches: int,
    rel_count: int = 1,
) -> list[list[str]]:
    """Split keys into batches balanced by estimated processing cost.

    The dominant cost is the per-shard extract+upload cycle (each shard
    triggers one upload_large_folder call). Byte size is a secondary factor
    for extraction time. The cost model weights shard count heavily and
    uses byte size as a tiebreaker to avoid concentrating all large
    partitions in one batch.

    Uses a greedy largest-fit algorithm: sorts shards by size descending,
    then assigns each to the batch with the smallest current cost.

    Falls back to positional slicing when sizes are unavailable.
    """
    if not keys:
        return []

    has_sizes = any(sizes.get(k, 0) > 0 for k in keys)

    if not has_sizes:
        # No size data — fall back to positional slicing
        n = -(-len(keys) // max(1, -(-len(keys) // max_batches)))
        return [keys[i:i + n] for i in range(0, len(keys), n)]

    # Sort by size descending (greedy: assign largest first)
    sorted_keys = sorted(keys, key=lambda k: sizes.get(k, 0), reverse=True)

    batches: list[list[str]] = [[] for _ in range(max_batches)]
    batch_counts: list[int] = [0] * max_batches  # primary: shard count
    batch_bytes: list[int] = [0] * max_batches  # secondary: total bytes

    for key in sorted_keys:
        size = sizes.get(key, 0)
        # Find the batch with the smallest count; break ties by bytes
        min_count = min(batch_counts)
        candidates = [i for i, c in enumerate(batch_counts) if c == min_count]
        min_idx = min(candidates, key=lambda i: batch_bytes[i])
        batches[min_idx].append(key)
        batch_counts[min_idx] += 1
        batch_bytes[min_idx] += size

    return batches


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
    """Fetch S3 manifest for an entity. Returns {s3_key: {"size": int}}."""
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
            meta = entry.get("meta", {})
            record: dict = {}
            # OpenAlex manifests include record_count in meta
            if isinstance(meta, dict):
                count = meta.get("record_count") or meta.get("content_length")
                if count is not None:
                    record["size"] = int(count)
            entries[key] = record
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
                entries[key] = {"size": obj.get("Size", 0)}
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


def _hf_entity_state(entity: str, cache_dir: Path | None = None) -> tuple[set[str], set[str]] | None:
    """List all files on HF for an entity in a single pass.

    Returns (source_files, parquet_rel_keys) or None if listing fails.
    When cache_dir is provided, caches the listing as JSON to avoid
    re-listing on subsequent calls within the same run or across runs.
    """
    from huggingface_hub import HfApi
    api = HfApi()

    source_files: set[str] = set()
    parquet_rel_keys: set[str] = set()

    # Check cache first
    if cache_dir is not None:
        cache_file = cache_dir / f"hf_state_{entity}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                source_files = set(cached.get("source_files", []))
                parquet_rel_keys = set(cached.get("parquet_rel_keys", []))
                log.info("Entity %s: loaded from cache (%d src, %d pq)",
                         entity, len(source_files), len(parquet_rel_keys))
                return source_files, parquet_rel_keys
            except (json.JSONDecodeError, KeyError):
                log.warning("Cache corrupt for %s, re-fetching", entity)

    def _cache_result(src: set[str], pq: set[str]) -> tuple[set[str], set[str]]:
        """Write result to cache if cache_dir is set."""
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"hf_state_{entity}.json"
            with open(cache_file, "w") as f:
                json.dump({
                    "source_files": sorted(src),
                    "parquet_rel_keys": sorted(pq),
                }, f)
            log.info("Entity %s: cached listing (%d src, %d pq)",
                     entity, len(src), len(pq))
        return src, pq

    try:
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
        return _cache_result(source_files, parquet_rel_keys)
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

        source_files.clear()
        parquet_rel_keys.clear()
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
        return _cache_result(source_files, parquet_rel_keys)
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

        # Symlink instead of copy — the file is read-only for extraction
        target.symlink_to(source_path.resolve())

        import sync.common as common
        from sync.extract import convert_relationships
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



# ── Detection ────────────────────────────────────────────────────────────


def detect_new_shards(
    entity_filter: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Compare S3 manifests against HF to find shards needing sync.

    A shard needs syncing if either:
    1. Its source .jsonl.gz file is not on HF, OR
    2. Its source file is on HF but its parquet extractions are incomplete

    Returns (new_shards, shard_sizes) where:
      new_shards: {entity: [s3_key, ...]} for each entity with gaps
      shard_sizes: {s3_key: size_in_bytes} from manifest or S3 listing
    """
    from huggingface_hub import HfApi
    api = HfApi()

    entities = (
        [entity_filter] if entity_filter else ENTITY_TYPES_BUILD_ORDER
    )
    new_shards: dict[str, list[str]] = {}
    shard_sizes: dict[str, int] = {}

    for entity in entities:
        manifest = _fetch_manifest(entity)
        if not manifest:
            manifest = _list_s3_shards(entity)
        if not manifest:
            log.info("No manifest for %s", entity)
            continue

        # Extract sizes from manifest metadata
        for s3_key, meta in manifest.items():
            if isinstance(meta, dict) and "size" in meta:
                shard_sizes[s3_key] = meta["size"]

        state = _hf_entity_state(entity, cache_dir=cache_dir)
        log.info("Entity %s: manifest=%d, state=%s", entity, len(manifest),
                 "None" if state is None else f"(src={len(state[0])}, pq={len(state[1])})")

        if state is not None:
            hf_source_files, parquet_rel_keys = state

            all_rel_types = {k.split("/", 1)[0] for k in parquet_rel_keys}
            # Use the known relationship type count for this entity, not
            # what's currently on HF. Deriving from HF means a shard
            # with only 10 of 18 rel types looks "complete" if those
            # 10 are the only rel types on HF — a false negative.
            expected_count = _ENTITY_REL_COUNTS.get(entity, len(all_rel_types))

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
                    log.debug(
                        "Shard %s: %d/%d rel types present",
                        shard_key, len(present_rels), expected_count,
                    )
                    missing_parquet.append(s3_key)

            log.info(
                "Entity %s: %d missing source, %d incomplete parquet "
                "(expected %d rel types, found %d on HF)",
                entity, len(missing_source), len(missing_parquet),
                expected_count, len(all_rel_types),
            )

            new_for_entity = missing_source + missing_parquet
        else:
            new_for_entity = []
            for s3_key in manifest:
                hf_path = _s3_key_to_hf_path(s3_key)
                if not api.file_exists(HF_REPO_ID, hf_path, repo_type="dataset"):
                    new_for_entity.append(s3_key)

        if new_for_entity:
            new_shards[entity] = new_for_entity

    return new_shards, shard_sizes


# ── Matrix generation ────────────────────────────────────────────────────


def prepare_matrix(
    entity_filter: str | None = None,
    shards_per_batch: int | None = None,
    cache_dir: Path | None = None,
) -> list[dict]:
    """Generate a GitHub Actions matrix from detect results.

    Each matrix entry is a (entity, batch_index) pair. The sync job handles
    ALL relationship types for its assigned batch of shards. Batch size is
    weighted by entity relationship type count to keep per-job output roughly
    equal.

    Returns list of matrix entries (dicts with string values).
    """
    new_shards, shard_sizes = detect_new_shards(entity_filter=entity_filter, cache_dir=cache_dir)

    if not new_shards:
        return []

    # Write detect results for sync jobs to reuse (avoids re-running detect)
    detect_path = os.environ.get("DETECT_RESULTS_PATH", "detect_results.json")
    with open(detect_path, "w") as f:
        json.dump({"shards": new_shards}, f)
    print(f"Detect results written to {detect_path}")

    matrix: list[dict] = []

    for entity, s3_keys in new_shards.items():
        spj = _shards_per_job_for_entity(entity, len(s3_keys))
        n_batches = -(-len(s3_keys) // spj)  # ceil div
        for i in range(n_batches):
            start = i * spj
            batch = s3_keys[start:start + spj]
            matrix.append({
                "entity": entity,
                "batch_index": str(i),
                "shard_count": str(len(batch)),
                "total_shards": str(len(s3_keys)),
                "label": f"{entity}-{i}",
            })

    return matrix


# ── Sync ─────────────────────────────────────────────────────────────────

SyncResult = dict  # {"entity": str, "s3_key": str, "status": str, ...}


def sync_shards(
    new_shards: dict[str, list[str]] | None = None,
    entity_filter: str | None = None,
    batch: int | None = None,
    shards_per_job: int | None = None,
    batch_keys: dict[str, list[str]] | None = None,
) -> None:
    """Download, extract, and upload new shards to HuggingFace.

    Processes shards for the given entity. If batch is set, only processes
    that slice (for matrix parallelism on large entities). Overlaps
    download of shard N+1 with extraction of shard N using a background
    thread. Uploads periodically (every 50 shards) via upload_large_folder
    so progress survives timeouts.

    Args:
        new_shards: Pre-computed detect results. If None, runs detect.
        entity_filter: Only process this entity.
        batch: Batch index within entity (0-indexed).
            If None, processes all shards for the entity.
        shards_per_job: Shards per batch (unused — weighted batching is
            computed from entity relationship counts).
    """
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi

    if new_shards is None:
        new_shards, _ = detect_new_shards(entity_filter=entity_filter, cache_dir=None)

    if not new_shards:
        print("No new shards to sync")
        return

    # Flatten to ordered list of (entity, s3_key) pairs
    queue: list[tuple[str, str]] = []
    if batch_keys:
        # Use pre-computed batch assignments from detect results
        for entity in ENTITY_TYPES_BUILD_ORDER:
            if entity in batch_keys:
                for s3_key in batch_keys[entity]:
                    queue.append((entity, s3_key))
    else:
        for entity in ENTITY_TYPES_BUILD_ORDER:
            if entity in new_shards:
                for s3_key in new_shards[entity]:
                    queue.append((entity, s3_key))

    # Apply batch slicing for large entities (only when no batch_keys)
    if batch is not None and not batch_keys:
        entity_name_for_batch = queue[0][0] if queue else "works"
        spj = _shards_per_job_for_entity(entity_name_for_batch, len(queue))
        batch_start = batch * spj
        queue = queue[batch_start:batch_start + spj]
        if not queue:
            print(f"Batch {batch} is empty, nothing to do")
            return

    total = len(queue)
    entity_name = queue[0][0] if queue else "unknown"
    print(f"Syncing {total} shards ({entity_name})")

    # Staging directory for all prepared files
    staging_root = Path(tempfile.mkdtemp(prefix="openalex-sync-"))
    upload_dir = staging_root / "upload"
    upload_dir.mkdir()

    # Upload every N shards to ensure progress survives timeouts
    UPLOAD_EVERY = 50

    results: list[SyncResult] = []
    succeeded = 0
    failed = 0
    processed = 0
    uploaded_count = 0

    # Prefetch function runs in background thread
    def _prefetch(s3_key: str, dest: Path) -> None:
        _download_shard(s3_key, dest)

    # Pre-download first shard
    first_entity, first_key = queue[0]
    first_dest = staging_root / Path(first_key).name
    _download_shard(first_key, first_dest)

    # Use a thread pool for overlapping download of next shard
    from concurrent.futures import Future
    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    outstanding_future: Future[None] | None = None

    try:
        for i, (entity, s3_key) in enumerate(queue):
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

            # Get the downloaded file (either pre-fetched or first)
            if i == 0:
                local_gz = first_dest
            else:
                local_gz = staging_root / Path(s3_key).name
                assert outstanding_future is not None
                outstanding_future.result()  # wait for prefetch to finish

            # Start prefetching next shard in background
            outstanding_future = None
            if i + 1 < len(queue):
                _, next_key = queue[i + 1]
                next_dest = staging_root / Path(next_key).name
                outstanding_future = prefetch_pool.submit(_prefetch, next_key, next_dest)

            try:
                # Rename to .jsonl.gz
                jsonl_gz = staging_root / (local_gz.stem + ".jsonl.gz")
                if local_gz != jsonl_gz:
                    local_gz.rename(jsonl_gz)

                # Place source file in upload dir
                hf_path = _s3_key_to_hf_path(s3_key)
                src_upload = upload_dir / hf_path
                src_upload.parent.mkdir(parents=True, exist_ok=True)
                src_upload.symlink_to(jsonl_gz.resolve())

                # Extract to parquet
                parquet_files = _extract_shard(
                    jsonl_gz, entity, staging_root / "parquet",
                )

                # Symlink parquets into upload dir
                for pq in parquet_files:
                    pq_rel = pq.relative_to(staging_root / "parquet")
                    pq_upload = upload_dir / "data" / pq_rel
                    pq_upload.parent.mkdir(parents=True, exist_ok=True)
                    pq_upload.symlink_to(pq.resolve())

                elapsed = time.monotonic() - t0
                n_files = 1 + len(parquet_files)
                result["status"] = "ok"
                result["files"] = n_files
                result["seconds"] = round(elapsed, 1)
                succeeded += 1

                print(
                    f"  [{processed}/{total}] prepared {s3_key} "
                    f"({n_files} files, {elapsed:.0f}s)"
                )

                # Clean up the .gz to free disk (parquets already symlinked)
                jsonl_gz.unlink(missing_ok=True)

            except Exception as exc:
                elapsed = time.monotonic() - t0
                result["status"] = "error"
                result["error"] = str(exc)
                result["seconds"] = round(elapsed, 1)
                failed += 1

                print(f"  [{processed}/{total}] FAILED {s3_key}: {exc}")

            results.append(result)

            # Periodic upload — flush prepared files to HF every N shards
            if succeeded > 0 and succeeded % UPLOAD_EVERY == 0:
                print(f"Uploading batch ({succeeded} shards prepared so far)...")
                api = HfApi()
                try:
                    api.upload_large_folder(
                        folder_path=str(upload_dir),
                        repo_id=HF_REPO_ID,
                        repo_type="dataset",
                        ignore_patterns=["._*"],
                    )
                    uploaded_count = succeeded
                    print("  Batch upload complete")
                    # Clean uploaded files from staging to free disk
                    for p in upload_dir.rglob("*"):
                        if p.is_file() and not p.is_symlink():
                            p.unlink(missing_ok=True)
                    for p in upload_dir.rglob("*"):
                        if p.is_symlink():
                            p.unlink(missing_ok=True)
                except Exception as exc:
                    print(f"  Batch upload FAILED: {exc}")

    finally:
        prefetch_pool.shutdown(wait=False)

    # Final upload for remaining files
    if succeeded > uploaded_count:
        remaining = succeeded - uploaded_count
        print(f"Uploading final batch ({remaining} shards)...")
        api = HfApi()
        t_upload = time.monotonic()
        try:
            api.upload_large_folder(
                folder_path=str(upload_dir),
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                ignore_patterns=["._*"],
            )
            upload_elapsed = time.monotonic() - t_upload
            uploaded_count = succeeded
            print(f"Upload complete ({upload_elapsed:.0f}s)")
        except Exception as exc:
            print(f"Upload FAILED: {exc}")
            for r in results[uploaded_count:]:
                if r["status"] == "ok":
                    r["status"] = "error"
                    r["error"] = f"upload failed: {exc}"
                    succeeded -= 1
                    failed += 1

    # Clean up
    shutil.rmtree(staging_root, ignore_errors=True)

    # Write results
    results_path = Path("sync_results.jsonl")
    with open(results_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Done: {succeeded} succeeded, {failed} failed out of {total}")
    if failed > 0:
        sys.exit(1)


# ── CLI ──────────────────────────────────────────────────────────────────


def _emit_matrix(entries: list[dict]) -> None:
    """Write matrix JSON and max_parallel to $GITHUB_OUTPUT or stdout."""
    matrix_json = json.dumps({"include": entries}, separators=(',', ':'))

    # Compute dynamic parallelism:
    # - Scale with job count: min(total, ceiling)
    # - Cap at GitHub concurrent limit (20 for free tier)
    # - Floor of 1
    # - Rate-limit safe: with per-shard uploads, each job makes multiple
    #   upload_large_folder calls. Cap keeps HF API pressure reasonable.
    GITHUB_MAX_CONCURRENT = 20
    max_parallel = min(len(entries), GITHUB_MAX_CONCURRENT) if entries else 1

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            if entries:
                f.write(f"has_new=true\nmatrix={matrix_json}\nmax_parallel={max_parallel}\n")
            else:
                f.write("has_new=false\nmatrix={\"include\":[]}\nmax_parallel=1\n")
    else:
        print(matrix_json)
        print(f"max_parallel={max_parallel}")


def _run_cli(args: argparse.Namespace) -> None:
    """Dispatch parsed CLI arguments."""
    if args.command == "detect":
        shards_found, _ = detect_new_shards(entity_filter=args.entity)
        if not shards_found:
            print("No new shards")
        else:
            total = sum(len(v) for v in shards_found.values())
            print(f"{total} new shards:")
            for entity, keys in shards_found.items():
                print(f"  {entity}: {len(keys)} new")
                for k in keys[:5]:
                    print(f"    {k}")
                if len(keys) > 5:
                    print(f"    ... and {len(keys) - 5} more")

    elif args.command == "detect-entities":
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        new_shards, shard_sizes = detect_new_shards(entity_filter=args.entity, cache_dir=cache_dir)

        detect_path = os.environ.get("DETECT_RESULTS_PATH", "detect_results.json")
        # Build batches first so we can store assignments in detect results
        batch_map: dict[str, list[list[str]]] = {}  # entity → [batch0_keys, batch1_keys, ...]
        for entity, keys in new_shards.items():
            if not keys:
                batch_map[entity] = []
                continue
            spj = _shards_per_job_for_entity(entity, len(keys))
            n_batches = -(-len(keys) // spj)
            if n_batches <= 1:
                batch_map[entity] = [keys]
            else:
                rel_count = _ENTITY_REL_COUNTS.get(entity, 1)
                batch_map[entity] = _size_aware_batch(keys, shard_sizes, n_batches, rel_count)

        with open(detect_path, "w") as f:
            json.dump({"shards": new_shards, "batches": batch_map}, f)
        print(f"Detect results written to {detect_path}")

        # Build matrix entries from batch assignments
        entries: list[dict] = []
        for entity, batches in batch_map.items():
            if not batches:
                continue
            if len(batches) == 1:
                entries.append({"entity": entity, "batch": "0", "label": entity})
            else:
                for i, batch in enumerate(batches):
                    if batch:
                        entries.append({
                            "entity": entity,
                            "batch": str(i),
                            "label": f"{entity}-{i}",
                        })

        _emit_matrix(entries)

        if entries:
            total = sum(len(v) for v in new_shards.values())
            print(f"Matrix: {len(entries)} entries")
            for entity, keys in new_shards.items():
                spj = _shards_per_job_for_entity(entity, len(keys))
                n = -(-len(keys) // spj)
                has_sizes = any(shard_sizes.get(k, 0) > 0 for k in keys)
                size_note = "size-aware" if has_sizes else "positional"
                print(f"  {entity}: {len(keys)} shards, {spj}/batch → {n} job(s) ({size_note})")
            print(f"Total: {total} shards across {len(new_shards)} entities")
        else:
            print("No new shards")

    elif args.command == "prepare-matrix":
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        matrix = prepare_matrix(
            entity_filter=args.entity,
            shards_per_batch=args.shards_per_batch,
            cache_dir=cache_dir,
        )
        _emit_matrix(matrix)
        print(f"Matrix: {len(matrix)} entries")

    elif args.command == "sync":
        loaded: dict[str, list[str]] | None = None
        if args.detect_file:
            with open(args.detect_file) as f:
                detect_data = json.load(f)
                if isinstance(detect_data, dict) and "shards" in detect_data:
                    loaded = detect_data["shards"]
                else:
                    loaded = detect_data
        sync_shards(
            new_shards=loaded,
            entity_filter=args.entity,
            batch=args.batch,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync new OpenAlex shards to HuggingFace"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    detect_parser = sub.add_parser("detect", help="Show new shards without syncing")
    detect_parser.add_argument("--entity", type=str, default=None)

    entity_parser = sub.add_parser("detect-entities",
        help="Detect new shards and output entity-level matrix for CI")
    entity_parser.add_argument("--entity", type=str, default=None)
    entity_parser.add_argument("--cache-dir", type=str, default=None)

    matrix_parser = sub.add_parser("prepare-matrix", help="Generate GitHub Actions matrix JSON")
    matrix_parser.add_argument("--entity", type=str, default=None)
    matrix_parser.add_argument("--shards-per-batch", type=int, default=None)
    matrix_parser.add_argument("--cache-dir", type=str, default=None,
        help="Directory to cache HF listing results across runs",
    )

    sync_parser = sub.add_parser("sync", help="Download, extract, and upload new shards")
    sync_parser.add_argument("--entity", type=str, default=None)
    sync_parser.add_argument("--batch", type=int, default=None,
        help="Batch index within entity (0-indexed, for matrix splitting)",
    )
    sync_parser.add_argument(
        "--detect-file", type=str, default=None,
        help="Load detect results from this file instead of re-running detect",
    )

    _run_cli(parser.parse_args())
