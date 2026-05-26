#!/usr/bin/env python3
"""Detect and sync new OpenAlex snapshot shards from S3 to HuggingFace.

Compares S3 manifests against the current HF dataset to find new shards,
then downloads, renames, extracts to parquet, and uploads each one.

Designed for CI: processes one shard at a time to keep disk bounded.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from sync.common import ENTITY_TYPES_BUILD_ORDER

S3_BUCKET = "openalex"
HF_REPO_ID = "Mearman/OpenAlex"


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
        # s3://openalex/data/works/updated_date=.../part_0000.gz
        if url.startswith(f"s3://{S3_BUCKET}/"):
            key = url[len(f"s3://{S3_BUCKET}/"):]
            entries[key] = entry.get("meta", {})
    return entries


def _hf_source_files_for_entity(entity: str) -> set[str] | None:
    """List .jsonl.gz files on HuggingFace for a specific entity.

    For large entities (works), uses shallow listing of partition directories
    then per-partition file enumeration to avoid timeouts on recursive listing
    of directories with tens of thousands of files.

    Returns None if the listing fails, signalling the caller should fall back
    to per-file checks.
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
    # 1. List partition directories (non-recursive — fast).
    # 2. For each partition, list files individually.
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


def _s3_key_to_hf_path(s3_key: str) -> str:
    """Convert S3 key to HF path with .jsonl.gz extension.

    s3 key:   data/works/updated_date=2024-01-13/part_0000.gz
    hf path:  data/works/updated_date=2024-01-13/part_0000.jsonl.gz
    """
    if s3_key.endswith(".gz") and not s3_key.endswith(".jsonl.gz"):
        return s3_key[:-3] + ".jsonl.gz"
    return s3_key


def _parquet_paths_for_source(hf_path: str) -> list[str]:
    """Derive expected parquet file paths for a source shard.

    Given data/works/updated_date=2024-01-13/part_0000.jsonl.gz,
    the parquet files are:
      data/works/{rel_type}/works__updated_date=2024-01-13__part_0000.parquet

    We can't know which rel_types exist without extracting, so we return
    the pattern prefix for deletion/cleanup purposes.
    """
    # data/works/updated_date=2024-01-13/part_0000.jsonl.gz
    # → entity=works, shard_key=works__updated_date=2024-01-13__part_0000
    parts = hf_path.split("/")
    entity = parts[1]
    filename = parts[-1]  # part_0000.jsonl.gz
    partition = parts[2]  # updated_date=2024-01-13
    part_stem = filename.rsplit(".", 2)[0]  # part_0000
    shard_key = f"{entity}__{partition}__{part_stem}"
    return shard_key


def _download_shard(s3_key: str, dest: Path) -> None:
    """Download a single shard from S3."""
    s3 = _s3_client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(S3_BUCKET, s3_key, str(dest))


def _extract_shard(source_path: Path, entity: str, output_dir: Path) -> list[Path]:
    """Extract a single source shard to parquet. Returns list of parquet files."""
    from sync.extract import convert_relationships

    # Extract to a temp dir, then move results
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Place the source file in the expected layout
        mock_data = tmp_dir / "data" / entity
        mock_data.mkdir(parents=True)

        # Preserve partition dir structure
        partition_dir = mock_data / source_path.parent.name
        partition_dir.mkdir(exist_ok=True)
        target = partition_dir / source_path.name

        # Copy or symlink the source file
        shutil.copy2(source_path, target)

        # Override SNAPSHOT_DIR temporarily
        import sync.common as common
        original_snapshot = common.SNAPSHOT_DIR
        common.SNAPSHOT_DIR = tmp_dir / "data"

        try:
            counts = convert_relationships(
                entity,
                force=True,
                workers=1,
            )
        finally:
            common.SNAPSHOT_DIR = original_snapshot

        # Collect parquet files from the mock data dir
        parquet_files = list((tmp_dir / "data").rglob("*.parquet"))
        result = []
        for pq in parquet_files:
            dest = output_dir / pq.relative_to(tmp_dir / "data")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pq), str(dest))
            result.append(dest)

        return result


def detect_new_shards() -> dict[str, list[str]]:
    """Compare S3 manifests against HF to find new shards.

    Returns {entity: [s3_key, ...]} for each entity with new files.
    """
    from huggingface_hub import HfApi
    api = HfApi()

    new_shards: dict[str, list[str]] = {}

    for entity in ENTITY_TYPES_BUILD_ORDER:
        manifest = _fetch_manifest(entity)
        if not manifest:
            continue

        # For small entities, list the directory tree.
        # For large ones, check individual files.
        hf_files = _hf_source_files_for_entity(entity)

        if hf_files is not None:
            # Tree listing worked
            new_for_entity = []
            for s3_key in manifest:
                hf_path = _s3_key_to_hf_path(s3_key)
                if hf_path not in hf_files:
                    new_for_entity.append(s3_key)
        else:
            # Tree listing failed — check each manifest entry individually
            new_for_entity = []
            for s3_key in manifest:
                hf_path = _s3_key_to_hf_path(s3_key)
                if not api.file_exists(HF_REPO_ID, hf_path, repo_type="dataset"):
                    new_for_entity.append(s3_key)

        if new_for_entity:
            new_shards[entity] = new_for_entity

    return new_shards


def sync_shards(new_shards: dict[str, list[str]] | None = None) -> None:
    """Download, extract, and upload new shards to HuggingFace.

    Processes one shard at a time to keep disk usage bounded.
    """
    from huggingface_hub import HfApi, CommitOperationAdd

    if new_shards is None:
        new_shards = detect_new_shards()

    if not new_shards:
        print("No new shards to sync")
        return

    total = sum(len(v) for v in new_shards.values())
    print(f"Syncing {total} new shards across {len(new_shards)} entities")

    api = HfApi()
    processed = 0

    for entity, s3_keys in new_shards.items():
        for s3_key in s3_keys:
            processed += 1
            hf_path = _s3_key_to_hf_path(s3_key)
            print(f"  [{processed}/{total}] {s3_key}")

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
                api.create_commit(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"feat: sync {hf_path}",
                )

                print(f"    uploaded {len(operations)} files")

    print(f"Done: synced {processed} shards")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync new OpenAlex shards to HuggingFace")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Show new shards without syncing")
    sync = sub.add_parser("sync", help="Download, extract, and upload new shards")

    args = parser.parse_args()

    if args.command == "detect":
        new = detect_new_shards()
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
        sync_shards()
