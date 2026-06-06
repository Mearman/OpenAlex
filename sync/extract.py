"""Extract nested arrays from JSONL into relationship Parquet tables.

Processes entity records (works, authors, sources, institutions,
publishers, funders, concepts, topics, subfields, fields) to produce
normalised relationship tables covering all OpenAlex data model
relationships.

Workers process disjoint chunks of the source-file list and each
write to their own ``part-WW-NNNNN.parquet`` shards under the
relationship-type directory, where ``WW`` is the worker index. The
parent process aggregates row counts and writes a single
``_provenance.json`` per relationship type.
"""

from __future__ import annotations

import json
import logging
import hashlib
import os
from functools import lru_cache
import multiprocessing
import shutil
import subprocess
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sync.common import (
    SNAPSHOT_DIR,
    STAGING_DIR,
    extract_id,
    iter_source_files,
    iter_jsonl,
    create_output_dir,
    get_skipped_missing_files,
    reset_skipped_files,
    rt_dir,
    nested_rt_path,
    format_size,
    _json_loads,
)
from sync.schema import EntitySchema, extract_relationships, get_entity_schema

log = logging.getLogger("openalex-sync")

# ── Batch thresholds ────────────────────────────────────────────────────

_BATCH_SIZE = 4_000_000
_PROGRESS_INTERVAL = 1_000_000
_DEFAULT_WORKERS = 6

# Memory model: reserve ~2 GB for the OS and assume each worker peaks
# around 6 GB resident (zstd compressor + PyArrow batch + Python heap).
# Empirically, 8 workers on 16 GB causes swap thrashing; 8 on 64 GB is fine.
_RAM_HEADROOM_GB = 2
_RAM_PER_WORKER_GB = 6

# Workers are I/O-bound (gzip decompress + JSON parse wait on disk), so we
# allow oversubscription: more workers than allocated CPUs. A factor of 3x
# means 16 allocated CPUs can run up to 48 workers if memory allows. The
# factor is capped to avoid diminishing returns from context-switch overhead.
_CPU_OVERSUBSCRIBE = 3


def _get_memory_limit_bytes() -> int | None:
    """Return the process's memory limit in bytes, or None if unlimited.

    Checks in order: cgroup v2, cgroup v1, Slurm env vars, POSIX RLIMIT_AS.
    Returns None if no limit is set (system memory is the effective limit).
    """
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            val = f.read().strip()
            if val != "max":
                limit = int(val)
                if limit < (1 << 62):
                    return limit
    except (FileNotFoundError, ValueError, OSError):
        pass

    # cgroup v1
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            limit = int(f.read().strip())
            if limit < (1 << 62):
                return limit
    except (FileNotFoundError, ValueError, OSError):
        pass

    # Slurm env vars (in MB)
    if "SLURM_MEM_PER_NODE" in os.environ:
        return int(os.environ["SLURM_MEM_PER_NODE"]) * 1024 * 1024
    if "SLURM_MEM_PER_CPU" in os.environ:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        return int(os.environ["SLURM_MEM_PER_CPU"]) * cpus * 1024 * 1024

    # POSIX RLIMIT_AS
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        if 0 < soft < (1 << 62):
            return soft
    except (ImportError, ValueError, OSError):
        pass

    return None


def _get_cpu_count() -> int:
    """Return CPUs available to this process, preferring Slurm-allocated cores."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    spec = line.split(":", 1)[1].strip()
                    count = 0
                    for part in spec.split(","):
                        if "-" in part:
                            a, b = part.split("-")
                            count += int(b) - int(a) + 1
                        else:
                            count += 1
                    if count > 0:
                        return count
    except (FileNotFoundError, ValueError):
        pass
    return os.cpu_count() or 1


def _auto_workers(reserve: int = 0) -> int:
    """Pick an extraction worker count based on available RAM and CPUs.

    Workers are I/O-bound (gzip decompression + JSON parsing spend most of
    their time waiting on disk I/O), so the primary limit is memory: the job's
    memory limit divided by the per-worker RAM budget.

    CPU count is not a hard cap (workers aren't CPU-bound) but we cap at
    ``_CPU_OVERSUBSCRIBE × allocated_cpus`` to avoid context-switch thrash
    from runaway oversubscription on large-memory nodes.

    On Slurm-managed nodes, uses the JOB's memory limit (cgroup or
    SLURM_MEM_PER_NODE) rather than the node's total physical memory.
    This prevents OOM kills when the sync runs as a small-memory job
    on a large-memory compute node.

    ``reserve`` withholds that many workers from the count so a concurrent
    stage (e.g. a background upload running alongside extraction) has memory
    to use without oversubscribing the allocation.
    """
    cpu_count = _get_cpu_count()
    cpu_cap = cpu_count * _CPU_OVERSUBSCRIBE

    limit_bytes = _get_memory_limit_bytes()
    if limit_bytes is not None:
        limit_gb = limit_bytes / (1024 ** 3)
        by_ram = max(1, int((limit_gb - _RAM_HEADROOM_GB) // _RAM_PER_WORKER_GB))
        chosen = max(1, min(cpu_cap, by_ram) - max(0, reserve))
        log.info(
            "Auto-sized workers to %d (%.0fGB job limit, %d CPUs, %d reserved, RAM cap %d, CPU cap %d)",
            chosen, limit_gb, cpu_count, reserve, by_ram, cpu_cap,
        )
        return chosen

    # Fall back to node physical memory
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total_bytes = page_size * phys_pages
    except (AttributeError, ValueError, OSError):
        chosen = max(1, _DEFAULT_WORKERS - max(0, reserve))
        log.info(
            "Auto-sized workers to %d (no limit detected, %d CPUs, %d reserved)",
            chosen, cpu_count, reserve,
        )
        return chosen

    total_gb = total_bytes / (1024 ** 3)
    by_ram = max(1, int((total_gb - _RAM_HEADROOM_GB) // _RAM_PER_WORKER_GB))
    chosen = max(1, min(cpu_cap, by_ram) - max(0, reserve))
    log.info(
        "Auto-sized workers to %d (%.0fGB node RAM, %d CPUs, %d reserved, RAM cap %d, CPU cap %d)",
        chosen, total_gb, cpu_count, reserve, by_ram, cpu_cap,
    )
    return chosen


# Per-entity extraction functions removed. All extraction is schema-driven
# via schema.py extract_relationships(). PyArrow write schemas are inferred
# from the first batch of rows rather than looked up from a hardcoded table.



# ── Provenance ──────────────────────────────────────────────────────────


def _git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None



# ── Source-file key & deterministic output identity ─────────────────────


def _source_file_key(source_path: Path) -> str:
    """Derive a deterministic output key from a source file path.

    The key is the path relative to ``SNAPSHOT_DIR`` with ``/`` replaced
    by ``__`` and the ``.gz`` extension stripped.  This is human-readable
    and collision-free within a single entity type.

    Example::

        works/updated_date=2026-01-09/part_0047.jsonl.gz
        → works__updated_date=2026-01-09__part_0047

    Workers with any ``--workers`` value produce the same shard name
    for the same source file, making output identity independent of
    scheduling.
    """
    try:
        rel = str(source_path.relative_to(SNAPSHOT_DIR))
    except ValueError:
        rel = str(source_path)
    # Strip .jsonl.gz or .gz suffix (the output is parquet, not gzip)
    if rel.endswith(".jsonl.gz"):
        rel = rel[:-9]
    elif rel.endswith(".gz"):
        rel = rel[:-3]
    elif rel.endswith(".jsonl"):
        rel = rel[:-6]
    return rel.replace("/", "__")


def _shard_path(output_dir: Path, source_key: str) -> Path:
    """Return the parquet shard path for a given source-file key."""
    return output_dir / f"{source_key}.parquet"


# ── Per-unit provenance ─────────────────────────────────────────────────


def _write_unit_provenance(
    output_dir: Path,
    *,
    source_key: str,
    source_file: str,
    content_length: int,
    row_count: int,
    status: str,
    relationship_type: str,
    output_hash: str | None = None,
    skipped: bool = False,
) -> None:
    """Write provenance for a single source-file unit.

    Stored as ``_units/{source_key}.json`` inside the relationship-type
    output directory.  Each unit record contains:

    - ``source_key``      — deterministic output key
    - ``source_file``     — original relative path
    - ``content_length``  — byte size of the source gzip file
    - ``row_count``       — rows extracted from this file
    - ``output_hash``     — SHA-256 (16 hex chars) of the parquet shard
    - ``status``          — ``"complete"`` | ``"skipped"`` | ``"empty"``
    - ``relationship_type``
    - ``git_commit``
    - ``timestamp``
    """
    units_dir = output_dir / "_units"
    units_dir.mkdir(parents=True, exist_ok=True)
    prov = {
        "source_key": source_key,
        "source_file": source_file,
        "content_length": content_length,
        "row_count": row_count,
        "status": status,
        "relationship_type": relationship_type,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if output_hash:
        prov["output_hash"] = output_hash
    path = units_dir / f"{source_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)


def _completed_source_keys(output_dir: Path) -> set[str]:
    """Return source keys that have a parquet shard on disk.

    Lists ``*.parquet`` files in *output_dir* by name only. Footer
    validation has been removed because it required opening every shard
    and reading metadata via PyArrow — on ExFAT with thousands of files
    per relationship this took several minutes per directory.

    Workers will encounter and skip any corrupt shard during actual use.
    Empty/zero-row parquets are valid completion markers (written
    deliberately for source files that produce no rows for a given
    relationship), so we no longer need to distinguish "has data" from
    "valid footer" at startup.
    """
    if not output_dir.exists():
        return set()
    return {
        shard.stem
        for shard in output_dir.glob("*.parquet")
        if not shard.name.startswith("._")
    }


def validate_shard(path: Path) -> bool:
    """Return True iff the parquet shard at *path* is readable.

    Opens the file with ``pyarrow.parquet.ParquetFile`` and accesses its
    ``.metadata`` (which reads and parses the footer). Any exception —
    truncation, corrupt footer, unreadable file — returns False.

    A zero-row shard with a valid footer returns True: empty shards are
    written deliberately as completion markers for source files that
    produce no rows for a given relationship, so a valid empty footer is
    not a failure.
    """
    try:
        pf = pq.ParquetFile(str(path))
        # Accessing metadata forces the footer to be read and parsed.
        _ = pf.metadata
        return True
    except Exception:
        return False


def _completed_source_keys_verified(output_dir: Path) -> set[str]:
    """Return source keys with a *readable* parquet shard on disk.

    Verify-aware variant of ``_completed_source_keys``: every ``*.parquet``
    shard is opened and its footer validated via ``validate_shard``. Any
    shard that fails validation is deleted (logged at warning level) so
    that it is treated as pending and re-extracted on this run.

    This is the slow path — it opens every shard. ``_completed_source_keys``
    remains the fast, filename-only default for verify=False.
    """
    if not output_dir.exists():
        return set()
    completed: set[str] = set()
    for shard in output_dir.glob("*.parquet"):
        if shard.name.startswith("._"):
            continue
        if validate_shard(shard):
            completed.add(shard.stem)
        else:
            log.warning(
                "Unreadable parquet shard %s — deleting so it re-extracts",
                shard,
            )
            shard.unlink()
    return completed


def _hf_completed_source_keys(
    entity: str,
    repo_id: str | None = None,
) -> dict[str, set[str]]:
    """Return parquet shard stems already present on the HuggingFace remote.

    Used for cross-machine resume: when M3 uploads parquets to HF, mini's
    next sync invocation will see them via this lookup and skip the
    corresponding source files instead of re-extracting.

    Returns ``{rel_name: set of shard_stems}`` matching the structure of
    ``_completed_source_keys`` so the two can be unioned per rel type.

    On any HF API failure, returns an empty dict and falls back silently
    to local-only resume.
    """
    if repo_id is None:
        repo_id = os.environ.get("OPENALEX_HF_REPO", "Mearman/OpenAlex")

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        all_files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        log.warning(
            "HF list_repo_files failed for %s (%s) — falling back to local-only resume",
            repo_id, exc,
        )
        return {}

    from sync.schema import _singular as _entity_singular
    singular = _entity_singular(entity)
    entity_prefix_path = f"data/{entity}/"

    by_rel: dict[str, set[str]] = {}
    for f in all_files:
        if not f.startswith(entity_prefix_path) or not f.endswith(".parquet"):
            continue
        rel_path = f[len(entity_prefix_path):]
        parts = rel_path.split("/", 1)
        if len(parts) != 2:
            continue
        subtable, filename = parts
        # Reconstruct rel_name: e.g. entity="authors", subtable="counts_by_year"
        # → rel_name "author_counts_by_year". Matches nested_rt_path's mapping.
        rel_name = f"{singular}_{subtable}"
        source_key = filename[:-len(".parquet")]
        by_rel.setdefault(rel_name, set()).add(source_key)

    total_keys = sum(len(s) for s in by_rel.values())
    log.info(
        "HF resume: %s — %d shard stems present on HF across %d rel types",
        entity, total_keys, len(by_rel),
    )
    return by_rel


def _count_shard_rows(output_dir: Path) -> tuple[int, int]:
    """Count number of valid shards in output_dir. Returns ``(0, shard_count)``.

    Previously also read each parquet footer to sum ``num_rows`` — useful for
    informational logging but extremely slow on APFS with thousands of files
    per relationship (and the totals were not used for any control-flow
    decision, only printed). Returns 0 for ``total_rows`` and lets the
    finalise step record shard count only.
    """
    if not output_dir.exists():
        return 0, 0
    shard_count = sum(
        1 for shard in output_dir.glob("*.parquet")
        if not shard.name.startswith("._")
    )
    return 0, shard_count


# Legacy loader retained for migration compatibility
def _load_unit_provenances(output_dir: Path) -> dict[str, dict]:
    """Load legacy per-unit provenance records (migration only).

    Returns ``{source_key → provenance_dict}``.
    """
    units_dir = output_dir / "_units"
    if not units_dir.exists():
        return {}
    result: dict[str, dict] = {}
    _json_load = json.load
    _json_decode_error = json.JSONDecodeError
    for prov_path in sorted(units_dir.glob("*.json")):
        if prov_path.name.startswith("._"):
            continue
        try:
            with open(prov_path) as f:
                prov = _json_load(f)
            key = prov.get("source_key", prov_path.stem)
            result[key] = prov
        except (_json_decode_error, KeyError, OSError, UnicodeDecodeError):
            log.warning("Corrupt unit provenance: %s, ignoring", prov_path)
    return result


def _compute_pending_source_files(
    source_files: list[Path],
    completed_keys: set[str],
) -> list[Path]:
    """Return source files whose shard does not yet exist on disk.

    Compares ``_source_file_key(f)`` against *completed_keys* (derived
    from existing valid parquet shards).
    """
    return [f for f in source_files if _source_file_key(f) not in completed_keys]


# ── Aggregate provenance ────────────────────────────────────────────────


def _write_provenance(
    output_dir: Path,
    *,
    relationship_type: str,
    record_count: int,
    source_entity: str,
    source_file_count: int,
) -> None:
    """Write a _provenance.json file recording conversion metadata."""
    provenance = {
        "relationship_type": relationship_type,
        "record_count": record_count,
        "source_entity": source_entity,
        "source_file_count": source_file_count,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    provenance_path = output_dir / "_provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    log.info("Wrote provenance: %s", provenance_path)


def _finalise_type(
    rt_dir: Path,
    *,
    relationship_type: str,
    source_entity: str,
    source_file_count: int,
    verify: bool = False,
) -> int:
    """Finalise a relationship type after all units are processed.

    Scans parquet shards on disk, computes totals, writes aggregate
    ``_provenance.json``, ``_lineage.json``, and
    ``_manifest_snapshot.json``.  Returns the total row count.

    When *verify* is True, every shard's footer is validated before the
    type is finalised. A shard that is still unreadable after the run has
    had its chance to re-extract it is a genuine failure: it is logged
    with full context, counted, and a ``RuntimeError`` is raised so the
    run fails loudly rather than recording corrupt output as complete.
    """
    if verify:
        unreadable = [
            shard
            for shard in sorted(rt_dir.glob("*.parquet"))
            if not shard.name.startswith("._") and not validate_shard(shard)
        ]
        if unreadable:
            for shard in unreadable:
                log.error(
                    "%s: shard %s still unreadable after re-extraction",
                    relationship_type, shard,
                )
            raise RuntimeError(
                f"{relationship_type}: {len(unreadable)} shard(s) remain "
                f"unreadable in {rt_dir} after verify re-extraction"
            )

    total_rows, shard_count = _count_shard_rows(rt_dir)

    _write_provenance(
        rt_dir,
        relationship_type=relationship_type,
        record_count=total_rows,
        source_entity=source_entity,
        source_file_count=source_file_count,
    )

    # Write lineage: shard_name (one shard per source file)
    lineage: dict[str, list[str]] = {}
    for shard in sorted(rt_dir.glob("*.parquet")):
        if shard.name.startswith("._"):
            continue
        key = shard.stem
        lineage.setdefault("", []).append(f"{key}.parquet")
    _write_lineage(rt_dir, lineage)

    # Write manifest snapshot for drift detection
    current_manifest = _load_entity_manifest(source_entity)
    _write_manifest_snapshot(rt_dir, current_manifest)

    log.info(
        "%s: finalised — %d/%d shards, %d total rows",
        relationship_type, shard_count, source_file_count, total_rows,
    )
    return total_rows


# ── Manifest snapshot & drift detection ─────────────────────────────────


def _manifest_key_to_path(file_rel: str) -> Path:
    """Convert a manifest key (S3-style ``.gz``) to a local filesystem path.

    The manifest stores keys like ``works/updated_date=.../part_0000.gz``
    (matching the S3 bucket).  On disk we use ``.jsonl.gz`` so the HF
    dataset viewer detects the inner format.  This helper bridges the two.
    """
    if file_rel.endswith(".gz") and not file_rel.endswith(".jsonl.gz"):
        file_rel = file_rel[:-3] + ".jsonl.gz"
    return SNAPSHOT_DIR / file_rel


def _load_entity_manifest(entity_type: str) -> dict[str, dict]:
    """Load the entity manifest and return {file_rel → {content_length, record_count}}.

    *file_rel* is relative to SNAPSHOT_DIR (S3-style), e.g.
    ``works/updated_date=2026-01-09/part_0000.gz``.
    """
    manifest_path = SNAPSHOT_DIR / entity_type / "manifest"
    if not manifest_path.exists():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    result: dict[str, dict] = {}
    for entry in raw.get("entries", []):
        url: str = entry.get("url", "")
        if url.startswith("s3://openalex/data/"):
            file_rel = url[len("s3://openalex/data/"):]
        else:
            continue
        meta = entry.get("meta", {})
        result[file_rel] = {
            "content_length": meta.get("content_length", 0),
            "record_count": meta.get("record_count", 0),
        }
    return result


def _write_manifest_snapshot(
    output_dir: Path,
    manifest: dict[str, dict],
) -> None:
    """Write a _manifest_snapshot.json for drift detection on future runs."""
    path = output_dir / "_manifest_snapshot.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.debug("Wrote manifest snapshot: %s (%d files)", path, len(manifest))


def _load_manifest_snapshot(output_dir: Path) -> dict[str, dict]:
    """Load a previously stored manifest snapshot."""
    path = output_dir / "_manifest_snapshot.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _detect_manifest_drift(
    current: dict[str, dict],
    previous: dict[str, dict],
) -> dict[str, str]:
    """Compare current manifest with previous snapshot.

    Returns ``{file_rel → drift_type}`` where *drift_type* is one of:

    - ``"added"``     — new file not in previous snapshot
    - ``"removed"``   — file gone from current manifest
    - ``"changed"``   — content_length differs
    """
    drift: dict[str, str] = {}
    all_keys = set(current) | set(previous)
    for key in sorted(all_keys):
        if key not in previous:
            drift[key] = "added"
        elif key not in current:
            drift[key] = "removed"
        elif current[key]["content_length"] != previous[key].get("content_length", -1):
            drift[key] = "changed"
    return drift


def _write_lineage(
    output_dir: Path,
    lineage: dict[str, list[str]],
) -> None:
    """Write a consolidated _lineage.json alongside the parquet files."""
    path = output_dir / "_lineage.json"
    lineage_sorted = {k: sorted(v) for k, v in sorted(lineage.items())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lineage_sorted, f, indent=2)
    log.debug("Wrote lineage: %s (%d source files)", path, len(lineage_sorted))


def _load_lineage(output_dir: Path) -> dict[str, list[str]]:
    """Load a previously stored lineage map."""
    path = output_dir / "_lineage.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ── Parquet writing — one shard per source file per type ─────────────────


def _extract_one_source_file(
    source_file: Path,
    entity_type: str,
    rel_types: frozenset[str],
    batch_size: int,
    output_base: Path,
    staging_dir: Path | None,
) -> dict[str, dict]:
    """Extract all relationship types from a single source file.

    For each relationship type, writes exactly one parquet shard named
    ``{source_key}.parquet`` and one unit provenance record.  The output
    shard name is derived from the source file path, so it is
    deterministic regardless of worker assignment.

    Returns ``{rel_type → {"source_key": str, "row_count": int}}``.
    """
    source_key = _source_file_key(source_file)
    schema = _entity_schema(entity_type)

    # Open one writer per relationship type for this source file
    writers: dict[str, _SourceFileWriter] = {}
    for rt in rel_types:
        out_dir = rt_dir(output_base, rt)
        writers[rt] = _SourceFileWriter(
            out_dir, rt, source_key, staging_dir=staging_dir,
        )

    buffers: dict[str, list[dict]] = {rt: [] for rt in rel_types}

    # The main entity table (one row per record) is written in a single batch
    # at file close so its column types are inferred from every row in the
    # partition — a column absent from the first batch but present later would
    # otherwise be locked to a null type. Identify it so the inner loop does
    # not mid-flush it.
    main_rel = next(
        (fs.rel_name for fs in schema.fields if fs.pattern == "scalar"), None
    )

    # Cache globals for inner-loop performance
    _buffers = buffers
    _writers = writers

    # Only compute the relationship types being written for this file (the
    # rest are already complete and would be discarded) — avoids the expensive
    # extraction of completed tables on the resume path.
    for record in iter_jsonl(source_file):
        rels = _extract_entity_relationships(record, schema, wanted=rel_types)
        for rt, rows in rels.items():
            buf = _buffers.get(rt)
            if buf is None:
                continue
            buf.extend(rows)
            if rt != main_rel and len(buf) >= batch_size:
                _writers[rt].write_batch(buf)
                _buffers[rt] = []

    results: dict[str, dict] = {}
    for rt in rel_types:
        if buffers[rt]:
            writers[rt].write_batch(buffers[rt])
        writers[rt].close()

        row_count = writers[rt].total_count
        results[rt] = {"source_key": source_key, "row_count": row_count}

    return results

def _hash_file(path: Path) -> str:
    """SHA-256 of file contents, truncated to 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class _SourceFileWriter:
    """Manages a single ParquetWriter for one source file × one type.

    Output filename is always ``{source_key}.parquet`` — deterministic
    regardless of which worker processes the file.
    """

    output_hash: str | None = None

    def __init__(
        self,
        output_dir: Path,
        rel_type: str,
        source_key: str,
        staging_dir: Path | None = None,
    ) -> None:
        self.rel_type = rel_type
        self.schema: pa.Schema | None = None  # inferred from first batch
        self.output_dir = output_dir
        self.staging_dir = staging_dir
        self.source_key = source_key
        self.total_count = 0
        self._writer: pq.ParquetWriter | None = None
        self._shard_name = f"{source_key}.parquet"
        self._current_path: Path | None = None

    def _rows_to_table(self, rows: list[dict]) -> pa.Table:
        """Convert rows to an Arrow table, inferring and caching schema on first call.

        First call: infers schema via from_pylist, promotes narrow int/float
        types to int64/float64, caches schema, and reuses the already-built
        table via cast() — avoiding a second conversion.

        Subsequent calls: uses columnar construction (from_pydict) which is
        faster than from_pylist for large batches because PyArrow builds each
        column from a homogeneous Python list using a C-level loop, avoiding
        per-row dict key resolution.
        """
        if self.schema is None:
            try:
                raw = pa.Table.from_pylist(rows)
            except (pa.ArrowTypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise RuntimeError(self._type_error_detail(rows, exc)) from exc
            fields = []
            for f in raw.schema:
                if f.type in (pa.int32(), pa.int16(), pa.uint8()):
                    fields.append(pa.field(f.name, pa.int64(), nullable=f.nullable))
                elif f.type == pa.float32():
                    fields.append(pa.field(f.name, pa.float64(), nullable=f.nullable))
                else:
                    fields.append(f)
            self.schema = pa.schema(fields)
            return raw.cast(self.schema)

        # Known schema — build column-at-a-time for better throughput
        try:
            return pa.Table.from_pydict(
                {name: [row.get(name) for row in rows] for name in self.schema.names},
                schema=self.schema,
            )
        except (pa.ArrowTypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise RuntimeError(self._type_error_detail(rows, exc)) from exc

    def _type_error_detail(self, rows: list[dict], exc: Exception) -> str:
        """Build a diagnostic naming the table, source, and the offending
        columns (those with more than one value type) with example values, so a
        type error points straight at the bad data instead of a bare Arrow error.
        """
        col_types: dict[str, dict[str, Any]] = {}
        for row in rows:
            for key, val in row.items():
                if val is None:
                    continue
                col_types.setdefault(key, {}).setdefault(type(val).__name__, val)
        mixed = {
            col: types for col, types in col_types.items() if len(types) > 1
        }
        return (
            f"Arrow type error building shard rel_type={self.rel_type} "
            f"source={self.source_key} ({len(rows)} rows): {exc}. "
            f"Mixed-type columns (name → type:example): {mixed or 'none detected'}"
        )

    def _ensure_writer(self) -> pq.ParquetWriter:
        assert self.schema is not None, f"Schema not yet inferred for {self.rel_type}"
        if self._writer is None:
            write_dir = self.staging_dir if self.staging_dir else self.output_dir
            write_dir.mkdir(parents=True, exist_ok=True)
            out_path = write_dir / self._shard_name
            self._writer = pq.ParquetWriter(out_path, self.schema, compression="snappy")
            self._current_path = out_path
        return self._writer

    def write_batch(self, rows: list[dict]) -> None:
        if not rows:
            return
        table = self._rows_to_table(rows)
        writer = self._ensure_writer()
        writer.write_table(table)
        self.total_count += len(rows)
        log.info(
            "%s: wrote row group (%d rows, %d total)",
            self.rel_type, len(rows), self.total_count,
        )

    def close(self) -> None:
        """Close writer, stage-to-final move, compute output hash."""
        # Always create a file — even with 0 rows — so the shard exists
        # on disk as a completion marker (replaces _units/ provenance).
        if self._writer is None:
            write_dir = self.staging_dir if self.staging_dir else self.output_dir
            write_dir.mkdir(parents=True, exist_ok=True)
            out_path = write_dir / self._shard_name
            # Write empty parquet as completion marker
            if self.schema is not None:
                with pq.ParquetWriter(out_path, self.schema, compression="snappy") as w:
                    pass  # empty file with schema
            else:
                # No data was ever written — create a minimal empty parquet
                empty_schema = pa.schema([pa.field("_placeholder", pa.int64())])
                empty_table = pa.table({"_placeholder": pa.array([], type=pa.int64())})
                pq.write_table(empty_table, out_path, compression="snappy")
            self._current_path = out_path

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        if self._current_path is not None:
            if self.staging_dir and self._current_path.parent != self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                dest = self.output_dir / self._current_path.name
                shutil.move(str(self._current_path), str(dest))
                self._current_path = dest
            # Compute content hash of the written parquet file
            self.output_hash = _hash_file(self._current_path)
            log.info(
                "%s: closed %s (%d rows, hash=%s)",
                self.rel_type, self._current_path.name, self.total_count,
                self.output_hash,
            )


# ── Worker entry point (must be module-level for multiprocessing) ───────


def _worker_process_files(
    args: tuple[int, list[Path], str, list[str], int, dict[str, set[str]], Path],
) -> dict:
    """Process a list of source files in one worker process.

    Each source file is processed independently — the worker opens and
    closes one writer per file per relationship type.  Output identity
    is ``{source_file_key}.parquet``, independent of worker_id.

    Returns ``{"worker_id": int, "results": {source_key: {rt: row_count}}}``.
    """
    worker_id, source_files, entity_type, rel_types_list, batch_size, type_completed_keys, output_base = args
    rel_types = frozenset(rel_types_list)

    log.info("[w%02d] processing %d source files", worker_id, len(source_files))

    # Cache globals for loop performance
    _extract = _extract_one_source_file
    _staging_dir = STAGING_DIR
    _tck = type_completed_keys

    all_results: dict[str, dict[str, int]] = {}
    for source_file in source_files:
        source_key = _source_file_key(source_file)

        # Skip types already completed for this source key
        pending_types = frozenset(
            rt for rt in rel_types
            if source_key not in _tck.get(rt, set())
        )
        if not pending_types:
            all_results[source_key] = {}
            continue

        file_results = _extract(
            source_file,
            entity_type,
            pending_types,
            batch_size,
            output_base,
            _staging_dir,
        )
        all_results[source_key] = {
            rt: info["row_count"] for rt, info in file_results.items()
        }

    return {
        "worker_id": worker_id,
        "results": all_results,
    }


# ── Relationship type helpers ───────────────────────────────────────────


@lru_cache(maxsize=None)
def _entity_schema(entity: str) -> EntitySchema:
    # Probe the entity-scoped source directory, not the whole snapshot root —
    # otherwise the probe mixes records from every entity and derives a schema
    # from whichever entity sorts first.
    return get_entity_schema(entity, source_dir=SNAPSHOT_DIR / entity)


def _extract_entity_relationships(
    record: dict, schema: EntitySchema, wanted: frozenset[str] | None = None,
) -> dict[str, list[dict]]:
    return extract_relationships(record, schema, wanted=wanted)


def _entity_rel_types(entity: str) -> frozenset[str]:
    return _entity_schema(entity).rel_type_names()


def _order_rel_types(rel_types: frozenset[str]) -> list[str]:
    return sorted(rel_types)

# ── Main conversion ────────────────────────────────────────────────────


def convert_relationships(
    entity_type: str,
    *,
    force: bool = False,
    exclude: frozenset[str] | None = None,
    include_inferred: bool = True,
    workers: int | None = None,
    batch_size: int | None = None,
    slice_index: int | None = None,
    slice_total: int | None = None,
    output_dir: Path | None = None,
    verify: bool = False,
) -> dict[str, int]:
    """Convert nested JSONL arrays to relationship Parquet tables.

    **Deterministic output identity.**  Each source file produces exactly
    one output shard per relationship type, named ``{source_key}.parquet``
    where *source_key* is derived from the file's path relative to
    ``SNAPSHOT_DIR``.  Worker count only affects scheduling — the same
    source file always produces the same output shard.

    **Incremental & resumable.**  Per-unit provenance records track
    completion status for each source file.  On restart, only source
    files without a ``"complete"`` unit record are reprocessed.

    **Manifest drift detection.**  When a type is fully complete, the
    entity manifest is snapshotted.  On the next run, the current
    manifest is compared against the snapshot — changed or removed files
    trigger targeted re-extraction of only affected units.

    Args:
        entity_type: The source entity type (e.g. "works", "authors").
        force: If True, re-extract all units regardless of provenance.
        exclude: Relationship type names to skip entirely.
        include_inferred: If False, skip inferred relationship types.
        workers: Number of parallel worker processes per type.
        batch_size: Rows per parquet row-group flush.
        output_dir: Directory to write parquet output to. Defaults to
            ``SNAPSHOT_DIR`` so that ``rt_dir(output_dir, rt)`` resolves
            to the same path as ``rt_dir(SNAPSHOT_DIR, rt)``.
        verify: If True, validate every candidate completed shard's parquet
            footer (via ``validate_shard``) before treating it as complete.
            Unreadable shards are deleted so they re-extract. When False
            (default), completion is decided by filename only — the fast
            path that does not open any shard.

    Returns:
        Mapping of relationship type name to number of rows written.
    """
    _output_dir = output_dir if output_dir is not None else SNAPSHOT_DIR

    # Select the completion-check strategy once. verify=True opens and
    # validates every shard footer (slow); verify=False uses the
    # filename-only fast path.
    completed_source_keys = (
        _completed_source_keys_verified if verify else _completed_source_keys
    )

    all_rel_types = _entity_rel_types(entity_type)

    # Apply exclusions
    effective_exclude = frozenset(exclude or ())
    if effective_exclude:
        skipped = all_rel_types & effective_exclude
        if skipped:
            log.info(
                "%s: excluding relationship types: %s",
                entity_type, ", ".join(sorted(skipped)),
            )
    rel_types = all_rel_types - effective_exclude
    if not rel_types:
        log.info("%s: all relationship types excluded, nothing to do", entity_type)
        return {}

    ordered_types = _order_rel_types(rel_types)

    source_files = iter_source_files(entity_type)
    if not source_files:
        log.warning("%s: no source files found", entity_type)
        return {}

    # Apply distributed slicing
    if slice_index is not None and slice_total is not None:
        source_files = [
            f for i, f in enumerate(source_files)
            if i % slice_total == slice_index
        ]
        log.info(
            "Slice %d/%d: %d source files assigned",
            slice_index, slice_total, len(source_files),
        )

    if workers is not None:
        n_workers = workers
    else:
        n_workers = min(_auto_workers(), len(source_files))
    n_workers = max(1, n_workers)
    effective_batch_size = batch_size if batch_size is not None else _BATCH_SIZE

    # ── Classify types: skip / drift-rebuild / incremental / full ───
    types_to_run: list[str] = []

    for rt in ordered_types:
        if force:
            types_to_run.append(rt)
            continue

        _rt_dir = rt_dir(_output_dir, rt)

        # Check which source files have valid parquet shards on disk
        all_completed_keys = completed_source_keys(_rt_dir)

        # When running with --slice-index, source_files is a subset of
        # the full set.  Only count completed shards that belong to this
        # slice — shards completed by other slices or by a previous full
        # run must not inflate the count.
        source_file_keys = {_source_file_key(f) for f in source_files}
        completed_keys = all_completed_keys & source_file_keys
        n_source = len(source_files)
        n_complete = len(completed_keys)

        # Check for aggregate provenance + manifest drift
        provenance_path = _rt_dir / "_provenance.json"
        if provenance_path.exists() and n_complete == n_source:
            # Fully complete — check manifest drift
            current_manifest = _load_entity_manifest(entity_type)
            previous_manifest = _load_manifest_snapshot(_rt_dir)
            drift = _detect_manifest_drift(current_manifest, previous_manifest)

            if drift:
                added = sum(1 for v in drift.values() if v == "added")
                removed = sum(1 for v in drift.values() if v == "removed")
                changed = sum(1 for v in drift.values() if v == "changed")
                log.info(
                    "%s: manifest drift — %d added, %d removed, %d changed",
                    rt, added, removed, changed,
                )
                # Delete affected units + shards, keep healthy ones
                for file_rel, drift_type in drift.items():
                    if drift_type in ("changed", "removed"):
                        key = _source_file_key(
                            _manifest_key_to_path(file_rel)
                        )
                        # Remove shard
                        shard = _shard_path(_rt_dir, key)
                        if shard.exists():
                            shard.unlink()
                            log.debug("Deleted drifted shard: %s", shard.name)
                        # Remove unit provenance
                        unit_prov = _rt_dir / "_units" / f"{key}.json"
                        if unit_prov.exists():
                            unit_prov.unlink()
                types_to_run.append(rt)
                continue

            log.info("%s: complete (%d/%d units, no drift), skipping", rt, n_complete, n_source)
            continue

        # Check source_file_count vs unit count
        if n_complete == n_source:
            # All units complete but no aggregate provenance — just finalise
            log.info("%s: all %d/%d units complete, finalising", rt, n_complete, n_source)
            continue

        if completed_keys:
            pending = n_source - n_complete
            log.info(
                "%s: %d/%d units complete, processing %d pending",
                rt, n_complete, n_source, pending,
            )
        else:
            log.info("%s: no provenance, full extraction", rt)

        types_to_run.append(rt)

    if not types_to_run:
        log.info(
            "%s relationships: all types complete, nothing to do",
            entity_type,
        )
        return {}

    log.info(
        "%s relationships: %d types to process (%s), %d source files, %d workers",
        entity_type,
        len(types_to_run),
        " → ".join(types_to_run),
        len(source_files),
        n_workers,
    )

    # ── Single-pass extraction across all pending types ───────────────
    #
    # Instead of processing each type sequentially (which reads every
    # source file once per type), we process all pending types in a
    # single pass.  Each source file is read and decompressed once, then
    # all relationship types are extracted from it simultaneously.
    # For 8 remaining types this eliminates 7/8 of the I/O.
    #
    result: dict[str, int] = {}

    # HF-aware resume: query the HuggingFace dataset once at startup and
    # treat shards already present on HF as completed. Avoids re-extracting
    # files another machine (e.g. a parallel worker on different hardware)
    # has already uploaded.
    hf_completed = _hf_completed_source_keys(entity_type)

    # Build per-type pending-file sets and union of all files needed
    all_pending_types: list[str] = []
    type_completed_keys: dict[str, set[str]] = {}
    files_needed: set[Path] = set()
    types_already_done: list[str] = []

    for rt in types_to_run:
        _rt_dir = rt_dir(_output_dir, rt)
        create_output_dir(_rt_dir)

        local_completed = completed_source_keys(_rt_dir)
        completed = local_completed | hf_completed.get(rt, set())
        pending = _compute_pending_source_files(source_files, completed)

        if not pending:
            # All done — just finalise
            total = _finalise_type(
                _rt_dir,
                relationship_type=rt,
                source_entity=entity_type,
                source_file_count=len(source_files),
                verify=verify,
            )
            result[rt] = total
            types_already_done.append(rt)
            continue

        log.info(
            "%s: %s — %d files to process (%d already done)",
            entity_type, rt,
            len(pending), len(completed),
        )
        all_pending_types.append(rt)
        type_completed_keys[rt] = completed
        files_needed.update(pending)

    if types_already_done:
        log.info(
            "%s: %d types already complete (%s)",
            entity_type, len(types_already_done),
            ", ".join(types_already_done),
        )

    if not all_pending_types:
        log.info(
            "%s relationships: all types complete, nothing to do",
            entity_type,
        )
        return result

    files_to_process = sorted(files_needed)
    log.info(
        "%s: single-pass extraction — %d types, %d unique files (%s)",
        entity_type, len(all_pending_types), len(files_to_process),
        " + ".join(all_pending_types),
    )

    # Distribute files across workers — each worker processes all types
    actual_workers = min(n_workers, len(files_to_process))
    actual_workers = max(1, actual_workers)

    reset_skipped_files()

    if actual_workers == 1:
        worker_results = [
            _worker_process_files(
                (0, files_to_process, entity_type, all_pending_types, effective_batch_size, type_completed_keys, _output_dir),
            ),
        ]
    else:
        chunks: list[list[Path]] = [
            [files_to_process[j] for j in range(i, len(files_to_process), actual_workers)]
            for i in range(actual_workers)
        ]

        worker_args = [
            (i, chunk, entity_type, all_pending_types, effective_batch_size, type_completed_keys, _output_dir)
            for i, chunk in enumerate(chunks)
            if chunk
        ]

        with multiprocessing.get_context("fork").Pool(
            processes=len(worker_args),
        ) as pool:
            worker_results = pool.map(
                _worker_process_files, worker_args,
            )

    # Finalise each type independently
    for rt in all_pending_types:
        _rt_dir = rt_dir(_output_dir, rt)

        new_row_count = sum(
            rt_counts.get(rt, 0)
            for wr in worker_results
            for rt_counts in wr.get("results", {}).values()
        )

        total_row_count = _finalise_type(
            _rt_dir,
            relationship_type=rt,
            source_entity=entity_type,
            source_file_count=len(source_files),
            verify=verify,
        )
        result[rt] = total_row_count

        log.info(
            "%s: %s complete -- %d rows (%d resumed + %d new)",
            entity_type, rt, total_row_count,
            total_row_count - new_row_count,
            new_row_count,
        )

    log.info(
        "%s relationships: all types complete -- %s",
        entity_type,
        ", ".join(f"{rt}={cnt}" for rt, cnt in sorted(result.items())),
    )

    # Update README.md dataset_info for this entity automatically
    try:
        from sync.metadata import update_entity
        update_entity(entity_type)
    except Exception as exc:
        log.warning("Failed to update metadata for %s: %s", entity_type, exc)

    return result


# ── Migration: legacy worker shards → deterministic unit shards ─────────


def _validate_unit(
    rt_dir: Path,
    source_key: str,
    row_count: int,
    expected_hash: str | None = None,
) -> bool:
    """Validate a migrated unit by re-extracting and comparing.

    Returns True if the shard matches fresh extraction.  If *expected_hash*
    is given and the shard's hash differs, the unit is invalid.
    """
    shard = _shard_path(rt_dir, source_key)
    if not shard.exists():
        return False
    if expected_hash and _hash_file(shard) != expected_hash:
        return False
    try:
        pf = pq.ParquetFile(str(shard))
        actual_rows = pf.metadata.num_rows
        return actual_rows == row_count
    except Exception:
        return False


def migrate_relationship_type(
    relationship_type: str,
    entity_type: str,
    *,
    batch_size: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Migrate a single relationship type from legacy worker shards to
    deterministic per-source-file units.

    **Algorithm**

    1. Scan for legacy ``part-WW-NNNNN.parquet`` shards.
    2. If no legacy shards found, the type is already in unit layout (or empty).
    3. For each source file in the entity snapshot:
       a. Compute the deterministic ``source_key``.
       b. If a valid unit provenance already exists, skip.
       c. Re-extract the source file for this relationship type only.
       d. Write new unit shard + provenance.
    4. Validate all units: hash + row count.
    5. If any unit is invalid, delete it and re-extract from source.
    6. If all 100% of source files have valid units:
       - Delete all legacy ``part-WW-*.parquet`` and ``part-NNNNN.parquet`` shards.
       - Delete old ``_provenance.json``, ``_provenance_worker_*.json``,
         ``_lineage.json``, ``_manifest_snapshot.json``.
       - Write new aggregate provenance, lineage, manifest snapshot.

    Args:
        relationship_type: e.g. ``"work_sdgs"``.
        entity_type: e.g. ``"works"``.
        batch_size: Rows per parquet row-group.
        force: Re-extract all units even if provenance exists.
        dry_run: Report what would happen without writing.

    Returns:
        ``{"migrated": int, "re_extracted": int, "validated": int, "total_source_files": int}``
    """
    _rt_dir = rt_dir(SNAPSHOT_DIR, relationship_type)
    if not _rt_dir.exists():
        log.warning("%s: output dir does not exist, nothing to migrate", relationship_type)
        return {"migrated": 0, "re_extracted": 0, "validated": 0, "total_source_files": 0}

    # Detect legacy shards
    legacy_shards = sorted(
        list(_rt_dir.glob("part-??-*.parquet"))  # part-WW-NNNNN.parquet
        + list(_rt_dir.glob("part-0.parquet"))   # edge case
    )
    # Filter: only files matching the worker-ID pattern part-NN-NNNNN.parquet
    legacy_shards = [
        s for s in legacy_shards
        if len(s.name.split("-")) >= 3 or (len(s.name.split("-")) == 2 and s.name.split("-")[1].replace(".parquet", "").isdigit())
    ]

    if not legacy_shards:
        log.info("%s: no legacy worker shards found, already in unit layout", relationship_type)
        # Still check if we need to finalise
        units = _load_unit_provenances(_rt_dir)
        source_files = iter_source_files(entity_type)
        if units and len(units) == len(source_files):
            _finalise_type(
                _rt_dir,
                relationship_type=relationship_type,
                source_entity=entity_type,
                source_file_count=len(source_files),
            )
        return {
            "migrated": 0, "re_extracted": 0, "validated": len(units),
            "total_source_files": len(source_files),
        }

    log.info(
        "%s: found %d legacy worker shards, beginning migration",
        relationship_type, len(legacy_shards),
    )

    source_files = iter_source_files(entity_type)
    if not source_files:
        log.warning("%s: no source files found for entity %s", relationship_type, entity_type)
        return {"migrated": 0, "re_extracted": 0, "validated": 0, "total_source_files": 0}

    effective_batch_size = batch_size or _BATCH_SIZE
    all_rel_types = _entity_rel_types(entity_type)
    if relationship_type not in all_rel_types:
        log.error("%s is not a valid relationship type for %s", relationship_type, entity_type)
        return {"migrated": 0, "re_extracted": 0, "validated": 0, "total_source_files": len(source_files)}

    # Re-extract each source file as a deterministic unit
    migrated = 0
    re_extracted = 0
    validated = 0

    for source_file in source_files:
        source_key = _source_file_key(source_file)

        # Check existing unit provenance
        existing_units = _load_unit_provenances(_rt_dir)
        existing = existing_units.get(source_key)

        if existing and existing.get("status") in ("complete", "empty") and not force:
            # Validate existing unit
            if _validate_unit(_rt_dir, source_key, existing.get("row_count", 0),
                              existing.get("output_hash")):
                validated += 1
                continue
            else:
                log.warning("%s: existing unit %s failed validation, re-extracting",
                            relationship_type, source_key)

        if dry_run:
            log.info("[dry-run] Would re-extract %s for %s", source_key, relationship_type)
            re_extracted += 1
            continue

        # Re-extract this source file
        try:
            file_results = _extract_one_source_file(
                source_file,
                entity_type,
                frozenset({relationship_type}),
                effective_batch_size,
                SNAPSHOT_DIR,
                STAGING_DIR,
            )
            info = file_results.get(relationship_type, {})
            if info:
                re_extracted += 1
            else:
                log.warning("%s: no results for %s", relationship_type, source_key)
        except Exception as exc:
            log.error("%s: failed to extract %s: %s", relationship_type, source_key, exc)
            continue

    # Validate all units
    units = _load_unit_provenances(_rt_dir)
    valid_count = 0
    for key, unit in units.items():
        if _validate_unit(_rt_dir, key, unit.get("row_count", 0), unit.get("output_hash")):
            valid_count += 1
        else:
            log.warning("%s: unit %s failed post-extraction validation", relationship_type, key)

    log.info(
        "%s: %d/%d units valid after re-extraction",
        relationship_type, valid_count, len(source_files),
    )

    if valid_count < len(source_files):
        log.error(
            "%s: only %d/%d units valid — NOT deleting legacy shards. "
            "Re-run to re-extract missing units.",
            relationship_type, valid_count, len(source_files),
        )
        return {
            "migrated": migrated, "re_extracted": re_extracted,
            "validated": valid_count, "total_source_files": len(source_files),
        }

    # 100% valid — hard cutover: delete legacy artifacts
    if not dry_run:
        log.info("%s: 100%% valid units (%d/%d) — deleting legacy shards",
                 relationship_type, valid_count, len(source_files))
        for shard in legacy_shards:
            shard.unlink()
            log.debug("Deleted legacy shard: %s", shard.name)

        # Delete old provenance formats
        for stale in ["_provenance.json", "_lineage.json", "_manifest_snapshot.json"]:
            p = _rt_dir / stale
            if p.exists():
                p.unlink()
        for wp in sorted(_rt_dir.glob("_provenance_worker_*.json")):
            wp.unlink()

        # Write new aggregate provenance
        _finalise_type(
            _rt_dir,
            relationship_type=relationship_type,
            source_entity=entity_type,
            source_file_count=len(source_files),
        )

    return {
        "migrated": len(legacy_shards),
        "re_extracted": re_extracted,
        "validated": valid_count,
        "total_source_files": len(source_files),
    }


# ── CLI entry point ─────────────────────────────────────────────────────


def _sync_provenance_from_remote(remote_spec: str) -> None:
    """Rsync _units/ provenance metadata from a remote machine.

    *remote_spec* is an rsync-compatible source path, e.g.
    ``mini:/Volumes/ExAPFS/OpenAlex/parquet``.  Only the ``*/_units/``
    subdirectories (tiny JSON files) are transferred — the parquet data
    stays on the remote machine.  This lets the local provenance check
    see what the remote has already completed and skip those units.

    Runs ``find`` on the remote to locate ``_units/`` directories,
    avoiding a full directory tree scan over the network.
    """
    import subprocess

    # Parse host:path from remote_spec
    if ":" not in remote_spec:
        log.error("--sync-provenance must be host:path (e.g. mini:/data/parquet)")
        return
    host, remote_path = remote_spec.split(":", 1)

    # Find _units/ directories on the remote — scanning happens there,
    # not over the network.
    log.info("Scanning remote %s for _units/ directories ...", host)
    find_result = subprocess.run(
        ["ssh", host, "find", remote_path, "-type", "d", "-name", "_units"],
        capture_output=True, text=True, timeout=60,
    )
    if find_result.returncode != 0:
        log.warning("Remote find failed: %s", find_result.stderr.strip())
        return

    units_dirs = [d.strip() for d in find_result.stdout.strip().splitlines() if d.strip()]
    if not units_dirs:
        log.info("No _units/ directories found on remote")
        return

    log.info("Found %d _units/ directories on remote", len(units_dirs))

    # Build --files-from input with paths relative to remote_path
    files_from_lines = []
    for d in units_dirs:
        rel = d
        if rel.startswith(remote_path):
            rel = rel[len(remote_path):].lstrip("/")
        files_from_lines.append(rel)
    files_from = "\n".join(files_from_lines)

    local_parquet = str(SNAPSHOT_DIR)
    remote_src = f"{host}:{remote_path}/"
    log.info("Syncing %d _units/ dirs from %s", len(units_dirs), remote_src)
    result = subprocess.run(
        [
            "rsync", "-az", "--compress",
            "--relative",
            "--files-from=-", remote_src, local_parquet,
        ],
        input=files_from, text=True, timeout=300,
    )
    if result.returncode != 0:
        log.warning("Provenance sync failed (exit %d)", result.returncode)
    else:
        log.info("Provenance sync complete")


def main(entity: str | None = None, force: bool = False, workers: int | None = None, batch_size: int | None = None, slice_index: int | None = None, slice_total: int | None = None, sync_provenance: str | None = None, output_dir: str | None = None, verify: bool = True) -> None:
    """Extract relationship tables from JSONL snapshot."""
    import sync.common as _common

    if output_dir:
        _common.SNAPSHOT_DIR = Path(output_dir)
        # Also update this module's reference
        global SNAPSHOT_DIR
        SNAPSHOT_DIR = _common.SNAPSHOT_DIR
        _entity_schema.cache_clear()

    if sync_provenance:
        _sync_provenance_from_remote(sync_provenance)

    from sync.schema import _discover_entities
    types = [entity] if entity else _discover_entities(SNAPSHOT_DIR)
    if entity is None and len(types) > 1:
        # Process smallest entity first (by source partition count) so a bug
        # surfaces on a cheap entity within seconds rather than only after the
        # largest one finishes; the big poles (authors, works) run last. Order
        # does not affect correctness — each entity extracts independently.
        types = sorted(types, key=lambda et: len(iter_source_files(et)))
        log.info("Entity order (smallest-first): %s", ", ".join(types))
    for et in types:
        counts = convert_relationships(et, force=force, workers=workers, batch_size=batch_size, slice_index=slice_index, slice_total=slice_total, verify=verify)
        for rt, cnt in sorted(counts.items()):
            log.info("Extracted %s: %d rows", rt, cnt)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Extract OpenAlex relationship tables to Parquet",
    )
    subparsers = parser.add_subparsers(dest="command")

    # extract (default)
    ext = subparsers.add_parser("extract", help="Extract relationships from snapshot")
    ext.add_argument("--entity", type=str, default=None)
    ext.add_argument("--force", action="store_true")
    ext.add_argument("--workers", type=int, default=None)
    ext.add_argument("--batch-size", type=int, default=None)
    ext.add_argument("--slice-index", type=int, default=None,
                      help="0-based slice index for distributed processing")
    ext.add_argument("--slice-total", type=int, default=None,
                      help="Total number of slices for distributed processing")
    ext.add_argument("--sync-provenance", type=str, default=None,
                      metavar="REMOTE",
                      help="Rsync _units/ provenance from remote (e.g. mini:/path/to/parquet)")
    ext.add_argument("--output-dir", type=str, default=None,
                      metavar="DIR",
                      help="Override SNAPSHOT_DIR for output (e.g. snapshot data dir for nested layout)")

    # migrate
    mig = subparsers.add_parser(
        "migrate",
        help="Migrate legacy worker shards to deterministic unit layout",
    )
    mig.add_argument("--relationship-type", type=str, required=True,
                      help="Relationship type to migrate (e.g. work_sdgs)")
    mig.add_argument("--entity", type=str, required=True,
                      help="Source entity type (e.g. works)")
    mig.add_argument("--batch-size", type=int, default=None)
    mig.add_argument("--force", action="store_true",
                      help="Re-extract all units even if provenance exists")
    mig.add_argument("--dry-run", action="store_true",
                      help="Report what would happen without writing")

    args = parser.parse_args()

    if args.command == "migrate":
        result = migrate_relationship_type(
            args.relationship_type,
            args.entity,
            batch_size=args.batch_size,
            force=args.force,
            dry_run=args.dry_run,
        )
        log.info("Migration result: %s", result)
    else:
        # Default: extract (also handles no subcommand for backward compat)
        entity = getattr(args, 'entity', None)
        force = getattr(args, 'force', False)
        workers = getattr(args, 'workers', None)
        batch_size = getattr(args, 'batch_size', None)
        slice_index = getattr(args, 'slice_index', None)
        slice_total = getattr(args, 'slice_total', None)
        sync_provenance = getattr(args, 'sync_provenance', None)
        output_dir = getattr(args, 'output_dir', None)
        main(entity=entity, force=force, workers=workers, batch_size=batch_size, slice_index=slice_index, slice_total=slice_total, sync_provenance=sync_provenance, output_dir=output_dir)
