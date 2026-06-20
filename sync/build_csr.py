"""Build Compressed Sparse Row matrices from relationship Parquet tables.

Reads per-source-file parquet shards produced by ``sync.extract`` and
builds a single CSR ``.npz`` file per relationship type using DuckDB
for fast aggregation. Output files are written to ``OPENALEX_CSR_DIR``
(defaults to ``<SYNC_ROOT>/csr/``).

Design constraints
------------------
**Deterministic**: explicit ``ORDER BY`` in DuckDB queries; sorted
shard list.  Byte-identical output given the same input parquets.

**Idempotent**: provenance ``.json`` records input shard hashes
(SHA-256 of sorted filenames + aggregate row count).  Skips
relationship types whose inputs are unchanged unless ``--force``.

**Atomic**: output written to a temp file via ``tempfile.mkstemp`` and
replaced with ``os.replace`` (atomic on POSIX).  Orphan temp files
from interrupted runs are cleaned up at startup.

**Memory-bounded**: DuckDB writes intermediate results (unique IDs,
deduplicated edges) to temporary parquet files, which are read back in
chunks via pyarrow.  This avoids materialising the full edge set in
Python — ``work_referenced_works`` alone has ~3B deduplicated edges
(~48 GB as raw uint64 pairs), far exceeding what ``fetchall()`` can
handle.  Peak Python memory is bounded by the batch size plus the CSR
index arrays.

**Platform-independent**: env vars ``OPENALEX_PARQUET_DIR`` and
``OPENALEX_CSR_DIR`` control paths.  DuckDB ``SET memory_limit``
honours the host.  No GPU, no MPI, no platform branching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import sparse

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore[assignment]

from sync.common import SNAPSHOT_DIR, rt_dir

log = logging.getLogger("sync.build_csr")

# ── Configuration ───────────────────────────────────────────────────────

_CSR_DIR_ENV = "OPENALEX_CSR_DIR"
_PARQUET_DIR_ENV = "OPENALEX_PARQUET_DIR"
_MEMORY_LIMIT_ENV = "OPENALEX_CSR_MEMORY_LIMIT"


def _resolve_csr_dir() -> Path:
    """Resolve CSR output directory from env or default."""
    env = os.environ.get(_CSR_DIR_ENV)
    if env:
        return Path(env)
    return SNAPSHOT_DIR.parent / "csr"


def _resolve_parquet_dir() -> Path:
    """Resolve parquet source directory from env or default."""
    env = os.environ.get(_PARQUET_DIR_ENV)
    if env:
        return Path(env)
    return SNAPSHOT_DIR / "data"


# Relationship types that produce directed CSR matrices.
# Each entry maps the relationship type name to (src_col, tgt_col).
# For bidirectional relationships (e.g. citations), both directions
# are listed as separate types.
_CSR_RELATIONSHIP_TYPES: dict[str, tuple[str, str]] = {
    # Works
    "work_referenced_works": ("work_id", "referenced_work_id"),
    "work_authorships": ("work_id", "authorship_id"),
    "work_topics": ("work_id", "topic_id"),
    "work_concepts": ("work_id", "concept_id"),
    "work_related": ("work_id", "related_work_id"),
    "work_funders": ("work_id", "funder_id"),
    # Institutions
    "institution_repositories": ("institution_id", "repositorie_id"),
    "institution_roles": ("institution_id", "role_entity_id"),
    # Publishers
    "publisher_lineage": ("publisher_id", "lineage_id"),
    "publisher_roles": ("publisher_id", "role_entity_id"),
    # Funders
    "funder_roles": ("funder_id", "role_entity_id"),
}


# ── Provenance ──────────────────────────────────────────────────────────


def _input_fingerprint(parquet_files: list[Path]) -> str:
    """Deterministic fingerprint of input parquet files.

    Uses SHA-256 of the sorted filenames (not content) so the check
    is fast even with thousands of shards.  This detects new/removed
    files; content changes within an existing file of the same name
    are not caught (the sync pipeline handles that independently).
    """
    names = sorted(f.name for f in parquet_files)
    h = hashlib.sha256("\n".join(names).encode())
    return h.hexdigest()[:16]


def _load_provenance(csr_path: Path) -> dict | None:
    """Load existing CSR provenance, or None if absent."""
    prov_path = csr_path.with_suffix(".provenance.json")
    if not prov_path.exists():
        return None
    try:
        with open(prov_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_provenance(
    csr_path: Path,
    *,
    rel_type: str,
    n_edges: int,
    n_nodes: int,
    shard_count: int,
    input_fingerprint: str,
) -> None:
    """Write provenance metadata alongside the CSR file."""
    prov = {
        "rel_type": rel_type,
        "n_edges": n_edges,
        "n_nodes": n_nodes,
        "shard_count": shard_count,
        "input_fingerprint": input_fingerprint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    prov_path = csr_path.with_suffix(".provenance.json")
    # Atomic write
    fd, tmp = tempfile.mkstemp(
        dir=csr_path.parent, prefix=".prov-", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(prov, f, indent=2)
        os.replace(tmp, prov_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Orphan cleanup ──────────────────────────────────────────────────────


def _clean_orphan_temps(output_dir: Path) -> None:
    """Remove leftover temp files from interrupted runs."""
    if not output_dir.exists():
        return
    for p in output_dir.iterdir():
        name = p.name
        if (name.startswith(".tmp-") or name.startswith(".prov-")
                or name.endswith(".tmp_ids.parquet")
                or name.endswith(".tmp_edges.parquet")):
            try:
                p.unlink()
                log.debug("Cleaned orphan temp: %s", name)
            except OSError:
                pass


# ── CSR build ───────────────────────────────────────────────────────────


def _find_parquet_shards(
    parquet_dir: Path, rel_type: str
) -> list[Path]:
    """Find all parquet shards for a relationship type.

    The sync pipeline writes per-source-file shards to
    ``<parquet_dir>/<entity>/<subtable>/*.parquet``.  The nested path
    is derived from the relationship type name via ``rt_dir``.
    """
    shard_dir = rt_dir(parquet_dir, rel_type)
    if not shard_dir.exists():
        return []
    return sorted(shard_dir.glob("*.parquet"))


def _build_csr_duckdb(
    parquet_files: list[Path],
    src_col: str,
    tgt_col: str,
    *,
    memory_limit: str | None = None,
    output_path: Path,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build a CSR matrix from parquet shards using DuckDB + pyarrow.

    Returns (csr_matrix, original_ids) where ``original_ids`` is a sorted
    numpy uint64 array whose index is the dense node index and whose value
    is the original OpenAlex ID.

    OpenAlex IDs are sparse (up to ~7B) so we remap them to a dense
    contiguous range [0, n_unique_nodes) to keep the indptr array
    small.  The remapping is deterministic: sorted by original ID.

    DuckDB does the whole remap.  It builds a dense ``id -> idx`` dimension
    table (``row_number() OVER (ORDER BY id)``), joins the deduplicated
    edges against it twice, and writes the already-dense, already-sorted
    ``(src_idx, tgt_idx)`` pairs to a temporary parquet file.  Python then
    streams that file straight into the preallocated CSR arrays — no
    per-batch binary search over the id array, which on the ~3B-edge
    work_referenced_works would dominate the build: each such lookup misses
    cache on the multi-GB sorted id array.  Python holds the id array (for
    the id map), ``indptr``, ``indices`` and one batch; the remap join and
    its spill stay inside DuckDB, bounded by ``memory_limit``.
    """
    if duckdb is None:
        raise ImportError("duckdb is required for CSR building")
    if pq is None:
        raise ImportError("pyarrow is required for CSR building")

    if not parquet_files:
        return sparse.csr_matrix((0, 0), dtype=np.float64), np.array([], dtype=np.uint64)

    shard_dir = parquet_files[0].parent
    glob_pattern = str(shard_dir / "*.parquet")
    tmp_ids = output_path.with_suffix(".tmp_ids.parquet")
    tmp_edges = output_path.with_suffix(".tmp_edges.parquet")

    try:
        con = duckdb.connect(":memory:")
        try:
            # Give the dim-table window, the joins and the final sort a spill
            # target so they can exceed RAM — essential under a tight
            # memory_limit, since an in-memory connection won't otherwise
            # spill.  output_path.parent is the (already-created) CSR dir.
            con.execute(f"SET temp_directory='{output_path.parent}'")
            if memory_limit:
                con.execute(f"SET memory_limit='{memory_limit}'")

            # Step 1: Build the dense id -> idx dimension table.  The dense
            # index is the rank of the ID in sorted order, so the mapping is
            # deterministic and id-sorted, exactly as the indptr layout needs.
            # DuckDB spills to disk if the set exceeds memory_limit.
            log.info("Collecting unique node IDs...")
            con.execute(f"""
                CREATE TABLE ids AS
                SELECT id, (row_number() OVER (ORDER BY id) - 1) AS idx
                FROM (
                    SELECT CAST("{src_col}" AS UBIGINT) AS id
                    FROM read_parquet('{glob_pattern}')
                    WHERE "{src_col}" IS NOT NULL
                    UNION
                    SELECT CAST("{tgt_col}" AS UBIGINT) AS id
                    FROM read_parquet('{glob_pattern}')
                    WHERE "{tgt_col}" IS NOT NULL
                )
            """)
            n_nodes = con.execute("SELECT COUNT(*) FROM ids").fetchone()[0]
            log.info("Unique nodes: %d", n_nodes)

            # Emit the id map in dense order (idx is monotone in id, so
            # ORDER BY idx == ORDER BY id): original_ids[dense_idx] = id.
            con.execute(
                f"COPY (SELECT id FROM ids ORDER BY idx) "
                f"TO '{tmp_ids}' (FORMAT PARQUET)"
            )

            # Step 2: Deduplicate edges, remap both endpoints to dense
            # indices via the dimension table, and sort by (src_idx, tgt_idx).
            # Because idx is monotone in id, this order equals the (src, tgt)
            # id order — the CSR row/column ordering the indptr build assumes.
            log.info("Deduplicating, remapping and sorting edges...")
            con.execute(f"""
                COPY (
                    SELECT s.idx AS src_idx, t.idx AS tgt_idx
                    FROM (
                        SELECT DISTINCT
                            CAST("{src_col}" AS UBIGINT) AS src,
                            CAST("{tgt_col}" AS UBIGINT) AS tgt
                        FROM read_parquet('{glob_pattern}')
                        WHERE "{src_col}" IS NOT NULL
                          AND "{tgt_col}" IS NOT NULL
                    ) e
                    JOIN ids s ON e.src = s.id
                    JOIN ids t ON e.tgt = t.id
                    ORDER BY src_idx, tgt_idx
                ) TO '{tmp_edges}' (FORMAT PARQUET)
            """)
            n_edges = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{tmp_edges}')"
            ).fetchone()[0]
            log.info("Deduplicated edges: %d", n_edges)

        finally:
            con.close()

        if n_edges == 0:
            return sparse.csr_matrix((0, 0), dtype=np.float64), np.array([], dtype=np.uint64)

        # Step 3: Read the dense-ordered IDs for the returned id map:
        # original_ids[dense_idx] = original OpenAlex ID.  The remap itself
        # already happened in DuckDB, so this array is only the output
        # mapping.  ~2 GB for 250M nodes (uint64), well within memory.
        original_ids = pq.read_table(
            str(tmp_ids), columns=["id"]
        ).column("id").to_numpy()
        # IDs read; delete temp file now since the finally block's
        # tmp_ids reference must stay valid for the error path.
        try:
            tmp_ids.unlink()
        except OSError:
            pass

        # Step 4: Stream the already-dense, already-sorted edges into the CSR
        # components.  The remap happened in DuckDB, so per batch this is just
        # a copy plus a bincount — no binary search.  Peak memory per batch:
        # ~40 MB (5M rows × 8 bytes).
        #
        # The CSR ``indices`` array is the full deduplicated tgt-index set
        # (~12 GB for work_referenced_works's ~3B edges), so it is
        # preallocated at its final size and filled slice-by-slice as the
        # batches stream in.  Collecting per-batch arrays in a list and
        # ``np.concatenate``-ing at the end would transiently hold both the
        # list and the joined array — doubling that 12 GB at the worst
        # possible moment.  The edge count is known exactly (``n_edges``),
        # so the single allocation is safe.
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        indices = np.empty(n_edges, dtype=np.int32)
        offset = 0
        batch_count = 0

        pf = pq.ParquetFile(str(tmp_edges))
        for batch in pf.iter_batches(batch_size=5_000_000):
            src_idx = batch.column("src_idx").to_numpy()
            tgt_idx = batch.column("tgt_idx").to_numpy().astype(np.int32)

            # Append this batch's column indices into the preallocated array.
            # Edges arrive in (src_idx, tgt_idx) order, so writing them in
            # batch order keeps ``indices`` grouped by source node — exactly
            # the CSR layout that ``indptr`` describes.
            indices[offset:offset + len(tgt_idx)] = tgt_idx
            offset += len(tgt_idx)

            # Count edges per source node for indptr construction.
            counts = np.bincount(src_idx, minlength=n_nodes)
            indptr[1:] += counts

            batch_count += 1
            log.debug("Processed batch %d (%d rows)", batch_count, len(src_idx))

        # All deduplicated edges must have been placed exactly once.
        assert offset == n_edges, f"placed {offset} indices, expected {n_edges}"

        # Build CSR indptr (cumulative sum of per-row edge counts).
        np.cumsum(indptr, out=indptr)

        data = np.ones(n_edges, dtype=np.float32)
        csr = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(n_nodes, n_nodes),
        )

        return csr, original_ids

    finally:
        # Clean up temp parquet files
        for tmp in [tmp_ids, tmp_edges]:
            if tmp is not None:
                try:
                    Path(tmp).unlink()
                except OSError:
                    pass


def build_csr(
    rel_type: str,
    *,
    parquet_dir: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
    memory_limit: str | None = None,
) -> dict:
    """Build a CSR matrix for a single relationship type.

    Returns a dict with build statistics.
    """
    _parquet_dir = parquet_dir or _resolve_parquet_dir()
    _output_dir = output_dir or _resolve_csr_dir()

    if rel_type not in _CSR_RELATIONSHIP_TYPES:
        return {"error": f"Unknown relationship type: {rel_type}"}

    src_col, tgt_col = _CSR_RELATIONSHIP_TYPES[rel_type]
    parquet_files = _find_parquet_shards(_parquet_dir, rel_type)

    if not parquet_files:
        return {"error": f"No parquet shards found for {rel_type}"}

    output_path = _output_dir / f"{rel_type}.npz"
    fingerprint = _input_fingerprint(parquet_files)

    # Idempotency check
    if not force and output_path.exists():
        existing = _load_provenance(output_path)
        if existing and existing.get("input_fingerprint") == fingerprint:
            log.info(
                "%s: up-to-date (%d shards, fingerprint %s), skipping",
                rel_type, len(parquet_files), fingerprint,
            )
            return {
                "rel_type": rel_type,
                "status": "skipped",
                "n_edges": existing.get("n_edges", 0),
                "n_nodes": existing.get("n_nodes", 0),
                "shard_count": len(parquet_files),
            }

    log.info(
        "%s: building CSR from %d shards (fingerprint %s)",
        rel_type, len(parquet_files), fingerprint,
    )
    t0 = time.time()

    _output_dir.mkdir(parents=True, exist_ok=True)

    csr, original_ids = _build_csr_duckdb(
        parquet_files, src_col, tgt_col,
        memory_limit=memory_limit,
        output_path=output_path,
    )

    n_edges = csr.nnz
    n_nodes = csr.shape[0]

    # Write CSR matrix (atomic: temp file then os.replace)
    fd, tmp_path = tempfile.mkstemp(
        dir=_output_dir, prefix=".tmp-", suffix=".npz"
    )
    try:
        os.close(fd)
        sparse.save_npz(tmp_path, csr)
        output_size = Path(tmp_path).stat().st_size
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Write ID mapping as numpy array (index = dense, value = original ID).
    # Uses .npy format — orders of magnitude faster than JSON for large
    # node sets (250M nodes → ~2 GB .npy vs ~5 GB JSON, and no Python
    # dict overhead).
    id_map_path = output_path.with_suffix(".id_map.npy")
    fd2, tmp_id = tempfile.mkstemp(
        dir=_output_dir, prefix=".tmp-id-", suffix=".npy"
    )
    try:
        os.close(fd2)
        np.save(tmp_id, original_ids)
        os.replace(tmp_id, id_map_path)
    except BaseException:
        try:
            os.unlink(tmp_id)
        except OSError:
            pass
        raise

    _write_provenance(
        output_path,
        rel_type=rel_type,
        n_edges=n_edges,
        n_nodes=n_nodes,
        shard_count=len(parquet_files),
        input_fingerprint=fingerprint,
    )

    elapsed = time.time() - t0
    size_gb = output_size / 1e9
    log.info(
        "%s: %d nodes, %d edges, %.2f GB, %.1fs",
        rel_type, n_nodes, n_edges, size_gb, elapsed,
    )

    return {
        "rel_type": rel_type,
        "status": "built",
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "shard_count": len(parquet_files),
        "output_path": str(output_path),
        "output_size_bytes": output_size,
        "elapsed_seconds": elapsed,
    }


def build_all_csr(
    *,
    parquet_dir: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
    memory_limit: str | None = None,
    rel_types: list[str] | None = None,
) -> list[dict]:
    """Build CSR matrices for all (or specified) relationship types."""
    _output_dir = output_dir or _resolve_csr_dir()
    _output_dir.mkdir(parents=True, exist_ok=True)

    _clean_orphan_temps(_output_dir)

    types = rel_types or list(_CSR_RELATIONSHIP_TYPES.keys())
    results: list[dict] = []

    for rel_type in types:
        result = build_csr(
            rel_type,
            parquet_dir=parquet_dir,
            output_dir=_output_dir,
            force=force,
            memory_limit=memory_limit,
        )
        results.append(result)
        if "error" in result:
            log.warning("%s: %s", rel_type, result["error"])

    built = sum(1 for r in results if r.get("status") == "built")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = sum(1 for r in results if "error" in r)
    log.info(
        "CSR build complete: %d built, %d skipped, %d failed",
        built, skipped, failed,
    )

    return results


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build CSR matrices from relationship Parquet tables",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Directory containing relationship parquet shards "
             f"(default: env {_PARQUET_DIR_ENV} or <SYNC_ROOT>/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output CSR files "
             f"(default: env {_CSR_DIR_ENV} or <SYNC_ROOT>/csr)",
    )
    parser.add_argument(
        "--rel-type",
        action="append",
        help="Relationship type to process (can specify multiple)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all known relationship types",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if output exists and inputs unchanged",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default=None,
        help="DuckDB memory limit (e.g. '32GB') "
             f"(default: env {_MEMORY_LIMIT_ENV} or none)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rel_types = args.rel_type
    if args.all or not rel_types:
        rel_types = None  # build_all_csr defaults to all known types

    memory = args.memory_limit or os.environ.get(_MEMORY_LIMIT_ENV)

    results = build_all_csr(
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        force=args.force,
        memory_limit=memory,
        rel_types=rel_types,
    )

    failed = sum(1 for r in results if "error" in r)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
