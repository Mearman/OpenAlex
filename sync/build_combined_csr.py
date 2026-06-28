#!/usr/bin/env python3
"""Build a combined CSR from multiple OpenAlex relationship types.

Extends build_csr.py to merge N relationship types into one unified CSR
with a single compact index space. DuckDB handles ID alignment naturally:
all unique node IDs across all relationship types are collected, sorted,
and remapped to [0, n_total_nodes).

Usage:
    python3 -m sync.build_combined_csr \
        --relationships work_referenced_works,work_authorships \
        --output combined_work_authors.npz

For the full structural graph:
    python3 -m sync.build_combined_csr \
        --relationships work_referenced_works,work_authorships,work_concepts,work_topics,work_funders,institution_roles,publisher_lineage \
        --output combined_full_structural.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from sync.common import SNAPSHOT_DIR, rt_dir

log = logging.getLogger("sync.build_combined_csr")

# Same mapping as build_csr.py
_RELATIONSHIP_TYPES: dict[str, tuple[str, str]] = {
    "work_referenced_works": ("work_id", "referenced_work_id"),
    "work_authorships": ("work_id", "authorship_id"),
    "work_topics": ("work_id", "topic_id"),
    "work_concepts": ("work_id", "concept_id"),
    "work_funders": ("work_id", "funder_id"),
    "institution_roles": ("institution_id", "role_entity_id"),
    "publisher_lineage": ("publisher_id", "lineage_id"),
    "funder_roles": ("funder_id", "role_entity_id"),
}


def build_combined_csr(
    rel_types: list[str],
    parquet_dir: Path,
    output_path: Path,
    memory_limit: str = "128G",
) -> None:
    """Build a single CSR merging multiple relationship types."""
    if duckdb is None:
        raise ImportError("duckdb required")
    if pq is None:
        raise ImportError("pyarrow required")

    # Resolve glob patterns for each relationship type
    rel_globs: list[tuple[str, str, str]] = []  # (rel_type, src_col, tgt_col, glob)
    for rt in rel_types:
        if rt not in _RELATIONSHIP_TYPES:
            raise ValueError(f"Unknown relationship type: {rt}")
        src_col, tgt_col = _RELATIONSHIP_TYPES[rt]
        shard_dir = rt_dir(parquet_dir, rt)
        if not shard_dir.exists():
            log.warning("Skipping %s: directory not found (%s)", rt, shard_dir)
            continue
        glob = str(shard_dir / "*.parquet")
        rel_globs.append((rt, src_col, tgt_col, glob))
        log.info("Including %s: src=%s tgt=%s glob=%s", rt, src_col, tgt_col, glob)

    if not rel_globs:
        raise RuntimeError("No valid relationship types found")

    tmp_ids = output_path.with_suffix(".tmp_ids.parquet")
    tmp_edges = output_path.with_suffix(".tmp_edges.parquet")

    try:
        con = duckdb.connect(":memory:")
        try:
            if memory_limit:
                con.execute(f"SET memory_limit='{memory_limit}'")

            # Step 1: Collect ALL unique node IDs across ALL relationship types.
            log.info("Collecting unique node IDs across %d relationship types...", len(rel_globs))
            id_unions = []
            for _rt, src_col, tgt_col, glob in rel_globs:
                id_unions.append(f"""
                    SELECT DISTINCT CAST("{src_col}" AS UBIGINT) AS id
                    FROM read_parquet('{glob}') WHERE "{src_col}" IS NOT NULL
                    UNION ALL
                    SELECT DISTINCT CAST("{tgt_col}" AS UBIGINT) AS id
                    FROM read_parquet('{glob}') WHERE "{tgt_col}" IS NOT NULL
                """)
            id_union_sql = "\n                    UNION ALL\n".join(id_unions)

            con.execute(f"""
                COPY (
                    WITH all_ids AS ({id_union_sql})
                    SELECT DISTINCT id FROM all_ids ORDER BY id
                ) TO '{tmp_ids}' (FORMAT PARQUET)
            """)
            n_nodes = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{tmp_ids}')"
            ).fetchone()[0]
            log.info("Total unique nodes across all relationship types: %d", n_nodes)

            # Step 2: Deduplicated edges from ALL relationship types, sorted.
            log.info("Deduplicating and sorting edges across all relationship types...")
            edge_unions = []
            for _rt, src_col, tgt_col, glob in rel_globs:
                edge_unions.append(f"""
                    SELECT DISTINCT
                        CAST("{src_col}" AS UBIGINT) AS src,
                        CAST("{tgt_col}" AS UBIGINT) AS tgt
                    FROM read_parquet('{glob}')
                    WHERE "{src_col}" IS NOT NULL AND "{tgt_col}" IS NOT NULL
                """)
            edge_union_sql = "\n                    UNION\n".join(edge_unions)

            con.execute(f"""
                COPY (
                    {edge_union_sql}
                    ORDER BY src, tgt
                ) TO '{tmp_edges}' (FORMAT PARQUET)
            """)
            n_edges = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{tmp_edges}')"
            ).fetchone()[0]
            log.info("Total deduplicated edges: %d", n_edges)

        finally:
            con.close()

        if n_edges == 0:
            log.warning("No edges found, writing empty CSR")
            csr = sparse.csr_matrix((0, 0), dtype=np.float64)
            sparse.save_npz(str(output_path), csr)
            return

        # Step 3: Read sorted unique IDs (the dense mapping).
        log.info("Reading sorted unique IDs (%d nodes)...", n_nodes)
        original_ids = pq.read_table(
            str(tmp_ids), columns=["id"]
        ).column("id").to_numpy()
        try:
            tmp_ids.unlink()
        except OSError:
            pass

        # Step 4: Read edges in chunks, remap to dense indices.
        log.info("Building CSR from %d edges...", n_edges)
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        indices_chunks: list[np.ndarray] = []

        batch_size = 5_000_000
        table = pq.ParquetFile(str(tmp_edges))
        batch_count = 0

        for batch in table.iter_batches(batch_size=batch_size, columns=["src", "tgt"]):
            src_orig = batch.column("src").to_numpy()
            tgt_orig = batch.column("tgt").to_numpy()

            # Binary search remap: original ID -> dense index
            src_dense = np.searchsorted(original_ids, src_orig)
            tgt_dense = np.searchsorted(original_ids, tgt_orig)

            # Count per-source for indptr
            np.add.at(indptr[1:], src_dense, 1)
            indices_chunks.append(tgt_dense.astype(np.int32))
            batch_count += 1
            if batch_count % 20 == 0:
                log.info("  Processed %d batches (%d edges)...",
                         batch_count, batch_count * batch_size)

        try:
            tmp_edges.unlink()
        except OSError:
            pass

        # Step 5: Cumulative sum indptr + concatenate indices.
        log.info("Finalising CSR (cumsum + sort per row)...")
        np.cumsum(indptr, out=indptr)
        indices = np.concatenate(indices_chunks)
        del indices_chunks

        # Sort indices within each row (CSR invariant)
        for i in range(n_nodes):
            lo = indptr[i]
            hi = indptr[i + 1]
            if hi > lo + 1:
                indices[lo:hi].sort()

        # Step 6: Save CSR
        csr = sparse.csr_matrix(
            (np.ones(n_edges, dtype=np.float32), indices, indptr),
            shape=(n_nodes, n_nodes),
        )
        log.info("Saving combined CSR to %s (%d nodes, %d edges)...",
                 output_path, n_nodes, n_edges)
        sparse.save_npz(str(output_path), csr)
        log.info("Done. Combined CSR: %d nodes, %d edges", n_nodes, n_edges)

    except BaseException:
        # Cleanup temp files on error
        for tmp in [tmp_ids, tmp_edges]:
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Build combined CSR from multiple OpenAlex relationship types")
    parser.add_argument(
        "--relationships", required=True,
        help="Comma-separated relationship types to merge (e.g. work_referenced_works,work_authorships)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .npz path"
    )
    parser.add_argument(
        "--parquet-dir",
        default=os.environ.get("OPENALEX_PARQUET_DIR", str(SNAPSHOT_DIR / "data")),
        help="Parquet source directory (default: OPENALEX_PARQUET_DIR or <SNAPSHOT>/data)"
    )
    parser.add_argument(
        "--memory-limit", default="128G",
        help="DuckDB memory limit (default: 128G)"
    )
    args = parser.parse_args()

    rel_types = [r.strip() for r in args.relationships.split(",")]
    parquet_dir = Path(args.parquet_dir)
    output_path = Path(args.output)

    log.info("Building combined CSR: %s -> %s", rel_types, output_path)
    build_combined_csr(rel_types, parquet_dir, output_path, args.memory_limit)


if __name__ == "__main__":
    main()
