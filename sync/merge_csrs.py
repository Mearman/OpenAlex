#!/usr/bin/env python3
"""Merge multiple per-relationship CSR matrices into one combined CSR.

This is the incremental build path for the full structural graph. The
previous approach (``build_combined_csr.py``) asked DuckDB to dedup and
sort all ~5B edges across 7 relationship types in a single SQL query,
which segfaulted even with ``--memory-limit 1000G`` and temp off-scratch.
DuckDB's external sort at the 5B-edge scale is the failure.

This script avoids that entirely. It consumes the **already-built**
per-relationship CSRs (``<rel>.npz`` + ``<rel>.id_map.npy`` produced by
``sync.build_csr``) and merges them in Python:

  1. Read each relationship's ``id_map`` (sorted uint64 OpenAlex IDs).
  2. Compute the union of all id_maps -> a single sorted ``global_ids``
     array, the combined compact index space ``[0, n_global_nodes)``.
  3. For each relationship CSR, remap its source/target dense indices
     (which index into that relationship's own id_map) into the global
     index space via the global id_map (binary search — order-preserving
     since global_ids is a sorted superset).
  4. Sum the remapped CSRs (scipy CSR + CSR, C-level merge of sorted
     rows) incrementally to bound peak memory.

No DuckDB, no 5B-edge SQL sort. The only sort is scipy's per-row CSR
merge, which is C code and incremental across the 7 inputs.

Memory profile on a 1.1 TB highmem node (worst case, full structural):
  - global_ids: ~617M uint64            ~5 GB
  - largest single CSR (refs, ~3B nnz): ~24 GB
  - running combined sum (~5B nnz):     ~40 GB
  - one remap intermediate at a time:   ~24 GB
Peak well under 200 GB.

Usage:
    python3 -m sync.merge_csrs \
        --relationships work_referenced_works,work_authorships,work_concepts,work_topics,work_funders,institution_roles,publisher_lineage \
        --output combined_full_structural.npz

Inputs (per relationship type ``<rt>``):
    <csr_dir>/<rt>.npz          the CSR matrix
    <csr_dir>/<rt>.id_map.npy   sorted uint64 OpenAlex IDs (dense -> original)

Output:
    <output>                    combined CSR (.npz)
    <output stem>.id_map.npy    global id map (optional, written alongside)
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

log = logging.getLogger("merge_csrs")


def _load_rel_csr(csr_dir: Path, rel_type: str) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Load a per-relationship CSR and its id_map.

    Returns (csr, id_map) where id_map is a sorted uint64 array:
    id_map[dense_idx] == original OpenAlex ID, and the CSR's
    indices/rows index into [0, len(id_map)).
    """
    csr_path = csr_dir / f"{rel_type}.npz"
    idmap_path = csr_dir / f"{rel_type}.id_map.npy"
    if not csr_path.exists():
        raise FileNotFoundError(f"CSR not found: {csr_path}")
    if not idmap_path.exists():
        raise FileNotFoundError(f"id_map not found: {idmap_path}")
    log.info("Loading %s CSR from %s", rel_type, csr_path)
    t0 = time.time()
    csr = sparse.load_npz(str(csr_path))
    csr = csr.tocsr()  # ensure canonical CSR
    id_map = np.load(str(idmap_path))
    log.info(
        "  %s: shape=%s nnz=%d id_map=%s (%.1fs)",
        rel_type, csr.shape, csr.nnz, id_map.shape, time.time() - t0,
    )
    if csr.shape[0] != id_map.shape[0] or csr.shape[1] != id_map.shape[0]:
        raise ValueError(
            f"{rel_type}: CSR shape {csr.shape} != id_map length {id_map.shape[0]}"
        )
    return csr, id_map


def _remap_to_global(
    csr: sparse.csr_matrix,
    local_id_map: np.ndarray,
    global_ids: np.ndarray,
) -> sparse.csr_matrix:
    """Remap a CSR's indices into the global compact index space.

    Each CSR row index i corresponds to local_id_map[i]; in the global
    space that node sits at searchsorted(global_ids, local_id_map[i]).
    The columns (indices array) likewise need remapping: indices[j] is a
    local dense index, so the original ID is local_id_map[indices[j]],
    and the global index is searchsorted(global_ids, that ID).

    Both remaps are order-preserving (global_ids is a sorted superset),
    so row ordering and within-row column ordering are preserved.

    Implementation (indptr-driven, no per-edge row expansion):
      - Remap the **columns** (indices array) directly: local dense idx
        -> original ID (gather via local_id_map) -> global idx (binary
        search). This is the only nnz-sized allocation needed.
      - Remap the **indptr** by counting edges per *global* row: the
        per-local-row degrees get scattered into a global-row histogram
        via np.add.at on the per-local-row global-row lookup, then
        cumsummed. No expansion to per-edge source rows.
    """
    n_global = global_ids.shape[0]
    n_edges = csr.nnz
    if n_edges == 0:
        return sparse.csr_matrix((n_global, n_global), dtype=csr.dtype)

    # Per-local-row -> global-row lookup (n_local entries, not nnz).
    global_row_of_local = np.searchsorted(global_ids, local_id_map)

    # Column remap: indices[j] (local dense) -> original ID -> global idx.
    edge_tgt_original = local_id_map[csr.indices]
    new_indices = np.searchsorted(global_ids, edge_tgt_original).astype(np.int32)
    del edge_tgt_original

    # Indptr remap: scatter per-local-row degrees into a global-row
    # histogram offset by one (so cumsum yields a 0-led indptr). Each
    # local row's degree is added at new_indptr[global_row+1].
    degrees = np.diff(csr.indptr).astype(np.int64)  # length n_local
    new_indptr = np.zeros(n_global + 1, dtype=np.int64)
    np.add.at(new_indptr, global_row_of_local + 1, degrees)
    np.cumsum(new_indptr, out=new_indptr)

    data = np.ones(n_edges, dtype=np.float32)
    remapped = sparse.csr_matrix(
        (data, new_indices, new_indptr),
        shape=(n_global, n_global),
    )
    return remapped


def merge_csrs(
    rel_types: list[str],
    csr_dir: Path,
    output_path: Path,
    *,
    write_id_map: bool = True,
) -> None:
    """Merge per-relationship CSRs into one combined CSR."""
    # ── Phase 1: load all id_maps, compute global node-id union ──────
    log.info("Phase 1: computing global node-id union across %d types", len(rel_types))
    t0 = time.time()
    id_maps: dict[str, np.ndarray] = {}
    for rt in rel_types:
        idmap_path = csr_dir / f"{rt}.id_map.npy"
        if not idmap_path.exists():
            raise FileNotFoundError(f"id_map not found for {rt}: {idmap_path}")
        id_maps[rt] = np.load(str(idmap_path))
        log.info("  %s id_map: %s", rt, id_maps[rt].shape)

    # Union of all sorted uint64 arrays. np.unique on the concatenation
    # gives the sorted, deduplicated global node set. Total input is
    # ~1.43B uint64 (~11 GB concat); np.unique returns ~617M (~5 GB).
    log.info("Concatenating + deduplicating global node IDs...")
    all_ids = np.concatenate(list(id_maps.values()))
    del id_maps
    global_ids = np.unique(all_ids)
    del all_ids
    n_global = global_ids.shape[0]
    log.info(
        "Global node union: %d nodes (computed in %.1fs)",
        n_global, time.time() - t0,
    )

    # ── Phase 2: load each CSR, remap, sum incrementally ─────────────
    # Process largest-first so the running accumulator (the expensive
    # object) is established early and later additions are cheap deltas.
    # We need sizes to order them; peek nnz via the npz without full load
    # by checking file size as a proxy is unreliable, so load metadata by
    # reading each CSR's indptr length cheaply is still a full load.
    # Instead, order by id_map length (correlates with nnz) descending.
    order = sorted(rel_types, key=lambda rt: -np.load(str(csr_dir / f"{rt}.id_map.npy"), mmap_mode="r").shape[0])
    log.info("Merge order (largest first): %s", order)

    combined: sparse.csr_matrix | None = None
    for idx, rt in enumerate(order, 1):
        log.info("Phase 2 [%d/%d]: %s", idx, len(order), rt)
        t1 = time.time()
        csr, local_id_map = _load_rel_csr(csr_dir, rt)
        t2 = time.time()
        remapped = _remap_to_global(csr, local_id_map, global_ids)
        t3 = time.time()
        # Free the per-relationship CSR + its id_map now that remap is done.
        del csr, local_id_map

        if combined is None:
            combined = remapped
        else:
            # CSR + CSR sum: scipy merges sorted rows in C, dedups
            # overlapping (src,tgt) pairs (e.g. a work->concept edge
            # present in two types collapses to one). dtype stays float.
            combined = combined + remapped
        del remapped
        log.info(
            "  %s merged (%.1fs load, %.1fs remap); combined nnz=%d",
            rt, t2 - t1, t3 - t2, combined.nnz,
        )

    assert combined is not None
    combined = combined.tocsr()
    # Structural graph: binary adjacency. Sum may produce values >1 if
    # scipy didn't fully dedup (it does for +, but be safe), so clip.
    combined.data[:] = 1.0
    log.info(
        "Combined CSR: %d nodes, %d edges", combined.shape[0], combined.nnz,
    )

    # ── Phase 3: write outputs ───────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing combined CSR to %s", output_path)
    sparse.save_npz(str(output_path), combined)

    if write_id_map:
        idmap_out = output_path.with_suffix(".id_map.npy")
        log.info("Writing global id_map to %s", idmap_out)
        np.save(str(idmap_out), global_ids)

    log.info(
        "Done. Combined CSR: %d nodes, %d edges -> %s",
        combined.shape[0], combined.nnz, output_path,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Merge per-relationship CSRs into one combined CSR",
    )
    parser.add_argument(
        "--relationships", required=True,
        help="Comma-separated relationship types to merge",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .npz path",
    )
    parser.add_argument(
        "--csr-dir",
        default=os.environ.get("OPENALEX_CSR_DIR",
                               os.environ.get("OPENALEX_SNAPSHOT_DIR",
                                              "/scratch/SCWF00070/b.abs217/openalex-snapshot") + "/csr"),
        help="Directory containing per-relationship <rt>.npz + <rt>.id_map.npy",
    )
    parser.add_argument(
        "--no-id-map", action="store_true",
        help="Do not write the global id_map alongside the output",
    )
    # Compatibility flags accepted from the build_combined_csr wrapper
    # but unused here: this script uses no DuckDB, so there is no temp
    # spill or memory limit to set. Recognised to avoid breaking the
    # existing run_csr_full.sh invocation when switched to this script.
    parser.add_argument(
        "--temp-dir", default=None,
        help="Ignored (no DuckDB); accepted for wrapper compatibility",
    )
    parser.add_argument(
        "--memory-limit", default=None,
        help="Ignored (no DuckDB); accepted for wrapper compatibility",
    )
    args = parser.parse_args()

    rel_types = [r.strip() for r in args.relationships.split(",") if r.strip()]
    csr_dir = Path(args.csr_dir)
    output_path = Path(args.output)

    log.info("Merging CSRs: %s -> %s (from %s)", rel_types, output_path, csr_dir)
    merge_csrs(
        rel_types, csr_dir, output_path,
        write_id_map=not args.no_id_map,
    )


if __name__ == "__main__":
    main()
