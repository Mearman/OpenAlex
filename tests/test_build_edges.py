"""Tests for sync.build_csr.build_edge_list — the edge-list Parquet export.

Pins the artifact's contract: a deduplicated, sorted edge list in original
OpenAlex IDs, queryable directly in DuckDB. build_csr's deps (duckdb/pyarrow)
are optional relative to the core pipeline; the file skips where absent.
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
duckdb = pytest.importorskip("duckdb")

from sync import build_csr  # noqa: E402

REL = "work_referenced_works"
SRC_COL, TGT_COL = "work_id", "referenced_work_id"


def _write_shard(shard_dir, edges, index=0):
    """Write ``edges`` (list of (src, tgt), None allowed) as one parquet shard."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    src = pa.array([e[0] for e in edges], type=pa.uint64())
    tgt = pa.array([e[1] for e in edges], type=pa.uint64())
    pq.write_table(
        pa.table({SRC_COL: src, TGT_COL: tgt}),
        shard_dir / f"part_{index:04d}.parquet",
    )


def _build(tmp_path, edges=None, *, shards=None, force=True):
    """Export the by_src edge list for ``edges`` (one shard) or ``shards``."""
    groups = shards if shards is not None else [edges]
    parquet_dir = tmp_path / "data"
    shard_dir = build_csr.rt_dir(parquet_dir, REL)
    for i, grp in enumerate(groups):
        _write_shard(shard_dir, grp, i)
    out_dir = tmp_path / "csr"
    result = build_csr.build_edge_list(
        REL, parquet_dir=parquet_dir, output_dir=out_dir, force=force
    )
    return result, out_dir / f"{REL}__by_src.parquet"


def _read(path):
    """Return the edge list as a list of (src, tgt) tuples, file order."""
    tbl = pq.read_table(path)
    return list(zip(tbl.column("src").to_pylist(), tbl.column("tgt").to_pylist()))


def test_dedups_sorts_and_drops_nulls(tmp_path):
    edges = [
        (10, 30), (10, 20), (40, 10), (10, 30),  # (10, 30) duplicated
        (30, 30), (20, 40),
        (50, None), (None, 60),  # null endpoints — dropped entirely
    ]
    result, path = _build(tmp_path, edges)
    got = _read(path)
    expected = sorted({(s, t) for s, t in edges if s is not None and t is not None})
    assert got == expected            # deduplicated, nulls dropped
    assert got == sorted(got)         # sorted by (src, tgt) for zonemap pruning
    assert result["n_edges"] == len(expected)


def test_uses_original_ids_not_dense_indices(tmp_path):
    # Sparse, OpenAlex-scale IDs must survive verbatim — no dense remapping.
    edges = [(7_000_000_001, 12), (12, 7_000_000_001)]
    _, path = _build(tmp_path, edges)
    got = _read(path)
    assert got == [(12, 7_000_000_001), (7_000_000_001, 12)]


def test_dedups_across_shards(tmp_path):
    # The same edge in different shards must collapse globally.
    shards = [[(1, 2), (3, 4)], [(1, 2), (5, 6)]]
    _, path = _build(tmp_path, shards=shards)
    assert _read(path) == [(1, 2), (3, 4), (5, 6)]


def test_is_queryable_by_src_in_duckdb(tmp_path):
    edges = [(10, 20), (10, 30), (10, 40), (50, 60)]
    _, path = _build(tmp_path, edges)
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT tgt FROM read_parquet('{path}') WHERE src = 10 ORDER BY tgt"
    ).fetchall()
    assert [r[0] for r in rows] == [20, 30, 40]


def test_skips_unchanged_inputs(tmp_path):
    edges = [(10, 20), (20, 30)]
    result1, path = _build(tmp_path, edges)
    assert result1["status"] == "built"
    mtime = path.stat().st_mtime_ns
    result2, _ = _build(tmp_path, edges, force=False)
    assert result2["status"] == "skipped"
    assert path.stat().st_mtime_ns == mtime  # not rewritten


def test_empty_relationship_yields_empty_edge_list(tmp_path):
    result, path = _build(tmp_path, [])
    assert result["n_edges"] == 0
    assert _read(path) == []
