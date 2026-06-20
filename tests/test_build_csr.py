"""Tests for sync.build_csr — CSR construction from relationship shards.

These pin the module's two load-bearing guarantees:

* **Correctness** — the dense-remapped CSR matches an independently computed
  reference, with nulls dropped, duplicate edges collapsed, and the sparse
  OpenAlex IDs mapped to a contiguous ``[0, n_nodes)`` range sorted by ID.
* **Determinism** — the same shards always produce a byte-identical ``.npz``
  (a documented invariant: explicit ``ORDER BY`` plus a sorted shard list).

build_csr depends on numpy, scipy, and duckdb, which are optional relative to
the core sync pipeline; the tests skip where those are unavailable.
"""

from __future__ import annotations

import hashlib

import pytest

# build_csr's deps (numpy/scipy/duckdb) are optional relative to the core sync
# pipeline; importorskip yields the modules and skips the file where any is
# absent. The first-party import then necessarily follows the guards (E402).
np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
sparse = pytest.importorskip("scipy.sparse")
pytest.importorskip("duckdb")

from sync import build_csr  # noqa: E402

REL = "work_referenced_works"
SRC_COL, TGT_COL = "work_id", "referenced_work_id"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_shard(shard_dir, edges, index=0):
    """Write ``edges`` (list of (src, tgt), None allowed) as one parquet shard."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    src = pa.array([e[0] for e in edges], type=pa.uint64())
    tgt = pa.array([e[1] for e in edges], type=pa.uint64())
    pq.write_table(
        pa.table({SRC_COL: src, TGT_COL: tgt}),
        shard_dir / f"part_{index:04d}.parquet",
    )


def _random_graph(seed, *, n_nodes=2_000, n_edges=8_000):
    """A reproducible sparse graph over OpenAlex-scale (uint64) IDs."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(1, 7_000_000_000), size=n_nodes, replace=False)
    return [
        (int(ids[i]), int(ids[j]))
        for i, j in zip(
            rng.integers(0, n_nodes, n_edges), rng.integers(0, n_nodes, n_edges)
        )
    ]


def _reference_csr(edges):
    """Build the expected CSR the long way, independent of build_csr's path.

    Mirrors the module contract: node set is every non-null src plus every
    non-null tgt; edges are the distinct rows with both endpoints present;
    dense index is the rank of the ID in sorted order.
    """
    nodes = {s for s, _ in edges if s is not None}
    nodes |= {t for _, t in edges if t is not None}
    sorted_ids = sorted(nodes)
    index = {nid: i for i, nid in enumerate(sorted_ids)}
    valid = sorted({(s, t) for s, t in edges if s is not None and t is not None})
    rows = [index[s] for s, _ in valid]
    cols = [index[t] for _, t in valid]
    n = len(sorted_ids)
    ref = sparse.csr_matrix(
        (np.ones(len(valid), dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )
    ref.sort_indices()
    return ref, np.array(sorted_ids, dtype=np.uint64)


def _build(tmp_path, edges=None, *, shards=None, force=True):
    """Build CSR for ``edges`` (one shard) or ``shards`` (one shard per list)."""
    groups = shards if shards is not None else [edges]
    parquet_dir = tmp_path / "data"
    shard_dir = build_csr.rt_dir(parquet_dir, REL)
    for i, grp in enumerate(groups):
        _write_shard(shard_dir, grp, i)
    out_dir = tmp_path / "csr"
    result = build_csr.build_csr(
        REL, parquet_dir=parquet_dir, output_dir=out_dir, force=force
    )
    return result, out_dir / f"{REL}.npz"


def test_csr_matches_independent_reference(tmp_path):
    # Sparse, OpenAlex-scale IDs; duplicate edge (10, 30); a null on each side.
    edges = [
        (10, 30), (10, 20), (40, 10), (10, 30),  # dup
        (30, 30), (20, 40), (40, 20),
        (50, None), (None, 60),  # endpoints seed nodes 50 & 60 but no edge
        (7_000_000_001, 10), (10, 7_000_000_001),
    ]
    _, npz = _build(tmp_path, edges)
    got = sparse.load_npz(npz)
    id_map = np.load(npz.with_suffix(".id_map.npy"))

    ref, ref_ids = _reference_csr(edges)
    assert got.shape == ref.shape
    assert got.nnz == ref.nnz
    assert got.indices.dtype == np.int32
    np.testing.assert_array_equal(got.indptr, ref.indptr)
    np.testing.assert_array_equal(got.indices, ref.indices)
    np.testing.assert_array_equal(got.data, ref.data)
    np.testing.assert_array_equal(id_map, ref_ids)
    # {10,20,30,40,50,60,7000000001}: 50 and 60 are seeded by null-partnered
    # rows, so they are nodes with no incident edge.
    assert got.shape[0] == 7


def test_node_set_includes_null_partnered_endpoints(tmp_path):
    edges = [(10, 20), (30, None), (None, 40)]
    _, npz = _build(tmp_path, edges)
    got = sparse.load_npz(npz)
    id_map = np.load(npz.with_suffix(".id_map.npy"))
    # 10,20 from the real edge; 30 and 40 seeded by their null-partnered rows
    np.testing.assert_array_equal(id_map, np.array([10, 20, 30, 40], dtype=np.uint64))
    assert got.shape == (4, 4)
    assert got.nnz == 1  # only (10,20) is a real edge


def test_duplicate_edges_collapse(tmp_path):
    edges = [(1, 2)] * 5 + [(2, 1)]
    _, npz = _build(tmp_path, edges)
    got = sparse.load_npz(npz)
    assert got.nnz == 2  # (1,2) and (2,1), the four duplicates dropped


def test_empty_relationship_yields_empty_matrix(tmp_path):
    result, npz = _build(tmp_path, [])
    got = sparse.load_npz(npz)
    assert got.shape == (0, 0)
    assert got.nnz == 0
    assert result["n_edges"] == 0


def test_duplicate_edges_collapse_across_shards(tmp_path):
    # Production always reads many shards; an edge repeated in *different* shards
    # must dedup globally, not just within one shard.
    shards = [[(1, 2), (3, 4)], [(1, 2), (5, 6)]]
    _, npz = _build(tmp_path, shards=shards)
    got = sparse.load_npz(npz)
    ref, ref_ids = _reference_csr([e for shard in shards for e in shard])
    assert got.nnz == 3  # (1, 2) once, plus (3, 4) and (5, 6)
    np.testing.assert_array_equal(got.indptr, ref.indptr)
    np.testing.assert_array_equal(got.indices, ref.indices)
    np.testing.assert_array_equal(np.load(npz.with_suffix(".id_map.npy")), ref_ids)


def test_larger_graph_matches_reference(tmp_path):
    # Validate the remap against the independent reference at a scale where
    # dense-index ordering bugs would surface, not just on the ~7-node fixture.
    edges = _random_graph(20260620)
    _, npz = _build(tmp_path, edges)
    got = sparse.load_npz(npz)
    ref, ref_ids = _reference_csr(edges)
    assert got.shape == ref.shape
    assert got.nnz == ref.nnz
    np.testing.assert_array_equal(got.indptr, ref.indptr)
    np.testing.assert_array_equal(got.indices, ref.indices)
    np.testing.assert_array_equal(got.data, ref.data)
    np.testing.assert_array_equal(np.load(npz.with_suffix(".id_map.npy")), ref_ids)


def test_unchanged_inputs_skip_and_preserve_output(tmp_path):
    # Idempotency: a second build with force=False detects unchanged inputs via
    # the provenance fingerprint, skips, and leaves the .npz byte-identical.
    edges = [(10, 20), (20, 30), (10, 30)]
    result1, npz = _build(tmp_path, edges)
    assert result1["status"] == "built"
    before = _sha256(npz)
    result2, _ = _build(tmp_path, edges, force=False)
    assert result2["status"] == "skipped"
    assert _sha256(npz) == before


def test_output_is_byte_identical_across_runs(tmp_path):
    edges = _random_graph(20260620)
    _, npz_a = _build(tmp_path / "a", edges)
    _, npz_b = _build(tmp_path / "b", edges)
    assert _sha256(npz_a) == _sha256(npz_b)
    assert _sha256(npz_a.with_suffix(".id_map.npy")) == _sha256(
        npz_b.with_suffix(".id_map.npy")
    )
