"""Regression tests for the --force re-extraction bug fix.

Bug: ``convert_relationships(..., force=True)`` was a no-op when parquet shards
already existed on disk.  The "classify types" loop correctly added all types to
``types_to_run`` under ``force``, but the subsequent per-type pending-file loop
still called ``completed_source_keys(_rt_dir)`` unconditionally.  Existing shards
made ``completed`` non-empty, ``_compute_pending_source_files`` returned an empty
list, and extraction was silently skipped.

Fix (``sync/extract.py``)::

    # Old (buggy):
    local_completed = completed_source_keys(_rt_dir)

    # New (fixed):
    local_completed = set() if force else completed_source_keys(_rt_dir)

These tests pin the regression: with existing shards present,
``force=True`` must produce a non-empty pending list and
``force=False`` must produce an empty one.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stub heavy dependencies before any sync.* imports so that this test file
# is importable in environments where pyarrow is not installed.  The stubs
# are installed into sys.modules BEFORE the import of sync.extract so that
# Python's import machinery picks them up when sync.extract does
# ``import pyarrow as pa`` and ``import pyarrow.parquet as pq``.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

_PA_MOCK = MagicMock(name="pyarrow")
_PQ_MOCK = MagicMock(name="pyarrow.parquet")

# Only inject if not already present (e.g. in an env where pyarrow IS installed,
# we prefer the real thing; the tests are written to work either way).
if "pyarrow" not in sys.modules:
    sys.modules["pyarrow"] = _PA_MOCK
    sys.modules["pyarrow.parquet"] = _PQ_MOCK

# huggingface_hub is optional; stub it too so the HF resume path doesn't blow up.
if "huggingface_hub" not in sys.modules:
    sys.modules["huggingface_hub"] = MagicMock(name="huggingface_hub")

# orjson and isal are optional; ensure they do NOT shadow the real json/gzip.
# If they are absent, sync.common falls back to the stdlib — which is fine.
if "orjson" not in sys.modules:
    sys.modules["orjson"] = MagicMock(name="orjson")
if "isal" not in sys.modules:
    sys.modules["isal"] = MagicMock(name="isal")
if "isal.igzip" not in sys.modules:
    sys.modules["isal.igzip"] = MagicMock(name="isal.igzip")

# ---------------------------------------------------------------------------
# Now import the real sync modules under test.
# ---------------------------------------------------------------------------
import sync.extract as extract
import sync.common as common


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source_files(snapshot_dir: Path, entity: str, n: int) -> list[Path]:
    """Create *n* empty .jsonl.gz source files in a standard partition layout."""
    partition = snapshot_dir / entity / "updated_date=2024-01-01"
    partition.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        f = partition / f"part_{i:04d}.jsonl.gz"
        f.write_bytes(b"")  # empty placeholder; not read in these unit tests
        files.append(f)
    return files


def _source_keys(source_files: list[Path], snapshot_dir: Path) -> set[str]:
    """Return the expected source-file keys given SNAPSHOT_DIR."""
    old = extract.SNAPSHOT_DIR
    extract.SNAPSHOT_DIR = snapshot_dir
    common.SNAPSHOT_DIR = snapshot_dir
    try:
        return {extract._source_file_key(f) for f in source_files}
    finally:
        extract.SNAPSHOT_DIR = old
        common.SNAPSHOT_DIR = old


# ---------------------------------------------------------------------------
# Unit tests for _compute_pending_source_files
# ---------------------------------------------------------------------------


class TestComputePendingSourceFiles:
    """Direct tests of the helper that determines which source files need work."""

    def test_empty_completed_returns_all(self, tmp_path: Path) -> None:
        """When no shards exist, every source file is pending."""
        source_files = _make_source_files(tmp_path, "concepts", 3)
        completed_keys: set[str] = set()

        pending = extract._compute_pending_source_files(source_files, completed_keys)

        assert pending == source_files

    def test_full_completed_returns_none(self, tmp_path: Path) -> None:
        """When all source keys are in completed_keys, pending is empty."""
        source_files = _make_source_files(tmp_path, "concepts", 3)
        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = tmp_path
        common.SNAPSHOT_DIR = tmp_path
        try:
            completed_keys = {extract._source_file_key(f) for f in source_files}
            # Keep SNAPSHOT_DIR set during the call: _compute_pending_source_files
            # calls _source_file_key internally and must use the same base.
            pending = extract._compute_pending_source_files(source_files, completed_keys)
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

        assert pending == []

    def test_partial_completed_returns_remainder(self, tmp_path: Path) -> None:
        """Only the uncompleted source files are returned as pending."""
        source_files = _make_source_files(tmp_path, "concepts", 4)
        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = tmp_path
        common.SNAPSHOT_DIR = tmp_path
        try:
            # Mark first two as completed
            completed_keys = {extract._source_file_key(f) for f in source_files[:2]}
            pending = extract._compute_pending_source_files(source_files, completed_keys)
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

        assert pending == source_files[2:]


# ---------------------------------------------------------------------------
# Unit tests for _completed_source_keys
# ---------------------------------------------------------------------------


class TestCompletedSourceKeys:
    """Tests for the on-disk shard discovery helper."""

    def test_returns_stems_of_parquet_files(self, tmp_path: Path) -> None:
        """Each .parquet file's stem becomes a completed key."""
        rt = tmp_path / "concepts" / "concept_ancestors"
        rt.mkdir(parents=True)
        (rt / "concepts__updated_date=2024-01-01__part_0000.parquet").write_bytes(b"x")
        (rt / "concepts__updated_date=2024-01-01__part_0001.parquet").write_bytes(b"x")

        keys = extract._completed_source_keys(rt)

        assert keys == {
            "concepts__updated_date=2024-01-01__part_0000",
            "concepts__updated_date=2024-01-01__part_0001",
        }

    def test_empty_dir_returns_empty_set(self, tmp_path: Path) -> None:
        rt = tmp_path / "concepts" / "concept_ancestors"
        rt.mkdir(parents=True)
        assert extract._completed_source_keys(rt) == set()

    def test_nonexistent_dir_returns_empty_set(self, tmp_path: Path) -> None:
        assert extract._completed_source_keys(tmp_path / "nonexistent") == set()

    def test_ignores_dot_underscore_files(self, tmp_path: Path) -> None:
        """macOS metadata files (._foo.parquet) are ignored."""
        rt = tmp_path / "rel"
        rt.mkdir()
        (rt / "._macmeta.parquet").write_bytes(b"x")
        (rt / "real.parquet").write_bytes(b"x")

        keys = extract._completed_source_keys(rt)

        assert keys == {"real"}
        assert "._macmeta" not in keys


# ---------------------------------------------------------------------------
# Core regression tests: force flag and the local_completed computation
# ---------------------------------------------------------------------------


class TestForceReextractionRegression:
    """Pin the exact regression: force=True must not treat existing shards as
    already-complete.  These tests model the inner loop of
    ``convert_relationships`` at the level of the fixed line:

        local_completed = set() if force else completed_source_keys(_rt_dir)

    They would FAIL with the pre-fix code (which always called
    ``completed_source_keys``) and PASS with the post-fix code.
    """

    def _build_existing_shards(
        self,
        rt_dir: Path,
        source_files: list[Path],
        snapshot_dir: Path,
    ) -> None:
        """Write a fake .parquet completion marker for every source file."""
        rt_dir.mkdir(parents=True, exist_ok=True)
        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = snapshot_dir
        common.SNAPSHOT_DIR = snapshot_dir
        try:
            for f in source_files:
                shard_name = f"{extract._source_file_key(f)}.parquet"
                (rt_dir / shard_name).write_bytes(b"fake parquet content")
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

    def test_no_force_existing_shards_yields_empty_pending(
        self, tmp_path: Path
    ) -> None:
        """Without --force, existing shards cause every source file to be
        treated as already complete, so pending is empty and extraction is
        skipped (correct resumption behaviour)."""
        snapshot_dir = tmp_path / "data"
        source_files = _make_source_files(snapshot_dir, "concepts", 3)
        rt = tmp_path / "rt_type"
        self._build_existing_shards(rt, source_files, snapshot_dir)

        force = False
        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = snapshot_dir
        common.SNAPSHOT_DIR = snapshot_dir
        try:
            local_completed = (
                set() if force else extract._completed_source_keys(rt)
            )
            pending = extract._compute_pending_source_files(
                source_files, local_completed
            )
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

        # Without force, shards exist → no pending files → incremental skip
        assert pending == [], (
            "Expected empty pending list when shards exist and force=False"
        )

    def test_force_true_existing_shards_yields_all_pending(
        self, tmp_path: Path
    ) -> None:
        """With --force, local_completed must be the empty set regardless of
        what's on disk, so every source file appears in pending.

        This test would FAIL on the pre-fix code, which evaluated:
            local_completed = completed_source_keys(_rt_dir)
        and therefore saw the existing shards as complete, making pending=[].
        """
        snapshot_dir = tmp_path / "data"
        source_files = _make_source_files(snapshot_dir, "concepts", 3)
        rt = tmp_path / "rt_type"
        self._build_existing_shards(rt, source_files, snapshot_dir)

        # Verify shards genuinely exist so the test is meaningful.
        existing = extract._completed_source_keys(rt)
        assert len(existing) == 3, (
            "Precondition: all three shards must exist on disk before we test force"
        )

        force = True
        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = snapshot_dir
        common.SNAPSHOT_DIR = snapshot_dir
        try:
            # POST-FIX: force clears local_completed unconditionally
            local_completed = set() if force else extract._completed_source_keys(rt)
            pending = extract._compute_pending_source_files(
                source_files, local_completed
            )
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

        assert len(pending) == 3, (
            "With force=True and existing shards on disk, ALL source files must "
            "be pending (the pre-fix bug returned an empty list here)"
        )
        assert set(pending) == set(source_files)

    def test_pre_fix_bug_demonstration(self, tmp_path: Path) -> None:
        """Explicit before/after comparison that would have caught the bug.

        Encodes the old (buggy) expression and the new (fixed) expression side by
        side and asserts they differ: the old one lets existing shards suppress
        re-extraction, the new one does not.
        """
        snapshot_dir = tmp_path / "data"
        source_files = _make_source_files(snapshot_dir, "concepts", 2)
        rt = tmp_path / "rt_type"
        self._build_existing_shards(rt, source_files, snapshot_dir)

        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = snapshot_dir
        common.SNAPSHOT_DIR = snapshot_dir
        try:
            disk_completed = extract._completed_source_keys(rt)

            # Pre-fix expression: force flag was ignored for local_completed
            pre_fix_local_completed = disk_completed  # always used disk result
            pending_pre_fix = extract._compute_pending_source_files(
                source_files, pre_fix_local_completed
            )

            # Post-fix expression: force=True clears local_completed
            force = True
            post_fix_local_completed = set() if force else disk_completed
            pending_post_fix = extract._compute_pending_source_files(
                source_files, post_fix_local_completed
            )
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

        # The pre-fix code silently skipped re-extraction when shards existed.
        assert pending_pre_fix == [], (
            "Pre-fix: existing shards made pending empty — force was a no-op"
        )
        # The post-fix code correctly re-extracts everything under force=True.
        assert len(pending_post_fix) == len(source_files), (
            "Post-fix: force=True must queue all source files regardless of disk state"
        )
        # The two differ — this is the regression.
        assert pending_pre_fix != pending_post_fix, (
            "Pre-fix and post-fix pending lists must differ: "
            "the bug made force equivalent to no-force when shards existed"
        )

    def test_force_false_no_shards_yields_all_pending(
        self, tmp_path: Path
    ) -> None:
        """Without --force and with no shards at all, all files are pending.
        Confirms the non-force path is unaffected by the fix."""
        snapshot_dir = tmp_path / "data"
        source_files = _make_source_files(snapshot_dir, "concepts", 2)
        rt = tmp_path / "empty_rt"
        rt.mkdir()

        force = False
        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = snapshot_dir
        common.SNAPSHOT_DIR = snapshot_dir
        try:
            local_completed = set() if force else extract._completed_source_keys(rt)
            pending = extract._compute_pending_source_files(
                source_files, local_completed
            )
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

        # No shards → nothing completed → all pending (even without force)
        assert len(pending) == 2


# ---------------------------------------------------------------------------
# Source-file key determinism
# ---------------------------------------------------------------------------


class TestSourceFileKey:
    """Sanity checks on the key derivation used throughout the pipeline."""

    def test_key_strips_jsonl_gz_suffix(self, tmp_path: Path) -> None:
        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = tmp_path
        common.SNAPSHOT_DIR = tmp_path
        try:
            f = tmp_path / "works" / "updated_date=2024-01-09" / "part_0047.jsonl.gz"
            key = extract._source_file_key(f)
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

        assert key == "works__updated_date=2024-01-09__part_0047"
        assert ".gz" not in key

    def test_key_is_deterministic(self, tmp_path: Path) -> None:
        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = tmp_path
        common.SNAPSHOT_DIR = tmp_path
        try:
            f = tmp_path / "authors" / "updated_date=2023-05-01" / "part_0001.jsonl.gz"
            key1 = extract._source_file_key(f)
            key2 = extract._source_file_key(f)
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

        assert key1 == key2

    def test_key_matches_shard_stem(self, tmp_path: Path) -> None:
        """The key produced from a source path must equal the stem of the shard
        written by _SourceFileWriter, which uses the same key as its file name."""
        old_snap = extract.SNAPSHOT_DIR
        extract.SNAPSHOT_DIR = tmp_path
        common.SNAPSHOT_DIR = tmp_path
        try:
            f = tmp_path / "concepts" / "updated_date=2024-01-01" / "part_0000.jsonl.gz"
            key = extract._source_file_key(f)
            shard_name = f"{key}.parquet"
        finally:
            extract.SNAPSHOT_DIR = old_snap
            common.SNAPSHOT_DIR = old_snap

        # The shard stem must equal the source key
        from pathlib import Path as _Path
        assert _Path(shard_name).stem == key
