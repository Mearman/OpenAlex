"""Integration guard: force must re-extract through the REAL convert_relationships.

Unlike the unit tests in test_force_reextraction.py (which re-implement the fixed
expression), this drives the actual pipeline against a tiny synthetic snapshot and
observes whether extraction is dispatched. Reverting the
`local_completed = set() if force else ...` fix makes test_force_reextracts FAIL
(force would skip dispatch when shards already exist).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import sync.common as common
import sync.extract as extract


def _write_source(snap_data: Path, entity: str, records: list[dict]) -> None:
    d = snap_data / entity / "updated_date=2020-01-01"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "part_0000.jsonl.gz", "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def tiny_snapshot(tmp_path, monkeypatch):
    data = tmp_path / "data"
    recs = [
        {"id": f"https://openalex.org/languages/{c}", "display_name": n}
        for c, n in [("en", "English"), ("fr", "French"), ("de", "German")]
    ]
    _write_source(data, "languages", recs)
    monkeypatch.setattr(common, "SNAPSHOT_DIR", data)
    monkeypatch.setattr(extract, "SNAPSHOT_DIR", data)
    monkeypatch.setenv("OPENALEX_SCHEMA_NOCACHE", "1")
    extract._entity_schema.cache_clear()
    extract._entity_arrow_schemas.cache_clear()
    yield data
    extract._entity_schema.cache_clear()
    extract._entity_arrow_schemas.cache_clear()


def _spy_extraction(monkeypatch):
    """Record the source-file lists passed to the real extraction dispatch."""
    dispatched: list[list] = []
    real = extract._worker_process_files

    def spy(args):
        # args = (worker_id, source_files, entity, rel_types, batch, tck, out)
        dispatched.append(list(args[1]))
        return real(args)

    monkeypatch.setattr(extract, "_worker_process_files", spy)
    return dispatched


def test_force_reextracts_via_real_pipeline(tiny_snapshot, monkeypatch):
    from sync.extract import convert_relationships

    # First run produces shards on disk.
    convert_relationships("languages", force=False, workers=1)

    # force=True must dispatch extraction again despite the shards existing.
    dispatched = _spy_extraction(monkeypatch)
    convert_relationships("languages", force=True, workers=1)
    assert dispatched and any(files for files in dispatched), (
        "force must re-dispatch extraction even though shards exist. Pre-fix bug: "
        "existing shards were counted as complete, so force skipped silently."
    )


def test_no_force_skips_existing_shards(tiny_snapshot, monkeypatch):
    from sync.extract import convert_relationships

    convert_relationships("languages", force=False, workers=1)
    dispatched = _spy_extraction(monkeypatch)
    convert_relationships("languages", force=False, workers=1)
    assert not any(files for files in dispatched), (
        "without force, existing shards must skip extraction (resume path)"
    )
