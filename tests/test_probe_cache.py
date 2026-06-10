"""Regression tests for the schema probe cache: version invalidation and
concurrency-safe writes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sync.schema as schema
from sync.schema import EntitySchema, FieldSchema


def _tiny_schema(entity: str = "widgets") -> EntitySchema:
    return EntitySchema(
        entity=entity,
        id_col="widget_id",
        id_path="id",
        id_type="int",
        fields=[
            FieldSchema(
                json_key="cited_by_count", pattern="scalar", rel_name="widget_main",
                scalar_cols=[{"col": "cited_by_count", "path": "cited_by_count", "type": "int"}],
            )
        ],
    )


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "probe-cache"
    monkeypatch.setattr(schema, "_PROBE_CACHE_DIR", d)
    return d


class TestVersionInvalidation:
    def test_roundtrip_same_version_loads(self, cache_dir):
        schema._store_probe_cache("widgets", "key0", _tiny_schema())
        loaded = schema._load_probe_cache("widgets", "key0")
        assert loaded is not None
        assert loaded.entity == "widgets"

    def test_stale_version_is_rejected(self, cache_dir, monkeypatch):
        # Written under the current version...
        schema._store_probe_cache("widgets", "key0", _tiny_schema())
        path = schema._probe_cache_path("widgets", "key0")
        assert path.is_file()
        # ...then the probe logic version moves on: the cache must be ignored,
        # not silently shadow the new behaviour (the cache_key is unchanged).
        monkeypatch.setattr(schema, "_SCHEMA_FILE_VERSION", schema._SCHEMA_FILE_VERSION + 1)
        assert schema._load_probe_cache("widgets", "key0") is None

    def test_cache_payload_records_version(self, cache_dir):
        schema._store_probe_cache("widgets", "key0", _tiny_schema())
        payload = json.loads(schema._probe_cache_path("widgets", "key0").read_text())
        assert payload["version"] == schema._SCHEMA_FILE_VERSION


class TestConcurrentWriteSafe:
    """A unique temp file per writer means a contended/blocked legacy shared
    ``<path>.tmp`` name can't break a write. Blocking that exact name (with a
    directory, so the old shared-tmp write would fail) deterministically
    distinguishes the fix from the racy original — no timing luck involved.
    """

    def test_write_schema_file_survives_blocked_shared_tmp(self, tmp_path, monkeypatch):
        sf = tmp_path / "openalex.schema.json"
        monkeypatch.setattr(schema, "_SCHEMA_FILE", sf)
        # Occupy the legacy shared temp name; the old code wrote here and would
        # raise (and it didn't catch the error, which is what aborted the job).
        (tmp_path / "openalex.schema.tmp").mkdir()
        schema._write_schema_file({"widgets": _tiny_schema().to_dict()})
        assert sf.is_file()
        payload = json.loads(sf.read_text())
        assert payload["version"] == schema._SCHEMA_FILE_VERSION
        assert "widgets" in payload["entities"]
        assert not list(tmp_path.glob(".openalex.schema.*.tmp")), "temp not cleaned up"

    def test_store_probe_cache_survives_blocked_shared_tmp(self, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = schema._probe_cache_path("widgets", "key0")
        # Block the legacy shared temp name (old: ``<path>.tmp``).
        Path(str(path) + ".tmp").mkdir()
        schema._store_probe_cache("widgets", "key0", _tiny_schema())
        # New code uses a unique temp, so the cache is written and loadable;
        # the old code would have failed to write it (and logged a warning).
        loaded = schema._load_probe_cache("widgets", "key0")
        assert loaded is not None and loaded.entity == "widgets"
        assert not list(cache_dir.glob("schema_widgets_key0.*.tmp")), "temp not cleaned up"
