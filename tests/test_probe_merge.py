"""Field discovery must merge all sampled records, not classify from one.

A field that is null or absent in the first record (or first list element) but
populated later was silently dropped before — these tests pin the fix.
"""
from __future__ import annotations

from sync.schema import probe_schema_multi, _representative_value


def _extra_cols(schema, json_key):
    for f in schema.fields:
        if f.json_key == json_key:
            return set(f.extra_cols)
    raise AssertionError(f"no field {json_key} in {[f.json_key for f in schema.fields]}")


class TestSparseAcrossRecords:
    def test_field_only_in_later_record_is_discovered(self):
        # First record's institution carries no lineage; a later one does.
        records = [
            {"id": "https://openalex.org/I1",
             "last_known_institutions": [
                 {"id": "https://openalex.org/I9", "ror": "a",
                  "country_code": "US", "type": "education"}]},
            {"id": "https://openalex.org/I2",
             "last_known_institutions": [
                 {"id": "https://openalex.org/I8", "ror": "b",
                  "country_code": "GB", "type": "education",
                  "lineage": ["https://openalex.org/I8", "https://openalex.org/I7"]}]},
        ]
        schema = probe_schema_multi("institutions", records)
        assert "lineage" in _extra_cols(schema, "last_known_institutions")


class TestSparseAcrossListElements:
    def test_field_only_in_later_element_is_discovered(self):
        # Within one record, the first element lacks the field, a sibling has it.
        records = [
            {"id": "https://openalex.org/I3",
             "last_known_institutions": [
                 {"id": "https://openalex.org/A", "type": "education"},
                 {"id": "https://openalex.org/B", "type": "education",
                  "lineage": ["https://openalex.org/B"]}]},
        ]
        schema = probe_schema_multi("institutions", records)
        assert "lineage" in _extra_cols(schema, "last_known_institutions")


class TestSkipListStillHonoured:
    def test_skip_nested_key_not_promoted_even_when_present(self):
        # countries is a _SKIP_NESTED_KEYS scalar list; merging must not leak it.
        records = [
            {"id": "https://openalex.org/W1",
             "authorships": [
                 {"author": {"id": "https://openalex.org/A1"},
                  "author_position": "first", "countries": ["US"]}]},
        ]
        schema = probe_schema_multi("works", records)
        assert "countries" not in _extra_cols(schema, "authorships")


class TestRepresentativeValue:
    def test_unions_dict_keys_across_elements(self):
        merged = _representative_value([
            [{"id": "x", "a": 1}],
            [{"id": "y", "b": ["p", "q"]}],
        ])
        assert isinstance(merged, list) and len(merged) == 1
        assert set(merged[0]) == {"id", "a", "b"}
        assert merged[0]["b"] == ["p", "q"]

    def test_scalar_list_kept_as_list(self):
        assert _representative_value([["p"], ["q", "r"]]) == ["p", "q", "r"]

    def test_all_null_returns_none(self):
        assert _representative_value([None, None]) is None
