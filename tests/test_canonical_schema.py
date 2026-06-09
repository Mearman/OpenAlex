"""Regression tests for canonical-schema pinning in the parquet writer.

Background
----------
Before the fix, the parquet writer inferred each shard's Arrow schema from its
own first batch of rows.  When a column was all-null in a source file it was
written with the degenerate Arrow ``null`` type; another file that had data
wrote it typed — leaving the dataset schema-unstable file-to-file.  Empty
shards carried a stray ``_placeholder`` column.  The instability forced a
downstream DuckDB build to widen every column to VARCHAR, which stringified
the nested ``positions`` list column in the inverted-index relationship.

The fix: ``build_canonical_schemas`` derives one stable Arrow schema per
relationship table from a deterministic sample and ``_SourceFileWriter`` is
pinned to it for every shard.

Each test documents the pre-fix behaviour it would have caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable even when conftest.py hasn't run yet
# (e.g. when the test file is executed directly).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sync.schema import (
    EntitySchema,
    FieldSchema,
    _ColumnTypeObservations,
    _scalar_type_name,
    _widen_scalar_type,
    build_canonical_schemas,
    extract_relationships,
)
from sync.extract import _SourceFileWriter


# ── Shared fixtures ──────────────────────────────────────────────────────


def _works_inverted_index_schema() -> EntitySchema:
    """Minimal EntitySchema for a 'works' entity with an inverted-index field.

    Only the fields needed for the test are included; no real data or files
    are required.
    """
    return EntitySchema(
        entity="works",
        id_col="work_id",
        id_path="id",
        id_type="int",
        fields=[
            FieldSchema(
                json_key="abstract_inverted_index",
                pattern="inverted_index",
                rel_name="work_abstracts",
                is_singular_dict=True,
            ),
        ],
    )


def _works_scalar_schema(scalar_cols: list[dict]) -> EntitySchema:
    """Minimal EntitySchema for 'works' with declared scalar columns."""
    return EntitySchema(
        entity="works",
        id_col="work_id",
        id_path="id",
        id_type="int",
        fields=[
            FieldSchema(
                json_key="__scalars__",
                pattern="scalar",
                rel_name="work_main",
                scalar_cols=scalar_cols,
            ),
        ],
    )


def _sources_issn_schema() -> EntitySchema:
    """Minimal EntitySchema for 'sources' with an issn field."""
    return EntitySchema(
        entity="sources",
        id_col="source_id",
        id_path="id",
        id_type="int",
        fields=[
            FieldSchema(
                json_key="issn",
                pattern="issn",
                rel_name="source_issns",
            ),
        ],
    )


def _topics_schema() -> EntitySchema:
    """Minimal EntitySchema for 'topics' with a string_list field."""
    return EntitySchema(
        entity="topics",
        id_col="topic_id",
        id_path="id",
        id_type="int",
        fields=[
            FieldSchema(
                json_key="keywords",
                pattern="string_list",
                rel_name="topic_keywords",
                col_renames={},
            ),
        ],
    )


# ── Helper: openalex-style numeric id string ─────────────────────────────


def _oa_id(prefix: str, n: int) -> str:
    return f"https://openalex.org/{prefix}{n}"


# ── Test 1: no pa.null() fields in the canonical schema ─────────────────


class TestNoNullTypeInCanonicalSchema:
    """build_canonical_schemas must never emit a pa.null()-typed field.

    Pre-fix behaviour: when every sampled record had None for a column the
    _ColumnTypeObservations.arrow_type() call defaulted to string (via
    _widen_scalar_type), but the old per-batch inference path called
    pa.Table.from_pylist on an all-None column which Arrow inferred as
    pa.null() — that null type was then cached on self.schema and propagated
    to the parquet footer.

    This test builds canonical schemas from records where one relationship
    column is all-None across every sample and asserts the resulting Arrow
    field is NOT of type pa.null().
    """

    def test_all_null_extra_col_resolves_to_string(self) -> None:
        """An extra_col that is None in every sampled record becomes pa.string()."""
        schema = EntitySchema(
            entity="works",
            id_col="work_id",
            id_path="id",
            id_type="int",
            fields=[
                FieldSchema(
                    json_key="topics",
                    pattern="id_ref",
                    rel_name="work_topics",
                    id_path="id",
                    extra_cols=["score"],
                ),
            ],
        )
        # The 'score' extra_col is present in the relationship rows but always
        # None — pattern _pattern_id_ref only copies extra_cols when val is
        # not None.  So the column will be absent from all rows, meaning
        # _ColumnTypeObservations.names stays empty → _widen_scalar_type({})
        # returns "str" → pa.string().
        records = [
            {
                "id": _oa_id("W", i),
                "topics": [{"id": _oa_id("T", i), "display_name": "ML"}],
                # score intentionally absent from each topic dict
            }
            for i in range(1, 6)
        ]
        arrow_schemas = build_canonical_schemas(schema, records)
        assert "work_topics" in arrow_schemas, "Expected work_topics in canonical schemas"
        wt_schema = arrow_schemas["work_topics"]
        for field in wt_schema:
            assert field.type != pa.null(), (
                f"Field '{field.name}' has degenerate null type — "
                "pre-fix behaviour: all-null columns were inferred as pa.null()"
            )

    def test_column_type_observations_empty_names_defaults_to_string(self) -> None:
        """_ColumnTypeObservations with no observed non-null values → pa.string().

        Directly tests the unit-level guarantee: an observation that never
        sees a non-null scalar must not produce pa.null().
        """
        obs = _ColumnTypeObservations()
        # Observe only None values — names stays empty
        obs.observe(None)
        obs.observe(None)
        result = obs.arrow_type()
        assert result != pa.null(), (
            "_ColumnTypeObservations.arrow_type() returned pa.null() for an "
            "all-null column; expected pa.string()"
        )
        assert result == pa.string(), (
            f"Expected pa.string() for all-null column, got {result}"
        )

    def test_main_table_all_null_scalars_use_declared_type(self) -> None:
        """Main-table columns get their declared type even when all-null in sample.

        The seeding step in build_canonical_schemas pre-populates main-table
        column observations from the scalar_cols declarations, so a column
        with no non-null sample values still gets its declared type rather
        than defaulting to string.  An 'int'-declared column must become
        pa.int64(), not pa.string() and definitely not pa.null().
        """
        scalar_cols = [
            {"col": "publication_year", "path": "publication_year", "type": "int"},
            {"col": "is_oa", "path": "is_oa", "type": "bool"},
            {"col": "cited_by_count", "path": "cited_by_count", "type": "int"},
            {"col": "title", "path": "title", "type": "str"},
        ]
        schema = _works_scalar_schema(scalar_cols)
        # Records have none of these scalar fields populated
        records = [{"id": _oa_id("W", i)} for i in range(1, 5)]
        arrow_schemas = build_canonical_schemas(schema, records)
        assert "work_main" in arrow_schemas, "Expected work_main table in canonical schemas"
        main_schema = arrow_schemas["work_main"]
        field_map = {f.name: f.type for f in main_schema}
        assert field_map.get("publication_year") == pa.int64(), (
            f"publication_year declared int but got {field_map.get('publication_year')}; "
            "pre-fix: all-null column fell through to pa.null() or wrong type"
        )
        assert field_map.get("is_oa") == pa.bool_(), (
            f"is_oa declared bool but got {field_map.get('is_oa')}"
        )
        assert field_map.get("cited_by_count") == pa.int64(), (
            f"cited_by_count declared int but got {field_map.get('cited_by_count')}"
        )


# ── Test 2: inverted-index positions column is pa.list_(pa.int64()) ───────


class TestInvertedIndexPositionsType:
    """The 'positions' column in an inverted-index relationship must be a
    properly nested list type, not a stringified scalar.

    Pre-fix behaviour: if the inverted-index table appeared in only some shards,
    the DuckDB union_by_name widening (triggered by schema instability) cast
    every column in work_abstracts to VARCHAR — turning the positions list
    [[0, 3, 12], [5]] into the string "[[0, 3, 12], [5]]".  Even within a
    single shard, a first-batch schema inferred from an all-null column would
    emit pa.null() which prevented proper casting.
    """

    def test_positions_is_list_of_int64(self) -> None:
        schema = _works_inverted_index_schema()
        records = [
            {
                "id": _oa_id("W", 1),
                "abstract_inverted_index": {
                    "machine": [0, 5, 12],
                    "learning": [1, 6],
                    "the": [2, 7, 13, 20],
                },
            },
        ]
        arrow_schemas = build_canonical_schemas(schema, records)
        assert "work_abstracts" in arrow_schemas, (
            "Expected work_abstracts in canonical schemas"
        )
        abstracts_schema = arrow_schemas["work_abstracts"]
        field_map = {f.name: f.type for f in abstracts_schema}
        positions_type = field_map.get("positions")
        assert positions_type is not None, "No 'positions' column in work_abstracts schema"
        assert positions_type == pa.list_(pa.int64()), (
            f"positions column is {positions_type}, expected pa.list_(pa.int64()); "
            "pre-fix: DuckDB schema-widening cast this to VARCHAR"
        )

    def test_word_column_is_string(self) -> None:
        schema = _works_inverted_index_schema()
        records = [
            {
                "id": _oa_id("W", 2),
                "abstract_inverted_index": {"neural": [0, 4], "network": [1, 5]},
            },
        ]
        arrow_schemas = build_canonical_schemas(schema, records)
        abstracts_schema = arrow_schemas["work_abstracts"]
        field_map = {f.name: f.type for f in abstracts_schema}
        assert field_map.get("word") == pa.string(), (
            f"word column is {field_map.get('word')}, expected pa.string()"
        )

    def test_positions_nesting_preserved_through_writer(self, tmp_path: Path) -> None:
        """Writing inverted-index rows through _SourceFileWriter preserves list typing."""
        schema = _works_inverted_index_schema()
        records = [
            {
                "id": _oa_id("W", 10),
                "abstract_inverted_index": {"deep": [0, 7], "learning": [1, 8, 15]},
            },
        ]
        arrow_schemas = build_canonical_schemas(schema, records)
        canonical = arrow_schemas["work_abstracts"]

        out_dir = tmp_path / "work_abstracts"
        writer = _SourceFileWriter(
            out_dir, "work_abstracts", "shard_001",
            canonical_schema=canonical,
        )
        rows = [
            {"work_id": 10, "word": "deep", "positions": [0, 7]},
            {"work_id": 10, "word": "learning", "positions": [1, 8, 15]},
        ]
        writer.write_batch(rows)
        writer.close()

        shard_path = out_dir / "shard_001.parquet"
        assert shard_path.exists(), "Shard not written"
        written = pq.read_table(str(shard_path))
        positions_field = written.schema.field("positions")
        assert positions_field.type == pa.list_(pa.int64()), (
            f"Written positions type is {positions_field.type}, expected list<int64>; "
            "pre-fix: schema instability caused DuckDB to widen to VARCHAR"
        )


# ── Test 3: string-list / issn column types ───────────────────────────────


class TestStringListAndIssnTypes:
    """list-of-strings relationship columns must yield pa.list_(pa.string())
    (for the stored-as-list pattern) or pa.string() (for the exploded pattern).

    The issn pattern explodes each ISSN into a separate row with a scalar
    'issn' string column.  The topic_keywords string_list pattern similarly
    explodes into rows with a scalar 'keyword' column.  Neither should ever
    produce a list-typed column in the relationship table — the list is
    exploded at extraction time.

    Pre-fix: schema instability caused no direct type error here, but verifying
    these types ensures the canonical schema builder handles both patterns
    correctly and never falls through to pa.null().
    """

    def test_issn_column_is_string_not_null(self) -> None:
        schema = _sources_issn_schema()
        records = [
            {
                "id": _oa_id("S", 1),
                "issn": ["2049-3630", "1556-5068"],
            },
        ]
        arrow_schemas = build_canonical_schemas(schema, records)
        assert "source_issns" in arrow_schemas, (
            "Expected source_issns in canonical schemas"
        )
        issns_schema = arrow_schemas["source_issns"]
        field_map = {f.name: f.type for f in issns_schema}
        issn_type = field_map.get("issn")
        assert issn_type is not None, "No 'issn' column in source_issns schema"
        assert issn_type == pa.string(), (
            f"issn column is {issn_type}, expected pa.string()"
        )
        assert issn_type != pa.null(), (
            "issn column is pa.null() — pre-fix all-null schema inference bug"
        )

    def test_issn_all_null_sample_still_string(self) -> None:
        """Even with no sampled ISSN data the canonical schema must not use pa.null()."""
        schema = _sources_issn_schema()
        # Records have no issn key at all — the relationship table will not
        # appear in arrow_schemas (no rows were extracted), so this test
        # validates the _ColumnTypeObservations guarantee independently.
        obs = _ColumnTypeObservations()
        # Never observe any value
        result = obs.arrow_type()
        assert result != pa.null(), (
            "All-null _ColumnTypeObservations must not produce pa.null()"
        )
        assert result == pa.string()

    def test_string_list_keyword_column_is_string(self) -> None:
        """string_list pattern produces a scalar string column, not pa.null()."""
        schema = _topics_schema()
        records = [
            {
                "id": _oa_id("T", 1),
                "keywords": ["machine learning", "deep learning"],
            },
        ]
        arrow_schemas = build_canonical_schemas(schema, records)
        assert "topic_keywords" in arrow_schemas, (
            "Expected topic_keywords in canonical schemas"
        )
        kw_schema = arrow_schemas["topic_keywords"]
        field_map = {f.name: f.type for f in kw_schema}
        # The string_list pattern explodes to (entity_id, keyword) rows.
        # col_renames maps "value" → json_key.rstrip("s") = "keyword"
        keyword_col = field_map.get("keyword")
        assert keyword_col is not None, (
            f"Expected a 'keyword' column in topic_keywords, got: {list(field_map)}"
        )
        assert keyword_col == pa.string(), (
            f"keyword column is {keyword_col}, expected pa.string()"
        )


# ── Test 4: main-table scalar column seeding from declared types ──────────


class TestMainTableScalarSeeding:
    """Declared scalar column types are used even when all-null in the sample.

    The seeding loop in build_canonical_schemas calls
    ``column(main_rel, sc["col"]).names.add(sc["type"])`` for every scalar_col
    before processing any record.  This guarantees that a column absent from
    every sampled record still gets its declared Arrow type.

    Pre-fix: there was no seeding.  Absent columns never entered
    _ColumnTypeObservations.names, so they defaulted to "str" → pa.string()
    regardless of declared type — an int column would appear as string in the
    written schema, silently changing downstream query semantics.
    """

    @pytest.mark.parametrize("declared_type,expected_arrow", [
        ("int",   pa.int64()),
        ("float", pa.float64()),
        ("bool",  pa.bool_()),
        ("str",   pa.string()),
    ])
    def test_declared_type_wins_when_all_null(
        self,
        declared_type: str,
        expected_arrow: pa.DataType,
    ) -> None:
        scalar_cols = [{"col": "my_col", "path": "my_col", "type": declared_type}]
        schema = _works_scalar_schema(scalar_cols)
        # All records have the id but not my_col
        records = [{"id": _oa_id("W", i)} for i in range(1, 4)]
        arrow_schemas = build_canonical_schemas(schema, records)
        assert "work_main" in arrow_schemas
        main_schema = arrow_schemas["work_main"]
        field_map = {f.name: f.type for f in main_schema}
        assert "my_col" in field_map, (
            "Declared scalar column absent from canonical schema — seeding failed"
        )
        assert field_map["my_col"] == expected_arrow, (
            f"Declared '{declared_type}' column got Arrow type {field_map['my_col']}, "
            f"expected {expected_arrow}; pre-fix: absent columns defaulted to pa.string()"
        )

    def test_id_column_is_int64_for_int_entity(self) -> None:
        """The entity id column uses the declared id_type."""
        scalar_cols: list[dict] = []
        schema = EntitySchema(
            entity="works",
            id_col="work_id",
            id_path="id",
            id_type="int",
            fields=[
                FieldSchema(
                    json_key="__scalars__",
                    pattern="scalar",
                    rel_name="work_main",
                    scalar_cols=scalar_cols,
                ),
            ],
        )
        records = [{"id": _oa_id("W", i)} for i in range(1, 3)]
        arrow_schemas = build_canonical_schemas(schema, records)
        assert "work_main" in arrow_schemas
        main_schema = arrow_schemas["work_main"]
        field_map = {f.name: f.type for f in main_schema}
        assert field_map["work_id"] == pa.int64(), (
            f"work_id is {field_map['work_id']}, expected pa.int64()"
        )


# ── Test 5: schema stability across batches with different missing columns ─


class TestSchemaStabilityAcrossShards:
    """Two shards written from batches with different missing columns must have
    identical parquet schemas, both matching the canonical schema.

    Pre-fix: each _SourceFileWriter inferred its schema from its own first
    batch.  A column absent from shard A's first batch was typed as null in
    that shard's schema while shard B (which happened to have data for it)
    typed it correctly — dataset-level schema instability that forced DuckDB
    to union_by_name-widen to VARCHAR.
    """

    def _make_author_schema(self) -> tuple[EntitySchema, pa.Schema]:
        """Return (EntitySchema, canonical_arrow_schema) for a simple id_ref."""
        entity_schema = EntitySchema(
            entity="works",
            id_col="work_id",
            id_path="id",
            id_type="int",
            fields=[
                FieldSchema(
                    json_key="authorships",
                    pattern="id_ref",
                    rel_name="work_authorships",
                    id_path="author.id",
                    target_col="author_id",
                    extra_cols=["author_position"],
                ),
            ],
        )
        # One record that has author_position so the canonical schema observes it
        seed_records = [
            {
                "id": _oa_id("W", 1),
                "authorships": [
                    {"author": {"id": _oa_id("A", 1)}, "author_position": "first"},
                    {"author": {"id": _oa_id("A", 2)}, "author_position": "middle"},
                ],
            },
        ]
        arrow_schemas = build_canonical_schemas(entity_schema, seed_records)
        return entity_schema, arrow_schemas["work_authorships"]

    def test_two_shards_have_identical_schemas(self, tmp_path: Path) -> None:
        _, canonical = self._make_author_schema()

        # Shard A: rows with author_position present
        rows_a = [
            {"work_id": 1, "author_id": 101, "author_position": "first"},
            {"work_id": 1, "author_id": 102, "author_position": "last"},
        ]
        # Shard B: rows with author_position absent (the column that caused instability)
        rows_b = [
            {"work_id": 2, "author_id": 201},   # author_position missing
            {"work_id": 2, "author_id": 202},
        ]

        out_dir = tmp_path / "work_authorships"

        writer_a = _SourceFileWriter(
            out_dir, "work_authorships", "shard_a",
            canonical_schema=canonical,
        )
        writer_a.write_batch(rows_a)
        writer_a.close()

        writer_b = _SourceFileWriter(
            out_dir, "work_authorships", "shard_b",
            canonical_schema=canonical,
        )
        writer_b.write_batch(rows_b)
        writer_b.close()

        schema_a = pq.read_schema(str(out_dir / "shard_a.parquet"))
        schema_b = pq.read_schema(str(out_dir / "shard_b.parquet"))

        assert schema_a == schema_b, (
            f"Shard A schema:\n{schema_a}\n"
            f"Shard B schema:\n{schema_b}\n"
            "Schemas differ — pre-fix behaviour: shard whose first batch lacked "
            "'author_position' wrote it as null, breaking dataset-level schema stability"
        )
        assert schema_a == canonical, (
            "Written schema differs from canonical schema"
        )

    def test_missing_column_filled_with_typed_nulls(self, tmp_path: Path) -> None:
        """A batch that omits a canonical column must be filled with typed nulls.

        _rows_to_table uses ``row.get(name)`` for every column in the canonical
        schema, so a missing key becomes None rather than an absent column.
        The resulting table must still conform to the canonical schema.
        """
        _, canonical = self._make_author_schema()

        rows = [{"work_id": 3, "author_id": 301}]   # author_position absent
        out_dir = tmp_path / "work_authorships_fill"
        writer = _SourceFileWriter(
            out_dir, "work_authorships", "shard_fill",
            canonical_schema=canonical,
        )
        writer.write_batch(rows)
        writer.close()

        written = pq.read_table(str(out_dir / "shard_fill.parquet"))
        assert written.schema == canonical, (
            "Written schema deviates from canonical even though canonical was supplied"
        )
        ap_col = written.column("author_position")
        assert ap_col[0].as_py() is None, (
            "Missing column value should be None (typed null), not an error"
        )


# ── Test 6: empty shard uses canonical schema, no _placeholder column ─────


class TestEmptyShardSchema:
    """close() on a writer that never received a batch must write a parquet
    shard carrying the canonical schema — not a ``_placeholder`` column.

    Pre-fix behaviour (lines now removed from close()):

        empty_schema = pa.schema([pa.field("_placeholder", pa.int64())])
        empty_table = pa.table({"_placeholder": pa.array([], type=pa.int64())})
        pq.write_table(empty_table, out_path, ...)

    This ran when ``self.schema is None`` — i.e. when the writer had not
    received any rows so schema inference had never fired.  The result was
    an empty shard with a useless ``_placeholder`` column and no relationship
    columns whatsoever, which DuckDB then had to union_by_name across
    well-typed shards.
    """

    def _canonical(self) -> pa.Schema:
        entity_schema = EntitySchema(
            entity="works",
            id_col="work_id",
            id_path="id",
            id_type="int",
            fields=[
                FieldSchema(
                    json_key="concepts",
                    pattern="id_ref",
                    rel_name="work_concepts",
                    id_path="id",
                    target_col="concept_id",
                    extra_cols=["score"],
                ),
            ],
        )
        records = [
            {
                "id": _oa_id("W", 1),
                "concepts": [{"id": _oa_id("C", 1), "score": 0.75}],
            },
        ]
        arrow_schemas = build_canonical_schemas(entity_schema, records)
        return arrow_schemas["work_concepts"]

    def test_empty_shard_has_canonical_schema(self, tmp_path: Path) -> None:
        canonical = self._canonical()
        out_dir = tmp_path / "work_concepts"
        writer = _SourceFileWriter(
            out_dir, "work_concepts", "empty_shard",
            canonical_schema=canonical,
        )
        # Do NOT call write_batch — close() with zero rows is the test scenario
        writer.close()

        shard_path = out_dir / "empty_shard.parquet"
        assert shard_path.exists(), "Empty shard file was not created"

        written_schema = pq.read_schema(str(shard_path))
        assert "_placeholder" not in written_schema.names, (
            "Empty shard has a '_placeholder' column — pre-fix behaviour: "
            "close() wrote a _placeholder shard when schema was None"
        )
        assert written_schema == canonical, (
            f"Empty shard schema {written_schema} != canonical {canonical}; "
            "pre-fix: empty shards had a completely different _placeholder schema"
        )
        # The file must be readable and contain zero rows
        written_table = pq.read_table(str(shard_path))
        assert written_table.num_rows == 0, (
            f"Empty shard should have 0 rows, got {written_table.num_rows}"
        )

    def test_empty_shard_no_rows_but_all_columns_present(self, tmp_path: Path) -> None:
        """All canonical column names appear in the empty shard's schema."""
        canonical = self._canonical()
        out_dir = tmp_path / "work_concepts_cols"
        writer = _SourceFileWriter(
            out_dir, "work_concepts", "empty_cols_shard",
            canonical_schema=canonical,
        )
        writer.close()

        written_schema = pq.read_schema(str(out_dir / "empty_cols_shard.parquet"))
        for field in canonical:
            assert field.name in written_schema.names, (
                f"Column '{field.name}' missing from empty shard schema"
            )


# ── Test 7: unexpected column raises loudly ───────────────────────────────


class TestUnexpectedColumnRaisesError:
    """_rows_to_table must raise when a row carries a column not in the
    canonical schema, rather than silently dropping the data.

    Pre-fix: the writer had no canonical schema and inferred from the first
    batch; any novel column appearing in a later batch was simply absent from
    self.schema.names and dropped by the ``row.get(name)`` loop in
    _rows_to_table, causing silent data loss.

    After the fix: the known-schema path explicitly checks for unexpected keys
    and raises RuntimeError with a diagnostic message.
    """

    def _canonical(self) -> pa.Schema:
        entity_schema = EntitySchema(
            entity="works",
            id_col="work_id",
            id_path="id",
            id_type="int",
            fields=[
                FieldSchema(
                    json_key="concepts",
                    pattern="id_ref",
                    rel_name="work_concepts",
                    id_path="id",
                    target_col="concept_id",
                    extra_cols=["score"],
                ),
            ],
        )
        records = [
            {
                "id": _oa_id("W", 1),
                "concepts": [{"id": _oa_id("C", 1), "score": 0.8}],
            },
        ]
        arrow_schemas = build_canonical_schemas(entity_schema, records)
        return arrow_schemas["work_concepts"]

    def test_extra_column_raises_runtime_error(self, tmp_path: Path) -> None:
        canonical = self._canonical()
        out_dir = tmp_path / "work_concepts_err"
        writer = _SourceFileWriter(
            out_dir, "work_concepts", "shard_err",
            canonical_schema=canonical,
        )
        # Row carries a column the canonical schema never saw
        rows_with_extra = [
            {"work_id": 1, "concept_id": 42, "score": 0.9, "surprise_column": "oops"},
        ]
        with pytest.raises(RuntimeError, match="canonical schema"):
            writer.write_batch(rows_with_extra)

    def test_extra_column_message_names_the_column(self, tmp_path: Path) -> None:
        canonical = self._canonical()
        out_dir = tmp_path / "work_concepts_err2"
        writer = _SourceFileWriter(
            out_dir, "work_concepts", "shard_err2",
            canonical_schema=canonical,
        )
        rows_with_extra = [
            {"work_id": 1, "concept_id": 55, "score": 0.5, "novel_field": "data"},
        ]
        with pytest.raises(RuntimeError) as exc_info:
            writer.write_batch(rows_with_extra)
        message = str(exc_info.value)
        assert "novel_field" in message, (
            "RuntimeError message should name the unexpected column; "
            "pre-fix: unexpected columns were silently dropped, losing data"
        )

    def test_known_columns_with_some_missing_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        """A row that is merely missing a canonical column (key absent) is fine —
        it results in a typed null, not an error.  Only columns ADDED beyond
        the schema should raise.
        """
        canonical = self._canonical()
        out_dir = tmp_path / "work_concepts_ok"
        writer = _SourceFileWriter(
            out_dir, "work_concepts", "shard_ok",
            canonical_schema=canonical,
        )
        # score is in the schema but absent from these rows — should not raise
        rows_missing_col = [
            {"work_id": 1, "concept_id": 10},  # score absent → typed null
            {"work_id": 2, "concept_id": 20, "score": 0.3},
        ]
        # Must not raise
        writer.write_batch(rows_missing_col)
        writer.close()
        written = pq.read_table(str(out_dir / "shard_ok.parquet"))
        assert written.schema == canonical


# ── Additional unit-level helpers ────────────────────────────────────────


class TestScalarTypeNameAndWiden:
    """Unit tests for _scalar_type_name and _widen_scalar_type."""

    def test_scalar_type_name_bool_before_int(self) -> None:
        """bool is a subclass of int; _scalar_type_name must classify it as 'bool'."""
        assert _scalar_type_name(True) == "bool"
        assert _scalar_type_name(False) == "bool"
        assert _scalar_type_name(1) == "int"
        assert _scalar_type_name(1.5) == "float"
        assert _scalar_type_name("hello") == "str"
        assert _scalar_type_name(None) is None
        assert _scalar_type_name([1, 2]) is None
        assert _scalar_type_name({"a": 1}) is None

    @pytest.mark.parametrize("types,expected", [
        (set(),           "str"),   # empty → default to string
        ({"bool"},        "bool"),
        ({"int"},         "int"),
        ({"float"},       "float"),
        ({"str"},         "str"),
        ({"bool", "int"}, "int"),   # int subsumes bool
        ({"int", "float"},"float"), # float subsumes int
        ({"str", "int"},  "str"),   # str wins
        ({"str", "float"},"str"),
        ({"str", "bool"}, "str"),
    ])
    def test_widen_scalar_type(self, types: set, expected: str) -> None:
        assert _widen_scalar_type(types) == expected, (
            f"_widen_scalar_type({types!r}) should be {expected!r}"
        )
