"""Schema-driven relationship extraction for OpenAlex entities.

Replaces per-entity extraction functions with a single generic extractor
parameterised by a schema inferred from sample data. The schema declares
which JSON fields to extract and which pattern to apply; the pattern
interpreter does the rest.

Entities and relationship types are discovered automatically — no
per-entity code is needed. New entities appearing in the OpenAlex
snapshot get picked up without changes to this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa

from sync.common import extract_id, iter_jsonl

log = logging.getLogger(__name__)

# ── Skip rules ──────────────────────────────────────────────────────────
# Fields that are NOT extracted as relationship tables.
# Fields skipped entirely: denormalised singular projections of list
# relationships (their data is already captured by the corresponding array
# field), so extracting them would duplicate rows. Everything else — including
# scalar-bearing metadata dicts like open_access, biblio, geo, summary_stats —
# is captured: scalars and the scalar leaves of such dicts flow into the main
# entity table via the "scalar" pattern.
_SKIP_FIELDS: frozenset[str] = frozenset({
    "primary_location",
    "best_oa_location",
    "primary_topic",
})

_SKIP_NESTED_KEYS: frozenset[str] = frozenset({
    "raw_affiliation_strings",
    "raw_author_name",
    "countries",
    "landing_page_url",
    "provenance",
})

# String-list field name suffixes that trigger string_list pattern.
_STRING_LIST_SUFFIXES: frozenset[str] = frozenset({
    "alternatives",
    "alternate_titles",
    "acronyms",
    "codes",
})


def _singular(entity: str) -> str:
    """Entity plural → singular for column naming (e.g. works → work)."""
    if entity.endswith("s") and not entity.endswith("ss"):
        return entity[:-1]
    return entity


def _py_type_name(value: Any) -> str:
    """Observed Python scalar type name for main-table column documentation.

    bool is checked before int because bool is a subclass of int.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _widen_scalar_type(types: set[str]) -> str:
    """Widen a set of observed scalar types to one column type.

    A string anywhere forces string (the lossless sink, since any scalar
    coerces to str); otherwise float subsumes int, and int subsumes bool.
    An all-null column defaults to string. This guarantees a single Arrow
    type per main-table column even when a field is typed inconsistently
    across records (e.g. a code stored as "12" in some rows and 12 in others).
    """
    if not types or "str" in types:
        return "str"
    if "float" in types:
        return "float"
    if "int" in types:
        return "int"
    if "bool" in types:
        return "bool"
    return "str"


def _coerce_scalar(value: Any, typ: str) -> Any:
    """Coerce a scalar to its declared (widened) main-table column type so
    every row in a shard is homogeneous. ``None`` passes through unchanged.

    Numeric/bool coercions only run on columns the probe found homogeneous;
    a genuinely non-coercible value raises loudly rather than being silently
    dropped — surfacing a real data anomaly the broad sample missed.
    """
    if value is None:
        return None
    if typ == "str":
        return value if isinstance(value, str) else str(value)
    if typ == "bool":
        return bool(value)
    if typ == "int":
        return int(value)
    if typ == "float":
        return float(value)
    return str(value)


def _derive_entity_id(record: dict, id_path: str) -> int | str | None:
    """Resolve an entity's own id from its record.

    Numeric OpenAlex ids (e.g. W123 → 123) extract to int; ids with no numeric
    suffix fall back to the final URL path segment as a string (slug entities
    like keywords, languages). A numeric slug therefore yields int while a
    textual one yields str — the caller widens/coerces to keep a column's type
    homogeneous (see EntitySchema.id_type).
    """
    raw = _resolve_nested(record, id_path)
    eid: int | str | None = extract_id(raw)
    if eid is None and isinstance(raw, str) and raw:
        eid = raw.rsplit("/", 1)[-1]
    return eid


# ── Schema types ────────────────────────────────────────────────────────


@dataclass
class FieldSchema:
    """Schema for one extractable field in an entity record.

    Attributes:
        json_key: Key in the JSON record (e.g. "authorships", "topics")
        pattern: Extraction pattern name (e.g. "id_ref", "url_list")
        rel_name: Output relationship table name (e.g. "work_authorships")
        id_path: Dot-path to the ID within each element (e.g. "author.id")
        target_col: Explicit name for the ID column (e.g. "author_id").
            When set, overrides the auto-derived column name in pattern
            handlers. Required for backward compatibility with existing
            parquet data whose column names don't follow the auto-derivation
            rules.
        extra_cols: Scalar fields to carry as columns (e.g. ["score", "count"])
        nested: Sub-array schemas for nested extraction
        col_renames: Rename columns from JSON key → parquet column
        is_singular_dict: True if the JSON value is a single dict, not an array
        scalar_cols: For the "scalar" pattern, the main-table columns this
            field contributes. Each entry is ``{"col", "path", "type"}`` where
            ``path`` is a dot-path resolved from the record root and ``type``
            is the observed Python type name ("str"/"int"/"float"/"bool").
            A top-level scalar contributes one column; a flattened metadata
            dict (e.g. open_access) contributes one per scalar leaf.
    """

    json_key: str
    pattern: str
    rel_name: str
    id_path: str | None = None
    target_col: str | None = None
    extra_cols: list[str] = field(default_factory=list)
    nested: list[FieldSchema] = field(default_factory=list)
    col_renames: dict[str, str] = field(default_factory=dict)
    is_singular_dict: bool = False
    scalar_cols: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "json_key": self.json_key,
            "pattern": self.pattern,
            "rel_name": self.rel_name,
        }
        if self.id_path:
            d["id_path"] = self.id_path
        if self.target_col:
            d["target_col"] = self.target_col
        if self.extra_cols:
            d["extra_cols"] = self.extra_cols
        if self.nested:
            d["nested"] = [n.to_dict() for n in self.nested]
        if self.col_renames:
            d["col_renames"] = self.col_renames
        if self.is_singular_dict:
            d["is_singular_dict"] = True
        if self.scalar_cols:
            d["scalar_cols"] = self.scalar_cols
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FieldSchema:
        return cls(
            json_key=d["json_key"],
            pattern=d["pattern"],
            rel_name=d["rel_name"],
            id_path=d.get("id_path"),
            target_col=d.get("target_col"),
            extra_cols=d.get("extra_cols", []),
            nested=[cls.from_dict(n) for n in d.get("nested", [])],
            col_renames=d.get("col_renames", {}),
            is_singular_dict=d.get("is_singular_dict", False),
            scalar_cols=d.get("scalar_cols", []),
        )


@dataclass
class EntitySchema:
    """Schema for one entity type.

    Attributes:
        entity: Entity name (e.g. "works")
        id_col: Column name for the entity ID (e.g. "work_id")
        id_path: Dot-path to the entity's own ID in the record (e.g. "id")
        fields: Schemas for extractable fields
        id_type: Widened type of the entity id ("int" or "str"), derived from
            data. Numeric-id entities (works W…, authors A…) are "int"; slug-id
            entities (keywords, languages) are "str". A few entities mix the two
            (a numeric slug yields int, others str) — widening to "str" keeps the
            id column homogeneous across every table that carries it.
    """

    entity: str
    id_col: str
    id_path: str
    fields: list[FieldSchema] = field(default_factory=list)
    id_type: str = "str"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "id_col": self.id_col,
            "id_path": self.id_path,
            "id_type": self.id_type,
            "fields": [f.to_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EntitySchema:
        return cls(
            entity=d["entity"],
            id_col=d["id_col"],
            id_path=d.get("id_path", "id"),
            id_type=d.get("id_type", "str"),
            fields=[FieldSchema.from_dict(f) for f in d.get("fields", [])],
        )

    def rel_type_names(self) -> frozenset[str]:
        """All relationship type names produced by this schema."""
        names: set[str] = set()
        for f in self.fields:
            names.add(f.rel_name)
            for n in f.nested:
                names.add(n.rel_name)
        return frozenset(names)


def _merge_unique_list(primary: list[str], secondary: list[str]) -> list[str]:
    merged = list(primary)
    seen = set(primary)
    for item in secondary:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def _merge_field_schema(primary: FieldSchema, secondary: FieldSchema | None) -> FieldSchema:
    if secondary is None:
        return primary

    nested_by_key = {field.json_key: field for field in secondary.nested}
    merged_nested: list[FieldSchema] = []
    seen_nested: set[str] = set()
    for field in primary.nested:
        merged_nested.append(_merge_field_schema(field, nested_by_key.pop(field.json_key, None)))
        seen_nested.add(field.json_key)
    for field in secondary.nested:
        if field.json_key not in seen_nested:
            merged_nested.append(field)

    merged_scalar_cols = list(primary.scalar_cols)
    seen_cols = {sc["col"] for sc in primary.scalar_cols}
    for sc in secondary.scalar_cols:
        if sc["col"] not in seen_cols:
            merged_scalar_cols.append(sc)
            seen_cols.add(sc["col"])

    return FieldSchema(
        json_key=primary.json_key,
        pattern=primary.pattern,
        rel_name=primary.rel_name,
        id_path=primary.id_path or secondary.id_path,
        target_col=primary.target_col or secondary.target_col,
        extra_cols=_merge_unique_list(primary.extra_cols, secondary.extra_cols),
        nested=merged_nested,
        col_renames={**secondary.col_renames, **primary.col_renames},
        is_singular_dict=primary.is_singular_dict or secondary.is_singular_dict,
        scalar_cols=merged_scalar_cols,
    )


def _merge_field_schemas(
    primary_fields: list[FieldSchema],
    secondary_fields: list[FieldSchema] | None,
) -> list[FieldSchema]:
    if not secondary_fields:
        return primary_fields

    secondary_by_key = {field.json_key: field for field in secondary_fields}
    merged: list[FieldSchema] = []
    seen: set[str] = set()

    for field in primary_fields:
        merged.append(_merge_field_schema(field, secondary_by_key.pop(field.json_key, None)))
        seen.add(field.json_key)

    for field in secondary_fields:
        if field.json_key not in seen:
            merged.append(field)

    return merged


# ── Pattern interpreter ────────────────────────────────────────────────


def _resolve_nested(obj: dict, path: str) -> Any:
    """Resolve a dot-path like 'author.id' from a dict."""
    parts = path.split(".")
    current: Any = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _pattern_url_list(
    entity_id: int,
    items: list[Any],
    fs: FieldSchema,
    entity_id_col: str,
    entity: str = "",
) -> dict[str, list[dict]]:
    """Pattern: list of OpenAlex URL strings → {entity}_id + {target}_id."""
    # Use explicit target_col if set, otherwise derive from json_key
    target_name = fs.target_col
    if target_name is None:
        target_name = fs.json_key.rstrip("s") + "_id"
    rows: list[dict] = []
    for url in items:
        target_id = extract_id(url)
        if target_id is not None:
            rows.append({entity_id_col: entity_id, target_name: target_id})
    return {fs.rel_name: rows} if rows else {}


def _pattern_id_ref(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
    entity: str = "",
) -> dict[str, list[dict]]:
    """Pattern: list of dicts with an ID field + optional scalar fields."""
    id_path = fs.id_path or "id"
    rows: list[dict] = []
    for item in items:
        target_id = _derive_entity_id(item, id_path)
        if target_id is None:
            continue
        row: dict[str, Any] = {entity_id_col: entity_id}
        # Use explicit target_col if set, otherwise auto-derive
        if fs.target_col:
            target_col = fs.target_col
        else:
            id_leaf = id_path.rsplit(".", 1)[-1]
            target_col = id_leaf + "_id" if id_leaf != "id" else fs.json_key.rstrip("s") + "_id"
        row[target_col] = target_id
        for col in fs.extra_cols:
            val = item.get(col)
            if val is not None:
                # Apply column rename if configured
                out_col = fs.col_renames.get(col, col)
                if isinstance(val, float):
                    row[out_col] = val
                elif isinstance(val, int):
                    row[out_col] = val
                elif isinstance(val, bool):
                    row[out_col] = val
                else:
                    row[out_col] = val
        rows.append(row)
    return {fs.rel_name: rows} if rows else {}


def _pattern_nested_id_ref(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
    entity: str = "",
) -> dict[str, list[dict]]:
    """Pattern: list of dicts with an ID field, scalar extras, and
    sub-arrays that themselves produce relationship tables.

    The parent element's ID is threaded into nested rows so they can
    be joined back.
    """
    id_path = fs.id_path or "id"
    result: dict[str, list[dict]] = {}
    parent_rows: list[dict] = []

    for item in items:
        target_id = _derive_entity_id(item, id_path)
        if target_id is None:
            continue
        row: dict[str, Any] = {entity_id_col: entity_id}
        # Use explicit target_col if set, otherwise auto-derive
        if fs.target_col:
            target_col = fs.target_col
        else:
            id_leaf = id_path.rsplit(".", 1)[-1]
            target_col = id_leaf + "_id" if id_leaf != "id" else fs.json_key.rstrip("s") + "_id"
        row[target_col] = target_id
        for col in fs.extra_cols:
            val = item.get(col)
            if val is not None:
                row[col] = val
        parent_rows.append(row)

        # Extract nested sub-arrays
        for nested_fs in fs.nested:
            sub_items = item.get(nested_fs.json_key)
            if not sub_items or not isinstance(sub_items, list):
                continue
            nested_id_path = nested_fs.id_path or "id"
            for sub in sub_items:
                sub_id = extract_id(_resolve_nested(sub, nested_id_path))
                if sub_id is None:
                    continue
                nrow: dict[str, Any] = {entity_id_col: entity_id}
                # Thread the parent target ID for join-back
                nrow[target_col] = target_id
                # Nested target col
                if nested_fs.target_col:
                    sub_col = nested_fs.target_col
                else:
                    sub_leaf = nested_id_path.rsplit(".", 1)[-1]
                    sub_auto = sub_leaf + "_id" if sub_leaf != "id" else nested_fs.json_key.rstrip("s") + "_id"
                    sub_col = sub_auto
                nrow[sub_col] = sub_id
                for col in nested_fs.extra_cols:
                    val = sub.get(col)
                    if val is not None:
                        nrow[col] = val
                result.setdefault(nested_fs.rel_name, []).append(nrow)

    if parent_rows:
        result[fs.rel_name] = parent_rows
    return result


def _pattern_time_series(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: list of dicts with 'year' + numeric fields."""
    rows: list[dict] = []
    for item in items:
        year = item.get("year")
        if year is None:
            continue
        row: dict[str, Any] = {entity_id_col: entity_id, "year": int(year)}
        for col in fs.extra_cols:
            val = item.get(col, 0)
            if isinstance(val, (int, float)):
                row[col] = int(val) if col != "score" else float(val)
            else:
                row[col] = 0
        rows.append(row)
    return {fs.rel_name: rows} if rows else {}


def _pattern_external_ids(
    entity_id: int,
    items: dict[str, Any],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: dict of {source: value} → {entity}_id, source, value."""
    rows: list[dict] = []
    for source, value in items.items():
        # Skip non-scalar values (issn is an array, handled separately)
        if isinstance(value, (list, dict)):
            continue
        if value:
            rows.append({
                entity_id_col: entity_id,
                "source": source,
                "value": str(value),
            })
    return {fs.rel_name: rows} if rows else {}


def _pattern_string_list(
    entity_id: int,
    items: list[Any],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: list of strings → {entity}_id, value."""
    col_name = fs.col_renames.get("value", fs.json_key.rstrip("s"))
    rows: list[dict] = []
    for val in items:
        if val and isinstance(val, str):
            rows.append({entity_id_col: entity_id, col_name: val})
    return {fs.rel_name: rows} if rows else {}


def _pattern_json_blob(
    entity_id: int,
    items: dict[str, Any],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: dict serialised as JSON string → one row per record."""
    blob = json.dumps(items, separators=(",", ":"))
    return {fs.rel_name: [{entity_id_col: entity_id, fs.json_key: blob}]}


def _pattern_inverted_index(
    entity_id: int,
    items: dict[str, Any],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: abstract inverted index → one row per word.

    The inverted index is a dict mapping word → list of token positions.
    Exploded into (entity_id, word, positions) rows so the data is typed
    and queryable rather than stored as a JSON blob.
    """
    rows: list[dict] = []
    for word, positions in items.items():
        rows.append({
            entity_id_col: entity_id,
            "word": word,
            "positions": positions,
        })
    return {fs.rel_name: rows} if rows else {}


def _pattern_generic_list(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: generic list of dicts → one row per item with all scalar fields.

    Recursively extracts nested sub-lists as separate relationship tables
    and flattens scalar leaves from nested dicts. Used as the fallback when
    no specific pattern matches a list-of-dicts field.
    """
    result: dict[str, list[dict]] = {}
    main_rows: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {entity_id_col: entity_id}
        _collect_scalar_leaves(item, row)
        main_rows.append(row)

        # Recursively extract nested sub-lists as separate relationship tables
        for k, v in item.items():
            if isinstance(v, list) and v:
                if isinstance(v[0], dict):
                    sub_rel = f"{fs.rel_name}_{k}"
                    sub_rows: list[dict[str, Any]] = []
                    for sub_item in v:
                        if not isinstance(sub_item, dict):
                            continue
                        sr: dict[str, Any] = {entity_id_col: entity_id}
                        _collect_scalar_leaves(sub_item, sr)
                        sub_rows.append(sr)
                    if sub_rows:
                        result.setdefault(sub_rel, []).extend(sub_rows)

    if main_rows:
        result[fs.rel_name] = main_rows
    return result


def _collect_scalar_leaves(d: dict, row: dict, prefix: str = "") -> None:
    """Flatten scalar values from a dict into row, recursing into sub-dicts.

    Nested dicts are traversed with column names like "source_id", "source_display_name".
    Non-scalar values (lists, nested complex structs) are skipped.
    """
    for k, v in d.items():
        col = f"{prefix}_{k}" if prefix else k
        if isinstance(v, (str, int, float, bool, type(None))):
            row[col] = v
        elif isinstance(v, dict):
            _collect_scalar_leaves(v, row, prefix=col)


def _collect_scalar_paths(
    d: dict,
    col_prefix: str,
    path_prefix: str = "",
) -> list[dict]:
    """Collect scalar leaf paths from a dict, recursing into sub-dicts.

    Returns a list of {col, path, type} dicts suitable for scalar_cols.
    col_prefix is underscore-separated (open_access_source_id),
    path_prefix is dot-separated (open_access.source.id).
    """
    if not path_prefix:
        path_prefix = col_prefix
    paths: list[dict] = []
    for k, v in d.items():
        col = f"{col_prefix}_{k}"
        path = f"{path_prefix}.{k}"
        if isinstance(v, (str, int, float, bool)):
            paths.append({"col": col, "path": path, "type": _py_type_name(v)})
        elif isinstance(v, dict):
            paths.extend(_collect_scalar_paths(v, col, path))
    return paths


def _pattern_roles(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: list of role dicts with id, role, works_count."""
    _ROLE_PREFIX: dict[str, str] = {
        "funder": "F",
        "publisher": "P",
        "institution": "I",
    }
    rows: list[dict] = []
    for role in items:
        role_id = extract_id(role.get("id"))
        role_type = role.get("role")
        if role_id is not None and role_type:
            rows.append({
                entity_id_col: entity_id,
                "role_entity_id": role_id,
                "role_type": role_type,
                "role_prefix": _ROLE_PREFIX.get(role_type, ""),
            })
    return {fs.rel_name: rows} if rows else {}


def _pattern_topic_share(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: list of dicts with id + value → topic_id, value."""
    rows: list[dict] = []
    for item in items:
        topic_id = extract_id(item.get("id"))
        if topic_id is not None:
            rows.append({
                entity_id_col: entity_id,
                "topic_id": topic_id,
                "value": float(item.get("value", 0.0)),
            })
    return {fs.rel_name: rows} if rows else {}


def _pattern_taxonomy_parent(
    entity_id: int,
    item: dict,
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: single dict with id → {entity}_id + {parent}_id."""
    id_path = fs.id_path or "id"
    parent_id = extract_id(_resolve_nested(item, id_path))
    if parent_id is None:
        return {}
    # Derive target column from the json_key
    target_col = fs.json_key + "_id"
    return {fs.rel_name: [{entity_id_col: entity_id, target_col: parent_id}]}


def _pattern_grants(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: grants array with funder URL string + award_id."""
    rows: list[dict] = []
    for grant in items:
        funder_id = extract_id(grant.get("funder"))
        if funder_id is not None:
            row: dict[str, Any] = {
                entity_id_col: entity_id,
                "funder_id": funder_id,
            }
            award_id = grant.get("award_id")
            if award_id is not None:
                row["award_id"] = award_id
            rows.append(row)
    return {fs.rel_name: rows} if rows else {}


def _pattern_mesh(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: MeSH terms with descriptor_ui, not OpenAlex IDs."""
    rows: list[dict] = []
    for mesh in items:
        descriptor_ui = mesh.get("descriptor_ui")
        if descriptor_ui:
            rows.append({
                entity_id_col: entity_id,
                "descriptor_ui": descriptor_ui,
                "descriptor_name": mesh.get("descriptor_name"),
                "qualifier_ui": mesh.get("qualifier_ui"),
                "qualifier_name": mesh.get("qualifier_name"),
                "is_major_topic": bool(mesh.get("is_major_topic", False)),
            })
    return {fs.rel_name: rows} if rows else {}


def _pattern_investigators(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: investigator dicts with optional nested affiliation."""
    result: dict[str, list[dict]] = {}

    # Handle lead/co_lead as single dicts (not arrays)
    if fs.is_singular_dict:
        items = [items] if isinstance(items, dict) else []

    for inv in items:
        if not isinstance(inv, dict) or not inv:
            continue
        aff = inv.get("affiliation")
        aff_name = None
        aff_country = None
        aff_id_rows: list[dict] = []

        if isinstance(aff, dict):
            aff_name = aff.get("name")
            aff_country = aff.get("country")
            for aid in (aff.get("ids") or []):
                if isinstance(aid, dict):
                    aff_id_rows.append({
                        entity_id_col: entity_id,
                        "investigator_orcid": inv.get("orcid"),
                        "investigator_family_name": inv.get("family_name"),
                        "affiliation_id": aid.get("id"),
                        "affiliation_type": aid.get("type"),
                        "asserted_by": aid.get("asserted_by"),
                    })
        elif isinstance(aff, str):
            aff_name = aff

        result.setdefault(fs.rel_name, []).append({
            entity_id_col: entity_id,
            "given_name": inv.get("given_name"),
            "family_name": inv.get("family_name"),
            "orcid": inv.get("orcid"),
            "affiliation_name": aff_name,
            "affiliation_country": aff_country,
            "role_start": inv.get("role_start"),
            "is_lead": fs.col_renames.get("is_lead", False),
            "is_co_lead": fs.col_renames.get("is_co_lead", False),
        })

        if aff_id_rows:
            # Find the nested affiliation schema
            for n in fs.nested:
                result.setdefault(n.rel_name, []).extend(aff_id_rows)

    return result


def _pattern_issn(
    entity_id: int,
    items: list[Any],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: ISSN string list → {entity}_id, issn."""
    rows: list[dict] = []
    for issn in items:
        if issn:
            rows.append({entity_id_col: entity_id, "issn": str(issn)})
    return {fs.rel_name: rows} if rows else {}


def _pattern_apc_prices(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: APC price dicts → {entity}_id, price, currency."""
    rows: list[dict] = []
    for apc in items:
        price = apc.get("price")
        if price is not None:
            rows.append({
                entity_id_col: entity_id,
                "price": float(price),
                "currency": apc.get("currency"),
            })
    return {fs.rel_name: rows} if rows else {}


def _pattern_societies(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: society dicts with organization + url."""
    rows: list[dict] = []
    for society in items:
        org = society.get("organization")
        if org:
            rows.append({
                entity_id_col: entity_id,
                "organization": org,
                "url": society.get("url"),
            })
    return {fs.rel_name: rows} if rows else {}


def _pattern_awards(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: work awards with id + display_name + funder fields."""
    rows: list[dict] = []
    for award in items:
        award_id = extract_id(award.get("id"))
        if award_id is not None:
            rows.append({
                entity_id_col: entity_id,
                "award_id": award_id,
                "display_name": award.get("display_name"),
                "funder_award_id": award.get("funder_award_id"),
                "funder_id": extract_id(award.get("funder_id")),
                "funder_display_name": award.get("funder_display_name"),
            })
    return {fs.rel_name: rows} if rows else {}


def _pattern_locations(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
) -> dict[str, list[dict]]:
    """Pattern: location dicts → source.id plus every scalar metadata field.

    Captures the source link as ``source_id`` and then every scalar leaf of the
    location (``is_oa``, ``is_accepted``, ``is_published``, ``license``,
    ``license_id``, ``pdf_url``, ``version`` …) rather than a hardcoded subset,
    so new OpenAlex location fields are picked up automatically. Locations
    without a source are still emitted (their metadata is real). The ``source``
    dict, the skip-listed keys, and ``raw_*`` projections are excluded.
    """
    rows: list[dict] = []
    for loc in items:
        if not isinstance(loc, dict):
            continue
        source = loc.get("source") or {}
        row: dict[str, Any] = {
            entity_id_col: entity_id,
            "source_id": extract_id(source.get("id")),
        }
        for k, v in loc.items():
            if k in ("source", "id") or k in _SKIP_NESTED_KEYS or k.startswith("raw_"):
                continue
            if isinstance(v, (str, int, float, bool)):
                row[k] = v
        rows.append(row)
    return {fs.rel_name: rows} if rows else {}


def _pattern_slug_ref(
    entity_id: int,
    items: list[dict],
    fs: FieldSchema,
    entity_id_col: str,
    entity: str = "",
) -> dict[str, list[dict]]:
    """Pattern: list of dicts with an ID URL → slug string + optional score.

    Unlike id_ref which extracts numeric IDs, this extracts the URL tail
    as a string slug (e.g. "machine-learning"). Used for keywords.
    """
    id_path = fs.id_path or "id"
    rows: list[dict] = []
    target_col = fs.target_col or (fs.json_key.rstrip("s") + "_id")
    for item in items:
        raw = _resolve_nested(item, id_path)
        slug = None
        if isinstance(raw, str) and raw:
            slug = raw.rsplit("/", 1)[-1]
        if slug:
            row: dict[str, Any] = {entity_id_col: entity_id, target_col: slug}
            for col in fs.extra_cols:
                val = item.get(col)
                if val is not None:
                    row[col] = float(val) if col == "score" else val
            rows.append(row)
    return {fs.rel_name: rows} if rows else {}


# ── Pattern dispatch ───────────────────────────────────────────────────

_PATTERN_DISPATCH: dict[str, Any] = {
    "url_list": _pattern_url_list,
    "id_ref": _pattern_id_ref,
    "nested_id_ref": _pattern_nested_id_ref,
    "slug_ref": _pattern_slug_ref,
    "time_series": _pattern_time_series,
    "external_ids": _pattern_external_ids,
    "string_list": _pattern_string_list,
    "json_blob": _pattern_json_blob,
    "inverted_index": _pattern_inverted_index,
    "generic_list": _pattern_generic_list,
    "roles": _pattern_roles,
    "topic_share": _pattern_topic_share,
    "taxonomy_parent": _pattern_taxonomy_parent,
    "grants": _pattern_grants,
    "mesh": _pattern_mesh,
    "investigators": _pattern_investigators,
    "issn": _pattern_issn,
    "apc_prices": _pattern_apc_prices,
    "societies": _pattern_societies,
    "awards": _pattern_awards,
    "locations": _pattern_locations,
}


# ── Generic extractor ──────────────────────────────────────────────────


def extract_relationships(
    record: dict,
    schema: EntitySchema,
    wanted: frozenset[str] | None = None,
) -> dict[str, list[dict]]:
    """Extract all relationship rows from a single record using its schema.

    Returns {rel_name: [row_dict, ...]} for each relationship type
    that has data in this record.

    ``wanted`` restricts extraction to a set of rel_names — fields whose output
    (including nested rel_names) is not wanted are skipped without being
    computed. On the resume path, where most relationship tables are already
    complete, this avoids the expensive extraction of rows that would be
    discarded. Output for the wanted tables is identical to a full extraction.
    """
    entity_id = _derive_entity_id(record, schema.id_path)
    if entity_id is None:
        return {}
    # Coerce to the entity's widened id type so every table carrying this id
    # (main and all relationship tables) has a homogeneous id column — a numeric
    # slug (int) and a textual one (str) must not coexist in one column.
    entity_id = _coerce_scalar(entity_id, schema.id_type)

    main_rel = f"{_singular(schema.entity)}_main"
    want_main = wanted is None or main_rel in wanted

    result: dict[str, list[dict]] = {}

    for fs in schema.fields:
        # Scalar fields are collected into the main entity table below, not
        # dispatched as relationships.
        if fs.pattern == "scalar":
            continue

        # Skip fields whose output table isn't wanted (resume fast path).
        if wanted is not None:
            field_rels = {fs.rel_name}
            field_rels.update(n.rel_name for n in fs.nested)
            if not (field_rels & wanted):
                continue

        # Resolve the field value. Dotted json_keys (e.g. "open_access.license")
        # are resolved from nested dicts; plain keys use direct lookup.
        value = (
            _resolve_nested(record, fs.json_key)
            if "." in fs.json_key
            else record.get(fs.json_key)
        )

        # Handle None / empty
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        if isinstance(value, dict) and len(value) == 0:
            continue

        handler = _PATTERN_DISPATCH.get(fs.pattern)
        if handler is None:
            log.warning("Unknown pattern %s for field %s", fs.pattern, fs.json_key)
            continue

        # Build handler args — only pass entity to handlers that accept it
        handler_args = (entity_id, value, fs, schema.id_col)
        import inspect as _inspect
        _sig = _inspect.signature(handler)
        kwargs = {}
        if "entity" in _sig.parameters:
            kwargs["entity"] = schema.entity

        # Some patterns expect a single dict, not an array
        if fs.is_singular_dict:
            if not isinstance(value, dict):
                continue
            extracted = handler(*handler_args, **kwargs)
        else:
            # List-expecting patterns: skip if value isn't a list
            # (records occasionally contain malformed scalar values where
            # an array is expected — e.g. a stray float in referenced_works).
            if not isinstance(value, list):
                continue
            extracted = handler(*handler_args, **kwargs)

        for rel_name, rows in extracted.items():
            result.setdefault(rel_name, []).extend(rows)

    # ── Main entity table: one row per record holding every scalar column ──
    # Columns are fixed by the schema's scalar fields, so each row carries the
    # same keys (missing values resolve to None) and the per-shard parquet has
    # a stable column set. Skipped entirely when the main table isn't wanted.
    if want_main:
        main_row: dict[str, Any] = {schema.id_col: entity_id}
        has_scalar = False
        for fs in schema.fields:
            if fs.pattern != "scalar":
                continue
            has_scalar = True
            for sc in fs.scalar_cols:
                main_row[sc["col"]] = _coerce_scalar(
                    _resolve_nested(record, sc["path"]), sc["type"],
                )
        if has_scalar:
            result[main_rel] = [main_row]

    return result


# ── Canonical write schema ──────────────────────────────────────────────
# The parquet writer must use one stable Arrow schema per relationship table
# across every source shard, otherwise a column that is all-null (or absent) in
# one shard is written with a degenerate null type while another shard writes it
# typed — leaving the dataset schema-unstable file-to-file. That instability is
# what forces DuckDB's union_by_name to widen everything to VARCHAR (destroying
# nested list columns). Deriving the schema once, up front, from a sample fixes
# it at the source.

_ARROW_BY_NAME: dict[str, pa.DataType] = {
    "str": pa.string(),
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
}


def _scalar_type_name(value: Any) -> str | None:
    """Type name for a non-null scalar; ``None`` for lists/dicts/None.

    bool is checked before int because bool is an int subclass.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return None


class _ColumnTypeObservations:
    """Accumulates the observed scalar types (and list-ness) of one column."""

    __slots__ = ("is_list", "names")

    def __init__(self) -> None:
        self.is_list = False
        self.names: set[str] = set()

    def observe(self, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, list):
            self.is_list = True
            for element in value:
                name = _scalar_type_name(element)
                if name is not None:
                    self.names.add(name)
            return
        name = _scalar_type_name(value)
        if name is not None:
            self.names.add(name)

    def arrow_type(self) -> pa.DataType:
        # _widen_scalar_type collapses the observed set to one column type and
        # defaults the all-null case to "str" — the same rule the main table
        # uses, so scalar and relationship columns stay consistent.
        base = _ARROW_BY_NAME[_widen_scalar_type(self.names)]
        return pa.list_(base) if self.is_list else base


def build_canonical_schemas(
    schema: EntitySchema, records: list[dict[str, Any]],
) -> dict[str, pa.Schema]:
    """Build one stable Arrow schema per relationship table from a record sample.

    Column names and types are observed by running the real extraction over the
    sample, so they match the writer's output for every pattern — including the
    dynamic ``generic_list`` flattening that cannot be enumerated statically.
    The main table is additionally seeded from its declared ``scalar_cols`` so
    columns that are all-null in the sample still get their declared type rather
    than defaulting to string. Column order places the entity id column first.
    """
    observed: dict[str, dict[str, _ColumnTypeObservations]] = {}
    order: dict[str, list[str]] = {}

    def column(rel: str, col: str) -> _ColumnTypeObservations:
        cols = observed.setdefault(rel, {})
        obs = cols.get(col)
        if obs is None:
            obs = _ColumnTypeObservations()
            cols[col] = obs
            order.setdefault(rel, []).append(col)
        return obs

    # Seed the main table from declared scalar columns: the id column first,
    # then every scalar column with its declared type. This guarantees the
    # main table's full, correctly-typed column set regardless of the sample.
    main_rel = f"{_singular(schema.entity)}_main"
    column(main_rel, schema.id_col).names.add(schema.id_type)
    for fs in schema.fields:
        if fs.pattern == "scalar":
            for sc in fs.scalar_cols:
                column(main_rel, sc["col"]).names.add(sc["type"])

    # Seed every declared relationship table with the entity id column so a
    # table absent from the sample (e.g. a deprecated, always-empty one like
    # concept counts_by_year) still gets a stable id-only schema rather than
    # leaving its empty shards to fall back to the _placeholder marker. If such
    # a table does carry data in some record, the extra columns are observed
    # below; if it carries data the sample never saw, the writer's
    # unexpected-column guard surfaces it loudly.
    for rel in schema.rel_type_names():
        column(rel, schema.id_col).names.add(schema.id_type)

    for record in records:
        for rel, rows in extract_relationships(record, schema).items():
            for row in rows:
                for col, value in row.items():
                    column(rel, col).observe(value)

    return {
        rel: pa.schema(
            [pa.field(col, observed[rel][col].arrow_type()) for col in cols]
        )
        for rel, cols in order.items()
    }


def canonical_schemas_from_source_dir(
    entity: str, source_dir: Path,
) -> dict[str, pa.Schema]:
    """Sample an entity's source shards and build its canonical write schemas.

    Uses the same deterministic, evenly-spaced sampling as the schema probe, so
    every worker derives an identical schema from an identical sample.
    """
    schema = get_entity_schema(entity, source_dir=source_dir)
    records = _sample_source_records(entity, source_dir)
    return build_canonical_schemas(schema, records)


# ── Schema probe ────────────────────────────────────────────────────────
# Infers an EntitySchema from a sample record. This is the only place
# where per-entity knowledge exists — and it's expressed as generic
# structural heuristics, not per-entity code.


def _add_empty_list_field(
    key: str,
    entity: str,
    rel_prefix: str,
    fields: list[FieldSchema],
) -> None:
    """Classify an empty-list field by name heuristics.

    When a list is empty in all sampled records, we can't infer its pattern
    from content. Instead, we use the field name to determine the pattern.
    """
    # Derive rel_name from the field name using the standard convention:
    #   {entity_singular}_{field_name}  (e.g. work_authorships, author_topics)
    rel_name = f"{rel_prefix}{key}"

    # Pattern detection from field name:
    #   - "issn" → issn pattern
    #   - "counts_by_year" → time_series
    #   - "ids" → external_ids
    #   - "abstract_inverted_index" → inverted_index
    #   - "funded_outputs" / url-like → url_list
    #   - "grants" → grants
    #   - "awards" → awards
    #   - "societies" → societies
    #   - "apc_prices" → apc_prices
    #   - "affiliations" → nested_id_ref (institution.id)
    #   - fields ending in known string-list suffixes → string_list
    #   - anything with "id" in the name → id_ref
    #   - fallback → string_list

    if key == "issn":
        fields.append(FieldSchema(json_key=key, pattern="issn", rel_name=rel_name))
    elif key == "counts_by_year":
        fields.append(FieldSchema(
            json_key=key, pattern="time_series", rel_name=rel_name,
            extra_cols=["works_count", "cited_by_count", "oa_works_count"],
        ))
    elif key == "ids":
        fields.append(FieldSchema(
            json_key=key, pattern="external_ids", rel_name=rel_name,
            is_singular_dict=True,
        ))
    elif key == "abstract_inverted_index":
        fields.append(FieldSchema(
            json_key=key, pattern="inverted_index", rel_name=f"{rel_prefix}abstracts",
            is_singular_dict=True,
        ))
    elif key == "funded_outputs":
        fields.append(FieldSchema(json_key=key, pattern="url_list", rel_name=rel_name))
    elif key == "grants":
        fields.append(FieldSchema(json_key=key, pattern="grants", rel_name=rel_name))
    elif key == "awards":
        fields.append(FieldSchema(json_key=key, pattern="awards", rel_name=rel_name))
    elif key == "societies":
        fields.append(FieldSchema(json_key=key, pattern="societies", rel_name=rel_name))
    elif key == "apc_prices":
        fields.append(FieldSchema(json_key=key, pattern="apc_prices", rel_name=rel_name))
    elif key == "affiliations":
        fields.append(FieldSchema(
            json_key=key, pattern="nested_id_ref", rel_name=rel_name,
            id_path="institution.id",
        ))
    elif any(key.endswith(s) for s in _STRING_LIST_SUFFIXES):
        fields.append(FieldSchema(json_key=key, pattern="string_list", rel_name=rel_name))
    # No default: a field that is null/empty in every sampled record and whose
    # name matches no known relationship is NOT assumed to be a relationship.
    # The old `else: id_ref` fallback fabricated empty relationship tables for
    # scalar fields that happened to be null in the sample (e.g. work.doi when
    # absent) and for any unrecognised key. Such fields are left unclassified
    # here; if they ever carry real data in another sampled record, the
    # type-aware pass in probe_schema_multi reclassifies them from that data.


def _classify_field(
    key: str,
    value: Any,
    entity: str,
    all_keys: set[str],
) -> list[FieldSchema]:
    """Classify a JSON field into extraction pattern(s).

    Returns a list because some fields produce multiple relationship
    tables (e.g. authorships → work_authorships + work_authorship_institutions).
    """
    sing = _singular(entity)
    rel_prefix = f"{sing}_"
    fields: list[FieldSchema] = []

    # Skip explicitly-ignored fields
    if key in _SKIP_FIELDS:
        return []
    # Skip singular versions of arrays
    if key.startswith("primary_") or key.startswith("best_"):
        array_key = key.replace("primary_", "").replace("best_", "")
        if array_key in all_keys:
            return []

    # ── None values (field exists but no data in this record) ──
    if value is None:
        # Classify by key name heuristics, same as empty lists
        _add_empty_list_field(key, entity, rel_prefix, fields)
        return fields

    # ── Scalar values → main entity table column ──
    # Strings, numbers and booleans are attributes of the entity itself, not
    # relationships. They are collected into a single main table (rel_name
    # "{singular}_main") keyed by the entity id. The entity's own "id" is
    # already captured as the id column, so skip it here.
    if isinstance(value, (str, int, float, bool)):
        if key == "id":
            return fields
        fields.append(FieldSchema(
            json_key=key, pattern="scalar",
            rel_name=f"{sing}_main",
            scalar_cols=[{"col": key, "path": key, "type": _py_type_name(value)}],
        ))
        return fields

    # ── Dict values ──
    if isinstance(value, dict):
        if key == "ids":
            fields.append(FieldSchema(
                json_key="ids", pattern="external_ids",
                rel_name=f"{rel_prefix}external_ids",
                is_singular_dict=True,
            ))
            # Check if ids contains an 'issn' array (source entities)
            if "issn" in value and isinstance(value.get("issn"), list):
                fields.append(FieldSchema(
                    json_key="issn", pattern="issn",
                    rel_name=f"{rel_prefix}issns",
                ))
        elif key == "abstract_inverted_index":
            fields.append(FieldSchema(
                json_key="abstract_inverted_index", pattern="inverted_index",
                rel_name=f"{rel_prefix}abstracts",
                is_singular_dict=True,
            ))
        # Taxonomy parent: single dict with "id" and "display_name"
        elif "id" in value and "display_name" in value:
            fields.append(FieldSchema(
                json_key=key, pattern="taxonomy_parent",
                rel_name=f"{rel_prefix}{key}s",
                id_path="id",
                is_singular_dict=True,
            ))
        # Single investigator: dict with given_name/family_name
        # (lead_investigator, co_lead_investigator in awards)
        elif "given_name" in value and "family_name" in value:
            inv_fs = FieldSchema(
                json_key=key, pattern="investigators",
                rel_name=f"{rel_prefix}investigators",
                is_singular_dict=True,
                col_renames={
                    "is_lead": key == "lead_investigator",
                    "is_co_lead": key == "co_lead_investigator",
                },
                nested=[
                    FieldSchema(
                        json_key="affiliation",
                        pattern="investigators",
                        rel_name=f"{rel_prefix}investigator_affiliations",
                    ),
                ],
            )
            fields.append(inv_fs)
        # Otherwise: a metadata wrapper dict (biblio, open_access, geo,
        # summary_stats, citation_normalized_percentile, …). Flatten ALL scalar
        # leaves — including those nested inside sub-dicts — into main-table
        # columns named "{key}_{subkey}_{subsubkey}". Nested sub-lists of dicts
        # become separate relationship tables.
        if not fields:
            scalar_cols = _collect_scalar_paths(value, key)
            if scalar_cols:
                fields.append(FieldSchema(
                    json_key=key, pattern="scalar",
                    rel_name=f"{sing}_main",
                    scalar_cols=scalar_cols,
                ))
            # Nested sub-lists → separate relationship tables
            for sk, sv in value.items():
                if isinstance(sv, list) and sv and isinstance(sv[0], dict):
                    fields.append(FieldSchema(
                        json_key=f"{key}.{sk}", pattern="generic_list",
                        rel_name=f"{rel_prefix}{key}_{sk}",
                    ))
        return fields

    # ── List values ──
    # ── List values ──
    if isinstance(value, list):
        if len(value) == 0:
            # Empty list: classify by name heuristics
            # These fields are known to exist even when empty in sample records
            _add_empty_list_field(key, entity, rel_prefix, fields)
            return fields

        first = value[0]

        # List of strings
        if isinstance(first, str):
            # ISSN list: key is "issn" and strings look like ISSNs (digits-xdigits)
            if key == "issn":
                fields.append(FieldSchema(
                    json_key=key, pattern="issn",
                    rel_name=f"{rel_prefix}issns",
                ))
                return fields

            # URL list: strings containing openalex.org or https://
            if any("openalex.org" in s or s.startswith("https://") for s in value[:5]):
                fields.append(FieldSchema(
                    json_key=key, pattern="url_list",
                    rel_name=f"{rel_prefix}{key}",
                ))
            # String list: matching known suffixes
            elif any(key.endswith(s) for s in _STRING_LIST_SUFFIXES) or key == "indexed_in" or key == "keywords":
                fields.append(FieldSchema(
                    json_key=key, pattern="string_list",
                    rel_name=f"{rel_prefix}{key}",
                ))
            return fields

        # List of dicts
        if isinstance(first, dict):
            # Time series: has 'year' key
            if "year" in first:
                numeric_cols = [
                    k for k in first
                    if k != "year" and isinstance(first.get(k), (int, float))
                ]
                fields.append(FieldSchema(
                    json_key=key, pattern="time_series",
                    rel_name=f"{rel_prefix}counts_by_year",
                    extra_cols=numeric_cols,
                ))
                return fields

            # Keyword-like dicts: have 'id' with /keywords/ URL + 'score'
            # These use slug IDs (strings), not numeric
            raw_id = first.get("id", "")
            if isinstance(raw_id, str) and "/keywords/" in raw_id:
                fields.append(FieldSchema(
                    json_key=key, pattern="slug_ref",
                    rel_name=f"{rel_prefix}{key}",
                    id_path="id",
                    target_col=f"{key.rstrip('s')}_id",
                    extra_cols=["score"],
                ))
                return fields

            # Roles: has 'id' + 'role'
            if "id" in first and "role" in first:
                fields.append(FieldSchema(
                    json_key=key, pattern="roles",
                    rel_name=f"{rel_prefix}roles",
                ))
                return fields

            # Topic share: has 'id' + 'value' + domain/field/subfield
            if "id" in first and "value" in first and "domain" in first:
                fields.append(FieldSchema(
                    json_key=key, pattern="topic_share",
                    rel_name=f"{rel_prefix}topic_share",
                ))
                return fields

            # MeSH terms: has descriptor_ui
            if "descriptor_ui" in first:
                fields.append(FieldSchema(
                    json_key=key, pattern="mesh",
                    rel_name=f"{rel_prefix}mesh",
                ))
                return fields

            # APC prices: has 'price' + 'currency'
            if "price" in first and "currency" in first:
                fields.append(FieldSchema(
                    json_key=key, pattern="apc_prices",
                    rel_name=f"{rel_prefix}apc_prices",
                ))
                return fields

            # Societies: has 'organization'
            if "organization" in first and "url" in first:
                fields.append(FieldSchema(
                    json_key=key, pattern="societies",
                    rel_name=f"{rel_prefix}societies",
                ))
                return fields

            # Grants: has 'funder' (string URL) + optional 'award_id'
            if "funder" in first and isinstance(first.get("funder"), str):
                fields.append(FieldSchema(
                    json_key=key, pattern="grants",
                    rel_name=f"{rel_prefix}funders",
                ))
                return fields

            # Awards: has 'id' + 'display_name' + 'funder_id'
            if "id" in first and "funder_id" in first:
                fields.append(FieldSchema(
                    json_key=key, pattern="awards",
                    rel_name=f"{rel_prefix}awards",
                ))
                return fields

            # Locations: has 'source' dict with 'id'
            if "source" in first and isinstance(first.get("source"), dict):
                fields.append(FieldSchema(
                    json_key=key, pattern="locations",
                    rel_name=f"{rel_prefix}locations",
                ))
                return fields

            # Investigators: has given_name/family_name/orcid
            if "given_name" in first and "family_name" in first:
                inv_fs = FieldSchema(
                    json_key=key, pattern="investigators",
                    rel_name=f"{rel_prefix}investigators",
                    nested=[
                        FieldSchema(
                            json_key="affiliation",
                            pattern="investigators",  # reused for nested
                            rel_name=f"{rel_prefix}investigator_affiliations",
                        ),
                    ],
                )
                fields.append(inv_fs)
                return fields

            # Generic id_ref: has 'id' field OR sub-dict with 'id'
            if "id" in first or any(
                isinstance(v, dict) and "id" in v
                for v in first.values()
            ):
                # Determine extra columns: every scalar field, plus lists whose
                # elements are scalars (e.g. an affiliation's ``years``) — these
                # are carried as native list columns so the data isn't dropped.
                # Dicts and lists-of-dicts are excluded (the former are id
                # sub-dicts, the latter become nested relationship tables).
                def _is_scalar(x: Any) -> bool:
                    return isinstance(x, (str, int, float, bool))

                def _is_scalar_list(x: Any) -> bool:
                    return (
                        isinstance(x, list)
                        and len(x) > 0
                        and all(_is_scalar(e) for e in x if e is not None)
                    )

                extra = [
                    k for k in first
                    if k != "id"
                    and (_is_scalar(first.get(k)) or _is_scalar_list(first.get(k)))
                    and not k.startswith("raw_")
                    and k not in {"display_name", "wikidata"}
                ]

                # Check for nested sub-arrays with id-bearing dicts
                nested: list[FieldSchema] = []
                for sub_key, sub_val in first.items():
                    if sub_key in _SKIP_NESTED_KEYS:
                        continue
                    if isinstance(sub_val, list) and sub_val:
                        sub_first = sub_val[0]
                        if isinstance(sub_first, dict) and "id" in sub_first:
                            nested.append(FieldSchema(
                                json_key=sub_key, pattern="id_ref",
                                rel_name=f"{rel_prefix}{key}_{sub_key}",
                                id_path="id",
                            ))

                # Also check for id-bearing sub-dicts (like author.id)
                id_subdict: str | None = None
                for sub_key, sub_val in first.items():
                    if isinstance(sub_val, dict) and "id" in sub_val:
                        id_subdict = sub_key
                        break

                if nested or id_subdict:
                    pattern = "nested_id_ref"
                    id_path = f"{id_subdict}.id" if id_subdict else "id"
                    # Adjust extra columns: remove sub-dict keys
                    extra = [
                        k for k in extra
                        if k != id_subdict and k not in {n.json_key for n in nested}
                    ]
                    fields.append(FieldSchema(
                        json_key=key, pattern=pattern,
                        rel_name=f"{rel_prefix}{key}",
                        id_path=id_path,
                        extra_cols=extra,
                        nested=nested,
                    ))
                else:
                    fields.append(FieldSchema(
                        json_key=key, pattern="id_ref",
                        rel_name=f"{rel_prefix}{key}",
                        id_path="id",
                        extra_cols=extra,
                    ))
                return fields

            # Fallback: generic relationship table. Extracts all scalar fields
            # from each dict item, flattens sub-dict scalars, and recursively
            # extracts nested sub-lists as separate relationship tables.
            fields.append(FieldSchema(
                json_key=key, pattern="generic_list",
                rel_name=f"{rel_prefix}{key}",
            ))
            return fields

        return fields

    return fields


def probe_schema(
    entity: str,
    record: dict,
    seed_schema: EntitySchema | None = None,
) -> EntitySchema:
    """Build an EntitySchema by probing a single sample record."""
    return probe_schema_multi(entity, [record], seed_schema=seed_schema)


def probe_schema_multi(
    entity: str,
    records: list[dict],
    seed_schema: EntitySchema | None = None,
) -> EntitySchema:
    """Build an EntitySchema by probing multiple sample records.

    Merges fields discovered across all records so that empty arrays
    in one record don't prevent a field from being classified. Fields
    are deduplicated by rel_name; the first classification wins since
    patterns are determined by structure, not content variation.

    When a seed schema is supplied, any fields it knows about but the
    local probe does not see are added back in, which lets API-derived
    knowledge fill gaps in sparse local samples.
    """
    if not records:
        raise ValueError(f"No records provided for {entity}")

    sing = _singular(entity)
    id_col = f"{sing}_id"
    id_path = "id"
    rel_prefix = f"{sing}_"

    if seed_schema is not None and seed_schema.entity != entity:
        log.warning(
            "Seed schema entity mismatch for %s: %s",
            entity, seed_schema.entity,
        )
        seed_schema = None

    # Collect all keys across all records for the skip-logic
    all_keys: set[str] = set()
    for record in records:
        all_keys.update(record.keys())

    seen_json_keys: dict[str, FieldSchema] = {}
    classified_real: set[str] = set()   # record keys classified from real data
    classified_empty: set[str] = set()  # record keys classified from null/empty

    for record in records:
        for key in sorted(record.keys()):
            value = record[key]
            is_empty = value is None or (
                isinstance(value, (list, dict)) and len(value) == 0
            )
            if is_empty:
                # Classify an empty value only if this key has never been seen
                # with real data nor already heuristically classified. Real data
                # observed later for the same key takes precedence.
                if key in classified_real or key in classified_empty:
                    continue
                new_fields = _classify_field(key, value, entity, all_keys)
                seen_json_keys.update({f.json_key: f for f in new_fields})
                classified_empty.add(key)
            else:
                # Real data always (re)classifies, overriding any prior
                # null-sample heuristic placeholder for this key, then locks
                # the key so later records skip re-classification.
                if key in classified_real:
                    continue
                new_fields = _classify_field(key, value, entity, all_keys)
                seen_json_keys.update({f.json_key: f for f in new_fields})
                classified_real.add(key)
                classified_empty.discard(key)

    observed_fields = list(seen_json_keys.values())

    # Widen each scalar column's type across all sampled records. A column that
    # is a string in any record must be typed as string so extraction coerces
    # losslessly; mixing str and int would otherwise break Arrow's one-type-
    # per-column inference at write time.
    for fs in observed_fields:
        if fs.pattern != "scalar":
            continue
        for sc in fs.scalar_cols:
            observed_types: set[str] = set()
            for rec in records:
                val = _resolve_nested(rec, sc["path"])
                if val is None:
                    continue
                observed_types.add(_py_type_name(val))
            sc["type"] = _widen_scalar_type(observed_types)

    seed_fields = seed_schema.fields if seed_schema is not None else None
    fields = _merge_field_schemas(observed_fields, seed_fields)

    # Widen the entity id type across all sampled records (a slug entity whose
    # ids are mostly textual but occasionally all-digits must be typed "str").
    id_types: set[str] = set()
    for record in records:
        eid = _derive_entity_id(record, id_path)
        if eid is not None:
            id_types.add(_py_type_name(eid))
    schema_id_type = _widen_scalar_type(id_types)
    if seed_schema is not None and seed_schema.id_type == "str":
        schema_id_type = "str"

    schema = EntitySchema(
        entity=entity,
        id_col=id_col,
        id_path=id_path,
        fields=fields,
        id_type=schema_id_type,
    )
    log.info(
        "Probed schema for %s (multi): %d fields, %d relationship types",
        entity, len(schema.fields), len(schema.rel_type_names()),
    )
    return schema



# ── Committed schema file ──────────────────────────────────────────────

# Bump this whenever the probe/extraction logic changes what columns a schema
# produces. Both the committed schema file and the on-disk probe cache record
# the version they were written with and are rejected on load if it no longer
# matches, so a logic change automatically invalidates stale caches instead of
# silently shadowing the new behaviour (the source-file hash alone can't catch
# a code change). v2: locations capture all scalar leaves; relationship probes
# carry lists-of-scalars as native list columns.
_SCHEMA_FILE_VERSION = 2
_SCHEMA_FILE = Path(__file__).resolve().parent.parent / "openalex.schema.json"
_PROBE_SAMPLE_SIZE = 100  # records sampled per file
_PROBE_SAMPLE_FILES = 25  # files sampled, evenly spaced across the date range


def _schema_file_path() -> Path:
    """Resolve the schema file path."""
    return _SCHEMA_FILE


@lru_cache(maxsize=1)
def _load_schema_file() -> dict[str, dict[str, Any]]:
    """Load entity schemas from the committed openalex.schema.json."""
    path = _schema_file_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Ignoring unreadable schema file %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict) or payload.get("version") != _SCHEMA_FILE_VERSION:
        log.warning("Schema file version mismatch in %s", path)
        return {}
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        return {}
    return {e: s for e, s in entities.items() if isinstance(s, dict)}


def _write_schema_file(cache: dict[str, dict[str, Any]]) -> None:
    """Write updated schema file atomically."""
    path = _schema_file_path()
    payload = {
        "version": _SCHEMA_FILE_VERSION,
        "entities": cache,
    }
    # Unique temp file per writer (not a shared "<path>.tmp") so concurrent
    # workers that all cache-miss and re-probe at once can't clobber each
    # other's temp and fail the rename. Failures are logged, not raised, so a
    # best-effort cache update never aborts the run.
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=".openalex.schema.", suffix=".tmp",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        log.warning("Cannot write schema file %s: %s", path, exc)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    _load_schema_file.cache_clear()


def _load_entity_schema(entity: str) -> EntitySchema | None:
    """Load a single entity schema from the committed schema file."""
    cache = _load_schema_file()
    raw = cache.get(entity)
    if raw is None:
        return None
    try:
        return EntitySchema.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("Ignoring corrupt schema entry for %s: %s", entity, exc)
        return None


def _store_entity_schema(entity: str, schema: EntitySchema) -> None:
    """Update the committed schema file with a new/updated entity schema."""
    cache = _load_schema_file()
    cache[entity] = schema.to_dict()
    _write_schema_file(cache)
    log.info("Updated schema for %s in %s", entity, _schema_file_path())


# ── Probe cache ─────────────────────────────────────────────────────────
# The probe walks JSON records — slow on cold reads (minutes per entity).
# Cache the final merged schema on disk, keyed by the source file set so
# the cache invalidates automatically when the snapshot changes.

_PROBE_CACHE_DIR = Path(
    os.environ.get(
        "OPENALEX_SCHEMA_CACHE_DIR",
        str(Path.home() / ".cache" / "openalex-sync"),
    )
).expanduser()


def _probe_cache_disabled() -> bool:
    """Return True when the cache should be bypassed via env override."""
    value = os.environ.get("OPENALEX_SCHEMA_NOCACHE")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _entity_source_files(entity: str, source_dir: Path) -> list[Path]:
    """List the .gz source files belonging to *entity* under *source_dir*.

    Mirrors the glob the probe itself uses, but scoped to the entity's
    own subdirectory so the cache key isn't perturbed by unrelated
    entities. Falls back to the parent when no entity subdir exists.
    """
    entity_dir = source_dir / entity
    search_root = entity_dir if entity_dir.is_dir() else source_dir
    return sorted(
        file for file in search_root.glob("**/*.gz")
        if not file.name.startswith("._")
        and (file.name.endswith(".jsonl.gz") or file.suffix == ".gz")
    )


def _probe_cache_key(entity: str, source_dir: Path) -> str:
    """SHA256 of the sorted relative source filenames for *entity*.

    Stable across runs as long as the file set is unchanged; changes the
    moment a file is added, removed, or renamed.
    """
    files = _entity_source_files(entity, source_dir)
    hasher = hashlib.sha256()
    for file in files:
        try:
            rel = file.relative_to(source_dir)
        except ValueError:
            rel = file
        hasher.update(str(rel).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _probe_cache_path(entity: str, key: str) -> Path:
    return _PROBE_CACHE_DIR / f"schema_{entity}_{key}.json"


def _load_probe_cache(entity: str, key: str) -> EntitySchema | None:
    """Return the cached merged schema for *entity* if present and current."""
    path = _probe_cache_path(entity, key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Ignoring unreadable schema probe cache %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_key") != key or payload.get("entity") != entity:
        return None
    # Reject caches written by a different probe-logic version — the source-file
    # hash is unchanged across a code change, so without this a stale cache would
    # shadow the updated extraction behaviour.
    if payload.get("version") != _SCHEMA_FILE_VERSION:
        return None
    schema_dict = payload.get("schema")
    if not isinstance(schema_dict, dict):
        return None
    try:
        return EntitySchema.from_dict(schema_dict)
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("Ignoring corrupt schema probe cache %s: %s", path, exc)
        return None


def _store_probe_cache(entity: str, key: str, schema: EntitySchema) -> None:
    """Persist *schema* to the probe cache, keyed by *key*."""
    try:
        _PROBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("Cannot create schema cache dir %s: %s", _PROBE_CACHE_DIR, exc)
        return
    path = _probe_cache_path(entity, key)
    payload = {
        "version": _SCHEMA_FILE_VERSION,
        "entity": entity,
        "cache_key": key,
        "schema": schema.to_dict(),
    }
    # Write to a unique temp file, then atomically rename onto the final path.
    # A shared "<path>.tmp" would let concurrent writers (e.g. a forked
    # extraction pool that all cache-miss at once) clobber each other's temp and
    # fail the rename with FileNotFoundError. Every writer produces the same
    # schema, so a per-writer temp plus last-writer-wins rename is safe.
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(_PROBE_CACHE_DIR), prefix=f"schema_{entity}_", suffix=".tmp",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        log.warning("Cannot write schema probe cache %s: %s", path, exc)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Schema probing from JSONL ───────────────────────────────────────────


def _sample_source_records(entity: str, source_dir: Path) -> list[dict[str, Any]]:
    """Read a deterministic, evenly-spaced sample of records from an entity's
    source shards.

    Shared by the schema probe and the canonical write-schema builder so both
    observe an identical sample (and therefore an identical schema).
    """
    # Use os.walk with followlinks=True because pathlib's glob("**/*.gz")
    # does not traverse symlinked directories by default. This matters when
    # a worker stages a partition into the snapshot dir via a symlink to
    # an external mount (e.g. SSD), where the entity dir contains a symlink
    # to the partition source files.
    # Prefer the updated_date=* partition layout: it is fast and, crucially,
    # does not descend into relationship parquet subdirectories. Fall back to a
    # recursive walk for layouts without date partitions or with symlink-staged
    # sources (followlinks=True, since pathlib glob does not follow symlinks).
    files: list[Path] = []
    if source_dir.is_dir():
        files = sorted(
            f for f in source_dir.glob("updated_date=*/*.gz")
            if not f.name.startswith("._")
            and (f.name.endswith(".jsonl.gz") or f.suffix == ".gz")
        )
    if not files and source_dir.is_dir():
        import os as _os
        found: list[Path] = []
        for root, _dirs, names in _os.walk(source_dir, followlinks=True):
            root_path = Path(root)
            for name in names:
                if name.startswith("._"):
                    continue
                if name.endswith(".jsonl.gz") or name.endswith(".gz"):
                    found.append(root_path / name)
        files = sorted(found)
    if not files:
        raise FileNotFoundError(f"No source files found for {entity} in {source_dir}")

    # Sample deterministically across the full sorted file list (the partitions
    # are date-ordered) rather than only the first file, so relationships and
    # scalar columns that appear only in newer or older partitions are observed.
    # Evenly-spaced strides over the sorted list make the sample reproducible.
    n = len(files)
    if n <= _PROBE_SAMPLE_FILES:
        sample_files = files
    else:
        stride = n / _PROBE_SAMPLE_FILES
        sample_files = [files[int(i * stride)] for i in range(_PROBE_SAMPLE_FILES)]

    records: list[dict[str, Any]] = []
    for file in sample_files:
        taken = 0
        for record in iter_jsonl(file):
            if isinstance(record, dict):
                records.append(record)
                taken += 1
            if taken >= _PROBE_SAMPLE_SIZE:
                break
    if not records:
        raise RuntimeError(f"No readable records found for {entity} in {source_dir}")
    return records


def _schema_from_source_dir(
    entity: str,
    source_dir: Path,
    seed_schema: EntitySchema | None = None,
) -> EntitySchema:
    records = _sample_source_records(entity, source_dir)
    return probe_schema_multi(entity, records, seed_schema=seed_schema)


def probe_schema_from_file(
    entity: str,
    jsonl_path: Path,
) -> EntitySchema | None:
    """Probe schema from a single JSONL file.

    Used by CI to discover schema from a downloaded shard when no
    committed schema entry exists for an entity. Returns None if the
    file contains no valid records.
    """
    records: list[dict[str, Any]] = []
    try:
        for record in iter_jsonl(jsonl_path):
            if isinstance(record, dict):
                records.append(record)
            if len(records) >= _PROBE_SAMPLE_SIZE:
                break
    except Exception as exc:
        log.warning("Failed to read %s for schema probe: %s", jsonl_path, exc)
        return None
    if not records:
        return None
    seed_schema = _load_entity_schema(entity)
    return probe_schema_multi(entity, records, seed_schema=seed_schema)


# ── Public API ──────────────────────────────────────────────────────────


def get_entity_schema(entity: str, *, source_dir: Path | None = None) -> EntitySchema:
    """Get the schema for an entity.

    Resolution order:
    1. Committed schema file (openalex.schema.json) merged with on-disk
       probe cache keyed by the source file set (skips the slow re-probe
       when the source files are unchanged).
    2. Local source directory probe (if source_dir given and exists);
       result is cached to disk and written back to the committed schema.

    The ``OPENALEX_SCHEMA_NOCACHE`` env var forces a re-probe.
    """
    cached_schema = _load_entity_schema(entity)

    if source_dir is not None:
        resolved = source_dir.expanduser().resolve()
        if resolved.exists():
            use_cache = not _probe_cache_disabled()
            cache_key = _probe_cache_key(entity, resolved) if use_cache else ""

            if use_cache:
                cached_probe = _load_probe_cache(entity, cache_key)
                if cached_probe is not None:
                    log.info(
                        "Using cached schema probe for %s (key %s…)",
                        entity, cache_key[:12],
                    )
                    return cached_probe

            schema = _schema_from_source_dir(
                entity, resolved, seed_schema=cached_schema,
            )
            _store_entity_schema(entity, schema)
            if use_cache:
                _store_probe_cache(entity, cache_key, schema)
            return schema

    if cached_schema is not None:
        return cached_schema

    raise RuntimeError(
        f"No schema for {entity}. "
        f"Add it to openalex.schema.json or provide source_dir."
    )


def _discover_entities(source_dir: Path) -> list[str]:
    """Discover entity types from subdirectories containing .jsonl.gz files."""
    if not source_dir.exists():
        return []
    entities = []
    for child in sorted(source_dir.iterdir()):
        if child.is_dir() and not child.name.startswith((".", "_")):
            if any(child.rglob("*.gz")):
                entities.append(child.name)
    return entities


def entity_rel_types(entity: str, *, source_dir: Path | None = None) -> frozenset[str]:
    """Return the relationship type names for an entity."""
    schema = get_entity_schema(entity, source_dir=source_dir)
    rel_types = schema.rel_type_names()
    if not rel_types:
        raise RuntimeError(f"No relationship types discovered for {entity}")
    return rel_types


def all_entity_rel_types(*, source_dir: Path | None = None) -> dict[str, frozenset[str]]:
    """Return {entity: rel_types} for every discovered entity type."""
    if source_dir is not None:
        resolved = source_dir.expanduser().resolve()
        entities = _discover_entities(resolved)
    else:
        entities = sorted(_load_schema_file().keys())
    return {entity: entity_rel_types(entity, source_dir=source_dir) for entity in entities}
