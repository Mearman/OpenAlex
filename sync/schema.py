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
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from sync.common import extract_id, iter_jsonl

log = logging.getLogger(__name__)

# ── Skip rules ──────────────────────────────────────────────────────────
# Fields that are NOT extracted as relationship tables.
_SKIP_FIELDS: frozenset[str] = frozenset({
    "summary_stats",
    "geo",
    "biblio",
    "open_access",
    "citation_normalized_percentile",
    "cited_by_percentile_year",
    "has_content",
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
        )


@dataclass
class EntitySchema:
    """Schema for one entity type.

    Attributes:
        entity: Entity name (e.g. "works")
        id_col: Column name for the entity ID (e.g. "work_id")
        id_path: Dot-path to the entity's own ID in the record (e.g. "id")
        fields: Schemas for extractable fields
    """

    entity: str
    id_col: str
    id_path: str
    fields: list[FieldSchema] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "id_col": self.id_col,
            "id_path": self.id_path,
            "fields": [f.to_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EntitySchema:
        return cls(
            entity=d["entity"],
            id_col=d["id_col"],
            id_path=d.get("id_path", "id"),
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
        target_id = extract_id(_resolve_nested(item, id_path))
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
        target_id = extract_id(_resolve_nested(item, id_path))
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
    """Pattern: location dicts with source.id + metadata."""
    rows: list[dict] = []
    for loc in items:
        source = loc.get("source") or {}
        source_id = extract_id(source.get("id"))
        if source_id is not None:
            rows.append({
                entity_id_col: entity_id,
                "source_id": source_id,
                "is_oa": bool(loc.get("is_oa", False)),
                "is_primary": bool(loc.get("is_primary", False)),
                "license": loc.get("license"),
                "version": loc.get("version"),
            })
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
) -> dict[str, list[dict]]:
    """Extract all relationship rows from a single record using its schema.

    Returns {rel_name: [row_dict, ...]} for each relationship type
    that has data in this record.
    """
    entity_id: int | str | None = extract_id(_resolve_nested(record, schema.id_path))
    if entity_id is None:
        # Non-numeric IDs (continents, countries, subfields) — use string tail
        raw_id = _resolve_nested(record, schema.id_path)
        if isinstance(raw_id, str) and raw_id:
            entity_id = raw_id.rsplit("/", 1)[-1]
        if entity_id is None:
            return {}

    result: dict[str, list[dict]] = {}

    for fs in schema.fields:
        value = record.get(fs.json_key)

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

    return result


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
    #   - "abstract_inverted_index" → json_blob
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
        fields.append(FieldSchema(json_key=key, pattern="external_ids", rel_name=rel_name))
    elif key == "abstract_inverted_index":
        fields.append(FieldSchema(json_key=key, pattern="json_blob", rel_name=rel_name))
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
    else:
        # Default: treat as id_ref (list of dicts with IDs)
        fields.append(FieldSchema(json_key=key, pattern="id_ref", rel_name=rel_name))


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

    # ── Dict values ──
    if isinstance(value, dict):
        if key == "ids":
            fields.append(FieldSchema(
                json_key="ids", pattern="external_ids",
                rel_name=f"{rel_prefix}external_ids",
            ))
            # Check if ids contains an 'issn' array (source entities)
            if "issn" in value and isinstance(value.get("issn"), list):
                fields.append(FieldSchema(
                    json_key="issn", pattern="issn",
                    rel_name=f"{rel_prefix}issns",
                ))
        elif key == "abstract_inverted_index":
            fields.append(FieldSchema(
                json_key="abstract_inverted_index", pattern="json_blob",
                rel_name=f"{rel_prefix}abstracts",
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
        # Otherwise skip (metadata wrapper like biblio, open_access, etc.)
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
                # Determine extra scalar columns
                extra = [
                    k for k in first
                    if k != "id"
                    and not isinstance(first.get(k), (list, dict))
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

    for record in records:
        for key in sorted(record.keys()):
            value = record[key]
            existing = seen_json_keys.get(key)
            # Skip if we already classified this key from real data
            if existing is not None and value is not None:
                continue
            # Allow re-classification: heuristic (from None/empty) → real data
            new_fields = _classify_field(key, value, entity, all_keys)
            seen_json_keys.update({f.json_key: f for f in new_fields})

    observed_fields = list(seen_json_keys.values())
    seed_fields = seed_schema.fields if seed_schema is not None else None
    fields = _merge_field_schemas(observed_fields, seed_fields)

    schema = EntitySchema(
        entity=entity,
        id_col=id_col,
        id_path=id_path,
        fields=fields,
    )
    log.info(
        "Probed schema for %s (multi): %d fields, %d relationship types",
        entity, len(schema.fields), len(schema.rel_type_names()),
    )
    return schema



# ── Committed schema file ──────────────────────────────────────────────

_SCHEMA_FILE_VERSION = 1
_SCHEMA_FILE = Path(__file__).resolve().parent.parent / "openalex.schema.json"
_PROBE_SAMPLE_SIZE = 100


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
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
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
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError as exc:
        log.warning("Cannot write schema probe cache %s: %s", path, exc)


# ── Schema probing from JSONL ───────────────────────────────────────────


def _schema_from_source_dir(
    entity: str,
    source_dir: Path,
    seed_schema: EntitySchema | None = None,
) -> EntitySchema:
    # Use os.walk with followlinks=True because pathlib's glob("**/*.gz")
    # does not traverse symlinked directories by default. This matters when
    # a worker stages a partition into the snapshot dir via a symlink to
    # an external mount (e.g. SSD), where the entity dir contains a symlink
    # to the partition source files.
    import os as _os
    found: list[Path] = []
    if source_dir.is_dir():
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
    records: list[dict[str, Any]] = []
    for file in files:
        for record in iter_jsonl(file):
            if isinstance(record, dict):
                records.append(record)
            if len(records) >= _PROBE_SAMPLE_SIZE:
                return probe_schema_multi(entity, records, seed_schema=seed_schema)
    if not records:
        raise RuntimeError(f"No readable records found for {entity} in {source_dir}")
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
