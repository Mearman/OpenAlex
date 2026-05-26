"""Extract nested arrays from JSONL into relationship Parquet tables.

Processes entity records (works, authors, sources, institutions,
publishers, funders, concepts, topics, subfields, fields) to produce
normalised relationship tables covering all OpenAlex data model
relationships.

Workers process disjoint chunks of the source-file list and each
write to their own ``part-WW-NNNNN.parquet`` shards under the
relationship-type directory, where ``WW`` is the worker index. The
parent process aggregates row counts and writes a single
``_provenance.json`` per relationship type.
"""

from __future__ import annotations

import json
import logging
import hashlib
import multiprocessing
import shutil
import subprocess
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sync.common import (
    SNAPSHOT_DIR,
    STAGING_DIR,
    extract_id,
    iter_source_files,
    iter_jsonl,
    create_output_dir,
    get_skipped_missing_files,
    reset_skipped_files,
    rt_dir,
    nested_rt_path,
    format_size,
    _json_loads,
)

try:
    import orjson as _orjson
    _json_dumps = lambda obj: _orjson.dumps(obj).decode()
except ImportError:
    _json_dumps = lambda obj: json.dumps(obj, separators=(",", ":"))

from sync.schemas import RELATIONSHIP_SCHEMAS

log = logging.getLogger("openalex-sync")

# ── Batch thresholds ────────────────────────────────────────────────────

_BATCH_SIZE = 4_000_000
_PROGRESS_INTERVAL = 1_000_000
_DEFAULT_WORKERS = 6

# ── Relationship extraction functions ───────────────────────────────────


def _extract_work_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract all relationship rows from a single work record."""
    work_id = extract_id(record.get("id"))
    if work_id is None:
        return {}

    result: dict[str, list[dict]] = {}

    # Authorships
    for authorship in (record.get("authorships") or []):
        author = authorship.get("author") or {}
        author_id = extract_id(author.get("id"))
        if author_id is not None:
            result.setdefault("work_authorships", []).append({
                "work_id": work_id,
                "author_id": author_id,
                "author_position": authorship.get("author_position"),
            })
            for inst in (authorship.get("institutions") or []):
                iid = extract_id(inst.get("id"))
                if iid is not None:
                    result.setdefault("work_authorship_institutions", []).append({
                        "work_id": work_id,
                        "author_id": author_id,
                        "institution_id": iid,
                    })

    # References
    for ref_url in (record.get("referenced_works") or []):
        ref_id = extract_id(ref_url)
        if ref_id is not None:
            result.setdefault("work_references", []).append({
                "work_id": work_id,
                "referenced_work_id": ref_id,
            })

    # Topics
    for topic in (record.get("topics") or []):
        topic_id = extract_id(topic.get("id"))
        if topic_id is not None:
            result.setdefault("work_topics", []).append({
                "work_id": work_id,
                "topic_id": topic_id,
                "score": topic.get("score", 0.0),
            })

    # Concepts
    for concept in (record.get("concepts") or []):
        concept_id = extract_id(concept.get("id"))
        if concept_id is not None:
            result.setdefault("work_concepts", []).append({
                "work_id": work_id,
                "concept_id": concept_id,
                "score": concept.get("score", 0.0),
            })

    # Locations
    primary_loc = record.get("primary_location")
    for loc in (record.get("locations") or []):
        source = loc.get("source") or {}
        source_id = extract_id(source.get("id"))
        _pl_source = primary_loc.get("source") if primary_loc else None
        is_primary = (
            primary_loc is not None
            and isinstance(_pl_source, dict)
            and isinstance(loc.get("source"), dict)
            and loc["source"].get("id") == _pl_source.get("id")
        )
        if source_id is not None:
            result.setdefault("work_locations", []).append({
                "work_id": work_id,
                "source_id": source_id,
                "is_oa": bool(loc.get("is_oa", False)),
                "is_primary": is_primary,
                "license": loc.get("license"),
                "version": loc.get("version"),
            })

    # Related works
    for rel_url in (record.get("related_works") or []):
        rel_id = extract_id(rel_url)
        if rel_id is not None:
            result.setdefault("work_related", []).append({
                "work_id": work_id,
                "related_work_id": rel_id,
            })

    # Funders (grants)
    for grant in (record.get("grants") or []):
        funder_id = extract_id(grant.get("funder"))
        if funder_id is not None:
            result.setdefault("work_funders", []).append({
                "work_id": work_id,
                "funder_id": funder_id,
                "award_id": grant.get("award_id"),
            })

    # Keywords
    for kw in (record.get("keywords") or []):
        kw_url = kw.get("id") or ""
        # keyword IDs are slug strings (e.g. "artificial-intelligence"),
        # not numeric — extract from URL tail
        keyword_id = kw_url.rsplit("/", 1)[-1] if kw_url else None
        if keyword_id:
            result.setdefault("work_keywords", []).append({
                "work_id": work_id,
                "keyword_id": keyword_id,
                "score": kw.get("score", 0.0),
            })

    # Sustainable Development Goals
    for sdg in (record.get("sustainable_development_goals") or []):
        sdg_id = extract_id(sdg.get("id"))
        if sdg_id is not None:
            result.setdefault("work_sdgs", []).append({
                "work_id": work_id,
                "sdg_id": sdg_id,
                "score": sdg.get("score", 0.0),
            })

    # MeSH terms. OpenAlex inherits these from PubMed, so most records
    # outside biomedicine carry no `mesh` array and yield no rows.
    for mesh in (record.get("mesh") or []):
        descriptor_ui = mesh.get("descriptor_ui")
        if descriptor_ui:
            result.setdefault("work_mesh", []).append({
                "work_id": work_id,
                "descriptor_ui": descriptor_ui,
                "descriptor_name": mesh.get("descriptor_name"),
                "qualifier_ui": mesh.get("qualifier_ui"),
                "qualifier_name": mesh.get("qualifier_name"),
                "is_major_topic": bool(mesh.get("is_major_topic", False)),
            })

    # Corresponding author IDs (subset of authorships, but flagged here for fast lookup)
    for ca_url in (record.get("corresponding_author_ids") or []):
        ca_id = extract_id(ca_url)
        if ca_id is not None:
            result.setdefault("work_corresponding_authors", []).append({
                "work_id": work_id,
                "author_id": ca_id,
            })

    # Corresponding institution IDs (subset of institutions, flagged for fast lookup)
    result["work_corresponding_institutions"] = [
        {"work_id": work_id, "institution_id": ci_id}
        for ci_url in (record.get("corresponding_institution_ids") or [])
        if (ci_id := extract_id(ci_url)) is not None
    ]

    # Counts by year (time-series citation data)
    result["work_counts_by_year"] = [
        {
            "work_id": work_id,
            "year": int(year),
            "cited_by_count": int(cby.get("cited_by_count", 0)),
        }
        for cby in (record.get("counts_by_year") or [])
        if (year := cby.get("year")) is not None
    ]

    # External identifiers (DOI/PMID/PMCID/MAG/OpenAlex)
    result["work_external_ids"] = [
        {"work_id": work_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    # Indexed-in directories (crossref, pubmed, datacite, …)
    result["work_indexed_in"] = [
        {"work_id": work_id, "index_name": str(index_name)}
        for index_name in (record.get("indexed_in") or [])
        if index_name
    ]

    # Awards (granular grant records with funder reference)
    result["work_awards"] = [
        {
            "work_id": work_id,
            "award_id": award_id,
            "display_name": award.get("display_name"),
            "funder_award_id": award.get("funder_award_id"),
            "funder_id": extract_id(award.get("funder_id")),
            "funder_display_name": award.get("funder_display_name"),
        }
        for award in (record.get("awards") or [])
        if (award_id := extract_id(award.get("id"))) is not None
    ]

    # Abstract inverted index (stored as JSON string; one row per work)
    aii = record.get("abstract_inverted_index")
    if aii:
        result.setdefault("work_abstracts", []).append({
            "work_id": work_id,
            "abstract_inverted_index": _json_dumps(aii),
        })

    return result


def _extract_author_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract institution affiliations, last-known institutions, and
    top topics from an author record."""
    author_id = extract_id(record.get("id"))
    if author_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    # Affiliations (full history)
    result["author_institutions"] = [
        {"author_id": author_id, "institution_id": inst_id}
        for aff in (record.get("affiliations") or [])
        if (inst_id := extract_id((aff.get("institution") or {}).get("id"))) is not None
    ]

    # Last-known institutions (most recent affiliations only)
    result["author_last_known_institutions"] = [
        {"author_id": author_id, "institution_id": inst_id}
        for inst in (record.get("last_known_institutions") or [])
        if (inst_id := extract_id(inst.get("id"))) is not None
    ]

    # Top topics (derived/aggregated by OpenAlex from author's works)
    result["author_topics"] = [
        {
            "author_id": author_id,
            "topic_id": topic_id,
            "count": int(topic.get("count", 0)),
            "score": float(topic.get("score", 0.0)),
        }
        for topic in (record.get("topics") or [])
        if (topic_id := extract_id(topic.get("id"))) is not None
    ]

    # Counts by year (time-series)
    for cby in (record.get("counts_by_year") or []):
        year = cby.get("year")
        if year is not None:
            result.setdefault("author_counts_by_year", []).append({
                "author_id": author_id,
                "year": int(year),
                "works_count": int(cby.get("works_count", 0)),
                "cited_by_count": int(cby.get("cited_by_count", 0)),
                "oa_works_count": int(cby.get("oa_works_count", 0)),
            })

    # External identifiers
    result["author_external_ids"] = [
        {"author_id": author_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    # Display-name alternatives
    result["author_name_alternatives"] = [
        {"author_id": author_id, "display_name_alternative": str(alt)}
        for alt in (record.get("display_name_alternatives") or [])
        if alt
    ]

    # Topic share (proportional contribution per topic)
    for ts in (record.get("topic_share") or []):
        topic_id = extract_id(ts.get("id"))
        if topic_id is not None:
            result.setdefault("author_topic_share", []).append({
                "author_id": author_id,
                "topic_id": topic_id,
                "value": float(ts.get("value", 0.0)),
            })

    # Sources where the author publishes
    for src in (record.get("sources") or []):
        src_id = extract_id(src.get("id"))
        if src_id is not None:
            result.setdefault("author_sources", []).append({
                "author_id": author_id,
                "source_id": src_id,
                "is_core": bool(src.get("is_core", False)) if src.get("is_core") is not None else None,
                "is_in_doaj": bool(src.get("is_in_doaj", False)) if src.get("is_in_doaj") is not None else None,
            })

    # x_concepts (deprecated concept taxonomy)
    for concept in (record.get("x_concepts") or []):
        concept_id = extract_id(concept.get("id"))
        if concept_id is not None:
            result.setdefault("author_concepts", []).append({
                "author_id": author_id,
                "concept_id": concept_id,
                "score": float(concept.get("score", 0.0)),
                "count": int(concept.get("count", 0)),
                "level": int(concept.get("level", 0)),
            })

    return result


# ── Roles prefix mapping ──────────────────────────────────────────────

_ROLE_TYPE_TO_PREFIX: dict[str, str] = {
    "funder": "F",
    "publisher": "P",
    "institution": "I",
}


def _extract_roles(
    record: dict,
    entity_id_key: str,
    entity_id: int,
) -> list[dict]:
    """Extract role rows from an entity record's roles[] array."""
    rows: list[dict] = []
    for role in (record.get("roles") or []):
        role_id_url = role.get("id")
        role_entity_id = extract_id(role_id_url)
        role_type = role.get("role")
        if role_entity_id is not None and role_type:
            rows.append({
                entity_id_key: entity_id,
                "role_entity_id": role_entity_id,
                "role_type": role_type,
                "role_prefix": _ROLE_TYPE_TO_PREFIX.get(role_type, ""),
            })
    return rows


def _extract_source_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract host lineage, top topics, and societies from a source record."""
    source_id = extract_id(record.get("id"))
    if source_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    # Host organisation lineage (chain up to publisher)
    for lineage_url in (record.get("host_organization_lineage") or []):
        pub_id = extract_id(lineage_url)
        if pub_id is not None:
            result.setdefault("source_host_lineage", []).append({
                "source_id": source_id,
                "publisher_id": pub_id,
            })

    # Top topics (derived/aggregated from published works)
    for topic in (record.get("topics") or []):
        topic_id = extract_id(topic.get("id"))
        if topic_id is not None:
            result.setdefault("source_topics", []).append({
                "source_id": source_id,
                "topic_id": topic_id,
                "count": int(topic.get("count", 0)),
                "score": float(topic.get("score", 0.0)),
            })

    # Societies (free-form organisation partnerships)
    for society in (record.get("societies") or []):
        org = society.get("organization")
        if org:
            result.setdefault("source_societies", []).append({
                "source_id": source_id,
                "organization": org,
                "url": society.get("url"),
            })

    # Counts by year (time-series)
    for cby in (record.get("counts_by_year") or []):
        year = cby.get("year")
        if year is not None:
            result.setdefault("source_counts_by_year", []).append({
                "source_id": source_id,
                "year": int(year),
                "works_count": int(cby.get("works_count", 0)),
                "cited_by_count": int(cby.get("cited_by_count", 0)),
                "oa_works_count": int(cby.get("oa_works_count", 0)),
            })

    # External identifiers (openalex, mag, wikidata, issn_l)
    for src, value in (record.get("ids") or {}).items():
        # The `issn` field in ids is an array, handled separately below
        if src == "issn":
            continue
        if value:
            result.setdefault("source_external_ids", []).append({
                "source_id": source_id,
                "source": src,
                "value": str(value),
            })

    # ISSNs (multiple per source)
    result["source_issns"] = [
        {"source_id": source_id, "issn": str(issn)}
        for issn in (record.get("issn") or [])
        if issn
    ]

    # APC prices (multi-currency)
    result["source_apc_prices"] = [
        {"source_id": source_id, "price": float(price), "currency": apc.get("currency")}
        for apc in (record.get("apc_prices") or [])
        if (price := apc.get("price")) is not None
    ]

    # Alternate titles
    result["source_alternate_titles"] = [
        {"source_id": source_id, "title": str(alt)}
        for alt in (record.get("alternate_titles") or [])
        if alt
    ]

    # Topic share (proportional contribution per topic)
    result["source_topic_share"] = [
        {"source_id": source_id, "topic_id": topic_id, "value": float(ts.get("value", 0.0))}
        for ts in (record.get("topic_share") or [])
        if (topic_id := extract_id(ts.get("id"))) is not None
    ]

    return result


def _extract_institution_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract associations, repositories, roles, lineage, and top topics
    from an institution record."""
    inst_id = extract_id(record.get("id"))
    if inst_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    # Associated institutions (peer relationships)
    result["institution_associations"] = [
        {
            "institution_id": inst_id,
            "associated_institution_id": assoc_id,
            "relationship_type": assoc.get("relationship"),
        }
        for assoc in (record.get("associated_institutions") or [])
        if (assoc_id := extract_id(assoc.get("id"))) is not None
    ]

    # Repositories (sources hosted by this institution)
    result["institution_repositories"] = [
        {"institution_id": inst_id, "source_id": repo_id}
        for repo in (record.get("repositories") or [])
        if (repo_id := extract_id(repo.get("id"))) is not None
    ]

    # Roles
    roles = _extract_roles(record, "institution_id", inst_id)
    if roles:
        result["institution_roles"] = roles

    # Lineage (ancestor institutions, e.g. faculty → university)
    result["institution_lineage"] = [
        {"institution_id": inst_id, "ancestor_institution_id": ancestor_id}
        for lineage_url in (record.get("lineage") or [])
        if (ancestor_id := extract_id(lineage_url)) is not None and ancestor_id != inst_id
    ]

    # Top topics (derived/aggregated from member-author works)
    result["institution_topics"] = [
        {
            "institution_id": inst_id,
            "topic_id": topic_id,
            "count": int(topic.get("count", 0)),
            "score": float(topic.get("score", 0.0)),
        }
        for topic in (record.get("topics") or [])
        if (topic_id := extract_id(topic.get("id"))) is not None
    ]

    # Counts by year (time-series)
    result["institution_counts_by_year"] = [
        {
            "institution_id": inst_id,
            "year": int(year),
            "works_count": int(cby.get("works_count", 0)),
            "cited_by_count": int(cby.get("cited_by_count", 0)),
            "oa_works_count": int(cby.get("oa_works_count", 0)),
        }
        for cby in (record.get("counts_by_year") or [])
        if (year := cby.get("year")) is not None
    ]

    # External identifiers
    result["institution_external_ids"] = [
        {"institution_id": inst_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    # Name alternatives + acronyms
    result["institution_name_alternatives"] = [
        {"institution_id": inst_id, "display_name_alternative": str(alt)}
        for alt in (record.get("display_name_alternatives") or [])
        if alt
    ]
    result["institution_name_acronyms"] = [
        {"institution_id": inst_id, "display_name_acronym": str(acr)}
        for acr in (record.get("display_name_acronyms") or [])
        if acr
    ]

    # Topic share
    result["institution_topic_share"] = [
        {"institution_id": inst_id, "topic_id": topic_id, "value": float(ts.get("value", 0.0))}
        for ts in (record.get("topic_share") or [])
        if (topic_id := extract_id(ts.get("id"))) is not None
    ]

    return result


def _extract_publisher_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract lineage, roles, and country codes from a publisher record."""
    pub_id = extract_id(record.get("id"))
    if pub_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    # Lineage (ancestor publishers)
    result["publisher_lineage"] = [
        {"publisher_id": pub_id, "ancestor_publisher_id": ancestor_id}
        for lineage_url in (record.get("lineage") or [])
        if (ancestor_id := extract_id(lineage_url)) is not None and ancestor_id != pub_id
    ]

    # Roles
    roles = _extract_roles(record, "publisher_id", pub_id)
    if roles:
        result["publisher_roles"] = roles

    # Country codes (publishers can operate from multiple countries)
    result["publisher_countries"] = [
        {"publisher_id": pub_id, "country_code": cc}
        for cc in (record.get("country_codes") or [])
        if cc
    ]

    # Counts by year
    result["publisher_counts_by_year"] = [
        {
            "publisher_id": pub_id,
            "year": int(year),
            "works_count": int(cby.get("works_count", 0)),
            "cited_by_count": int(cby.get("cited_by_count", 0)),
        }
        for cby in (record.get("counts_by_year") or [])
        if (year := cby.get("year")) is not None
    ]

    # External identifiers
    result["publisher_external_ids"] = [
        {"publisher_id": pub_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    # Alternate titles
    result["publisher_alternate_titles"] = [
        {"publisher_id": pub_id, "title": str(alt)}
        for alt in (record.get("alternate_titles") or [])
        if alt
    ]

    return result


def _extract_funder_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract roles, counts-by-year, external IDs, alternate titles."""
    funder_id = extract_id(record.get("id"))
    if funder_id is None:
        return {}
    result: dict[str, list[dict]] = {}
    roles = _extract_roles(record, "funder_id", funder_id)
    if roles:
        result["funder_roles"] = roles

    # Counts by year
    result["funder_counts_by_year"] = [
        {
            "funder_id": funder_id,
            "year": int(year),
            "works_count": int(cby.get("works_count", 0)),
            "cited_by_count": int(cby.get("cited_by_count", 0)),
            "oa_works_count": int(cby.get("oa_works_count", 0)),
        }
        for cby in (record.get("counts_by_year") or [])
        if (year := cby.get("year")) is not None
    ]

    # External identifiers
    result["funder_external_ids"] = [
        {"funder_id": funder_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    # Alternate titles
    result["funder_alternate_titles"] = [
        {"funder_id": funder_id, "title": str(alt)}
        for alt in (record.get("alternate_titles") or [])
        if alt
    ]

    return result


def _extract_topic_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract hierarchical links, keywords, and external IDs from a topic record."""
    topic_id = extract_id(record.get("id"))
    if topic_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    subfield = record.get("subfield") or {}
    sf_id = extract_id(subfield.get("id"))
    if sf_id is not None:
        result["topic_subfields"] = [{"topic_id": topic_id, "subfield_id": sf_id}]

    field = record.get("field") or {}
    f_id = extract_id(field.get("id"))
    if f_id is not None:
        result["topic_fields"] = [{"topic_id": topic_id, "field_id": f_id}]

    domain = record.get("domain") or {}
    d_id = extract_id(domain.get("id"))
    if d_id is not None:
        result["topic_domains"] = [{"topic_id": topic_id, "domain_id": d_id}]

    result["topic_keywords"] = [
        {"topic_id": topic_id, "keyword": str(kw)}
        for kw in (record.get("keywords") or [])
        if kw
    ]

    result["topic_external_ids"] = [
        {"topic_id": topic_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    return result


def _extract_subfield_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract hierarchical links, external IDs, and name alternatives."""
    sf_id = extract_id(record.get("id"))
    if sf_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    field = record.get("field") or {}
    f_id = extract_id(field.get("id"))
    if f_id is not None:
        result["subfield_fields"] = [{"subfield_id": sf_id, "field_id": f_id}]

    domain = record.get("domain") or {}
    d_id = extract_id(domain.get("id"))
    if d_id is not None:
        result["subfield_domains"] = [{"subfield_id": sf_id, "domain_id": d_id}]

    result["subfield_external_ids"] = [
        {"subfield_id": sf_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    result["subfield_name_alternatives"] = [
        {"subfield_id": sf_id, "display_name_alternative": str(alt)}
        for alt in (record.get("display_name_alternatives") or [])
        if alt
    ]

    return result


def _extract_field_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract hierarchical link, external IDs, and name alternatives."""
    f_id = extract_id(record.get("id"))
    if f_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    domain = record.get("domain") or {}
    d_id = extract_id(domain.get("id"))
    if d_id is not None:
        result["field_domains"] = [{"field_id": f_id, "domain_id": d_id}]

    result["field_external_ids"] = [
        {"field_id": f_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    result["field_name_alternatives"] = [
        {"field_id": f_id, "display_name_alternative": str(alt)}
        for alt in (record.get("display_name_alternatives") or [])
        if alt
    ]

    return result


def _extract_domain_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract external IDs and name alternatives from a domain record."""
    d_id = extract_id(record.get("id"))
    if d_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    result["domain_external_ids"] = [
        {"domain_id": d_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    result["domain_name_alternatives"] = [
        {"domain_id": d_id, "display_name_alternative": str(alt)}
        for alt in (record.get("display_name_alternatives") or [])
        if alt
    ]

    return result


def _extract_sdg_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract external IDs from an SDG record."""
    sdg_id = extract_id(record.get("id"))
    if sdg_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    result["sdg_external_ids"] = [
        {"sdg_id": sdg_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    return result


def _append_investigator(
    result: dict[str, list[dict]],
    award_id: int,
    inv: dict,
    *,
    is_lead: bool,
    is_co_lead: bool,
) -> None:
    """Append an investigator row and any nested affiliation IDs.

    Affiliation can be a string or a nested dict ({name, country, ids[]});
    flatten name/country and split the IDs into a separate sub-table.
    """
    if not inv:
        return
    aff = inv.get("affiliation")
    aff_name = None
    aff_country = None
    aff_id_rows: list[dict] = []
    if isinstance(aff, dict):
        aff_name = aff.get("name")
        aff_country = aff.get("country")
        aff_id_rows = [
            {
                "award_id": award_id,
                "investigator_orcid": inv.get("orcid"),
                "investigator_family_name": inv.get("family_name"),
                "affiliation_id": aid.get("id"),
                "affiliation_type": aid.get("type"),
                "asserted_by": aid.get("asserted_by"),
            }
            for aid in (aff.get("ids") or [])
            if isinstance(aid, dict)
        ]
    elif isinstance(aff, str):
        aff_name = aff

    result.setdefault("award_investigators", []).append({
        "award_id": award_id,
        "given_name": inv.get("given_name"),
        "family_name": inv.get("family_name"),
        "orcid": inv.get("orcid"),
        "affiliation_name": aff_name,
        "affiliation_country": aff_country,
        "role_start": inv.get("role_start"),
        "is_lead": is_lead,
        "is_co_lead": is_co_lead,
    })
    if aff_id_rows:
        result.setdefault("award_investigator_affiliations", []).extend(aff_id_rows)


def _extract_continent_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract member countries, external IDs, name alternatives from a continent."""
    raw_id = record.get("id") or ""
    continent_id = raw_id.rsplit("/", 1)[-1] if raw_id else None
    if not continent_id:
        return {}
    result: dict[str, list[dict]] = {}

    result["continent_countries"] = [
        {"continent_id": continent_id, "country_id": str(url).rsplit("/", 1)[-1]}
        for raw_url in (record.get("countries") or [])
        for url in [raw_url.get("id") if isinstance(raw_url, dict) else raw_url]
        if url
    ]

    result["continent_external_ids"] = [
        {"continent_id": continent_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    result["continent_name_alternatives"] = [
        {"continent_id": continent_id, "display_name_alternative": str(alt)}
        for alt in (record.get("display_name_alternatives") or [])
        if alt
    ]

    return result


def _extract_country_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract external IDs and name alternatives from a country record."""
    raw_id = record.get("id") or ""
    country_id = raw_id.rsplit("/", 1)[-1] if raw_id else None
    if not country_id:
        return {}
    result: dict[str, list[dict]] = {}

    result["country_external_ids"] = [
        {"country_id": country_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    result["country_name_alternatives"] = [
        {"country_id": country_id, "display_name_alternative": str(alt)}
        for alt in (record.get("display_name_alternatives") or [])
        if alt
    ]

    return result


def _extract_award_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract investigators (and their affiliation IDs) plus funded outputs."""
    award_id = extract_id(record.get("id"))
    if award_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    _append_investigator(result, award_id, record.get("lead_investigator") or {},
                         is_lead=True, is_co_lead=False)
    _append_investigator(result, award_id, record.get("co_lead_investigator") or {},
                         is_lead=False, is_co_lead=True)
    for inv in (record.get("investigators") or []):
        if isinstance(inv, dict):
            _append_investigator(result, award_id, inv, is_lead=False, is_co_lead=False)

    result["award_funded_outputs"] = [
        {"award_id": award_id, "work_id": wid}
        for out_url in (record.get("funded_outputs") or [])
        if (wid := extract_id(out_url)) is not None
    ]

    return result


def _extract_concept_relationships(record: dict) -> dict[str, list[dict]]:
    """Extract ancestors and related concepts from a concept record."""
    concept_id = extract_id(record.get("id"))
    if concept_id is None:
        return {}
    result: dict[str, list[dict]] = {}

    # Ancestors
    result["concept_ancestors"] = [
        {"concept_id": concept_id, "ancestor_concept_id": ancestor_id}
        for ancestor in (record.get("ancestors") or [])
        if (ancestor_id := extract_id(ancestor.get("id"))) is not None
    ]

    # Related concepts
    result["concept_related"] = [
        {
            "concept_id": concept_id,
            "related_concept_id": related_id,
            "score": related.get("score", 0.0),
        }
        for related in (record.get("related_concepts") or [])
        if (related_id := extract_id(related.get("id"))) is not None
    ]

    # Counts by year
    result["concept_counts_by_year"] = [
        {
            "concept_id": concept_id,
            "year": int(year),
            "works_count": int(cby.get("works_count", 0)),
            "cited_by_count": int(cby.get("cited_by_count", 0)),
            "oa_works_count": int(cby.get("oa_works_count", 0)),
        }
        for cby in (record.get("counts_by_year") or [])
        if (year := cby.get("year")) is not None
    ]

    # External identifiers
    result["concept_external_ids"] = [
        {"concept_id": concept_id, "source": source, "value": str(value)}
        for source, value in (record.get("ids") or {}).items()
        if value
    ]

    return result


# ── Provenance ──────────────────────────────────────────────────────────


def _git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None



# ── Source-file key & deterministic output identity ─────────────────────


def _source_file_key(source_path: Path) -> str:
    """Derive a deterministic output key from a source file path.

    The key is the path relative to ``SNAPSHOT_DIR`` with ``/`` replaced
    by ``__`` and the ``.gz`` extension stripped.  This is human-readable
    and collision-free within a single entity type.

    Example::

        works/updated_date=2026-01-09/part_0047.jsonl.gz
        → works__updated_date=2026-01-09__part_0047

    Workers with any ``--workers`` value produce the same shard name
    for the same source file, making output identity independent of
    scheduling.
    """
    try:
        rel = str(source_path.relative_to(SNAPSHOT_DIR))
    except ValueError:
        rel = str(source_path)
    # Strip .jsonl.gz or .gz suffix (the output is parquet, not gzip)
    if rel.endswith(".jsonl.gz"):
        rel = rel[:-9]
    elif rel.endswith(".gz"):
        rel = rel[:-3]
    elif rel.endswith(".jsonl"):
        rel = rel[:-6]
    return rel.replace("/", "__")


def _shard_path(output_dir: Path, source_key: str) -> Path:
    """Return the parquet shard path for a given source-file key."""
    return output_dir / f"{source_key}.parquet"


# ── Per-unit provenance ─────────────────────────────────────────────────


def _write_unit_provenance(
    output_dir: Path,
    *,
    source_key: str,
    source_file: str,
    content_length: int,
    row_count: int,
    status: str,
    relationship_type: str,
    output_hash: str | None = None,
    skipped: bool = False,
) -> None:
    """Write provenance for a single source-file unit.

    Stored as ``_units/{source_key}.json`` inside the relationship-type
    output directory.  Each unit record contains:

    - ``source_key``      — deterministic output key
    - ``source_file``     — original relative path
    - ``content_length``  — byte size of the source gzip file
    - ``row_count``       — rows extracted from this file
    - ``output_hash``     — SHA-256 (16 hex chars) of the parquet shard
    - ``status``          — ``"complete"`` | ``"skipped"`` | ``"empty"``
    - ``relationship_type``
    - ``git_commit``
    - ``timestamp``
    """
    units_dir = output_dir / "_units"
    units_dir.mkdir(parents=True, exist_ok=True)
    prov = {
        "source_key": source_key,
        "source_file": source_file,
        "content_length": content_length,
        "row_count": row_count,
        "status": status,
        "relationship_type": relationship_type,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if output_hash:
        prov["output_hash"] = output_hash
    path = units_dir / f"{source_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)


def _completed_source_keys(output_dir: Path) -> set[str]:
    """Return source keys that have a valid parquet shard on disk.

    Scans for ``*.parquet`` files in *output_dir* (excluding ``_units/``
    subdirectory), validates each has a readable footer, and extracts
    the source key from the filename.
    """
    result: set[str] = set()
    if not output_dir.exists():
        return result
    for shard in output_dir.glob("*.parquet"):
        if shard.name.startswith("._"):
            continue
        try:
            pf = pq.ParquetFile(str(shard))
            _ = pf.metadata  # validate footer is readable
            result.add(shard.stem)
        except Exception:
            log.warning("Invalid parquet shard: %s, ignoring", shard.name)
    return result


def _count_shard_rows(output_dir: Path) -> tuple[int, int]:
    """Count total rows and number of valid shards in output_dir.

    Returns ``(total_rows, shard_count)``.
    """
    total_rows = 0
    shard_count = 0
    if not output_dir.exists():
        return 0, 0
    for shard in output_dir.glob("*.parquet"):
        if shard.name.startswith("._"):
            continue
        try:
            pf = pq.ParquetFile(str(shard))
            total_rows += pf.metadata.num_rows
            shard_count += 1
        except Exception:
            pass
    return total_rows, shard_count


# Legacy loader retained for migration compatibility
def _load_unit_provenances(output_dir: Path) -> dict[str, dict]:
    """Load legacy per-unit provenance records (migration only).

    Returns ``{source_key → provenance_dict}``.
    """
    units_dir = output_dir / "_units"
    if not units_dir.exists():
        return {}
    result: dict[str, dict] = {}
    _json_load = json.load
    _json_decode_error = json.JSONDecodeError
    for prov_path in sorted(units_dir.glob("*.json")):
        if prov_path.name.startswith("._"):
            continue
        try:
            with open(prov_path) as f:
                prov = _json_load(f)
            key = prov.get("source_key", prov_path.stem)
            result[key] = prov
        except (_json_decode_error, KeyError, OSError, UnicodeDecodeError):
            log.warning("Corrupt unit provenance: %s, ignoring", prov_path)
    return result


def _compute_pending_source_files(
    source_files: list[Path],
    completed_keys: set[str],
) -> list[Path]:
    """Return source files whose shard does not yet exist on disk.

    Compares ``_source_file_key(f)`` against *completed_keys* (derived
    from existing valid parquet shards).
    """
    return [f for f in source_files if _source_file_key(f) not in completed_keys]


# ── Aggregate provenance ────────────────────────────────────────────────


def _write_provenance(
    output_dir: Path,
    *,
    relationship_type: str,
    record_count: int,
    source_entity: str,
    source_file_count: int,
) -> None:
    """Write a _provenance.json file recording conversion metadata."""
    provenance = {
        "relationship_type": relationship_type,
        "record_count": record_count,
        "source_entity": source_entity,
        "source_file_count": source_file_count,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    provenance_path = output_dir / "_provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    log.info("Wrote provenance: %s", provenance_path)


def _finalise_type(
    rt_dir: Path,
    *,
    relationship_type: str,
    source_entity: str,
    source_file_count: int,
) -> int:
    """Finalise a relationship type after all units are processed.

    Scans parquet shards on disk, computes totals, writes aggregate
    ``_provenance.json``, ``_lineage.json``, and
    ``_manifest_snapshot.json``.  Returns the total row count.
    """
    total_rows, shard_count = _count_shard_rows(rt_dir)

    _write_provenance(
        rt_dir,
        relationship_type=relationship_type,
        record_count=total_rows,
        source_entity=source_entity,
        source_file_count=source_file_count,
    )

    # Write lineage: shard_name (one shard per source file)
    lineage: dict[str, list[str]] = {}
    for shard in sorted(rt_dir.glob("*.parquet")):
        if shard.name.startswith("._"):
            continue
        key = shard.stem
        lineage.setdefault("", []).append(f"{key}.parquet")
    _write_lineage(rt_dir, lineage)

    # Write manifest snapshot for drift detection
    current_manifest = _load_entity_manifest(source_entity)
    _write_manifest_snapshot(rt_dir, current_manifest)

    log.info(
        "%s: finalised — %d/%d shards, %d total rows",
        relationship_type, shard_count, source_file_count, total_rows,
    )
    return total_rows


# ── Manifest snapshot & drift detection ─────────────────────────────────


def _manifest_key_to_path(file_rel: str) -> Path:
    """Convert a manifest key (S3-style ``.gz``) to a local filesystem path.

    The manifest stores keys like ``works/updated_date=.../part_0000.gz``
    (matching the S3 bucket).  On disk we use ``.jsonl.gz`` so the HF
    dataset viewer detects the inner format.  This helper bridges the two.
    """
    if file_rel.endswith(".gz") and not file_rel.endswith(".jsonl.gz"):
        file_rel = file_rel[:-3] + ".jsonl.gz"
    return SNAPSHOT_DIR / file_rel


def _load_entity_manifest(entity_type: str) -> dict[str, dict]:
    """Load the entity manifest and return {file_rel → {content_length, record_count}}.

    *file_rel* is relative to SNAPSHOT_DIR (S3-style), e.g.
    ``works/updated_date=2026-01-09/part_0000.gz``.
    """
    manifest_path = SNAPSHOT_DIR / entity_type / "manifest"
    if not manifest_path.exists():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    result: dict[str, dict] = {}
    for entry in raw.get("entries", []):
        url: str = entry.get("url", "")
        if url.startswith("s3://openalex/data/"):
            file_rel = url[len("s3://openalex/data/"):]
        else:
            continue
        meta = entry.get("meta", {})
        result[file_rel] = {
            "content_length": meta.get("content_length", 0),
            "record_count": meta.get("record_count", 0),
        }
    return result


def _write_manifest_snapshot(
    output_dir: Path,
    manifest: dict[str, dict],
) -> None:
    """Write a _manifest_snapshot.json for drift detection on future runs."""
    path = output_dir / "_manifest_snapshot.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.debug("Wrote manifest snapshot: %s (%d files)", path, len(manifest))


def _load_manifest_snapshot(output_dir: Path) -> dict[str, dict]:
    """Load a previously stored manifest snapshot."""
    path = output_dir / "_manifest_snapshot.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _detect_manifest_drift(
    current: dict[str, dict],
    previous: dict[str, dict],
) -> dict[str, str]:
    """Compare current manifest with previous snapshot.

    Returns ``{file_rel → drift_type}`` where *drift_type* is one of:

    - ``"added"``     — new file not in previous snapshot
    - ``"removed"``   — file gone from current manifest
    - ``"changed"``   — content_length differs
    """
    drift: dict[str, str] = {}
    all_keys = set(current) | set(previous)
    for key in sorted(all_keys):
        if key not in previous:
            drift[key] = "added"
        elif key not in current:
            drift[key] = "removed"
        elif current[key]["content_length"] != previous[key].get("content_length", -1):
            drift[key] = "changed"
    return drift


def _write_lineage(
    output_dir: Path,
    lineage: dict[str, list[str]],
) -> None:
    """Write a consolidated _lineage.json alongside the parquet files."""
    path = output_dir / "_lineage.json"
    lineage_sorted = {k: sorted(v) for k, v in sorted(lineage.items())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lineage_sorted, f, indent=2)
    log.debug("Wrote lineage: %s (%d source files)", path, len(lineage_sorted))


def _load_lineage(output_dir: Path) -> dict[str, list[str]]:
    """Load a previously stored lineage map."""
    path = output_dir / "_lineage.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ── Parquet writing — one shard per source file per type ─────────────────


def _extract_one_source_file(
    source_file: Path,
    entity_type: str,
    rel_types: frozenset[str],
    batch_size: int,
    output_base: Path,
    staging_dir: Path | None,
) -> dict[str, dict]:
    """Extract all relationship types from a single source file.

    For each relationship type, writes exactly one parquet shard named
    ``{source_key}.parquet`` and one unit provenance record.  The output
    shard name is derived from the source file path, so it is
    deterministic regardless of worker assignment.

    Returns ``{rel_type → {"source_key": str, "row_count": int}}``.
    """
    source_key = _source_file_key(source_file)
    try:
        source_file_rel = str(source_file.relative_to(SNAPSHOT_DIR))
    except ValueError:
        source_file_rel = str(source_file)

    content_length = source_file.stat().st_size if source_file.exists() else 0

    _, extractor = _ENTITY_DISPATCH[entity_type]

    # Open one writer per relationship type for this source file
    writers: dict[str, _SourceFileWriter] = {}
    for rt in rel_types:
        out_dir = rt_dir(output_base, rt)
        writers[rt] = _SourceFileWriter(
            out_dir, rt, source_key, staging_dir=staging_dir,
        )

    buffers: dict[str, list[dict]] = {rt: [] for rt in rel_types}
    record_count = 0

    # Cache globals for inner-loop performance
    _buffers = buffers
    _writers = writers

    for record in iter_jsonl(source_file):
        rels = extractor(record)
        for rt, rows in rels.items():
            buf = _buffers.get(rt)
            if buf is None:
                continue
            buf.extend(rows)
            if len(buf) >= batch_size:
                _writers[rt].write_batch(buf)
                _buffers[rt] = []
        record_count += 1

    results: dict[str, dict] = {}
    for rt in rel_types:
        if buffers[rt]:
            writers[rt].write_batch(buffers[rt])
        writers[rt].close()

        row_count = writers[rt].total_count
        results[rt] = {"source_key": source_key, "row_count": row_count}

    return results


def _hash_file(path: Path) -> str:
    """SHA-256 of file contents, truncated to 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class _SourceFileWriter:
    """Manages a single ParquetWriter for one source file × one type.

    Output filename is always ``{source_key}.parquet`` — deterministic
    regardless of which worker processes the file.
    """

    output_hash: str | None = None

    def __init__(
        self,
        output_dir: Path,
        rel_type: str,
        source_key: str,
        staging_dir: Path | None = None,
    ) -> None:
        self.rel_type = rel_type
        self.schema = RELATIONSHIP_SCHEMAS[rel_type]
        self.output_dir = output_dir
        self.staging_dir = staging_dir
        self.source_key = source_key
        self.total_count = 0
        self._writer: pq.ParquetWriter | None = None
        self._shard_name = f"{source_key}.parquet"
        self._current_path: Path | None = None

    def _ensure_writer(self) -> pq.ParquetWriter:
        if self._writer is None:
            write_dir = self.staging_dir if self.staging_dir else self.output_dir
            write_dir.mkdir(parents=True, exist_ok=True)
            out_path = write_dir / self._shard_name
            self._writer = pq.ParquetWriter(out_path, self.schema, compression="snappy")
            self._current_path = out_path
        return self._writer

    def write_batch(self, rows: list[dict]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=self.schema)
        writer = self._ensure_writer()
        writer.write_table(table)
        self.total_count += len(rows)
        log.info(
            "%s: wrote row group (%d rows, %d total)",
            self.rel_type, len(rows), self.total_count,
        )

    def close(self) -> None:
        """Close writer, stage-to-final move, compute output hash."""
        # Always create a file — even with 0 rows — so the shard exists
        # on disk as a completion marker (replaces _units/ provenance).
        if self._writer is None:
            write_dir = self.staging_dir if self.staging_dir else self.output_dir
            write_dir.mkdir(parents=True, exist_ok=True)
            out_path = write_dir / self._shard_name
            # Write schema-only parquet (0 rows)
            with pq.ParquetWriter(out_path, self.schema, compression="snappy") as w:
                pass  # empty file with schema
            self._current_path = out_path

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        if self._current_path is not None:
            if self.staging_dir and self._current_path.parent != self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                dest = self.output_dir / self._current_path.name
                shutil.move(str(self._current_path), str(dest))
                self._current_path = dest
            # Compute content hash of the written parquet file
            self.output_hash = _hash_file(self._current_path)
            log.info(
                "%s: closed %s (%d rows, hash=%s)",
                self.rel_type, self._current_path.name, self.total_count,
                self.output_hash,
            )


# ── Worker entry point (must be module-level for multiprocessing) ───────


def _worker_process_files(
    args: tuple[int, list[Path], str, list[str], int],
) -> dict:
    """Process a list of source files in one worker process.

    Each source file is processed independently — the worker opens and
    closes one writer per file per relationship type.  Output identity
    is ``{source_file_key}.parquet``, independent of worker_id.

    Returns ``{"worker_id": int, "results": {source_key: {rt: row_count}}}``.
    """
    worker_id, source_files, entity_type, rel_types_list, batch_size, type_completed_keys = args
    rel_types = frozenset(rel_types_list)

    log.info("[w%02d] processing %d source files", worker_id, len(source_files))

    # Cache globals for loop performance
    _extract = _extract_one_source_file
    _snapshot_dir = SNAPSHOT_DIR
    _staging_dir = STAGING_DIR
    _tck = type_completed_keys

    all_results: dict[str, dict[str, int]] = {}
    for source_file in source_files:
        source_key = _source_file_key(source_file)

        # Skip types already completed for this source key
        pending_types = frozenset(
            rt for rt in rel_types
            if source_key not in _tck.get(rt, set())
        )
        if not pending_types:
            all_results[source_key] = {}
            continue

        file_results = _extract(
            source_file,
            entity_type,
            pending_types,
            batch_size,
            _snapshot_dir,
            _staging_dir,
        )
        all_results[source_key] = {
            rt: info["row_count"] for rt, info in file_results.items()
        }

    return {
        "worker_id": worker_id,
        "results": all_results,
    }


# ── Relationship type sets per source entity ────────────────────────────
_WORK_RELATIONSHIP_TYPES = frozenset({
    "work_authorships",
    "work_authorship_institutions",
    "work_references",
    "work_topics",
    "work_concepts",
    "work_locations",
    "work_related",
    "work_funders",
    "work_keywords",
    "work_sdgs",
    "work_mesh",
    "work_corresponding_authors",
    "work_corresponding_institutions",
    # Lossless projection (replaces _json column)
    "work_counts_by_year",
    "work_external_ids",
    "work_indexed_in",
    "work_awards",
    "work_abstracts",
})

_AUTHOR_RELATIONSHIP_TYPES = frozenset({
    "author_institutions",
    "author_last_known_institutions",
    "author_topics",
    # Lossless projection (replaces _json column)
    "author_counts_by_year",
    "author_external_ids",
    "author_name_alternatives",
    "author_topic_share",
    "author_sources",
    "author_concepts",
})

_SOURCE_RELATIONSHIP_TYPES = frozenset({
    "source_host_lineage",
    "source_topics",
    "source_societies",
    # Lossless projection (replaces _json column)
    "source_counts_by_year",
    "source_external_ids",
    "source_issns",
    "source_apc_prices",
    "source_alternate_titles",
    "source_topic_share",
})

_INSTITUTION_RELATIONSHIP_TYPES = frozenset({
    "institution_associations",
    "institution_repositories",
    "institution_roles",
    "institution_lineage",
    "institution_topics",
    # Lossless projection (replaces _json column)
    "institution_counts_by_year",
    "institution_external_ids",
    "institution_name_alternatives",
    "institution_name_acronyms",
    "institution_topic_share",
})

_PUBLISHER_RELATIONSHIP_TYPES = frozenset({
    "publisher_lineage",
    "publisher_roles",
    "publisher_countries",
    "publisher_counts_by_year",
    "publisher_external_ids",
    "publisher_alternate_titles",
})

_FUNDER_RELATIONSHIP_TYPES = frozenset({
    "funder_roles",
    "funder_counts_by_year",
    "funder_external_ids",
    "funder_alternate_titles",
})

_CONCEPT_RELATIONSHIP_TYPES = frozenset({
    "concept_ancestors",
    "concept_related",
    "concept_counts_by_year",
    "concept_external_ids",
})

_TOPIC_RELATIONSHIP_TYPES = frozenset({
    "topic_subfields",
    "topic_fields",
    "topic_domains",
    "topic_keywords",
    "topic_external_ids",
})

_SUBFIELD_RELATIONSHIP_TYPES = frozenset({
    "subfield_fields",
    "subfield_domains",
    "subfield_external_ids",
    "subfield_name_alternatives",
})

_FIELD_RELATIONSHIP_TYPES = frozenset({
    "field_domains",
    "field_external_ids",
    "field_name_alternatives",
})

_DOMAIN_RELATIONSHIP_TYPES = frozenset({
    "domain_external_ids",
    "domain_name_alternatives",
})

_SDG_RELATIONSHIP_TYPES = frozenset({
    "sdg_external_ids",
})

_AWARD_RELATIONSHIP_TYPES = frozenset({
    "award_investigators",
    "award_investigator_affiliations",
    "award_funded_outputs",
})

_CONTINENT_RELATIONSHIP_TYPES = frozenset({
    "continent_countries",
    "continent_external_ids",
    "continent_name_alternatives",
})

_COUNTRY_RELATIONSHIP_TYPES = frozenset({
    "country_external_ids",
    "country_name_alternatives",
})

# Inferred (algorithmically derived) relationships — excluded by default.
# These are aggregated by OpenAlex from underlying primary data, not
# explicitly recorded as edges.
INFERRED_RELATIONSHIP_TYPES = frozenset({
    "work_topics",
    "work_concepts",
    "work_related",
    "work_keywords",
    "concept_related",
    "author_topics",
    "institution_topics",
    "source_topics",
})

# Core structural graph edges: citation links, authorship edges, institutional
# affiliations, funding links, locations.  These form the citation/collaboration
# graph and are the foundation for downstream graph-analytic methods.
# Everything NOT in this set and NOT inferred is an annotation/metadata table.
_GRAPH_RELATIONSHIP_TYPES = frozenset({
    # Works
    "work_authorships",
    "work_authorship_institutions",
    "work_references",
    "work_funders",
    "work_locations",
    "work_corresponding_institutions",
    "work_corresponding_authors",
    # Authors
    "author_institutions",
    "author_last_known_institutions",
    "author_sources",
    # Sources
    "source_host_lineage",
    # Institutions
    "institution_repositories",
    "institution_lineage",
    "institution_associations",
    # Publishers
    "publisher_lineage",
    # Concepts
    "concept_ancestors",
    "concept_related",
    # Topics
    "topic_subfields",
    "topic_fields",
    "topic_domains",
    # Subfields / Fields / Domains
    "subfield_fields",
    "subfield_domains",
    "field_domains",
    # Awards
    "award_investigators",
    "award_investigator_affiliations",
    "award_funded_outputs",
    # Geography
    "continent_countries",
})


# Rough row-count estimates for ordering: smallest first so quick types
# finish early and free disk space.  Exact values don't matter — only the
# relative ordering within each category (structural / inferred) does.
_SIZE_ESTIMATE: dict[str, int] = {
    # ── Works graph edges (structural core) ─────────────────────────────
    "work_corresponding_institutions": 1,
    "work_corresponding_authors":      2,
    "work_funders":                    3,
    "work_locations":                  4,
    "work_authorships":                5,
    "work_authorship_institutions":    6,  # same order as work_authorships
    "work_references":                 7,
    # ── Works annotation/metadata (non-structural) ──────────────────────
    "work_sdgs":                       8,
    "work_mesh":                       9,
    "work_awards":                    10,
    "work_indexed_in":                11,
    "work_counts_by_year":            12,
    "work_external_ids":              13,
    "work_abstracts":                 14,
    # ── Works inferred ──────────────────────────────────────────────────
    "work_keywords":                  15,
    "work_related":                   16,
    "work_concepts":                  17,
    "work_topics":                    18,
    # ── Authors ─────────────────────────────────────────────────────────
    "author_name_alternatives":        1,
    "author_external_ids":             2,
    "author_institutions":            3,
    "author_last_known_institutions": 4,
    "author_counts_by_year":          5,
    "author_sources":                 6,
    "author_topic_share":             7,
    "author_concepts":                8,
    "author_topics":                  9,
    # ── Sources ─────────────────────────────────────────────────────────
    "source_societies":                1,
    "source_issns":                    2,
    "source_alternate_titles":         3,
    "source_external_ids":             4,
    "source_host_lineage":             5,
    "source_counts_by_year":           6,
    "source_topic_share":              7,
    "source_topics":                   8,
    # ── Institutions ────────────────────────────────────────────────────
    "institution_name_acronyms":       1,
    "institution_name_alternatives":   2,
    "institution_external_ids":        3,
    "institution_repositories":        4,
    "institution_roles":               5,
    "institution_associations":        6,
    "institution_lineage":             7,
    "institution_counts_by_year":      8,
    "institution_topic_share":         9,
    "institution_topics":             10,
    # ── Other entities (small volumes, alphabetical is fine) ────────────
    "publisher_alternate_titles":      1,
    "publisher_external_ids":          2,
    "publisher_countries":             3,
    "publisher_roles":                 4,
    "publisher_lineage":               5,
    "publisher_counts_by_year":        6,
    "funder_alternate_titles":         1,
    "funder_external_ids":             2,
    "funder_roles":                    3,
    "funder_counts_by_year":           4,
    "concept_ancestors":               1,
    "concept_external_ids":            2,
    "concept_counts_by_year":          3,
    "concept_related":                 4,
    "topic_keywords":                  1,
    "topic_external_ids":              2,
    "topic_subfields":                 3,
    "topic_fields":                    4,
    "topic_domains":                   5,
    "subfield_name_alternatives":      1,
    "subfield_external_ids":           2,
    "subfield_fields":                 3,
    "subfield_domains":                4,
    "field_name_alternatives":         1,
    "field_external_ids":              2,
    "field_domains":                   3,
    "domain_name_alternatives":        1,
    "domain_external_ids":             2,
    "sdg_external_ids":                1,
    "award_investigators":             1,
    "award_investigator_affiliations": 2,
    "award_funded_outputs":            3,
    "continent_name_alternatives":     1,
    "continent_countries":             2,
    "continent_external_ids":          3,
    "country_name_alternatives":       1,
    "country_external_ids":            2,
}


def _order_rel_types(rel_types: frozenset[str]) -> list[str]:
    """Order: graph edges (smallest first), then annotations, then inferred."""
    def sort_key(rt: str) -> tuple[int, int, int]:
        if rt in _GRAPH_RELATIONSHIP_TYPES:
            tier = 0
        elif rt in INFERRED_RELATIONSHIP_TYPES:
            tier = 2
        else:
            tier = 1  # annotation / metadata
        return (tier, _SIZE_ESTIMATE.get(rt, 999), hash(rt))
    return sorted(rel_types, key=sort_key)


_ENTITY_DISPATCH: dict[str, tuple[frozenset[str], object]] = {
    "works":        (_WORK_RELATIONSHIP_TYPES,        _extract_work_relationships),
    "authors":      (_AUTHOR_RELATIONSHIP_TYPES,      _extract_author_relationships),
    "sources":      (_SOURCE_RELATIONSHIP_TYPES,      _extract_source_relationships),
    "institutions": (_INSTITUTION_RELATIONSHIP_TYPES, _extract_institution_relationships),
    "publishers":   (_PUBLISHER_RELATIONSHIP_TYPES,   _extract_publisher_relationships),
    "funders":      (_FUNDER_RELATIONSHIP_TYPES,      _extract_funder_relationships),
    "concepts":     (_CONCEPT_RELATIONSHIP_TYPES,     _extract_concept_relationships),
    "topics":       (_TOPIC_RELATIONSHIP_TYPES,       _extract_topic_relationships),
    "subfields":    (_SUBFIELD_RELATIONSHIP_TYPES,    _extract_subfield_relationships),
    "fields":       (_FIELD_RELATIONSHIP_TYPES,       _extract_field_relationships),
    "domains":      (_DOMAIN_RELATIONSHIP_TYPES,      _extract_domain_relationships),
    "sdgs":         (_SDG_RELATIONSHIP_TYPES,         _extract_sdg_relationships),
    "awards":       (_AWARD_RELATIONSHIP_TYPES,       _extract_award_relationships),
    "continents":   (_CONTINENT_RELATIONSHIP_TYPES,   _extract_continent_relationships),
    "countries":    (_COUNTRY_RELATIONSHIP_TYPES,     _extract_country_relationships),
}


# ── Main conversion ────────────────────────────────────────────────────


def convert_relationships(
    entity_type: str,
    *,
    force: bool = False,
    exclude: frozenset[str] | None = None,
    include_inferred: bool = True,
    workers: int | None = None,
    batch_size: int | None = None,
    slice_index: int | None = None,
    slice_total: int | None = None,
) -> dict[str, int]:
    """Convert nested JSONL arrays to relationship Parquet tables.

    **Deterministic output identity.**  Each source file produces exactly
    one output shard per relationship type, named ``{source_key}.parquet``
    where *source_key* is derived from the file's path relative to
    ``SNAPSHOT_DIR``.  Worker count only affects scheduling — the same
    source file always produces the same output shard.

    **Incremental & resumable.**  Per-unit provenance records track
    completion status for each source file.  On restart, only source
    files without a ``"complete"`` unit record are reprocessed.

    **Manifest drift detection.**  When a type is fully complete, the
    entity manifest is snapshotted.  On the next run, the current
    manifest is compared against the snapshot — changed or removed files
    trigger targeted re-extraction of only affected units.

    Args:
        entity_type: The source entity type (e.g. "works", "authors").
        force: If True, re-extract all units regardless of provenance.
        exclude: Relationship type names to skip entirely.
        include_inferred: If False, skip inferred relationship types.
        workers: Number of parallel worker processes per type.
        batch_size: Rows per parquet row-group flush.

    Returns:
        Mapping of relationship type name to number of rows written.
    """
    if entity_type not in _ENTITY_DISPATCH:
        log.info("%s: no relationship extraction defined, skipping", entity_type)
        return {}

    all_rel_types, _ = _ENTITY_DISPATCH[entity_type]

    # Apply exclusions
    effective_exclude = frozenset(exclude or ())
    if not include_inferred:
        effective_exclude = effective_exclude | INFERRED_RELATIONSHIP_TYPES
    if effective_exclude:
        skipped = all_rel_types & effective_exclude
        if skipped:
            log.info(
                "%s: excluding relationship types: %s",
                entity_type, ", ".join(sorted(skipped)),
            )
    rel_types = all_rel_types - effective_exclude
    if not rel_types:
        log.info("%s: all relationship types excluded, nothing to do", entity_type)
        return {}

    ordered_types = _order_rel_types(rel_types)

    source_files = iter_source_files(entity_type)
    if not source_files:
        log.warning("%s: no source files found", entity_type)
        return {}

    # Apply distributed slicing
    if slice_index is not None and slice_total is not None:
        source_files = [
            f for i, f in enumerate(source_files)
            if i % slice_total == slice_index
        ]
        log.info(
            "Slice %d/%d: %d source files assigned",
            slice_index, slice_total, len(source_files),
        )

    n_workers = workers if workers is not None else min(_DEFAULT_WORKERS, len(source_files))
    n_workers = max(1, n_workers)
    effective_batch_size = batch_size if batch_size is not None else _BATCH_SIZE

    # ── Classify types: skip / drift-rebuild / incremental / full ───
    types_to_run: list[str] = []

    for rt in ordered_types:
        if force:
            types_to_run.append(rt)
            continue

        _rt_dir = rt_dir(SNAPSHOT_DIR, rt)

        # Check which source files have valid parquet shards on disk
        all_completed_keys = _completed_source_keys(_rt_dir)

        # When running with --slice-index, source_files is a subset of
        # the full set.  Only count completed shards that belong to this
        # slice — shards completed by other slices or by a previous full
        # run must not inflate the count.
        source_file_keys = {_source_file_key(f) for f in source_files}
        completed_keys = all_completed_keys & source_file_keys
        n_source = len(source_files)
        n_complete = len(completed_keys)

        # Check for aggregate provenance + manifest drift
        provenance_path = _rt_dir / "_provenance.json"
        if provenance_path.exists() and n_complete == n_source:
            # Fully complete — check manifest drift
            current_manifest = _load_entity_manifest(entity_type)
            previous_manifest = _load_manifest_snapshot(_rt_dir)
            drift = _detect_manifest_drift(current_manifest, previous_manifest)

            if drift:
                added = sum(1 for v in drift.values() if v == "added")
                removed = sum(1 for v in drift.values() if v == "removed")
                changed = sum(1 for v in drift.values() if v == "changed")
                log.info(
                    "%s: manifest drift — %d added, %d removed, %d changed",
                    rt, added, removed, changed,
                )
                # Delete affected units + shards, keep healthy ones
                for file_rel, drift_type in drift.items():
                    if drift_type in ("changed", "removed"):
                        key = _source_file_key(
                            _manifest_key_to_path(file_rel)
                        )
                        # Remove shard
                        shard = _shard_path(_rt_dir, key)
                        if shard.exists():
                            shard.unlink()
                            log.debug("Deleted drifted shard: %s", shard.name)
                        # Remove unit provenance
                        unit_prov = _rt_dir / "_units" / f"{key}.json"
                        if unit_prov.exists():
                            unit_prov.unlink()
                types_to_run.append(rt)
                continue

            log.info("%s: complete (%d/%d units, no drift), skipping", rt, n_complete, n_source)
            continue

        # Check source_file_count vs unit count
        if n_complete == n_source:
            # All units complete but no aggregate provenance — just finalise
            log.info("%s: all %d/%d units complete, finalising", rt, n_complete, n_source)
            continue

        if completed_keys:
            pending = n_source - n_complete
            log.info(
                "%s: %d/%d units complete, processing %d pending",
                rt, n_complete, n_source, pending,
            )
        else:
            log.info("%s: no provenance, full extraction", rt)

        types_to_run.append(rt)

    if not types_to_run:
        log.info(
            "%s relationships: all types complete, nothing to do",
            entity_type,
        )
        return {}

    log.info(
        "%s relationships: %d types to process (%s), %d source files, %d workers",
        entity_type,
        len(types_to_run),
        " → ".join(types_to_run),
        len(source_files),
        n_workers,
    )

    # ── Single-pass extraction across all pending types ───────────────
    #
    # Instead of processing each type sequentially (which reads every
    # source file once per type), we process all pending types in a
    # single pass.  Each source file is read and decompressed once, then
    # all relationship types are extracted from it simultaneously.
    # For 8 remaining types this eliminates 7/8 of the I/O.
    #
    result: dict[str, int] = {}

    # Build per-type pending-file sets and union of all files needed
    all_pending_types: list[str] = []
    type_completed_keys: dict[str, set[str]] = {}
    files_needed: set[Path] = set()
    types_already_done: list[str] = []

    for rt in types_to_run:
        _rt_dir = rt_dir(SNAPSHOT_DIR, rt)
        create_output_dir(_rt_dir)

        completed = _completed_source_keys(_rt_dir)
        pending = _compute_pending_source_files(source_files, completed)

        if not pending:
            # All done — just finalise
            total = _finalise_type(
                _rt_dir,
                relationship_type=rt,
                source_entity=entity_type,
                source_file_count=len(source_files),
            )
            result[rt] = total
            types_already_done.append(rt)
            continue

        log.info(
            "%s: %s [%s] — %d files to process (%d already done)",
            entity_type, rt,
            "inferred" if rt in INFERRED_RELATIONSHIP_TYPES else "structural",
            len(pending), len(completed),
        )
        all_pending_types.append(rt)
        type_completed_keys[rt] = completed
        files_needed.update(pending)

    if types_already_done:
        log.info(
            "%s: %d types already complete (%s)",
            entity_type, len(types_already_done),
            ", ".join(types_already_done),
        )

    if not all_pending_types:
        log.info(
            "%s relationships: all types complete, nothing to do",
            entity_type,
        )
        return result

    files_to_process = sorted(files_needed)
    log.info(
        "%s: single-pass extraction — %d types, %d unique files (%s)",
        entity_type, len(all_pending_types), len(files_to_process),
        " + ".join(all_pending_types),
    )

    # Distribute files across workers — each worker processes all types
    actual_workers = min(n_workers, len(files_to_process))
    actual_workers = max(1, actual_workers)

    reset_skipped_files()

    if actual_workers == 1:
        worker_results = [
            _worker_process_files(
                (0, files_to_process, entity_type, all_pending_types, effective_batch_size, type_completed_keys),
            ),
        ]
    else:
        chunks: list[list[Path]] = [
            [files_to_process[j] for j in range(i, len(files_to_process), actual_workers)]
            for i in range(actual_workers)
        ]

        worker_args = [
            (i, chunk, entity_type, all_pending_types, effective_batch_size, type_completed_keys)
            for i, chunk in enumerate(chunks)
            if chunk
        ]

        with multiprocessing.get_context("fork").Pool(
            processes=len(worker_args),
        ) as pool:
            worker_results = pool.map(
                _worker_process_files, worker_args,
            )

    # Finalise each type independently
    for rt in all_pending_types:
        _rt_dir = rt_dir(SNAPSHOT_DIR, rt)

        new_row_count = sum(
            rt_counts.get(rt, 0)
            for wr in worker_results
            for rt_counts in wr.get("results", {}).values()
        )

        total_row_count = _finalise_type(
            _rt_dir,
            relationship_type=rt,
            source_entity=entity_type,
            source_file_count=len(source_files),
        )
        result[rt] = total_row_count

        log.info(
            "%s: %s complete -- %d rows (%d resumed + %d new)",
            entity_type, rt, total_row_count,
            total_row_count - new_row_count,
            new_row_count,
        )

    log.info(
        "%s relationships: all types complete -- %s",
        entity_type,
        ", ".join(f"{rt}={cnt}" for rt, cnt in sorted(result.items())),
    )

    # Update README.md dataset_info for this entity automatically
    try:
        from sync.metadata import update_entity
        update_entity(entity_type)
    except Exception as exc:
        log.warning("Failed to update metadata for %s: %s", entity_type, exc)

    return result


# ── Migration: legacy worker shards → deterministic unit shards ─────────


def _validate_unit(
    rt_dir: Path,
    source_key: str,
    row_count: int,
    expected_hash: str | None = None,
) -> bool:
    """Validate a migrated unit by re-extracting and comparing.

    Returns True if the shard matches fresh extraction.  If *expected_hash*
    is given and the shard's hash differs, the unit is invalid.
    """
    shard = _shard_path(rt_dir, source_key)
    if not shard.exists():
        return False
    if expected_hash and _hash_file(shard) != expected_hash:
        return False
    try:
        pf = pq.ParquetFile(str(shard))
        actual_rows = pf.metadata.num_rows
        return actual_rows == row_count
    except Exception:
        return False


def migrate_relationship_type(
    relationship_type: str,
    entity_type: str,
    *,
    batch_size: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Migrate a single relationship type from legacy worker shards to
    deterministic per-source-file units.

    **Algorithm**

    1. Scan for legacy ``part-WW-NNNNN.parquet`` shards.
    2. If no legacy shards found, the type is already in unit layout (or empty).
    3. For each source file in the entity snapshot:
       a. Compute the deterministic ``source_key``.
       b. If a valid unit provenance already exists, skip.
       c. Re-extract the source file for this relationship type only.
       d. Write new unit shard + provenance.
    4. Validate all units: hash + row count.
    5. If any unit is invalid, delete it and re-extract from source.
    6. If all 100% of source files have valid units:
       - Delete all legacy ``part-WW-*.parquet`` and ``part-NNNNN.parquet`` shards.
       - Delete old ``_provenance.json``, ``_provenance_worker_*.json``,
         ``_lineage.json``, ``_manifest_snapshot.json``.
       - Write new aggregate provenance, lineage, manifest snapshot.

    Args:
        relationship_type: e.g. ``"work_sdgs"``.
        entity_type: e.g. ``"works"``.
        batch_size: Rows per parquet row-group.
        force: Re-extract all units even if provenance exists.
        dry_run: Report what would happen without writing.

    Returns:
        ``{"migrated": int, "re_extracted": int, "validated": int, "total_source_files": int}``
    """
    _rt_dir = rt_dir(SNAPSHOT_DIR, relationship_type)
    if not _rt_dir.exists():
        log.warning("%s: output dir does not exist, nothing to migrate", relationship_type)
        return {"migrated": 0, "re_extracted": 0, "validated": 0, "total_source_files": 0}

    # Detect legacy shards
    legacy_shards = sorted(
        list(_rt_dir.glob("part-??-*.parquet"))  # part-WW-NNNNN.parquet
        + list(_rt_dir.glob("part-0.parquet"))   # edge case
    )
    # Filter: only files matching the worker-ID pattern part-NN-NNNNN.parquet
    legacy_shards = [
        s for s in legacy_shards
        if len(s.name.split("-")) >= 3 or (len(s.name.split("-")) == 2 and s.name.split("-")[1].replace(".parquet", "").isdigit())
    ]

    if not legacy_shards:
        log.info("%s: no legacy worker shards found, already in unit layout", relationship_type)
        # Still check if we need to finalise
        units = _load_unit_provenances(rt_dir)
        source_files = iter_source_files(entity_type)
        if units and len(units) == len(source_files):
            _finalise_type(
                rt_dir,
                relationship_type=relationship_type,
                source_entity=entity_type,
                source_file_count=len(source_files),
            )
        return {
            "migrated": 0, "re_extracted": 0, "validated": len(units),
            "total_source_files": len(source_files),
        }

    log.info(
        "%s: found %d legacy worker shards, beginning migration",
        relationship_type, len(legacy_shards),
    )

    source_files = iter_source_files(entity_type)
    if not source_files:
        log.warning("%s: no source files found for entity %s", relationship_type, entity_type)
        return {"migrated": 0, "re_extracted": 0, "validated": 0, "total_source_files": 0}

    effective_batch_size = batch_size or _BATCH_SIZE
    all_rel_types, extractor = _ENTITY_DISPATCH[entity_type]
    if relationship_type not in all_rel_types:
        log.error("%s is not a valid relationship type for %s", relationship_type, entity_type)
        return {"migrated": 0, "re_extracted": 0, "validated": 0, "total_source_files": len(source_files)}

    # Re-extract each source file as a deterministic unit
    migrated = 0
    re_extracted = 0
    validated = 0

    for source_file in source_files:
        source_key = _source_file_key(source_file)

        # Check existing unit provenance
        existing_units = _load_unit_provenances(rt_dir)
        existing = existing_units.get(source_key)

        if existing and existing.get("status") in ("complete", "empty") and not force:
            # Validate existing unit
            if _validate_unit(rt_dir, source_key, existing.get("row_count", 0),
                              existing.get("output_hash")):
                validated += 1
                continue
            else:
                log.warning("%s: existing unit %s failed validation, re-extracting",
                            relationship_type, source_key)

        if dry_run:
            log.info("[dry-run] Would re-extract %s for %s", source_key, relationship_type)
            re_extracted += 1
            continue

        # Re-extract this source file
        try:
            file_results = _extract_one_source_file(
                source_file,
                entity_type,
                frozenset({relationship_type}),
                effective_batch_size,
                SNAPSHOT_DIR,
                STAGING_DIR,
            )
            info = file_results.get(relationship_type, {})
            if info:
                re_extracted += 1
            else:
                log.warning("%s: no results for %s", relationship_type, source_key)
        except Exception as exc:
            log.error("%s: failed to extract %s: %s", relationship_type, source_key, exc)
            continue

    # Validate all units
    units = _load_unit_provenances(rt_dir)
    valid_count = 0
    for key, unit in units.items():
        if _validate_unit(rt_dir, key, unit.get("row_count", 0), unit.get("output_hash")):
            valid_count += 1
        else:
            log.warning("%s: unit %s failed post-extraction validation", relationship_type, key)

    log.info(
        "%s: %d/%d units valid after re-extraction",
        relationship_type, valid_count, len(source_files),
    )

    if valid_count < len(source_files):
        log.error(
            "%s: only %d/%d units valid — NOT deleting legacy shards. "
            "Re-run to re-extract missing units.",
            relationship_type, valid_count, len(source_files),
        )
        return {
            "migrated": migrated, "re_extracted": re_extracted,
            "validated": valid_count, "total_source_files": len(source_files),
        }

    # 100% valid — hard cutover: delete legacy artifacts
    if not dry_run:
        log.info("%s: 100%% valid units (%d/%d) — deleting legacy shards",
                 relationship_type, valid_count, len(source_files))
        for shard in legacy_shards:
            shard.unlink()
            log.debug("Deleted legacy shard: %s", shard.name)

        # Delete old provenance formats
        for stale in ["_provenance.json", "_lineage.json", "_manifest_snapshot.json"]:
            p = rt_dir / stale
            if p.exists():
                p.unlink()
        for wp in sorted(rt_dir.glob("_provenance_worker_*.json")):
            wp.unlink()

        # Write new aggregate provenance
        _finalise_type(
            rt_dir,
            relationship_type=relationship_type,
            source_entity=entity_type,
            source_file_count=len(source_files),
        )

    return {
        "migrated": len(legacy_shards),
        "re_extracted": re_extracted,
        "validated": valid_count,
        "total_source_files": len(source_files),
    }


# ── CLI entry point ─────────────────────────────────────────────────────


def _sync_provenance_from_remote(remote_spec: str) -> None:
    """Rsync _units/ provenance metadata from a remote machine.

    *remote_spec* is an rsync-compatible source path, e.g.
    ``mini:/Volumes/ExAPFS/OpenAlex/parquet``.  Only the ``*/_units/``
    subdirectories (tiny JSON files) are transferred — the parquet data
    stays on the remote machine.  This lets the local provenance check
    see what the remote has already completed and skip those units.

    Runs ``find`` on the remote to locate ``_units/`` directories,
    avoiding a full directory tree scan over the network.
    """
    import subprocess

    # Parse host:path from remote_spec
    if ":" not in remote_spec:
        log.error("--sync-provenance must be host:path (e.g. mini:/data/parquet)")
        return
    host, remote_path = remote_spec.split(":", 1)

    # Find _units/ directories on the remote — scanning happens there,
    # not over the network.
    log.info("Scanning remote %s for _units/ directories ...", host)
    find_result = subprocess.run(
        ["ssh", host, "find", remote_path, "-type", "d", "-name", "_units"],
        capture_output=True, text=True, timeout=60,
    )
    if find_result.returncode != 0:
        log.warning("Remote find failed: %s", find_result.stderr.strip())
        return

    units_dirs = [d.strip() for d in find_result.stdout.strip().splitlines() if d.strip()]
    if not units_dirs:
        log.info("No _units/ directories found on remote")
        return

    log.info("Found %d _units/ directories on remote", len(units_dirs))

    # Build --files-from input with paths relative to remote_path
    files_from_lines = []
    for d in units_dirs:
        rel = d
        if rel.startswith(remote_path):
            rel = rel[len(remote_path):].lstrip("/")
        files_from_lines.append(rel)
    files_from = "\n".join(files_from_lines)

    local_parquet = str(SNAPSHOT_DIR)
    remote_src = f"{host}:{remote_path}/"
    log.info("Syncing %d _units/ dirs from %s", len(units_dirs), remote_src)
    result = subprocess.run(
        [
            "rsync", "-az", "--compress",
            "--relative",
            "--files-from=-", remote_src, local_parquet,
        ],
        input=files_from, text=True, timeout=300,
    )
    if result.returncode != 0:
        log.warning("Provenance sync failed (exit %d)", result.returncode)
    else:
        log.info("Provenance sync complete")


def main(entity: str | None = None, force: bool = False, workers: int | None = None, batch_size: int | None = None, slice_index: int | None = None, slice_total: int | None = None, sync_provenance: str | None = None, output_dir: str | None = None) -> None:
    """Extract relationship tables from JSONL snapshot."""
    import sync.common as _common

    if output_dir:
        _common.SNAPSHOT_DIR = Path(output_dir)
        # Also update this module's reference
        global SNAPSHOT_DIR
        SNAPSHOT_DIR = _common.SNAPSHOT_DIR

    if sync_provenance:
        _sync_provenance_from_remote(sync_provenance)

    types = [entity] if entity else _common.ENTITY_TYPES_BUILD_ORDER
    for et in types:
        counts = convert_relationships(et, force=force, workers=workers, batch_size=batch_size, slice_index=slice_index, slice_total=slice_total)
        for rt, cnt in sorted(counts.items()):
            log.info("Extracted %s: %d rows", rt, cnt)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Extract OpenAlex relationship tables to Parquet",
    )
    subparsers = parser.add_subparsers(dest="command")

    # extract (default)
    ext = subparsers.add_parser("extract", help="Extract relationships from snapshot")
    ext.add_argument("--entity", type=str, default=None)
    ext.add_argument("--force", action="store_true")
    ext.add_argument("--workers", type=int, default=None)
    ext.add_argument("--batch-size", type=int, default=None)
    ext.add_argument("--slice-index", type=int, default=None,
                      help="0-based slice index for distributed processing")
    ext.add_argument("--slice-total", type=int, default=None,
                      help="Total number of slices for distributed processing")
    ext.add_argument("--sync-provenance", type=str, default=None,
                      metavar="REMOTE",
                      help="Rsync _units/ provenance from remote (e.g. mini:/path/to/parquet)")
    ext.add_argument("--output-dir", type=str, default=None,
                      metavar="DIR",
                      help="Override SNAPSHOT_DIR for output (e.g. snapshot data dir for nested layout)")

    # migrate
    mig = subparsers.add_parser(
        "migrate",
        help="Migrate legacy worker shards to deterministic unit layout",
    )
    mig.add_argument("--relationship-type", type=str, required=True,
                      help="Relationship type to migrate (e.g. work_sdgs)")
    mig.add_argument("--entity", type=str, required=True,
                      help="Source entity type (e.g. works)")
    mig.add_argument("--batch-size", type=int, default=None)
    mig.add_argument("--force", action="store_true",
                      help="Re-extract all units even if provenance exists")
    mig.add_argument("--dry-run", action="store_true",
                      help="Report what would happen without writing")

    args = parser.parse_args()

    if args.command == "migrate":
        result = migrate_relationship_type(
            args.relationship_type,
            args.entity,
            batch_size=args.batch_size,
            force=args.force,
            dry_run=args.dry_run,
        )
        log.info("Migration result: %s", result)
    else:
        # Default: extract (also handles no subcommand for backward compat)
        entity = getattr(args, 'entity', None)
        force = getattr(args, 'force', False)
        workers = getattr(args, 'workers', None)
        batch_size = getattr(args, 'batch_size', None)
        slice_index = getattr(args, 'slice_index', None)
        slice_total = getattr(args, 'slice_total', None)
        sync_provenance = getattr(args, 'sync_provenance', None)
        output_dir = getattr(args, 'output_dir', None)
        main(entity=entity, force=force, workers=workers, batch_size=batch_size, slice_index=slice_index, slice_total=slice_total, sync_provenance=sync_provenance, output_dir=output_dir)
