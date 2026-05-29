#!/usr/bin/env python3
"""Verify schema-driven extraction produces identical output to hardcoded functions.

For each entity, loads a sample record, runs both extractors, and diffs the results.
Reports any mismatches in column names, row counts, or data values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add sync/ to path
sys.path.insert(0, str(Path(__file__).parent))

from sync.extract import (
    _ENTITY_DISPATCH,
    _extract_work_relationships,
    _extract_author_relationships,
    _extract_source_relationships,
    _extract_institution_relationships,
    _extract_publisher_relationships,
    _extract_funder_relationships,
    _extract_concept_relationships,
    _extract_topic_relationships,
    _extract_subfield_relationships,
    _extract_field_relationships,
    _extract_domain_relationships,
    _extract_sdg_relationships,
    _extract_award_relationships,
)
from sync.schema import probe_schema, extract_relationships


def normalise(result: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Sort rows within each rel type for deterministic comparison."""
    out = {}
    for rt, rows in result.items():
        # Sort by string representation for determinism
        out[rt] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str))
    return out


def compare(entity: str, record: dict) -> bool:
    """Compare hardcoded vs schema extraction for one record. Returns True if match."""
    _, hardcoded_fn = _ENTITY_DISPATCH.get(entity, (frozenset(), None))
    if hardcoded_fn is None:
        print(f"  SKIP: no hardcoded extractor for {entity}")
        return True

    hardcoded = normalise(hardcoded_fn(record))
    schema = probe_schema(entity, record)
    generic = normalise(extract_relationships(record, schema))

    # Check rel type coverage
    hc_types = set(hardcoded.keys())
    gen_types = set(generic.keys())

    ok = True
    missing = hc_types - gen_types
    extra = gen_types - hc_types

    if missing:
        print(f"  MISSING rel types (schema doesn't produce): {sorted(missing)}")
        ok = False
    if extra:
        print(f"  EXTRA rel types (schema produces but hardcoded doesn't): {sorted(extra)}")
        ok = False

    # Compare shared rel types
    for rt in sorted(hc_types & gen_types):
        hc_rows = hardcoded[rt]
        gen_rows = generic[rt]

        if len(hc_rows) != len(gen_rows):
            print(f"  {rt}: row count mismatch: hardcoded={len(hc_rows)}, schema={len(gen_rows)}")
            ok = False
            continue

        if hc_rows != gen_rows:
            print(f"  {rt}: row content mismatch")
            for i, (h, g) in enumerate(zip(hc_rows, gen_rows)):
                if h != g:
                    # Show diff
                    h_keys = set(h.keys())
                    g_keys = set(g.keys())
                    if h_keys != g_keys:
                        print(f"    row {i}: key diff: hardcoded={sorted(h_keys)}, schema={sorted(g_keys)}")
                    else:
                        diffs = {k: (h.get(k), g.get(k)) for k in h_keys if h.get(k) != g.get(k)}
                        print(f"    row {i}: value diff: {diffs}")
            ok = False

    if ok:
        print(f"  OK: {len(hc_types)} rel types, all match")

    # Report schema discovery
    print(f"  Schema discovered: {len(schema.fields)} fields, {len(schema.rel_type_names())} rel types")
    print(f"  Rel types: {sorted(schema.rel_type_names())}")

    return ok


# Sample records for each entity type (minimal but covering all field types)
SAMPLES = {
    "works": {
        "id": "https://openalex.org/W12345",
        "authorships": [
            {
                "author": {"id": "https://openalex.org/A100", "display_name": "Test Author"},
                "author_position": "first",
                "institutions": [
                    {"id": "https://openalex.org/I200", "display_name": "Test Uni"}
                ],
            }
        ],
        "referenced_works": ["https://openalex.org/W99999"],
        "related_works": ["https://openalex.org/W88888"],
        "topics": [{"id": "https://openalex.org/T100", "score": 0.85, "count": 50, "display_name": "Test Topic"}],
        "concepts": [{"id": "https://openalex.org/C100", "score": 0.72}],
        "locations": [
            {
                "source": {"id": "https://openalex.org/S100", "display_name": "Test Journal"},
                "is_oa": True,
                "is_primary": True,
                "license": "cc-by",
                "version": "publishedVersion",
            }
        ],
        "primary_location": {
            "source": {"id": "https://openalex.org/S100"},
        },
        "grants": [{"funder": "https://openalex.org/F100", "award_id": "grant-123"}],
        "keywords": [{"id": "https://openalex.org/keywords/machine-learning", "score": 0.9, "display_name": "ML"}],
        "sustainable_development_goals": [{"id": "https://openalex.org/SDG3", "score": 0.6}],
        "mesh": [{"descriptor_ui": "D001", "descriptor_name": "Test", "is_major_topic": True}],
        "corresponding_author_ids": ["https://openalex.org/A100"],
        "corresponding_institution_ids": ["https://openalex.org/I200"],
        "counts_by_year": [{"year": 2024, "cited_by_count": 10}],
        "ids": {"doi": "10.1234/test", "openalex": "W12345"},
        "indexed_in": ["crossref", "pubmed"],
        "awards": [
            {
                "id": "https://openalex.org/A123456",
                "display_name": "Test Award",
                "funder_award_id": "FA-001",
                "funder_id": "https://openalex.org/F100",
                "funder_display_name": "Test Funder",
            }
        ],
        "abstract_inverted_index": {"test": [0, 1]},
    },
    "authors": {
        "id": "https://openalex.org/A100",
        "affiliations": [{"institution": {"id": "https://openalex.org/I200"}}],
        "last_known_institutions": [{"id": "https://openalex.org/I200"}],
        "topics": [{"id": "https://openalex.org/T100", "count": 5, "score": 0.8}],
        "counts_by_year": [{"year": 2024, "works_count": 10, "cited_by_count": 50, "oa_works_count": 3}],
        "ids": {"orcid": "0000-0001-1234-5678"},
        "display_name_alternatives": ["J. Test"],
        "topic_share": [{"id": "https://openalex.org/T100", "value": 0.3, "domain": {}, "field": {}, "subfield": {}}],
        "sources": [{"id": "https://openalex.org/S100", "is_core": True, "is_in_doaj": False}],
        "x_concepts": [{"id": "https://openalex.org/C100", "score": 0.5, "count": 3, "level": 2}],
    },
    "sources": {
        "id": "https://openalex.org/S100",
        "host_organization_lineage": ["https://openalex.org/P100"],
        "topics": [{"id": "https://openalex.org/T100", "count": 5, "score": 0.8}],
        "societies": [{"organization": "Test Society", "url": "https://example.com"}],
        "counts_by_year": [{"year": 2024, "works_count": 100, "cited_by_count": 500, "oa_works_count": 30}],
        "ids": {"openalex": "S100", "issn_l": "1234-5679"},
        "issn": ["1234-5679", "9876-5432"],
        "apc_prices": [{"price": 500.0, "currency": "USD"}],
        "alternate_titles": ["Test Alt Title"],
        "topic_share": [{"id": "https://openalex.org/T100", "value": 0.3, "domain": {}, "field": {}, "subfield": {}}],
    },
    "institutions": {
        "id": "https://openalex.org/I200",
        "associated_institutions": [
            {"id": "https://openalex.org/I300", "relationship": "parent", "display_name": "Parent"}
        ],
        "repositories": [{"id": "https://openalex.org/S100", "display_name": "Repo"}],
        "roles": [{"id": "https://openalex.org/F100", "role": "funder"}],
        "lineage": ["https://openalex.org/I300"],
        "topics": [{"id": "https://openalex.org/T100", "count": 5, "score": 0.8}],
        "counts_by_year": [{"year": 2024, "works_count": 100, "cited_by_count": 500, "oa_works_count": 30}],
        "ids": {"ror": "https://ror.org/test"},
        "display_name_alternatives": ["Test Uni Alt"],
        "display_name_acronyms": ["TU"],
        "topic_share": [{"id": "https://openalex.org/T100", "value": 0.3, "domain": {}, "field": {}, "subfield": {}}],
    },
    "publishers": {
        "id": "https://openalex.org/P100",
        "lineage": ["https://openalex.org/P200"],
        "roles": [{"id": "https://openalex.org/I200", "role": "institution"}],
        "country_codes": ["GB", "US"],
        "counts_by_year": [{"year": 2024, "works_count": 100, "cited_by_count": 500}],
        "ids": {"openalex": "P100"},
        "alternate_titles": ["Test Pub Alt"],
    },
    "funders": {
        "id": "https://openalex.org/F100",
        "roles": [{"id": "https://openalex.org/I200", "role": "institution"}],
        "counts_by_year": [{"year": 2024, "works_count": 50, "cited_by_count": 200, "oa_works_count": 10}],
        "ids": {"openalex": "F100"},
        "alternate_titles": ["Test Fund Alt"],
    },
    "concepts": {
        "id": "https://openalex.org/C100",
        "ancestors": [{"id": "https://openalex.org/C200", "display_name": "Parent"}],
        "related_concepts": [{"id": "https://openalex.org/C300", "score": 0.5}],
        "counts_by_year": [{"year": 2024, "works_count": 100, "cited_by_count": 500, "oa_works_count": 30}],
        "ids": {"openalex": "C100", "wikidata": "Q12345"},
    },
    "topics": {
        "id": "https://openalex.org/T100",
        "subfield": {"id": "https://openalex.org/Sub100", "display_name": "Test Subfield"},
        "field": {"id": "https://openalex.org/Field100", "display_name": "Test Field"},
        "domain": {"id": "https://openalex.org/Domain100", "display_name": "Test Domain"},
        "keywords": ["machine learning", "deep learning"],
        "ids": {"openalex": "T100"},
    },
    "subfields": {
        "id": "https://openalex.org/Sub100",
        "field": {"id": "https://openalex.org/Field100", "display_name": "Test Field"},
        "domain": {"id": "https://openalex.org/Domain100", "display_name": "Test Domain"},
        "ids": {"openalex": "Sub100"},
        "display_name_alternatives": ["Test SF Alt"],
    },
    "fields": {
        "id": "https://openalex.org/Field100",
        "domain": {"id": "https://openalex.org/Domain100", "display_name": "Test Domain"},
        "ids": {"openalex": "Field100"},
        "display_name_alternatives": ["Test Field Alt"],
    },
    "domains": {
        "id": "https://openalex.org/Domain100",
        "ids": {"openalex": "Domain100"},
        "display_name_alternatives": ["Test Domain Alt"],
    },
    "sdgs": {
        "id": "https://openalex.org/SDG3",
        "ids": {"openalex": "SDG3", "un_sdgs": "https://sdgs.un.org/goals/goal3"},
    },
    "awards": {
        "id": "https://openalex.org/AW100",
        "lead_investigator": {
            "given_name": "Jane",
            "family_name": "Doe",
            "orcid": "0000-0001-1234-5678",
            "affiliation": {
                "name": "Test Uni",
                "country": "GB",
                "ids": [{"id": "I200", "type": "openalex", "asserted_by": "pii"}],
            },
        },
        "co_lead_investigator": {
            "given_name": "John",
            "family_name": "Smith",
            "orcid": None,
            "affiliation": "Another Uni",
        },
        "investigators": [
            {"given_name": "Bob", "family_name": "Jones", "orcid": None, "affiliation": None},
        ],
        "funded_outputs": ["https://openalex.org/W12345"],
    },
}


def main():
    all_ok = True
    for entity, record in SAMPLES.items():
        print(f"\n{'='*60}")
        print(f"Entity: {entity}")
        print(f"{'='*60}")
        ok = compare(entity, record)
        if not ok:
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("ALL ENTITIES: OK ✓")
    else:
        print("SOME MISMATCHES FOUND ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
