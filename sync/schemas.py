"""PyArrow schemas for OpenAlex entity and relationship Parquet tables."""

from __future__ import annotations

import pyarrow as pa

# ── Simple entity schema (topics, concepts, publishers, funders, etc.) ───

SIMPLE_ENTITY_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("display_name", pa.string()),
])

# ── Entity table schemas ────────────────────────────────────────────────

_WORKS_SCHEMA = pa.schema([
    # Identity + core
    pa.field("id", pa.int64(), nullable=False),
    pa.field("doi", pa.string()),
    pa.field("display_name", pa.string()),
    pa.field("title", pa.string()),
    pa.field("publication_year", pa.int32()),
    pa.field("publication_date", pa.string()),
    pa.field("type", pa.string()),
    pa.field("language", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
    # Citation metrics
    pa.field("cited_by_count", pa.int64()),
    pa.field("fwci", pa.float64()),
    pa.field("cnp_value", pa.float64()),
    pa.field("cnp_is_in_top_10_percent", pa.bool_()),
    pa.field("cnp_is_in_top_1_percent", pa.bool_()),
    pa.field("cbpy_min", pa.float64()),
    pa.field("cbpy_max", pa.float64()),
    # Flags
    pa.field("is_retracted", pa.bool_()),
    pa.field("is_paratext", pa.bool_()),
    pa.field("is_xpac", pa.bool_()),
    pa.field("has_fulltext", pa.bool_()),
    pa.field("has_grobid_xml", pa.bool_()),
    pa.field("has_pdf", pa.bool_()),
    # Open-access (flattened from open_access)
    pa.field("is_oa", pa.bool_()),
    pa.field("oa_status", pa.string()),
    pa.field("oa_any_repository_has_fulltext", pa.bool_()),
    pa.field("oa_url", pa.string()),
    # Counts (aggregates over sub-tables — denormalised for fast filtering)
    pa.field("authors_count", pa.int64()),
    pa.field("countries_distinct_count", pa.int64()),
    pa.field("institutions_distinct_count", pa.int64()),
    pa.field("locations_count", pa.int64()),
    pa.field("referenced_works_count", pa.int64()),
    # Biblio (flattened)
    pa.field("biblio_volume", pa.string()),
    pa.field("biblio_issue", pa.string()),
    pa.field("biblio_first_page", pa.string()),
    pa.field("biblio_last_page", pa.string()),
    # Primary topic shortcut (full graph in work_topics)
    pa.field("primary_topic_id", pa.int64()),
    pa.field("primary_topic_score", pa.float64()),
    # APC (article processing charge)
    pa.field("apc_list_value", pa.float64()),
    pa.field("apc_list_currency", pa.string()),
    pa.field("apc_list_value_usd", pa.float64()),
    pa.field("apc_paid_value", pa.float64()),
    pa.field("apc_paid_currency", pa.string()),
    pa.field("apc_paid_value_usd", pa.float64()),
])

_AUTHORS_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("orcid", pa.string()),
    pa.field("display_name", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
    pa.field("works_api_url", pa.string()),
    # Summary stats (flattened from summary_stats)
    pa.field("summary_stats_2yr_mean_citedness", pa.float64()),
    pa.field("summary_stats_h_index", pa.int32()),
    pa.field("summary_stats_i10_index", pa.int32()),
])

_INSTITUTIONS_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("ror", pa.string()),
    pa.field("display_name", pa.string()),
    pa.field("type", pa.string()),
    pa.field("type_id", pa.string()),
    pa.field("country_code", pa.string()),
    pa.field("homepage_url", pa.string()),
    pa.field("image_url", pa.string()),
    pa.field("image_thumbnail_url", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
    pa.field("status", pa.string()),
    pa.field("is_super_system", pa.bool_()),
    # Geo (flattened from geo)
    pa.field("geo_city", pa.string()),
    pa.field("geo_country", pa.string()),
    pa.field("geo_country_code", pa.string()),
    pa.field("geo_geonames_city_id", pa.string()),
    pa.field("geo_latitude", pa.float64()),
    pa.field("geo_longitude", pa.float64()),
    pa.field("geo_region", pa.string()),
    # Summary stats (flattened)
    pa.field("summary_stats_2yr_mean_citedness", pa.float64()),
    pa.field("summary_stats_h_index", pa.int32()),
    pa.field("summary_stats_i10_index", pa.int32()),
])

_SOURCES_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("type", pa.string()),
    pa.field("country_code", pa.string()),
    pa.field("homepage_url", pa.string()),
    pa.field("issn_l", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("oa_works_count", pa.int64()),
    pa.field("apc_usd", pa.float64()),
    pa.field("first_publication_year", pa.int32()),
    pa.field("last_publication_year", pa.int32()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
    pa.field("works_api_url", pa.string()),
    pa.field("host_organization_id", pa.int64()),
    pa.field("host_organization_name", pa.string()),
    pa.field("host_organization_type", pa.string()),  # "publisher" or "institution"
    # Flags
    pa.field("is_oa", pa.bool_()),
    pa.field("is_core", pa.bool_()),
    pa.field("is_high_oa_rate", pa.bool_()),
    pa.field("is_high_oa_rate_since_year", pa.int32()),
    pa.field("is_in_doaj", pa.bool_()),
    pa.field("is_in_doaj_since_year", pa.int32()),
    pa.field("is_in_scielo", pa.bool_()),
    pa.field("is_ojs", pa.bool_()),
    pa.field("oa_flip_year", pa.int32()),
    # Summary stats (flattened)
    pa.field("summary_stats_2yr_mean_citedness", pa.float64()),
    pa.field("summary_stats_h_index", pa.int32()),
    pa.field("summary_stats_i10_index", pa.int32()),
])

_TOPICS_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("subfield_id", pa.int64()),
    pa.field("field_id", pa.int64()),
    pa.field("domain_id", pa.int64()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
])

_CONCEPTS_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("level", pa.int32()),
    pa.field("wikidata", pa.string()),
    pa.field("image_url", pa.string()),
    pa.field("image_thumbnail_url", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
    pa.field("summary_stats_2yr_mean_citedness", pa.float64()),
    pa.field("summary_stats_h_index", pa.int32()),
    pa.field("summary_stats_i10_index", pa.int32()),
])

_PUBLISHERS_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("parent_publisher_id", pa.int64()),
    pa.field("hierarchy_level", pa.int32()),
    pa.field("homepage_url", pa.string()),
    pa.field("image_url", pa.string()),
    pa.field("image_thumbnail_url", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("sources_api_url", pa.string()),
    pa.field("ror_id", pa.string()),
    pa.field("wikidata_id", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
    pa.field("summary_stats_2yr_mean_citedness", pa.float64()),
    pa.field("summary_stats_h_index", pa.int32()),
    pa.field("summary_stats_i10_index", pa.int32()),
])

_FUNDERS_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("country_code", pa.string()),
    pa.field("homepage_url", pa.string()),
    pa.field("image_url", pa.string()),
    pa.field("image_thumbnail_url", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("awards_count", pa.int64()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
    pa.field("summary_stats_2yr_mean_citedness", pa.float64()),
    pa.field("summary_stats_h_index", pa.int32()),
    pa.field("summary_stats_i10_index", pa.int32()),
])

_TAXONOMY_NODE_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
])

# ── String-ID entity schemas (non-numeric OpenAlex IDs) ──────────────────

# Shared shape for simple count-only string-keyed reference vocabularies
_REFERENCE_VOCAB_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
])

# Languages: "https://openalex.org/languages/sc" → id="sc"
_LANGUAGES_SCHEMA = _REFERENCE_VOCAB_SCHEMA

# Source types: "https://openalex.org/source-types/other" → id="other"
_SOURCE_TYPES_SCHEMA = _REFERENCE_VOCAB_SCHEMA

# Institution types: "https://openalex.org/institution-types/other" → id="other"
_INSTITUTION_TYPES_SCHEMA = _REFERENCE_VOCAB_SCHEMA

# Keywords: "https://openalex.org/keywords/td-scdma" → id="td-scdma"
_KEYWORDS_SCHEMA = _REFERENCE_VOCAB_SCHEMA

# Licences: "https://openalex.org/licenses/mit" → id="mit"
_LICENSES_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("url", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
])

# Work types: "https://openalex.org/types/libguides" → id="libguides"
_WORK_TYPES_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
])

# Continents: "https://openalex.org/continents/Q18" → id="Q18"
_CONTINENTS_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("wikidata_id", pa.string()),
    pa.field("wikidata_url", pa.string()),
    pa.field("wikipedia_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
])

# Countries: "https://openalex.org/countries/WF" → id="WF"
_COUNTRIES_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("full_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("country_code", pa.string()),
    pa.field("alpha_3", pa.string()),
    pa.field("numeric", pa.string()),
    pa.field("continent_id", pa.string()),
    pa.field("is_global_south", pa.bool_()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("authors_api_url", pa.string()),
    pa.field("institutions_api_url", pa.string()),
    pa.field("wikidata_url", pa.string()),
    pa.field("wikipedia_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
])

# Awards: "https://openalex.org/G1346583" → id=1346583 (numeric, with G prefix)
_AWARDS_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("doi", pa.string()),
    pa.field("landing_page_url", pa.string()),
    pa.field("funder_award_id", pa.string()),
    pa.field("amount", pa.float64()),
    pa.field("currency", pa.string()),
    pa.field("funding_type", pa.string()),
    pa.field("funder_scheme", pa.string()),
    pa.field("provenance", pa.string()),
    pa.field("start_date", pa.string()),
    pa.field("end_date", pa.string()),
    pa.field("start_year", pa.int32()),
    pa.field("end_year", pa.int32()),
    pa.field("funded_outputs_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
    # Funder (flattened from funder dict)
    pa.field("funder_id", pa.int64()),
    pa.field("funder_display_name", pa.string()),
    pa.field("funder_doi", pa.string()),
    pa.field("funder_ror_id", pa.string()),
    # Lead investigator (flattened from lead_investigator dict;
    # affiliation flattened to name + country — ids list and investigators
    # list are nested and require their own relationship tables, not in scope here)
    pa.field("lead_investigator_given_name", pa.string()),
    pa.field("lead_investigator_family_name", pa.string()),
    pa.field("lead_investigator_orcid", pa.string()),
    pa.field("lead_investigator_affiliation_name", pa.string()),
    pa.field("lead_investigator_affiliation_country", pa.string()),
    pa.field("lead_investigator_role_start", pa.string()),
])

# SDGs: "https://openalex.org/sdgs/8" → id=8 (numeric)
_SDGS_SCHEMA = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("display_name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("image_url", pa.string()),
    pa.field("image_thumbnail_url", pa.string()),
    pa.field("works_count", pa.int64()),
    pa.field("cited_by_count", pa.int64()),
    pa.field("works_api_url", pa.string()),
    pa.field("created_date", pa.string()),
    pa.field("updated_date", pa.string()),
])

ENTITY_SCHEMAS: dict[str, pa.Schema] = {
    "works": _WORKS_SCHEMA,
    "authors": _AUTHORS_SCHEMA,
    "institutions": _INSTITUTIONS_SCHEMA,
    "sources": _SOURCES_SCHEMA,
    "topics": _TOPICS_SCHEMA,
    "concepts": _CONCEPTS_SCHEMA,
    "publishers": _PUBLISHERS_SCHEMA,
    "funders": _FUNDERS_SCHEMA,
    "subfields": _TAXONOMY_NODE_SCHEMA,
    "fields": _TAXONOMY_NODE_SCHEMA,
    "domains": _TAXONOMY_NODE_SCHEMA,
    # Full-coverage additions
    "awards": _AWARDS_SCHEMA,
    "languages": _LANGUAGES_SCHEMA,
    "licenses": _LICENSES_SCHEMA,
    "work-types": _WORK_TYPES_SCHEMA,
    "source-types": _SOURCE_TYPES_SCHEMA,
    "institution-types": _INSTITUTION_TYPES_SCHEMA,
    "keywords": _KEYWORDS_SCHEMA,
    "sdgs": _SDGS_SCHEMA,
    "continents": _CONTINENTS_SCHEMA,
    "countries": _COUNTRIES_SCHEMA,
}

# ── Relationship table schemas ──────────────────────────────────────────

RELATIONSHIP_SCHEMAS: dict[str, pa.Schema] = {
    "work_authorships": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("author_position", pa.string()),
    ]),
    "work_authorship_institutions": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("institution_id", pa.int64(), nullable=False),
    ]),
    "work_references": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("referenced_work_id", pa.int64(), nullable=False),
    ]),
    "work_topics": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("score", pa.float32()),
    ]),
    "work_concepts": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("concept_id", pa.int64(), nullable=False),
        pa.field("score", pa.float32()),
    ]),
    "work_locations": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("is_oa", pa.bool_()),
        pa.field("is_primary", pa.bool_()),
        pa.field("license", pa.string()),
        pa.field("version", pa.string()),
    ]),
    "work_related": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("related_work_id", pa.int64(), nullable=False),
    ]),
    "author_institutions": pa.schema([
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("institution_id", pa.int64(), nullable=False),
    ]),
    "work_funders": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("funder_id", pa.int64(), nullable=False),
        pa.field("award_id", pa.string()),
    ]),
    "work_keywords": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("keyword_id", pa.string(), nullable=False),
        pa.field("score", pa.float32()),
    ]),
    "source_host_lineage": pa.schema([
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("publisher_id", pa.int64(), nullable=False),
    ]),
    "institution_associations": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("associated_institution_id", pa.int64(), nullable=False),
        pa.field("relationship_type", pa.string()),
    ]),
    "institution_repositories": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("source_id", pa.int64(), nullable=False),
    ]),
    "institution_roles": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("role_entity_id", pa.int64(), nullable=False),
        pa.field("role_type", pa.string()),
        pa.field("role_prefix", pa.string()),
    ]),
    "publisher_lineage": pa.schema([
        pa.field("publisher_id", pa.int64(), nullable=False),
        pa.field("ancestor_publisher_id", pa.int64(), nullable=False),
    ]),
    "publisher_roles": pa.schema([
        pa.field("publisher_id", pa.int64(), nullable=False),
        pa.field("role_entity_id", pa.int64(), nullable=False),
        pa.field("role_type", pa.string()),
        pa.field("role_prefix", pa.string()),
    ]),
    "funder_roles": pa.schema([
        pa.field("funder_id", pa.int64(), nullable=False),
        pa.field("role_entity_id", pa.int64(), nullable=False),
        pa.field("role_type", pa.string()),
        pa.field("role_prefix", pa.string()),
    ]),
    "concept_ancestors": pa.schema([
        pa.field("concept_id", pa.int64(), nullable=False),
        pa.field("ancestor_concept_id", pa.int64(), nullable=False),
    ]),
    "concept_related": pa.schema([
        pa.field("concept_id", pa.int64(), nullable=False),
        pa.field("related_concept_id", pa.int64(), nullable=False),
        pa.field("score", pa.float32()),
    ]),
    # ── Newly added relationships (audit 2026-05-16) ────────────────────
    "work_sdgs": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("sdg_id", pa.int64(), nullable=False),
        pa.field("score", pa.float32()),
    ]),
    "work_mesh": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("descriptor_ui", pa.string(), nullable=False),
        pa.field("descriptor_name", pa.string()),
        pa.field("qualifier_ui", pa.string()),
        pa.field("qualifier_name", pa.string()),
        pa.field("is_major_topic", pa.bool_()),
    ]),
    "work_corresponding_authors": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("author_id", pa.int64(), nullable=False),
    ]),
    "work_corresponding_institutions": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("institution_id", pa.int64(), nullable=False),
    ]),
    "author_topics": pa.schema([
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("count", pa.int64()),
        pa.field("score", pa.float32()),
    ]),
    "author_last_known_institutions": pa.schema([
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("institution_id", pa.int64(), nullable=False),
    ]),
    "institution_topics": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("count", pa.int64()),
        pa.field("score", pa.float32()),
    ]),
    "institution_lineage": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("ancestor_institution_id", pa.int64(), nullable=False),
    ]),
    "source_topics": pa.schema([
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("count", pa.int64()),
        pa.field("score", pa.float32()),
    ]),
    "source_societies": pa.schema([
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("organization", pa.string(), nullable=False),
        pa.field("url", pa.string()),
    ]),
    "publisher_countries": pa.schema([
        pa.field("publisher_id", pa.int64(), nullable=False),
        pa.field("country_code", pa.string(), nullable=False),
    ]),
    # Topic hierarchy (extracted from topic/subfield/field entity records)
    "topic_subfields": pa.schema([
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("subfield_id", pa.int64(), nullable=False),
    ]),
    "topic_fields": pa.schema([
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("field_id", pa.int64(), nullable=False),
    ]),
    "topic_domains": pa.schema([
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("domain_id", pa.int64(), nullable=False),
    ]),
    "subfield_fields": pa.schema([
        pa.field("subfield_id", pa.int64(), nullable=False),
        pa.field("field_id", pa.int64(), nullable=False),
    ]),
    "subfield_domains": pa.schema([
        pa.field("subfield_id", pa.int64(), nullable=False),
        pa.field("domain_id", pa.int64(), nullable=False),
    ]),
    "field_domains": pa.schema([
        pa.field("field_id", pa.int64(), nullable=False),
        pa.field("domain_id", pa.int64(), nullable=False),
    ]),
    # ── Lossless projection of work record (replaces _json) ──────────
    "work_counts_by_year": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("cited_by_count", pa.int64()),
    ]),
    "work_external_ids": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),  # mag, doi, pmid, pmcid, openalex
        pa.field("value", pa.string(), nullable=False),
    ]),
    "work_indexed_in": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("index_name", pa.string(), nullable=False),  # crossref, pubmed, datacite, ...
    ]),
    "work_awards": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("award_id", pa.int64(), nullable=False),
        pa.field("display_name", pa.string()),
        pa.field("funder_award_id", pa.string()),
        pa.field("funder_id", pa.int64()),
        pa.field("funder_display_name", pa.string()),
    ]),
    "work_abstracts": pa.schema([
        pa.field("work_id", pa.int64(), nullable=False),
        pa.field("abstract_inverted_index", pa.string()),  # JSON-encoded dict
    ]),
    # ── Lossless projection of author record ─────────────────────────
    "author_counts_by_year": pa.schema([
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("works_count", pa.int64()),
        pa.field("cited_by_count", pa.int64()),
        pa.field("oa_works_count", pa.int64()),
    ]),
    "author_external_ids": pa.schema([
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),  # openalex, orcid, mag, scopus
        pa.field("value", pa.string(), nullable=False),
    ]),
    "author_name_alternatives": pa.schema([
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("display_name_alternative", pa.string(), nullable=False),
    ]),
    "author_topic_share": pa.schema([
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("value", pa.float64()),
    ]),
    "author_sources": pa.schema([
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("is_core", pa.bool_()),
        pa.field("is_in_doaj", pa.bool_()),
    ]),
    "author_concepts": pa.schema([
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("concept_id", pa.int64(), nullable=False),
        pa.field("score", pa.float64()),
        pa.field("count", pa.int64()),
        pa.field("level", pa.int32()),
    ]),
    # ── Lossless projection of source record ─────────────────────────
    "source_counts_by_year": pa.schema([
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("works_count", pa.int64()),
        pa.field("cited_by_count", pa.int64()),
        pa.field("oa_works_count", pa.int64()),
    ]),
    "source_external_ids": pa.schema([
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),  # openalex, issn_l, mag, wikidata, issn (array source)
        pa.field("value", pa.string(), nullable=False),
    ]),
    "source_issns": pa.schema([
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("issn", pa.string(), nullable=False),
    ]),
    "source_apc_prices": pa.schema([
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("price", pa.float64()),
        pa.field("currency", pa.string()),
    ]),
    "source_alternate_titles": pa.schema([
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
    ]),
    "source_topic_share": pa.schema([
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("value", pa.float64()),
    ]),
    # ── Lossless projection of institution record ─────────────────────
    "institution_counts_by_year": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("works_count", pa.int64()),
        pa.field("cited_by_count", pa.int64()),
        pa.field("oa_works_count", pa.int64()),
    ]),
    "institution_external_ids": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),  # openalex, ror, grid, wikidata, wikipedia
        pa.field("value", pa.string(), nullable=False),
    ]),
    "institution_name_alternatives": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("display_name_alternative", pa.string(), nullable=False),
    ]),
    "institution_name_acronyms": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("display_name_acronym", pa.string(), nullable=False),
    ]),
    "institution_topic_share": pa.schema([
        pa.field("institution_id", pa.int64(), nullable=False),
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("value", pa.float64()),
    ]),
    # ── Lossless projection of topic record ──────────────────────────
    "topic_keywords": pa.schema([
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("keyword", pa.string(), nullable=False),
    ]),
    "topic_external_ids": pa.schema([
        pa.field("topic_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]),
    # ── Lossless projection of concept record ────────────────────────
    "concept_counts_by_year": pa.schema([
        pa.field("concept_id", pa.int64(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("works_count", pa.int64()),
        pa.field("cited_by_count", pa.int64()),
        pa.field("oa_works_count", pa.int64()),
    ]),
    "concept_external_ids": pa.schema([
        pa.field("concept_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),  # openalex, mag, umls_aui, umls_cui, wikidata, wikipedia
        pa.field("value", pa.string(), nullable=False),
    ]),
    # ── Lossless projection of publisher record ──────────────────────
    "publisher_counts_by_year": pa.schema([
        pa.field("publisher_id", pa.int64(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("works_count", pa.int64()),
        pa.field("cited_by_count", pa.int64()),
    ]),
    "publisher_external_ids": pa.schema([
        pa.field("publisher_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]),
    "publisher_alternate_titles": pa.schema([
        pa.field("publisher_id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
    ]),
    # ── Lossless projection of funder record ─────────────────────────
    "funder_counts_by_year": pa.schema([
        pa.field("funder_id", pa.int64(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("works_count", pa.int64()),
        pa.field("cited_by_count", pa.int64()),
        pa.field("oa_works_count", pa.int64()),
    ]),
    "funder_external_ids": pa.schema([
        pa.field("funder_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]),
    "funder_alternate_titles": pa.schema([
        pa.field("funder_id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
    ]),
    # ── Lossless projection of subfield/field/domain records ─────────
    "subfield_external_ids": pa.schema([
        pa.field("subfield_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]),
    "subfield_name_alternatives": pa.schema([
        pa.field("subfield_id", pa.int64(), nullable=False),
        pa.field("display_name_alternative", pa.string(), nullable=False),
    ]),
    "field_external_ids": pa.schema([
        pa.field("field_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]),
    "field_name_alternatives": pa.schema([
        pa.field("field_id", pa.int64(), nullable=False),
        pa.field("display_name_alternative", pa.string(), nullable=False),
    ]),
    "domain_external_ids": pa.schema([
        pa.field("domain_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]),
    "domain_name_alternatives": pa.schema([
        pa.field("domain_id", pa.int64(), nullable=False),
        pa.field("display_name_alternative", pa.string(), nullable=False),
    ]),
    # ── Lossless projection of SDG record ────────────────────────────
    "sdg_external_ids": pa.schema([
        pa.field("sdg_id", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),  # openalex, un, wikidata
        pa.field("value", pa.string(), nullable=False),
    ]),
    # ── Lossless projection of award record ──────────────────────────
    "award_investigators": pa.schema([
        pa.field("award_id", pa.int64(), nullable=False),
        pa.field("given_name", pa.string()),
        pa.field("family_name", pa.string()),
        pa.field("orcid", pa.string()),
        pa.field("affiliation_name", pa.string()),
        pa.field("affiliation_country", pa.string()),
        pa.field("role_start", pa.string()),
        pa.field("is_lead", pa.bool_()),
        pa.field("is_co_lead", pa.bool_()),
    ]),
    "award_investigator_affiliations": pa.schema([
        pa.field("award_id", pa.int64(), nullable=False),
        pa.field("investigator_orcid", pa.string()),  # nullable join key
        pa.field("investigator_family_name", pa.string()),
        pa.field("affiliation_id", pa.string()),  # e.g. nrid/ror URL
        pa.field("affiliation_type", pa.string()),  # nrid, ror, ...
        pa.field("asserted_by", pa.string()),
    ]),
    "award_funded_outputs": pa.schema([
        pa.field("award_id", pa.int64(), nullable=False),
        pa.field("work_id", pa.int64(), nullable=False),
    ]),
    # ── Lossless projection of continent / country records ───────────
    "continent_countries": pa.schema([
        pa.field("continent_id", pa.string(), nullable=False),
        pa.field("country_id", pa.string(), nullable=False),
    ]),
    "continent_external_ids": pa.schema([
        pa.field("continent_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]),
    "continent_name_alternatives": pa.schema([
        pa.field("continent_id", pa.string(), nullable=False),
        pa.field("display_name_alternative", pa.string(), nullable=False),
    ]),
    "country_external_ids": pa.schema([
        pa.field("country_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]),
    "country_name_alternatives": pa.schema([
        pa.field("country_id", pa.string(), nullable=False),
        pa.field("display_name_alternative", pa.string(), nullable=False),
    ]),
}
