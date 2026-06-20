"""Generate the README.md for the OpenAlex HuggingFace dataset.

Reads the committed schema (openalex.schema.json) and scans the local
data directory to build the YAML frontmatter with configs: entries for
every entity+rel_type parquet table and every entity's source .jsonl.gz
files. The body contains dataset documentation.

Called by the sync pipeline after extraction, before upload, so the
generated README reflects the actual data on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from sync.common import SYNC_ROOT, nested_rt_path

# The schema lives in the OpenAlex repo alongside sync/, not inside the
# snapshot data root (which may be a plain folder on an external drive).
_REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = _REPO_ROOT / "openalex.schema.json"
DATA_DIR = SYNC_ROOT / "data"

# ── YAML frontmatter ──────────────────────────────────────────────────


def _entity_dir_name(entity: str) -> str:
    """Map entity name to its directory name under data/."""
    # Some entities use hyphens in dir names (e.g. institution-types)
    # The directory name matches the S3 prefix / entity key.
    # Try the entity name as-is first, then with hyphens.
    if (DATA_DIR / entity).is_dir():
        return entity
    return entity


def _scan_rel_types(entity: str) -> list[str]:
    """Scan the data directory for parquet rel-type subdirs of an entity."""
    entity_dir = DATA_DIR / _entity_dir_name(entity)
    if not entity_dir.is_dir():
        return []
    rel_types = []
    for child in sorted(entity_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        # Skip source data dirs (updated_date=YYYY-MM-DD) and metadata
        if name.startswith("updated_date=") or name == "manifest":
            continue
        # Only include if it actually contains parquets
        has_parquets = any(child.rglob("*.parquet"))
        if has_parquets:
            rel_types.append(name)
    return rel_types


def _has_source_files(entity: str) -> bool:
    """Check if entity has .jsonl.gz source files."""
    entity_dir = DATA_DIR / _entity_dir_name(entity)
    if not entity_dir.is_dir():
        return False
    return any(entity_dir.rglob("*.jsonl.gz"))


def _generate_configs(schema: dict) -> list[dict]:
    """Generate all config entries from schema.

    Uses the schema as the source of truth for entity+rel_type combinations.
    Also adds a __source config for every entity.
    """
    entities_data = schema.get("entities", {})
    configs: list[dict] = []

    for entity_name in sorted(entities_data.keys()):
        dir_name = _entity_dir_name(entity_name)
        entity_schema = entities_data[entity_name]

        # Collect rel_type names from the schema's fields
        seen_rels: set[str] = set()
        for field in entity_schema.get("fields", []):
            rel = field.get("rel_name")
            if rel and rel not in seen_rels:
                seen_rels.add(rel)
                # The data lives where extraction wrote it: rt_dir uses
                # nested_rt_path(rel) = "{entity_plural}/{subtable}" (e.g.
                # author_sources -> authors/sources), NOT "{entity}/{rel}".
                # Using the literal rel_name produced data/authors/author_sources
                # which matches no files, breaking the dataset viewer.
                configs.append({
                    "config_name": f"{entity_name}__{rel}",
                    "data_files": [{"split": "train", "path": f"data/{nested_rt_path(rel)}/*.parquet"}],
                })

        # Source config for every entity
        configs.append({
            "config_name": f"{entity_name}__source",
            "data_files": [{"split": "train", "path": f"data/{dir_name}/**/*.jsonl.gz"}],
        })

    return configs


def _yaml_dump_configs(configs: list[dict]) -> str:
    """Serialise configs to YAML without requiring pyyaml."""
    lines: list[str] = []
    for cfg in configs:
        lines.append(f"- config_name: {cfg['config_name']}")
        lines.append("  data_files:")
        for df in cfg["data_files"]:
            lines.append(f"  - split: {df['split']}")
            lines.append(f'    path: "{df["path"]}"')
    return "\n".join(lines)


# ── README body ────────────────────────────────────────────────────────

_BODY = r"""pretty_name: OpenAlex Snapshot
license: cc0-1.0
papers:
- https://arxiv.org/abs/2205.01833
annotations_creators:
- found
- machine-generated
language_creators:
- found
source_datasets:
- original
tags:
- academic
- scholarly-metadata
- citation-data
- bibliometrics
- open-science
- tabular
language:
- en
- multilingual
size_categories:
- n>1T
task_categories:
- tabular-classification
- feature-extraction
doi: 10.57967/hf/7682
---

# OpenAlex Snapshot

Mirror of the OpenAlex scholarly metadata snapshot — a free, open catalogue of 250M+ scholarly works, 100M+ authors, and related entities.

Hosted on HuggingFace via Xet for content-addressable deduplication.

**Source**: [s3://openalex](https://registry.opendata.aws/openalex/) (public, anonymous S3 bucket)

## Dataset subsets

Each entity type is a separate subset (config). For each entity, there is:
- One `__source` subset containing the raw `.jsonl.gz` source files
- One subset per extracted relationship table (e.g. `works__main`, `works__abstracts`, `authors__affiliations`)

| Entity | Description |
|--------|-------------|
| works | Scholarly works (papers, datasets, etc.) |
| authors | Authors of scholarly works |
| institutions | Universities, research orgs |
| publishers | Academic publishers |
| sources | Journals, repositories, conferences |
| awards | Grant/funding awards |
| concepts | Legacy concept taxonomy (Wikidata) |
| topics | Topic taxonomy |
| domains | Top-level topic domains |
| fields | Topic fields |
| subfields | Topic subfields |
| funders | Funding organisations |
| keywords | Machine-learning keywords |
| continents | Continents |
| countries | Countries |
| languages | Languages |
| licenses | Licences |
| sdgs | Sustainable Development Goals |
| institution-types | Institution types |
| source-types | Source types |
| work-types | Work types |

## Data format

Each shard is a gzip-compressed JSON Lines file at:

```
data/{entity}/updated_date=YYYY-MM-DD/part_XXXX.jsonl.gz
```

The `.jsonl.gz` extension allows the HuggingFace dataset viewer to detect the inner format automatically. On S3, files are named `part_XXXX.gz`; the download pipeline renames them on save.

Each line is a JSON object representing one entity record. Fields vary by entity type. See the [OpenAlex data model](https://docs.openalex.org/about-the-data) for field definitions.

### Extracted Parquet tables

The sync pipeline extracts relationship tables from each entity into Parquet files. Each entity has a `main` table (scalar attributes, one row per entity) plus separate tables for each list/dict-valued field:

```
data/{entity}/
  updated_date=YYYY-MM-DD/part_XXXX.jsonl.gz       # source data
  main/
    {entity}__updated_date=...__part_XXXX.parquet   # scalar attributes
  {relationship_type}/
    {entity}__updated_date=...__part_XXXX.parquet   # one edge table per list/dict field
```

The dataset viewer provides one subset per entity+relationship combination (e.g. `works__main`, `works__abstracts`, `authors__affiliations`) and one `__source` subset per entity for the raw JSONL.

### CSR matrices

The sync pipeline also produces Compressed Sparse Row (CSR) matrices for relationship tables that encode directed edges between integer node IDs. These are stored as scipy `.npz` files:

```
csr/
  work_referenced_works.npz    # citation graph (work → referenced work)
  work_authorships.npz         # authorship graph
  work_topics.npz              # work-topic associations
  ...
```

Each file contains a single sparse adjacency matrix. Row/column indices correspond to OpenAlex entity IDs. Provenance metadata is stored alongside as `.provenance.json`.

Build with:

```bash
python3 -m sync.build_csr --all
python3 -m sync.build_csr --rel-type work_referenced_works
```

### Edge-list Parquet

For querying the graph in DuckDB/Arrow without loading a whole matrix into memory, `--edge-list` exports each relationship as a sorted, deduplicated edge list in the original OpenAlex IDs (so it joins the `main` and relationship tables directly), in both directions:

```
csr/
  work_referenced_works__by_src.parquet   # (src, tgt), sorted by src
  work_referenced_works__by_tgt.parquet   # (src, tgt), sorted by tgt
  ...
```

Each file's bounded, sorted row groups let zonemaps prune that direction's lookups to a few row groups — millisecond range scans against the full graph. `by_src` answers "what X cites" (`WHERE src = X`); `by_tgt` answers "who cites X" (`WHERE tgt = X`).

```bash
python3 -m sync.build_csr --all --edge-list
duckdb -c "SELECT tgt FROM 'csr/work_referenced_works__by_src.parquet' WHERE src = 2741809807"  # what it cites
duckdb -c "SELECT src FROM 'csr/work_referenced_works__by_tgt.parquet' WHERE tgt = 2741809807"  # who cites it
```

### Example: Work record fields

`id`, `doi`, `title`, `display_name`, `publication_year`, `type`, `language`, `authorships`, `concepts`, `topics`, `keywords`, `cited_by_count`, `referenced_works`, `related_works`, `locations`, `open_access`, `funders`, `awards`, `mesh`, `sustainable_development_goals`, `counts_by_year`, `updated_date`, and more.

## Sync and extraction pipeline

The [sync/](https://github.com/Mearman/OpenAlex/tree/main/sync) directory contains a Python pipeline for downloading from S3 and extracting relationship tables to Parquet:

```bash
# Full sync (all entities)
python3 -m sync

# Limit to one entity
python3 -m sync --entity works

# Split extraction across machines
python3 -m sync --slice-index 0 --slice-total 2   # machine 1
python3 -m sync --slice-index 1 --slice-total 2   # machine 2
```

## License

OpenAlex data is released under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). See the [OpenAlex terms](https://openalex.org/about) for details.

## Links

- [OpenAlex documentation](https://docs.openalex.org)
- [OpenAlex API](https://api.openalex.org)
- [AWS Open Data Registry](https://registry.opendata.aws/openalex/)
- [Extraction pipeline source](https://github.com/Mearman/OpenAlex)
"""


# ── Public API ─────────────────────────────────────────────────────────


def generate_readme() -> str:
    """Generate the full README.md content with YAML frontmatter."""
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    configs = _generate_configs(schema)
    parquet = sum(1 for c in configs if "__source" not in c["config_name"])
    source = sum(1 for c in configs if "__source" in c["config_name"])

    yaml_configs = _yaml_dump_configs(configs)
    readme = f"---\nconfigs:\n{yaml_configs}\n{_BODY}"

    print(f"README generated: {parquet} parquet configs, {source} source configs, {len(configs)} total")
    return readme


def update_readme_on_hf() -> None:
    """Generate and upload the README to HuggingFace."""
    from huggingface_hub import HfApi

    readme = generate_readme()

    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id="Mearman/OpenAlex",
        repo_type="dataset",
        commit_message="docs: regenerate dataset viewer configs from schema",
    )
    print("README uploaded to HuggingFace")
