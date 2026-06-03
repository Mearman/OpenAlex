# OpenAlex Research Data

Sync/extraction tooling for the OpenAlex scholarly metadata snapshot. The dataset itself lives on [HuggingFace](https://huggingface.co/datasets/Mearman/OpenAlex) (Git LFS via Xet storage).

## What's here

| Path | Description |
|------|-------------|
| `sync/` | Python tooling — download from S3, extract relationship tables to Parquet, manage the snapshot |
| `openalex-snapshot/` | [Git submodule](https://huggingface.co/datasets/Mearman/OpenAlex) — source data and extracted tables |

## Quick Start

`python3 -m sync` runs from this directory (the repo root). The submodule must be initialised so `openalex-snapshot/data/` exists.

```bash
git clone https://github.com/Mearman/OpenAlex.git
cd OpenAlex
git submodule update --init
```

### Install dependencies

```bash
pip install -r sync/requirements.txt
```

### Run the sync

One idempotent command does everything — download sources from S3, extract Parquet, commit/push, and reconcile the HuggingFace dataset. There are no subcommands; re-running converges the local tree, git, and HF to the canonical state and resumes where it left off:

```bash
# Full sync (all entities)
python3 -m sync

# Limit to one entity
python3 -m sync --entity works

# Skip the HuggingFace upload (local extraction only)
python3 -m sync --no-upload

# Deep self-heal: content-verify sources and Parquet shards, re-fetching corruption
python3 -m sync --verify

# Split extraction across two machines
python3 -m sync --slice-index 0 --slice-total 2   # machine 1
python3 -m sync --slice-index 1 --slice-total 2   # machine 2
```

Source files are saved as `part_XXXX.jsonl.gz` (renamed from S3's `part_XXXX.gz` so HuggingFace's dataset viewer detects the format).

The extractor derives each entity's schema by scanning the source data — there is no hardcoded field list. Scalar attributes (id, doi, title, language, publication year, type, FWCI, open-access and bibliographic metadata, …) are collected into a single **main** table per entity; every list- or dict-valued field becomes its own **relationship** table. Each source shard produces one Parquet file per table. The HuggingFace upload runs **in the background, overlapping extraction** — completed shards are pushed while later ones are still being written, so it adds little to the wall-clock rather than running as a serial tail. The prune (deleting remote Parquet files that no longer exist locally) and the git-ref sync run once at the end against the final set (`--no-prune` to upload additively, `--no-upload` to skip HuggingFace entirely).

### Entity layout

```
data/{entity}/
  updated_date=YYYY-MM-DD/part_XXXX.jsonl.gz       # source data (from S3)
  main/
    {entity}__updated_date=...__part_XXXX.parquet   # scalar attributes, one row per entity
  {relationship_type}/
    {entity}__updated_date=...__part_XXXX.parquet   # one edge table per list/dict field
```

The schema is data-derived and committed to `openalex.schema.json`; re-scanning the data reproduces it deterministically, so a field's presence in the schema is decided by the data, not a hardcoded list. For example, works yields a `main` table (doi, title, language, year, type, FWCI, …) alongside relationship tables for abstracts, authorships, references, concepts, keywords, locations, and more.

## Dataset

| | |
|---|---|
| **Host** | https://huggingface.co/datasets/Mearman/OpenAlex |
| **Format** | JSONL source (`.jsonl.gz`) + Parquet tables (a `main` scalar-attribute table and relationship tables per entity) |
| **License** | CC0 (public domain) |

### Entities

Works, Authors, Sources, Institutions, Publishers, Funders, Awards, Topics, Concepts, Fields, Subfields, Domains.

## External links

- [OpenAlex documentation](https://docs.openalex.org)
- [OpenAlex API](https://api.openalex.org)
- [AWS Open Data Registry](https://registry.opendata.aws/openalex/)
- [HuggingFace dataset](https://huggingface.co/datasets/Mearman/OpenAlex)
