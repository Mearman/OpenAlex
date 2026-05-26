# OpenAlex Research Data

Sync/extraction tooling for the OpenAlex scholarly metadata snapshot. The dataset itself lives on [HuggingFace](https://huggingface.co/datasets/Mearman/OpenAlex) (Git LFS via Xet storage).

## What's here

| Path | Description |
|------|-------------|
| `sync/` | Python tooling — download from S3, extract relationship tables to Parquet, manage the snapshot |
| `openalex-snapshot/` | [Git submodule](https://huggingface.co/datasets/Mearman/OpenAlex) — source data and extracted tables |

## Quick Start

All `sync` commands run from this directory (the repo root). The submodule must be initialised so `openalex-snapshot/data/` exists.

```bash
git clone https://github.com/Mearman/OpenAlex.git
cd OpenAlex
git submodule update --init
```

### Install dependencies

```bash
pip install -r sync/requirements.txt
```

### Download from S3

OpenAlex publishes the snapshot on AWS S3, freely accessible:

```bash
# Download all entities
python3 -m sync download

# Download a single entity
python3 -m sync download --entity works
```

Files are saved as `part_XXXX.jsonl.gz` (renamed from S3's `part_XXXX.gz` so HuggingFace's dataset viewer detects the format).

### Extract relationship tables to Parquet

Each source shard produces one Parquet file per relationship type (e.g. `work_abstracts`, `author_institutions`):

```bash
# Extract everything (skips already-completed shards)
python3 -m sync extract

# Extract a single entity
python3 -m sync extract --entity works

# Distributed across two machines
python3 -m sync extract --slice-index 0 --slice-total 2   # machine 1
python3 -m sync extract --slice-index 1 --slice-total 2   # machine 2
```

### Entity layout

```
data/{entity}/
  updated_date=YYYY-MM-DD/part_XXXX.jsonl.gz      # source data (from S3)
  {relationship_type}/
    {entity}__updated_date=...__part_XXXX.parquet  # extracted tables
```

For example, works produces relationship tables for abstracts, authorships, references, concepts, keywords, locations, and more.

## Dataset

| | |
|---|---|
| **Host** | https://huggingface.co/datasets/Mearman/OpenAlex |
| **Format** | JSONL source (`.jsonl.gz`) + Parquet relationship tables |
| **License** | CC0 (public domain) |

### Entities

Works, Authors, Sources, Institutions, Publishers, Funders, Awards, Topics, Concepts, Fields, Subfields, Domains.

## External links

- [OpenAlex documentation](https://docs.openalex.org)
- [OpenAlex API](https://api.openalex.org)
- [AWS Open Data Registry](https://registry.opendata.aws/openalex/)
- [HuggingFace dataset](https://huggingface.co/datasets/Mearman/OpenAlex)
