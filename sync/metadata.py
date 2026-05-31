"""Generate HuggingFace dataset metadata from parquet shards.

Reads actual parquet footers for schemas, row counts, and file sizes.
Outputs YAML ``dataset_info`` blocks for the README.md frontmatter.

Called automatically by ``extract.py`` after each entity's extraction
completes — no manual step required.

Standalone usage (e.g. after a batch git-add of parquets)::

    PYTHONPATH=. python3 -m sync.metadata
    PYTHONPATH=. python3 -m sync.metadata --entity works
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sync.common import (
    DATA_DIR,
    ENTITY_TYPES,
    SNAPSHOT_DIR,
)

log = logging.getLogger(__name__)

# ── PyArrow → HF dtype mapping ───────────────────────────────────────

_PA_TO_HF: dict[pa.DataType, str] = {
    pa.int8(): "int8",
    pa.int16(): "int16",
    pa.int32(): "int32",
    pa.int64(): "int64",
    pa.uint8(): "uint8",
    pa.uint16(): "uint16",
    pa.uint32(): "uint32",
    pa.uint64(): "uint64",
    pa.float32(): "float32",
    pa.float64(): "float64",
    pa.bool_(): "bool",
    pa.string(): "string",
    pa.large_string(): "string",
    pa.binary(): "binary",
    pa.large_binary(): "binary",
}


def _pa_dtype_to_hf(dtype: pa.DataType) -> dict:
    """Convert a PyArrow type to an HF dataset_info feature dict."""
    if dtype in _PA_TO_HF:
        return {"dtype": _PA_TO_HF[dtype]}

    if pa.types.is_list(dtype):
        return {
            "sequence": _pa_dtype_to_hf(dtype.value_type),
        }

    if pa.types.is_struct(dtype):
        return {
            "struct": [
                {"name": f.name, **_pa_dtype_to_hf(f.type)}
                for f in dtype
            ]
        }

    # Fallback
    return {"dtype": "string"}


# ── Per-sub-table info ───────────────────────────────────────────────

def _estimate_total_rows(parquets: list[Path], sample_size: int = 20) -> int:
    """Estimate total rows by sampling parquet footers.

    Reading all footers on ExFAT is slow for large shard counts.
    We sample a spread, compute average rows per shard, and extrapolate.
    """
    n = len(parquets)
    if n == 0:
        return 0

    if n <= sample_size:
        sample = parquets
    else:
        indices = [i * (n - 1) // (sample_size - 1) for i in range(sample_size)]
        sample = [parquets[i] for i in indices]

    total_sample_rows = 0
    for p in sample:
        try:
            pf = pq.ParquetFile(str(p))
            total_sample_rows += pf.metadata.num_rows
        except Exception:
            pass

    avg = total_sample_rows / len(sample)
    return int(avg * n)


def _estimate_total_bytes(parquets: list[Path], sample_size: int = 10) -> int:
    """Estimate total bytes by sampling a few files.

    ExFAT stat() is extremely slow for bulk operations, so we
    stat only a sample and extrapolate.
    """
    n = len(parquets)
    if n == 0:
        return 0

    # Stat a spread of files across the list
    if n <= sample_size:
        sample = parquets
    else:
        indices = [i * (n - 1) // (sample_size - 1) for i in range(sample_size)]
        sample = [parquets[i] for i in indices]

    sample_bytes = sum(p.stat().st_size for p in sample)
    avg = sample_bytes / len(sample)
    return int(avg * n)

def _sub_table_info(sub_dir: Path) -> dict | None:
    """Read parquet footer metadata for a sub-table directory.

    Returns dict with features, num_rows, num_bytes, num_shards,
    or None if directory is empty or missing.
    """
    if not sub_dir.is_dir():
        return None

    parquets = sorted(
        f for f in sub_dir.glob("*.parquet") if not f.name.startswith("._")
    )
    if not parquets:
        return None

    # Read schema and row count from first shard
    sample = parquets[0]
    try:
        pf = pq.ParquetFile(str(sample))
        schema = pf.schema_arrow
    except Exception:
        log.warning("Cannot read parquet: %s", sample)
        return None

    # Build features list
    features = []
    for field in schema:
        feat = {"name": field.name}
        feat.update(_pa_dtype_to_hf(field.type))
        features.append(feat)

    # Aggregate row count by sampling footers.
    # Reading all 38k works footers on ExFAT is too slow,
    # so sample a spread and extrapolate.
    total_rows = _estimate_total_rows(parquets)

    # File size: stat only a sample of files and extrapolate.
    total_bytes = _estimate_total_bytes(parquets)

    return {
        "features": features,
        "num_rows": total_rows,
        "num_bytes": total_bytes,
        "num_shards": len(parquets),
    }


# ── Per-entity info ──────────────────────────────────────────────────

def entity_parquet_info(entity: str) -> list[dict]:
    """Generate dataset_info entries for all sub-tables of an entity.

    Returns a list of dicts, each suitable for a HF config entry.
    """
    entity_dir = DATA_DIR / entity
    if not entity_dir.is_dir():
        return []

    # Find parquet sub-directories (skip updated_date= partitions)
    sub_dirs = sorted(
        d for d in entity_dir.iterdir()
        if d.is_dir() and not d.name.startswith("updated_date=") and not d.name.startswith("_")
    )

    results = []
    for sub_dir in sub_dirs:
        info = _sub_table_info(sub_dir)
        if info is None:
            continue

        # Config name: entity_subtable (e.g. works_abstracts)
        config_name = f"{entity}_{sub_dir.name}"
        results.append({
            "config_name": config_name,
            "data_dir": str(sub_dir.relative_to(SNAPSHOT_DIR)),
            **info,
        })

    return results


def all_entities_info() -> dict[str, list[dict]]:
    """Generate dataset_info for all entities in build order.

    Returns ``{entity_name: [sub_table_info, ...]}``.
    """
    result = {}
    for entity in ENTITY_TYPES:
        info = entity_parquet_info(entity)
        if info:
            result[entity] = info
    return result


# ── YAML output ──────────────────────────────────────────────────────

def _yaml_value(value) -> str:
    """Format a Python value as YAML scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        # Simple strings that don't need quoting
        if value.isidentifier() or value.replace("-", "").isalnum():
            return value
        return f'"{value}"'
    return str(value)


def _yaml_features(features: list[dict], indent: int = 8) -> list[str]:
    """Format a features list as YAML."""
    lines: list[str] = []
    prefix = " " * indent
    for feat in features:
        name = feat["name"]
        if "dtype" in feat:
            lines.append(f"{prefix}- name: {name}")
            lines.append(f"{prefix}  dtype: {feat['dtype']}")
        elif "sequence" in feat:
            lines.append(f"{prefix}- name: {name}")
            seq = feat["sequence"]
            if "dtype" in seq:
                lines.append(f"{prefix}  sequence: {seq['dtype']}")
            elif "struct" in seq:
                lines.append(f"{prefix}  sequence:")
                for sf in seq["struct"]:
                    lines.append(f"{prefix}    - name: {sf['name']}")
                    lines.append(f"{prefix}      dtype: {sf['dtype']}")
        elif "struct" in feat:
            lines.append(f"{prefix}- name: {name}")
            lines.append(f"{prefix}  struct:")
            for sf in feat["struct"]:
                lines.append(f"{prefix}    - name: {sf['name']}")
                lines.append(f"{prefix}      dtype: {sf['dtype']}")
    return lines


def generate_dataset_info_yaml(entities: dict[str, list[dict]] | None = None) -> str:
    """Generate the dataset_info YAML block for README.md.

    If *entities* is None, generates for all entities.
    """
    if entities is None:
        entities = all_entities_info()

    lines: list[str] = []
    lines.append("dataset_info:")

    for entity_name, sub_tables in entities.items():
        for st in sub_tables:
            lines.append(f"  - config_name: {st['config_name']}")
            lines.append(f"    data_dir: {st['data_dir']}")
            lines.append("    features:")
            lines.extend(_yaml_features(st["features"], indent=6))
            lines.append(f"    num_rows: {st['num_rows']}")
            lines.append(f"    num_bytes: {st['num_bytes']}")
            lines.append(f"    num_shards: {st['num_shards']}")

    return "\n".join(lines)


def generate_config_entries(entities: dict[str, list[dict]] | None = None) -> str:
    """Generate config YAML entries for the parquet sub-tables.

    These go in the ``configs:`` section of README.md frontmatter,
    alongside the existing .jsonl.gz configs.
    """
    if entities is None:
        entities = all_entities_info()

    lines: list[str] = []

    for entity_name, sub_tables in entities.items():
        for st in sub_tables:
            lines.append(f"  - config_name: {st['config_name']}")
            lines.append(f"    data_dir: {st['data_dir']}")

    return "\n".join(lines)


# ── README.md update ─────────────────────────────────────────────────

README_PATH = Path(__file__).resolve().parent.parent / "README.md"


def update_readme(
    entities: dict[str, list[dict]] | None = None,
    readme_path: Path = README_PATH,
) -> None:
    """Inject dataset_info and parquet configs into README.md frontmatter.

    The README is expected to have YAML frontmatter delimited by ``---``.
    This function:

    1. Adds/replaces the ``dataset_info:`` block with fresh parquet metadata.
    2. Appends parquet config entries to the existing ``configs:`` section
       (skipping any that already exist).
    """
    if entities is None:
        entities = all_entities_info()

    text = readme_path.read_text(encoding="utf-8")

    # Tolerantly locate the opening frontmatter delimiter:
    #   - skip a UTF-8 BOM
    #   - skip leading blank lines
    # Then look for `---` followed by a newline.
    BOM = "﻿"
    prefix = ""
    body_text = text
    if body_text.startswith(BOM):
        prefix += BOM
        body_text = body_text[len(BOM):]
    stripped = body_text.lstrip("\n")
    if stripped != body_text:
        prefix += body_text[: len(body_text) - len(stripped)]
        body_text = stripped

    has_open_delim = body_text.startswith("---\n") or body_text == "---" or body_text.startswith("---\r\n")
    end_fm = body_text.find("\n---", 3) if has_open_delim else -1

    if not has_open_delim or end_fm < 0:
        # No (well-formed) frontmatter — insert a fresh block at the top.
        # The dataset_info generator will fill it in below.
        if has_open_delim and end_fm < 0:
            log.warning(
                "Malformed YAML frontmatter in %s (opening --- found, no closing ---); "
                "inserting a fresh block above existing content",
                readme_path,
            )
        else:
            log.warning(
                "No YAML frontmatter found in %s; inserting a fresh block",
                readme_path,
            )
        frontmatter = ""
        # Drop only the BOM from the preserved body; keep any user blank
        # lines intact below the new frontmatter delimiter.
        body = text[len(BOM):] if text.startswith(BOM) else text
    else:
        frontmatter = body_text[3:end_fm]
        body = body_text[end_fm + 4:]

    # Generate fresh dataset_info
    ds_info_yaml = generate_dataset_info_yaml(entities)

    # Remove existing dataset_info block
    lines = frontmatter.split("\n")
    filtered: list[str] = []
    in_dataset_info = False
    for line in lines:
        if line.startswith("dataset_info:"):
            in_dataset_info = True
            continue
        if in_dataset_info:
            # dataset_info entries are indented 2+ spaces
            if line.startswith("  ") or line == "":
                continue
            in_dataset_info = False
        filtered.append(line)

    frontmatter = "\n".join(filtered).rstrip()

    # Collapse consecutive blank lines (artifact of removing dataset_info block)
    import re as _re
    frontmatter = _re.sub(r'\n{3,}', '\n\n', frontmatter)

    # Append dataset_info at end of frontmatter
    frontmatter += "\n" + ds_info_yaml

    # Add parquet config entries (skip existing)
    existing_configs = {
        line.split("config_name:")[1].strip()
        for line in frontmatter.split("\n")
        if "config_name:" in line
    }

    new_config_lines: list[str] = []
    for entity_name, sub_tables in entities.items():
        for st in sub_tables:
            if st["config_name"] not in existing_configs:
                new_config_lines.append(
                    f"  - config_name: {st['config_name']}\n"
                    f"    data_dir: {st['data_dir']}"
                )

    if new_config_lines:
        # Find the configs: section and append after last config entry
        fm_lines = frontmatter.split("\n")
        last_config_idx = -1
        for i, line in enumerate(fm_lines):
            if "config_name:" in line:
                last_config_idx = i
            elif line.startswith("    data_files:") or line.startswith("    data_dir:") or line.startswith("    default:"):
                if last_config_idx >= 0:
                    last_config_idx = i

        if last_config_idx >= 0:
            # Find the end of the last config block
            # (last line that's indented under the config)
            insert_idx = last_config_idx + 1
            # Skip any remaining lines belonging to this config
            while insert_idx < len(fm_lines) and (
                fm_lines[insert_idx].startswith("    ")
                or fm_lines[insert_idx].startswith("  - ")
            ):
                if fm_lines[insert_idx].startswith("  - config_name:"):
                    break
                insert_idx += 1

            for cl in new_config_lines:
                fm_lines.insert(insert_idx, cl)
                insert_idx += 1

            frontmatter = "\n".join(fm_lines)
        else:
            # No existing configs: open a fresh configs: section at the
            # top of the frontmatter so the parquet sub-tables are
            # discoverable by the HF dataset loader.
            frontmatter = (
                "configs:\n"
                + "\n".join(new_config_lines)
                + ("\n" + frontmatter if frontmatter else "")
            )

    # Reassemble — normalise blank lines between frontmatter and body
    body = body.lstrip('\n')
    readme_path.write_text(f'---\n{frontmatter}\n---\n\n{body}')
    log.info("Updated %s with dataset_info for %d entities", readme_path, len(entities))


# ── Targeted README update (called by extract.py) ───────────────────


def update_entity(entity: str) -> None:
    """Update dataset_info for a single entity's sub-tables in README.md.

    Reads the current README frontmatter, replaces only the configs
    belonging to *entity*, and writes back.  Faster than regenerating
    all 86+ configs when only one entity changed.
    """
    info = entity_parquet_info(entity)
    if not info:
        log.debug("No parquet data for entity: %s, skipping metadata update", entity)
        return

    entities = {entity: info}
    _update_readme_entities(entities)
    log.info("Updated README.md metadata for entity: %s (%d sub-tables)", entity, len(info))


def update_all() -> None:
    """Regenerate dataset_info for all entities and update README.md."""
    entities = all_entities_info()
    if not entities:
        log.info("No parquet data found, skipping metadata update")
        return
    _update_readme_entities(entities)
    log.info("Updated README.md metadata for %d entities", len(entities))


def _update_readme_entities(entities: dict[str, list[dict]]) -> None:
    """Regenerate the full dataset_info block in README.md.

    The targeted-entity approach had ordering issues (appending at the end
    instead of preserving position).  Since metadata.py uses sampling
    (not reading all 38k footers), regenerating the full block is fast
    enough (< 2 seconds) and avoids ordering drift.
    """
    # Gather all entity info — use provided entities plus whatever's
    # already on disk for non-targeted entities.
    full_entities: dict[str, list[dict]] = {}
    for entity in ENTITY_TYPES:
        if entity in entities:
            full_entities[entity] = entities[entity]
        else:
            info = entity_parquet_info(entity)
            if info:
                full_entities[entity] = info

    update_readme(full_entities)


# ── CLI (standalone use, e.g. after batch git-add) ───────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate HuggingFace dataset metadata from parquet shards",
    )
    parser.add_argument(
        "--entity", type=str, default=None,
        help="Generate metadata for a single entity (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generated YAML without modifying README.md",
    )

    args = parser.parse_args()

    if args.entity:
        info = {args.entity: entity_parquet_info(args.entity)}
        if not info[args.entity]:
            log.error("No parquet data found for entity: %s", args.entity)
            sys.exit(1)
    else:
        info = all_entities_info()

    if args.dry_run:
        print(generate_dataset_info_yaml(info))
    else:
        if args.entity:
            update_entity(args.entity)
        else:
            update_all()


if __name__ == "__main__":
    main()
