"""Shared infrastructure for OpenAlex snapshot data management.

Self-contained — no imports from the thesis experiments pipeline.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

# ── Paths ────────────────────────────────────────────────────────────────

# The data root is the openalex-snapshot/ directory (sibling of sync/)
# sync/ lives in the parent GitHub repo; openalex-snapshot/ is the HF dataset.
# Lazy resolution: these compute on first attribute access so that importing
# sync.common on CI (where openalex-snapshot/ doesn't exist) doesn't fail.
# CI code overrides SNAPSHOT_DIR before use (via env var or direct assignment).


def _resolve_sync_root() -> Path:
    """Lazily resolve SYNC_ROOT. Overridable via OPENALEX_SYNC_ROOT env var."""
    env = os.environ.get("OPENALEX_SYNC_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "openalex-snapshot"


def _resolve_snapshot_dir() -> Path:
    """Lazily resolve SNAPSHOT_DIR. Overridable via OPENALEX_SNAPSHOT env var."""
    env = os.environ.get("OPENALEX_SNAPSHOT")
    if env:
        return Path(env)
    return _resolve_sync_root() / "data"


class _LazyPath:
    """Descriptor that resolves a path on first access, then caches it."""

    def __init__(self, resolver):
        self._resolver = resolver
        self._attr_name = None

    def __set_name__(self, owner, name):
        self._attr_name = f"_lazy_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        cached = getattr(obj, self._attr_name, self)
        if cached is self:
            cached = self._resolver()
            object.__setattr__(obj, self._attr_name, cached)
        return cached

    def __set__(self, obj, value):
        object.__setattr__(obj, self._attr_name, value)


class _Paths:
    """Module-level lazy path container.

    SYNC_ROOT and SNAPSHOT_DIR resolve on first access.
    Assigning to them (e.g. ``common.SNAPSHOT_DIR = tmp_dir``) overrides
    the lazy resolution permanently.
    """

    SYNC_ROOT: Path = _LazyPath(_resolve_sync_root)       # type: ignore[assignment]
    SNAPSHOT_DIR: Path = _LazyPath(_resolve_snapshot_dir)  # type: ignore[assignment]

    @property
    def DATA_DIR(self) -> Path:
        return self.SYNC_ROOT / "data"


# Module-level singleton. Importing sync.common does not trigger resolution.
_paths = _Paths()


def __getattr__(name):
    """Expose _Paths attributes at module level for backward compatibility."""
    if name in ("SYNC_ROOT", "DATA_DIR", "SNAPSHOT_DIR"):
        return getattr(_paths, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __setattr__(name, value):
    """Allow ``common.SNAPSHOT_DIR = ...`` to override lazy resolution."""
    if name in ("SYNC_ROOT", "SNAPSHOT_DIR"):
        setattr(_paths, name, value)
    else:
        globals()[name] = value


def __delattr__(name):
    raise AttributeError(f"cannot delete {name!r} from {__name__!r}")

# Staging directory for parquet writes — fast local storage (SSD/NVMe).
# Completed files are moved to the output dir on close.
STAGING_DIR = (
    Path(os.environ.get("OPENALEX_STAGING", ""))
    if os.environ.get("OPENALEX_STAGING")
    else None
)

# ── I/O acceleration: orjson > json fallback ─────────────────────────────

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

try:
    from isal import igzip

    def _gzip_open_isal(path, mode="rb", **kwargs):
        if "compresslevel" in kwargs and kwargs["compresslevel"] > 3:
            kwargs["compresslevel"] = 3
        return igzip.open(path, mode, **kwargs)

    _gzip_open: Callable[..., Any] = _gzip_open_isal
except ImportError:
    _gzip_open = gzip.open

# ── Entity types ─────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "works",
    "authors",
    "institutions",
    "sources",
    "topics",
    "subfields",
    "fields",
    "domains",
    "publishers",
    "funders",
    "concepts",
]

ENTITY_PREFIX_MAP: dict[str, str] = {
    "works": "W",
    "authors": "A",
    "sources": "S",
    "institutions": "I",
    "concepts": "C",
    "topics": "T",
    "publishers": "P",
    "funders": "F",
    "domains": "Do",
    "fields": "Fi",
    "subfields": "Sf",
}

def nested_rt_path(rt: str) -> str:
    """Map a flat relationship type name to its nested entity/subtable path.

    Convention: ``{entity_singular}_{subtable}`` → ``{entity_plural}/{subtable}``.
    The plural is derived by appending 's' to the singular prefix
    (e.g. ``work_authorships`` → ``works/authorships``).
    Also handles hyphenated variants (``work-types`` → ``works/types``).
    Returns the raw ``rt`` unchanged if no separator is found.
    """
    for sep in ("_", "-"):
        idx = rt.find(sep)
        if idx > 0:
            singular = rt[:idx]
            subtable = rt[idx + 1:]
            # Derive plural: strip trailing 's' if present, then add 's'
            # Handles irregular singulars like "sdg" → "sdgs"
            if singular.endswith("ss"):
                plural = singular + "es"
            elif singular.endswith("s"):
                plural = singular
            else:
                plural = singular + "s"
            return f"{plural}/{subtable}"
    return rt


def rt_dir(base: Path, rt: str) -> Path:
    """Construct the output directory for a relationship type."""
    return base / nested_rt_path(rt)


# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("openalex-sync")

# ── Skipped-file tracking ────────────────────────────────────────────────

_skipped_missing_files: list[str] = []


# ── Helpers ──────────────────────────────────────────────────────────────


def extract_id(value: str | int | None) -> int | None:
    """Strip an OpenAlex URL or identifier to its numeric suffix."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    part = value.rsplit("/", 1)[-1]
    match = re.search(r"(\d+)$", part)
    if match is None:
        return None
    return int(match.group(1))


def format_size(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def iter_source_files(entity_type: str) -> list[Path]:
    """List source files for an entity type from SNAPSHOT_DIR."""
    entity_dir = SNAPSHOT_DIR / entity_type
    if not entity_dir.is_dir():
        log.warning("Source directory not found: %s", entity_dir)
        return []
    return sorted(f for f in entity_dir.rglob("*.gz") if not f.name.startswith(".") and (f.name.endswith(".jsonl.gz") or f.suffix == ".gz"))


def _entity_from_path(path: Path) -> str | None:
    """Derive entity type from a snapshot file path."""
    try:
        rel = path.relative_to(SNAPSHOT_DIR)
    except ValueError:
        return None
    parts = rel.parts
    return parts[0] if len(parts) >= 2 else None


def _file_in_manifest(entity: str, filename: str, partition_dir: str) -> bool:
    """Check whether a file appears in the local entity manifest."""
    manifest_path = SNAPSHOT_DIR / entity / "manifest"
    if not manifest_path.exists():
        return True
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True
    expected_suffix = f"{entity}/{partition_dir}/{filename}"
    for entry in manifest.get("entries", []):
        url: str = entry.get("url", "")
        if url.endswith(expected_suffix):
            return True
    return False


def _redownload_corrupt(path: Path) -> None:
    """Redownload a corrupt file from OpenAlex S3."""
    from sync.download import S3_BUCKET, _get_s3_client

    try:
        rel = path.relative_to(SNAPSHOT_DIR)
    except ValueError:
        raise FileNotFoundError(f"Not under SNAPSHOT_DIR: {path}")

    s3_key = f"data/{rel}"
    log.info("Redownloading corrupt file from s3://%s/%s", S3_BUCKET, s3_key)
    try:
        from botocore.exceptions import ClientError
        s3 = _get_s3_client()
        path.unlink(missing_ok=True)
        s3.download_file(S3_BUCKET, s3_key, str(path))
        log.info("Redownloaded %s (%d bytes)", path.name, path.stat().st_size)
    except Exception as exc:
        error_str = str(exc)
        if "404" in error_str:
            entity = _entity_from_path(path)
            if entity:
                try:
                    rel2 = path.relative_to(SNAPSHOT_DIR / entity)
                    partition_dir = rel2.parts[0]
                except (ValueError, IndexError):
                    partition_dir = ""
                in_manifest = _file_in_manifest(entity, path.name, partition_dir)
                if not in_manifest:
                    log.warning("File %s not on S3 and absent from %s manifest — stale, skipping", path.name, entity)
                else:
                    log.warning("File %s IS in %s manifest but S3 returned 404", path.name, entity)
            raise FileNotFoundError(f"Not found on S3: {path.name}") from exc
        raise


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream JSON objects from a .gz or .jsonl file.

    On EOFError (truncated gzip), redownloads and retries once.
    """
    opener = _gzip_open if path.name.endswith(".jsonl.gz") or path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield _json_loads(line)
                    except (json.JSONDecodeError, ValueError):
                        pass
    except EOFError:
        log.warning("Truncated gzip file: %s — attempting redownload", path.name)
        try:
            _redownload_corrupt(path)
            with opener(path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield _json_loads(line)
                        except (json.JSONDecodeError, ValueError):
                            pass
        except FileNotFoundError:
            try:
                rel = str(path.relative_to(SNAPSHOT_DIR))
            except ValueError:
                rel = str(path)
            _skipped_missing_files.append(rel)
            log.warning("Skipping missing file: %s (total skipped: %d)", path.name, len(_skipped_missing_files))


def get_skipped_missing_files() -> list[str]:
    return list(_skipped_missing_files)


def reset_skipped_files() -> None:
    _skipped_missing_files.clear()


def create_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relative_source_path(path: Path) -> str:
    """Return a stable relative path string for progress tracking."""
    try:
        rel = str(path.relative_to(SNAPSHOT_DIR))
    except ValueError:
        rel = str(path)
    if rel.endswith(".jsonl.gz"):
        rel = rel[:-8] + ".gz"
    elif rel.endswith(".jsonl"):
        rel = rel[:-6] + ".gz"
    return rel
