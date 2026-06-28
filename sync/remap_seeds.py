#!/usr/bin/env python3
"""Remap seed pairs from OpenAlex IDs to a graph's compact index space.

The canonical ``seed-pairs-compact.csv`` carries both the Graph-1 compact
indices (``seed_a``/``seed_b``) AND the original OpenAlex work IDs
(``oa_id_a``/``oa_id_b``). To run any at-scale phase on Graph 2 or Graph 3,
the seeds must be re-expressed in THAT graph's compact index space.

Each graph's CSR ships an ``id_map.npy``: a sorted ``uint64`` array where
``id_map[compact_index] == OpenAlex ID``. So the reverse map is a binary
search: ``compact = searchsorted(id_map, oa_id)`` (with a presence check).

Usage:
    python3 -m sync.remap_seeds \\
        --seed-csv .../seed-pairs-compact.csv \\
        --id-map   .../csr/combined_work_authors.id_map.npy \\
        --output   .../seeds/seed-pairs-graph2.csv
"""

from __future__ import annotations

import argparse
import csv

import numpy as np


def remap(seed_csv: str, id_map_path: str, output_csv: str) -> None:
    # mmap: the id_map is multi-GB; searchsorted only needs sequential reads.
    id_map = np.load(id_map_path, mmap_mode="r")
    n = len(id_map)

    with open(seed_csv, newline="") as f:
        rows_in = list(csv.DictReader(f))

    # Preserve column order; overwrite seed_a/seed_b with the new compact ids.
    fieldnames = list(rows_in[0].keys()) if rows_in else []
    out_rows: list[dict] = []
    missing = 0
    for r in rows_in:
        try:
            oa_a = int(r["oa_id_a"])
            oa_b = int(r["oa_id_b"])
        except (KeyError, ValueError):
            missing += 1
            continue
        ca = int(np.searchsorted(id_map, oa_a))
        cb = int(np.searchsorted(id_map, oa_b))
        # Presence check: the oa_id must actually be in this graph's id_map.
        if ca >= n or int(id_map[ca]) != oa_a or cb >= n or int(id_map[cb]) != oa_b:
            missing += 1
            continue
        r["seed_a"] = str(ca)
        r["seed_b"] = str(cb)
        out_rows.append(r)

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(
        f"remapped {len(out_rows)}/{len(rows_in)} pairs "
        f"({missing} dropped — a node absent from this graph's id_map); "
        f"wrote {output_csv}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed-csv", required=True, help="seed-pairs-compact.csv (with oa_id_a/oa_id_b)")
    p.add_argument("--id-map", required=True, help="graph <name>.id_map.npy (sorted uint64)")
    p.add_argument("--output", required=True, help="output seed-pairs-<graph>.csv")
    a = p.parse_args()
    remap(a.seed_csv, a.id_map, a.output)


if __name__ == "__main__":
    main()
