#!/usr/bin/env python3
"""Convert a CSR .npz to uncompressed .npy files for np.memmap.

scipy.sparse.load_npz fully materialises the CSR into RAM (~30min for the
12GB citation graph on slow Lustre). Saving indptr/indices as raw .npy lets
the loader use np.memmap — the OS loads pages on-demand as the algorithm
accesses them (near-instant startup, only accessed pages faulted in).

Output files (alongside the .npz):
  <stem>.offsets.npy   — indptr (uint64, shape (n+1,))
  <stem>.targets.npy   — indices (uint32, shape (nnz,))
  <stem>.shape.npy     — [n_rows, n_cols] (int64, shape (2,))

Usage:
    python3 -m sync.convert_csr_npy path/to/work_referenced_works.npz [more.npz ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse


def convert(npz_path: str) -> None:
    npz_path = Path(npz_path)
    print(f"Loading {npz_path.name}...", flush=True)
    csr = sparse.load_npz(str(npz_path))
    csr.eliminate_zeros()

    offsets = csr.indptr.astype(np.uint64)
    targets = csr.indices.astype(np.uint32)
    shape = np.array(csr.shape, dtype=np.int64)

    offsets_path = npz_path.with_suffix(".offsets.npy")
    targets_path = npz_path.with_suffix(".targets.npy")
    shape_path = npz_path.with_suffix(".shape.npy")

    print(f"  Writing {offsets_path.name} ({offsets.shape}, {offsets.dtype})...", flush=True)
    offsets.tofile(str(offsets_path))
    print(f"  Writing {targets_path.name} ({targets.shape}, {targets.dtype})...", flush=True)
    targets.tofile(str(targets_path))
    shape.tofile(str(shape_path))
    print(f"  Done. CSR: {csr.shape[0]:,} nodes, {csr.nnz:,} edges", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 -m sync.convert_csr_npy <file.npz> [more.npz ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        convert(p)


if __name__ == "__main__":
    main()
