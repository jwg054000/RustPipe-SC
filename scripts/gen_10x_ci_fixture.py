#!/usr/bin/env python3
"""Write testdata/filtered_feature_bc_matrix.h5 — tiny synthetic 10x /matrix H5.

Layout matches Cell Ranger filtered_feature_bc_matrix.h5 so rustpipe-sc
load_input → read_10x_h5 runs. Sized to pass default pipeline flags:

  --min-genes 200 --max-pct-mt 5 --n-hvg 2000 --n-pcs 50 --knn-k 15

Not a public PBMC file. Regenerates the committed binary; do not vendor RustQC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

# Human MT- prefixes so qc.rs pct_counts_mt is real and stays well under 5%.
MT_NAMES = (
    "MT-ND1",
    "MT-ND2",
    "MT-CO1",
    "MT-CO2",
    "MT-ATP6",
    "MT-ATP8",
    "MT-CYB",
    "MT-ND4",
)

N_CELLS = 80
N_GENES = 2120  # 8 MT + 2112 others; default --n-hvg 2000 needs ≥2000 after min_cells=3
N_GENES_PER_CELL = 240
SEED = 42


def gene_names() -> list[bytes]:
    names = [n.encode("ascii") for n in MT_NAMES]
    for i in range(len(MT_NAMES), N_GENES):
        names.append(f"GENE{i:04d}".encode("ascii"))
    return names


def barcodes() -> list[bytes]:
    # 16-base + "-1", Cell Ranger style, unique and deterministic.
    out = []
    alphabet = b"ACGT"
    for i in range(N_CELLS):
        seq = bytearray(16)
        n = i
        for pos in range(16):
            seq[pos] = alphabet[n & 3]
            n >>= 2
        out.append(bytes(seq) + b"-1")
    return out


def build_csc(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSC genes × cells. Every gene in ≥3 cells; every cell has 240 genes."""
    cells_for_gene: list[set[int]] = [set() for _ in range(N_GENES)]
    genes_for_cell: list[set[int]] = [set() for _ in range(N_CELLS)]

    # Round-robin so min_cells=3 holds for all genes (HVG default 2000).
    for g in range(N_GENES):
        for k in range(3):
            c = (g + k * 17) % N_CELLS
            cells_for_gene[g].add(c)
            genes_for_cell[c].add(g)

    for c in range(N_CELLS):
        while len(genes_for_cell[c]) < N_GENES_PER_CELL:
            g = int(rng.integers(0, N_GENES))
            genes_for_cell[c].add(g)
            cells_for_gene[g].add(c)

    data: list[int] = []
    indices: list[int] = []
    indptr = np.zeros(N_CELLS + 1, dtype=np.int64)
    for c in range(N_CELLS):
        genes = sorted(genes_for_cell[c])
        for g in genes:
            if g < len(MT_NAMES):
                count = 1  # ~0.2% MT if the rest average ~4 UMIs
            else:
                count = 1 + int(rng.integers(0, 8))
            data.append(count)
            indices.append(g)
        indptr[c + 1] = len(data)

    return (
        np.asarray(data, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
        indptr,
    )


def write_h5(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    data, indices, indptr = build_csc(rng)
    names = gene_names()
    bc = barcodes()
    ids = [f"ENSG{i:011d}".encode("ascii") for i in range(N_GENES)]
    feature_type = [b"Gene Expression"] * N_GENES
    genome = [b"GRCh38"] * N_GENES

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        matrix = f.create_group("matrix")
        matrix.create_dataset("shape", data=np.array([N_GENES, N_CELLS], dtype=np.int64))
        matrix.create_dataset("data", data=data)
        matrix.create_dataset("indices", data=indices)
        matrix.create_dataset("indptr", data=indptr)
        matrix.create_dataset("barcodes", data=np.array(bc, dtype="S18"))
        features = matrix.create_group("features")
        features.create_dataset("name", data=np.array(names, dtype="S16"))
        features.create_dataset("id", data=np.array(ids, dtype="S15"))
        features.create_dataset("feature_type", data=np.array(feature_type, dtype="S16"))
        features.create_dataset("genome", data=np.array(genome, dtype="S8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "testdata" / "filtered_feature_bc_matrix.h5",
    )
    args = parser.parse_args()
    write_h5(args.output)
    size = args.output.stat().st_size
    print(f"wrote {args.output} ({size} bytes)  {N_CELLS} cells x {N_GENES} genes")


if __name__ == "__main__":
    main()
