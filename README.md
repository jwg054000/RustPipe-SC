# RustPipe-SC

Fast single-cell and spatial transcriptomics in Rust. Accelerates the compute-bound steps of scRNA-seq analysis (QC, normalization, HVG, PCA, kNN, Leiden clustering, marker detection) while maintaining concordance with Scanpy.

Designed to run downstream of [Seqera](https://seqera.io/) / [nf-core/scrnaseq](https://nf-co.re/scrnaseq) pipelines that output H5AD or 10x CellRanger H5 files.

## Performance

Measured on Apple M3 Pro, single-threaded where noted. All timings are wall-clock.

### PBMC 3K (2,700 cells x 32,738 genes)

| Step | Scanpy | RustPipe-SC | Speedup |
|------|-------:|------------:|--------:|
| QC + Filter | 76 ms | 88 ms | ~1x |
| Normalize | 11 ms | 10 ms | ~1x |
| HVG | 305 ms | 12 ms | 25x |
| Scale + PCA | 4,150 ms | 150 ms | 28x |
| kNN (k=15) | 7,501 ms | 31 ms | 242x |
| Leiden | 531 ms | 17 ms | 31x |
| Markers | — | 155 ms | — |
| **Total pipeline** | **12,574 ms** | **526 ms** | **24x** |

### Scaling (synthetic data, 15K genes)

| Cells | Normalize | HVG | Scale+PCA | kNN | Leiden | Total |
|------:|----------:|----:|----------:|----:|-------:|------:|
| 3K | 0.03s | 0.03s | 0.18s | 0.04s | 0.06s | 0.38s |
| 10K | 0.09s | 0.13s | 0.62s | 0.36s | 0.23s | 1.57s |
| 20K | 0.34s | 0.37s | 1.16s | 1.41s | 0.81s | 4.43s |
| 50K | 0.64s | 0.95s | 2.58s | 11.31s | 3.80s | 20.13s |

At 10K cells, RustPipe-SC achieves **22x speedup** over Scanpy on compute-bound steps (PCA, kNN, Leiden, markers). The value compounds with iterative workflows: a 20-parameter sweep at 10K cells saves 3+ minutes.

### Concordance with Scanpy

| Metric | Value |
|--------|-------|
| PCA PC1-5 Pearson \|r\| | 1.000000 |
| Leiden ARI | 0.90 |
| QC cell count (min_genes=200, max_mt=5%) | 2643/2700 (exact match) |

## Input Formats

| Format | Extension | Source |
|--------|-----------|--------|
| H5AD (AnnData) | `.h5ad` | Scanpy, nf-core/scrnaseq |
| 10x CellRanger H5 | `.h5` | CellRanger, cellranger-arc |
| CSV (cells x genes) | `.csv` | Custom pipelines |
| CSV (genes x cells) | `.csv` | Legacy format (auto-transposed) |

Auto-detection: `.h5` and `.h5ad` files are automatically detected as 10x or AnnData format.

### HDF5 Support

H5AD and 10x H5 reading requires the `hdf5` feature and a system HDF5 library:

```bash
# macOS
brew install hdf5

# Build with HDF5 support
HDF5_DIR=$(brew --prefix hdf5) cargo build --release --features hdf5
```

Without the `hdf5` feature, CSV input is still supported.

## Installation

```bash
git clone https://github.com/jwg054000/RustPipe-SC.git
cd rustpipe-sc

# Without HDF5 (CSV only)
cargo build --release

# With HDF5 support (recommended)
HDF5_DIR=$(brew --prefix hdf5) cargo build --release --features hdf5
```

## Usage

### Full Pipeline

```bash
# H5AD input (Scanpy/nf-core output)
rustpipe-sc pipeline \
  --input sample.h5ad \
  --output results/ \
  --n-hvg 2000 --n-pcs 50 --knn-k 15

# 10x CellRanger H5 input
rustpipe-sc pipeline \
  --input filtered_feature_bc_matrix.h5 \
  --output results/ \
  --max-pct-mt 20.0

# CSV input
rustpipe-sc pipeline \
  --input expression.csv \
  --output results/
```

### Individual Steps

```bash
# QC metrics + cell filtering
rustpipe-sc qc --input raw.h5ad --output qc/ --min-genes 200 --max-pct-mt 5.0

# Normalize + log1p
rustpipe-sc normalize --input qc/filtered.csv --output norm/

# HVG selection (Seurat v3 method)
rustpipe-sc hvg --input norm/normalized.csv --output hvg/ --n-top-genes 2000

# PCA
rustpipe-sc pca --input hvg/expression_hvg.csv --output pca/ --n-components 50

# kNN graph
rustpipe-sc knn --input pca/pca_scores.csv --output knn/ --k 15

# Leiden clustering
rustpipe-sc leiden --input knn/knn.csv --output clusters/ --resolution 1.0

# Spatial autocorrelation (Moran's I)
rustpipe-sc morans-i \
  --input expression.csv \
  --coords spatial_coords.csv \
  --output spatial/
```

### Global Options

```
--threads N    Number of threads (0 = all cores, default: 0)
--seed N       Random seed for reproducibility (default: 42)
-v/-vv/-vvv    Verbosity level
```

## Pipeline Outputs

```
output/
  qc_metrics.csv          # Per-cell QC: n_genes, total_counts, pct_mt
  filtered_cells.csv      # Cells passing QC
  normalized.csv          # Library-size normalized, log1p transformed
  hvg_genes.csv           # Selected highly variable genes
  pca_scores.csv          # PCA embeddings (cells x PCs)
  pca_loadings.csv        # Gene loadings per PC
  pca_variance.csv        # Variance explained per PC
  knn.csv                 # k-nearest neighbor graph
  clusters.csv            # Leiden cluster assignments
  markers.csv             # Cluster marker genes (Wilcoxon rank-sum)
```

## Architecture

```
src/
  main.rs       CLI + pipeline orchestration
  h5ad.rs       H5AD and 10x H5 I/O (feature-gated)
  io.rs         CSV I/O
  sparse.rs     CSR sparse matrix ops, implicit-centering matmul
  qc.rs         Per-cell QC metrics, MAD filtering
  normalize.rs  Library-size normalization + log1p
  hvg_sc.rs     Seurat v3 HVG selection
  scale.rs      Z-score scaling with clip
  pca.rs        Randomized SVD (dense path, sparse-ready)
  knn.rs        Brute-force k-nearest neighbors
  graph.rs      CSR graph, kNN→graph conversion
  leiden.rs     Leiden community detection
  markers.rs    Wilcoxon rank-sum marker gene detection
  morans_i.rs   Moran's I spatial autocorrelation
  gsea.rs       Gene set enrichment analysis
  stats_sc.rs   Statistical utilities (Welford, BH, MAD)
```

## Seqera Integration

RustPipe-SC reads the standard output formats from nf-core/scrnaseq:

```
nf-core/scrnaseq (cellranger/alevin/starsolo)
  --> filtered_feature_bc_matrix.h5  (10x format)
  --> adata.h5ad                      (AnnData format)
      --> rustpipe-sc pipeline --input adata.h5ad --output results/
```

## Testing

```bash
cargo test                    # 104 unit tests, ~0.06s
cargo test --features hdf5    # Includes H5AD I/O tests (requires HDF5)
```

## Known Limitations

- kNN is brute-force O(n^2): fast up to ~50K cells, needs approximate NN (HNSW) for larger
- No UMAP embedding (use Scanpy or umap-rs downstream)
- HVG Seurat v3 overlap with Scanpy is ~50% due to binning boundary sensitivity
- Spatial segmentation not yet implemented

## License

GPL-3.0 - see [LICENSE](LICENSE)

## Citation

Prairie Bio, 2026. RustPipe-SC: Fast single-cell and spatial transcriptomics in Rust.
