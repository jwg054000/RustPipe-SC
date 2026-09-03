# Packet contract (10x Chromium / scRNA)

RustPipe-SC runs **downstream** of [Cell Ranger](https://www.10xgenomics.com/support/software/cell-ranger) or [nf-core/scrnaseq](https://nf-co.re/scrnaseq). It accelerates QC → normalize → HVG → PCA → kNN → Leiden → markers.

It is **not** a Cell Ranger replacement: no BCL → FASTQ, no barcode/UMI counting, no STARSolo. It is **not** prairie-rna-stream (bulk STAR → rustqc → rustpipe). Never vendor this repo into prairie-rna-stream.

## Inputs (`--input`)

Exactly one cell-count matrix:

| path | source |
| --- | --- |
| `filtered_feature_bc_matrix.h5` (or other 10x `/matrix` H5) | Cell Ranger, cellranger-arc |
| `*.h5ad` | Scanpy / nf-core/scrnaseq AnnData |
| cells × genes CSV (`barcode,gene1,gene2,…`) | custom / Scanpy export |
| genes × cells CSV (`gene,cell1,cell2,…`) | legacy; auto-transposed |

`--qc-bam` / `--qc-gtf` are optional library QC (PATH `rustqc rna` into `output/libqc/`). They are **not** the cell matrix. `libqc/featurecounts/` must be ignored as cells.

## Fail closed (wrong assay)

Reject `--input` and exit non-zero. Do not transpose, do not “fix”, do not count a BAM.

| handed in | why |
| --- | --- |
| `*.bam` / `*.sam` / `*.cram` (including STAR `*.Aligned.sortedByCoord.out.bam`) | alignments; bulk STAR belongs in prairie-rna-stream |
| Cell Ranger `possorted_genome_bam.bam` as `--input` | that path is `--qc-bam` only |
| rustqc `*.featureCounts.tsv` (header `Geneid,Chr,Start,End,Strand,Length` or `# Program:featureCounts`) | library QC table, not a cell matrix |
| `matrix.mtx` | MTX directory is not an input; use the 10x H5 or H5AD |

A prairie-rna-stream packet (`counts.parquet` + `rustpipe_out/` + `run.json`) is the wrong product. Do not ingest it here.

## Outputs (`rustpipe-sc pipeline --output DIR`)

What this crate actually writes today (v0.3.0). Column names are as emitted; do not invent aliases.

| file | columns / role |
| --- | --- |
| `qc_metrics.csv` | `barcode`, `n_genes_by_counts`, `total_counts`, `pct_counts_mt` |
| `hvg_genes.csv` | `gene` |
| `pca_scores.csv` | `sample` (cell barcode values), `PC1`…`PCk` |
| `knn.csv` | `neighbor_1`,`distance_1`,… for k neighbors |
| `clusters.csv` | `barcode`, `cluster` |
| `markers.csv` | `cluster`, `gene`, `score`, `pval`, `pval_adj`, `log2fc` |
| `pipeline_timings.json` | `pipeline`=`rustpipe-sc`, `version`, `seed`, `input`, cell/gene counts, `hvg_flavor`, `n_pcs`, `knn_k`, `steps[{step,seconds}]`, `total_seconds`. Step names are unchanged (`load`, `qc_filter`, `normalize`, `hvg`, `scale`, `pca`, `knn`, `leiden`, `markers`). Internally QC+normalize+HVG share two CSR nnz passes; `scale` is mean/std only (z-score+clip is implicit in PCA). |
| `libqc/` | only if `--qc-bam`; rustqc tree. Not the cell matrix. |

The `qc` subcommand also writes `filtered_barcodes.csv`. The full `pipeline` subcommand does **not** write `filtered_cells.csv` / `normalized.csv` / `pca_loadings.csv` (those names in older README lists are not produced).

No hashed `run.json` yet. `pipeline_timings.json` is the provenance file. A Prairie SDK ingest should fail-closed if `pipeline` ≠ `"rustpipe-sc"`, if a bulk STAR/featureCounts file is present as the matrix, or if `clusters.csv` / `pca_scores.csv` / `pipeline_timings.json` are missing.

## Image pin

Pull the published linux/amd64 image. Never `docker build` this tree from prairie-rna-stream.

```
ghcr.io/jwg054000/rustpipe-sc:0.3.0@sha256:c1d9cc7581ac521e1dae5aae6ac2a63d457c822c4ddcc8507e457d44a21dc8c2
```

GHCR is public. See README (Container).
