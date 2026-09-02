# CI 10x H5 fixture

`filtered_feature_bc_matrix.h5` is a **synthetic** Cell Ranger `/matrix` file
(80 cells × 2120 genes). It is not a public PBMC download.

Regenerate:

```bash
python3 scripts/gen_10x_ci_fixture.py
```

Sized so default `rustpipe-sc pipeline` flags succeed:

- each cell has 240 detected genes (`--min-genes 200`)
- `MT-*` genes are named and low-count (`--max-pct-mt 5`)
- ≥2000 genes pass `min_cells=3` (`--n-hvg 2000`)
- 80 cells (`--knn-k 15`; `n_pcs` already clamps to rank)
