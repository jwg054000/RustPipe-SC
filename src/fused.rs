//! Two-pass fused QC → (optional) normalize → HVG stats over CSR nonzeros.
//!
//! Replaces three full matrix scans (`compute_qc_metrics`, `normalize_log1p_sparse`,
//! `sparse_gene_stats`) plus a separate gene-filter scan. Biology is unchanged:
//! same QC thresholds, same per-cell totals on gene-filtered counts, same
//! seurat / seurat_v3 ranking.

use crate::hvg_sc::{self, HvgResult};
use crate::normalize;
use crate::qc::{self, QcMetrics};
use crate::sparse::SpMat;
use anyhow::Result;

/// Filtered, optionally log-normalized matrix plus HVG-ready gene stats.
pub struct FusedPrep {
    pub metrics: QcMetrics,
    pub keep_cells: Vec<bool>,
    pub filtered_names: Vec<String>,
    pub kept_var: Vec<String>,
    pub mat: SpMat,
    pub gene_means: Vec<f32>,
    pub gene_variances: Vec<f32>,
}

/// QC + gene filter + normalize + per-gene mean/variance in two CSR passes.
pub fn qc_normalize_hvg_stats(
    mat: &SpMat,
    var_names: &[String],
    obs_names: &[String],
    min_genes: u32,
    max_pct_mt: f32,
    min_cells: usize,
    target_sum: f32,
    skip_normalize: bool,
) -> FusedPrep {
    let (metrics, keep_cells, gene_counts) =
        qc::qc_metrics_and_kept_gene_counts(mat, var_names, obs_names, min_genes, max_pct_mt);
    let keep_genes = qc::gene_keep_from_counts(&gene_counts, min_cells);

    let filtered_names: Vec<String> = obs_names
        .iter()
        .zip(keep_cells.iter())
        .filter(|(_, &k)| k)
        .map(|(n, _)| n.clone())
        .collect();
    let kept_var: Vec<String> = var_names
        .iter()
        .zip(keep_genes.iter())
        .filter(|(_, &k)| k)
        .map(|(n, _)| n.clone())
        .collect();

    let (normed, gene_means, gene_variances) = normalize::normalize_log1p_subset_with_stats(
        mat,
        &keep_cells,
        &keep_genes,
        target_sum,
        skip_normalize,
    );

    FusedPrep {
        metrics,
        keep_cells,
        filtered_names,
        kept_var,
        mat: normed,
        gene_means,
        gene_variances,
    }
}

/// Rank HVGs from fused stats (no extra matrix scan).
pub fn select_hvg_from_stats(
    prep: &FusedPrep,
    n_top_genes: usize,
    hvg_flavor: &str,
) -> Result<HvgResult> {
    let n_top = n_top_genes.min(prep.kept_var.len());
    match hvg_flavor {
        "seurat_v3" => hvg_sc::select_hvg_vst_from_stats(
            &prep.gene_means,
            &prep.gene_variances,
            &prep.kept_var,
            n_top,
            prep.mat.rows(),
        ),
        _ => hvg_sc::select_hvg_seurat_from_stats(
            &prep.gene_means,
            &prep.gene_variances,
            &prep.kept_var,
            n_top,
            20,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize_log1p_sparse;
    use crate::qc::{apply_cell_filter, compute_qc_metrics, filter_cells_fixed, filter_genes};
    use crate::sparse::from_triplets;
    use rand::prelude::*;

    fn toy_sc_matrix() -> (SpMat, Vec<String>, Vec<String>) {
        let mat = from_triplets(
            4,
            5,
            &[0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
            &[0, 1, 4, 0, 2, 1, 3, 4, 0, 1],
            &[10.0, 20.0, 5.0, 15.0, 25.0, 30.0, 10.0, 8.0, 3.0, 2.0],
        );
        let genes = vec![
            "BRCA1".into(),
            "TP53".into(),
            "EGFR".into(),
            "MYC".into(),
            "MT-CO1".into(),
        ];
        let cells = vec![
            "CELL_A".into(),
            "CELL_B".into(),
            "CELL_C".into(),
            "CELL_D".into(),
        ];
        (mat, genes, cells)
    }

    fn make_sc_test_matrix(
        n_cells: usize,
        n_high: usize,
        n_low: usize,
        seed: u64,
    ) -> (SpMat, Vec<String>, Vec<String>) {
        let n_genes = n_high + n_low;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for c in 0..n_cells {
            for g in 0..n_genes {
                let val = if g < n_high {
                    if rng.random::<f32>() < 0.9 {
                        rng.random::<f32>() * 500.0 + 10.0
                    } else {
                        0.0
                    }
                } else if rng.random::<f32>() < 0.9 {
                    50.0 + rng.random::<f32>() * 2.0
                } else {
                    0.0
                };
                if val > 0.0 {
                    rows.push(c);
                    cols.push(g);
                    vals.push(val);
                }
            }
        }

        let mat = from_triplets(n_cells, n_genes, &rows, &cols, &vals);
        let gene_names: Vec<String> = (0..n_genes).map(|g| format!("GENE{:05}", g)).collect();
        let cell_names: Vec<String> = (0..n_cells).map(|c| format!("CELL{:04}", c)).collect();
        (mat, gene_names, cell_names)
    }

    fn sequential_prep(
        mat: &SpMat,
        var_names: &[String],
        obs_names: &[String],
        min_genes: u32,
        max_pct_mt: f32,
        min_cells: usize,
        target_sum: f32,
        skip_normalize: bool,
    ) -> (QcMetrics, Vec<bool>, Vec<String>, Vec<String>, SpMat) {
        let metrics = compute_qc_metrics(mat, var_names, obs_names);
        let keep = filter_cells_fixed(&metrics, min_genes, max_pct_mt);
        let (filtered_mat, filtered_names) = apply_cell_filter(mat, obs_names, &keep);
        let (gene_filtered, kept_genes) = filter_genes(&filtered_mat, min_cells);
        let kept_var: Vec<String> = kept_genes.iter().map(|&i| var_names[i].clone()).collect();
        let normed = if skip_normalize {
            gene_filtered
        } else {
            normalize_log1p_sparse(&gene_filtered, target_sum)
        };
        (metrics, keep, filtered_names, kept_var, normed)
    }

    fn assert_qc_close(a: &QcMetrics, b: &QcMetrics) {
        assert_eq!(a.barcodes, b.barcodes);
        assert_eq!(a.n_genes_by_counts, b.n_genes_by_counts);
        assert_eq!(a.total_counts.len(), b.total_counts.len());
        for i in 0..a.total_counts.len() {
            assert!(
                (a.total_counts[i] - b.total_counts[i]).abs() < 1e-5,
                "total_counts[{}] fused={} sequential={}",
                i,
                a.total_counts[i],
                b.total_counts[i]
            );
            assert!(
                (a.pct_counts_mt[i] - b.pct_counts_mt[i]).abs() < 1e-5,
                "pct_mt[{}] fused={} sequential={}",
                i,
                a.pct_counts_mt[i],
                b.pct_counts_mt[i]
            );
        }
    }

    fn assert_sparse_close(a: &SpMat, b: &SpMat) {
        assert_eq!(a.rows(), b.rows());
        assert_eq!(a.cols(), b.cols());
        assert_eq!(a.nnz(), b.nnz());
        for i in 0..a.rows() {
            let ra = a.outer_view(i).unwrap();
            let rb = b.outer_view(i).unwrap();
            let va: Vec<(usize, f32)> = ra.iter().map(|(c, &v)| (c, v)).collect();
            let vb: Vec<(usize, f32)> = rb.iter().map(|(c, &v)| (c, v)).collect();
            assert_eq!(va.len(), vb.len(), "row {} nnz", i);
            for ((ca, xa), (cb, xb)) in va.iter().zip(vb.iter()) {
                assert_eq!(ca, cb, "row {} col mismatch", i);
                assert!(
                    (xa - xb).abs() < 1e-5,
                    "row {} col {}: fused={} sequential={}",
                    i,
                    ca,
                    xa,
                    xb
                );
            }
        }
    }

    #[test]
    fn fused_matches_sequential_on_qc_fixture() {
        let (mat, genes, cells) = toy_sc_matrix();
        let fused = qc_normalize_hvg_stats(&mat, &genes, &cells, 3, 100.0, 2, 1e4, false);
        let (metrics, keep, names, kept_var, normed) =
            sequential_prep(&mat, &genes, &cells, 3, 100.0, 2, 1e4, false);

        assert_qc_close(&fused.metrics, &metrics);
        assert_eq!(fused.keep_cells, keep);
        assert_eq!(fused.filtered_names, names);
        assert_eq!(fused.kept_var, kept_var);
        assert_sparse_close(&fused.mat, &normed);
    }

    #[test]
    fn fused_hvg_matches_sequential_seurat_fixture() {
        let (mat, genes, cells) = make_sc_test_matrix(200, 10, 90, 42);
        let fused = qc_normalize_hvg_stats(&mat, &genes, &cells, 1, 100.0, 3, 1e4, false);
        let (_m, _k, _n, kept_var, normed) =
            sequential_prep(&mat, &genes, &cells, 1, 100.0, 3, 1e4, false);

        assert_eq!(fused.kept_var, kept_var);
        let n_top = 15.min(kept_var.len());
        let hvg_fused = select_hvg_from_stats(&fused, n_top, "seurat").unwrap();
        let hvg_seq = hvg_sc::select_hvg_seurat(&normed, &kept_var, n_top, 20).unwrap();
        assert_eq!(hvg_fused.gene_names, hvg_seq.gene_names);
        assert_eq!(hvg_fused.gene_indices, hvg_seq.gene_indices);
        assert_eq!(hvg_fused.variances_norm.len(), hvg_seq.variances_norm.len());
        for (i, (a, b)) in hvg_fused
            .variances_norm
            .iter()
            .zip(hvg_seq.variances_norm.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-5,
                "disp_norm[{}] fused={} sequential={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn fused_hvg_matches_sequential_seurat_v3_fixture() {
        let (mat, genes, cells) = make_sc_test_matrix(200, 10, 90, 42);
        let fused = qc_normalize_hvg_stats(&mat, &genes, &cells, 1, 100.0, 3, 1e4, false);
        let (_m, _k, _n, kept_var, normed) =
            sequential_prep(&mat, &genes, &cells, 1, 100.0, 3, 1e4, false);

        let n_top = 15.min(kept_var.len());
        let hvg_fused = select_hvg_from_stats(&fused, n_top, "seurat_v3").unwrap();
        let hvg_seq = hvg_sc::select_hvg_sparse(&normed, &kept_var, n_top).unwrap();
        assert_eq!(hvg_fused.gene_names, hvg_seq.gene_names);
        assert_eq!(hvg_fused.gene_indices, hvg_seq.gene_indices);
    }

    #[test]
    fn fused_skip_normalize_matches_sequential() {
        let (mat, genes, cells) = make_sc_test_matrix(80, 8, 32, 7);
        let fused = qc_normalize_hvg_stats(&mat, &genes, &cells, 1, 100.0, 3, 1e4, true);
        let (_m, _k, _n, kept_var, raw) =
            sequential_prep(&mat, &genes, &cells, 1, 100.0, 3, 1e4, true);
        assert_eq!(fused.kept_var, kept_var);
        assert_sparse_close(&fused.mat, &raw);
    }
}
