//! Quality control metrics and cell filtering for single-cell RNA-seq.
//!
//! Computes per-cell QC metrics (n_genes, total_counts, pct_mito) and
//! applies threshold-based or MAD-based filtering.

use crate::sparse::{self, SpMat};
use crate::stats_sc;
use anyhow::Result;

/// Per-cell quality control metrics.
pub struct QcMetrics {
    pub barcodes: Vec<String>,
    pub n_genes_by_counts: Vec<u32>,
    pub total_counts: Vec<f32>,
    pub pct_counts_mt: Vec<f32>,
}

/// Compute QC metrics from a sparse count matrix.
///
/// Mitochondrial genes are identified by prefix "MT-" (human) or "mt-" (mouse).
/// Each cell's CSR row is iterated once: O(nnz) total.
pub fn compute_qc_metrics(mat: &SpMat, var_names: &[String], obs_names: &[String]) -> QcMetrics {
    let n_cells = mat.rows();

    // Build mito gene mask
    let is_mito: Vec<bool> = var_names
        .iter()
        .map(|g| g.starts_with("MT-") || g.starts_with("mt-"))
        .collect();

    let mut n_genes_by_counts = Vec::with_capacity(n_cells);
    let mut total_counts = Vec::with_capacity(n_cells);
    let mut pct_counts_mt = Vec::with_capacity(n_cells);

    for i in 0..n_cells {
        let row = mat.outer_view(i).unwrap();
        let mut n_genes = 0u32;
        let mut total = 0.0f32;
        let mut mito_total = 0.0f32;

        for (col, &val) in row.iter() {
            if val > 0.0 {
                n_genes += 1;
                total += val;
                if is_mito[col] {
                    mito_total += val;
                }
            }
        }

        n_genes_by_counts.push(n_genes);
        total_counts.push(total);
        pct_counts_mt.push(if total > 0.0 {
            (mito_total / total) * 100.0
        } else {
            0.0
        });
    }

    QcMetrics {
        barcodes: obs_names.to_vec(),
        n_genes_by_counts,
        total_counts,
        pct_counts_mt,
    }
}

/// Filter cells based on fixed thresholds.
///
/// Returns a boolean mask: `true` = keep cell.
pub fn filter_cells_fixed(metrics: &QcMetrics, min_genes: u32, max_pct_mt: f32) -> Vec<bool> {
    metrics
        .n_genes_by_counts
        .iter()
        .zip(metrics.pct_counts_mt.iter())
        .map(|(&ng, &pmt)| ng >= min_genes && pmt <= max_pct_mt)
        .collect()
}

/// MAD-based threshold: threshold = median +/- n_mads * MAD.
///
/// `upper`: if true, returns upper threshold (for max_pct_mt).
/// `lower`: if true, returns lower threshold (for min_genes, min_counts).
#[allow(dead_code)]
pub fn mad_threshold_upper(values: &[f32], n_mads: f32) -> f32 {
    let med = stats_sc::median(values);
    let mad_val = stats_sc::mad(values);
    med + n_mads * 1.4826 * mad_val // 1.4826 = consistency constant for normal
}

#[allow(dead_code)]
pub fn mad_threshold_lower(values: &[f32], n_mads: f32) -> f32 {
    let med = stats_sc::median(values);
    let mad_val = stats_sc::mad(values);
    med - n_mads * 1.4826 * mad_val
}

/// Apply a boolean mask to filter the sparse matrix and barcodes.
pub fn apply_cell_filter(mat: &SpMat, obs_names: &[String], keep: &[bool]) -> (SpMat, Vec<String>) {
    let filtered_mat = sparse::filter_rows(mat, keep);
    let filtered_names: Vec<String> = obs_names
        .iter()
        .zip(keep.iter())
        .filter(|(_, &k)| k)
        .map(|(n, _)| n.clone())
        .collect();
    (filtered_mat, filtered_names)
}

/// Filter genes: keep only genes expressed in at least `min_cells` cells.
pub fn filter_genes(mat: &SpMat, min_cells: usize) -> (SpMat, Vec<usize>) {
    let n_genes = mat.cols();
    let mut gene_cell_counts = vec![0usize; n_genes];

    for i in 0..mat.rows() {
        let row = mat.outer_view(i).unwrap();
        for (col, &val) in row.iter() {
            if val > 0.0 {
                gene_cell_counts[col] += 1;
            }
        }
    }

    let keep: Vec<bool> = gene_cell_counts.iter().map(|&c| c >= min_cells).collect();
    sparse::filter_cols(mat, &keep)
}

/// Write QC metrics to CSV (matching Scanpy reference format).
pub fn write_qc_csv(metrics: &QcMetrics, path: &std::path::Path) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "barcode",
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
    ])?;

    for i in 0..metrics.barcodes.len() {
        wtr.write_record([
            &metrics.barcodes[i],
            &metrics.n_genes_by_counts[i].to_string(),
            &format!("{:.1}", metrics.total_counts[i]),
            &format!("{:.7}", metrics.pct_counts_mt[i]),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

// =====================================================================
//  Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::from_triplets;

    fn toy_sc_matrix() -> (SpMat, Vec<String>, Vec<String>) {
        // 4 cells x 5 genes, gene 4 is "MT-CO1" (mitochondrial)
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

    #[test]
    fn test_qc_metrics() {
        let (mat, genes, cells) = toy_sc_matrix();
        let qc = compute_qc_metrics(&mat, &genes, &cells);

        // Cell A: genes [BRCA1, TP53, MT-CO1], total=35, mito=5, pct=14.28%
        assert_eq!(qc.n_genes_by_counts[0], 3);
        assert!((qc.total_counts[0] - 35.0).abs() < 0.1);
        assert!((qc.pct_counts_mt[0] - 5.0 / 35.0 * 100.0).abs() < 0.1);

        // Cell D: genes [BRCA1, TP53], total=5, mito=0, pct=0%
        assert_eq!(qc.n_genes_by_counts[3], 2);
        assert!((qc.pct_counts_mt[3]).abs() < 0.01);
    }

    #[test]
    fn test_filter_cells_fixed() {
        let (mat, genes, cells) = toy_sc_matrix();
        let qc = compute_qc_metrics(&mat, &genes, &cells);

        // min_genes=3 keeps cells with >= 3 genes detected
        let keep = filter_cells_fixed(&qc, 3, 100.0);
        assert_eq!(keep, vec![true, false, true, false]);
    }

    #[test]
    fn test_apply_cell_filter() {
        let (mat, genes, cells) = toy_sc_matrix();
        let keep = vec![true, false, true, false];
        let (filtered, filtered_names) = apply_cell_filter(&mat, &cells, &keep);

        assert_eq!(filtered.rows(), 2);
        assert_eq!(filtered_names.len(), 2);
        assert_eq!(filtered_names[0], "CELL_A");
        assert_eq!(filtered_names[1], "CELL_C");
    }

    #[test]
    fn test_filter_genes() {
        let (mat, _, _) = toy_sc_matrix();
        // Gene 3 (MYC) only expressed in 1 cell → should be removed with min_cells=2
        let (filtered, kept) = filter_genes(&mat, 2);
        assert!(filtered.cols() < mat.cols());
        // MYC (index 3) should not be in kept
        assert!(!kept.contains(&3));
    }
}
