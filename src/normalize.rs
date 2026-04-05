//! Library-size normalization and log1p transform for single-cell data.
//!
//! Equivalent to `sc.pp.normalize_total(adata, target_sum=1e4)` followed by
//! `sc.pp.log1p(adata)`. Preserves sparsity: log1p(0) = 0.
//!
//! Builds output CSR directly from input CSR (no TriMat sort overhead).

use crate::sparse::SpMat;
use sprs::CsMat;

/// Normalize each cell to `target_sum` total counts, then apply log1p.
///
/// For each cell (CSR row):
///   x_norm = log1p(x / total_counts * target_sum)
///
/// Since log1p(0) = 0, zero entries remain zero — sparsity is preserved.
/// Builds output CSR directly from input CSR (avoids O(nnz log nnz) TriMat sort).
pub fn normalize_log1p_sparse(mat: &SpMat, target_sum: f32) -> SpMat {
    let n_cells = mat.rows();
    let n_genes = mat.cols();

    // Pre-allocate output CSR arrays (same capacity as input — nnz is preserved)
    let mut indptr: Vec<usize> = Vec::with_capacity(n_cells + 1);
    let mut indices: Vec<usize> = Vec::with_capacity(mat.nnz());
    let mut data: Vec<f32> = Vec::with_capacity(mat.nnz());
    indptr.push(0);

    for i in 0..n_cells {
        let row = mat.outer_view(i).unwrap();

        // Compute cell total counts
        let cell_total: f32 = row.iter().map(|(_, &v)| v).sum();

        if cell_total > 0.0 {
            let scale = target_sum / cell_total;
            for (col, &val) in row.iter() {
                if val > 0.0 {
                    let normalized = (val * scale + 1.0).ln();
                    if normalized > 0.0 {
                        indices.push(col);
                        data.push(normalized);
                    }
                }
            }
        }
        indptr.push(indices.len());
    }

    // Safety: we build from sorted CSR input, so indices are already sorted per row
    CsMat::new((n_cells, n_genes), indptr, indices, data)
}

/// Normalize without log transform (just scale to target_sum per cell).
/// Builds output CSR directly (no TriMat overhead).
#[allow(dead_code)]
pub fn normalize_total_sparse(mat: &SpMat, target_sum: f32) -> SpMat {
    let n_cells = mat.rows();
    let n_genes = mat.cols();

    let mut indptr: Vec<usize> = Vec::with_capacity(n_cells + 1);
    let mut indices: Vec<usize> = Vec::with_capacity(mat.nnz());
    let mut data: Vec<f32> = Vec::with_capacity(mat.nnz());
    indptr.push(0);

    for i in 0..n_cells {
        let row = mat.outer_view(i).unwrap();
        let cell_total: f32 = row.iter().map(|(_, &v)| v).sum();

        if cell_total > 0.0 {
            let scale = target_sum / cell_total;
            for (col, &val) in row.iter() {
                if val > 0.0 {
                    indices.push(col);
                    data.push(val * scale);
                }
            }
        }
        indptr.push(indices.len());
    }

    CsMat::new((n_cells, n_genes), indptr, indices, data)
}

// =====================================================================
//  Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::from_triplets;

    #[test]
    fn test_normalize_preserves_sparsity() {
        // 2 cells x 3 genes
        let mat = from_triplets(
            2,
            3,
            &[0, 0, 1, 1],
            &[0, 2, 0, 1],
            &[100.0, 200.0, 300.0, 100.0],
        );

        let normed = normalize_log1p_sparse(&mat, 1e4);

        // Should still be sparse (same nnz or fewer)
        assert!(normed.nnz() <= mat.nnz());
        assert_eq!(normed.rows(), 2);
        assert_eq!(normed.cols(), 3);
    }

    #[test]
    fn test_normalize_values_correct() {
        // Single cell: [100, 200, 0] — total = 300
        // After normalize_total(target=1000): [333.33, 666.67, 0]
        // After log1p: [ln(334.33), ln(667.67), 0]
        let mat = from_triplets(1, 3, &[0, 0], &[0, 1], &[100.0, 200.0]);
        let normed = normalize_log1p_sparse(&mat, 1000.0);

        let row = normed.outer_view(0).unwrap();
        let val0 = row.get(0).copied().unwrap_or(0.0);
        let val1 = row.get(1).copied().unwrap_or(0.0);

        let expected0 = (100.0f32 / 300.0 * 1000.0 + 1.0).ln();
        let expected1 = (200.0f32 / 300.0 * 1000.0 + 1.0).ln();

        assert!(
            (val0 - expected0).abs() < 0.01,
            "val0={}, expected={}",
            val0,
            expected0
        );
        assert!(
            (val1 - expected1).abs() < 0.01,
            "val1={}, expected={}",
            val1,
            expected1
        );
    }

    #[test]
    fn test_normalize_total_only() {
        let mat = from_triplets(1, 2, &[0, 0], &[0, 1], &[100.0, 400.0]);
        let normed = normalize_total_sparse(&mat, 1e4);

        let row = normed.outer_view(0).unwrap();
        let val0 = row.get(0).copied().unwrap_or(0.0);
        let val1 = row.get(1).copied().unwrap_or(0.0);

        // 100/500 * 10000 = 2000, 400/500 * 10000 = 8000
        assert!((val0 - 2000.0).abs() < 0.1);
        assert!((val1 - 8000.0).abs() < 0.1);
    }
}
