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

/// Pass 2 of the fused prep: subset kept cells/genes, optionally
/// normalize+log1p, and accumulate per-gene mean/variance.
///
/// One scan over kept-cell nonzeros. Cell totals use only kept genes
/// (same biology as `filter_genes` then `normalize_log1p_sparse`).
/// Variance uses Bessel correction, matching `sparse_gene_stats`.
pub fn normalize_log1p_subset_with_stats(
    mat: &SpMat,
    keep_cells: &[bool],
    keep_genes: &[bool],
    target_sum: f32,
    skip_normalize: bool,
) -> (SpMat, Vec<f32>, Vec<f32>) {
    let n_new_cells = keep_cells.iter().filter(|&&b| b).count();
    let n_new_genes = keep_genes.iter().filter(|&&b| b).count();

    let mut col_map = vec![None; mat.cols()];
    let mut new_idx = 0usize;
    for (orig, &kept) in keep_genes.iter().enumerate() {
        if kept {
            col_map[orig] = Some(new_idx);
            new_idx += 1;
        }
    }

    let mut indptr: Vec<usize> = Vec::with_capacity(n_new_cells + 1);
    let mut indices: Vec<usize> = Vec::with_capacity(mat.nnz());
    let mut data: Vec<f32> = Vec::with_capacity(mat.nnz());
    indptr.push(0);

    let mut sums = vec![0.0f64; n_new_genes];
    let mut sum_sq = vec![0.0f64; n_new_genes];

    for i in 0..mat.rows() {
        if !keep_cells[i] {
            continue;
        }
        let row = mat.outer_view(i).unwrap();

        if skip_normalize {
            for (col, &val) in row.iter() {
                if val > 0.0 {
                    if let Some(new_col) = col_map[col] {
                        indices.push(new_col);
                        data.push(val);
                        let v = val as f64;
                        sums[new_col] += v;
                        sum_sq[new_col] += v * v;
                    }
                }
            }
        } else {
            let mut cell_total = 0.0f32;
            for (col, &val) in row.iter() {
                if val > 0.0 && keep_genes[col] {
                    cell_total += val;
                }
            }
            if cell_total > 0.0 {
                let scale = target_sum / cell_total;
                for (col, &val) in row.iter() {
                    if val > 0.0 {
                        if let Some(new_col) = col_map[col] {
                            let normalized = (val * scale + 1.0).ln();
                            if normalized > 0.0 {
                                indices.push(new_col);
                                data.push(normalized);
                                let v = normalized as f64;
                                sums[new_col] += v;
                                sum_sq[new_col] += v * v;
                            }
                        }
                    }
                }
            }
        }
        indptr.push(indices.len());
    }

    let n = n_new_cells as f64;
    let means: Vec<f32> = if n > 0.0 {
        sums.iter().map(|&s| (s / n) as f32).collect()
    } else {
        vec![0.0; n_new_genes]
    };
    let variances: Vec<f32> = if n > 1.0 {
        sums.iter()
            .zip(sum_sq.iter())
            .map(|(&s, &sq)| {
                let mean = s / n;
                let var = (sq - n * mean * mean) / (n - 1.0);
                var.max(0.0) as f32
            })
            .collect()
    } else {
        vec![0.0; n_new_genes]
    };

    let out = CsMat::new((n_new_cells, n_new_genes), indptr, indices, data);
    (out, means, variances)
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

    #[test]
    fn test_subset_with_stats_matches_normalize() {
        let mat = from_triplets(
            2,
            3,
            &[0, 0, 1, 1],
            &[0, 2, 0, 1],
            &[100.0, 200.0, 300.0, 100.0],
        );
        let keep_cells = vec![true, true];
        let keep_genes = vec![true, true, true];
        let (fused, means, vars) =
            normalize_log1p_subset_with_stats(&mat, &keep_cells, &keep_genes, 1e4, false);
        let seq = normalize_log1p_sparse(&mat, 1e4);
        let (seq_means, seq_vars) = crate::sparse::sparse_gene_stats(&seq, seq.rows());

        assert_eq!(fused.nnz(), seq.nnz());
        for i in 0..2 {
            let ra = fused.outer_view(i).unwrap();
            let rb = seq.outer_view(i).unwrap();
            for ((c1, &v1), (c2, &v2)) in ra.iter().zip(rb.iter()) {
                assert_eq!(c1, c2);
                assert!((v1 - v2).abs() < 1e-6);
            }
        }
        for i in 0..3 {
            assert!((means[i] - seq_means[i]).abs() < 1e-6);
            assert!((vars[i] - seq_vars[i]).abs() < 1e-5);
        }
    }
}
