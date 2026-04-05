//! Core sparse matrix types and operations for single-cell RNA-seq.
//!
//! CSR format for cells x genes matrices. All count data stored as f32
//! (counts are integers, f32 is sufficient and halves memory vs f64).
//! Provides implicit-centering sparse-dense matmul for PCA without
//! materializing the centered matrix.

use ndarray::{Array1, Array2, Axis};
use sprs::{CsMat, TriMat};

/// Sparse matrix type: cells (rows) x genes (columns), CSR layout, f32 values.
pub type SpMat = CsMat<f32>;

/// Build a CSR sparse matrix from COO triplets.
///
/// # Arguments
/// * `n_rows` — number of cells (rows)
/// * `n_cols` — number of genes (columns)
/// * `rows` — row indices for each non-zero entry
/// * `cols` — column indices for each non-zero entry
/// * `vals` — values for each non-zero entry
pub fn from_triplets(
    n_rows: usize,
    n_cols: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[f32],
) -> SpMat {
    let mut tri = TriMat::new((n_rows, n_cols));
    for i in 0..rows.len() {
        if vals[i] != 0.0 {
            tri.add_triplet(rows[i], cols[i], vals[i]);
        }
    }
    tri.to_csr()
}

/// Compute per-gene (column) mean and variance in a single pass over non-zero entries.
///
/// Accounts for implicit zeros: for a gene with `nnz` stored values out of
/// `n_cells` total, the remaining `n_cells - nnz` values are zero.
/// Uses Welford-like two-pass (sum then variance) for numerical stability.
pub fn sparse_gene_stats(mat: &SpMat, n_cells: usize) -> (Vec<f32>, Vec<f32>) {
    let n_genes = mat.cols();
    let mut sums = vec![0.0f64; n_genes];
    let mut sum_sq = vec![0.0f64; n_genes];

    // CSR: iterate rows (cells), accumulate per-column (gene) stats
    for row_idx in 0..mat.rows() {
        let row = mat.outer_view(row_idx).unwrap();
        for (col_idx, &val) in row.iter() {
            let v = val as f64;
            sums[col_idx] += v;
            sum_sq[col_idx] += v * v;
        }
    }

    let n = n_cells as f64;
    let means: Vec<f32> = sums.iter().map(|&s| (s / n) as f32).collect();
    let variances: Vec<f32> = sums
        .iter()
        .zip(sum_sq.iter())
        .map(|(&s, &sq)| {
            let mean = s / n;
            // var = E[x^2] - E[x]^2, with Bessel correction
            let var = (sq - n * mean * mean) / (n - 1.0);
            var.max(0.0) as f32
        })
        .collect();

    (means, variances)
}

/// Per-cell total counts (library sizes) — sum of each CSR row.
#[allow(dead_code)]
pub fn cell_total_counts(mat: &SpMat) -> Vec<f32> {
    (0..mat.rows())
        .map(|i| {
            let row = mat.outer_view(i).unwrap();
            row.iter().map(|(_, &v)| v).sum()
        })
        .collect()
}

/// Per-cell number of genes detected (count > 0).
#[allow(dead_code)]
pub fn cell_n_genes(mat: &SpMat) -> Vec<u32> {
    (0..mat.rows())
        .map(|i| {
            let row = mat.outer_view(i).unwrap();
            row.iter().filter(|(_, &v)| v > 0.0).count() as u32
        })
        .collect()
}

/// Y = (A - mu) * Omega — sparse-dense matmul with implicit centering.
///
/// Computes `A * Omega` (sparse x dense) then subtracts the rank-1 correction
/// `ones * (mu^T * Omega)` without materializing the centered matrix.
///
/// # Arguments
/// * `mat` — sparse CSR matrix, cells x genes
/// * `gene_means` — per-gene mean vector, length n_genes
/// * `omega` — dense Gaussian random matrix, n_genes x k
///
/// # Returns
/// Dense matrix of shape cells x k
#[allow(dead_code)]
pub fn spmm_centered(mat: &SpMat, gene_means: &[f32], omega: &Array2<f32>) -> Array2<f32> {
    let n_cells = mat.rows();
    let k = omega.ncols();

    // Step 1: correction = mu^T * Omega  (1 x k row vector)
    // Each element: correction[j] = sum_g(gene_means[g] * omega[g, j])
    let correction: Vec<f32> = (0..k)
        .map(|j| {
            gene_means
                .iter()
                .zip(omega.column(j).iter())
                .map(|(&m, &o)| m * o)
                .sum()
        })
        .collect();

    // Step 2: Y = A * Omega (sparse-dense matmul, row by row)
    let mut y = Array2::<f32>::zeros((n_cells, k));

    for i in 0..n_cells {
        let row = mat.outer_view(i).unwrap();
        for (g, &val) in row.iter() {
            for j in 0..k {
                y[[i, j]] += val * omega[[g, j]];
            }
        }
        // Step 3: subtract correction (same for every cell)
        for j in 0..k {
            y[[i, j]] -= correction[j];
        }
    }

    y
}

#[allow(dead_code)]
/// Z = (A - mu)^T * Q — transposed sparse-dense matmul with implicit centering.
///
/// Computes `A^T * Q` then subtracts `mu * (1^T * Q)` (column sums of Q, scaled by mu).
///
/// # Arguments
/// * `mat` — sparse CSR, cells x genes
/// * `gene_means` — per-gene means, length n_genes
/// * `q` — dense orthonormal matrix, cells x k
///
/// # Returns
/// Dense matrix of shape genes x k
pub fn spmm_at_centered(mat: &SpMat, gene_means: &[f32], q: &Array2<f32>) -> Array2<f32> {
    let n_genes = mat.cols();
    let k = q.ncols();
    let n_cells = mat.rows();

    // Step 1: A^T * Q (accumulate from CSR rows)
    let mut z = Array2::<f32>::zeros((n_genes, k));

    for i in 0..n_cells {
        let row = mat.outer_view(i).unwrap();
        for (g, &val) in row.iter() {
            for j in 0..k {
                z[[g, j]] += val * q[[i, j]];
            }
        }
    }

    // Step 2: col_sums = 1^T * Q = sum of each column of Q
    let col_sums: Array1<f32> = q.sum_axis(Axis(0));

    // Step 3: subtract mu * col_sums^T
    for g in 0..n_genes {
        for j in 0..k {
            z[[g, j]] -= gene_means[g] * col_sums[j];
        }
    }

    z
}

/// Subset a sparse matrix to selected gene (column) indices.
///
/// Returns a new CSR matrix with only the selected columns, in the order given.
pub fn subset_genes(mat: &SpMat, gene_indices: &[usize]) -> SpMat {
    let n_cells = mat.rows();
    let n_new_genes = gene_indices.len();

    // Build reverse map: original gene index -> new gene index
    let mut col_map = vec![None; mat.cols()];
    for (new_idx, &orig_idx) in gene_indices.iter().enumerate() {
        col_map[orig_idx] = Some(new_idx);
    }

    let mut tri = TriMat::new((n_cells, n_new_genes));
    for i in 0..n_cells {
        let row = mat.outer_view(i).unwrap();
        for (col, &val) in row.iter() {
            if let Some(new_col) = col_map[col] {
                tri.add_triplet(i, new_col, val);
            }
        }
    }

    tri.to_csr()
}

/// Filter sparse matrix to keep only selected rows (cells).
/// `keep` is a boolean mask of length n_cells.
pub fn filter_rows(mat: &SpMat, keep: &[bool]) -> SpMat {
    let n_new_rows = keep.iter().filter(|&&b| b).count();
    let n_cols = mat.cols();

    let mut tri = TriMat::new((n_new_rows, n_cols));
    let mut new_row = 0;
    for (old_row, &kept) in keep.iter().enumerate() {
        if kept {
            let row = mat.outer_view(old_row).unwrap();
            for (col, &val) in row.iter() {
                tri.add_triplet(new_row, col, val);
            }
            new_row += 1;
        }
    }

    tri.to_csr()
}

/// Filter sparse matrix to keep only selected columns (genes).
/// `keep` is a boolean mask of length n_genes.
pub fn filter_cols(mat: &SpMat, keep: &[bool]) -> (SpMat, Vec<usize>) {
    let kept_indices: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    let sub = subset_genes(mat, &kept_indices);
    (sub, kept_indices)
}

// =====================================================================
//  Unit tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_matrix() -> SpMat {
        // 3 cells x 4 genes
        // Cell 0: [1, 0, 2, 0]
        // Cell 1: [0, 3, 0, 4]
        // Cell 2: [5, 0, 6, 0]
        from_triplets(
            3,
            4,
            &[0, 0, 1, 1, 2, 2],
            &[0, 2, 1, 3, 0, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
    }

    #[test]
    fn test_from_triplets_shape() {
        let mat = toy_matrix();
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 4);
        assert_eq!(mat.nnz(), 6);
    }

    #[test]
    fn test_sparse_gene_stats() {
        let mat = toy_matrix();
        let (means, vars) = sparse_gene_stats(&mat, 3);

        // Gene 0: values [1, 0, 5], mean = 2.0
        assert!((means[0] - 2.0).abs() < 1e-5, "gene 0 mean = {}", means[0]);
        // Gene 1: values [0, 3, 0], mean = 1.0
        assert!((means[1] - 1.0).abs() < 1e-5, "gene 1 mean = {}", means[1]);
        // Gene 2: values [2, 0, 6], mean = 8/3
        assert!(
            (means[2] - 8.0 / 3.0).abs() < 0.01,
            "gene 2 mean = {}",
            means[2]
        );

        // Variance of gene 0: var([1,0,5]) = ((1-2)^2 + (0-2)^2 + (5-2)^2) / 2 = 7.0
        assert!((vars[0] - 7.0).abs() < 0.1, "gene 0 var = {}", vars[0]);
    }

    #[test]
    fn test_cell_total_counts() {
        let mat = toy_matrix();
        let counts = cell_total_counts(&mat);
        assert!((counts[0] - 3.0).abs() < 1e-5);
        assert!((counts[1] - 7.0).abs() < 1e-5);
        assert!((counts[2] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_cell_n_genes() {
        let mat = toy_matrix();
        let ng = cell_n_genes(&mat);
        assert_eq!(ng[0], 2);
        assert_eq!(ng[1], 2);
        assert_eq!(ng[2], 2);
    }

    #[test]
    fn test_spmm_centered_vs_dense() {
        // Build a small dense matrix, center it, multiply by omega.
        // Compare with spmm_centered on the sparse version.
        let mat = toy_matrix();
        let (means, _) = sparse_gene_stats(&mat, 3);

        let omega =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]).unwrap();

        let y_sparse = spmm_centered(&mat, &means, &omega);

        // Dense reference: explicit centering
        let dense = vec![
            1.0, 0.0, 2.0, 0.0, // cell 0
            0.0, 3.0, 0.0, 4.0, // cell 1
            5.0, 0.0, 6.0, 0.0, // cell 2
        ];
        let n_cells = 3;
        let n_genes = 4;
        let k = 2;

        let mut y_dense = Array2::<f32>::zeros((n_cells, k));
        for i in 0..n_cells {
            for j in 0..k {
                let mut sum = 0.0f32;
                for g in 0..n_genes {
                    sum += (dense[i * n_genes + g] - means[g]) * omega[[g, j]];
                }
                y_dense[[i, j]] = sum;
            }
        }

        for i in 0..n_cells {
            for j in 0..k {
                assert!(
                    (y_sparse[[i, j]] - y_dense[[i, j]]).abs() < 1e-4,
                    "mismatch at [{}, {}]: sparse={}, dense={}",
                    i,
                    j,
                    y_sparse[[i, j]],
                    y_dense[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_subset_genes() {
        let mat = toy_matrix();
        let sub = subset_genes(&mat, &[0, 2]); // keep genes 0 and 2

        assert_eq!(sub.rows(), 3);
        assert_eq!(sub.cols(), 2);

        // Cell 0: [1, 2], Cell 1: [0, 0], Cell 2: [5, 6]
        let row0 = sub.outer_view(0).unwrap();
        let row0_dense: Vec<f32> = (0..2)
            .map(|c| row0.get(c).copied().unwrap_or(0.0))
            .collect();
        assert_eq!(row0_dense, vec![1.0, 2.0]);
    }

    #[test]
    fn test_filter_rows() {
        let mat = toy_matrix();
        let filtered = filter_rows(&mat, &[true, false, true]);
        assert_eq!(filtered.rows(), 2);
        assert_eq!(filtered.cols(), 4);
    }
}
