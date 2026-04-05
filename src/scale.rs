//! Gene scaling: z-score normalization with clipping.
//!
//! Equivalent to `sc.pp.scale(adata, max_value=10)`.
//! For each gene (column), subtract the mean, divide by the standard deviation,
//! then clip to [-max_value, max_value].
//!
//! Uses **population** std (ddof=0) to match Scanpy's implementation,
//! which calls `np.std(axis=0)` (default ddof=0).
//!
//! The result is a dense matrix because centering fills zeros, destroying sparsity.

use ndarray::Array2;

use crate::sparse::SpMat;

/// Scale gene expression to zero mean and unit variance, with clipping.
///
/// Converts sparse -> dense (scaling fills zeros, destroying sparsity).
///
/// # Arguments
/// * `mat` - sparse CSR matrix (cells x genes), e.g. log-normalized HVG subset
/// * `max_value` - clip scaled values to [-max_value, max_value] (Scanpy default: 10)
///
/// # Returns
/// * `scaled` - dense cells x genes Array2<f32>
/// * `means` - per-gene population means (length n_genes)
/// * `stds` - per-gene population stds (length n_genes)
pub fn scale_sparse(mat: &SpMat, max_value: f32) -> (Array2<f32>, Vec<f32>, Vec<f32>) {
    let n_cells = mat.rows();
    let n_genes = mat.cols();

    // Step 1: Compute per-gene sum and sum-of-squares from sparse data.
    // Use population statistics (ddof=0) to match Scanpy.
    let mut sums = vec![0.0f64; n_genes];
    let mut sum_sq = vec![0.0f64; n_genes];

    for row_idx in 0..n_cells {
        let row = mat.outer_view(row_idx).unwrap();
        for (col_idx, &val) in row.iter() {
            let v = val as f64;
            sums[col_idx] += v;
            sum_sq[col_idx] += v * v;
        }
    }

    let n = n_cells as f64;
    let means: Vec<f32> = sums.iter().map(|&s| (s / n) as f32).collect();
    let stds: Vec<f32> = sums
        .iter()
        .zip(sum_sq.iter())
        .map(|(&s, &sq)| {
            let mean = s / n;
            // Population variance: E[x^2] - E[x]^2
            let var = (sq / n) - mean * mean;
            let std = var.max(0.0).sqrt();
            // Floor at 1e-12 to avoid division by zero
            std.max(1e-12) as f32
        })
        .collect();

    // Step 2: Build dense scaled matrix.
    // Most entries are zero in the sparse matrix, so we first fill the entire
    // matrix with the z-score of zero for each gene, then overwrite non-zeros.
    let mut scaled = Array2::<f32>::zeros((n_cells, n_genes));

    // Pre-compute z-score and clipped value of zero for each gene
    for g in 0..n_genes {
        let z_zero = -means[g] / stds[g];
        let z_clipped = z_zero.clamp(-max_value, max_value);
        // Fill column with this value (the z-score of all the implicit zeros)
        for c in 0..n_cells {
            scaled[[c, g]] = z_clipped;
        }
    }

    // Overwrite non-zero entries with their actual z-scores
    for c in 0..n_cells {
        let row = mat.outer_view(c).unwrap();
        for (g, &val) in row.iter() {
            let z = (val - means[g]) / stds[g];
            scaled[[c, g]] = z.clamp(-max_value, max_value);
        }
    }

    (scaled, means, stds)
}

// =====================================================================
//  Unit tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse;

    /// Same toy matrix as sparse.rs:
    /// Cell 0: [1, 0, 2, 0]
    /// Cell 1: [0, 3, 0, 4]
    /// Cell 2: [5, 0, 6, 0]
    fn toy_matrix() -> SpMat {
        sparse::from_triplets(
            3,
            4,
            &[0, 0, 1, 1, 2, 2],
            &[0, 2, 1, 3, 0, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
    }

    #[test]
    fn test_scale_means_near_zero() {
        let mat = toy_matrix();
        let (scaled, _means, _stds) = scale_sparse(&mat, 10.0);

        // After scaling, each gene column should have mean approximately 0
        let n_cells = scaled.nrows();
        for g in 0..scaled.ncols() {
            let col_mean: f32 = (0..n_cells).map(|c| scaled[[c, g]]).sum::<f32>() / n_cells as f32;
            assert!(
                col_mean.abs() < 1e-4,
                "gene {} mean = {} (expected ~0)",
                g,
                col_mean
            );
        }
    }

    #[test]
    fn test_scale_stds_near_one() {
        let mat = toy_matrix();
        let (scaled, _means, _stds) = scale_sparse(&mat, 100.0); // large max_value to avoid clipping

        let n_cells = scaled.nrows();
        for g in 0..scaled.ncols() {
            let col_mean: f32 = (0..n_cells).map(|c| scaled[[c, g]]).sum::<f32>() / n_cells as f32;
            let col_var: f32 = (0..n_cells)
                .map(|c| {
                    let d = scaled[[c, g]] - col_mean;
                    d * d
                })
                .sum::<f32>()
                / n_cells as f32; // population var
            let col_std = col_var.sqrt();
            assert!(
                (col_std - 1.0).abs() < 0.05,
                "gene {} std = {} (expected ~1.0)",
                g,
                col_std
            );
        }
    }

    #[test]
    fn test_scale_clipping() {
        let mat = toy_matrix();
        let max_val = 0.5; // very aggressive clipping
        let (scaled, _means, _stds) = scale_sparse(&mat, max_val);

        for c in 0..scaled.nrows() {
            for g in 0..scaled.ncols() {
                assert!(
                    scaled[[c, g]] >= -max_val && scaled[[c, g]] <= max_val,
                    "value {} at [{}, {}] exceeds clip bounds",
                    scaled[[c, g]],
                    c,
                    g
                );
            }
        }
    }

    #[test]
    fn test_scale_zero_variance_gene() {
        // Gene with all identical values should produce all zeros after scaling
        // (std is floored at 1e-12, so z-scores are huge, but they should all be
        // the same value, and mean-centering makes them ~0 before division)
        let mat = sparse::from_triplets(
            3,
            2,
            &[0, 1, 2, 0, 1, 2],
            &[0, 0, 0, 1, 1, 1],
            &[5.0, 5.0, 5.0, 1.0, 2.0, 3.0],
        );

        let (scaled, _means, _stds) = scale_sparse(&mat, 10.0);

        // Gene 0: all values are 5.0, mean=5.0, z = (5-5)/eps ≈ 0
        for c in 0..3 {
            assert!(
                scaled[[c, 0]].abs() < 1e-3,
                "zero-variance gene: cell {} = {} (expected ~0)",
                c,
                scaled[[c, 0]]
            );
        }

        // Gene 1: values [1,2,3], should have non-zero spread
        let col1_vals: Vec<f32> = (0..3).map(|c| scaled[[c, 1]]).collect();
        let range = col1_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - col1_vals.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            range > 0.1,
            "gene 1 should have spread, got range={}",
            range
        );
    }

    #[test]
    fn test_scale_shape() {
        let mat = toy_matrix();
        let (scaled, means, stds) = scale_sparse(&mat, 10.0);

        assert_eq!(scaled.nrows(), 3);
        assert_eq!(scaled.ncols(), 4);
        assert_eq!(means.len(), 4);
        assert_eq!(stds.len(), 4);
    }

    #[test]
    fn test_scale_population_std() {
        // Verify we use population std (ddof=0), not sample std (ddof=1).
        // Gene 0 in toy_matrix: values [1, 0, 5]
        //   mean = 2.0
        //   pop_var = ((1-2)^2 + (0-2)^2 + (5-2)^2) / 3 = (1+4+9)/3 = 14/3
        //   pop_std = sqrt(14/3) ≈ 2.16025
        //   sample_std = sqrt(14/2) ≈ 2.64575  (different!)
        let mat = toy_matrix();
        let (_scaled, _means, stds) = scale_sparse(&mat, 10.0);

        let expected_pop_std = (14.0_f32 / 3.0).sqrt(); // ≈ 2.16025
        assert!(
            (stds[0] - expected_pop_std).abs() < 0.01,
            "gene 0 std = {} (expected pop_std {})",
            stds[0],
            expected_pop_std
        );
    }
}
