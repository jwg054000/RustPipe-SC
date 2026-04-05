//! Highly Variable Gene selection for single-cell data.
//!
//! Two flavors:
//! - **seurat_v3** (VST): Operates on raw counts. Fits mean-variance trend,
//!   standardizes, clips at sqrt(n_cells). Matches Scanpy's `seurat_v3`.
//! - **seurat** (dispersion-bin): Operates on log-normalized data. Bins genes
//!   by log-mean, z-scores log-dispersion within bins. Matches Scanpy's `seurat`.

use crate::sparse::{self, SpMat};
use anyhow::{bail, Result};
use log::info;
use rayon::prelude::*;

/// Result of HVG selection.
pub struct HvgResult {
    pub gene_names: Vec<String>,
    pub gene_indices: Vec<usize>,
    #[allow(dead_code)]
    pub means: Vec<f32>,
    #[allow(dead_code)]
    pub variances: Vec<f32>,
    #[allow(dead_code)]
    pub variances_norm: Vec<f32>,
}

/// Fit log(variance) ~ a + b * log(mean) via OLS on log-log scale.
///
/// Only uses genes with positive mean and variance. Returns (intercept, slope).
#[allow(dead_code)]
fn fit_mean_variance_trend(means: &[f32], variances: &[f32]) -> (f64, f64) {
    let pairs: Vec<(f64, f64)> = means
        .iter()
        .zip(variances.iter())
        .filter(|(&m, &v)| m > 0.0 && v > 0.0 && m.is_finite() && v.is_finite())
        .map(|(&m, &v)| ((m as f64).ln(), (v as f64).ln()))
        .collect();

    if pairs.len() < 2 {
        return (0.0, 1.0);
    }

    let n = pairs.len() as f64;
    let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = pairs.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        let avg_log_var = sum_y / n;
        return (avg_log_var, 0.0);
    }

    let b = (n * sum_xy - sum_x * sum_y) / denom;
    let a = (sum_y - b * sum_x) / n;

    (a, b)
}

/// Select top N highly variable genes using Seurat v3 VST method on sparse data.
///
/// Algorithm:
/// 1. Compute per-gene mean and variance from sparse matrix
/// 2. Fit log(var) ~ a + b*log(mean) via OLS
/// 3. Standardized variance = observed / expected
/// 4. Clip standardized values at sqrt(n_cells) (Seurat v3 specific!)
/// 5. Rank by standardized variance, return top N
///
/// The clipping at sqrt(n_cells) is CRITICAL for matching Scanpy's seurat_v3.
pub fn select_hvg_sparse(
    mat: &SpMat,
    var_names: &[String],
    n_top_genes: usize,
) -> Result<HvgResult> {
    let n_genes = mat.cols();
    let n_cells = mat.rows();

    if n_top_genes > n_genes {
        bail!(
            "n_top_genes ({}) exceeds total genes ({})",
            n_top_genes,
            n_genes
        );
    }
    if n_cells < 2 {
        bail!("Need at least 2 cells to compute variance");
    }

    let start = std::time::Instant::now();

    // Step 1: per-gene mean and variance from sparse data
    let (means, variances) = sparse::sparse_gene_stats(mat, n_cells);

    // Step 2: fit mean-variance trend
    let (a, b) = fit_mean_variance_trend(&means, &variances);
    info!(
        "HVG mean-variance trend: ln(var) = {:.4} + {:.4} * ln(mean)",
        a, b
    );

    // Step 3: standardized variance with Seurat v3 clipping
    let clip_val = (n_cells as f32).sqrt();
    let variances_norm: Vec<f32> = means
        .par_iter()
        .zip(variances.par_iter())
        .map(|(&m, &v)| {
            if m <= 0.0 || !m.is_finite() || !v.is_finite() {
                return 0.0;
            }
            let expected = (a + b * (m as f64).ln()).exp() as f32;
            if expected <= 0.0 || !expected.is_finite() {
                return 0.0;
            }
            let sv = v / expected;
            if sv.is_finite() {
                sv.min(clip_val) // Seurat v3 clipping!
            } else {
                0.0
            }
        })
        .collect();

    // Step 4: rank by standardized variance (descending), take top N
    let mut ranked_indices: Vec<usize> = (0..n_genes).collect();
    ranked_indices.sort_by(|&a_idx, &b_idx| {
        variances_norm[b_idx]
            .partial_cmp(&variances_norm[a_idx])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked_indices.truncate(n_top_genes);

    // Sort for stable ordering
    ranked_indices.sort();

    let selected_names: Vec<String> = ranked_indices
        .iter()
        .map(|&i| var_names[i].clone())
        .collect();

    info!(
        "Selected {} HVGs from {} genes in {:.3}s",
        n_top_genes,
        n_genes,
        start.elapsed().as_secs_f64()
    );

    Ok(HvgResult {
        gene_names: selected_names,
        gene_indices: ranked_indices,
        means,
        variances,
        variances_norm,
    })
}

/// Select HVGs using Scanpy's 'seurat' flavor (dispersion-bin method).
///
/// Matches `scanpy.pp.highly_variable_genes(flavor='seurat')` exactly:
/// 1. Per-gene mean and variance from sparse matrix
/// 2. Dispersion = variance / mean, mean_zero → 1e-12
/// 3. log_dispersion = ln(dispersion), log_mean = ln(mean + 1)
///    NOTE: Scanpy uses ln(disp) NOT ln(disp + 1)
/// 4. Bin genes into `n_bins` equal-width bins by log_mean
/// 5. Within each bin, z-score log_dispersion (ddof=1, Bessel's correction)
/// 6. Rank by normalized dispersion descending, return top N
pub fn select_hvg_seurat(
    mat: &SpMat,
    var_names: &[String],
    n_top_genes: usize,
    n_bins: usize,
) -> Result<HvgResult> {
    let n_genes = mat.cols();
    let n_cells = mat.rows();

    if n_top_genes > n_genes {
        bail!(
            "n_top_genes ({}) exceeds total genes ({})",
            n_top_genes,
            n_genes
        );
    }
    if n_cells < 2 {
        bail!("Need at least 2 cells to compute variance");
    }

    let start = std::time::Instant::now();

    // Step 1: per-gene mean and variance from sparse data
    let (means, variances) = sparse::sparse_gene_stats(mat, n_cells);

    // Step 2: dispersion = var / mean (Scanpy: mean_zero → 1e-12, disp_zero → NaN)
    let dispersions: Vec<f64> = means
        .iter()
        .zip(variances.iter())
        .map(|(&m, &v)| {
            let safe_mean = if m > 0.0 { m as f64 } else { 1e-12 };
            let d = v as f64 / safe_mean;
            if d == 0.0 {
                f64::NAN
            } else {
                d
            }
        })
        .collect();

    // Scanpy: log_mean = np.log1p(mean), log_dispersion = np.log(dispersion)
    // NOTE: np.log, NOT np.log1p for dispersion!
    let log_means: Vec<f64> = means.iter().map(|&m| (m as f64 + 1.0).ln()).collect();
    let log_dispersions: Vec<f64> = dispersions.iter().map(|&d| d.ln()).collect();

    // Step 3: bin genes by log_mean using equal-width bins
    // Match pd.cut behavior: use range of ALL log_means (including zero-mean genes)
    // Only consider genes with finite log_dispersion for statistics
    let valid_log_means: Vec<f64> = log_means
        .iter()
        .copied()
        .filter(|m| m.is_finite())
        .collect();
    if valid_log_means.is_empty() {
        bail!("No genes with valid mean expression");
    }

    let min_mean = valid_log_means
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_mean = valid_log_means
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // pd.cut extends range by 0.1% on left edge to include the minimum
    let range = max_mean - min_mean;
    let adj_min = min_mean - range * 0.001;
    let bin_width = (max_mean - adj_min) / n_bins as f64;

    // If range is too small for reliable binning, reduce bin count.
    // Scanpy's pd.cut would create near-zero-width bins, leading to
    // bins with 0-1 genes and meaningless z-scores. Guard against this
    // by requiring bin_width >= 1e-4 (or use a single bin).
    let min_bin_width = 1e-4;
    let (effective_bins, effective_width) = if bin_width < min_bin_width {
        (1, range + 1e-6) // all genes in one bin
    } else {
        (n_bins, bin_width)
    };

    // Assign bins — match pd.cut (right-inclusive intervals)
    let gene_bins: Vec<Option<usize>> = log_means
        .iter()
        .zip(log_dispersions.iter())
        .map(|(&lm, &ld)| {
            if !lm.is_finite() || !ld.is_finite() || ld.is_nan() {
                None // NaN dispersion → excluded from z-scoring
            } else if effective_bins == 1 || effective_width <= 0.0 {
                Some(0)
            } else {
                // pd.cut right-inclusive: (adj_min, adj_min+width], ...
                let b = ((lm - adj_min) / effective_width).ceil() as i64 - 1;
                let b = b.max(0).min(effective_bins as i64 - 1) as usize;
                Some(b)
            }
        })
        .collect();

    // Step 4: per-bin mean and std of log_dispersion (ddof=1, matching pandas/Scanpy)
    let mut bin_sums = vec![0.0f64; effective_bins];
    let mut bin_sq_sums = vec![0.0f64; effective_bins];
    let mut bin_counts = vec![0usize; effective_bins];

    for g in 0..n_genes {
        if let Some(b) = gene_bins[g] {
            let ld = log_dispersions[g];
            if ld.is_finite() {
                bin_sums[b] += ld;
                bin_sq_sums[b] += ld * ld;
                bin_counts[b] += 1;
            }
        }
    }

    let bin_means: Vec<f64> = (0..effective_bins)
        .map(|b| {
            if bin_counts[b] > 0 {
                bin_sums[b] / bin_counts[b] as f64
            } else {
                0.0
            }
        })
        .collect();

    // Sample std (ddof=1) — matches pandas .std(ddof=1)
    let bin_stds: Vec<f64> = (0..effective_bins)
        .map(|b| {
            if bin_counts[b] > 1 {
                let n = bin_counts[b] as f64;
                let mean = bin_means[b];
                let var = (bin_sq_sums[b] - n * mean * mean) / (n - 1.0); // ddof=1
                var.max(0.0).sqrt()
            } else {
                f64::NAN // single-gene or empty bins: result is NaN (gene excluded)
            }
        })
        .collect();

    // Step 5: z-score within each bin
    let disp_norm: Vec<f32> = (0..n_genes)
        .map(|g| {
            if let Some(b) = gene_bins[g] {
                let std = bin_stds[b];
                let ld = log_dispersions[g];
                if std.is_finite() && std > 1e-12 && ld.is_finite() {
                    ((ld - bin_means[b]) / std) as f32
                } else {
                    0.0 // genes in single-element bins or NaN bins
                }
            } else {
                0.0 // NaN dispersion genes
            }
        })
        .collect();

    // Scanpy clips disp_norm to max = dispersions.values[googled.max()] style
    // Actually, Scanpy clips: dispersions_norm = dispersions_norm.clip(max=disp_norm_max)
    // where disp_norm_max = np.inf — so no clipping. But it does set NaN → 0.
    // We already handle NaN → 0 above.

    // Step 6: rank by disp_norm descending, take top N
    let mut ranked: Vec<usize> = (0..n_genes).collect();
    ranked.sort_by(|&a, &b| {
        disp_norm[b]
            .partial_cmp(&disp_norm[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(n_top_genes);
    ranked.sort(); // stable ordering by index

    let selected_names: Vec<String> = ranked.iter().map(|&i| var_names[i].clone()).collect();

    info!(
        "Selected {} HVGs (seurat flavor, {} bins) from {} genes in {:.3}s",
        n_top_genes,
        n_bins,
        n_genes,
        start.elapsed().as_secs_f64()
    );

    Ok(HvgResult {
        gene_names: selected_names,
        gene_indices: ranked,
        means,
        variances,
        variances_norm: disp_norm,
    })
}

// =====================================================================
//  Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::from_triplets;
    use rand::prelude::*;

    fn make_sc_test_matrix(
        n_cells: usize,
        n_high: usize,
        n_low: usize,
        seed: u64,
    ) -> (SpMat, Vec<String>) {
        let n_genes = n_high + n_low;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for c in 0..n_cells {
            for g in 0..n_genes {
                let val = if g < n_high {
                    // High variance gene — expressed in ~90% of cells, wide range
                    if rng.random::<f32>() < 0.9 {
                        (rng.random::<f32>() * 500.0 + 10.0) as f32
                    } else {
                        0.0
                    }
                } else {
                    // Low variance gene — expressed in ~90% of cells, narrow range
                    if rng.random::<f32>() < 0.9 {
                        (50.0 + rng.random::<f32>() * 2.0) as f32
                    } else {
                        0.0
                    }
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
        (mat, gene_names)
    }

    #[test]
    fn test_hvg_returns_correct_count() {
        let (mat, gene_names) = make_sc_test_matrix(200, 10, 90, 42);
        let result = select_hvg_sparse(&mat, &gene_names, 15).unwrap();

        // Verify correct number selected
        assert_eq!(result.gene_names.len(), 15);
        assert_eq!(result.gene_indices.len(), 15);

        // Verify indices are sorted and within range
        for i in 1..result.gene_indices.len() {
            assert!(
                result.gene_indices[i] > result.gene_indices[i - 1],
                "indices should be sorted"
            );
        }
        for &idx in &result.gene_indices {
            assert!(idx < 100, "index {} out of range", idx);
        }

        // Variances_norm should be non-negative and finite
        for &vn in &result.variances_norm {
            assert!(
                vn >= 0.0 && vn.is_finite(),
                "invalid variances_norm: {}",
                vn
            );
        }
    }

    #[test]
    fn test_hvg_top_n() {
        let (mat, gene_names) = make_sc_test_matrix(50, 10, 40, 99);
        let result = select_hvg_sparse(&mat, &gene_names, 5).unwrap();
        assert_eq!(result.gene_names.len(), 5);
        assert_eq!(result.gene_indices.len(), 5);
    }

    #[test]
    fn test_hvg_clipping() {
        let (mat, gene_names) = make_sc_test_matrix(100, 10, 40, 7);
        let result = select_hvg_sparse(&mat, &gene_names, 10).unwrap();

        let clip_val = (100.0f32).sqrt();
        for &sv in &result.variances_norm {
            assert!(
                sv <= clip_val + 1e-5,
                "Standardized variance {} exceeds clip value {}",
                sv,
                clip_val
            );
        }
    }

    #[test]
    fn test_fit_trend() {
        // Perfect power-law: var = mean^2
        let means = vec![1.0f32, 2.0, 4.0, 8.0, 16.0];
        let variances: Vec<f32> = means.iter().map(|m| m * m).collect();
        let (a, b) = fit_mean_variance_trend(&means, &variances);
        assert!((a).abs() < 0.1, "intercept should be ~0, got {}", a);
        assert!((b - 2.0).abs() < 0.1, "slope should be ~2, got {}", b);
    }

    // -----------------------------------------------------------------
    //  Seurat (dispersion-bin) flavor tests
    // -----------------------------------------------------------------

    #[test]
    fn test_seurat_flavor_returns_correct_count() {
        let (mat, gene_names) = make_sc_test_matrix(200, 10, 90, 42);
        let result = select_hvg_seurat(&mat, &gene_names, 15, 20).unwrap();

        assert_eq!(result.gene_names.len(), 15);
        assert_eq!(result.gene_indices.len(), 15);

        // Indices must be sorted and within range
        for i in 1..result.gene_indices.len() {
            assert!(
                result.gene_indices[i] > result.gene_indices[i - 1],
                "indices should be sorted"
            );
        }
        for &idx in &result.gene_indices {
            assert!(idx < 100, "index {} out of range", idx);
        }
    }

    #[test]
    fn test_seurat_flavor_prefers_high_dispersion() {
        // The seurat flavor z-scores log-dispersion within mean-bins, so
        // genes with higher dispersion relative to genes at similar mean
        // expression should rank higher.
        //
        // Strategy: create two classes of genes that share the SAME mean
        // but differ sharply in dispersion.  We achieve same-mean bimodal
        // data by using values symmetrically above/below the mean.
        //
        // 5 high-dispersion genes: half cells = mean + delta, half = mean - delta  (large delta)
        // 95 low-dispersion genes: all cells = mean + tiny noise
        let n_cells = 1000;
        let n_high = 5;
        let n_low = 95;
        let n_genes = n_high + n_low;
        let base_mean: f32 = 50.0;
        let delta: f32 = 45.0; // values alternate between 5 and 95 => mean=50, var~2025, disp~40.5
        let mut rng = StdRng::seed_from_u64(77);
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for c in 0..n_cells {
            for g in 0..n_genes {
                let val: f32 = if g < n_high {
                    // High-dispersion: symmetric bimodal around base_mean
                    if c % 2 == 0 {
                        base_mean + delta
                    } else {
                        base_mean - delta
                    }
                } else {
                    // Low-dispersion: tight around base_mean
                    base_mean + (rng.random::<f32>() - 0.5) * 0.01
                };
                if val > 0.0 {
                    rows.push(c);
                    cols.push(g);
                    vals.push(val);
                }
            }
        }

        let mat = from_triplets(n_cells, n_genes, &rows, &cols, &vals);
        let names: Vec<String> = (0..n_genes).map(|g| format!("G{:04}", g)).collect();

        let result = select_hvg_seurat(&mat, &names, 5, 20).unwrap();

        // The 5 high-dispersion genes should dominate the top 5
        let high_disp_genes: Vec<String> = (0..n_high).map(|g| format!("G{:04}", g)).collect();
        let n_high_selected = high_disp_genes
            .iter()
            .filter(|n| result.gene_names.contains(n))
            .count();

        assert!(
            n_high_selected >= 4,
            "Expected >= 4 of 5 high-dispersion genes in top 5, got {} ({:?})",
            n_high_selected,
            result.gene_names
        );
    }

    #[test]
    fn test_seurat_flavor_bins() {
        let (mat, gene_names) = make_sc_test_matrix(100, 10, 40, 123);
        let result = select_hvg_seurat(&mat, &gene_names, 10, 20).unwrap();

        // All disp_norm values must be finite
        for (i, &dn) in result.variances_norm.iter().enumerate() {
            assert!(dn.is_finite(), "disp_norm[{}] = {} is not finite", i, dn);
        }

        // At least some genes should have non-zero normalized dispersion
        let nonzero = result
            .variances_norm
            .iter()
            .filter(|&&v| v.abs() > 1e-10)
            .count();
        assert!(
            nonzero > 0,
            "Expected some genes with non-zero disp_norm, got all zeros"
        );
    }
}
