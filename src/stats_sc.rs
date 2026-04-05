//! Statistical utilities for single-cell analysis.
//!
//! Self-contained implementations — no external stats crate dependency.
//! Welford mean/variance, BH FDR correction, median, MAD.

/// Welford's single-pass mean and variance (f64).
#[allow(dead_code)]
pub fn welford_mean_var(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (data[0], 0.0);
    }
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (i, &x) in data.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    (mean, m2 / (n - 1) as f64)
}

/// Welford's single-pass mean and variance (f32).
#[allow(dead_code)]
pub fn welford_mean_var_f32(data: &[f32]) -> (f32, f32) {
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (data[0], 0.0);
    }
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (i, &x) in data.iter().enumerate() {
        let x = x as f64;
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    (mean as f32, (m2 / (n - 1) as f64) as f32)
}

/// Welford mean/variance for sparse data: given only the non-zero values
/// and the total number of elements (including zeros).
#[allow(dead_code)]
pub fn welford_mean_var_sparse(nonzero_vals: &[f32], n_total: usize) -> (f32, f32) {
    if n_total == 0 {
        return (0.0, 0.0);
    }
    if n_total == 1 {
        return if nonzero_vals.is_empty() {
            (0.0, 0.0)
        } else {
            (nonzero_vals[0], 0.0)
        };
    }

    let sum: f64 = nonzero_vals.iter().map(|&v| v as f64).sum();
    let sum_sq: f64 = nonzero_vals.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let n = n_total as f64;
    let mean = sum / n;
    let var = (sum_sq - n * mean * mean) / (n - 1.0);
    (mean as f32, var.max(0.0) as f32)
}

/// Benjamini-Hochberg FDR correction (in-place).
///
/// Adjusts p-values so that `adjusted_p[i]` controls the false discovery rate.
/// p-values are modified in-place. Non-finite values are set to 1.0.
pub fn bh_adjust(pvalues: &mut [f64]) {
    let n = pvalues.len();
    if n == 0 {
        return;
    }

    // Get sorted indices (ascending p-value)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        pvalues[a]
            .partial_cmp(&pvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Adjust: p_adj[i] = p[i] * n / rank
    // Then enforce monotonicity from the bottom up
    let mut adjusted = vec![0.0f64; n];
    for (rank_minus_1, &idx) in indices.iter().enumerate() {
        let rank = (rank_minus_1 + 1) as f64;
        adjusted[idx] = pvalues[idx] * n as f64 / rank;
    }

    // Enforce monotonicity: walk from largest p-value to smallest
    let mut cummin = f64::INFINITY;
    for &idx in indices.iter().rev() {
        if adjusted[idx] < cummin {
            cummin = adjusted[idx];
        } else {
            adjusted[idx] = cummin;
        }
    }

    // Clamp to [0, 1]
    for (i, &adj) in adjusted.iter().enumerate() {
        pvalues[i] = adj.min(1.0).max(0.0);
        if !pvalues[i].is_finite() {
            pvalues[i] = 1.0;
        }
    }
}

/// Compute median of a slice (sorts a copy).
#[allow(dead_code)]
pub fn median(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Median of f64 values.
#[allow(dead_code)]
pub fn median_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Median Absolute Deviation: MAD = median(|x_i - median(x)|)
#[allow(dead_code)]
pub fn mad(data: &[f32]) -> f32 {
    let med = median(data);
    let deviations: Vec<f32> = data.iter().map(|&x| (x - med).abs()).collect();
    median(&deviations)
}

// =====================================================================
//  Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_basic() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, var) = welford_mean_var(&data);
        assert!((mean - 5.0).abs() < 1e-10);
        assert!((var - 4.571428571).abs() < 1e-6);
    }

    #[test]
    fn test_welford_sparse() {
        // Full data: [3.0, 0.0, 0.0, 5.0, 0.0]
        let nonzero = vec![3.0f32, 5.0];
        let (mean, var) = welford_mean_var_sparse(&nonzero, 5);
        // mean = 8/5 = 1.6
        assert!((mean - 1.6).abs() < 1e-4, "mean = {}", mean);
        // var = ((3-1.6)^2 + (0-1.6)^2*3 + (5-1.6)^2) / 4
        let expected_var =
            ((3.0 - 1.6_f64).powi(2) + 3.0 * (1.6_f64).powi(2) + (5.0 - 1.6_f64).powi(2)) / 4.0;
        assert!(
            (var - expected_var as f32).abs() < 0.1,
            "var = {}, expected = {}",
            var,
            expected_var
        );
    }

    #[test]
    fn test_bh_adjust() {
        let mut pvals = vec![0.05, 0.01, 0.10, 0.001];
        bh_adjust(&mut pvals);
        // Sorted: 0.001, 0.01, 0.05, 0.10
        // Adjusted: 0.004, 0.02, 0.0667, 0.10
        assert!(pvals[3] < 0.01, "smallest p should adjust to < 0.01");
        assert!(pvals[1] < 0.05);
    }

    #[test]
    fn test_median() {
        assert!((median(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-6);
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-6);
        assert!((median(&[5.0]) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mad() {
        // Data: [1, 2, 3, 4, 5], median=3, deviations=[2,1,0,1,2], MAD=1
        assert!((mad(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 1.0).abs() < 1e-6);
    }
}
