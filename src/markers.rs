//! Marker gene detection via Wilcoxon rank-sum test.
//!
//! Equivalent to Scanpy's `sc.tl.rank_genes_groups()` with method='wilcoxon'.
//! For each cluster vs rest, computes U-statistic (normal approximation),
//! log2 fold change, and BH-adjusted p-values. Genes are ranked by adjusted
//! p-value ascending, then fold change descending for ties.

use crate::sparse::SpMat;
use crate::stats_sc;
use anyhow::Result;
use rayon::prelude::*;

/// Result of marker gene detection for a single cluster.
pub struct MarkerResult {
    #[allow(dead_code)]
    pub cluster_id: usize,
    #[allow(dead_code)]
    pub gene_names: Vec<String>,
    #[allow(dead_code)]
    pub scores: Vec<f64>, // z-score from normal approximation
    #[allow(dead_code)]
    pub pvals: Vec<f64>,
    #[allow(dead_code)]
    pub pvals_adj: Vec<f64>, // BH-adjusted
    #[allow(dead_code)]
    pub log2fc: Vec<f64>, // log2 fold change (cluster vs rest)
}

/// Wilcoxon rank-sum test for a single gene: cluster vs rest.
///
/// Uses the normal approximation for large samples (which is standard
/// for scRNA-seq where cell counts are typically in the thousands).
///
/// Returns (z-score, two-sided p-value).
fn wilcoxon_rank_sum(values: &[f32], is_cluster: &[bool]) -> (f64, f64) {
    let n = values.len();

    // Build index-value pairs and sort by value for ranking
    let mut indexed: Vec<(usize, f32)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (1-based), averaging ties
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n
            && indexed[j]
                .1
                .partial_cmp(&indexed[i].1)
                .unwrap_or(std::cmp::Ordering::Equal)
                == std::cmp::Ordering::Equal
        {
            j += 1;
        }
        // Positions i..j have the same value; average rank = (i+1 + j) / 2
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }

    // Sum ranks in cluster, count sizes
    let mut rank_sum_cluster = 0.0f64;
    let mut n1 = 0usize; // cluster size
    let mut n2 = 0usize; // rest size
    for idx in 0..n {
        if is_cluster[idx] {
            rank_sum_cluster += ranks[idx];
            n1 += 1;
        } else {
            n2 += 1;
        }
    }

    if n1 == 0 || n2 == 0 {
        return (0.0, 1.0);
    }

    // U statistic for cluster group
    let u1 = rank_sum_cluster - (n1 as f64 * (n1 as f64 + 1.0)) / 2.0;

    // Normal approximation: mean and variance of U under H0
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let mu = n1f * n2f / 2.0;
    let sigma = ((n1f * n2f * (n1f + n2f + 1.0)) / 12.0).sqrt();

    if sigma < 1e-15 {
        return (0.0, 1.0);
    }

    let z = (u1 - mu) / sigma;
    // Two-sided p-value, clamped to [0, 1]
    let p = (2.0 * normal_cdf(-z.abs())).clamp(0.0, 1.0);

    (z, p)
}

/// Standard normal CDF via the error function approximation.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun, formula 7.1.26).
///
/// Maximum error: |epsilon(x)| <= 1.5e-7.
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Extract dense gene expression vectors from sparse CSR matrix.
///
/// Returns a Vec of length n_genes, each containing an f32 vec of length n_cells.
/// This extracts all genes in a single pass over the sparse matrix, which is
/// much more efficient than per-gene column access on CSR layout.
fn extract_gene_columns(mat: &SpMat) -> Vec<Vec<f32>> {
    let n_cells = mat.rows();
    let n_genes = mat.cols();
    let mut columns: Vec<Vec<f32>> = vec![vec![0.0f32; n_cells]; n_genes];

    for cell in 0..n_cells {
        let row = mat.outer_view(cell).unwrap();
        for (gene_idx, &val) in row.iter() {
            columns[gene_idx][cell] = val;
        }
    }

    columns
}

fn score_cluster(
    cluster_id: usize,
    gene_columns: &[Vec<f32>],
    clusters: &[usize],
    var_names: &[String],
) -> MarkerResult {
    let n_cells = clusters.len();
    let n_genes = gene_columns.len();
    let is_cluster: Vec<bool> = clusters.iter().map(|&c| c == cluster_id).collect();
    let n_in = is_cluster.iter().filter(|&&b| b).count();
    let n_out = n_cells - n_in;

    let mut scores = Vec::with_capacity(n_genes);
    let mut pvals = Vec::with_capacity(n_genes);
    let mut log2fcs = Vec::with_capacity(n_genes);

    for g in 0..n_genes {
        let vals = &gene_columns[g];
        let (score, pval) = wilcoxon_rank_sum(vals, &is_cluster);

        let mean_in: f64 = vals
            .iter()
            .enumerate()
            .filter(|(i, _)| is_cluster[*i])
            .map(|(_, &v)| v as f64)
            .sum::<f64>()
            / n_in.max(1) as f64;
        let mean_out: f64 = vals
            .iter()
            .enumerate()
            .filter(|(i, _)| !is_cluster[*i])
            .map(|(_, &v)| v as f64)
            .sum::<f64>()
            / n_out.max(1) as f64;

        let log2fc = if mean_out > 1e-10 {
            (mean_in / mean_out).log2()
        } else if mean_in > 1e-10 {
            10.0
        } else {
            0.0
        };

        scores.push(score);
        pvals.push(pval);
        log2fcs.push(log2fc);
    }

    let mut pvals_adj = pvals.clone();
    stats_sc::bh_adjust(&mut pvals_adj);

    MarkerResult {
        cluster_id,
        gene_names: var_names.to_vec(),
        scores,
        pvals,
        pvals_adj,
        log2fc: log2fcs,
    }
}

/// Compute marker genes for all clusters via Wilcoxon rank-sum test.
///
/// One cluster per rayon task. Ranking is deterministic (no RNG): same
/// top-N genes and order as the serial path.
///
/// # Arguments
/// * `mat` - Sparse CSR matrix (cells x genes), normalized log1p expression
/// * `clusters` - Cluster assignment for each cell (0-indexed)
/// * `var_names` - Gene names corresponding to matrix columns
/// * `_n_top` - Reserved for future use (top N filtering done at write time)
pub fn find_markers(
    mat: &SpMat,
    clusters: &[usize],
    var_names: &[String],
    _n_top: usize,
) -> Result<Vec<MarkerResult>> {
    find_markers_impl(mat, clusters, var_names, true)
}

/// Serial Wilcoxon markers (same ranking as `find_markers`; for tests).
pub fn find_markers_serial(
    mat: &SpMat,
    clusters: &[usize],
    var_names: &[String],
    _n_top: usize,
) -> Result<Vec<MarkerResult>> {
    find_markers_impl(mat, clusters, var_names, false)
}

fn find_markers_impl(
    mat: &SpMat,
    clusters: &[usize],
    var_names: &[String],
    parallel: bool,
) -> Result<Vec<MarkerResult>> {
    let n_clusters = *clusters.iter().max().unwrap_or(&0) + 1;
    let gene_columns = extract_gene_columns(mat);

    let results = if parallel {
        (0..n_clusters)
            .into_par_iter()
            .map(|cluster_id| score_cluster(cluster_id, &gene_columns, clusters, var_names))
            .collect()
    } else {
        (0..n_clusters)
            .map(|cluster_id| score_cluster(cluster_id, &gene_columns, clusters, var_names))
            .collect()
    };

    Ok(results)
}

/// Write marker results to CSV (top N genes per cluster).
///
/// Genes are sorted by adjusted p-value ascending, then by absolute score
/// descending for tie-breaking. Only the top `n_top` genes per cluster
/// are written.
pub fn write_markers_csv(
    results: &[MarkerResult],
    n_top: usize,
    path: &std::path::Path,
) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["cluster", "gene", "score", "pval", "pval_adj", "log2fc"])?;

    for result in results {
        // Sort gene indices: by pval_adj ascending, then |score| descending
        let mut indices: Vec<usize> = (0..result.gene_names.len()).collect();
        indices.sort_by(|&a, &b| {
            result.pvals_adj[a]
                .partial_cmp(&result.pvals_adj[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    result.log2fc[b]
                        .abs()
                        .partial_cmp(&result.log2fc[a].abs())
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        for &i in indices.iter().take(n_top) {
            wtr.write_record([
                &result.cluster_id.to_string(),
                &result.gene_names[i],
                &format!("{:.4}", result.scores[i]),
                &format!("{:.6e}", result.pvals[i]),
                &format!("{:.6e}", result.pvals_adj[i]),
                &format!("{:.4}", result.log2fc[i]),
            ])?;
        }
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
    use crate::sparse;

    #[test]
    fn test_erf_values() {
        // erf(0) = 0
        assert!((erf(0.0)).abs() < 1e-7);
        // erf(1) ~ 0.8427
        assert!((erf(1.0) - 0.8427007929).abs() < 1e-5);
        // erf(-1) ~ -0.8427
        assert!((erf(-1.0) + 0.8427007929).abs() < 1e-5);
        // erf(inf) -> 1
        assert!((erf(10.0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_normal_cdf_values() {
        // CDF(0) = 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        // CDF(-inf) -> 0
        assert!(normal_cdf(-10.0) < 1e-10);
        // CDF(+inf) -> 1
        assert!((normal_cdf(10.0) - 1.0).abs() < 1e-10);
        // CDF(1.96) ~ 0.975
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.001);
    }

    #[test]
    fn test_wilcoxon_identical_groups() {
        // All values the same -> p = 1.0
        let values = vec![1.0f32; 10];
        let is_cluster = vec![
            true, true, true, true, true, false, false, false, false, false,
        ];
        let (_, p) = wilcoxon_rank_sum(&values, &is_cluster);
        assert!(
            (p - 1.0).abs() < 1e-10,
            "identical values should give p=1.0, got {}",
            p
        );
    }

    #[test]
    fn test_wilcoxon_perfect_separation() {
        // Cluster has all high values, rest has all low -> very small p
        let mut values = vec![0.0f32; 20];
        let mut is_cluster = vec![false; 20];
        for i in 0..10 {
            values[i] = 100.0;
            is_cluster[i] = true;
        }
        for i in 10..20 {
            values[i] = 1.0;
        }
        let (z, p) = wilcoxon_rank_sum(&values, &is_cluster);
        assert!(z > 0.0, "cluster with higher values should have positive z");
        assert!(
            p < 0.001,
            "perfect separation should give p < 0.001, got {}",
            p
        );
    }

    #[test]
    fn test_wilcoxon_empty_group() {
        let values = vec![1.0f32, 2.0, 3.0];
        let is_cluster = vec![false, false, false];
        let (z, p) = wilcoxon_rank_sum(&values, &is_cluster);
        assert_eq!(z, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_wilcoxon_ranks_with_ties() {
        // Values: [1, 1, 2, 2, 3]
        // Sorted: [1, 1, 2, 2, 3]
        // Ranks:  [1.5, 1.5, 3.5, 3.5, 5]
        let values = vec![1.0f32, 1.0, 2.0, 2.0, 3.0];
        let is_cluster = vec![true, false, true, false, true];
        let (z, p) = wilcoxon_rank_sum(&values, &is_cluster);
        // Cluster has values [1, 2, 3], rest has [1, 2]
        // Cluster rank sum = 1.5 + 3.5 + 5.0 = 10.0
        // n1=3, n2=2, U = 10 - 3*4/2 = 10 - 6 = 4
        // mu = 3*2/2 = 3, sigma = sqrt(3*2*6/12) = sqrt(3)
        // z = (4-3)/sqrt(3) ~ 0.577
        assert!((z - 0.5774).abs() < 0.01, "expected z ~ 0.577, got {}", z);
        // With such a small sample, p should be fairly large
        assert!(p > 0.1, "small sample, no clear separation: p={}", p);
    }

    #[test]
    fn test_find_markers_basic() {
        // 6 cells x 3 genes, two clusters
        // Cluster 0 (cells 0-2): gene 0 high, gene 1 low
        // Cluster 1 (cells 3-5): gene 0 low, gene 1 high
        // Gene 2: uniform (not a marker)
        let mat = sparse::from_triplets(
            6,
            3,
            &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[
                10.0, 1.0, 5.0, // cell 0 (cluster 0)
                9.0, 2.0, 5.0, // cell 1 (cluster 0)
                11.0, 1.0, 5.0, // cell 2 (cluster 0)
                1.0, 10.0, 5.0, // cell 3 (cluster 1)
                2.0, 9.0, 5.0, // cell 4 (cluster 1)
                1.0, 11.0, 5.0, // cell 5 (cluster 1)
            ],
        );

        let clusters = vec![0, 0, 0, 1, 1, 1];
        let var_names: Vec<String> = vec!["GeneA".into(), "GeneB".into(), "GeneC".into()];

        let results = find_markers(&mat, &clusters, &var_names, 3).unwrap();
        assert_eq!(results.len(), 2);

        // Cluster 0: GeneA should be the top marker (highest in cluster 0)
        let c0 = &results[0];
        assert_eq!(c0.cluster_id, 0);
        // GeneA (index 0) should have positive z-score for cluster 0
        assert!(
            c0.scores[0] > 0.0,
            "GeneA should be upregulated in cluster 0"
        );
        // GeneA should have positive log2fc for cluster 0
        assert!(
            c0.log2fc[0] > 0.0,
            "GeneA log2fc should be positive in cluster 0"
        );
        // GeneB (index 1) should have negative z-score for cluster 0
        assert!(
            c0.scores[1] < 0.0,
            "GeneB should be downregulated in cluster 0"
        );

        // Cluster 1: GeneB should be the top marker
        let c1 = &results[1];
        assert!(
            c1.scores[1] > 0.0,
            "GeneB should be upregulated in cluster 1"
        );
        assert!(
            c1.log2fc[1] > 0.0,
            "GeneB log2fc should be positive in cluster 1"
        );
    }

    #[test]
    fn test_find_markers_pvals_adjusted() {
        // Verify that adjusted p-values are >= raw p-values
        let mat = sparse::from_triplets(
            6,
            3,
            &[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            &[
                10.0, 1.0, 9.0, 2.0, 11.0, 1.0, 1.0, 10.0, 2.0, 9.0, 1.0, 11.0,
            ],
        );
        let clusters = vec![0, 0, 0, 1, 1, 1];
        let var_names: Vec<String> = vec!["G1".into(), "G2".into(), "G3".into()];

        let results = find_markers(&mat, &clusters, &var_names, 3).unwrap();
        for result in &results {
            for i in 0..result.pvals.len() {
                // BH adjustment should generally not decrease p-values.
                // Allow a small floating-point tolerance.
                assert!(
                    result.pvals_adj[i] >= result.pvals[i] - 1e-10,
                    "adjusted p-value ({}) should be >= raw p-value ({}) for gene {}",
                    result.pvals_adj[i],
                    result.pvals[i],
                    result.gene_names[i],
                );
            }
        }
    }

    #[test]
    fn test_write_markers_csv() {
        let result = MarkerResult {
            cluster_id: 0,
            gene_names: vec!["A".into(), "B".into(), "C".into()],
            scores: vec![3.0, -1.0, 0.5],
            pvals: vec![0.001, 0.1, 0.05],
            pvals_adj: vec![0.003, 0.1, 0.075],
            log2fc: vec![2.0, -0.5, 0.3],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("markers.csv");
        write_markers_csv(&[result], 2, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // Header + 2 top genes
        assert_eq!(lines.len(), 3, "should have header + 2 data rows");
        assert!(lines[0].contains("cluster"));
        assert!(lines[0].contains("gene"));
        assert!(lines[0].contains("pval_adj"));
        assert!(lines[0].contains("log2fc"));
        // First data row should be gene A (lowest pval_adj = 0.003)
        assert!(lines[1].contains(",A,"), "top gene should be A");
    }

    #[test]
    fn test_extract_gene_columns() {
        let mat = sparse::from_triplets(3, 2, &[0, 1, 2], &[0, 1, 0], &[1.0, 2.0, 3.0]);
        let cols = extract_gene_columns(&mat);
        assert_eq!(cols.len(), 2);
        // Gene 0: [1, 0, 3]
        assert_eq!(cols[0], vec![1.0, 0.0, 3.0]);
        // Gene 1: [0, 2, 0]
        assert_eq!(cols[1], vec![0.0, 2.0, 0.0]);
    }

    fn ranked_top_genes(result: &MarkerResult, n_top: usize) -> Vec<String> {
        let mut indices: Vec<usize> = (0..result.gene_names.len()).collect();
        indices.sort_by(|&a, &b| {
            result.pvals_adj[a]
                .partial_cmp(&result.pvals_adj[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    result.log2fc[b]
                        .abs()
                        .partial_cmp(&result.log2fc[a].abs())
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
        indices
            .iter()
            .take(n_top)
            .map(|&i| result.gene_names[i].clone())
            .collect()
    }

    #[test]
    fn test_find_markers_parallel_matches_serial() {
        // 12 cells x 5 genes, 3 clusters — small enough to compare bit-stable ranks.
        let mat = sparse::from_triplets(
            12,
            5,
            &[
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
            ],
            &[
                0, 1, 0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 2, 4, 3, 4, 3, 0, 4, 0, 4, 1, 0, 4,
            ],
            &[
                9.0, 1.0, 8.0, 2.0, 10.0, 1.0, 2.0, 8.0, 1.0, 9.0, 2.0, 7.0, 8.0, 1.0, 9.0, 2.0,
                7.0, 1.0, 10.0, 1.0, 8.0, 2.0, 1.0, 9.0,
            ],
        );
        let clusters = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let names: Vec<String> = (0..5).map(|g| format!("G{g}")).collect();

        let parallel = find_markers(&mat, &clusters, &names, 3).unwrap();
        let serial = find_markers_serial(&mat, &clusters, &names, 3).unwrap();
        assert_eq!(parallel.len(), serial.len());
        for (p, s) in parallel.iter().zip(serial.iter()) {
            assert_eq!(p.cluster_id, s.cluster_id);
            assert_eq!(ranked_top_genes(p, 3), ranked_top_genes(s, 3));
            for i in 0..p.gene_names.len() {
                assert!((p.scores[i] - s.scores[i]).abs() < 1e-10);
                assert!((p.pvals[i] - s.pvals[i]).abs() < 1e-12);
                assert!((p.pvals_adj[i] - s.pvals_adj[i]).abs() < 1e-12);
                let same_fc =
                    p.log2fc[i] == s.log2fc[i] || ((p.log2fc[i] - s.log2fc[i]).abs() < 1e-10);
                assert!(
                    same_fc,
                    "log2fc[{}] cluster {} parallel={} serial={}",
                    i, p.cluster_id, p.log2fc[i], s.log2fc[i]
                );
            }
        }
    }
}
