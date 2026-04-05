//! Moran's I spatial autocorrelation statistic.
//!
//! Computes Moran's I for each gene across spatial spots with permutation-based
//! p-values. Parallelized across genes with rayon. Expected 100x+ speedup over
//! Squidpy (484s → ~3-5s for 2000 genes on 3798 Visium spots).

use crate::graph::CsrGraph;
use crate::stats_sc;
use anyhow::Result;
use rand::prelude::*;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

/// Results of Moran's I computation.
pub struct MoransResult {
    pub gene_names: Vec<String>,
    pub i_values: Vec<f64>,
    pub expected: Vec<f64>,
    pub pval_sim: Vec<f64>,
    pub pval_sim_fdr: Vec<f64>,
}

/// Build a spatial neighbor graph from 2D spot coordinates using k-nearest neighbors.
///
/// For Visium hexagonal grids, `n_neighbors=6` captures the lattice structure.
/// Uses brute-force kNN in 2D Euclidean space (n ~ 4000 spots is small).
/// Returns a CsrGraph with binary weights (w = 1.0).
pub fn spatial_neighbor_graph(coords: &[(f64, f64)], n_neighbors: usize) -> CsrGraph {
    let n = coords.len();
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n * n_neighbors);

    for i in 0..n {
        // Compute distances to all other spots
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                (j, dx * dx + dy * dy) // squared distance
            })
            .collect();

        // Partial sort to find k nearest
        let k = n_neighbors.min(dists.len());
        dists.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        for &(j, _) in &dists[..k] {
            edges.push((i, j, 1.0)); // binary weight
        }
    }

    CsrGraph::from_edge_list(&edges, n)
}

/// Compute Moran's I for a single gene across spatial spots.
///
/// I = (n / W) * sum_ij(w_ij * z_i * z_j) / sum_i(z_i^2)
/// where z_i = x_i - mean(x)
fn morans_i_single(x: &[f32], graph: &CsrGraph, w_sum: f64) -> f64 {
    let n = x.len() as f64;
    let mean: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n;

    // Denominator: sum of squared deviations
    let denom: f64 = x
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum();

    if denom < 1e-15 {
        return 0.0;
    }

    // Numerator: weighted cross-deviation sum
    let mut numer = 0.0f64;
    for i in 0..graph.n_nodes {
        let zi = x[i] as f64 - mean;
        let range = graph.neighbor_range(i);
        for (&j, &w) in graph.targets[range.clone()]
            .iter()
            .zip(graph.weights[range].iter())
        {
            let zj = x[j] as f64 - mean;
            numer += w * zi * zj;
        }
    }

    (n / w_sum) * numer / denom
}

/// Compute permutation-based p-value for Moran's I.
fn permutation_pvalue(
    x: &[f32],
    graph: &CsrGraph,
    w_sum: f64,
    observed_i: f64,
    n_perm: usize,
    rng: &mut StdRng,
) -> f64 {
    let mut x_perm: Vec<f32> = x.to_vec();
    let mut count_ge = 0usize;

    for _ in 0..n_perm {
        x_perm.shuffle(rng);
        let i_perm = morans_i_single(&x_perm, graph, w_sum);
        if i_perm >= observed_i {
            count_ge += 1;
        }
    }

    // Phipson-Smyth correction: never return p=0
    (count_ge as f64 + 1.0) / (n_perm as f64 + 1.0)
}

/// Compute Moran's I for each gene, with permutation p-values and BH FDR.
///
/// Gene-level parallelism via rayon. Each gene gets a deterministic RNG
/// seeded from (global_seed + gene_index).
///
/// # Arguments
/// * `expression` — expression values per gene: `expression[g][s]` = gene g at spot s
/// * `graph` — spatial neighbor graph (from `spatial_neighbor_graph`)
/// * `gene_names` — gene identifiers
/// * `n_perm` — number of permutations for p-value (typically 999)
/// * `seed` — global seed for reproducibility
pub fn compute_morans_i(
    expression: &[Vec<f32>],
    graph: &CsrGraph,
    gene_names: &[String],
    n_perm: usize,
    seed: u64,
) -> MoransResult {
    let t0 = Instant::now();
    let n_genes = expression.len();
    let n_spots = graph.n_nodes;

    // Precompute total weight sum
    let w_sum = graph.total_weight * 2.0; // total_weight is sum/2 in CsrGraph

    // Expected value under null hypothesis
    let expected_i = -1.0 / (n_spots as f64 - 1.0);

    // Parallel computation across genes
    let results: Vec<(f64, f64)> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(g as u64));
            let i_obs = morans_i_single(&expression[g], graph, w_sum);
            let pval = if n_perm > 0 {
                permutation_pvalue(&expression[g], graph, w_sum, i_obs, n_perm, &mut rng)
            } else {
                0.0 // skip permutation test
            };
            (i_obs, pval)
        })
        .collect();

    let i_values: Vec<f64> = results.iter().map(|(i, _)| *i).collect();
    let pval_sim: Vec<f64> = results.iter().map(|(_, p)| *p).collect();

    // BH FDR correction
    let mut pval_fdr = pval_sim.clone();
    stats_sc::bh_adjust(&mut pval_fdr);

    let elapsed = t0.elapsed().as_secs_f64();
    log::info!(
        "Moran's I: {} genes, {} spots, {} permutations in {:.2}s",
        n_genes,
        n_spots,
        n_perm,
        elapsed
    );

    MoransResult {
        gene_names: gene_names.to_vec(),
        i_values,
        expected: vec![expected_i; n_genes],
        pval_sim,
        pval_sim_fdr: pval_fdr,
    }
}

/// Write Moran's I results to CSV.
pub fn write_morans_csv(result: &MoransResult, path: &std::path::Path) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["gene", "I", "pval_sim", "pval_sim_fdr", "E_I"])?;

    for i in 0..result.gene_names.len() {
        wtr.write_record([
            &result.gene_names[i],
            &format!("{:.15}", result.i_values[i]),
            &format!("{:.15}", result.pval_sim[i]),
            &format!("{:.15}", result.pval_sim_fdr[i]),
            &format!("{:.15}", result.expected[i]),
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

    fn simple_line_graph() -> CsrGraph {
        // 5 nodes in a line: 0-1-2-3-4
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)];
        CsrGraph::from_edge_list(&edges, 5)
    }

    #[test]
    fn test_expected_value() {
        let n = 100;
        let expected = -1.0 / (n as f64 - 1.0);
        assert!((expected - (-0.010101)).abs() < 0.001);
    }

    #[test]
    fn test_morans_i_perfect_cluster() {
        // Values increase along the line → high positive autocorrelation
        let graph = simple_line_graph();
        let w_sum = graph.total_weight * 2.0;
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let i = morans_i_single(&x, &graph, w_sum);
        assert!(i > 0.3, "clustered pattern should have I > 0, got {}", i);
    }

    #[test]
    fn test_morans_i_dispersed() {
        // Alternating high/low values → negative autocorrelation
        let graph = simple_line_graph();
        let w_sum = graph.total_weight * 2.0;
        let x = vec![10.0f32, 0.0, 10.0, 0.0, 10.0];
        let i = morans_i_single(&x, &graph, w_sum);
        assert!(i < 0.0, "dispersed pattern should have I < 0, got {}", i);
    }

    #[test]
    fn test_morans_i_constant() {
        // All same value → I should be 0 (no autocorrelation possible)
        let graph = simple_line_graph();
        let w_sum = graph.total_weight * 2.0;
        let x = vec![5.0f32, 5.0, 5.0, 5.0, 5.0];
        let i = morans_i_single(&x, &graph, w_sum);
        assert!(
            (i).abs() < 1e-10,
            "constant field should have I = 0, got {}",
            i
        );
    }

    #[test]
    fn test_spatial_neighbor_graph() {
        // 4 spots in a square: (0,0), (1,0), (0,1), (1,1)
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let graph = spatial_neighbor_graph(&coords, 2);

        assert_eq!(graph.n_nodes, 4);
        // Each spot should have exactly 2 neighbors
        for i in 0..4 {
            assert_eq!(
                graph.neighbor_range(i).len(),
                2,
                "spot {} should have 2 neighbors",
                i
            );
        }
    }

    #[test]
    fn test_compute_morans_i_deterministic() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let graph = spatial_neighbor_graph(&coords, 2);

        let expression = vec![vec![1.0f32, 2.0, 3.0, 4.0, 5.0]];
        let gene_names = vec!["GENE1".to_string()];

        let r1 = compute_morans_i(&expression, &graph, &gene_names, 99, 42);
        let r2 = compute_morans_i(&expression, &graph, &gene_names, 99, 42);

        assert!(
            (r1.i_values[0] - r2.i_values[0]).abs() < 1e-15,
            "Moran's I should be deterministic"
        );
        assert!(
            (r1.pval_sim[0] - r2.pval_sim[0]).abs() < 1e-15,
            "p-values should be deterministic"
        );
    }
}
