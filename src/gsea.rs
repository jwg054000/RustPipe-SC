/// GSEA (Gene Set Enrichment Analysis) — preranked, Subramanian et al. 2005
///
/// Replaces R `fgsea` with Rust + rayon parallel permutations.
/// Target: beat fgsea's 5.5 s on 65 pathways x 18,168 genes x 10,000 permutations.
///
/// Algorithm:
///   1. Walk the ranked gene list. Genes in the set increment a running sum
///      proportional to |stat|; genes outside decrement by a constant.
///   2. ES = maximum deviation from zero in the running sum.
///   3. Null distribution via label permutation (sample random hit indices).
///   4. NES = ES / mean(|ES_null|) for matching sign.
///   5. p-value from null, then BH FDR across pathways.
use std::collections::HashMap;
use std::time::Instant;

use rand::seq::index::sample;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Results for a single pathway.
#[allow(dead_code)]
pub struct PathwayResult {
    pub name: String,
    pub es: f64,
    pub nes: f64,
    pub pvalue: f64,
    pub padj: f64,
    pub n_genes: usize,
    pub leading_edge: Vec<String>,
}

/// Timing information for the GSEA run.
#[allow(dead_code)]
pub struct GseaTimings {
    pub total_ms: f64,
    pub per_pathway_ms: f64,
}

/// Container for full GSEA output.
#[allow(dead_code)]
pub struct GseaResults {
    pub pathways: Vec<PathwayResult>,
    pub timings: GseaTimings,
}

// ---------------------------------------------------------------------------
// Entry point called from main.rs
// ---------------------------------------------------------------------------

/// CLI wrapper: read inputs, run GSEA, write CSV.
pub fn run(
    ranks_path: &str,
    gmt_path: &str,
    nperm: usize,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let ranked_genes = crate::io::read_ranked_list(ranks_path)?;
    let gene_sets = crate::io::read_gmt(gmt_path)?;

    let results = run_gsea(&ranked_genes, &gene_sets, nperm, 15, 500);

    write_gsea_csv(&results, output_path)?;

    eprintln!(
        "[gsea] {} pathways tested, {:.1} ms total ({:.2} ms/pathway)",
        results.pathways.len(),
        results.timings.total_ms,
        results.timings.per_pathway_ms,
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Core GSEA engine
// ---------------------------------------------------------------------------

/// Run preranked GSEA across all gene sets.
///
/// `ranked_genes` must be sorted descending by statistic.
/// Gene sets outside `[min_size, max_size]` overlap with the ranked list are skipped.
pub fn run_gsea(
    ranked_genes: &[(String, f64)],
    gene_sets: &[(String, Vec<String>)],
    nperm: usize,
    min_size: usize,
    max_size: usize,
) -> GseaResults {
    let t0 = Instant::now();

    let n_total = ranked_genes.len();

    // 1. Build gene -> rank index map for O(1) lookup.
    let gene_index: HashMap<&str, usize> = ranked_genes
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.as_str(), i))
        .collect();

    // 2. Extract absolute statistics vector (rank-ordered).
    //    Pre-computing avoids repeated abs() calls inside the hot ES walk.
    let abs_stats: Vec<f64> = ranked_genes.iter().map(|(_, s)| s.abs()).collect();

    // 3. For each gene set, resolve hit indices against the ranked list.
    //    Collect (name, sorted_hit_indices) for sets that pass the size filter.
    let resolved: Vec<(&str, Vec<usize>)> = gene_sets
        .iter()
        .filter_map(|(name, genes)| {
            let mut hits: Vec<usize> = genes
                .iter()
                .filter_map(|g| gene_index.get(g.as_str()).copied())
                .collect();
            hits.sort_unstable();
            hits.dedup();
            if hits.len() >= min_size && hits.len() <= max_size {
                Some((name.as_str(), hits))
            } else {
                None
            }
        })
        .collect();

    // Gene names vector for leading edge annotation.
    let gene_names: Vec<&str> = ranked_genes.iter().map(|(n, _)| n.as_str()).collect();

    // 4. Parallel GSEA over pathways (rayon).
    //    Each pathway's permutation loop is fully independent.
    let mut pathway_results: Vec<PathwayResult> = resolved
        .par_iter()
        .map(|(name, hit_indices)| {
            let n_hits = hit_indices.len();

            // --- Observed ES ---
            let (es, le_indices) = compute_es(&abs_stats, hit_indices, n_total);

            // --- Null distribution via label permutation ---
            // For each permutation, randomly sample `n_hits` indices from 0..n_total
            // without replacement. This is equivalent to shuffling gene labels while
            // keeping the statistics in place (Fisher-Yates partial shuffle inside
            // rand::seq::index::sample).
            let mut rng = rand::rng();
            let mut null_es: Vec<f64> = Vec::with_capacity(nperm);

            for _ in 0..nperm {
                let mut perm_hits: Vec<usize> =
                    sample(&mut rng, n_total, n_hits).into_iter().collect();
                perm_hits.sort_unstable();

                let (es_null, _) = compute_es(&abs_stats, &perm_hits, n_total);
                null_es.push(es_null);
            }

            // --- NES and p-value ---
            let (nes, pvalue) = compute_significance(es, &null_es);

            // --- Leading edge gene names ---
            let leading_edge: Vec<String> = le_indices
                .iter()
                .map(|&i| gene_names[i].to_owned())
                .collect();

            PathwayResult {
                name: name.to_string(),
                es,
                nes,
                pvalue,
                padj: 0.0, // placeholder; filled after BH correction
                n_genes: n_hits,
                leading_edge,
            }
        })
        .collect();

    // 5. BH FDR correction across all tested pathways.
    let pvals: Vec<f64> = pathway_results.iter().map(|p| p.pvalue).collect();
    let padj = bh_adjust(&pvals);
    for (pr, adj) in pathway_results.iter_mut().zip(padj.iter()) {
        pr.padj = *adj;
    }

    // 6. Sort by padj ascending, then by |NES| descending for readability.
    pathway_results.sort_by(|a, b| {
        a.padj
            .partial_cmp(&b.padj)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.nes
                    .abs()
                    .partial_cmp(&a.nes.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let n_tested = pathway_results.len().max(1);

    GseaResults {
        pathways: pathway_results,
        timings: GseaTimings {
            total_ms: elapsed_ms,
            per_pathway_ms: elapsed_ms / n_tested as f64,
        },
    }
}

// ---------------------------------------------------------------------------
// ES computation (the running-sum walk)
// ---------------------------------------------------------------------------

/// Compute enrichment score via the running-sum walk (Subramanian et al. 2005).
///
/// `abs_stats` — pre-computed |statistic| for each rank position.
/// `hit_indices` — **sorted** indices of gene set members in the ranked list.
/// `n_total` — total number of genes in the ranked list.
///
/// Returns `(es, leading_edge_indices)`.
///
/// ES is the running sum value with the largest absolute deviation from zero.
/// Leading edge contains the hit indices encountered up to (and including)
/// the peak for positive ES, or from the peak to the end for negative ES.
fn compute_es(abs_stats: &[f64], hit_indices: &[usize], n_total: usize) -> (f64, Vec<usize>) {
    let n_hits = hit_indices.len();
    if n_hits == 0 {
        return (0.0, Vec::new());
    }

    // N_H = sum of |stat_i|^p for all hit genes. With p = 1 this is just the
    // sum of absolute statistics for the set members.
    let sum_hits: f64 = hit_indices.iter().map(|&i| abs_stats[i]).sum();

    // Guard: if every hit gene has stat = 0, the enrichment is undefined.
    if sum_hits == 0.0 {
        return (0.0, Vec::new());
    }

    let miss_penalty = 1.0 / (n_total - n_hits) as f64;

    // Walk through the ranked list. We use a pointer into the sorted
    // hit_indices slice so each step is O(1) — no HashSet needed.
    let mut hit_ptr: usize = 0;
    let mut running_sum: f64 = 0.0;
    let mut max_dev: f64 = 0.0;
    let mut min_dev: f64 = 0.0;
    let mut max_pos: usize = 0;
    let mut min_pos: usize = 0;

    for i in 0..n_total {
        if hit_ptr < n_hits && hit_indices[hit_ptr] == i {
            running_sum += abs_stats[i] / sum_hits;
            hit_ptr += 1;
        } else {
            running_sum -= miss_penalty;
        }

        if running_sum > max_dev {
            max_dev = running_sum;
            max_pos = i;
        }
        if running_sum < min_dev {
            min_dev = running_sum;
            min_pos = i;
        }
    }

    // ES = the deviation with the largest absolute value.
    let (es, peak_pos) = if max_dev.abs() >= min_dev.abs() {
        (max_dev, max_pos)
    } else {
        (min_dev, min_pos)
    };

    // Leading edge: for positive ES, the hit genes before (and at) the peak
    // are driving the enrichment at the top of the list. For negative ES,
    // the hit genes after (and at) the peak are driving enrichment at the bottom.
    let leading_edge: Vec<usize> = if es >= 0.0 {
        hit_indices
            .iter()
            .copied()
            .filter(|&i| i <= peak_pos)
            .collect()
    } else {
        hit_indices
            .iter()
            .copied()
            .filter(|&i| i >= peak_pos)
            .collect()
    };

    (es, leading_edge)
}

// ---------------------------------------------------------------------------
// Statistical significance
// ---------------------------------------------------------------------------

/// Compute Normalized Enrichment Score (NES) and empirical p-value.
///
/// NES = ES / mean(|ES_null|) for null values with the same sign as ES.
/// p-value = fraction of same-sign null values at least as extreme as ES.
///
/// Uses the Phipson & Smyth (2010) correction: add 1 to both numerator and
/// denominator to avoid reporting p = 0.
fn compute_significance(es: f64, null_es: &[f64]) -> (f64, f64) {
    if null_es.is_empty() || es == 0.0 {
        return (0.0, 1.0);
    }

    if es > 0.0 {
        let pos_nulls: Vec<f64> = null_es.iter().copied().filter(|&x| x > 0.0).collect();
        if pos_nulls.is_empty() {
            // Every null ES is negative/zero — the observed positive ES is
            // maximally significant given the permutation depth.
            return (es, 1.0 / (null_es.len() as f64 + 1.0));
        }
        let mean_pos: f64 = pos_nulls.iter().sum::<f64>() / pos_nulls.len() as f64;
        let nes = es / mean_pos;

        let n_extreme = pos_nulls.iter().filter(|&&x| x >= es).count();
        let pvalue = (n_extreme as f64 + 1.0) / (pos_nulls.len() as f64 + 1.0);

        (nes, pvalue)
    } else {
        let neg_nulls: Vec<f64> = null_es.iter().copied().filter(|&x| x < 0.0).collect();
        if neg_nulls.is_empty() {
            return (es, 1.0 / (null_es.len() as f64 + 1.0));
        }
        let mean_neg: f64 = neg_nulls.iter().map(|x| x.abs()).sum::<f64>() / neg_nulls.len() as f64;
        let nes = es / mean_neg; // negative / positive => negative NES

        let n_extreme = neg_nulls.iter().filter(|&&x| x <= es).count();
        let pvalue = (n_extreme as f64 + 1.0) / (neg_nulls.len() as f64 + 1.0);

        (nes, pvalue)
    }
}

// ---------------------------------------------------------------------------
// Multiple testing correction
// ---------------------------------------------------------------------------

/// Benjamini-Hochberg FDR adjustment.
///
/// Input: raw p-values in the same order as the pathway results.
/// Output: adjusted p-values, same length and order as input.
///
/// Procedure:
///   1. Sort p-values ascending, preserving original indices.
///   2. Apply BH formula: padj_i = p_i * n / rank_i  (rank is 1-based).
///   3. Enforce monotonicity from right to left (cumulative min).
///   4. Cap at 1.0.
fn bh_adjust(pvalues: &[f64]) -> Vec<f64> {
    let n = pvalues.len();
    if n == 0 {
        return Vec::new();
    }

    // (original_index, pvalue), sorted ascending by pvalue.
    let mut indexed: Vec<(usize, f64)> = pvalues.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut adjusted = vec![0.0_f64; n];
    let mut cummin = f64::INFINITY;

    // Walk from largest rank to smallest, enforcing monotonicity.
    for i in (0..n).rev() {
        let (orig_idx, pval) = indexed[i];
        let rank = i + 1; // 1-based
        let raw_adj = (pval * n as f64) / rank as f64;
        cummin = cummin.min(raw_adj).min(1.0);
        adjusted[orig_idx] = cummin;
    }

    adjusted
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

/// Write GSEA results to CSV.
///
/// Columns: pathway, es, nes, pvalue, padj, n_genes, leading_edge.
/// Leading edge genes are semicolon-separated within the column.
pub fn write_gsea_csv(results: &GseaResults, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)
        .map_err(|e| format!("Cannot create GSEA output '{}': {}", path, e))?;

    wtr.write_record([
        "pathway",
        "es",
        "nes",
        "pvalue",
        "padj",
        "n_genes",
        "leading_edge",
    ])
    .map_err(|e| format!("Write error to '{}': {}", path, e))?;

    for pr in &results.pathways {
        let le_str = pr.leading_edge.join(";");
        wtr.write_record([
            &pr.name,
            &format!("{:.6}", pr.es),
            &format!("{:.6}", pr.nes),
            &format!("{:.6e}", pr.pvalue),
            &format!("{:.6e}", pr.padj),
            &pr.n_genes.to_string(),
            &le_str,
        ])
        .map_err(|e| format!("Write error to '{}': {}", path, e))?;
    }

    wtr.flush()
        .map_err(|e| format!("Flush error for '{}': {}", path, e))?;

    eprintln!(
        "[gsea] wrote {} pathway results to '{}'",
        results.pathways.len(),
        path
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic tiny example: 10 genes, set members at top => positive ES.
    #[test]
    fn test_es_positive_enrichment() {
        let abs_stats: Vec<f64> = vec![10.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2];
        let hit_indices: Vec<usize> = vec![0, 1, 3];
        let n_total = 10;

        let (es, le) = compute_es(&abs_stats, &hit_indices, n_total);

        assert!(es > 0.0, "Expected positive ES, got {}", es);
        assert!(!le.is_empty(), "Leading edge should not be empty");
    }

    /// Set members at the bottom of the list => negative ES.
    #[test]
    fn test_es_negative_enrichment() {
        let abs_stats: Vec<f64> = vec![10.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2];
        let hit_indices: Vec<usize> = vec![7, 8, 9];
        let n_total = 10;

        let (es, le) = compute_es(&abs_stats, &hit_indices, n_total);

        assert!(es < 0.0, "Expected negative ES, got {}", es);
        assert!(!le.is_empty());
    }

    /// Empty hit set => ES = 0.
    #[test]
    fn test_es_empty_set() {
        let abs_stats: Vec<f64> = vec![10.0, 5.0, 1.0];
        let (es, le) = compute_es(&abs_stats, &[], 3);
        assert_eq!(es, 0.0);
        assert!(le.is_empty());
    }

    /// Verify the running sum returns to zero at the end of the walk.
    /// This is a mathematical invariant: sum of increments = 1, sum of decrements = 1.
    #[test]
    fn test_es_running_sum_returns_to_zero() {
        let abs_stats: Vec<f64> = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let hit_indices: Vec<usize> = vec![1, 3];
        let n_total = 5;
        let n_hits = 2;

        let sum_hits: f64 = hit_indices.iter().map(|&i| abs_stats[i]).sum();
        let miss_penalty = 1.0 / (n_total - n_hits) as f64;

        let mut running_sum = 0.0;
        let mut hit_ptr = 0;
        for i in 0..n_total {
            if hit_ptr < n_hits && hit_indices[hit_ptr] == i {
                running_sum += abs_stats[i] / sum_hits;
                hit_ptr += 1;
            } else {
                running_sum -= miss_penalty;
            }
        }

        assert!(
            running_sum.abs() < 1e-10,
            "Running sum should return to ~0, got {}",
            running_sum
        );
    }

    /// BH adjustment: single p-value should be unchanged.
    #[test]
    fn test_bh_single() {
        let adj = bh_adjust(&[0.05]);
        assert_eq!(adj.len(), 1);
        assert!((adj[0] - 0.05).abs() < 1e-10);
    }

    /// BH adjustment: known multi-p-value example.
    #[test]
    fn test_bh_multiple() {
        let pvals = vec![0.01, 0.04, 0.03, 0.20];
        let adj = bh_adjust(&pvals);

        assert_eq!(adj.len(), 4);

        // Adjusted values must be >= raw values.
        for (raw, adjusted) in pvals.iter().zip(adj.iter()) {
            assert!(*adjusted >= *raw - 1e-10, "adj {} < raw {}", adjusted, raw);
        }

        // Adjusted values must be <= 1.0.
        for a in &adj {
            assert!(*a <= 1.0 + 1e-10, "adj {} > 1.0", a);
        }

        // The smallest raw p-value (0.01, index 0) should have the smallest adj.
        assert!(adj[0] <= adj[1] + 1e-10);
        assert!(adj[0] <= adj[2] + 1e-10);
        assert!(adj[0] <= adj[3] + 1e-10);
    }

    /// BH adjustment preserves monotonicity in sorted order.
    #[test]
    fn test_bh_monotonicity() {
        let pvals = vec![0.001, 0.01, 0.02, 0.04, 0.05, 0.10, 0.50];
        let adj = bh_adjust(&pvals);

        // When input is already sorted ascending, output should also be
        // non-decreasing (monotonicity guarantee of BH).
        for i in 1..adj.len() {
            assert!(
                adj[i] >= adj[i - 1] - 1e-10,
                "Monotonicity violated: adj[{}]={} < adj[{}]={}",
                i,
                adj[i],
                i - 1,
                adj[i - 1]
            );
        }
    }

    /// BH on empty input.
    #[test]
    fn test_bh_empty() {
        let adj = bh_adjust(&[]);
        assert!(adj.is_empty());
    }

    /// Significance computation: strong positive ES with weak nulls => small p.
    #[test]
    fn test_significance_strong_positive() {
        let es = 0.8;
        let null_es: Vec<f64> = (0..1000).map(|i| 0.1 + 0.001 * i as f64).collect();
        let (nes, pval) = compute_significance(es, &null_es);
        assert!(nes > 0.0);
        assert!(pval < 0.5, "Expected small p-value, got {}", pval);
    }

    /// Significance: strong negative ES.
    #[test]
    fn test_significance_strong_negative() {
        let es = -0.7;
        let null_es: Vec<f64> = (0..1000).map(|i| -0.1 - 0.0005 * i as f64).collect();
        let (nes, pval) = compute_significance(es, &null_es);
        assert!(nes < 0.0, "Expected negative NES, got {}", nes);
        assert!(pval < 1.0);
    }

    /// Significance: ES = 0 => NES = 0, p = 1.
    #[test]
    fn test_significance_zero_es() {
        let (nes, pval) = compute_significance(0.0, &[0.1, -0.2, 0.3]);
        assert_eq!(nes, 0.0);
        assert_eq!(pval, 1.0);
    }

    /// Significance: empty null distribution => NES = 0, p = 1.
    #[test]
    fn test_significance_empty_null() {
        let (nes, pval) = compute_significance(0.5, &[]);
        assert_eq!(nes, 0.0);
        assert_eq!(pval, 1.0);
    }

    /// Phipson-Smyth correction: p-value should never be exactly zero.
    #[test]
    fn test_pvalue_never_zero() {
        let es = 100.0; // absurdly large
        let null_es: Vec<f64> = vec![0.01; 10000]; // all nulls far below
        let (_nes, pval) = compute_significance(es, &null_es);
        assert!(
            pval > 0.0,
            "p-value must never be exactly 0 (Phipson-Smyth), got {}",
            pval
        );
    }

    /// Full run_gsea on a small synthetic dataset: enriched set at top.
    #[test]
    fn test_run_gsea_smoke() {
        let ranked: Vec<(String, f64)> = (0..100)
            .map(|i| (format!("GENE{}", i), 50.0 - i as f64))
            .collect();

        let gene_sets: Vec<(String, Vec<String>)> = vec![(
            "TOP_SET".to_string(),
            (0..20).map(|i| format!("GENE{}", i)).collect(),
        )];

        let results = run_gsea(&ranked, &gene_sets, 1000, 5, 500);

        assert_eq!(results.pathways.len(), 1);
        let pr = &results.pathways[0];
        assert_eq!(pr.name, "TOP_SET");
        assert!(pr.es > 0.0, "Expected positive ES for top-enriched set");
        assert!(pr.nes > 0.0);
        assert!(
            pr.pvalue < 0.05,
            "Expected significant p-value, got {}",
            pr.pvalue
        );
        assert_eq!(pr.n_genes, 20);
        assert!(!pr.leading_edge.is_empty());
    }

    /// run_gsea with a set enriched at the bottom => negative ES/NES.
    #[test]
    fn test_run_gsea_bottom_enrichment() {
        let ranked: Vec<(String, f64)> = (0..100)
            .map(|i| (format!("GENE{}", i), 50.0 - i as f64))
            .collect();

        let gene_sets: Vec<(String, Vec<String>)> = vec![(
            "BOTTOM_SET".to_string(),
            (80..100).map(|i| format!("GENE{}", i)).collect(),
        )];

        let results = run_gsea(&ranked, &gene_sets, 1000, 5, 500);

        assert_eq!(results.pathways.len(), 1);
        let pr = &results.pathways[0];
        assert!(
            pr.es < 0.0,
            "Expected negative ES for bottom set, got {}",
            pr.es
        );
        assert!(pr.nes < 0.0);
    }

    /// Pathways outside the size bounds are filtered out.
    #[test]
    fn test_size_filter() {
        let ranked: Vec<(String, f64)> = (0..50)
            .map(|i| (format!("G{}", i), 25.0 - i as f64))
            .collect();

        let gene_sets: Vec<(String, Vec<String>)> = vec![
            // Too small: 2 genes, min_size = 5.
            ("TINY".to_string(), vec!["G0".to_string(), "G1".to_string()]),
            // Just right.
            (
                "OK".to_string(),
                (0..10).map(|i| format!("G{}", i)).collect(),
            ),
            // Too large: max_size = 15, set has 20 overlapping genes.
            (
                "HUGE".to_string(),
                (0..20).map(|i| format!("G{}", i)).collect(),
            ),
        ];

        let results = run_gsea(&ranked, &gene_sets, 100, 5, 15);

        assert_eq!(results.pathways.len(), 1);
        assert_eq!(results.pathways[0].name, "OK");
    }

    /// Gene set members not present in the ranked list are silently ignored.
    #[test]
    fn test_missing_genes_ignored() {
        let ranked: Vec<(String, f64)> = (0..50)
            .map(|i| (format!("G{}", i), 25.0 - i as f64))
            .collect();

        // 10 genes in the ranked list + 5 that don't exist.
        let mut genes: Vec<String> = (0..10).map(|i| format!("G{}", i)).collect();
        genes.extend((100..105).map(|i| format!("G{}", i)));

        let gene_sets = vec![("MIXED".to_string(), genes)];

        let results = run_gsea(&ranked, &gene_sets, 100, 5, 500);

        assert_eq!(results.pathways.len(), 1);
        // Only the 10 genes found in the ranked list should count.
        assert_eq!(results.pathways[0].n_genes, 10);
    }

    /// Multiple pathways: BH correction should make padj >= pvalue.
    #[test]
    fn test_multiple_pathways_bh() {
        let ranked: Vec<(String, f64)> = (0..200)
            .map(|i| (format!("GENE{}", i), 100.0 - i as f64))
            .collect();

        let gene_sets: Vec<(String, Vec<String>)> = vec![
            (
                "SET_A".to_string(),
                (0..20).map(|i| format!("GENE{}", i)).collect(),
            ),
            (
                "SET_B".to_string(),
                (50..70).map(|i| format!("GENE{}", i)).collect(),
            ),
            (
                "SET_C".to_string(),
                (180..200).map(|i| format!("GENE{}", i)).collect(),
            ),
        ];

        let results = run_gsea(&ranked, &gene_sets, 500, 5, 500);

        assert_eq!(results.pathways.len(), 3);
        for pr in &results.pathways {
            assert!(
                pr.padj >= pr.pvalue - 1e-10,
                "padj ({}) should be >= pvalue ({}) for {}",
                pr.padj,
                pr.pvalue,
                pr.name
            );
        }
    }

    /// Timings struct is populated.
    #[test]
    fn test_timings_populated() {
        let ranked: Vec<(String, f64)> = (0..50)
            .map(|i| (format!("G{}", i), 25.0 - i as f64))
            .collect();
        let gene_sets = vec![(
            "S".to_string(),
            (0..10).map(|i| format!("G{}", i)).collect(),
        )];

        let results = run_gsea(&ranked, &gene_sets, 100, 5, 500);
        assert!(results.timings.total_ms >= 0.0);
        assert!(results.timings.per_pathway_ms >= 0.0);
    }
}
