/// Leiden community detection for single-cell / spatial transcriptomics graphs.
///
/// Implements the Leiden algorithm (Traag, Waltman, van Eck, 2019) with:
///   - Fast local moving phase (queue-based, O(m) per pass)
///   - Refinement phase (singleton merging within partition communities)
///   - Graph aggregation for multi-level recursion
///
/// Designed to consume kNN graph CSV output from `rustpipe knn`.
/// The CSR graph from `crate::graph::CsrGraph` is cache-friendly and lets
/// us parallelize the refinement phase with rayon.
///
/// Performance targets: 5-10x faster than igraph/leidenalg on 100K cells.
use crate::graph::CsrGraph;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// =====================================================================
//  Result types
// =====================================================================

pub struct LeidenResult {
    pub communities: Vec<usize>,
    pub n_communities: usize,
    pub modularity: f64,
    #[allow(dead_code)]
    pub n_iterations: usize,
    #[allow(dead_code)]
    pub timings: LeidenTimings,
}

pub struct LeidenTimings {
    #[allow(dead_code)]
    pub total_ms: f64,
    #[allow(dead_code)]
    pub move_nodes_ms: f64,
    #[allow(dead_code)]
    pub refine_ms: f64,
    #[allow(dead_code)]
    pub aggregate_ms: f64,
}

// =====================================================================
//  Phase 1: Fast local moving
// =====================================================================

/// Queue-based fast local moving of nodes between communities.
///
/// Maintains `community_totals[c]` for O(1) lookup of the total node
/// strength of each community.  Uses an `in_queue` bitset to avoid
/// duplicate queue entries.
///
/// Returns `true` if any node was moved.
fn move_nodes_fast(
    graph: &CsrGraph,
    communities: &mut [usize],
    community_totals: &mut Vec<f64>,
    resolution: f64,
    rng: &mut StdRng,
) -> bool {
    let n = graph.n_nodes;
    let m = graph.total_weight;
    if m < 1e-14 {
        return false;
    }
    let m2 = 2.0 * m;
    let inv_m = 1.0 / m;
    let inv_m2_sq = 1.0 / (m2 * m2);

    // Random permutation for initial queue order.
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(rng);

    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);
    let mut in_queue = vec![true; n];
    for &node in &order {
        queue.push_back(node);
    }

    let mut any_moved = false;

    // Temporary storage for per-community edge weights from node v.
    // Reused across iterations to avoid repeated allocation.
    let mut neighbor_comm_weights: HashMap<usize, f64> = HashMap::with_capacity(64);

    while let Some(v) = queue.pop_front() {
        in_queue[v] = false;

        let c_v = communities[v];
        let s_v = graph.node_strengths[v];

        // Collect sum of edge weights from v to each neighboring community.
        neighbor_comm_weights.clear();
        let range = graph.neighbor_range(v);
        for idx in range {
            let c_neighbor = communities[graph.targets[idx]];
            *neighbor_comm_weights.entry(c_neighbor).or_insert(0.0) += graph.weights[idx];
        }

        // Weight from v to its own community.
        let k_v_to_cv = neighbor_comm_weights.get(&c_v).copied().unwrap_or(0.0);

        // Total strength of v's current community (excluding v itself for the
        // null-model term).
        let sigma_tot_cv_minus_sv = community_totals[c_v] - s_v;

        // Cost of removing v from its current community.
        // delta_remove = -k_v_to_cv / m + gamma * s_v * (S_cv - s_v) / (2m)^2
        let delta_remove =
            -k_v_to_cv * inv_m + resolution * s_v * sigma_tot_cv_minus_sv * inv_m2_sq;

        let mut best_community = c_v;
        let mut best_delta = 0.0f64;

        for (&c, &k_v_to_c) in &neighbor_comm_weights {
            if c == c_v {
                continue;
            }

            let sigma_tot_c = community_totals[c];

            // Gain of adding v to community c.
            // delta_add = k_v_to_c / m - gamma * s_v * S_c / (2m)^2
            let delta_add = k_v_to_c * inv_m - resolution * s_v * sigma_tot_c * inv_m2_sq;

            let delta_q = delta_remove + delta_add;

            // Numerical stability: require improvement > epsilon.
            // Tie-breaking: prefer lower community ID for determinism.
            if delta_q > best_delta + 1e-12 || (delta_q > best_delta - 1e-12 && c < best_community)
            {
                best_delta = delta_q;
                best_community = c;
            }
        }

        if best_community != c_v && best_delta > 1e-12 {
            // Move v from c_v to best_community.
            communities[v] = best_community;
            community_totals[c_v] -= s_v;
            community_totals[best_community] += s_v;
            any_moved = true;

            // Add neighbors not in the new community back to the queue.
            let range = graph.neighbor_range(v);
            for idx in range {
                let neighbor = graph.targets[idx];
                if communities[neighbor] != best_community && !in_queue[neighbor] {
                    queue.push_back(neighbor);
                    in_queue[neighbor] = true;
                }
            }
        }
    }

    any_moved
}

// =====================================================================
//  Phase 2: Refinement
// =====================================================================

/// Refine the partition by checking for disconnected communities and
/// merging singletons within each partition community.
///
/// For each community from Phase 1, starts with singletons and greedily
/// merges nodes within the same partition community if it improves modularity.
///
/// Parallelized over communities with rayon since they are independent.
fn refine_partition(
    graph: &CsrGraph,
    partition: &[usize],
    resolution: f64,
    rng_seed: u64,
) -> Vec<usize> {
    let n = graph.n_nodes;
    let m = graph.total_weight;
    if m < 1e-14 {
        return (0..n).collect();
    }
    let inv_m = 1.0 / m;
    let m2 = 2.0 * m;
    let inv_m2_sq = 1.0 / (m2 * m2);

    // Group nodes by their Phase 1 community.
    let n_partition = partition.iter().copied().max().unwrap_or(0) + 1;
    let mut community_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_partition];
    for (node, &c) in partition.iter().enumerate() {
        community_nodes[c].push(node);
    }

    // Process each partition community independently.
    // Returns Vec<(global_node_id, local_subcommunity_id)>.
    let refined_local: Vec<Vec<(usize, usize)>> = community_nodes
        .par_iter()
        .enumerate()
        .map(|(comm_idx, nodes)| {
            if nodes.len() <= 1 {
                return nodes.iter().map(|&n| (n, 0)).collect();
            }

            let mut local_rng = StdRng::seed_from_u64(rng_seed.wrapping_add(comm_idx as u64));

            // Map global node IDs to local indices for this community.
            let node_to_local: HashMap<usize, usize> =
                nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

            // Each node starts in its own singleton subcommunity.
            let mut sub_comm: Vec<usize> = (0..nodes.len()).collect();
            let mut sub_totals: Vec<f64> = nodes.iter().map(|&n| graph.node_strengths[n]).collect();

            // Process nodes in random order.
            let mut order: Vec<usize> = (0..nodes.len()).collect();
            order.shuffle(&mut local_rng);

            // Temporary map for per-subcommunity edge weights.
            let mut neighbor_sub: HashMap<usize, f64> = HashMap::with_capacity(32);

            for &local_v in &order {
                let v = nodes[local_v];
                let c_v = sub_comm[local_v];
                let s_v = graph.node_strengths[v];

                // Collect edge weights to subcommunities within this partition community.
                neighbor_sub.clear();
                let range = graph.neighbor_range(v);
                for idx in range {
                    let neighbor = graph.targets[idx];
                    if let Some(&local_n) = node_to_local.get(&neighbor) {
                        let sc = sub_comm[local_n];
                        *neighbor_sub.entry(sc).or_insert(0.0) += graph.weights[idx];
                    }
                }

                let k_v_to_cv = neighbor_sub.get(&c_v).copied().unwrap_or(0.0);
                let sigma_tot_cv_minus_sv = sub_totals[c_v] - s_v;

                let delta_remove =
                    -k_v_to_cv * inv_m + resolution * s_v * sigma_tot_cv_minus_sv * inv_m2_sq;

                let mut best_sub = c_v;
                let mut best_delta = 0.0f64;

                for (&sc, &k_v_to_sc) in &neighbor_sub {
                    if sc == c_v {
                        continue;
                    }
                    let sigma_tot_sc = sub_totals[sc];
                    let delta_add = k_v_to_sc * inv_m - resolution * s_v * sigma_tot_sc * inv_m2_sq;
                    let delta_q = delta_remove + delta_add;

                    if delta_q > best_delta + 1e-12 {
                        best_delta = delta_q;
                        best_sub = sc;
                    }
                }

                if best_sub != c_v && best_delta > 1e-12 {
                    sub_comm[local_v] = best_sub;
                    sub_totals[c_v] -= s_v;
                    sub_totals[best_sub] += s_v;
                }
            }

            nodes
                .iter()
                .enumerate()
                .map(|(local_i, &node)| (node, sub_comm[local_i]))
                .collect()
        })
        .collect();

    // Map subcommunity IDs to globally unique IDs.
    let mut refined = vec![0usize; n];
    let mut global_id = 0usize;

    for local_results in &refined_local {
        let mut local_to_global: HashMap<usize, usize> = HashMap::new();
        for &(node, local_sub) in local_results {
            let gid = *local_to_global.entry(local_sub).or_insert_with(|| {
                let id = global_id;
                global_id += 1;
                id
            });
            refined[node] = gid;
        }
    }

    refined
}

// =====================================================================
//  Compact community IDs to contiguous range [0, n)
// =====================================================================

fn compact_communities(communities: &mut [usize]) -> usize {
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0usize;

    for c in communities.iter_mut() {
        let new_id = *mapping.entry(*c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        *c = new_id;
    }

    next_id
}

// =====================================================================
//  Main Leiden entry point
// =====================================================================

/// Run the Leiden algorithm on a CSR graph.
///
/// Parameters:
/// - `graph`: the input graph (from `crate::graph::CsrGraph`)
/// - `resolution`: modularity resolution gamma, typically 1.0
/// - `n_iterations`: maximum number of outer iterations, typically 10
/// - `seed`: random seed for reproducibility
pub fn run_leiden(
    graph: &CsrGraph,
    resolution: f64,
    n_iterations: usize,
    seed: u64,
) -> LeidenResult {
    let total_t0 = Instant::now();
    let mut move_ms = 0.0f64;
    let mut refine_ms = 0.0f64;
    let mut aggregate_ms = 0.0f64;

    let n = graph.n_nodes;

    if n == 0 {
        return LeidenResult {
            communities: Vec::new(),
            n_communities: 0,
            modularity: 0.0,
            n_iterations: 0,
            timings: LeidenTimings {
                total_ms: 0.0,
                move_nodes_ms: 0.0,
                refine_ms: 0.0,
                aggregate_ms: 0.0,
            },
        };
    }

    // Initialize: each node in its own community.
    let mut communities: Vec<usize> = (0..n).collect();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut actual_iterations = 0usize;

    for iter in 0..n_iterations {
        actual_iterations = iter + 1;

        // --- Phase 1: Fast local moving ---
        let t0 = Instant::now();

        // Build community_totals from current partition.
        let n_comm_current = communities.iter().copied().max().unwrap_or(0) + 1;
        let totals_len = n_comm_current.max(n);
        let mut community_totals = vec![0.0f64; totals_len];
        for node in 0..n {
            community_totals[communities[node]] += graph.node_strengths[node];
        }

        let changed = move_nodes_fast(
            graph,
            &mut communities,
            &mut community_totals,
            resolution,
            &mut rng,
        );
        move_ms += t0.elapsed().as_secs_f64() * 1000.0;

        if !changed {
            eprintln!("[leiden] iteration {}: no moves, converged", iter + 1);
            break;
        }

        // Compact community IDs after moving phase.
        let n_comm = compact_communities(&mut communities);

        eprintln!(
            "[leiden] iteration {}: {} communities after local moving",
            iter + 1,
            n_comm
        );

        // --- Phase 2: Refinement ---
        let t0 = Instant::now();
        let partition = communities.clone();
        let refined = refine_partition(
            graph,
            &partition,
            resolution,
            seed.wrapping_add(iter as u64),
        );
        refine_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // --- Phase 3: Aggregation ---
        let t0 = Instant::now();
        let mut refined_compact = refined;
        let n_refined = compact_communities(&mut refined_compact);

        if n_refined >= n {
            // No aggregation possible (singletons), stop.
            communities = refined_compact;
            aggregate_ms += t0.elapsed().as_secs_f64() * 1000.0;
            break;
        }

        // Build aggregated graph from the refined partition.
        let agg_graph = graph.aggregate(&refined_compact);

        // Map the Phase 1 partition to aggregated graph level.
        // Each aggregated node = a refined subcommunity. Map it to the
        // Phase 1 community of any member node (they all have the same
        // Phase 1 community by construction of the refinement).
        let mut agg_partition: Vec<usize> = vec![0; n_refined];
        for node in 0..n {
            agg_partition[refined_compact[node]] = partition[node];
        }
        compact_communities(&mut agg_partition);

        // Run local moving on the aggregated graph.
        let n_agg_comm = agg_partition.iter().copied().max().unwrap_or(0) + 1;
        let agg_totals_len = n_agg_comm.max(agg_graph.n_nodes);
        let mut agg_totals = vec![0.0f64; agg_totals_len];
        for node in 0..agg_graph.n_nodes {
            agg_totals[agg_partition[node]] += agg_graph.node_strengths[node];
        }

        move_nodes_fast(
            &agg_graph,
            &mut agg_partition,
            &mut agg_totals,
            resolution,
            &mut rng,
        );

        // Map aggregated communities back to original nodes.
        for node in 0..n {
            communities[node] = agg_partition[refined_compact[node]];
        }
        compact_communities(&mut communities);

        aggregate_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }

    compact_communities(&mut communities);
    let n_communities = communities.iter().copied().max().map_or(0, |m| m + 1);
    let modularity = graph.modularity(&communities, resolution);
    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "[leiden] done: {} communities, Q={:.6}, {} iterations in {:.1}ms \
         (move={:.1}, refine={:.1}, agg={:.1})",
        n_communities, modularity, actual_iterations, total_ms, move_ms, refine_ms, aggregate_ms
    );

    LeidenResult {
        communities,
        n_communities,
        modularity,
        n_iterations: actual_iterations,
        timings: LeidenTimings {
            total_ms,
            move_nodes_ms: move_ms,
            refine_ms,
            aggregate_ms,
        },
    }
}

// =====================================================================
//  CLI entry point
// =====================================================================

/// Entry point called from main.rs.
///
/// 1. Builds CsrGraph from kNN CSV
/// 2. Runs Leiden clustering
/// 3. Writes output CSV with columns: cell_id,cluster
/// 4. Prints summary
pub fn run(
    input: &str,
    resolution: f64,
    n_iterations: usize,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let graph = CsrGraph::from_knn_csv(input)?;

    eprintln!(
        "[leiden] running: resolution={}, max_iterations={}, nodes={}, edges={}",
        resolution, n_iterations, graph.n_nodes, graph.n_edges
    );

    let result = run_leiden(&graph, resolution, n_iterations, 42);

    // Write output CSV.
    let mut file = File::create(output)
        .map_err(|e| format!("Cannot create output file '{}': {}", output, e))?;
    writeln!(file, "cell_id,cluster")?;
    for (node, &cluster) in result.communities.iter().enumerate() {
        writeln!(file, "{},{}", node, cluster)?;
    }

    eprintln!(
        "[leiden] wrote {} assignments to '{}'",
        result.communities.len(),
        output
    );
    eprintln!(
        "[leiden] summary: n_clusters={}, modularity={:.6}, iterations={}, total_time={:.1}ms",
        result.n_communities, result.modularity, result.n_iterations, result.timings.total_ms,
    );
    eprintln!(
        "[leiden] timings: move_nodes={:.1}ms, refine={:.1}ms, aggregate={:.1}ms",
        result.timings.move_nodes_ms, result.timings.refine_ms, result.timings.aggregate_ms,
    );

    Ok(())
}

// =====================================================================
//  Unit tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Two 5-cliques connected by a single weak edge.
    /// Expected: 2 communities.
    fn two_cliques_graph() -> CsrGraph {
        let edges: Vec<(usize, usize, f64)> = {
            let mut e = Vec::new();
            // Clique A: nodes 0-4
            for i in 0..5usize {
                for j in (i + 1)..5 {
                    e.push((i, j, 1.0));
                }
            }
            // Clique B: nodes 5-9
            for i in 5..10usize {
                for j in (i + 1)..10 {
                    e.push((i, j, 1.0));
                }
            }
            // Weak bridge
            e.push((4, 5, 0.1));
            e
        };
        CsrGraph::from_edge_list(&edges, 10)
    }

    /// Three triangles connected by weak bridges.
    fn three_triangles_graph() -> CsrGraph {
        let edges = vec![
            // Triangle 0
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
            // Triangle 1
            (3, 4, 1.0),
            (4, 5, 1.0),
            (3, 5, 1.0),
            // Triangle 2
            (6, 7, 1.0),
            (7, 8, 1.0),
            (6, 8, 1.0),
            // Weak bridges
            (2, 3, 0.05),
            (5, 6, 0.05),
        ];
        CsrGraph::from_edge_list(&edges, 9)
    }

    /// Single triangle (all nodes should be 1 community).
    fn single_triangle_graph() -> CsrGraph {
        CsrGraph::from_edge_list(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)], 3)
    }

    // -------------------------------------------------------------------
    //  Leiden algorithm tests
    // -------------------------------------------------------------------

    #[test]
    fn leiden_two_cliques_finds_two_communities() {
        let g = two_cliques_graph();
        let result = run_leiden(&g, 1.0, 10, 42);

        assert_eq!(
            result.n_communities, 2,
            "expected 2 communities, got {}",
            result.n_communities
        );

        // Clique A nodes should share a community.
        let c_a = result.communities[0];
        for i in 0..5 {
            assert_eq!(
                result.communities[i], c_a,
                "node {} not in same community as node 0",
                i
            );
        }
        // Clique B nodes should share a community.
        let c_b = result.communities[5];
        for i in 5..10 {
            assert_eq!(
                result.communities[i], c_b,
                "node {} not in same community as node 5",
                i
            );
        }
        assert_ne!(c_a, c_b, "cliques should be in different communities");
    }

    #[test]
    fn leiden_three_triangles_finds_three() {
        let g = three_triangles_graph();
        let result = run_leiden(&g, 1.0, 10, 42);

        assert_eq!(
            result.n_communities, 3,
            "expected 3 communities for three triangles, got {}",
            result.n_communities
        );
    }

    #[test]
    fn leiden_single_triangle_one_community() {
        let g = single_triangle_graph();
        let result = run_leiden(&g, 1.0, 10, 42);

        assert_eq!(
            result.n_communities, 1,
            "single triangle should be 1 community, got {}",
            result.n_communities
        );
    }

    #[test]
    fn leiden_modularity_positive_for_two_cliques() {
        let g = two_cliques_graph();
        let result = run_leiden(&g, 1.0, 10, 42);
        assert!(
            result.modularity > 0.0,
            "modularity should be positive, got {}",
            result.modularity
        );
    }

    #[test]
    fn leiden_modularity_nondecreasing_across_iterations() {
        let g = two_cliques_graph();
        let r1 = run_leiden(&g, 1.0, 1, 42);
        let r10 = run_leiden(&g, 1.0, 10, 42);
        assert!(
            r10.modularity >= r1.modularity - 1e-9,
            "modularity should not decrease: 1-iter={}, 10-iter={}",
            r1.modularity,
            r10.modularity
        );
    }

    #[test]
    fn leiden_terminates_within_max_iterations() {
        let g = two_cliques_graph();
        let result = run_leiden(&g, 1.0, 100, 42);
        assert!(
            result.n_iterations <= 100,
            "ran {} iterations, max was 100",
            result.n_iterations
        );
    }

    #[test]
    fn leiden_empty_graph() {
        let g = CsrGraph::from_edge_list(&[], 0);
        let result = run_leiden(&g, 1.0, 10, 42);
        assert_eq!(result.n_communities, 0);
        assert_eq!(result.communities.len(), 0);
    }

    #[test]
    fn leiden_single_node() {
        let g = CsrGraph::from_edge_list(&[], 1);
        let result = run_leiden(&g, 1.0, 10, 42);
        assert_eq!(result.n_communities, 1);
        assert_eq!(result.communities, vec![0]);
    }

    #[test]
    fn leiden_disconnected_components() {
        // Two isolated triangles -- should find 2 communities.
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (3, 5, 1.0),
        ];
        let g = CsrGraph::from_edge_list(&edges, 6);
        let result = run_leiden(&g, 1.0, 10, 42);

        assert_eq!(
            result.n_communities, 2,
            "disconnected components should be separate, got {}",
            result.n_communities
        );
        // Nodes 0-2 together.
        assert_eq!(result.communities[0], result.communities[1]);
        assert_eq!(result.communities[1], result.communities[2]);
        // Nodes 3-5 together.
        assert_eq!(result.communities[3], result.communities[4]);
        assert_eq!(result.communities[4], result.communities[5]);
        // Different groups.
        assert_ne!(result.communities[0], result.communities[3]);
    }

    #[test]
    fn leiden_deterministic_with_same_seed() {
        let g = two_cliques_graph();
        let r1 = run_leiden(&g, 1.0, 10, 12345);
        let r2 = run_leiden(&g, 1.0, 10, 12345);
        assert_eq!(
            r1.communities, r2.communities,
            "same seed should produce identical results"
        );
    }

    #[test]
    fn leiden_resolution_effect() {
        // Higher resolution should produce more (or equal) communities.
        let g = two_cliques_graph();
        let r_low = run_leiden(&g, 0.1, 10, 42);
        let r_high = run_leiden(&g, 5.0, 10, 42);
        assert!(
            r_high.n_communities >= r_low.n_communities,
            "higher resolution should give >= communities: low={}, high={}",
            r_low.n_communities,
            r_high.n_communities
        );
    }

    // -------------------------------------------------------------------
    //  CSV round-trip test
    // -------------------------------------------------------------------

    #[test]
    fn csv_round_trip() {
        let g = two_cliques_graph();
        let result = run_leiden(&g, 1.0, 10, 42);

        let tmp = "/tmp/rustpipe_leiden_test.csv";
        {
            let mut file = File::create(tmp).unwrap();
            writeln!(file, "cell_id,cluster").unwrap();
            for (node, &cluster) in result.communities.iter().enumerate() {
                writeln!(file, "{},{}", node, cluster).unwrap();
            }
        }

        let contents = std::fs::read_to_string(tmp).unwrap();
        let lines: Vec<&str> = contents.trim().lines().collect();
        assert_eq!(lines.len(), 11, "expected header + 10 data rows");
        assert_eq!(lines[0], "cell_id,cluster");

        for (i, line) in lines[1..].iter().enumerate() {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 2);
            let cell_id: usize = fields[0].parse().unwrap();
            assert_eq!(cell_id, i);
            let _cluster: usize = fields[1].parse().unwrap();
        }
    }

    // -------------------------------------------------------------------
    //  Timing fields are populated
    // -------------------------------------------------------------------

    #[test]
    fn timings_populated() {
        let g = two_cliques_graph();
        let result = run_leiden(&g, 1.0, 10, 42);
        assert!(result.timings.total_ms >= 0.0);
        assert!(result.timings.move_nodes_ms >= 0.0);
        assert!(result.timings.refine_ms >= 0.0);
        assert!(result.timings.aggregate_ms >= 0.0);
    }

    // -------------------------------------------------------------------
    //  Larger structured graph: ring of 8 cliques
    // -------------------------------------------------------------------

    #[test]
    fn leiden_ring_of_cliques() {
        // 8 cliques of 10 nodes each, connected in a ring by weak edges.
        let n_cliques = 8;
        let clique_size = 10;
        let n_nodes = n_cliques * clique_size;
        let mut edges = Vec::new();

        for c in 0..n_cliques {
            let base = c * clique_size;
            // Intra-clique: complete graph.
            for i in 0..clique_size {
                for j in (i + 1)..clique_size {
                    edges.push((base + i, base + j, 1.0));
                }
            }
            // Inter-clique bridge: last node of this clique -> first node of next.
            let next_base = ((c + 1) % n_cliques) * clique_size;
            edges.push((base + clique_size - 1, next_base, 0.05));
        }

        let g = CsrGraph::from_edge_list(&edges, n_nodes);
        let result = run_leiden(&g, 1.0, 10, 42);

        // Should find 8 communities (one per clique).
        assert_eq!(
            result.n_communities, n_cliques,
            "expected {} communities in ring-of-cliques, got {}",
            n_cliques, result.n_communities
        );

        // Each clique's nodes should all be in the same community.
        for c in 0..n_cliques {
            let base = c * clique_size;
            let comm = result.communities[base];
            for i in 1..clique_size {
                assert_eq!(
                    result.communities[base + i],
                    comm,
                    "node {} (clique {}) disagrees with node {} (same clique)",
                    base + i,
                    c,
                    base
                );
            }
        }
    }
}
