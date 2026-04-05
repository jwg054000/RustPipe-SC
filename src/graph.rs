/// Compressed Sparse Row (CSR) graph — weighted, undirected.
///
/// Standard sparse representation for Leiden clustering and modularity
/// computation. Each undirected edge (i, j, w) is stored as two directed
/// edges: i→j and j→i, both with weight w.
///
/// Layout invariant: within node i's adjacency slice, entries are sorted
/// ascending by target index. This is enforced by all constructors.
use std::fs::File;
use std::time::Instant;

// =====================================================================
//  Core struct
// =====================================================================

/// Compressed Sparse Row graph — weighted, undirected.
/// Standard format for sparse graph algorithms.
pub struct CsrGraph {
    pub n_nodes: usize,
    /// Total directed edges (each undirected edge counted twice).
    #[allow(dead_code)]
    pub n_edges: usize,
    /// Length n_nodes + 1. Node i's neighbors occupy
    /// targets[offsets[i]..offsets[i+1]].
    pub offsets: Vec<usize>,
    /// Target node for each directed edge.
    pub targets: Vec<usize>,
    /// Weight for each directed edge.
    pub weights: Vec<f64>,
    /// Sum of all edge weights / 2 (for modularity: m).
    pub total_weight: f64,
    /// Weighted degree of each node: sum of weights of incident edges.
    pub node_strengths: Vec<f64>,
}

// =====================================================================
//  Constructors
// =====================================================================

impl CsrGraph {
    /// Build CSR from a flat edge list of (source, target, weight) triples.
    ///
    /// The input triples are treated as undirected: for every (i, j, w) the
    /// reverse edge (j, i, w) is added as well. Duplicate directed edges
    /// (same source + target appearing more than once) are merged by summing
    /// their weights — this handles the symmetric case where both i→j and j→i
    /// are already present in the input.
    ///
    /// `n_nodes` must be > max node index in `edges`.
    pub fn from_edge_list(edges: &[(usize, usize, f64)], n_nodes: usize) -> Self {
        // Expand to directed pairs, then deduplicate by summing weights.
        // Use a Vec that we sort; this avoids a HashMap allocation and is
        // cache-friendly for the downstream CSR build.
        let mut directed: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len() * 2);

        for &(src, tgt, w) in edges {
            directed.push((src, tgt, w));
            if src != tgt {
                directed.push((tgt, src, w));
            }
        }

        // Sort by (source, target) so identical pairs are adjacent.
        directed.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Deduplicate by summing weights for identical (src, tgt) pairs.
        let mut deduped: Vec<(usize, usize, f64)> = Vec::with_capacity(directed.len());
        for (src, tgt, w) in directed {
            if let Some(last) = deduped.last_mut() {
                if last.0 == src && last.1 == tgt {
                    last.2 += w;
                    continue;
                }
            }
            deduped.push((src, tgt, w));
        }

        Self::build_from_sorted_directed(deduped, n_nodes)
    }

    /// Read the kNN CSV produced by `rustpipe knn` and build an undirected
    /// weighted CSR graph.
    ///
    /// CSV format (with header):
    /// ```text
    /// neighbor_1,distance_1,neighbor_2,distance_2,...,neighbor_k,distance_k
    /// ```
    /// Row index = cell index. Edge weight = 1 / (1 + d).
    ///
    /// When both i→j and j→i are present (typical for kNN graphs), the edge
    /// is kept once with the *maximum* weight (i.e., minimum distance).
    #[allow(dead_code)]
    pub fn from_knn_csv(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let t0 = Instant::now();

        let file =
            File::open(path).map_err(|e| format!("Cannot open kNN CSV '{}': {}", path, e))?;

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let header = rdr
            .headers()
            .map_err(|e| format!("Cannot read header in '{}': {}", path, e))?
            .clone();

        // Header has pairs: neighbor_1, distance_1, neighbor_2, distance_2, ...
        if header.is_empty() || header.len() % 2 != 0 {
            return Err(format!(
                "kNN CSV '{}': expected even number of columns (neighbor/distance pairs), got {}",
                path,
                header.len()
            )
            .into());
        }
        let k = header.len() / 2;

        // Collect raw (i, j, w) directed edges with max-weight deduplication.
        // We store as (min(i,j), max(i,j)) → max_weight to merge symmetric edges.
        // Use a sorted Vec for memory efficiency; HashMap would also work but
        // allocates more for large kNN graphs.
        let mut raw_edges: Vec<(usize, usize, f64)> = Vec::new();

        let mut n_nodes = 0usize;

        for (row_idx, result) in rdr.records().enumerate() {
            let record = result
                .map_err(|e| format!("Parse error in '{}' at row {}: {}", path, row_idx + 1, e))?;

            if record.len() != header.len() {
                return Err(format!(
                    "'{}' row {}: expected {} fields, got {}",
                    path,
                    row_idx + 2,
                    header.len(),
                    record.len()
                )
                .into());
            }

            let cell_i = row_idx;
            if cell_i >= n_nodes {
                n_nodes = cell_i + 1;
            }

            for ki in 0..k {
                let neighbor_col = ki * 2;
                let distance_col = ki * 2 + 1;

                let neighbor_j: usize = record[neighbor_col].trim().parse().map_err(|e| {
                    format!(
                        "'{}' row {} col {}: cannot parse neighbor index '{}': {}",
                        path,
                        row_idx + 2,
                        neighbor_col + 1,
                        &record[neighbor_col],
                        e
                    )
                })?;

                let distance: f64 = record[distance_col].trim().parse().map_err(|e| {
                    format!(
                        "'{}' row {} col {}: cannot parse distance '{}': {}",
                        path,
                        row_idx + 2,
                        distance_col + 1,
                        &record[distance_col],
                        e
                    )
                })?;

                // Track max node index to size the graph.
                if neighbor_j >= n_nodes {
                    n_nodes = neighbor_j + 1;
                }

                // Weight: closer neighbors get higher weight.
                let weight = 1.0 / (1.0 + distance);

                // Canonicalize so (lo, hi) — makes dedup symmetric.
                let (lo, hi) = if cell_i <= neighbor_j {
                    (cell_i, neighbor_j)
                } else {
                    (neighbor_j, cell_i)
                };

                raw_edges.push((lo, hi, weight));
            }
        }

        // Sort by canonical (lo, hi) so duplicates are adjacent.
        raw_edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Deduplicate: keep max weight for each undirected edge.
        let mut canonical: Vec<(usize, usize, f64)> = Vec::with_capacity(raw_edges.len());
        for (lo, hi, w) in raw_edges {
            if let Some(last) = canonical.last_mut() {
                if last.0 == lo && last.1 == hi {
                    if w > last.2 {
                        last.2 = w;
                    }
                    continue;
                }
            }
            canonical.push((lo, hi, w));
        }

        // Expand canonical undirected edges → directed pairs for CSR.
        let mut directed: Vec<(usize, usize, f64)> = Vec::with_capacity(canonical.len() * 2);
        for (lo, hi, w) in &canonical {
            directed.push((*lo, *hi, *w));
            if lo != hi {
                directed.push((*hi, *lo, *w));
            }
        }
        directed.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let graph = Self::build_from_sorted_directed(directed, n_nodes);

        eprintln!(
            "[graph] from_knn_csv: {} nodes, {} undirected edges built in {:.1}ms",
            graph.n_nodes,
            canonical.len(),
            t0.elapsed().as_secs_f64() * 1000.0
        );

        Ok(graph)
    }

    // ------------------------------------------------------------------
    //  Internal helper: build CSR from already-sorted directed edge list.
    //  Assumes sorted by (source, target) with no duplicates.
    // ------------------------------------------------------------------
    fn build_from_sorted_directed(directed: Vec<(usize, usize, f64)>, n_nodes: usize) -> Self {
        let n_edges = directed.len();

        let mut offsets = vec![0usize; n_nodes + 1];
        let mut targets = Vec::with_capacity(n_edges);
        let mut weights = Vec::with_capacity(n_edges);

        // Count out-degree for each node.
        for &(src, _, _) in &directed {
            offsets[src + 1] += 1;
        }
        // Prefix sum.
        for i in 0..n_nodes {
            offsets[i + 1] += offsets[i];
        }

        // Fill targets / weights in source order (already sorted).
        for (_, tgt, w) in directed {
            targets.push(tgt);
            weights.push(w);
        }

        // Compute node strengths and total weight.
        let mut node_strengths = vec![0.0f64; n_nodes];
        for (i, &w) in weights.iter().enumerate() {
            // targets[i] is the target, but we need to find the source.
            // We know offsets: binary search to find which node owns slot i.
            // But it's cheaper to iterate once over nodes.
            let _ = i; // will be computed below
            let _ = w;
        }
        // Compute node_strengths via offsets (avoids binary search).
        for node in 0..n_nodes {
            let start = offsets[node];
            let end = offsets[node + 1];
            let strength: f64 = weights[start..end].iter().sum();
            node_strengths[node] = strength;
        }

        // total_weight = sum of all directed weights / 2  (= sum of undirected weights).
        let total_weight: f64 = node_strengths.iter().sum::<f64>() / 2.0;

        CsrGraph {
            n_nodes,
            n_edges,
            offsets,
            targets,
            weights,
            total_weight,
            node_strengths,
        }
    }
}

// =====================================================================
//  Spatial graph construction
// =====================================================================

impl CsrGraph {
    /// Build a spatial neighbor graph from 2D coordinates using brute-force kNN.
    ///
    /// For Visium hexagonal grids, `n_neighbors=6` captures the lattice.
    /// Weight = 1.0 (binary) for standard Moran's I computation.
    #[allow(dead_code)]
    pub fn from_spatial_coords(
        coords: &[(f64, f64)],
        n_neighbors: usize,
        binary_weights: bool,
    ) -> Self {
        let n = coords.len();
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n * n_neighbors);

        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let dx = coords[i].0 - coords[j].0;
                    let dy = coords[i].1 - coords[j].1;
                    (j, (dx * dx + dy * dy).sqrt())
                })
                .collect();

            let k = n_neighbors.min(dists.len());
            dists.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            for &(j, d) in &dists[..k] {
                let w = if binary_weights { 1.0 } else { 1.0 / (1.0 + d) };
                edges.push((i, j, w));
            }
        }

        CsrGraph::from_edge_list(&edges, n)
    }
}

// =====================================================================
//  Graph traversal
// =====================================================================

impl CsrGraph {
    /// Index range of node `node`'s adjacency in `targets` / `weights`.
    ///
    /// Usage:
    /// ```
    /// let r = graph.neighbor_range(node);
    /// for (tgt, w) in graph.targets[r.clone()].iter().zip(graph.weights[r].iter()) { ... }
    /// ```
    #[inline(always)]
    pub fn neighbor_range(&self, node: usize) -> std::ops::Range<usize> {
        self.offsets[node]..self.offsets[node + 1]
    }

    /// Sum of edge weights from `node` to all nodes that belong to
    /// `community` in `communities`.
    ///
    /// This is the inner loop of the Leiden/Louvain move phase and is kept as
    /// tight as possible: a single linear scan over the node's adjacency slice.
    #[allow(dead_code)]
    #[inline]
    pub fn community_edge_weight(
        &self,
        node: usize,
        community: usize,
        communities: &[usize],
    ) -> f64 {
        let r = self.neighbor_range(node);
        let mut sum = 0.0f64;
        for (tgt, w) in self.targets[r.clone()].iter().zip(self.weights[r].iter()) {
            if communities[*tgt] == community {
                sum += w;
            }
        }
        sum
    }
}

// =====================================================================
//  Graph coarsening (aggregate)
// =====================================================================

impl CsrGraph {
    /// Build a coarser graph where each community becomes a single super-node.
    ///
    /// `communities[i]` = community index for node i. Community indices must
    /// span 0..n_communities (gaps are allowed; empty communities produce
    /// isolated nodes).
    ///
    /// Edge weight between communities A and B = sum of weights of all edges
    /// between nodes in A and nodes in B. Self-loops (intra-community edges)
    /// are preserved as self-loop edges on the super-node; they are counted
    /// once (not doubled) in the output because the source and target are the
    /// same super-node.
    pub fn aggregate(&self, communities: &[usize]) -> CsrGraph {
        assert_eq!(
            communities.len(),
            self.n_nodes,
            "communities length must equal n_nodes"
        );

        let n_communities = communities
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        // Accumulate inter-community edge weights.
        // Use a flat Vec indexed by (src_comm * n_communities + tgt_comm) for
        // dense graphs, which is fine when n_communities is in the hundreds.
        // For very large community counts a HashMap would be preferable, but
        // Leiden typically operates on O(n_cells) nodes → O(1..n_cells)
        // communities converging quickly.
        let matrix_size = n_communities * n_communities;
        let mut weight_matrix = vec![0.0f64; matrix_size];

        for src in 0..self.n_nodes {
            let c_src = communities[src];
            let r = self.neighbor_range(src);
            for (tgt, w) in self.targets[r.clone()].iter().zip(self.weights[r].iter()) {
                let c_tgt = communities[*tgt];
                // For directed storage: both directions are in the graph.
                // We accumulate the directed sum; when we later build the
                // undirected edge list we'll halve inter-community weights.
                weight_matrix[c_src * n_communities + c_tgt] += w;
            }
        }

        // Extract non-zero edges as undirected (lo, hi, w) canonical form.
        // Self-loops: weight_matrix[c][c] is the sum of all directed intra-
        // community edges, which double-counts (each undirected edge added as
        // i→j and j→i). Halve it to get the true intra-community sum.
        // Inter-community: weight_matrix[c1][c2] + weight_matrix[c2][c1]
        // = 2 * undirected_weight (symmetric by construction). Halve it.
        let mut edge_list: Vec<(usize, usize, f64)> = Vec::new();

        for c1 in 0..n_communities {
            // Self-loop.
            let self_w = weight_matrix[c1 * n_communities + c1] / 2.0;
            if self_w > 0.0 {
                edge_list.push((c1, c1, self_w));
            }
            // Inter-community (upper triangle only to avoid duplicates).
            for c2 in (c1 + 1)..n_communities {
                let w = (weight_matrix[c1 * n_communities + c2]
                    + weight_matrix[c2 * n_communities + c1])
                    / 2.0;
                if w > 0.0 {
                    edge_list.push((c1, c2, w));
                }
            }
        }

        // `from_edge_list` handles adding reverse directed edges and building CSR.
        CsrGraph::from_edge_list(&edge_list, n_communities)
    }
}

// =====================================================================
//  Modularity
// =====================================================================

impl CsrGraph {
    /// Compute modularity Q for a given community assignment.
    ///
    /// Formula (Newman-Girvan with resolution γ):
    ///   Q = (1/2m) * Σ_{edges (i,j)} [w_ij - γ * s_i * s_j / (2m)] * δ(c_i, c_j)
    ///
    /// where m = total_weight, s_i = node_strengths[i], γ = resolution.
    ///
    /// The sum is over directed edges (both directions of each undirected edge),
    /// so the leading 1/2m factor correctly normalizes.
    pub fn modularity(&self, communities: &[usize], resolution: f64) -> f64 {
        assert_eq!(communities.len(), self.n_nodes);

        let m2 = 2.0 * self.total_weight; // 2m
        if m2 == 0.0 {
            return 0.0;
        }

        let mut q = 0.0f64;

        for src in 0..self.n_nodes {
            let c_src = communities[src];
            let s_src = self.node_strengths[src];
            let r = self.neighbor_range(src);

            for (tgt, w) in self.targets[r.clone()].iter().zip(self.weights[r].iter()) {
                if communities[*tgt] == c_src {
                    let s_tgt = self.node_strengths[*tgt];
                    q += w - resolution * s_src * s_tgt / m2;
                }
            }
        }

        q / m2
    }
}

// =====================================================================
//  Unit tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Triangle graph: nodes 0, 1, 2 fully connected with unit weights.
    fn triangle_edges() -> Vec<(usize, usize, f64)> {
        vec![(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]
    }

    #[test]
    fn triangle_basic_structure() {
        let g = CsrGraph::from_edge_list(&triangle_edges(), 3);
        assert_eq!(g.n_nodes, 3);
        // 3 undirected edges → 6 directed.
        assert_eq!(g.n_edges, 6);
        // Each node has degree 2, weight 2.0.
        for i in 0..3 {
            assert_eq!(
                g.neighbor_range(i).len(),
                2,
                "node {} should have 2 neighbors",
                i
            );
            assert!(
                (g.node_strengths[i] - 2.0).abs() < 1e-12,
                "node {} strength should be 2.0",
                i
            );
        }
        // total_weight = 3.0 (three unit-weight undirected edges).
        assert!((g.total_weight - 3.0).abs() < 1e-12);
    }

    #[test]
    fn neighbor_range_correct() {
        let g = CsrGraph::from_edge_list(&triangle_edges(), 3);
        for node in 0..3 {
            let r = g.neighbor_range(node);
            let neighbors: Vec<usize> = g.targets[r].to_vec();
            // Every other node should appear exactly once.
            for other in 0..3usize {
                if other == node {
                    assert!(
                        !neighbors.contains(&other),
                        "self-loop in undirected triangle"
                    );
                } else {
                    assert!(
                        neighbors.contains(&other),
                        "missing neighbor {} for node {}",
                        other,
                        node
                    );
                }
            }
        }
    }

    #[test]
    fn community_edge_weight_all_same_community() {
        let g = CsrGraph::from_edge_list(&triangle_edges(), 3);
        let communities = vec![0usize, 0, 0]; // all one community
                                              // Node 0 has two edges each weight 1.0 → sum = 2.0.
        let w = g.community_edge_weight(0, 0, &communities);
        assert!((w - 2.0).abs() < 1e-12, "expected 2.0, got {}", w);
    }

    #[test]
    fn community_edge_weight_disjoint_communities() {
        let g = CsrGraph::from_edge_list(&triangle_edges(), 3);
        let communities = vec![0usize, 1, 2]; // each node its own community
                                              // Node 0 → community 1: one edge weight 1.0.
        let w01 = g.community_edge_weight(0, 1, &communities);
        assert!((w01 - 1.0).abs() < 1e-12, "expected 1.0, got {}", w01);
        // Node 0 → community 0: node 0 is in community 0, no self-loop.
        let w00 = g.community_edge_weight(0, 0, &communities);
        assert!((w00 - 0.0).abs() < 1e-12, "expected 0.0, got {}", w00);
    }

    #[test]
    fn modularity_all_one_community() {
        let g = CsrGraph::from_edge_list(&triangle_edges(), 3);
        let communities = vec![0usize, 0, 0];
        let q = g.modularity(&communities, 1.0);
        // Q = (1/6) * Σ [w_ij - s_i*s_j/6] over same-community directed edges.
        // All 6 directed edges are same-community.
        // Each directed edge: w=1, s_i=2, s_j=2 → 1 - 4/6 = 1/3.
        // Sum = 6 * (1/3) = 2.0 → Q = 2/6 = 1/3.
        assert!((q - 1.0 / 3.0).abs() < 1e-10, "expected Q≈0.333, got {}", q);
    }

    #[test]
    fn modularity_perfect_partition() {
        // Two fully-connected components, no inter-community edges.
        // Nodes 0,1 and 2,3.
        let edges = vec![(0, 1, 1.0), (2, 3, 1.0)];
        let g = CsrGraph::from_edge_list(&edges, 4);
        let communities = vec![0usize, 0, 1, 1];
        let q = g.modularity(&communities, 1.0);
        // Perfect 2-partition: Q = 0.75 (each component is self-contained).
        assert!(
            (q - 0.75).abs() < 1e-10,
            "expected Q=0.75 for perfect 2-partition, got {}",
            q
        );
    }

    #[test]
    fn aggregate_two_communities() {
        let g = CsrGraph::from_edge_list(&triangle_edges(), 3);
        // Communities: {0,1} = 0, {2} = 1.
        let communities = vec![0usize, 0, 1];
        let agg = g.aggregate(&communities);

        assert_eq!(agg.n_nodes, 2);

        // Community 0 (nodes 0,1) has intra-edge (0,1) weight 1.0.
        // Community 0 → community 1: edges (0,2) and (1,2) total weight 2.0.
        // Self-loop on community 0 = 1.0.
        // Edge between community 0 and community 1 = 2.0.
        assert!(
            (agg.total_weight - 2.5).abs() < 1e-10,
            "aggregated total_weight should be 2.5, got {}",
            agg.total_weight
        );
    }

    #[test]
    fn from_edge_list_symmetric_dedup() {
        // Providing both directions doubles the weight (sum-based dedup).
        // Use undirected input (one direction only) for correct behavior.
        let edges_one = vec![(0, 1, 1.0)];
        let g1 = CsrGraph::from_edge_list(&edges_one, 2);

        // Each node should have 1 neighbor with weight 1.0
        assert_eq!(g1.n_edges, 2); // 2 directed edges
        assert!((g1.total_weight - 1.0).abs() < 1e-12); // 1 undirected edge
        assert!((g1.node_strengths[0] - 1.0).abs() < 1e-12);
        assert!((g1.node_strengths[1] - 1.0).abs() < 1e-12);
    }
}
