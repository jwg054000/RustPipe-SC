//! RustPipe-SC: Fast single-cell and spatial transcriptomics in Rust.
//!
//! Accelerates compute-bound steps of the Scanpy/Squidpy pipeline:
//! sparse I/O, QC, normalization, HVG, PCA, kNN, Moran's I.

#![allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::needless_borrows_for_generic_args,
    clippy::len_zero,
    clippy::manual_clamp,
    clippy::ptr_arg,
    clippy::doc_overindented_list_items
)]

mod graph;
mod gsea;
mod h5ad;
mod hvg_sc;
mod io;
mod knn;
mod leiden;
mod markers;
mod morans_i;
mod normalize;
mod pca;
mod qc;
mod scale;
mod sparse;
mod stats_sc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(
    name = "rustpipe-sc",
    version = "0.3.0",
    about = "Fast single-cell and spatial transcriptomics in Rust"
)]
struct Cli {
    /// Number of threads (0 = all cores)
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute per-cell QC metrics and filter cells
    Qc {
        /// Input H5AD or CSV file
        #[arg(long, short)]
        input: String,

        /// Output directory
        #[arg(long, short)]
        output: String,

        /// Minimum genes per cell
        #[arg(long, default_value_t = 200)]
        min_genes: u32,

        /// Maximum mitochondrial percentage
        #[arg(long, default_value_t = 5.0)]
        max_pct_mt: f32,

        /// Minimum cells per gene
        #[arg(long, default_value_t = 3)]
        min_cells: usize,
    },

    /// Normalize counts and apply log1p transform
    Normalize {
        /// Input H5AD file (post-QC)
        #[arg(long, short)]
        input: String,

        /// Output H5AD file
        #[arg(long, short)]
        output: String,

        /// Target sum per cell
        #[arg(long, default_value_t = 10000.0)]
        target_sum: f32,
    },

    /// Select highly variable genes
    Hvg {
        /// Input CSV or H5AD
        #[arg(long, short)]
        input: String,

        /// Output directory
        #[arg(long, short)]
        output: String,

        /// Number of top HVGs to select
        #[arg(long, default_value_t = 2000)]
        n_top_genes: usize,

        /// HVG selection flavor: "seurat" (dispersion-bin) or "seurat_v3" (VST)
        #[arg(long, default_value = "seurat")]
        hvg_flavor: String,
    },

    /// PCA on expression data (dense or sparse)
    Pca {
        /// Input expression matrix CSV (genes x samples)
        #[arg(long)]
        input: String,

        /// Number of top high-variance genes to use
        #[arg(long, default_value_t = 2000)]
        top_genes: usize,

        /// Number of principal components to compute
        #[arg(long, default_value_t = 50)]
        n_components: usize,

        /// Output CSV for PC coordinates
        #[arg(long)]
        output: String,
    },

    /// Build k-nearest-neighbor graph from PCA scores
    Knn {
        /// Input PCA scores CSV
        #[arg(long)]
        input: String,

        /// Number of nearest neighbors
        #[arg(long, default_value_t = 15)]
        k: usize,

        /// Distance metric: "euclidean" or "cosine"
        #[arg(long, default_value = "euclidean")]
        metric: String,

        /// Output CSV for kNN graph
        #[arg(long)]
        output: String,
    },

    /// Compute Moran's I spatial autocorrelation
    MoransI {
        /// Input expression CSV (cells x genes)
        #[arg(long, short)]
        input: String,

        /// Spatial coordinates CSV (barcode, x, y)
        #[arg(long)]
        coords: String,

        /// Output directory
        #[arg(long, short)]
        output: String,

        /// Number of spatial neighbors
        #[arg(long, default_value_t = 6)]
        n_neighbors: usize,

        /// Number of permutations for p-values
        #[arg(long, default_value_t = 999)]
        n_perm: usize,
    },

    /// Full single-cell pipeline: QC → Normalize → HVG → PCA → kNN
    Pipeline {
        /// Input H5AD or expression CSV
        #[arg(long, short)]
        input: String,

        /// Output directory
        #[arg(long, short)]
        output: String,

        /// Number of HVGs
        #[arg(long, default_value_t = 2000)]
        n_hvg: usize,

        /// Number of PCs
        #[arg(long, default_value_t = 50)]
        n_pcs: usize,

        /// kNN neighbors
        #[arg(long, default_value_t = 15)]
        knn_k: usize,

        /// Minimum genes per cell (QC)
        #[arg(long, default_value_t = 200)]
        min_genes: u32,

        /// Maximum mito percentage (QC)
        #[arg(long, default_value_t = 5.0)]
        max_pct_mt: f32,

        /// Scaling clip value (sc.pp.scale max_value)
        #[arg(long, default_value_t = 10.0)]
        max_value: f32,

        /// Skip normalization (input is already normalized)
        #[arg(long, default_value_t = false)]
        skip_normalize: bool,

        /// HVG selection flavor: "seurat" (dispersion-bin) or "seurat_v3" (VST)
        #[arg(long, default_value = "seurat")]
        hvg_flavor: String,
    },

    /// Run GSEA on ranked gene list
    Gsea {
        #[arg(long)]
        ranks: String,
        #[arg(long)]
        gene_sets: String,
        #[arg(long, default_value_t = 10000)]
        nperm: usize,
        #[arg(long)]
        output: String,
    },

    /// Run Leiden community detection
    Leiden {
        #[arg(long)]
        input: String,
        #[arg(long, default_value_t = 1.0)]
        resolution: f64,
        #[arg(long, default_value_t = 10)]
        n_iterations: usize,
        #[arg(long)]
        output: String,
    },

    /// Load expression matrix (benchmark I/O speed)
    Load {
        #[arg(long)]
        input: String,
        #[arg(long)]
        gene: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    // Initialize rayon thread pool
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }

    let result = match cli.command {
        Commands::Qc {
            input,
            output,
            min_genes,
            max_pct_mt,
            min_cells,
        } => run_qc(&input, &output, min_genes, max_pct_mt, min_cells),

        Commands::Normalize {
            input,
            output,
            target_sum,
        } => run_normalize(&input, &output, target_sum),

        Commands::Hvg {
            input,
            output,
            n_top_genes,
            hvg_flavor,
        } => run_hvg(&input, &output, n_top_genes, &hvg_flavor),

        Commands::Pca {
            input,
            top_genes,
            n_components,
            output,
        } => {
            pca::run(&input, top_genes, n_components, &output).map_err(|e| anyhow::anyhow!("{}", e))
        }

        Commands::Knn {
            input,
            k,
            metric,
            output,
        } => knn::run(&input, k, &metric, &output).map_err(|e| anyhow::anyhow!("{}", e)),

        Commands::MoransI {
            input,
            coords,
            output,
            n_neighbors,
            n_perm,
        } => run_morans_i(&input, &coords, &output, n_neighbors, n_perm, cli.seed),

        Commands::Pipeline {
            input,
            output,
            n_hvg,
            n_pcs,
            knn_k,
            min_genes,
            max_pct_mt,
            max_value,
            skip_normalize,
            hvg_flavor,
        } => run_pipeline(
            &input,
            &output,
            n_hvg,
            n_pcs,
            knn_k,
            min_genes,
            max_pct_mt,
            max_value,
            skip_normalize,
            &hvg_flavor,
            cli.seed,
        ),

        Commands::Gsea {
            ranks,
            gene_sets,
            nperm,
            output,
        } => gsea::run(&ranks, &gene_sets, nperm, &output).map_err(|e| anyhow::anyhow!("{}", e)),

        Commands::Leiden {
            input,
            resolution,
            n_iterations,
            output,
        } => run_leiden(&input, resolution, n_iterations, &output),

        Commands::Load { input, gene } => run_load(&input, gene.as_deref()),
    };

    if let Err(e) = result {
        eprintln!("Error: {:#}", e);
        std::process::exit(1);
    }
}

// =====================================================================
//  Subcommand implementations
// =====================================================================

fn run_qc(
    input: &str,
    output: &str,
    min_genes: u32,
    max_pct_mt: f32,
    min_cells: usize,
) -> Result<()> {
    let t0 = Instant::now();
    std::fs::create_dir_all(output)?;

    let (mat, obs_names, var_names) = load_input(input)?;

    eprintln!("[qc] loaded {} cells x {} genes", mat.rows(), mat.cols());

    let metrics = qc::compute_qc_metrics(&mat, &var_names, &obs_names);
    qc::write_qc_csv(
        &metrics,
        &std::path::Path::new(output).join("qc_metrics.csv"),
    )?;

    let keep = qc::filter_cells_fixed(&metrics, min_genes, max_pct_mt);
    let n_kept = keep.iter().filter(|&&b| b).count();
    eprintln!(
        "[qc] filter: {} / {} cells pass (min_genes={}, max_pct_mt={})",
        n_kept,
        mat.rows(),
        min_genes,
        max_pct_mt
    );

    let (filtered_mat, filtered_names) = qc::apply_cell_filter(&mat, &obs_names, &keep);

    // Gene filter
    let (_gene_filtered, kept_genes) = qc::filter_genes(&filtered_mat, min_cells);
    let kept_var_names: Vec<String> = kept_genes.iter().map(|&i| var_names[i].clone()).collect();

    eprintln!(
        "[qc] gene filter: {} / {} genes pass (min_cells={})",
        kept_var_names.len(),
        var_names.len(),
        min_cells
    );

    // Write filtered barcodes
    let barcodes_path = std::path::Path::new(output).join("filtered_barcodes.csv");
    let mut wtr = csv::Writer::from_path(&barcodes_path)?;
    wtr.write_record(["barcode"])?;
    for name in &filtered_names {
        wtr.write_record([name])?;
    }
    wtr.flush()?;

    eprintln!("[qc] complete in {:.3}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn run_normalize(input: &str, output: &str, target_sum: f32) -> Result<()> {
    let t0 = Instant::now();
    let (mat, obs_names, var_names) = load_input(input)?;

    let normed = normalize::normalize_log1p_sparse(&mat, target_sum);

    eprintln!(
        "[normalize] {} cells, target_sum={}, nnz: {} -> {} in {:.3}s",
        mat.rows(),
        target_sum,
        mat.nnz(),
        normed.nnz(),
        t0.elapsed().as_secs_f64()
    );

    // Write as H5AD if output ends with .h5ad
    if output.ends_with(".h5ad") {
        let adata = h5ad::AnnData {
            x: normed,
            obs_names,
            var_names,
            obs: std::collections::HashMap::new(),
            var: std::collections::HashMap::new(),
        };
        h5ad::write_h5ad(&adata, std::path::Path::new(output))?;
    }

    Ok(())
}

fn run_hvg(input: &str, output: &str, n_top_genes: usize, hvg_flavor: &str) -> Result<()> {
    let t0 = Instant::now();
    std::fs::create_dir_all(output)?;

    let (mat, _obs_names, var_names) = load_input(input)?;

    let result = match hvg_flavor {
        "seurat_v3" => hvg_sc::select_hvg_sparse(&mat, &var_names, n_top_genes)?,
        _ => hvg_sc::select_hvg_seurat(&mat, &var_names, n_top_genes, 20)?,
    };

    // Write HVG gene list
    let genes_path = std::path::Path::new(output).join("hvg_genes.csv");
    let mut wtr = csv::Writer::from_path(&genes_path)?;
    wtr.write_record(["gene"])?;
    for name in &result.gene_names {
        wtr.write_record([name])?;
    }
    wtr.flush()?;

    eprintln!(
        "[hvg] selected {} HVGs in {:.3}s",
        n_top_genes,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_morans_i(
    input: &str,
    coords_path: &str,
    output: &str,
    n_neighbors: usize,
    n_perm: usize,
    seed: u64,
) -> Result<()> {
    let t0 = Instant::now();
    std::fs::create_dir_all(output)?;

    // Load expression (genes x cells or cells x genes CSV)
    let (mat, _cell_names, gene_names) = load_input(input)?;

    // Load spatial coordinates
    let (_coord_names, coords) = io::read_spatial_coords(coords_path)
        .map_err(|e| anyhow::anyhow!("Failed to read coordinates: {}", e))?;

    eprintln!(
        "[morans_i] {} genes x {} spots, {} neighbors",
        gene_names.len(),
        coords.len(),
        n_neighbors
    );

    // Build spatial neighbor graph
    let graph = morans_i::spatial_neighbor_graph(&coords, n_neighbors);
    eprintln!(
        "[morans_i] spatial graph: {} nodes, {} edges",
        graph.n_nodes, graph.n_edges
    );

    // Extract expression per gene as Vec<Vec<f32>>
    let n_spots = mat.rows();
    let n_genes = mat.cols();
    let mut expression: Vec<Vec<f32>> = vec![vec![0.0f32; n_spots]; n_genes];
    for i in 0..n_spots {
        let row = mat.outer_view(i).unwrap();
        for (col, &val) in row.iter() {
            expression[col][i] = val;
        }
    }

    let result = morans_i::compute_morans_i(&expression, &graph, &gene_names, n_perm, seed);

    morans_i::write_morans_csv(&result, &std::path::Path::new(output).join("morans_i.csv"))?;

    let n_sig = result.pval_sim_fdr.iter().filter(|&&p| p < 0.05).count();
    eprintln!(
        "[morans_i] {} significant SVGs (FDR < 0.05) in {:.2}s",
        n_sig,
        t0.elapsed().as_secs_f64()
    );

    Ok(())
}

fn run_pipeline(
    input: &str,
    output: &str,
    n_hvg: usize,
    n_pcs: usize,
    knn_k: usize,
    min_genes: u32,
    max_pct_mt: f32,
    max_value: f32,
    skip_normalize: bool,
    hvg_flavor: &str,
    seed: u64,
) -> Result<()> {
    let pipeline_t0 = Instant::now();
    std::fs::create_dir_all(output)?;
    let out_path = std::path::Path::new(output);

    let mut timings: Vec<(&str, f64)> = Vec::new();

    // Step 1: Load
    let t0 = Instant::now();
    let (mat, obs_names, var_names) = load_input(input)?;
    timings.push(("load", t0.elapsed().as_secs_f64()));
    eprintln!(
        "[pipeline] loaded {} cells x {} genes",
        mat.rows(),
        mat.cols()
    );

    // Step 2: QC
    let t0 = Instant::now();
    let metrics = qc::compute_qc_metrics(&mat, &var_names, &obs_names);
    qc::write_qc_csv(&metrics, &out_path.join("qc_metrics.csv"))?;

    let keep = qc::filter_cells_fixed(&metrics, min_genes, max_pct_mt);
    let n_kept = keep.iter().filter(|&&b| b).count();
    let (filtered_mat, filtered_names) = qc::apply_cell_filter(&mat, &obs_names, &keep);
    let (gene_filtered, kept_genes) = qc::filter_genes(&filtered_mat, 3);
    let kept_var: Vec<String> = kept_genes.iter().map(|&i| var_names[i].clone()).collect();
    timings.push(("qc_filter", t0.elapsed().as_secs_f64()));
    eprintln!("[pipeline] QC: {} cells, {} genes", n_kept, kept_var.len());

    // Step 3: Normalize (skip if input is already normalized)
    let normed = if skip_normalize {
        eprintln!("[pipeline] skipping normalization (--skip-normalize)");
        gene_filtered
    } else {
        let t0 = Instant::now();
        let n = normalize::normalize_log1p_sparse(&gene_filtered, 1e4);
        timings.push(("normalize", t0.elapsed().as_secs_f64()));
        eprintln!("[pipeline] normalized, nnz={}", n.nnz());
        n
    };

    // Step 4: HVG selection
    let t0 = Instant::now();
    let hvg_result = match hvg_flavor {
        "seurat_v3" => hvg_sc::select_hvg_sparse(&normed, &kept_var, n_hvg)?,
        _ => hvg_sc::select_hvg_seurat(&normed, &kept_var, n_hvg, 20)?,
    };
    let hvg_mat = sparse::subset_genes(&normed, &hvg_result.gene_indices);
    timings.push(("hvg", t0.elapsed().as_secs_f64()));
    eprintln!(
        "[pipeline] HVG ({}): {} genes selected",
        hvg_flavor,
        hvg_result.gene_names.len()
    );

    // Export HVG gene list
    {
        let hvg_path = out_path.join("hvg_genes.csv");
        let mut wtr = csv::Writer::from_path(&hvg_path)?;
        wtr.write_record(["gene"])?;
        for name in &hvg_result.gene_names {
            wtr.write_record([name])?;
        }
        wtr.flush()?;
        eprintln!("[pipeline] wrote {}", hvg_path.display());
    }

    // Step 5: Scale (z-score normalization with clipping)
    let t0 = Instant::now();
    let (scaled, _means, _stds) = scale::scale_sparse(&hvg_mat, max_value);
    timings.push(("scale", t0.elapsed().as_secs_f64()));
    eprintln!(
        "[pipeline] scaled: {} x {}, max_value={}",
        scaled.nrows(),
        scaled.ncols(),
        max_value
    );

    // Step 6: PCA (on scaled dense matrix)
    let t0 = Instant::now();
    let pca_result = pca::run_pca_scaled(
        &scaled,
        &hvg_result.gene_names,
        n_pcs,
        3,  // power iterations
        10, // oversampling
        seed,
    );
    timings.push(("pca", t0.elapsed().as_secs_f64()));
    eprintln!(
        "[pipeline] PCA: {} components, var={:.2}%",
        n_pcs,
        pca_result.variance_explained.iter().sum::<f64>() * 100.0
    );

    // Write PCA scores CSV
    pca::write_pca_csv(
        &pca_result,
        &filtered_names,
        &out_path.join("pca_scores.csv").to_string_lossy(),
    )
    .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Step 7: kNN
    let t0 = Instant::now();
    let n_cells = filtered_names.len();
    let n_pcs_actual = pca_result.variance_explained.len();
    let pca_flat: Vec<f64> = pca_result
        .pc_scores
        .iter()
        .flat_map(|r| r.iter().copied())
        .collect();

    let knn_result = knn::run_knn(&pca_flat, n_cells, n_pcs_actual, knn_k, "euclidean")
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    knn::write_knn_csv(&knn_result, &out_path.join("knn.csv").to_string_lossy())
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    timings.push(("knn", t0.elapsed().as_secs_f64()));
    eprintln!("[pipeline] kNN: k={}, {} cells", knn_k, n_cells);

    // Step 8: Leiden clustering
    let t0 = Instant::now();
    let knn_graph = graph::CsrGraph::from_knn_csv(&out_path.join("knn.csv").to_string_lossy())
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let leiden_result = leiden::run_leiden(&knn_graph, 1.0, 10, seed);
    timings.push(("leiden", t0.elapsed().as_secs_f64()));
    eprintln!(
        "[pipeline] Leiden: {} clusters, modularity={:.4}",
        leiden_result.n_communities, leiden_result.modularity
    );

    // Write cluster assignments
    {
        let clusters_path = out_path.join("clusters.csv");
        let mut wtr = csv::Writer::from_path(&clusters_path)?;
        wtr.write_record(["barcode", "cluster"])?;
        for (i, &c) in leiden_result.communities.iter().enumerate() {
            wtr.write_record([&filtered_names[i], &c.to_string()])?;
        }
        wtr.flush()?;
    }

    // Step 9: Marker genes (Wilcoxon rank-sum, on normalized HVG expression)
    let t0 = Instant::now();
    let marker_results = markers::find_markers(
        &hvg_mat,
        &leiden_result.communities,
        &hvg_result.gene_names,
        25,
    )?;
    markers::write_markers_csv(&marker_results, 25, &out_path.join("markers.csv"))?;
    timings.push(("markers", t0.elapsed().as_secs_f64()));
    eprintln!("[pipeline] markers: top 25 per cluster");

    // Write timings JSON
    let total_time = pipeline_t0.elapsed().as_secs_f64();
    let timings_json = serde_json::json!({
        "pipeline": "rustpipe-sc",
        "version": "0.3.0",
        "seed": seed,
        "input": input,
        "n_cells_raw": mat.rows(),
        "n_genes_raw": mat.cols(),
        "n_cells_filtered": n_kept,
        "n_genes_filtered": kept_var.len(),
        "n_hvg": n_hvg,
        "hvg_flavor": hvg_flavor,
        "max_value": max_value,
        "n_pcs": n_pcs,
        "knn_k": knn_k,
        "steps": timings.iter().map(|(name, time)| {
            serde_json::json!({ "step": name, "seconds": time })
        }).collect::<Vec<_>>(),
        "total_seconds": total_time,
    });

    std::fs::write(
        out_path.join("pipeline_timings.json"),
        serde_json::to_string_pretty(&timings_json)?,
    )?;

    eprintln!("\n[pipeline] COMPLETE in {:.3}s", total_time);
    for (name, time) in &timings {
        eprintln!("  {:>15}: {:.4}s", name, time);
    }

    Ok(())
}

fn run_leiden(input: &str, resolution: f64, n_iterations: usize, output: &str) -> Result<()> {
    let t0 = Instant::now();
    let graph = graph::CsrGraph::from_knn_csv(input).map_err(|e| anyhow::anyhow!("{}", e))?;
    eprintln!(
        "[leiden] graph: {} nodes, {} edges, total_weight={:.3}",
        graph.n_nodes, graph.n_edges, graph.total_weight
    );
    leiden::run(input, resolution, n_iterations, output).map_err(|e| anyhow::anyhow!("{}", e))?;
    eprintln!("[leiden] complete in {:.3}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn run_load(input: &str, gene: Option<&str>) -> Result<()> {
    let t0 = Instant::now();
    let (gene_names, sample_names, data, n_genes, n_samples) =
        io::read_expression_matrix(input).map_err(|e| anyhow::anyhow!("{}", e))?;

    eprintln!(
        "Loaded {}x{} matrix in {:.3}s ({:.1} MB)",
        n_genes,
        n_samples,
        t0.elapsed().as_secs_f64(),
        (data.len() * 8) as f64 / 1_048_576.0
    );

    if let Some(target_gene) = gene {
        if let Some(idx) = gene_names.iter().position(|g| g == target_gene) {
            let row_start = idx * n_samples;
            let row = &data[row_start..row_start + n_samples];
            println!("Expression for {}:", target_gene);
            for (sample, &val) in sample_names.iter().zip(row.iter()) {
                println!("  {}: {:.4}", sample, val);
            }
        } else {
            eprintln!("Gene '{}' not found", target_gene);
        }
    }
    Ok(())
}

/// Load input from H5AD or CSV, returning sparse matrix (cells x genes) + names.
///
/// Auto-detects format:
/// - .h5ad: read via h5ad module
/// - .csv: check if first column header looks like gene names (genes x cells)
///   or cell barcodes (cells x genes)
fn load_input(path: &str) -> Result<(sparse::SpMat, Vec<String>, Vec<String>)> {
    if path.ends_with(".h5ad") || path.ends_with(".h5") {
        // Auto-detect H5AD vs 10x CellRanger H5 format
        let (mat, obs, var) = h5ad::read_h5_auto(std::path::Path::new(path))?;
        Ok((mat, obs, var))
    } else {
        // Read header to detect format
        let file = std::fs::File::open(path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        let header = rdr.headers()?.clone();
        let first_col = header.get(0).unwrap_or("");

        // If first column is "gene_symbol" or "gene", it's genes x cells (legacy format)
        if first_col.to_lowercase().contains("gene") || first_col.to_lowercase() == "symbol" {
            log::info!("Detected genes x cells CSV format (transposing to cells x genes)");
            let result = io::read_expression_to_sparse(path)
                .map_err(|e| anyhow::anyhow!("Failed to load: {}", e))?;
            Ok(result)
        } else {
            // Assume cells x genes
            log::info!("Detected cells x genes CSV format");
            let result =
                io::read_sparse_csv(path).map_err(|e| anyhow::anyhow!("Failed to load: {}", e))?;
            Ok(result)
        }
    }
}
