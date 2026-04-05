/// Randomized PCA via truncated SVD (Halko, Martinsson & Tropp, 2011).
///
/// Replaces R's `prcomp()` for expression matrices.  Uses a randomized
/// power-iteration scheme identical to scikit-learn's
/// `PCA(svd_solver='randomized')`.  No external linear-algebra crate is
/// required -- all decompositions (QR via modified Gram-Schmidt, Jacobi SVD
/// for the small sketch matrix) are implemented inline.
///
/// Dependencies: rayon (parallel variance), rand + rand_distr (Gaussian
/// random matrix), csv (output).
use rayon::prelude::*;
use std::time::Instant;

// =====================================================================
//  Public types
// =====================================================================

pub struct PcaResult {
    pub pc_scores: Vec<Vec<f64>>,     // n_samples x n_components
    pub variance_explained: Vec<f64>, // fraction of variance per PC
    #[allow(dead_code)]
    pub selected_genes: Vec<String>, // genes kept after variance filter
    #[allow(dead_code)]
    pub timings: PcaTimings,
}

pub struct PcaTimings {
    #[allow(dead_code)]
    pub variance_calc_ms: f64,
    #[allow(dead_code)]
    pub gene_selection_ms: f64,
    #[allow(dead_code)]
    pub centering_ms: f64,
    #[allow(dead_code)]
    pub svd_ms: f64,
    #[allow(dead_code)]
    pub total_ms: f64,
}

// =====================================================================
//  Matrix helpers -- flat row-major
// =====================================================================

/// C = A * B  where A is (m x k), B is (k x n), C is (m x n).
/// All stored as flat row-major Vec<f64>.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        let a_off = i * k;
        let c_off = i * n;
        for p in 0..k {
            let a_val = a[a_off + p];
            if a_val == 0.0 {
                continue;
            }
            let b_off = p * n;
            for j in 0..n {
                c[c_off + j] += a_val * b[b_off + j];
            }
        }
    }
    c
}

/// C = A^T * B  where A is stored as (m x k), so A^T is (k x m).
/// B is (m x n), result C is (k x n).
fn matmul_at_b(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; k * n];
    for p in 0..m {
        let a_off = p * k;
        let b_off = p * n;
        for i in 0..k {
            let a_val = a[a_off + i];
            if a_val == 0.0 {
                continue;
            }
            let c_off = i * n;
            for j in 0..n {
                c[c_off + j] += a_val * b[b_off + j];
            }
        }
    }
    c
}

// =====================================================================
//  QR decomposition -- modified Gram-Schmidt
// =====================================================================

/// Modified Gram-Schmidt QR factorisation of an (m x n) row-major matrix.
/// Returns (Q, R) where Q is (m x n) orthonormal and R is (n x n) upper
/// triangular.  Only Q is needed downstream but R is cheap to keep.
fn qr_mgs(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    // Work on columns for simpler inner products.
    let mut cols: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..m).map(|i| a[i * n + j]).collect())
        .collect();

    let mut r = vec![0.0f64; n * n];

    for j in 0..n {
        // Orthogonalise against all earlier columns.
        for i in 0..j {
            let dot: f64 = cols[j].iter().zip(cols[i].iter()).map(|(a, b)| a * b).sum();
            r[i * n + j] = dot;
            for row in 0..m {
                cols[j][row] -= dot * cols[i][row];
            }
        }
        // Normalise.
        let norm: f64 = cols[j].iter().map(|v| v * v).sum::<f64>().sqrt();
        r[j * n + j] = norm;
        if norm > 1e-14 {
            let inv = 1.0 / norm;
            for row in 0..m {
                cols[j][row] *= inv;
            }
        }
    }

    // Pack Q back to row-major.
    let mut q = vec![0.0f64; m * n];
    for j in 0..n {
        for i in 0..m {
            q[i * n + j] = cols[j][i];
        }
    }
    (q, r)
}

// =====================================================================
//  One-sided Jacobi SVD for small matrices
// =====================================================================

/// Thin SVD of a small (m x n) matrix B where m << n.
///
/// Strategy: form G = B B^T (m x m), eigendecompose G via Jacobi
/// rotations, then recover the SVD factors.
///
/// Returns (U, sigma, Vt):
///   U     : m x m  row-major (left singular vectors)
///   sigma : length m (singular values, descending)
///   Vt    : m x n  row-major (first m right singular vectors)
fn small_svd(b: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // -- 1. Gram matrix G = B * B^T  (m x m, symmetric) --
    let mut g = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut dot = 0.0f64;
            let bi = i * n;
            let bj = j * n;
            for p in 0..n {
                dot += b[bi + p] * b[bj + p];
            }
            g[i * m + j] = dot;
            g[j * m + i] = dot;
        }
    }

    // -- 2. Jacobi eigendecomposition of symmetric G --
    let mut eigvecs = vec![0.0f64; m * m];
    for i in 0..m {
        eigvecs[i * m + i] = 1.0;
    }

    let max_sweeps = 200;
    let tol = 1e-12;

    for _sweep in 0..max_sweeps {
        // Off-diagonal Frobenius norm.
        let mut off_norm = 0.0f64;
        for i in 0..m {
            for j in (i + 1)..m {
                off_norm += g[i * m + j] * g[i * m + j];
            }
        }
        if off_norm < tol {
            break;
        }

        for p in 0..m {
            for q in (p + 1)..m {
                let g_pp = g[p * m + p];
                let g_qq = g[q * m + q];
                let g_pq = g[p * m + q];

                if g_pq.abs() < 1e-15 {
                    continue;
                }

                // Jacobi rotation angle.
                let tau = (g_qq - g_pp) / (2.0 * g_pq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal and off-diagonal for (p, q).
                let new_pp = c * c * g_pp - 2.0 * s * c * g_pq + s * s * g_qq;
                let new_qq = s * s * g_pp + 2.0 * s * c * g_pq + c * c * g_qq;
                g[p * m + p] = new_pp;
                g[q * m + q] = new_qq;
                g[p * m + q] = 0.0;
                g[q * m + p] = 0.0;

                // Rotate rows/cols for all other indices.
                for r in 0..m {
                    if r == p || r == q {
                        continue;
                    }
                    let g_rp = g[r * m + p];
                    let g_rq = g[r * m + q];
                    let new_rp = c * g_rp - s * g_rq;
                    let new_rq = s * g_rp + c * g_rq;
                    g[r * m + p] = new_rp;
                    g[p * m + r] = new_rp;
                    g[r * m + q] = new_rq;
                    g[q * m + r] = new_rq;
                }

                // Accumulate eigenvectors.
                for r in 0..m {
                    let vp = eigvecs[r * m + p];
                    let vq = eigvecs[r * m + q];
                    eigvecs[r * m + p] = c * vp - s * vq;
                    eigvecs[r * m + q] = s * vp + c * vq;
                }
            }
        }
    }

    // -- 3. Sort eigenvalues descending, extract singular values --
    let mut eigen_pairs: Vec<(f64, usize)> = (0..m).map(|i| (g[i * m + i], i)).collect();
    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let sigma: Vec<f64> = eigen_pairs
        .iter()
        .map(|(ev, _)| if *ev > 0.0 { ev.sqrt() } else { 0.0 })
        .collect();

    // U columns = sorted eigenvectors.
    let mut u = vec![0.0f64; m * m];
    for (new_col, &(_, old_col)) in eigen_pairs.iter().enumerate() {
        for row in 0..m {
            u[row * m + new_col] = eigvecs[row * m + old_col];
        }
    }

    // -- 4. Vt = diag(1/sigma) * U^T * B --
    let mut vt = vec![0.0f64; m * n];
    for j in 0..m {
        if sigma[j] < 1e-14 {
            continue;
        }
        let inv_s = 1.0 / sigma[j];
        for col in 0..n {
            let mut dot = 0.0f64;
            for row in 0..m {
                dot += u[row * m + j] * b[row * n + col];
            }
            vt[j * n + col] = dot * inv_s;
        }
    }

    (u, sigma, vt)
}

// =====================================================================
//  Randomized PCA core
// =====================================================================

pub fn run_pca(
    data: &[f64],
    gene_names: &[String],
    _sample_names: &[String],
    n_genes: usize,
    n_samples: usize,
    top_n_genes: usize,
    n_components: usize,
) -> PcaResult {
    let total_t0 = Instant::now();

    // -----------------------------------------------------------------
    //  Step 1: per-gene variance (rayon parallel)
    // -----------------------------------------------------------------
    let var_t0 = Instant::now();
    let variances: Vec<f64> = (0..n_genes)
        .into_par_iter()
        .map(|i| {
            let start = i * n_samples;
            let row = &data[start..start + n_samples];
            let n = n_samples as f64;
            let mean = row.iter().sum::<f64>() / n;
            row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
        })
        .collect();
    let variance_calc_ms = var_t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[pca] variance calculation: {:.1}ms ({} genes x {} samples)",
        variance_calc_ms, n_genes, n_samples
    );

    // -----------------------------------------------------------------
    //  Step 2: select top variable genes
    // -----------------------------------------------------------------
    let sel_t0 = Instant::now();
    let effective_top = top_n_genes.min(n_genes);

    let mut indices: Vec<usize> = (0..n_genes).collect();
    indices.sort_unstable_by(|&a, &b| {
        variances[b]
            .partial_cmp(&variances[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_indices = &indices[..effective_top];

    let selected_genes: Vec<String> = top_indices.iter().map(|&i| gene_names[i].clone()).collect();
    let gene_selection_ms = sel_t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[pca] gene selection: {:.1}ms (kept {} / {})",
        gene_selection_ms, effective_top, n_genes
    );

    // -----------------------------------------------------------------
    //  Step 3: build submatrix and center (subtract column means)
    // -----------------------------------------------------------------
    let center_t0 = Instant::now();
    let ng = effective_top; // rows in sub-matrix
    let ns = n_samples; // cols in sub-matrix

    // Extract submatrix: ng x ns, row-major.
    let mut sub = vec![0.0f64; ng * ns];
    for (new_row, &orig_row) in top_indices.iter().enumerate() {
        let src = orig_row * ns;
        sub[new_row * ns..(new_row + 1) * ns].copy_from_slice(&data[src..src + ns]);
    }

    // Row means (mean of each gene across samples).
    // This matches R's prcomp(t(X), center=TRUE) which centers each gene.
    let mut row_means = vec![0.0f64; ng];
    for i in 0..ng {
        let off = i * ns;
        let sum: f64 = sub[off..off + ns].iter().sum();
        row_means[i] = sum / ns as f64;
    }

    // Subtract row means (center each gene across samples).
    for i in 0..ng {
        let off = i * ns;
        let mean = row_means[i];
        for j in 0..ns {
            sub[off + j] -= mean;
        }
    }
    let centering_ms = center_t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[pca] centering: {:.1}ms", centering_ms);

    // -----------------------------------------------------------------
    //  Step 4: randomized SVD
    // -----------------------------------------------------------------
    let svd_t0 = Instant::now();
    let oversampling = 10usize;
    let k = (n_components + oversampling).min(ng).min(ns); // sketch rank, bounded by matrix dims

    // 4a. Gaussian random matrix Omega (ns x k).
    let omega = {
        use rand::Rng;
        use rand_distr::StandardNormal;
        let mut rng = rand::rng();
        let mut omega = vec![0.0f64; ns * k];
        for v in omega.iter_mut() {
            *v = rng.sample(StandardNormal);
        }
        omega
    };

    // 4b. Y = A * Omega       (ng x ns) * (ns x k) = (ng x k)
    let y = matmul(&sub, &omega, ng, ns, k);

    // 4c. QR(Y) -> Q (ng x k), orthonormal basis for range(A).
    let (q, _r) = qr_mgs(&y, ng, k);

    // 4d. B = Q^T * A          (k x ng)^T ... but Q is (ng x k),
    //     so Q^T A is (k x ns).
    let b_mat = matmul_at_b(&q, &sub, ng, k, ns);

    // 4e. Full SVD of the small matrix B (k x ns).
    let (u_b, sigma, _vt) = small_svd(&b_mat, k, ns);

    // 4f. U_approx = Q * U_B   (ng x k) * (k x k) = (ng x k)
    let u_approx = matmul(&q, &u_b, ng, k, k);

    // 4g. PC scores = A^T * U_approx
    //     sub is (ng x ns), sub^T is (ns x ng).  We want (ns x n_components).
    //     scores[s][c] = sum_g sub[g * ns + s] * u_approx[g * k + c]
    let mut pc_scores: Vec<Vec<f64>> = Vec::with_capacity(ns);
    for s in 0..ns {
        let mut row = Vec::with_capacity(n_components);
        for c in 0..n_components {
            let mut dot = 0.0f64;
            for g in 0..ng {
                dot += sub[g * ns + s] * u_approx[g * k + c];
            }
            row.push(dot);
        }
        pc_scores.push(row);
    }

    // 4h. Variance explained = sigma_i^2 / sum(sub^2)
    //     Total variance from the centered submatrix (Frobenius norm squared).
    let total_var: f64 = sub.iter().map(|x| x * x).sum::<f64>();
    let var_explained: Vec<f64> = sigma[..n_components]
        .iter()
        .map(|s| (s * s) / total_var)
        .collect();

    let svd_ms = svd_t0.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

    let sigma_sq_sum: f64 = sigma.iter().map(|s| s * s).sum();

    eprintln!("[pca] randomized SVD: {:.1}ms", svd_ms);
    eprintln!("[pca] total: {:.1}ms", total_ms);
    eprintln!(
        "[pca] variance captured by {} PCs: {:.2}% (of top-{} subspace)",
        n_components,
        var_explained.iter().sum::<f64>() * 100.0,
        effective_top
    );
    eprintln!(
        "[pca] sketch captured {:.2}% of subspace variance (k={})",
        sigma_sq_sum / total_var * 100.0,
        k
    );

    PcaResult {
        pc_scores,
        variance_explained: var_explained,
        selected_genes,
        timings: PcaTimings {
            variance_calc_ms,
            gene_selection_ms,
            centering_ms,
            svd_ms,
            total_ms,
        },
    }
}

// =====================================================================
//  CSV output
// =====================================================================

pub fn write_pca_csv(
    result: &PcaResult,
    sample_names: &[String],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;

    let n_components = result.variance_explained.len();

    // Header: sample, PC1, PC2, ..., PCn
    let mut header: Vec<String> = Vec::with_capacity(1 + n_components);
    header.push("sample".to_string());
    for i in 1..=n_components {
        header.push(format!("PC{}", i));
    }
    wtr.write_record(&header)?;

    // Data rows.
    for (i, scores) in result.pc_scores.iter().enumerate() {
        let mut row = Vec::with_capacity(1 + n_components);
        row.push(sample_names[i].clone());
        for s in scores {
            row.push(format!("{:.6}", s));
        }
        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}

// =====================================================================
//  Entry point (called from main.rs)
// =====================================================================

pub fn run(
    input: &str,
    top_genes: usize,
    n_components: usize,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let (gene_names, sample_names, data, n_genes, n_samples) =
        crate::io::read_expression_matrix(input)?;
    eprintln!(
        "[pca] loaded {}x{} matrix in {:.1}ms",
        n_genes,
        n_samples,
        t0.elapsed().as_secs_f64() * 1000.0
    );

    if n_components > top_genes {
        return Err(format!(
            "n_components ({}) must be <= top_genes ({})",
            n_components, top_genes
        )
        .into());
    }

    let result = run_pca(
        &data,
        &gene_names,
        &sample_names,
        n_genes,
        n_samples,
        top_genes,
        n_components,
    );

    // Variance explained table.
    eprintln!("\nVariance explained per PC:");
    let mut cumulative = 0.0f64;
    for (i, &v) in result.variance_explained.iter().enumerate() {
        cumulative += v;
        eprintln!(
            "  PC{:>2}: {:>6.2}%  (cumulative: {:>6.2}%)",
            i + 1,
            v * 100.0,
            cumulative * 100.0
        );
    }

    write_pca_csv(&result, &sample_names, output)?;
    eprintln!(
        "\n[pca] wrote {} samples x {} PCs to {}",
        n_samples, n_components, output
    );

    Ok(())
}

// =====================================================================
//  Sparse PCA — randomized SVD with implicit centering
// =====================================================================

/// Run PCA on a sparse matrix (cells x genes) with implicit centering.
///
/// Uses the same Halko-Martinsson-Tropp algorithm as dense PCA, but replaces
/// dense matmul with sparse-dense products that never materialize the centered
/// matrix. Includes power iteration for accuracy.
///
/// # Arguments
/// * `mat` — sparse CSR (cells x genes), already normalized + log1p
/// * `gene_means` — per-gene means from `sparse_gene_stats`
/// * `gene_names` — gene identifiers for selected genes
/// * `n_components` — number of PCs to compute (e.g., 50)
/// * `n_power_iter` — power iteration rounds (3 recommended)
/// * `oversampling` — sketch oversampling (10 default)
/// * `seed` — RNG seed for reproducibility
#[allow(dead_code)]
pub fn run_pca_sparse(
    mat: &crate::sparse::SpMat,
    gene_means: &[f32],
    gene_names: &[String],
    n_components: usize,
    n_power_iter: usize,
    oversampling: usize,
    seed: u64,
) -> PcaResult {
    use crate::sparse::{spmm_at_centered, spmm_centered};
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;

    let total_t0 = Instant::now();
    let n_cells = mat.rows();
    let n_genes = mat.cols();
    let k = n_components + oversampling;

    eprintln!(
        "[pca-sparse] {}x{} matrix, {} components, {} power iterations",
        n_cells, n_genes, n_components, n_power_iter
    );

    // Step 1: Gaussian random matrix Omega (genes x k)
    let svd_t0 = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let omega = Array2::from_shape_fn((n_genes, k), |_| {
        use rand::Rng;
        rng.sample::<f32, _>(StandardNormal)
    });

    // Step 2: Y = (A - mu) * Omega  (cells x k) via implicit centering
    let mut y = spmm_centered(mat, gene_means, &omega);

    // Step 3: Power iteration for accuracy
    for _iter in 0..n_power_iter {
        // QR decomposition of Y
        let (q_flat, _) = qr_mgs_f32(&y);
        let q = Array2::from_shape_vec((n_cells, k), q_flat).unwrap();

        // Z = (A - mu)^T * Q  (genes x k)
        let z = spmm_at_centered(mat, gene_means, &q);

        // QR decomposition of Z
        let (z_q_flat, _) = qr_mgs_f32(&z);
        let z_q = Array2::from_shape_vec((n_genes, k), z_q_flat).unwrap();

        // Y = (A - mu) * Z_Q
        y = spmm_centered(mat, gene_means, &z_q);
    }

    // Step 4: Final QR of Y
    let (q_flat, _) = qr_mgs_f32(&y);
    let q = Array2::from_shape_vec((n_cells, k), q_flat).unwrap();

    // Step 5: B = Q^T * (A - mu) = ((A - mu)^T * Q)^T  (k x genes... but we compute genes x k then transpose)
    let b_t = spmm_at_centered(mat, gene_means, &q); // genes x k

    // Convert to f64 flat for small_svd (reuse existing dense SVD)
    let k_actual = k.min(n_genes).min(n_cells);
    let mut b_flat = vec![0.0f64; k_actual * n_genes];
    for i in 0..k_actual {
        for j in 0..n_genes {
            b_flat[i * n_genes + j] = b_t[[j, i]] as f64;
        }
    }

    // Step 6: SVD of B (k x genes) — small matrix
    let (u_b, sigma, _vt) = small_svd(&b_flat, k_actual, n_genes);

    // Step 7: PC scores = Q * U_B  (cells x k) * (k x k) = cells x k
    // Then truncate to n_components
    let mut pc_scores: Vec<Vec<f64>> = Vec::with_capacity(n_cells);
    for c in 0..n_cells {
        let mut row = Vec::with_capacity(n_components);
        for pc in 0..n_components {
            let mut dot = 0.0f64;
            for j in 0..k_actual {
                dot += q[[c, j]] as f64 * u_b[j * k_actual + pc];
            }
            // Scale by singular value to get scores
            row.push(dot * sigma[pc]);
        }
        pc_scores.push(row);
    }

    // Step 8: Variance explained
    // Total variance = sum over all genes of sum_cells((x_gc - mean_g)^2)
    // Efficient: total_var = sum_g(sum_sq_g - n_cells * mean_g^2)
    let total_var: f64 = {
        let mut sum_sq = vec![0.0f64; n_genes];
        for i in 0..n_cells {
            let row = mat.outer_view(i).unwrap();
            for (g, &val) in row.iter() {
                sum_sq[g] += (val as f64) * (val as f64);
            }
        }
        let mut tv = 0.0f64;
        for g in 0..n_genes {
            tv += sum_sq[g] - n_cells as f64 * (gene_means[g] as f64) * (gene_means[g] as f64);
        }
        tv
    };

    let var_explained: Vec<f64> = if total_var > 0.0 {
        sigma[..n_components]
            .iter()
            .map(|s| (s * s) / total_var)
            .collect()
    } else {
        vec![0.0; n_components]
    };

    let svd_ms = svd_t0.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("[pca-sparse] SVD: {:.1}ms", svd_ms);
    eprintln!("[pca-sparse] total: {:.1}ms", total_ms);
    eprintln!(
        "[pca-sparse] variance captured: {:.2}%",
        var_explained.iter().sum::<f64>() * 100.0
    );

    PcaResult {
        pc_scores,
        variance_explained: var_explained,
        selected_genes: gene_names.to_vec(),
        timings: PcaTimings {
            variance_calc_ms: 0.0,
            gene_selection_ms: 0.0,
            centering_ms: 0.0,
            svd_ms,
            total_ms,
        },
    }
}

// =====================================================================
//  PCA on pre-scaled dense matrix (from scale::scale_sparse)
// =====================================================================

/// Run PCA on a pre-scaled dense matrix (already zero-mean, unit-variance, clipped).
///
/// Uses the same Halko-Martinsson-Tropp randomized SVD algorithm but operates
/// directly on a dense f32 matrix via ndarray `.dot()`. No centering is needed
/// because the input has already been centered and scaled.
///
/// # Arguments
/// * `scaled` - dense cells x genes Array2<f32> from `scale::scale_sparse`
/// * `gene_names` - gene identifiers (length = ncols)
/// * `n_components` - number of PCs to compute
/// * `n_power_iter` - power iteration rounds (3 recommended)
/// * `oversampling` - sketch oversampling (10 default)
/// * `seed` - RNG seed for reproducibility
pub fn run_pca_scaled(
    scaled: &ndarray::Array2<f32>,
    gene_names: &[String],
    n_components: usize,
    n_power_iter: usize,
    oversampling: usize,
    seed: u64,
) -> PcaResult {
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;

    let total_t0 = Instant::now();
    let n_cells = scaled.nrows();
    let n_genes = scaled.ncols();
    let k = (n_components + oversampling).min(n_genes).min(n_cells);

    eprintln!(
        "[pca-scaled] {}x{} dense matrix, {} components, {} power iterations",
        n_cells, n_genes, n_components, n_power_iter
    );

    let svd_t0 = Instant::now();

    // Step 1: Gaussian random matrix Omega (genes x k)
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let omega = Array2::from_shape_fn((n_genes, k), |_| {
        use rand::Rng;
        rng.sample::<f32, _>(StandardNormal)
    });

    // Step 2: Y = scaled * Omega  (cells x k)
    let mut y = scaled.dot(&omega);

    // Step 3: Power iteration
    for _iter in 0..n_power_iter {
        let (q_flat, _) = qr_mgs_f32(&y);
        let q = Array2::from_shape_vec((n_cells, k), q_flat).unwrap();

        // Z = scaled^T * Q  (genes x k)
        let z = scaled.t().dot(&q);

        let (z_q_flat, _) = qr_mgs_f32(&z);
        let z_q = Array2::from_shape_vec((n_genes, k), z_q_flat).unwrap();

        // Y = scaled * Z_Q
        y = scaled.dot(&z_q);
    }

    // Step 4: Final QR of Y
    let (q_flat, _) = qr_mgs_f32(&y);
    let q = Array2::from_shape_vec((n_cells, k), q_flat).unwrap();

    // Step 5: B = Q^T * scaled  (k x genes)
    let b_mat = q.t().dot(scaled); // k x genes, f32

    // Convert to f64 flat for small_svd
    let k_actual = k.min(n_genes).min(n_cells);
    let mut b_flat = vec![0.0f64; k_actual * n_genes];
    for i in 0..k_actual {
        for j in 0..n_genes {
            b_flat[i * n_genes + j] = b_mat[[i, j]] as f64;
        }
    }

    // Step 6: SVD of B (k x genes)
    let (u_b, sigma, _vt) = small_svd(&b_flat, k_actual, n_genes);

    // Step 7: PC scores = Q * U_B * Sigma
    let mut pc_scores: Vec<Vec<f64>> = Vec::with_capacity(n_cells);
    for c in 0..n_cells {
        let mut row = Vec::with_capacity(n_components);
        for pc in 0..n_components {
            let mut dot = 0.0f64;
            for j in 0..k_actual {
                dot += q[[c, j]] as f64 * u_b[j * k_actual + pc];
            }
            row.push(dot * sigma[pc]);
        }
        pc_scores.push(row);
    }

    // Step 8: Variance explained
    // Total variance = sum of squared values in the scaled matrix
    let total_var: f64 = scaled.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();

    let var_explained: Vec<f64> = if total_var > 0.0 {
        sigma[..n_components]
            .iter()
            .map(|s| (s * s) / total_var)
            .collect()
    } else {
        vec![0.0; n_components]
    };

    let svd_ms = svd_t0.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("[pca-scaled] SVD: {:.1}ms", svd_ms);
    eprintln!("[pca-scaled] total: {:.1}ms", total_ms);
    eprintln!(
        "[pca-scaled] variance captured: {:.2}%",
        var_explained.iter().sum::<f64>() * 100.0
    );

    PcaResult {
        pc_scores,
        variance_explained: var_explained,
        selected_genes: gene_names.to_vec(),
        timings: PcaTimings {
            variance_calc_ms: 0.0,
            gene_selection_ms: 0.0,
            centering_ms: 0.0,
            svd_ms,
            total_ms,
        },
    }
}

/// QR decomposition via modified Gram-Schmidt for f32 ndarray.
/// Returns (q_flat_f32, r_flat_f32) as Vec<f32>.
fn qr_mgs_f32(a: &ndarray::Array2<f32>) -> (Vec<f32>, Vec<f32>) {
    let m = a.nrows();
    let n = a.ncols();

    let mut cols: Vec<Vec<f32>> = (0..n).map(|j| a.column(j).to_vec()).collect();

    let mut r = vec![0.0f32; n * n];

    for j in 0..n {
        for i in 0..j {
            let dot: f32 = cols[j].iter().zip(cols[i].iter()).map(|(a, b)| a * b).sum();
            r[i * n + j] = dot;
            for row in 0..m {
                cols[j][row] -= dot * cols[i][row];
            }
        }
        let norm: f32 = cols[j].iter().map(|v| v * v).sum::<f32>().sqrt();
        r[j * n + j] = norm;
        if norm > 1e-7 {
            let inv = 1.0 / norm;
            for row in 0..m {
                cols[j][row] *= inv;
            }
        }
    }

    let mut q = vec![0.0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            q[i * n + j] = cols[j][i];
        }
    }
    (q, r)
}

// =====================================================================
//  Unit tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small matrix where the first gene has high variance and
    /// the second has low variance, so gene selection is deterministic.
    fn toy_matrix() -> (Vec<f64>, Vec<String>, Vec<String>, usize, usize) {
        // 3 genes x 4 samples
        let data = vec![
            10.0, 20.0, 30.0, 40.0, // gene A -- high variance
            1.0, 1.1, 1.2, 1.3, // gene B -- low variance
            5.0, 15.0, 25.0, 35.0, // gene C -- medium variance
        ];
        let genes = vec!["A".into(), "B".into(), "C".into()];
        let samples = vec!["s1".into(), "s2".into(), "s3".into(), "s4".into()];
        (data, genes, samples, 3, 4)
    }

    #[test]
    fn gene_selection_picks_high_variance() {
        let (data, genes, samples, ng, ns) = toy_matrix();
        let res = run_pca(&data, &genes, &samples, ng, ns, 2, 1);
        // Top-2 genes by variance should be A and C (not B).
        assert!(res.selected_genes.contains(&"A".to_string()));
        assert!(res.selected_genes.contains(&"C".to_string()));
        assert!(!res.selected_genes.contains(&"B".to_string()));
    }

    #[test]
    fn variance_explained_positive() {
        let (data, genes, samples, ng, ns) = toy_matrix();
        let res = run_pca(&data, &genes, &samples, ng, ns, 2, 1);
        // Variance explained should be positive
        for &v in &res.variance_explained {
            assert!(v >= 0.0, "variance explained should be >= 0, got {}", v);
        }
    }

    #[test]
    fn pc_scores_shape() {
        let (data, genes, samples, ng, ns) = toy_matrix();
        let res = run_pca(&data, &genes, &samples, ng, ns, 3, 2);
        assert_eq!(res.pc_scores.len(), ns);
        for row in &res.pc_scores {
            assert_eq!(row.len(), 2);
        }
    }

    #[test]
    fn matmul_identity() {
        // I (2x2) * [1,2; 3,4] should give [1,2; 3,4]
        let id = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = matmul(&id, &b, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-12);
        assert!((c[1] - 2.0).abs() < 1e-12);
        assert!((c[2] - 3.0).abs() < 1e-12);
        assert!((c[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn matmul_at_b_simple() {
        // A = [[1,0],[0,1],[1,1]]  (3x2)
        // B = [[1],[2],[3]]        (3x1)
        // A^T B = [[1*1+0*2+1*3],[0*1+1*2+1*3]] = [[4],[5]]
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = matmul_at_b(&a, &b, 3, 2, 1);
        assert!((c[0] - 4.0).abs() < 1e-12);
        assert!((c[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn qr_orthonormality() {
        let a = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]; // 3 x 2
        let (q, _r) = qr_mgs(&a, 3, 2);
        // Q^T Q should be ~ I (2x2)
        let qtq = matmul_at_b(&q, &q, 3, 2, 2);
        assert!((qtq[0] - 1.0).abs() < 1e-10);
        assert!((qtq[3] - 1.0).abs() < 1e-10);
        assert!(qtq[1].abs() < 1e-10);
        assert!(qtq[2].abs() < 1e-10);
    }

    #[test]
    fn small_svd_recovers_rank1() {
        // B = [[3, 0], [0, 0]]  -- rank 1, sigma = [3, 0]
        let b = vec![3.0, 0.0, 0.0, 0.0];
        let (_, sigma, _) = small_svd(&b, 2, 2);
        assert!((sigma[0] - 3.0).abs() < 1e-8, "sigma[0] = {}", sigma[0]);
        assert!(sigma[1].abs() < 1e-8, "sigma[1] = {}", sigma[1]);
    }
}
