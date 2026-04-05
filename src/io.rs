use crate::sparse::{self, SpMat};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

/// Read an expression matrix CSV with the layout:
///   gene_symbol, sample_1, sample_2, ..., sample_N
///   GENE_A,      5.41,     5.66,     ..., 5.28
///   ...
///
/// Returns (gene_names, sample_names, flat_data_row_major, n_genes, n_samples).
/// flat_data[i * n_samples + j] is the expression of gene i in sample j.
pub fn read_expression_matrix(
    path: &str,
) -> Result<(Vec<String>, Vec<String>, Vec<f64>, usize, usize), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let file =
        File::open(path).map_err(|e| format!("Cannot open expression matrix '{}': {}", path, e))?;

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    // --- parse header ---
    let header = rdr
        .headers()
        .map_err(|e| format!("Cannot read header row in '{}': {}", path, e))?
        .clone();

    // First column is gene_symbol; the rest are sample IDs.
    if header.len() < 2 {
        return Err(format!(
            "Expression matrix '{}' must have at least 2 columns (gene_symbol + 1 sample)",
            path
        )
        .into());
    }

    let sample_names: Vec<String> = header.iter().skip(1).map(|s| s.to_owned()).collect();
    let n_samples = sample_names.len();

    // --- parse rows ---
    let mut gene_names: Vec<String> = Vec::with_capacity(20_000);
    // Pre-allocate for the expected 18,168 x 3,211 = ~58M floats (~465 MB f64)
    let mut data: Vec<f64> = Vec::with_capacity(20_000 * n_samples);

    for (row_idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| {
            format!(
                "Parse error in '{}' at data row {}: {}",
                path,
                row_idx + 1,
                e
            )
        })?;

        if record.len() != n_samples + 1 {
            return Err(format!(
                "'{}' row {}: expected {} fields, got {}",
                path,
                row_idx + 2, // +1 for header, +1 for 1-based
                n_samples + 1,
                record.len()
            )
            .into());
        }

        gene_names.push(record[0].to_owned());

        for col_idx in 1..=n_samples {
            let val: f64 = record[col_idx].trim().parse().map_err(|e| {
                format!(
                    "'{}' row {} col {}: cannot parse '{}' as f64: {}",
                    path,
                    row_idx + 2,
                    col_idx + 1,
                    &record[col_idx],
                    e
                )
            })?;
            data.push(val);
        }
    }

    let n_genes = gene_names.len();
    if n_genes == 0 {
        return Err(format!("Expression matrix '{}' contains no data rows", path).into());
    }

    let elapsed = t0.elapsed();
    eprintln!(
        "[io] read_expression_matrix: {}x{} in {:.3}s  ({:.0} MB flat vec)",
        n_genes,
        n_samples,
        elapsed.as_secs_f64(),
        (data.len() * std::mem::size_of::<f64>()) as f64 / 1_048_576.0
    );

    Ok((gene_names, sample_names, data, n_genes, n_samples))
}

/// Read a 2-column CSV (gene, statistic), pre-sorted descending by statistic.
/// Returns Vec<(gene_name, statistic)>.
pub fn read_ranked_list(path: &str) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let file =
        File::open(path).map_err(|e| format!("Cannot open ranked list '{}': {}", path, e))?;

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut ranks: Vec<(String, f64)> = Vec::with_capacity(20_000);

    for (row_idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| {
            format!(
                "Parse error in ranked list '{}' at row {}: {}",
                path,
                row_idx + 2,
                e
            )
        })?;

        if record.len() < 2 {
            return Err(format!(
                "Ranked list '{}' row {}: expected 2 columns (gene, statistic), got {}",
                path,
                row_idx + 2,
                record.len()
            )
            .into());
        }

        let gene = record[0].trim().to_owned();
        let stat: f64 = record[1].trim().parse().map_err(|e| {
            format!(
                "Ranked list '{}' row {}: cannot parse statistic '{}' as f64: {}",
                path,
                row_idx + 2,
                &record[1],
                e
            )
        })?;

        ranks.push((gene, stat));
    }

    let elapsed = t0.elapsed();
    eprintln!(
        "[io] read_ranked_list: {} genes in {:.3}s",
        ranks.len(),
        elapsed.as_secs_f64()
    );

    Ok(ranks)
}

/// Read a GMT file (Broad/MSigDB format).
/// Each line: pathway_name\tdescription\tgene1\tgene2\t...
/// Returns Vec<(pathway_name, gene_list)>.
pub fn read_gmt(path: &str) -> Result<Vec<(String, Vec<String>)>, Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let file = File::open(path).map_err(|e| format!("Cannot open GMT file '{}': {}", path, e))?;

    let reader = BufReader::new(file);
    let mut pathways: Vec<(String, Vec<String>)> = Vec::new();

    for (line_idx, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| {
            format!(
                "Read error in GMT '{}' at line {}: {}",
                path,
                line_idx + 1,
                e
            )
        })?;

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            return Err(format!(
                "GMT '{}' line {}: expected at least 3 tab-separated fields \
                 (name, description, gene...), got {}",
                path,
                line_idx + 1,
                fields.len()
            )
            .into());
        }

        let name = fields[0].trim().to_owned();
        // fields[1] is the description — skip it, keep genes only
        let genes: Vec<String> = fields[2..]
            .iter()
            .map(|g| g.trim().to_owned())
            .filter(|g| !g.is_empty())
            .collect();

        if genes.is_empty() {
            eprintln!(
                "[io] read_gmt: warning — pathway '{}' has no genes (line {})",
                name,
                line_idx + 1
            );
            continue;
        }

        pathways.push((name, genes));
    }

    let elapsed = t0.elapsed();
    eprintln!(
        "[io] read_gmt: {} pathways in {:.3}s",
        pathways.len(),
        elapsed.as_secs_f64()
    );

    Ok(pathways)
}

/// Read a cells x genes CSV (Scanpy export format) into a sparse matrix.
///
/// CSV format: first column is cell barcode, remaining columns are gene names.
/// Most values are zero (single-cell dropout). Returns sparse CSR matrix.
pub fn read_sparse_csv(
    path: &str,
) -> Result<(SpMat, Vec<String>, Vec<String>), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let file = File::open(path).map_err(|e| format!("Cannot open sparse CSV '{}': {}", path, e))?;

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let header = rdr.headers()?.clone();
    let gene_names: Vec<String> = header.iter().skip(1).map(|s| s.to_owned()).collect();
    let n_genes = gene_names.len();

    let mut cell_names: Vec<String> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();

    for (row_idx, result) in rdr.records().enumerate() {
        let record = result?;
        cell_names.push(record[0].to_owned());

        for col_idx in 0..n_genes {
            let val: f32 = record[col_idx + 1].trim().parse().unwrap_or(0.0);
            if val != 0.0 {
                rows.push(row_idx);
                cols.push(col_idx);
                vals.push(val);
            }
        }
    }

    let n_cells = cell_names.len();
    let mat = sparse::from_triplets(n_cells, n_genes, &rows, &cols, &vals);

    eprintln!(
        "[io] read_sparse_csv: {}x{}, nnz={} in {:.3}s",
        n_cells,
        n_genes,
        mat.nnz(),
        t0.elapsed().as_secs_f64()
    );

    Ok((mat, cell_names, gene_names))
}

/// Read spatial coordinates CSV: columns are barcode, x, y.
pub fn read_spatial_coords(
    path: &str,
) -> Result<(Vec<String>, Vec<(f64, f64)>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut names = Vec::new();
    let mut coords = Vec::new();

    for result in rdr.records() {
        let record = result?;
        names.push(record[0].to_owned());
        let x: f64 = record[1].trim().parse()?;
        let y: f64 = record[2].trim().parse()?;
        coords.push((x, y));
    }

    Ok((names, coords))
}

/// Read the genes x cells CSV format (rustpipe-sc legacy: genes as rows).
/// Transposes to cells x genes sparse matrix.
pub fn read_expression_to_sparse(
    path: &str,
) -> Result<(SpMat, Vec<String>, Vec<String>), Box<dyn std::error::Error>> {
    let (gene_names, sample_names, data, n_genes, n_samples) = read_expression_matrix(path)?;

    // data is genes x samples row-major. Convert to cells x genes sparse.
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for g in 0..n_genes {
        for s in 0..n_samples {
            let val = data[g * n_samples + s] as f32;
            if val != 0.0 {
                rows.push(s); // cell index
                cols.push(g); // gene index
                vals.push(val);
            }
        }
    }

    let mat = sparse::from_triplets(n_samples, n_genes, &rows, &cols, &vals);
    Ok((mat, sample_names, gene_names))
}

/// Generic CSV writer.
/// `headers` — column header strings.
/// `data`    — each inner Vec<f64> is one row; must match headers.len().
#[allow(dead_code)]
pub fn write_csv_matrix(
    path: &str,
    headers: &[&str],
    data: &[Vec<f64>],
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let mut file =
        File::create(path).map_err(|e| format!("Cannot create output file '{}': {}", path, e))?;

    // Write header row
    writeln!(file, "{}", headers.join(","))
        .map_err(|e| format!("Write error to '{}': {}", path, e))?;

    // Write data rows
    for (row_idx, row) in data.iter().enumerate() {
        if row.len() != headers.len() {
            return Err(format!(
                "write_csv_matrix: row {} has {} values but {} headers were provided",
                row_idx,
                row.len(),
                headers.len()
            )
            .into());
        }

        let fields: Vec<String> = row.iter().map(|v| format!("{:.6}", v)).collect();
        writeln!(file, "{}", fields.join(","))
            .map_err(|e| format!("Write error to '{}' at row {}: {}", path, row_idx, e))?;
    }

    let elapsed = t0.elapsed();
    eprintln!(
        "[io] write_csv_matrix: {} rows x {} cols to '{}' in {:.3}s",
        data.len(),
        headers.len(),
        path,
        elapsed.as_secs_f64()
    );

    Ok(())
}
