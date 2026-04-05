//! H5AD (AnnData) and 10x Genomics H5 file I/O for single-cell RNA-seq.
//!
//! Supports two input formats:
//! - **H5AD** (AnnData): `/X/{data,indices,indptr}` CSR sparse matrix + `/obs` + `/var`
//! - **10x H5** (CellRanger): `/matrix/{data,indices,indptr}` CSC sparse + barcodes + features
//!
//! Both are read into our standard cells-x-genes CSR SpMat.

#[cfg(feature = "hdf5")]
use hdf5_metno as hdf5;

use crate::sparse::SpMat;
#[cfg(feature = "hdf5")]
use anyhow::Context;
use anyhow::{bail, Result};
use std::path::Path;
#[cfg(feature = "hdf5")]
use std::str::FromStr;

/// Minimal AnnData representation.
pub struct AnnData {
    #[allow(dead_code)]
    pub x: SpMat,
    #[allow(dead_code)]
    pub obs_names: Vec<String>,
    #[allow(dead_code)]
    pub var_names: Vec<String>,
    #[allow(dead_code)]
    pub obs: std::collections::HashMap<String, Vec<String>>,
    #[allow(dead_code)]
    pub var: std::collections::HashMap<String, Vec<String>>,
}

// =====================================================================
//  H5AD reading (AnnData format)
// =====================================================================

/// Read an H5AD file: X matrix (CSR) + obs_names (barcodes) + var_names (genes).
///
/// H5AD structure:
///   /X/data     — f32 or f64 non-zero values
///   /X/indices  — i32 or i64 column indices
///   /X/indptr   — i32 or i64 row pointers (len = n_cells + 1)
///   /obs        — compound dataset with "index" field (barcodes)
///   /var        — compound dataset with "index" field (gene names)
///
/// Falls back to dense /X if sparse arrays aren't found.
#[cfg(feature = "hdf5")]
#[allow(dead_code)]
pub fn read_h5ad_x(path: &Path) -> Result<(SpMat, Vec<String>, Vec<String>)> {
    let t0 = std::time::Instant::now();
    let file = hdf5::File::open(path)
        .with_context(|| format!("Cannot open H5AD file: {}", path.display()))?;

    // Read obs names (barcodes) from /obs compound dataset
    let obs_names =
        read_h5ad_index(&file, "obs").with_context(|| "Failed to read /obs barcodes")?;

    // Read var names (genes) from /var compound dataset
    let var_names =
        read_h5ad_index(&file, "var").with_context(|| "Failed to read /var gene names")?;

    let n_cells = obs_names.len();
    let n_genes = var_names.len();

    // Read sparse X matrix
    let x_group = file
        .group("X")
        .with_context(|| "No /X group found in H5AD")?;

    let data: Vec<f32> =
        read_dataset_as_f32(&x_group, "data").with_context(|| "Failed to read /X/data")?;
    let indices: Vec<usize> =
        read_dataset_as_usize(&x_group, "indices").with_context(|| "Failed to read /X/indices")?;
    let indptr: Vec<usize> =
        read_dataset_as_usize(&x_group, "indptr").with_context(|| "Failed to read /X/indptr")?;

    // Validate dimensions
    if indptr.len() != n_cells + 1 {
        bail!(
            "H5AD indptr length {} != n_cells + 1 ({}). Expected {} cells.",
            indptr.len(),
            n_cells + 1,
            n_cells
        );
    }
    if data.len() != indices.len() {
        bail!(
            "H5AD data length {} != indices length {}",
            data.len(),
            indices.len()
        );
    }

    // Build CSR matrix (H5AD stores as CSR by default)
    let mat = sprs::CsMat::new((n_cells, n_genes), indptr, indices, data);

    eprintln!(
        "[h5ad] read {} cells x {} genes, nnz={} in {:.3}s",
        n_cells,
        n_genes,
        mat.nnz(),
        t0.elapsed().as_secs_f64()
    );

    Ok((mat, obs_names, var_names))
}

/// Read the "index" field from an H5AD obs or var compound dataset.
///
/// H5AD stores /obs and /var as HDF5 compound datasets where the first
/// field (usually "index" or "_index") contains the names.
#[cfg(feature = "hdf5")]
fn read_h5ad_index(file: &hdf5::File, group_name: &str) -> Result<Vec<String>> {
    // Try /obs/_index or /var/_index first (newer AnnData format)
    let index_path = format!("{}/{}", group_name, "_index");
    if let Ok(ds) = file.dataset(&index_path) {
        return read_string_dataset(&ds);
    }

    // Try /obs/index or /var/index (another common location)
    let index_path2 = format!("{}/{}", group_name, "index");
    if let Ok(ds) = file.dataset(&index_path2) {
        return read_string_dataset(&ds);
    }

    // As a last resort, try reading the group dataset itself as a compound type
    // (older H5AD files store obs/var as compound datasets with an "index" field)
    if let Ok(ds) = file.dataset(group_name) {
        // Try direct string read first
        if let Ok(strings) = read_string_dataset(&ds) {
            return Ok(strings);
        }
        // Try reading compound dataset using HDF5 C API
        if let Ok(strings) = read_compound_index_field(&ds) {
            return Ok(strings);
        }
    }

    bail!(
        "Cannot read /{} names — tried _index, index, and compound parsing",
        group_name
    )
}

// =====================================================================
//  10x Genomics H5 reading (CellRanger format)
// =====================================================================

/// Read a 10x Genomics H5 file (CellRanger filtered_feature_bc_matrix.h5).
///
/// Structure:
///   /matrix/data     — i32 counts (CSC format: genes x cells)
///   /matrix/indices  — i32 row indices (gene indices)
///   /matrix/indptr   — i64 column pointers (len = n_cells + 1)
///   /matrix/barcodes — fixed-length strings
///   /matrix/features/name — gene symbols
///   /matrix/features/id   — Ensembl IDs
///   /matrix/shape    — [n_genes, n_cells]
///
/// Output is transposed to cells x genes CSR.
#[cfg(feature = "hdf5")]
pub fn read_10x_h5(path: &Path) -> Result<(SpMat, Vec<String>, Vec<String>)> {
    let t0 = std::time::Instant::now();
    let file = hdf5::File::open(path)
        .with_context(|| format!("Cannot open 10x H5 file: {}", path.display()))?;

    let matrix = file
        .group("matrix")
        .with_context(|| "No /matrix group found — is this a CellRanger H5 file?")?;

    // Read shape: [n_genes, n_cells] (CSC is genes x cells)
    let shape: Vec<i64> = matrix.dataset("shape")?.read_raw()?;
    let n_genes = shape[0] as usize;
    let n_cells = shape[1] as usize;

    // Read barcodes
    let barcodes_ds = matrix.dataset("barcodes")?;
    let barcodes = read_fixed_strings(&barcodes_ds)?;

    // Read gene names (prefer "name" over "id")
    let features = matrix.group("features")?;
    let gene_names = if let Ok(ds) = features.dataset("name") {
        read_fixed_strings(&ds)?
    } else {
        read_fixed_strings(&features.dataset("id")?)?
    };

    // Read CSC data (genes x cells)
    let data_raw: Vec<i32> = matrix.dataset("data")?.read_raw()?;
    let data: Vec<f32> = data_raw.iter().map(|&v| v as f32).collect();
    let indices: Vec<usize> = {
        let raw: Vec<i32> = matrix.dataset("indices")?.read_raw()?;
        raw.iter().map(|&v| v as usize).collect()
    };
    let indptr: Vec<usize> = {
        // indptr can be i32 or i64 depending on CellRanger version
        if let Ok(raw) = matrix.dataset("indptr")?.read_1d::<i64>() {
            raw.iter().map(|&v| v as usize).collect()
        } else {
            let raw: Vec<i32> = matrix.dataset("indptr")?.read_raw()?;
            raw.iter().map(|&v| v as usize).collect()
        }
    };

    // Convert CSC (genes x cells) to CSR (cells x genes).
    // Build CSR directly: each CSC column j corresponds to cell j's entries.
    // For each cell, collect its (gene_idx, value) pairs, sort by gene_idx,
    // and append to CSR arrays.
    let mut csr_indptr: Vec<usize> = Vec::with_capacity(n_cells + 1);
    let mut csr_indices: Vec<usize> = Vec::with_capacity(data.len());
    let mut csr_data: Vec<f32> = Vec::with_capacity(data.len());
    csr_indptr.push(0);

    for j in 0..n_cells {
        let start = indptr[j];
        let end = indptr[j + 1];
        // Collect (gene_idx, value) pairs for this cell
        let mut pairs: Vec<(usize, f32)> = (start..end).map(|k| (indices[k], data[k])).collect();
        pairs.sort_unstable_by_key(|&(idx, _)| idx);
        for (idx, val) in pairs {
            csr_indices.push(idx);
            csr_data.push(val);
        }
        csr_indptr.push(csr_indices.len());
    }

    let mat = sprs::CsMat::new((n_cells, n_genes), csr_indptr, csr_indices, csr_data);

    eprintln!(
        "[10x-h5] read {} cells x {} genes, nnz={} in {:.3}s",
        n_cells,
        n_genes,
        mat.nnz(),
        t0.elapsed().as_secs_f64()
    );

    Ok((mat, barcodes, gene_names))
}

// =====================================================================
//  Helper functions
// =====================================================================

/// Read an HDF5 dataset as Vec<f32>, handling f32 and f64 source types.
#[cfg(feature = "hdf5")]
fn read_dataset_as_f32(group: &hdf5::Group, name: &str) -> Result<Vec<f32>> {
    let ds = group.dataset(name)?;
    // Try f32 first (most common for scRNA-seq)
    if let Ok(data) = ds.read_1d::<f32>() {
        return Ok(data.to_vec());
    }
    // Fall back to f64
    if let Ok(data) = ds.read_1d::<f64>() {
        return Ok(data.iter().map(|&v| v as f32).collect());
    }
    // Fall back to i32 (raw counts)
    if let Ok(data) = ds.read_1d::<i32>() {
        return Ok(data.iter().map(|&v| v as f32).collect());
    }
    bail!("Cannot read {}/{} — unsupported dtype", group.name(), name)
}

/// Read an HDF5 dataset as Vec<usize>, handling i32 and i64 source types.
#[cfg(feature = "hdf5")]
fn read_dataset_as_usize(group: &hdf5::Group, name: &str) -> Result<Vec<usize>> {
    let ds = group.dataset(name)?;
    if let Ok(data) = ds.read_1d::<i32>() {
        return Ok(data.iter().map(|&v| v as usize).collect());
    }
    if let Ok(data) = ds.read_1d::<i64>() {
        return Ok(data.iter().map(|&v| v as usize).collect());
    }
    bail!("Cannot read {}/{} as integer array", group.name(), name)
}

/// Read a dataset of fixed-length strings using HDF5 C API.
///
/// The hdf5-metno safe API can't read fixed-length string datasets as u8 bytes,
/// so we use H5Dread directly with the file's native type.
#[cfg(feature = "hdf5")]
fn read_fixed_strings(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    let shape = ds.shape();
    let n = shape[0];
    let dtype = ds.dtype()?;
    let elem_size = dtype.size();
    let total_bytes = n * elem_size;

    let mut buf: Vec<u8> = vec![0u8; total_bytes];

    unsafe {
        const H5S_ALL: i64 = 0;
        const H5P_DEFAULT: i64 = 0;

        extern "C" {
            fn H5Dread(
                dset_id: i64,
                mem_type_id: i64,
                mem_space_id: i64,
                file_space_id: i64,
                xfer_plist_id: i64,
                buf: *mut std::ffi::c_void,
            ) -> i32;
            fn H5Dget_type(dset_id: i64) -> i64;
            fn H5Tclose(type_id: i64) -> i32;
        }

        let ds_id = ds.id();
        let file_type = H5Dget_type(ds_id);
        if file_type < 0 {
            bail!("H5Dget_type failed for fixed string dataset");
        }

        let status = H5Dread(
            ds_id,
            file_type,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            buf.as_mut_ptr() as *mut std::ffi::c_void,
        );

        H5Tclose(file_type);

        if status < 0 {
            bail!("H5Dread failed for fixed string dataset");
        }
    }

    let mut strings = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * elem_size;
        let end = start + elem_size;
        let bytes = &buf[start..end];
        let s = std::str::from_utf8(bytes)
            .unwrap_or("")
            .trim_end_matches('\0')
            .to_string();
        strings.push(s);
    }
    Ok(strings)
}

/// Read the "index" field from a compound dataset using HDF5 C API.
///
/// H5AD obs/var are often compound datasets like `[('index', 'S16')]`.
/// The hdf5-metno safe API can't read compound types as raw bytes,
/// so we use the C API directly: H5Dread with the file's native type.
#[cfg(feature = "hdf5")]
fn read_compound_index_field(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    use hdf5::types::TypeDescriptor;

    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor()?;

    // Must be a compound type
    let compound = match desc {
        TypeDescriptor::Compound(ref ct) => ct,
        _ => bail!("Dataset is not a compound type"),
    };

    // Find the "index" field (first field, or named "index")
    let index_field = compound
        .fields
        .iter()
        .find(|f| f.name == "index" || f.name == "_index")
        .or_else(|| compound.fields.first())
        .ok_or_else(|| anyhow::anyhow!("No fields in compound dataset"))?;

    let field_offset = index_field.offset;
    let field_size = match &index_field.ty {
        TypeDescriptor::FixedAscii(n) => *n,
        TypeDescriptor::FixedUnicode(n) => *n,
        other => bail!("Index field has unexpected type: {:?}", other),
    };
    let record_size = compound.size;

    let shape = ds.shape();
    let n_items = shape[0];

    // Allocate buffer and read raw bytes using HDF5 C API.
    // hdf5-metno's safe API can't read compound types as raw bytes,
    // so we call H5Dread directly with the file's native type.
    let total_bytes = n_items * record_size;
    let mut buf: Vec<u8> = vec![0u8; total_bytes];

    unsafe {
        // HDF5 C API constants: H5S_ALL = 0, H5P_DEFAULT = 0
        const H5S_ALL: i64 = 0;
        const H5P_DEFAULT: i64 = 0;

        extern "C" {
            fn H5Dread(
                dset_id: i64,
                mem_type_id: i64,
                mem_space_id: i64,
                file_space_id: i64,
                xfer_plist_id: i64,
                buf: *mut std::ffi::c_void,
            ) -> i32;
            fn H5Dget_type(dset_id: i64) -> i64;
            fn H5Tclose(type_id: i64) -> i32;
        }

        let ds_id = ds.id();
        let file_type = H5Dget_type(ds_id);
        if file_type < 0 {
            bail!("H5Dget_type failed");
        }

        let status = H5Dread(
            ds_id,
            file_type,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            buf.as_mut_ptr() as *mut std::ffi::c_void,
        );

        H5Tclose(file_type);

        if status < 0 {
            bail!("H5Dread failed for compound dataset");
        }
    }

    // Parse fixed-length strings from the buffer
    let mut names = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let start = i * record_size + field_offset;
        let end = start + field_size;
        let bytes = &buf[start..end];
        let s = std::str::from_utf8(bytes)
            .unwrap_or("")
            .trim_end_matches('\0')
            .to_string();
        names.push(s);
    }

    Ok(names)
}

/// Read a string dataset (variable or fixed length).
#[cfg(feature = "hdf5")]
fn read_string_dataset(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    // Try reading as variable-length strings first
    if let Ok(data) = ds.read_1d::<hdf5::types::VarLenUnicode>() {
        return Ok(data.iter().map(|s| s.to_string()).collect());
    }
    // Fall back to fixed-length
    read_fixed_strings(ds)
}

// =====================================================================
//  Write support (minimal, for round-trip testing)
// =====================================================================

/// Write AnnData to H5AD format.
#[cfg(feature = "hdf5")]
pub fn write_h5ad(adata: &AnnData, path: &Path) -> Result<()> {
    let file = hdf5::File::create(path)
        .with_context(|| format!("Cannot create H5AD file: {}", path.display()))?;

    let n_cells = adata.obs_names.len();
    let n_genes = adata.var_names.len();

    // Write X as CSR sparse
    let x_group = file.create_group("X")?;

    // Get CSR components from sprs
    let indptr_view = adata.x.indptr();
    let indptr_slice = indptr_view.as_slice().unwrap();
    let indices_slice = adata.x.indices();
    let data_slice = adata.x.data();

    x_group
        .new_dataset_builder()
        .with_data(&indptr_slice)
        .create("indptr")?;
    x_group
        .new_dataset_builder()
        .with_data(indices_slice)
        .create("indices")?;
    x_group
        .new_dataset_builder()
        .with_data(data_slice)
        .create("data")?;

    // Write shape attribute
    let shape_data = [n_cells as i64, n_genes as i64];
    x_group
        .new_attr_builder()
        .with_data(&shape_data)
        .create("h5sparse_shape")?;

    // Write obs names
    let obs_group = file.create_group("obs")?;
    let obs_strings: Vec<hdf5::types::VarLenUnicode> = adata
        .obs_names
        .iter()
        .map(|s| hdf5::types::VarLenUnicode::from_str(s).unwrap())
        .collect();
    obs_group
        .new_dataset_builder()
        .with_data(&obs_strings)
        .create("_index")?;

    // Write var names
    let var_group = file.create_group("var")?;
    let var_strings: Vec<hdf5::types::VarLenUnicode> = adata
        .var_names
        .iter()
        .map(|s| hdf5::types::VarLenUnicode::from_str(s).unwrap())
        .collect();
    var_group
        .new_dataset_builder()
        .with_data(&var_strings)
        .create("_index")?;

    eprintln!(
        "[h5ad] wrote {} cells x {} genes to {}",
        n_cells,
        n_genes,
        path.display()
    );
    Ok(())
}

// =====================================================================
//  Fallback stubs when HDF5 feature is disabled
// =====================================================================

#[cfg(not(feature = "hdf5"))]
#[allow(dead_code)]
pub fn read_h5ad_x(_path: &Path) -> Result<(SpMat, Vec<String>, Vec<String>)> {
    bail!(
        "H5AD support requires the 'hdf5' feature. \
         Build with: cargo build --features hdf5\n\
         Requires system HDF5 library (brew install hdf5)"
    )
}

#[cfg(not(feature = "hdf5"))]
#[allow(dead_code)]
pub fn read_10x_h5(_path: &Path) -> Result<(SpMat, Vec<String>, Vec<String>)> {
    bail!(
        "10x H5 support requires the 'hdf5' feature. \
         Build with: cargo build --features hdf5\n\
         Requires system HDF5 library (brew install hdf5)"
    )
}

#[cfg(not(feature = "hdf5"))]
pub fn write_h5ad(_adata: &AnnData, _path: &Path) -> Result<()> {
    bail!("H5AD writing requires the 'hdf5' feature.")
}

#[cfg(not(feature = "hdf5"))]
#[allow(dead_code)]
pub fn read_h5ad(_path: &Path) -> Result<AnnData> {
    bail!("H5AD reading requires the 'hdf5' feature.")
}

// =====================================================================
//  Auto-detection: H5AD vs 10x H5
// =====================================================================

/// Detect whether an H5 file is H5AD (AnnData) or 10x CellRanger format
/// and read accordingly.
#[cfg(feature = "hdf5")]
pub fn read_h5_auto(path: &Path) -> Result<(SpMat, Vec<String>, Vec<String>)> {
    let file = hdf5::File::open(path)
        .with_context(|| format!("Cannot open H5 file: {}", path.display()))?;

    // 10x CellRanger has /matrix group; H5AD has /X group
    if file.group("matrix").is_ok() {
        eprintln!("[io] detected 10x CellRanger H5 format");
        drop(file);
        read_10x_h5(path)
    } else if file.group("X").is_ok() {
        eprintln!("[io] detected H5AD (AnnData) format");
        drop(file);
        read_h5ad_x(path)
    } else {
        bail!(
            "Unrecognized H5 format: {} (expected /X for H5AD or /matrix for 10x)",
            path.display()
        )
    }
}

#[cfg(not(feature = "hdf5"))]
pub fn read_h5_auto(path: &Path) -> Result<(SpMat, Vec<String>, Vec<String>)> {
    bail!(
        "HDF5 support requires the 'hdf5' feature. \
         Build with: cargo build --features hdf5\n\
         Input file: {}",
        path.display()
    )
}
