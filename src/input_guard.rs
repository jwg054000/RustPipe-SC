//! Fail-closed input checks for the 10x / scRNA lane.
//!
//! `--input` is a cell-count matrix (Cell Ranger H5, H5AD, or cell CSV).
//! A bulk STAR BAM or rustqc featureCounts TSV is the wrong assay.

use anyhow::{bail, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const FEATURECOUNTS_COLS: [&str; 6] = ["Geneid", "Chr", "Start", "End", "Strand", "Length"];

/// Reject bulk STAR / rustqc files handed in as the cell matrix.
pub fn reject_wrong_assay(path: &Path) -> Result<()> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if name.ends_with(".bam") || name.ends_with(".sam") || name.ends_with(".cram") {
        bail!(
            "refusing alignment file as --input ({}). \
             RustPipe-SC consumes Cell Ranger H5 / H5AD / cell-count CSV, not BAM/SAM/CRAM. \
             A bulk STAR BAM belongs in prairie-rna-stream. \
             A Cell Ranger possorted_genome_bam.bam is optional --qc-bam (library QC), never the cell matrix.",
            path.display()
        );
    }

    if name.ends_with(".mtx") {
        bail!(
            "refusing MTX as --input ({}). \
             Use Cell Ranger filtered_feature_bc_matrix.h5 or an H5AD from nf-core/scrnaseq. \
             This crate does not read the MTX directory.",
            path.display()
        );
    }

    if name.contains("featurecounts") {
        bail!("{}", featurecounts_msg(path));
    }

    if should_peek_text(&name) {
        if let Some(reason) = peek_featurecounts(path)? {
            bail!("{} ({})", featurecounts_msg(path), reason);
        }
    }

    Ok(())
}

fn should_peek_text(lower_name: &str) -> bool {
    !lower_name.ends_with(".h5")
        && !lower_name.ends_with(".h5ad")
        && !lower_name.ends_with(".bam")
        && !lower_name.ends_with(".sam")
        && !lower_name.ends_with(".cram")
        && !lower_name.ends_with(".mtx")
}

fn featurecounts_msg(path: &Path) -> String {
    format!(
        "refusing rustqc/featureCounts table as --input ({}). \
         That file is bulk library QC (Geneid/Chr/Start/End/Strand/Length), not a cell-count matrix. \
         Use filtered_feature_bc_matrix.h5 or an H5AD from Cell Ranger / nf-core/scrnaseq.",
        path.display()
    )
}

fn peek_featurecounts(path: &Path) -> Result<Option<&'static str>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    for _ in 0..8 {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            if trimmed.to_ascii_lowercase().contains("featurecounts") {
                return Ok(Some("comment line names featureCounts"));
            }
            continue;
        }
        if looks_like_featurecounts_header(trimmed) {
            return Ok(Some("header has Geneid/Chr/Start/End/Strand/Length"));
        }
        break;
    }
    Ok(None)
}

fn looks_like_featurecounts_header(header: &str) -> bool {
    let cols: Vec<&str> = header
        .split(['\t', ','])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    FEATURECOUNTS_COLS
        .iter()
        .all(|need| cols.iter().any(|got| got.eq_ignore_ascii_case(need)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_with(name: &str, body: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .prefix("rp_sc_")
            .suffix(name)
            .tempfile()
            .unwrap();
        f.write_all(body.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn rejects_star_bam_by_extension() {
        let err = reject_wrong_assay(Path::new("WT_REP1.Aligned.sortedByCoord.out.bam"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("alignment file"), "{err}");
        assert!(err.contains("prairie-rna-stream"), "{err}");
    }

    #[test]
    fn rejects_cellranger_bam_as_input() {
        let err = reject_wrong_assay(Path::new("possorted_genome_bam.bam"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--qc-bam"), "{err}");
    }

    #[test]
    fn rejects_featurecounts_filename() {
        let err = reject_wrong_assay(Path::new("RAP1_IAA.featureCounts.tsv"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("featureCounts"), "{err}");
    }

    #[test]
    fn rejects_featurecounts_header() {
        let f = tmp_with(
            ".tsv",
            "Geneid\tChr\tStart\tEnd\tStrand\tLength\tWT.bam\nYAL069W\tI\t335\t649\t+\t315\t19\n",
        );
        let err = reject_wrong_assay(f.path()).unwrap_err().to_string();
        assert!(err.contains("featureCounts"), "{err}");
    }

    #[test]
    fn rejects_featurecounts_comment() {
        let f = tmp_with(
            ".tsv",
            "# Program:featureCounts v2.0.6, generated by RustQC v0.2.1\n\
             Geneid\tChr\tStart\tEnd\tStrand\tLength\tWT.bam\n",
        );
        let err = reject_wrong_assay(f.path()).unwrap_err().to_string();
        assert!(err.contains("featureCounts"), "{err}");
    }

    #[test]
    fn rejects_mtx() {
        let err = reject_wrong_assay(Path::new("matrix.mtx"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("MTX"), "{err}");
    }

    #[test]
    fn accepts_cell_csv() {
        let f = tmp_with(
            ".csv",
            "barcode,GENE_A,GENE_B\nAAACCTGAGAAACCAT-1,1,0\n",
        );
        reject_wrong_assay(f.path()).unwrap();
    }

    #[test]
    fn does_not_peek_h5_as_text() {
        reject_wrong_assay(Path::new("filtered_feature_bc_matrix.h5")).unwrap();
        reject_wrong_assay(Path::new("adata.h5ad")).unwrap();
    }
}
