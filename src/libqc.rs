//! Optional Seqera rustqc library QC sidecar.
//!
//! Spawns PATH `rustqc rna` into `<pipeline_output>/libqc/`. Never reads
//! rustqc featureCounts as the cell matrix.

use anyhow::{bail, Context, Result};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

const RUSTQC_BIN_ENV: &str = "RUSTPIPE_SC_RUSTQC";

pub fn rustqc_bin() -> PathBuf {
    std::env::var_os(RUSTQC_BIN_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("rustqc"))
}

/// argv after the binary: `rna <bam> --gtf <gtf> --outdir <libqc> --skip-dup-check`
/// No `--paired`: Cell Ranger BAMs are not the bulk PE contract.
pub fn build_rustqc_argv(bam: &Path, gtf: &Path, libqc_dir: &Path) -> Vec<OsString> {
    vec![
        OsString::from("rna"),
        bam.as_os_str().to_os_string(),
        OsString::from("--gtf"),
        gtf.as_os_str().to_os_string(),
        OsString::from("--outdir"),
        libqc_dir.as_os_str().to_os_string(),
        OsString::from("--skip-dup-check"),
    ]
}

pub fn run_libqc(bam: &Path, gtf: &Path, pipeline_output: &Path) -> Result<()> {
    let libqc_dir = pipeline_output.join("libqc");
    std::fs::create_dir_all(&libqc_dir)
        .with_context(|| format!("cannot create {}", libqc_dir.display()))?;

    let bin = rustqc_bin();
    let args = build_rustqc_argv(bam, gtf, &libqc_dir);
    let status = Command::new(&bin).args(&args).status().with_context(|| {
        format!(
            "failed to spawn rustqc ({bin}). Install Seqera rustqc on PATH or set {RUSTQC_BIN_ENV}",
            bin = bin.display()
        )
    })?;

    if !status.success() {
        bail!(
            "rustqc library QC failed with {}. featureCounts under {} is not the cell matrix",
            status,
            libqc_dir.display()
        );
    }

    eprintln!(
        "[libqc] rustqc rna wrote {} (ignore featurecounts/ as cell matrix)",
        libqc_dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn argv_strings(bam: &Path, gtf: &Path, out: &Path) -> Vec<String> {
        build_rustqc_argv(bam, gtf, out)
            .into_iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn argv_is_rna_sidecar_without_paired() {
        let args = argv_strings(
            Path::new("/data/possorted_genome_bam.bam"),
            Path::new("/ref/genes.gtf"),
            Path::new("/out/libqc"),
        );
        assert_eq!(args[0], "rna");
        assert_eq!(args[1], "/data/possorted_genome_bam.bam");
        assert!(args.contains(&"--gtf".to_string()));
        assert!(args.contains(&"--outdir".to_string()));
        assert!(args.contains(&"/out/libqc".to_string()));
        assert!(args.contains(&"--skip-dup-check".to_string()));
        assert!(!args.iter().any(|a| a == "--paired" || a == "-p"));
        assert!(!args.iter().any(|a| a.contains("featureCounts")));
    }

    #[test]
    fn missing_binary_errors_with_path_hint() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var(RUSTQC_BIN_ENV, dir.path().join("no-such-rustqc"));
        let err = run_libqc(
            Path::new("/data/possorted_genome_bam.bam"),
            Path::new("/ref/genes.gtf"),
            dir.path(),
        )
        .unwrap_err()
        .to_string();
        std::env::remove_var(RUSTQC_BIN_ENV);
        assert!(
            err.contains("PATH") || err.contains("RUSTPIPE_SC_RUSTQC") || err.contains("spawn"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn stub_binary_succeeds_and_does_not_require_featurecounts() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let stub = dir.path().join("fake-rustqc");
        std::fs::write(
            &stub,
            "#!/bin/sh\n# ignore featureCounts\nexit 0\n",
        )
        .unwrap();
        let mut perm = std::fs::metadata(&stub).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&stub, perm).unwrap();

        std::env::set_var(RUSTQC_BIN_ENV, &stub);
        run_libqc(
            Path::new("/data/possorted_genome_bam.bam"),
            Path::new("/ref/genes.gtf"),
            dir.path(),
        )
        .unwrap();
        std::env::remove_var(RUSTQC_BIN_ENV);
        assert!(dir.path().join("libqc").is_dir());
    }
}
