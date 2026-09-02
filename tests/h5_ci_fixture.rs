//! Exercise `rustpipe-sc pipeline` on the committed 10x H5 fixture.
//!
//! Gated on `--features hdf5`. Without HDF5 the binary cannot read `.h5`.

#[cfg(feature = "hdf5")]
mod with_hdf5 {
    use std::path::PathBuf;
    use std::process::Command;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/filtered_feature_bc_matrix.h5")
    }

    #[test]
    fn pipeline_h5_writes_packet_artifacts() {
        let fixture = fixture();
        assert!(
            fixture.is_file(),
            "missing committed 10x fixture at {}",
            fixture.display()
        );

        let out = tempfile::tempdir().expect("tempdir");
        let status = Command::new(env!("CARGO_BIN_EXE_rustpipe-sc"))
            .args([
                "pipeline",
                "--input",
                fixture.to_str().unwrap(),
                "--output",
                out.path().to_str().unwrap(),
            ])
            .status()
            .expect("spawn rustpipe-sc");
        assert!(status.success(), "pipeline exited {status}");

        for name in [
            "qc_metrics.csv",
            "pca_scores.csv",
            "clusters.csv",
            "markers.csv",
            "pipeline_timings.json",
        ] {
            let path = out.path().join(name);
            assert!(path.is_file(), "PACKET.md artifact missing: {name}");
            assert!(
                path.metadata().unwrap().len() > 0,
                "PACKET.md artifact empty: {name}"
            );
        }

        let timings = std::fs::read_to_string(out.path().join("pipeline_timings.json")).unwrap();
        assert!(
            timings.contains("\"pipeline\": \"rustpipe-sc\""),
            "pipeline_timings.json must name rustpipe-sc"
        );
    }
}
