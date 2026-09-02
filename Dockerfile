# linux/amd64 image for ghcr.io/jwg054000/rustpipe-sc.
# Runtime copies the stripped binary only (no rustc). HDF5 is Debian bookworm
# 1.10 via apt — not Homebrew HDF5 2.x.
#
# Downstream pipelines pull the published digest. Do not rebuild this image
# inside prairie-rna-stream and do not vendor this repo there.

FROM rust:1.88-bookworm AS builder
WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends libhdf5-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo build --release --features hdf5 && strip target/release/rustpipe-sc

FROM debian:bookworm-slim@sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867
LABEL org.opencontainers.image.source="https://github.com/jwg054000/RustPipe-SC"
LABEL org.opencontainers.image.description="Fast single-cell RNA-seq downstream of Cell Ranger / nf-core/scrnaseq"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL org.opencontainers.image.version="0.3.0"
RUN apt-get update \
    && apt-get install -y --no-install-recommends libhdf5-103-1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/rustpipe-sc /usr/local/bin/rustpipe-sc
RUN rustpipe-sc --version
ENTRYPOINT ["rustpipe-sc"]
