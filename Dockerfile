FROM rust:1.88-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo build --release && strip target/release/rustpipe-sc

FROM debian:bookworm-slim
LABEL org.opencontainers.image.source="https://github.com/jwg054000/RustPipe-SC"
LABEL org.opencontainers.image.description="Fast single-cell and spatial transcriptomics in Rust"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL org.opencontainers.image.version="0.3.0"
COPY --from=builder /build/target/release/rustpipe-sc /usr/local/bin/rustpipe-sc
RUN rustpipe-sc --version
ENTRYPOINT ["rustpipe-sc"]
