#!/usr/bin/env bash
set -euo pipefail
: "${WISENT_SOURCE_DIR:?Stado must supply the immutable source directory}"
: "${WISENT_OUTPUT_DIR:?Stado must supply the release output directory}"
: "${CARGO_TARGET_DIR:?Stado must supply the product build cache}"
cargo build --manifest-path "$WISENT_SOURCE_DIR/Cargo.toml" --locked --release --bins
install -d "$WISENT_OUTPUT_DIR"
for binary in singularity singularity-bootstrap singularity-repo-mcp singularity-finance-mcp singularity-finance-executor-http; do
    install -m 755 "$CARGO_TARGET_DIR/release/$binary" "$WISENT_OUTPUT_DIR/$binary"
done
