#!/bin/sh
set -eu

: "${WISENT_VERSION:?WISENT_VERSION is required}"
: "${WISENT_SOURCE_DIR:?WISENT_SOURCE_DIR is required}"
: "${WISENT_OUTPUT_DIR:?WISENT_OUTPUT_DIR is required}"
: "${WISENT_PLATFORM:?WISENT_PLATFORM is required}"
: "${WISENT_INPUTS_DIR:?WISENT_INPUTS_DIR is required}"

case "$WISENT_PLATFORM" in
  darwin-arm64|linux-amd64) ;;
  *)
    echo "unsupported release platform: $WISENT_PLATFORM" >&2
    exit 64
    ;;
esac

mkdir -p "$WISENT_OUTPUT_DIR/cargo-target-$WISENT_PLATFORM"
rm -f "$WISENT_OUTPUT_DIR"/singularity*
export CARGO_INCREMENTAL=0
export SOURCE_DATE_EPOCH=1
export CARGO_TARGET_DIR="$WISENT_OUTPUT_DIR/cargo-target-$WISENT_PLATFORM"
cd "$WISENT_SOURCE_DIR"
cargo build --release --locked --manifest-path "$WISENT_SOURCE_DIR/Cargo.toml" --bins
for binary in singularity singularity-bootstrap singularity-repo-mcp singularity-finance-mcp singularity-finance-executor-http; do
  install -m 0755 "$CARGO_TARGET_DIR/release/$binary" "$WISENT_OUTPUT_DIR/$binary"
done
