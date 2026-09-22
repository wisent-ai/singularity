#!/usr/bin/env bash
set -euo pipefail
: "${WISENT_SOURCE_DIR:?Stado must supply the immutable source directory}"
: "${WISENT_TEST_EVIDENCE_DIR:?Stado must retain qualification evidence}"
cargo test --manifest-path "$WISENT_SOURCE_DIR/Cargo.toml" --locked --all-targets
