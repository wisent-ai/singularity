#!/bin/sh
# Bootstrap a being in a throwaway sandbox with no Brama gateway, no Las
# checkout, and no credentials — the executed transcript behind
# walkthrough-first-cycle.md. Every refusal is printed with its exit code.
#
# Usage: SINGULARITY_BIN=target/release/singularity docs/examples/sandbox-being.sh
set -u

BIN=${SINGULARITY_BIN:-singularity}
STUB=$(cd "$(dirname "$0")" && pwd)/stub-las.py
SBX=$(mktemp -d /tmp/singularity-walkthrough.XXXXXX)
mkdir -p "$SBX/workspace" "$SBX/las"
printf 'walkthrough-not-a-real-secret\n' > "$SBX/brama.hmac"
chmod 600 "$SBX/brama.hmac"
cp "$STUB" "$SBX/las/stub-las.py"
for f in release.manifest.json release.manifest.sig release.trust release.watermark; do
  printf 'stub\n' > "$SBX/las/$f"
done

hex() { python3 -c "print('$1' * 64)"; }

export SINGULARITY_AGENT_ID=walkthrough-being SINGULARITY_ROLE=walkthrough \
  SINGULARITY_ENVIRONMENT=local-walkthrough SINGULARITY_HOST=local \
  SINGULARITY_WORKLOAD_ID=walkthrough-workload \
  SINGULARITY_WORKLOAD_PUBLIC_KEY="$(hex a)" SINGULARITY_EXECUTABLE_SHA256="$(hex b)" \
  SINGULARITY_CODE_SHA256="$(hex c)" SINGULARITY_POLICY_SHA256="$(hex d)" \
  SINGULARITY_POLICY_SEQUENCE=1 \
  SINGULARITY_STATE_DIR="$SBX/state" SINGULARITY_WORKSPACE="$SBX/workspace" \
  BRAMA_BASE_URL=http://127.0.0.1:9 BRAMA_HMAC_SECRET_FILE="$SBX/brama.hmac" \
  LAS_COMMAND=/usr/bin/python3 LAS_MCP_ENTRYPOINT="$SBX/las/stub-las.py" \
  LAS_RELEASE_MANIFEST_FILE="$SBX/las/release.manifest.json" \
  LAS_RELEASE_MANIFEST_SIGNATURE_FILE="$SBX/las/release.manifest.sig" \
  LAS_RELEASE_TRUST_STORE_FILE="$SBX/las/release.trust" \
  LAS_RELEASE_WATERMARK_FILE="$SBX/las/release.watermark" \
  SINGULARITY_REQUIRED_SURFACES=

step() { printf '\n### %s\n' "$1"; }

step 'boot gate: missing identity (clap)'
out=$(env -u SINGULARITY_AGENT_ID "$BIN" once 2>&1); code=$?
printf '%s\n' "$out" | head -3; echo "exit=$code"

step 'boot gate: malformed digest'
SINGULARITY_WORKLOAD_PUBLIC_KEY=not-hex "$BIN" once 2>&1; echo "exit=$?"

step 'boot gate: group/world-readable secret'
chmod 644 "$SBX/brama.hmac"; "$BIN" once 2>&1; echo "exit=$?"
chmod 600 "$SBX/brama.hmac"

step 'boot gate: missing Las entrypoint'
LAS_MCP_ENTRYPOINT="$SBX/las/missing.py" "$BIN" once 2>&1; echo "exit=$?"

step 'boot gate: relative Las release pinning path'
LAS_RELEASE_MANIFEST_FILE=las/release.manifest.json "$BIN" once 2>&1; echo "exit=$?"

step 'the compiled-in catalogue (stub Las advertises zero tools)'
"$BIN" tools --format table 2>&1; echo "exit=$?"

step 'required surface the stub cannot offer'
SINGULARITY_REQUIRED_SURFACES=finance "$BIN" once 2>&1; echo "exit=$?"

step 'doctor: nothing listening on the Brama port'
"$BIN" doctor 2>&1; echo "exit=$?"

step 'once: the being is created, only cognition fails'
"$BIN" once 2>&1; echo "exit=$?"
ls -la "$SBX/state"

step 'resume gate: state already exists'
"$BIN" once 2>&1; echo "exit=$?"

step 'resume gate: identity mismatch'
SINGULARITY_ROLE=other-role "$BIN" once --resume 2>&1; echo "exit=$?"

step 'resume accepted: same being, one cycle older'
"$BIN" once --resume 2>&1; echo "exit=$?"
python3 -c "import json;print('cycle =', json.load(open('$SBX/state/state.json'))['cycle'])"

step 'resume gate: no state to resume'
SINGULARITY_STATE_DIR="$SBX/state-none" "$BIN" once --resume 2>&1; echo "exit=$?"

printf '\nsandbox: %s\n' "$SBX"
