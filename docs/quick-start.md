# Quick start

How do you go from a checkout to one completed autonomous cycle? This page is
the one happy path: build the binaries, satisfy the three prerequisites
(identity, Brama, Las), verify with `doctor`, then run `once`. The command
surface is [cli](cli.md); every variable named here is specified in
[configuration](configuration.md).

## Build

```bash
cargo build --locked
cargo install --path . --locked
```

The package builds five binaries: `singularity` (the being runtime),
`singularity-bootstrap` (managed launch), `singularity-repo-mcp`,
`singularity-finance-mcp`, and `singularity-finance-executor-http`.
Rust 1.85 or newer is required.

## Prerequisite 1: an identity

The runtime refuses to start anonymous. Every `run`, `once`, and `doctor`
needs the immutable workload identity — who this being is and which exact
code and policy it runs:

```bash
export SINGULARITY_AGENT_ID=<agent-id>
export SINGULARITY_ROLE=<role>
export SINGULARITY_ENVIRONMENT=<environment>
export SINGULARITY_HOST=<host-label>
export SINGULARITY_WORKLOAD_ID=<workload-id>
export SINGULARITY_WORKLOAD_PUBLIC_KEY=<64-hex>
export SINGULARITY_EXECUTABLE_SHA256=<64-hex>
export SINGULARITY_CODE_SHA256=<64-hex>
export SINGULARITY_POLICY_SHA256=<64-hex>
export SINGULARITY_POLICY_SEQUENCE=<positive-integer>
```

Identity components are 1–128 characters of ASCII alphanumerics plus
`- _ . :`; the key and digests must be exactly 64 lowercase hex characters.
In a managed deployment [`singularity-bootstrap`](bootstrap.md) injects all
of these from a signed manifest; for a local first run you set them yourself.

The being's persona is separate and optional: `SINGULARITY_AGENT_NAME`
(default `MyAgent`), `SINGULARITY_AGENT_TICKER` (default `AGENT`),
`SINGULARITY_AGENT_TYPE` and `SINGULARITY_SPECIALTY` (default `general`).

## Prerequisite 2: Brama

Cognition goes through a Brama gateway, authenticated per request with an
HMAC secret:

```bash
export BRAMA_BASE_URL=<brama-origin>        # default http://127.0.0.1:8081
export BRAMA_MODEL=any                      # or an exact advertised model
export BRAMA_HMAC_SECRET_FILE=<path>        # owner-only file, not group/world readable
```

Without `BRAMA_HMAC_SECRET_FILE` the runtime falls back to the
`WISENT_APP_AGENT_AUTH_SECRET` environment variable; one of the two is
required. To account real spend, also set
`BRAMA_INPUT_PRICE_USD_PER_MILLION` and `BRAMA_OUTPUT_PRICE_USD_PER_MILLION`
(both default `0`).

## Prerequisite 3: Las

Skills come from a Las checkout, spawned as a child MCP process and pinned to
a signed release:

```bash
export LAS_MCP_ENTRYPOINT=<path-to>/las/src/mcp.mjs   # default ../las/src/mcp.mjs
export LAS_RELEASE_MANIFEST_FILE=<absolute-path>
export LAS_RELEASE_MANIFEST_SIGNATURE_FILE=<absolute-path>
export LAS_RELEASE_TRUST_STORE_FILE=<absolute-path>
export LAS_RELEASE_WATERMARK_FILE=<absolute-path>
```

The manifest, signature, and trust store must be absolute paths to existing
regular files; the watermark must be an absolute path. By default the
runtime requires the `skarbiec` and `finance` surfaces to be present
(`SINGULARITY_REQUIRED_SURFACES=skarbiec,finance`); for a first run without
those surfaces, narrow both lists, e.g.
`LAS_ONLY=weles,most SINGULARITY_REQUIRED_SURFACES=`.

## Verify the world

```bash
singularity doctor
```

`doctor` checks Brama `/health`, confirms the configured model is either a
selector (`any`, `any-vision-capable`, `task:*`) or actually advertised by
`/v1/models`, checks Most send-capability when `MOST_SERVICE_TOKEN_FILE` is
configured, spawns Las, and confirms every required surface exposes at least
one tool. Success prints one JSON object:

```json
{"ok":true,"brama_model":"any","most":null,"las_tools":42}
```

To inspect the exact catalogue the model would see:

```bash
singularity tools --format table
```

## Run one cycle

```bash
singularity once
```

`once` boots the being — refusing to overwrite an existing `state.json`; use
`--resume` or a fresh `SINGULARITY_STATE_DIR` (default `.singularity`) —
executes one autonomous cycle, shuts down cleanly, and prints the cycle
report:

```json
{
  "cycle": 1,
  "status": "completed",
  "final_content": "…",
  "balance_usd": "9.98",
  "earned_usd": "0",
  "net_profit_usd": "-0.02",
  "total_tokens": 1874,
  "actions": ["singularity_memory_remember"]
}
```

`status` is `completed` when the model ended the cycle with a plain reply,
`tool_round_limit` when it hit `SINGULARITY_MAX_TOOL_ROUNDS` (default 8), or
`budget_exhausted` when the balance reached zero.

## Let it live

```bash
singularity run
```

`run` repeats cycles, sleeping `SINGULARITY_CYCLE_INTERVAL_SECS` (default 5)
between them, while the balance stays above zero and until Ctrl-C. The
starting balance is `SINGULARITY_STARTING_BALANCE_USD` (default `10`).
Everything the being does lands in the state directory: `state.json` (the
being), `activity.jsonl` (the journal), and `children/<id>/` for any child
beings. Continue with [the being](being.md) for what that state means and
[the loop and solvency](loop.md) for how a cycle spends and earns.
