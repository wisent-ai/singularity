# Configuration

Everything is configured through environment variables; each also exists as
a `--flag` on the corresponding subcommand (`--agent-id` for
`SINGULARITY_AGENT_ID`, and so on — `singularity <command> --help` prints
the mapping). This page lists every variable each binary reads, with
defaults and the validation the runtime enforces.

## Secret files

Every secret is read from a file, never from a bare variable (the one
fallback is noted below). A secret file must be a regular file, must not be
group- or world-accessible, and must be non-empty after stripping trailing
newlines; violations are refused at boot with exit code 2.

## `singularity` — identity and accounting

| Variable | Default | Meaning |
|---|---|---|
| `SINGULARITY_AGENT_ID` | required | Immutable agent identifier. |
| `SINGULARITY_AGENT_NAME` | `MyAgent` | Persona name, woven into the initial prompt. |
| `SINGULARITY_AGENT_TICKER` | `AGENT` | Persona ticker. |
| `SINGULARITY_AGENT_TYPE` | `general` | Persona type label. |
| `SINGULARITY_SPECIALTY` | `general` | Specialty, woven into the initial prompt. |
| `SINGULARITY_ROLE` | required | Workload role. |
| `SINGULARITY_ENVIRONMENT` | required | Workload environment. |
| `SINGULARITY_HOST` | required | Host label. |
| `SINGULARITY_WORKLOAD_ID` | required | Workload identifier. |
| `SINGULARITY_WORKLOAD_PUBLIC_KEY` | required | 64 lowercase hex. |
| `SINGULARITY_EXECUTABLE_SHA256` | required | 64 lowercase hex. |
| `SINGULARITY_CODE_SHA256` | required | 64 lowercase hex. |
| `SINGULARITY_POLICY_SHA256` | required | 64 lowercase hex. |
| `SINGULARITY_POLICY_SEQUENCE` | required | Policy sequence number. |
| `SINGULARITY_STARTING_BALANCE_USD` | `10` | Initial budget; must not be negative. |
| `SINGULARITY_INSTANCE_USD_PER_HOUR` | `0` | Instance cost rate; must not be negative. |
| `SINGULARITY_STIMULUS` | unset | One external observation for the first cycle; ≤65536 bytes, no NUL. |

Identity components (`agent_id`, `role`, `environment`, `host`,
`workload_id`) are 1–128 characters of ASCII alphanumerics plus `- _ . :`.

## `singularity` — loop and state

| Variable | Default | Meaning |
|---|---|---|
| `SINGULARITY_CYCLE_INTERVAL_SECS` | `5` | Sleep between cycles in `run`. |
| `SINGULARITY_MAX_TOOL_ROUNDS` | `8` | Model rounds per cycle; must be positive. |
| `SINGULARITY_STATE_DIR` | `.singularity` | Owner-only state directory. |
| `SINGULARITY_WORKSPACE` | `.` | Root for `singularity_file_read`/`write`; must be an existing directory. |
| `SINGULARITY_RESUME` | `false` | Continue existing state; identity must match. |
| `SINGULARITY_HTTP_TIMEOUT_SECS` | `120` | Brama and Most HTTP timeout. |
| `SINGULARITY_MCP_TIMEOUT_SECS` | `120` | Per-request Las deadline. |
| `SINGULARITY_SHUTDOWN_GRACE_SECS` | `10` | Grace before the Las child is killed. |

## `singularity` — Brama

| Variable | Default | Meaning |
|---|---|---|
| `BRAMA_BASE_URL` | `http://127.0.0.1:8081` | Gateway origin; http or https. |
| `BRAMA_MODEL` | `any` | Exact model or a selector (`any`, `any-vision-capable`, `task:*`). |
| `BRAMA_HMAC_SECRET_FILE` | see below | HMAC secret file for request signing. |
| `BRAMA_MAX_TOKENS` | `2048` | Completion token cap. |
| `BRAMA_TEMPERATURE` | `0.2` | Must be finite, between 0 and 2. |
| `BRAMA_INPUT_PRICE_USD_PER_MILLION` | `0` | Prompt-token price. |
| `BRAMA_OUTPUT_PRICE_USD_PER_MILLION` | `0` | Completion-token price. |

Requests are signed HMAC-SHA256 over `agent_id:timestamp:body_sha256` and
sent with `x-agent-id`, `x-agent-timestamp`, `x-agent-body-sha256`, and
`x-agent-signature` headers. When `BRAMA_HMAC_SECRET_FILE` is unset, the
secret comes from `WISENT_APP_AGENT_AUTH_SECRET`; one of the two is
required.

## `singularity` — Las and Most

| Variable | Default | Meaning |
|---|---|---|
| `LAS_COMMAND` | `node` | Interpreter for the Las entrypoint. |
| `LAS_MCP_ENTRYPOINT` | `../las/src/mcp.mjs` | Must be an existing file. |
| `LAS_ONLY` | `weles,skarbiec,tama,stado,lem,echo,most,probierz,byk,brama,warsztat,finance` | Surface selection passed to Las. |
| `LAS_SKIP` | unset | Surfaces to exclude. |
| `LAS_RELEASE_MANIFEST_FILE` | required | Absolute regular file. |
| `LAS_RELEASE_MANIFEST_SIGNATURE_FILE` | required | Absolute regular file. |
| `LAS_RELEASE_TRUST_STORE_FILE` | required | Absolute regular file. |
| `LAS_RELEASE_WATERMARK_FILE` | required | Absolute path. |
| `SINGULARITY_REQUIRED_SURFACES` | `skarbiec,finance` | Surfaces that must expose tools at boot. |
| `MOST_BASE_URL` | `http://127.0.0.1:8080` | Most origin; http or https. |
| `MOST_SERVICE_TOKEN_FILE` | unset | Enables the direct Most tools when set. |

When the `skarbiec` surface is selected, the `SKARBIEC_*` capability
variables present in the environment are forwarded to the Las child (see
[skills](skills.md)); everything else is scrubbed.

## `singularity-finance-mcp`

| Variable | Meaning |
|---|---|
| `SINGULARITY_FINANCE_POLICY_FILE` | Absolute path to the Ed25519-signed policy. |
| `SINGULARITY_FINANCE_ENABLE_LEASE_FILE` | Absolute path to the signed enable lease. |
| `SINGULARITY_FINANCE_STATE_DIR` | Absolute path to the owner-only state directory. |
| `SINGULARITY_FINANCE_VERIFY_KEY_HEX` | 32-byte Ed25519 verifying key, hex. |
| `SINGULARITY_FINANCE_EXECUTOR` | Absolute path to the executor; must be an executable file. |

All five are required at startup; semantics are in
[the finance boundary](finance.md).

## `singularity-finance-executor-http`

| Variable | Meaning |
|---|---|
| `SINGULARITY_FINANCE_CUSTODY_URL` | Credential-free HTTPS custody endpoint. |
| `SINGULARITY_FINANCE_CUSTODY_TOKEN_FILE` | Owner-only bearer token file. |

## `singularity-repo-mcp`

| Variable | Meaning |
|---|---|
| `JEDEN_REPO_POLICY_FILE` | Absolute path to the repository policy. |
| `JEDEN_REPO_STATE_DIR` | Absolute path to the proposal state directory. |

See [repo surface](repo-surface.md).

## `singularity-bootstrap`

| Variable | Meaning |
|---|---|
| `SINGULARITY_BOOTSTRAP_MANIFEST` | Signed launch manifest (owner-only). |
| `SINGULARITY_BOOTSTRAP_MANIFEST_SIG` | 64-byte Ed25519 signature, hex (owner-only). |
| `SINGULARITY_BOOTSTRAP_TRUST_ROOT` | 32-byte verifying key, hex (owner-only). |
| `SINGULARITY_RUNTIME_ROOT` | Absolute directory for ephemeral credential materialization. |

The bootstrap launches the runtime with a scrubbed environment and injects
the identity and credential-path variables itself; see
[bootstrap](bootstrap.md).
