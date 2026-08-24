# CLI reference

The `singularity` binary has four subcommands. Every option is also an
environment variable — the tables in [configuration](configuration.md) are
the authoritative list — and `--help` on any subcommand prints the canonical
options. Logging uses `tracing` with the standard `RUST_LOG` env filter and
defaults to `info`.

| Command | Purpose |
|---|---|
| `singularity run` | Live continuously while solvent. |
| `singularity once` | Execute one autonomous cycle and print its report. |
| `singularity doctor` | Verify Brama, Most, Las, and required surfaces. |
| `singularity tools` | Print the dynamic and built-in tool catalogue. |

## Exit codes

Failures print `singularity: <error>` on stderr and exit with a class-stable
code:

| Code | Class |
|---|---|
| `0` | Success. |
| `2` | Configuration or secret-file error — a missing or invalid variable, a secret file that is empty or group/world readable. |
| `3` | An upstream surface failed: Brama, the Las MCP channel, or Most. |
| `4` | State or I/O error — unreadable state, unsupported schema, filesystem failure. |
| `5` | Tool, runtime, or JSON error. |

## `singularity run`

Boots the being and repeats autonomous cycles while the budget's remaining
balance is above zero. Between cycles it sleeps `--cycle-interval-secs`
(default 5). Ctrl-C (SIGINT) cancels the loop; shutdown then saves state,
stops the Las child within `--shutdown-grace-secs` (default 10, then kill),
and appends a final `stopped` journal event with status `stopped` or
`exhausted`.

Boot is strict about state:

- No existing state and no `--resume`: a fresh being is created from the
  configured identity and starting balance.
- Existing state without `--resume`: refused with
  `state already exists at <path>; use --resume or a new directory`.
- `--resume` without existing state: refused.
- `--resume` with existing state: refused unless the stored identity exactly
  matches the configured identity.

`--stimulus <text>` (or `SINGULARITY_STIMULUS`) injects one external
observation into the first cycle only, as
`External observation for this cycle: <text>`. It is capped at 65536 bytes
and must contain no NUL.

A permanent Brama error (for example a rejected signature) aborts the run;
any other cycle error is recorded as a `warning` journal event and the loop
continues after the normal sleep.

## `singularity once`

Identical boot and shutdown, but exactly one cycle, then prints the cycle
report as pretty JSON on stdout:

| Field | Meaning |
|---|---|
| `cycle` | Cycle counter, persisted across resumes. |
| `status` | `completed`, `tool_round_limit`, or `budget_exhausted`. |
| `final_content` | The model's closing reply, when `status` is `completed`. |
| `balance_usd` | Remaining balance after the cycle. |
| `earned_usd` | Lifetime credited revenue. |
| `net_profit_usd` | Earned minus API spend minus instance spend. |
| `total_tokens` | Lifetime token usage. |
| `actions` | Tool names invoked this cycle, in order. |

## `singularity doctor`

Read-only preflight; it starts no cycle and writes no being state. In order:

1. `GET /health` on Brama must answer `{"status":"ok"}`.
2. `GET /v1/models` must list the configured `--brama-model`, unless the
   model is a selector: `any`, `any-vision-capable`, or a `task:` prefix.
3. When `--most-token-file` is configured, Most health must report at least
   one send-capable backend (`backends` non-empty and not `none`).
4. Las is spawned exactly as `run` would spawn it, and every surface in
   `--required-surfaces` must expose at least one `<surface>__` tool.

Success prints `{"ok":true,"brama_model":…,"most":…,"las_tools":…}`.

## `singularity tools`

Spawns Las, builds the same catalogue the model would receive — minus the
Most tools, which require a Most credential — and prints it.

| Option | Meaning |
|---|---|
| `--format json` | Full tool definitions with JSON Schemas (default). |
| `--format table` | One `name<TAB>description` line per tool. |
| `--agent-id` | Optional Las agent identity; required when the `skarbiec` surface is selected. |
| `--las-command`, `--las-entrypoint`, `--las-only`, `--las-skip` | Las spawn selection, as in [skills](skills.md). |
| `--las-release-manifest`, `--las-release-manifest-signature`, `--las-release-trust-store`, `--las-release-watermark` | Signed Las release pinning files. |

Unlike `run`, `tools` enforces no required surfaces and uses a fixed 120 s
MCP deadline.

## Companion binaries

| Binary | Role |
|---|---|
| `singularity-bootstrap` | Verify a signed launch manifest, redeem short-lived Brama and Most credentials, launch the runtime. See [bootstrap](bootstrap.md). |
| `singularity-finance-mcp` | Policy-bound finance MCP server; also `owner-event <file>` ingestion. See [finance](finance.md). |
| `singularity-finance-executor-http` | Isolated executor adapter that forwards canonical intents to an HTTPS custody endpoint. See [finance](finance.md). |
| `singularity-repo-mcp` | Policy-bound repository proposal MCP server. See [repo surface](repo-surface.md). |
