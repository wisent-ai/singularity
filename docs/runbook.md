# Runbook

Something refused, aborted, or went quiet — which sentence did it print, and
what does it mean? Every string below exists verbatim in the source or was
captured from a live run; grep for it. Failures print
`<binary>: <error>` on stderr; the runtime's exit codes are class-stable
(see the table at the end).

## The runtime refuses to boot (exit 2)

Configuration and secret gates run before anything is spawned. The
sentences and their repairs:

| Sentence | Meaning / repair |
|---|---|
| `error: the following required arguments were not provided: …` | clap: a required identity or Las-pinning flag/env is missing entirely. |
| `configuration: <label> is not a valid immutable identifier` | Identity component is empty, > 128 bytes, or has characters outside ASCII alphanumerics `- _ . :`. |
| `configuration: <label> must be 64 lowercase hexadecimal characters` | Key or digest is not exactly 64 lowercase hex. |
| `configuration: stimulus must be at most 65536 bytes and contain no NUL` | Trim the stimulus. |
| `configuration: workspace: <io-error>` / `workspace must be a directory` | `SINGULARITY_WORKSPACE` must canonicalize to an existing directory. |
| `configuration: max tool rounds must be positive` | `SINGULARITY_MAX_TOOL_ROUNDS=0`. |
| `configuration: prices and balance cannot be negative` | One of the four price/balance decimals is negative. |
| `configuration: temperature must be finite and between zero and two` | `BRAMA_TEMPERATURE` out of range. |
| `configuration: LAS entrypoint not found: <path>` | `LAS_MCP_ENTRYPOINT` is not an existing file. |
| `configuration: LAS release manifest must be an absolute regular file` | Also for the signature and trust store; the watermark only needs to be absolute: `LAS release watermark must be an absolute path`. |
| `configuration: BRAMA_BASE_URL must use http or https` / `<name>: <parse-error>` | Also `MOST_BASE_URL`. |
| `secret file: not a regular file: <path>` | Secret paths must name regular files. |
| `secret file: <path> must not be group/world accessible` | `chmod 600` the secret. |
| `secret file: <path> is empty` | The file must be non-empty after stripping trailing newlines. |
| `secret file: BRAMA_HMAC_SECRET_FILE or WISENT_APP_AGENT_AUTH_SECRET is required` | No Brama secret anywhere. |

## Boot fails at Las (exit 3)

| Sentence | Meaning |
|---|---|
| `mcp: cannot start Las: <os-error>` | The `LAS_COMMAND` executable was not found or not runnable. Remember the child runs from a scrubbed environment with the fixed `PATH` `/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin` — an interpreter outside those directories needs an absolute `LAS_COMMAND`. |
| `mcp: Las negotiated an unsupported MCP version` | Las did not answer `protocolVersion` `2024-11-05`. |
| `mcp: required Las surface unavailable: <surface>` | No tool named `<surface>__*` in the catalogue. Fix `LAS_ONLY`/`LAS_SKIP`, or narrow `SINGULARITY_REQUIRED_SURFACES`. |
| `mcp: Skarbiec requires an explicit immutable Las agent identity` | The `skarbiec` surface is selected but no agent id was supplied (`singularity tools` without `--agent-id`). |
| `mcp: invalid immutable Las agent identity` | The agent id fails the identifier rules. |
| `tool: duplicate tool: <name>` / `invalid tool name: <name>` / `tool schema is not an object: <name>` | Las offered a catalogue the runtime refuses to merge (exit 5). |

## The run aborts — or doesn't

Brama errors are classified, and only one class kills a `run`:

- **Permanent** (HTTP 4xx, malformed or contradictory completions —
  `unsupported finish reason`, `tool_calls finish reason without calls`,
  `stop finish reason with tool calls`, `response must contain exactly one
  choice`, `usage total does not match prompt plus completion`, a rejected
  signature): the run aborts, exit 3.
- **Transient** (connection errors, HTTP 429/5xx): the cycle is journaled
  as `{"type":"warning", …}` and the loop sleeps
  `SINGULARITY_CYCLE_INTERVAL_SECS`, then continues.
- **Indeterminate** (the request was sent, the response never arrived):
  same tolerated path for cognition; for tool calls the outcome is recorded
  as `indeterminate` and never automatically replayed.

`once` propagates the first error directly (captured:
`singularity: brama: error sending request for url
(http://127.0.0.1:9/v1/chat/completions)`, exit 3) — but state and journal
are already on disk.

`maximum tool rounds reached` as a warning event means the model was still
calling tools when `SINGULARITY_MAX_TOOL_ROUNDS` ran out; the report status
is `tool_round_limit`. Raise the limit or let the next cycle continue the
work.

## State refusals (exit 4)

| Sentence | Meaning |
|---|---|
| `state: state already exists at <path>; use --resume or a new directory` | Protecting an existing being from silent overwrite. |
| `state: resume requested but no state exists` | `--resume` against an empty directory. |
| `state: resume identity does not match configuration` | Any identity field differs — including persona and digests. |
| `state: unsupported state schema <version>` | Only `being-v1` loads; older supervisor state is not a compatibility path. |

## A tool call failed — from the model's side

| `error_code` in the tool outcome | Meaning |
|---|---|
| `invalid_arguments` | Arguments not a JSON object, or a field failed its bound (`<key> is invalid`). |
| `unknown_tool` | Name not in the catalogue; never reaches Las. |
| `remote_tool` | The Las tool answered `isError`. |
| `mcp` | Transport failure mid-call → status `indeterminate`, never replayed. |
| `most_unavailable` | `Most credential is not configured`. |
| `model_unavailable` | `Brama does not advertise that model`. |
| `invalid_path` / `workspace_boundary` | File tools: path not relative-normal, or `path leaves the workspace`. |
| `sensitive_output_rejected` | The result violated the output boundary (oversize, too deep, forbidden key, private-key material, raw local path) and was replaced entirely. The reason is in the runtime log, not shown to the model. |
| `child_state` / `child_executable` / `child_spawn` | Child creation failed at directory, executable resolution, or OS spawn. |

## `singularity-finance-mcp` will not start (exit 1)

Captured sentences, in the order the gates run:

```
policy_denied: SINGULARITY_FINANCE_POLICY_FILE is required        # likewise LEASE_FILE, STATE_DIR, EXECUTOR, VERIFY_KEY_HEX
policy_denied: SINGULARITY_FINANCE_EXECUTOR must name an executable file
policy_denied: protected file must be owner-only, current-user-owned, regular, and not a symlink
policy_denied: policy signature verification failed
policy_denied: signed policy rollback or equivocation detected
state_error: invalid audit record: …
state_error: audit hash chain validation failed
```

The last three matter most: a **rollback** refusal means the state store
has already anchored a newer policy/lease than the one on disk — restore
the newer document, never delete the anchor. An **audit** refusal means a
record under `state/audit/` was altered or truncated; the service refuses
to serve over a broken chain. Recover from your own copy of the store; the
chain is the evidence.

## Finance refusals in flight

All captured in [walkthrough-finance](walkthrough-finance.md):

| Sentence | Gate |
|---|---|
| `policy_denied: finance enable lease is absent, expired, disabled, or mismatched` | The [lease](concepts/lease.md), re-read on every propose/execute/privileged owner event. |
| `policy_denied: signed enable lease rollback or equivocation detected` | An older lease presented after a newer one was anchored. |
| `policy_denied: beneficiary is not in signed policy` / `asset is not in signed policy` | Unknown ids. |
| `policy_denied: beneficiary is disabled, outside its validity window, or disallows the asset or purpose` | Beneficiary gate. |
| `policy_denied: per-transaction limit exceeded` | Asset or beneficiary per-transaction cap. |
| `policy_denied: rolling, daily, or lifetime limit exceeded` (also the `beneficiary …` variant) | Window arithmetic over all reserving and executed transactions. |
| `policy_denied: protected reserve would be breached` | `spendable − reserve − reserved` cannot cover the amount. |
| `policy_denied: proposal TTL exceeds signed policy` / `proposal validity exceeds beneficiary validity` | TTL gates. |
| `policy_denied: parameters cannot override protected intent fields` | `parameters` tried to smuggle `destination`, `amount_minor`, `secret`, … |
| `invalid_state: request_id was already used with different intent` | Idempotency conflict; pick a new `request_id`. |
| `invalid_state: execution requires signed state and completed reconciliation` | `finance_execute` before `signed`, or with reconciliation pending. |
| `invalid_state: transaction can no longer be cancelled` / `cancellation deadline has passed` | Cancel after signing or after the deadline. |
| `internal_error: executor refused: <stderr>` | The isolated executor exited non-zero **after** dispatch — the transaction is already durably `indeterminate` with `reconciliation_required: true`; only a reconciler owner event moves it. |
| `policy_denied: owner event does not approve exact intent hash` | The signed event names a different intent. |
| `policy_denied: owner event timestamp outside acceptance window` | `occurred_at` older than 24 h or more than 5 min in the future. |
| `invalid_state: approval requires an independently accepted simulation` / `simulation event invalid in current state` / `signing requires completed timelock and ready state` / `submission requires signed state and completed reconciliation` / `confirmation requires submitted, indeterminate, or quarantined state` | Lifecycle-order gates on owner events. |
| `invalid_state: terminal transaction state is immutable` | `confirmed`/`rejected`/`cancelled`/`expired`/`failed` never change again. |
| `policy_denied: custody authority is not authorized by signed policy` / `independent custody signature verification failed` | Wrong authority id or signature. |
| `policy_denied: external WORM receipt is not bound to the exact execution event` | Receipt field mismatch (sink, kind, tx, intent, reference, timestamp). |

## `singularity-finance-executor-http` refused (exit 1)

```
SINGULARITY_FINANCE_CUSTODY_URL is required
SINGULARITY_FINANCE_CUSTODY_URL must be a credential-free HTTPS URL
SINGULARITY_FINANCE_CUSTODY_TOKEN_FILE must name an absolute regular non-symlink file
SINGULARITY_FINANCE_CUSTODY_TOKEN_FILE must be owner-only and owned by the current user
custody token is empty
execution request exceeds size limit          # stdin > 64 KiB
custody service refused with HTTP <status>
custody response exceeds size limit
custody response failed validation            # executor id, 64-hex reference, 128-hex signature, absolute receipt path
```

The adapter never retries; an ambiguous outcome surfaces to the finance
service as the dispatch failure above.

## `singularity-repo-mcp` refusals

Startup (exit 1): `policy_denied: JEDEN_REPO_POLICY_FILE is required`,
`policy must be owned by the current user and mode 0600 (or stricter)`,
`repository root is not a git checkout`, `branch_prefix cannot be a
protected branch`, `github_head_owner must match the owner of
github_repository`, `required check "<name>" is not defined`.

In flight (all captured in [repo-surface](repo-surface.md)):

| Sentence | Gate |
|---|---|
| `policy_denied: repository is not allowlisted` | Unknown `repo_id`. |
| `invalid_state: source repository is not clean` | The source checkout has any status output; commit or stash first. |
| `policy_denied: repository config contains executable Git filters` | Clean/smudge/process filters in the source repo are refused outright. |
| `policy_denied: path is outside allowed_paths` | Read or patch touching paths the policy does not own. |
| `invalid_state: cannot seal an empty diff` | Nothing to propose. |
| `invalid_state: workspace is not sealed` / `staged tree changed after seal` / `worktree changed after seal` / `untracked files appeared after seal` | The seal is an exact tree; any drift voids it. |
| `policy_denied: check is not allowlisted` | Check name not in policy. |
| `invalid_state: required check "<name>" has not run` / `lacks successful exact evidence` | Evidence must exist, be successful, and match the sealed fingerprint. |
| `invalid_state: workspace is already committed` | One commit per workspace; open a new one. |
| `invalid_state: branch must be published before opening a pull request` | Order gate. |
| `invalid_state: remote branch exists at a different commit` / `published proposal branch no longer matches the recorded commit` | Reconciliation against the remote failed; someone moved the branch. |
| `command_failed: <operation> failed (exit <code>): <stderr>` | An underlying `git`/`gh` invocation failed; the operation name says which. |

## Exit codes (`singularity`, `singularity-bootstrap`)

| Code | Class |
|---|---|
| `0` | Success. |
| `2` | Configuration or secret-file error (also clap usage errors). |
| `3` | An upstream surface failed: Brama, the Las MCP channel, or Most. |
| `4` | State or I/O error. |
| `5` | Tool, runtime, or JSON error. |

`singularity-bootstrap` exits with the launched runtime's own status once
it launches; its own refusals (`configuration: bootstrap manifest is
invalid or expired`, `bootstrap requires absolute regular owner-only
files`, `singularity executable digest mismatch`, `secret file: capability
redemption denied`, …) use the same table. The MCP servers exit `1` on
startup failure and otherwise answer errors in-band as
`{"isError": true}` tool results.
