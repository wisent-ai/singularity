# Skills via Las

Singularity ships almost no world-facing tools of its own. The dynamic
catalogue — internet actions, messaging, compute, capabilities, repository
work, finance — comes from Las, a separate supervised process serving
namespaced MCP tools. This page is how Las is spawned, what the runtime
requires of it, and the boundary every tool output crosses before the model
sees it.

## The supervised Las process

At boot the runtime spawns `LAS_COMMAND` (default `node`) with
`LAS_MCP_ENTRYPOINT` (default `../las/src/mcp.mjs`) as a child process
speaking MCP over stdio, protocol version `2024-11-05` — any other
negotiated version is refused. The child starts from a scrubbed
environment: everything is cleared, then exactly these are set:

- a fixed `PATH` (`/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin`)
  and `LANG`/`LC_ALL` of `C.UTF-8`;
- `LAS_ONLY` (surface selection; default
  `weles,skarbiec,tama,stado,lem,echo,most,probierz,byk,brama,warsztat,finance`)
  and `LAS_SKIP` when configured;
- the signed release pinning files: `LAS_RELEASE_MANIFEST_FILE`,
  `LAS_RELEASE_MANIFEST_SIGNATURE_FILE`, `LAS_RELEASE_TRUST_STORE_FILE`,
  `LAS_RELEASE_WATERMARK_FILE`;
- when the `skarbiec` surface is selected: the `SKARBIEC_*` capability
  paths present in the parent environment, plus `SKARBIEC_MCP_AGENT_ID` set
  to the being's agent id. Skarbiec without an explicit immutable agent
  identity is refused before spawn.

Each MCP request carries a deadline (`SINGULARITY_MCP_TIMEOUT_SECS`, default
120). Shutdown closes stdin and waits `SINGULARITY_SHUTDOWN_GRACE_SECS`
(default 10) before killing the child.

## Required surfaces

After `tools/list`, the runtime verifies that every surface named in
`SINGULARITY_REQUIRED_SURFACES` (default `skarbiec,finance`) exposes at
least one tool with the `<surface>__` name prefix; a missing surface is a
permanent boot failure. This is the existence guarantee behind the finance
boundary: a being configured to require `finance` cannot come up without
[the finance tools](finance.md).

## One catalogue, strict names

Las tools and the built-in tools (see [the being](being.md), plus the Most
tools below) merge into a single catalogue sent to Brama on every model
round. Tool names must be non-empty ASCII alphanumerics with `_` or `-`,
schemas must be objects, and a duplicate name anywhere — including a Las
tool colliding with a built-in — fails boot rather than shadowing. A call
to a name not in the catalogue fails as `unknown_tool` without reaching Las.

## The output boundary

Every tool result becomes a model-visible message only after validation.
The raw result is size- and shape-checked:

- at most 64 KiB when serialized, at most 8 levels deep;
- no object key — normalized to lowercase with `-` and spaces as `_` —
  equal to or suffixed by: `secret`, `password`, `passwd`, `token`,
  `access_token`, `refresh_token`, `api_key`, `authorization`, `cookie`,
  `private_key`, `privatekey`, `credential_path`, `secret_path`, `key_path`;
- no string containing a NUL, a private-key PEM block, or a raw local path
  (any whitespace-delimited token starting with `/` or `file://`);
- strings that look like JSON are parsed and re-checked recursively.

A violating result is not truncated or redacted — it is replaced entirely
with `{"status":"failed","error_code":"sensitive_output_rejected"}`, and the
rejection is logged. Secret material and machine-local paths therefore never
enter the conversation, the persistent state, or a future prompt.

## Outcome classification

| Status | Meaning |
|---|---|
| `success` | The tool returned a result without `isError`. |
| `failed` | The tool answered with an error (`remote_tool`), or arguments were invalid. |
| `indeterminate` | The MCP transport failed mid-call: the remote effect is unknown. Recorded, surfaced to the model, never automatically replayed. |

Every outcome is journaled as `tool_finished` and appended to the last-100
action window the next cycle's framing message summarizes.

## Most direct tools

Besides the `most__*` Las surface, the runtime registers three first-class
Most tools when — and only when — `MOST_SERVICE_TOKEN_FILE` is configured:
`most_health`, `most_create_chat` (from, recipients, text, optional
preferred service among `iMessage`/`SMS`/`RCS`), and `most_send_message`
(chat id, text, optional preferred service). Chat and message ids returned
by these calls are tracked in the being's `created_resources`, so the state
records what the being has created in the world.
