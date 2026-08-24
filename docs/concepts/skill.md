# Skill

Where does a being's ability to act on the world come from? Not from its own
binary: every world-facing capability is a skill — one namespaced MCP tool
served by the supervised Las child process — plus a small compiled-in set of
self-tools. The catalogue is dynamic: what Las offers at boot is what the
model can call.

## What it is

A skill is an `McpTool` from Las `tools/list`:

| Field | Meaning |
|---|---|
| `name` | `<surface>__<tool>`, e.g. `finance__finance_propose`, `warsztat__workspace_create`. |
| `description` | Free text shown to the model. |
| `inputSchema` | JSON Schema of the arguments (defaults to an empty object schema). |

Surfaces are selected by `LAS_ONLY` (default
`weles,skarbiec,tama,stado,lem,echo,most,probierz,byk,brama,warsztat,finance`)
minus `LAS_SKIP`.

## Catalogue rules

Las tools and the built-in being tools merge into one catalogue with strict
registration:

- names must be non-empty ASCII alphanumerics with `_` or `-`
  (`tool: invalid tool name: <name>`);
- schemas must be objects (`tool: tool schema is not an object: <name>`);
- a duplicate name anywhere — including a Las tool colliding with a
  built-in — fails boot (`tool: duplicate tool: <name>`) rather than
  shadowing;
- a call to a name not in the catalogue fails as `unknown_tool` without
  reaching Las.

Every surface named in `SINGULARITY_REQUIRED_SURFACES` (default
`skarbiec,finance`) must expose at least one `<surface>__` tool, or boot
fails permanently:

```
singularity: mcp: required Las surface unavailable: <surface>
```

(exit 3; captured in [walkthrough-first-cycle](../walkthrough-first-cycle.md)).

## Execution and outcome

A skill call is forwarded to Las as `tools/call` under the per-request
deadline (`SINGULARITY_MCP_TIMEOUT_SECS`, default 120 s) and classified:

| Status | Meaning |
|---|---|
| `success` | Result without `isError`. |
| `failed` | The tool answered with an error (`remote_tool`) or the arguments were invalid. |
| `indeterminate` | The MCP transport failed mid-call (`mcp` error code): the remote effect is unknown, recorded, surfaced to the model, and never automatically replayed. |

Before the model sees any result, it crosses the output boundary — size,
depth, forbidden-key, private-key and raw-path screening — described in
[skills](../skills.md); a violating result is replaced entirely with
`{"status":"failed","error_code":"sensitive_output_rejected"}`.

## The built-in tools

Nine self-tools are compiled in and always present — memory, prompt, rule,
learning, model switch, child spawn, and workspace-confined file read/write
(see [mind](mind.md), [rule](rule.md), [child](child.md)) — plus three
direct Most tools (`most_health`, `most_create_chat`, `most_send_message`)
registered only when `MOST_SERVICE_TOKEN_FILE` is configured. Captured with
a stub Las offering zero tools:

```
$ singularity tools --format table
singularity_memory_remember	Persist a memory owned by this digital being
singularity_memory_recall	Recall persistent memories containing a query
singularity_self_set_prompt	Replace this being's persistent system prompt
singularity_self_add_rule	Add a persistent self-imposed rule
singularity_self_add_learning	Record a persistent learning that changes future decisions
singularity_self_switch_model	Switch future cognition calls to another available Brama model
singularity_spawn_child	Create and start a child digital being with separate state
singularity_file_read	Read one UTF-8 file inside the configured workspace
singularity_file_write	Atomically create or replace one UTF-8 file inside the configured workspace
```

## Commands

```bash
singularity tools --format table   # the exact catalogue the model would see
singularity doctor                 # proves required surfaces exist
```

## Not to be confused with

- **A built-in tool** — acts on the being's own state or workspace; a skill
  acts on the world through a surface with its own authority.
- **A surface** — the service behind a namespace prefix (Weles, Skarbiec,
  Finance…); a skill is one tool of one surface.
- **A capability** — Skarbiec-brokered authority a surface may hold; the
  being holds none of them directly.
