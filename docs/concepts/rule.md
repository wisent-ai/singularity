# Rule

Nobody hands this being a policy for its own behavior — it writes one, one
sentence at a time. A rule is a self-imposed, persistent constraint the
being adds to its own mind and re-reads on every model round.

## What it is

One string in `mind.rules`, appended by the built-in tool
`singularity_self_add_rule`:

- argument `rule`, trimmed, non-empty, ≤ 4096 bytes, no control characters;
  a violation fails as `invalid_arguments` with `rule is invalid`;
- the successful response echoes the stored rule: `{"rule": "<text>"}`.

## Lifecycle

Append-only. There is no rule-removal or rule-editing tool: the catalogue
exposes `singularity_self_add_rule` and nothing that deletes from `rules`.
Replacing the system prompt (`singularity_self_set_prompt`) does not touch
rules — they are a separate field and a separate prompt section. A rule
therefore outlives every prompt rewrite and every process restart, for as
long as the being's state exists.

## Where it appears

Every model round, all rules are rendered into the system message under the
exact header `Self-imposed rules:`, one `- ` line each, between the system
prompt and the learnings (see [mind](mind.md)). The model reads its own past
constraints before choosing the next action; that is the entire enforcement
mechanism — a rule binds cognition, not the runtime. Hard boundaries (money,
paths, secrets) are enforced elsewhere and do not depend on the being's
self-discipline: see [the finance boundary](../finance.md) and
[skills](../skills.md).

## Where it lives

`state.json → mind.rules`, saved atomically after the tool call that added
it, and journaled as a `tool_finished` event for
`singularity_self_add_rule`.

```bash
jq .mind.rules .singularity/state.json
```

## Not to be confused with

- **A learning** — `singularity_self_add_learning` records what happened
  (≤ 8192 bytes, own prompt section `Persistent learnings:`); a rule
  prescribes what to do. The runtime treats them identically except for
  size and section.
- **The system prompt** — replaceable wholesale; rules are append-only.
- **The signed finance policy** — an operator-signed document enforced by a
  separate process ([lease](lease.md), [finance](../finance.md)); a rule is
  self-written and enforced only by the being's own reading of it.
