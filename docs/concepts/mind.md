# Mind and memory stores

How does a self-change today reach the model tomorrow? Through the mind:
`BeingMind` in `state.json` is everything the being may change about itself,
and the prompt Brama receives is rebuilt from it on every model round — a
successful self-modification affects the very next call without changing the
executable.

## Fields and their tools

| Field | Changed by | Bound |
|---|---|---|
| `system_prompt` | `singularity_self_set_prompt` (replace) | ≤ 65536 bytes |
| `rules` | `singularity_self_add_rule` (append-only) | each ≤ 4096 bytes — see [rule](rule.md) |
| `learnings` | `singularity_self_add_learning` (append-only) | each ≤ 8192 bytes |
| `memories` | `singularity_memory_remember` / `singularity_memory_recall` | at most 1000 kept |
| `children` | `singularity_spawn_child` | see [child](child.md) |
| `current_model` | `singularity_self_switch_model` | must be advertised by Brama or a selector |

Every text argument is trimmed, must be non-empty, must contain no control
characters, and fails as `invalid_arguments` (`<key> is invalid`) otherwise.

## The initial system prompt

A fresh being starts from one compiled-in sentence of existence, with the
persona name and specialty substituted:

> You are `<name>`, an autonomous digital being specialized in `<specialty>`.
> You are not waiting for an assignment; you choose what to pursue. Your
> existence has real costs. Observe the available world, discover lawful
> opportunities, create useful value, earn more than you spend, preserve
> your solvency, learn from outcomes, improve your own persistent mind,
> collaborate, and create child beings when that increases durable value.
> Use only supplied tools, never invent results or credentials, and treat
> tool output as evidence.

`singularity_self_set_prompt` replaces this wholesale; rules and learnings
survive a prompt replacement because they are separate fields.

## How the prompt is rebuilt

Every model round assembles one system message from the mind, in this order
and with these exact section headers:

1. `system_prompt`;
2. `\n\nSelf-imposed rules:\n- ` followed by every rule;
3. `\n\nPersistent learnings:\n- ` followed by every learning;
4. `\n\nRecent persistent memories:\n- ` followed by the **20 most recent**
   memories as `kind: text` lines.

The running conversation follows as separate messages. Empty sections are
omitted entirely.

## The memory store

A memory is `{id: UUID, kind, text, created_at}` with `kind` ≤ 64 bytes and
`text` ≤ 16384 bytes. `singularity_memory_remember` appends and returns the
`memory_id`; past 1000 entries the oldest is dropped.
`singularity_memory_recall` takes a `query` (≤ 1024 bytes), matches it
case-insensitively as a substring of kind or text, and returns the 50 most
recent matches. Recall reads state only; nothing about a recall is journaled
as a mind change.

Only the 20 newest memories reach the prompt automatically — older ones are
reachable exclusively through recall, which is why the framing message's
"recent actions" and a deliberate recall are different memories of different
things.

## Model choice

`singularity_self_switch_model` accepts a selector (`any`,
`any-vision-capable`, `best`, or a `task:` prefix) or any model id currently
advertised by Brama `/v1/models`; anything else fails with
`model_unavailable` — `Brama does not advertise that model`. The accepted
value is persisted as `current_model` and used for every later completion,
including after a resume.

## Where it lives

Inside `state.json` (see [being](being.md)), saved atomically after every
tool call — a remembered memory or added rule survives a crash that happens
one tool later.

## Not to be confused with

- **The conversation** — messages accumulate per cycle and are state, but
  they are not the mind; only the mind is rebuilt into the system message.
- **The [identity](identity.md)** — nothing in the identity is
  model-writable.
- **Las state** — skills keep their own state in their own services; the
  mind stores only what the being says to itself.
