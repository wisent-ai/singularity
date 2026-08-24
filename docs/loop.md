# The loop and solvency

The being lives cycle by cycle. A cycle is one user-visible unit of
autonomous behavior: a fresh framing message, up to `max_tool_rounds` model
rounds, and a durable record of everything spent, earned and done. This page
is the exact anatomy of a cycle, the cost and revenue accounting, and how
the loop ends.

## Anatomy of a cycle

1. **Solvency gate.** If the remaining balance is not above zero, the cycle
   does not start; the being becomes `exhausted` and the report says
   `budget_exhausted`.
2. **Framing.** The cycle counter increments and a user message is appended:
   cycle number, balance, lifetime earnings, net profit, current model, and
   the last 10 actions as `tool:status` pairs, ending with the reminder that
   a plain response ends only this cycle. On cycle 1 only, a configured
   `--stimulus` is appended as one external observation.
3. **Model rounds.** Up to `SINGULARITY_MAX_TOOL_ROUNDS` (default 8) times:
   the solvency gate is re-checked, the cognition messages are rebuilt from
   the mind (see [the being](being.md)), and Brama is called with the full
   tool catalogue. Cost is debited immediately after every completion.
   - If the reply has no tool calls, the cycle completes with the reply as
     `final_content`.
   - Otherwise every tool call executes in order through the catalogue
     ([skills](skills.md) or a built-in), the outcome is appended to the
     conversation, the action is recorded, and state is saved after each
     tool.
4. **Round limit.** If the rounds run out with tool calls still coming, a
   `maximum tool rounds reached` warning is journaled and the report status
   is `tool_round_limit`.

Every step writes its journal event: `cycle_started`, `model_completed`
(token usage), `cost_debited` (amount), `tool_finished` (tool, status),
`revenue_credited` (amount, source), `warning`.

## Cost: debited exactly once per completion

The budget carries `starting`, `remaining`, `api_spent`, `instance_spent`,
`total_tokens`, and `earned`. After each Brama completion the debit is:

```
api      = prompt_tokens   × BRAMA_INPUT_PRICE_USD_PER_MILLION  / 1,000,000
         + completion_tokens × BRAMA_OUTPUT_PRICE_USD_PER_MILLION / 1,000,000
instance = elapsed_call_time × SINGULARITY_INSTANCE_USD_PER_HOUR / 1 hour
remaining -= api + instance
```

Arithmetic is exact decimal, not floating point. The runtime also refuses a
Brama usage block whose `total_tokens` disagrees with
`prompt_tokens + completion_tokens`, so token accounting cannot drift.

## Revenue: credited only from trusted evidence

The model cannot talk itself richer. A credit happens only when:

- the tool that just returned is namespaced `finance__*` or `trading__*`, and
- its result contains a `revenue_usd` or `realized_profit_usd` field
  (searched recursively) whose value is positive.

The first such positive amount is credited to `earned` and `remaining` and
journaled as `revenue_credited` with the tool as source. Output of every
other tool — including the being's own file writes and messages — has no
financial effect. What it takes for a `finance__*` tool to actually move
money is [the finance boundary](finance.md).

## How the loop continues and ends

`singularity run` repeats cycles while `remaining > 0` and no Ctrl-C has
arrived, sleeping `SINGULARITY_CYCLE_INTERVAL_SECS` (default 5) between
cycles. Error handling is class-based:

| Error class | Behavior |
|---|---|
| Permanent Brama error (bad signature, unsupported response, 4xx) | The run aborts; exit code 3. |
| Any other cycle error (transient network, MCP failure, state I/O) | Journaled as `warning`; the loop sleeps and continues. |

An MCP transport failure during a tool call does not fail the cycle at all:
the tool outcome is recorded as `indeterminate` — the effect may or may not
have happened remotely — and is never automatically replayed. The model sees
the indeterminate outcome and decides what to do next.

Shutdown (Ctrl-C, end of `once`, or solvency exhaustion) saves state as
`stopping`, stops the Las child within the shutdown grace, then persists the
final status: `stopped` if still solvent, `exhausted` if not, with a final
`stopped` journal event. Because `state.json` is saved atomically after
every tool call, a crash at any point loses at most the round in flight, and
`run --resume` continues the same being.

## Solvency is the only mission

There is no assigned objective anywhere in the runtime. The framing message
states the balance and profit, the initial system prompt demands earning
more than is spent, and the loop's only unconditional stop is
`remaining ≤ 0`. Everything else — what to pursue, which rules to adopt,
which model to think with, whether to spawn children — is the being's own
persistent choice, recorded in its mind.
