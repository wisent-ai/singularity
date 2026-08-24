# Earning and cost record

Every model call costs money and almost nothing earns it — so the accounting
is the being's most load-bearing state. The record is the `budget` object in
`state.json` plus the paired journal events, written exactly once per event
and in exact decimal arithmetic, never floating point.

## The budget object

| Field | Meaning |
|---|---|
| `starting` | `SINGULARITY_STARTING_BALANCE_USD` (default `10`); refused if negative. |
| `remaining` | Current balance; the [solvency](solvency.md) gate reads only this. |
| `api_spent` | Cumulative token cost. |
| `instance_spent` | Cumulative wall-clock instance cost. |
| `total_tokens` | Lifetime token count (saturating). |
| `earned` | Cumulative credited revenue. |

Net profit is derived, never stored: `earned − api_spent − instance_spent`.

## Cost: debited exactly once per completion

Immediately after every Brama completion:

```
api      = prompt_tokens     × BRAMA_INPUT_PRICE_USD_PER_MILLION  / 1,000,000
         + completion_tokens × BRAMA_OUTPUT_PRICE_USD_PER_MILLION / 1,000,000
instance = elapsed_call_nanoseconds × SINGULARITY_INSTANCE_USD_PER_HOUR / 3.6e12
remaining -= api + instance
```

and two journal events are appended: `model_completed` (the usage block)
and `cost_debited` (the amount). The runtime also refuses a Brama usage
block whose non-zero `total_tokens` disagrees with
`prompt_tokens + completion_tokens`
(`brama: usage total does not match prompt plus completion`), so token
accounting cannot drift silently.

## Revenue: credited only from trusted evidence

The model cannot talk itself richer. After a tool call, a credit happens
only when all of these hold:

1. the tool name starts with `finance__` or `trading__`;
2. its result contains a `revenue_usd` or `realized_profit_usd` key —
   searched recursively through objects and arrays, accepted as a decimal
   string or a float;
3. the first such value found is strictly positive.

That amount is added to `earned` and `remaining` and journaled as
`revenue_credited` with the tool name as `source`. A negative credit is a
hard state error (`revenue cannot be negative`). Output of every other tool
— including the being's own file writes and Most messages — has no
financial effect. What it takes for a `finance__*` tool to report real
revenue at all is [the finance boundary](../finance.md).

## Where it lives

- `state.json → budget` — the current record, saved atomically with the
  rest of the being;
- `activity.jsonl` — the history: `model_completed`, `cost_debited`,
  `revenue_credited` per event, fsynced on append;
- every `once` report repeats the headline numbers: `balance_usd`,
  `earned_usd`, `net_profit_usd`, `total_tokens`.

```bash
jq .budget .singularity/state.json
grep -E '"(cost_debited|revenue_credited)"' .singularity/activity.jsonl
```

## Not to be confused with

- **[Solvency](solvency.md)** — the gate that reads `remaining`; this page
  is the ledger it reads.
- **The finance transaction ledger** — `singularity-finance-mcp` keeps its
  own hash-chained audit of money moving in the world; the budget records
  what existence costs and what tools report as realized revenue.
- **Brama's own billing** — Singularity prices tokens itself from the two
  `*_PRICE_USD_PER_MILLION` variables; with both at their default `0`, model
  calls debit only instance time (and nothing at the default instance rate
  of `0`).
