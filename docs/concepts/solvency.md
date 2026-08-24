# Solvency

The runtime has exactly one unconditional stop, and it is not a task being
finished — it is `remaining ≤ 0`. Solvency is the rule that existence must
be paid for, applied as a gate at every point where the being would spend.

## The gate

`can_call` is `budget.remaining > 0`, checked in exact decimal:

| Checkpoint | On failure |
|---|---|
| Before a cycle starts | The cycle does not start; status becomes `exhausted`; the report says `budget_exhausted`. |
| Before **every** model round inside a cycle | Same — a being can go broke halfway through a cycle, after a tool has already run. |
| The `run` loop condition | `run` repeats cycles only while solvent and not cancelled. |
| At shutdown | Final status is `stopped` if still solvent, `exhausted` if not. |

There is no debt: costs are debited after each completion, so `remaining`
can cross zero mid-cycle, and the next gate catches it. An `exhausted`
being's state remains intact and inspectable; nothing is deleted.

## What restores solvency

Only two things move `remaining` upward: a fresh start with a higher
`SINGULARITY_STARTING_BALANCE_USD` (a new being), or credited revenue on
the existing being (see [earning and cost record](earnings.md)). Resuming an
exhausted being does not reset the balance — `--resume` continues the same
budget, and the first gate will immediately report `budget_exhausted` again
unless revenue arrived in the meantime. Since revenue only arrives through
tool results, an exhausted being cannot earn its own way back: solvency is
recoverable only by the operator's choice.

## How it reads

The framing message opens every cycle with the numbers the model must
respect:

```
Begin autonomous cycle <n>. Balance: <remaining> USD. Earned: <earned> USD.
Net profit: <net> USD. Current model: <model>. Recent actions: <tool:status …>.
Inspect opportunities and choose the next useful action. A plain response
ends only this cycle; the being continues living while solvent.
```

and the `once` report repeats them (`balance_usd`, `earned_usd`,
`net_profit_usd`). The initial system prompt makes the demand explicit:
*earn more than you spend, preserve your solvency*.

## Where it lives

```bash
jq '{status, remaining: .budget.remaining}' .singularity/state.json
```

`status` is `exhausted` exactly when the last gate failed; the final
journal event is `{"type":"stopped", "status":"exhausted", …}`.

## Not to be confused with

- **The [budget](earnings.md)** — the ledger; solvency is the predicate over
  it.
- **The protected reserve** — a finance-policy floor on real-world assets
  enforced by `singularity-finance-mcp`; solvency is about the being's own
  operating balance.
- **`tool_round_limit`** — a cycle that ran out of rounds, not money; the
  being continues with the next cycle.
