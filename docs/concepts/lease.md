# Lease

The signed finance policy says what may ever move; who says money may move
*right now*? The enable lease: a second, short-lived signed document that
must be present and healthy at the moment of every consequential finance
call. Deleting one file halts the money without touching the policy.

## What it is

`EnableLease` is an Ed25519-signed JSON document at
`SINGULARITY_FINANCE_ENABLE_LEASE_FILE`, verified against the same
`SINGULARITY_FINANCE_VERIFY_KEY_HEX` as the policy:

| Field | Meaning |
|---|---|
| `policy_id`, `policy_version` | Must equal the loaded policy exactly. |
| `lease_id` | Identifier (ASCII alphanumerics, `-`, `_`; ≤ 128). |
| `issued_at`, `expires_at` | Validity window; `expires_at` may not outlive the policy. |
| `enabled` | Must be `true`. |
| `kill_switch` | Must be `false`. |

Any failed condition — absent file, bad signature, wrong policy binding,
not yet issued, expired, disabled, or kill-switched — produces one exact
sentence:

```
policy_denied: finance enable lease is absent, expired, disabled, or mismatched
```

## Who checks it, and when

The lease is re-read from disk inside every call that could change the
world: `finance_propose`, `finance_execute`, and the privileged owner
events (`simulation_accepted`, `approval_granted`, `signed`, `submitted`).
`finance_status` and `finance_cancel` do not require it — in-flight
transactions stay readable and withdrawable while the money is halted.

## Rollback protection

Every accepted lease is anchored in the state store
(`lease-anchor.json` with `lease_id`, `issued_at`, and the document hash).
A later call presenting a lease older than the anchor, or a different lease
document with the same `issued_at`, is refused:

```
policy_denied: signed enable lease rollback or equivocation detected
```

The policy has the same anchor (`policy-anchor.json`) with the same
discipline: `policy_denied: signed policy rollback or equivocation detected`.

Captured from a sandbox session (kill switch flipped, fresh lease accepted,
old lease restored — see [walkthrough-finance](../walkthrough-finance.md)):

```
### finance_propose refusal (kill switch lease)
<- ERROR policy_denied: finance enable lease is absent, expired, disabled, or mismatched

### finance_propose accepted again under a fresh lease (anchor advances)
<- {"transaction_id": "fin_9564d5dceb3b6c6b3d219e59447418e2", "status": "approval_pending", ...}

### finance_propose refusal (older lease restored: rollback detected)
<- ERROR policy_denied: signed enable lease rollback or equivocation detected
```

## Where it lives

The lease file itself is operator-managed, owner-only (`0600`), and
non-symlink; the anchor lives in `SINGULARITY_FINANCE_STATE_DIR`. Rotating
the lease means signing a new document with a strictly newer `issued_at`.

## Not to be confused with

- **The signed policy** — the standing law: beneficiaries, limits,
  authorities. The lease is the consent that the law may be applied right
  now; both must hold simultaneously.
- **A mutation lease in other Wisent products** — those serialize competing
  writers; this lease authorizes nobody and races nobody — it is a dead-man
  switch for the whole finance surface.
- **The timelock** — a per-transaction delay between approval and signing;
  the lease gates every transaction at once.
