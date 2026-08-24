# Repo surface

`singularity-repo-mcp` is a stdio MCP server for policy-bound repository
work: an agent gets an isolated worktree, edits only allowed paths, seals
the exact tree it proposes, proves its checks ran against that sealed tree,
and opens a pull request — with external CI and human review remaining the
final gates. It speaks MCP protocol `2024-11-05` and caps requests at
2 MiB.

## Configuration

| Variable | Meaning |
|---|---|
| `JEDEN_REPO_POLICY_FILE` | Absolute path to the repository policy file. |
| `JEDEN_REPO_STATE_DIR` | Absolute path to the durable proposal state directory. |

The policy declares, per `repo_id`: the source repository `root`, `remote`,
`base_branch`, `branch_prefix`, the GitHub repository and head owner, the
`allowed_paths` an agent may touch, the available `checks` (each with a
kind and timeout), and which of them are `required_checks`. Unknown fields
are rejected.

## Tools

| Tool | Contract |
|---|---|
| `workspace_create` | Create an isolated policy-bound worktree from a clean source repository (`repo_id`, `workspace_id`, `request_id`). |
| `workspace_read` | Read one bounded UTF-8 file inside `allowed_paths`. |
| `workspace_apply_patch` | Apply one bounded unified diff restricted to `allowed_paths`. |
| `workspace_diff` | Return the bounded unified diff for the workspace. |
| `workspace_seal` | Stage allowed roots and seal the exact Git tree object and bounded diff. |
| `workspace_check` | Run the fixed `git_diff_check` against the exact sealed index and retain tree-bound evidence. |
| `commit_create` | Commit the sealed index without restaging, only after exact successful required evidence. |
| `branch_publish` | Reconcile then publish the committed proposal branch without force. |
| `pull_request_open` | Reconcile then open a pull request to the policy base branch. |
| `proposal_status` | Return durable proposal lifecycle state without changing it. |

The mutating tools take a `request_id`, following the same idempotency
pattern as [the finance boundary](finance.md): recorded requests replay
their recorded response instead of re-executing.

## Where it fits

The being does not talk to this server directly; repository work reaches the
model as namespaced tools through [Las](skills.md) (the `warsztat` surface
in the default `LAS_ONLY` selection), keeping `singularity-repo-mcp` a
separate process with its own policy and failure behavior, like every other
surface with authority.
