# Support Matrix and Operational Limits

This page explains what ActPlane can enforce today, what requires BPF-LSM, and
what should be reviewed with `actplane compile --explain` before use.

Always start with:

```bash
actplane compile --explain --report-out docs/actplane-review.txt
actplane doctor
```

Use `--json` for CI:

```bash
actplane compile --json > actplane-compile-report.json
```

## Effects

| Effect | What it means | Requires BPF-LSM | Use when |
| --- | --- | ---: | --- |
| `notify` | Record feedback; operation proceeds | No | Observe-first rollout, reminders, non-fatal workflow guidance |
| `block` | Pre-operation denial in a BPF-LSM hook | Yes | Security-sensitive file/network/process policy that must not commit |
| `kill` | Terminate the matching task and emit feedback | No for tracepoint-backed operations | Argv-sensitive exec rules such as `git push`, `git commit`, `git branch` |

If several clauses match one event, ActPlane chooses the strongest effect:

```text
kill > block > notify
```

## Operation Coverage

| Operation pattern | Pre-op `block` | `kill` / `notify` | Notes |
| --- | --- | --- | --- |
| `exec "git"` | Yes, with BPF-LSM | Yes | Executable identity can be denied pre-op. |
| `exec "git" "push"` | Not pre-op today | Yes | Argv-token predicates are observed after exec; use `kill`. |
| `open file PAT` | Yes, with BPF-LSM | Yes | Use for mandatory mediation and protected files. |
| `read file PAT` | Yes, with BPF-LSM | Yes | Also introduces source labels from file sources. |
| `write file PAT` | Yes, with BPF-LSM | Yes | Covers writes, creates, truncates when hooks are active. |
| `unlink file PAT` | Yes, with BPF-LSM | Yes | Use for destructive-operation policy. |
| `connect endpoint PAT` | Yes for supported endpoint forms, with BPF-LSM | Yes | Numeric IPv4 support is strongest today. |
| `recv endpoint PAT` | Yes for connected IPv4 recv, with BPF-LSM | Yes | Endpoint-source ingress support depends on hook profile. |

## Pattern Support

| Pattern class | Support | Notes |
| --- | --- | --- |
| Exec basename, e.g. `exec "git"` | Supported | Treated like `exec "**/git"`. |
| Exec path glob, e.g. `exec "**/pytest"` | Supported | Review lowered matchers with `compile --explain`. |
| Single argv token, e.g. `exec "git" "push"` | Supported post-exec | Use `kill` or `notify`; not pre-op `block`. |
| File exact/prefix/suffix/any globs | Supported | Real `(dev,inode)` identity in LSM mode where available. |
| Endpoint numeric IPv4, e.g. `10.0.0.` or `127.` | Supported | This is the recommended endpoint policy form today. |
| Hostname endpoint glob | Surface syntax accepted, kernel support limited | `compile --explain` reports support details. |
| IPv6 endpoint glob | Surface syntax accepted, kernel support limited | `compile --explain` reports support details. |
| Path contains/suffix in runtime deltas | Requires profile reservation | Deltas cannot introduce hook/matcher classes not reserved at load time. |

## Hook Profiles

ActPlane loads the hook classes needed by the compiled policy. Some runtime
deltas can only be accepted later if the relevant hook and matcher classes were
reserved when the engine loaded.

| Setting | Effect |
| --- | --- |
| default profile | Load the policy-selected attach set. |
| `ACTPLANE_RESERVE_FILE_FLOW=1` | Reserve file-flow hooks for later runtime deltas. |
| `ACTPLANE_ENABLE_ADVANCED_HOOKS=1` | Enable advanced file-flow hooks. |
| `ACTPLANE_HOOK_PROFILE=full` | Enable file flow, network, and block hook classes for future deltas. |

Use the full profile for long-running MCP/watch sessions that will accept child
domain deltas whose final policy is not known at startup.

## BPF-LSM vs Tracepoint Mode

| Capability | BPF-LSM active | Tracepoint-only mode |
| --- | --- | --- |
| Pre-op denial with `block` | Yes | No |
| Post-event `notify` | Yes | Yes |
| Post-event `kill` for supported events | Yes | Yes |
| Real file identity | Strongest | Available for many fd-backed events; fallback uses path hash |
| Argv-token exec policy | `kill`/`notify` after exec | `kill`/`notify` after exec |
| Security claim for "operation never committed" | Use `block` | Do not claim pre-op denial |

Tracepoint-only mode is still useful for observation, corrective feedback, and
many harness-level policies. Use BPF-LSM for hard security boundaries.

## Data-Flow Semantics

Labels propagate across:

- fork and exec lineage
- file reads and writes
- supported network receive/send paths
- supported fd duplication and IPC paths in advanced profiles

This is conservative. A process that reads a small secret value can taint later
files or processes even if some later output does not literally contain the
secret. That over-tainting is intentional: ActPlane favors preserving
provenance over silently missing a derived flow.

## Temporal Gates

`after exec G` is latching: once it has happened in a lineage, it remains true.

Use `since` for freshness:

```text
after exec "**/pytest" exits 0 since write "src/**"
```

This means the gate is valid only if `pytest` exited 0 after the latest write to
`src/**`. Any later matching write makes the gate stale.

## Runtime Deltas

Runtime deltas are append-only policy changes applied to a running engine:

```bash
actplane control delta add --target-id <domain-id> --delta policy-delta.dsl
```

Allowed delta shape:

- add local bindings
- add labels
- add gates
- add restrictions
- narrow scope
- create child domains within delegated authority

Rejected delta shape:

- remove inherited rules
- weaken parent policy
- widen scope
- remove labels or gates
- mutate an existing rule definition
- introduce hook or matcher classes not reserved at load time

If configured, runtime-delta admission can require metadata:

```bash
actplane control delta add \
  --target-id <domain-id> \
  --delta policy-delta.dsl \
  --approved-by alice \
  --approval-ref REVIEW-123 \
  --generated-by codex
```

The current admission gate is deterministic local metadata checking. It is not
a cryptographic signature or external ticket-system verifier.

## Attach Limits

`actplane attach --pid <pid>` is post-hoc:

```bash
sudo -E actplane attach --pid <pid>
```

It protects future events from the attached process tree, but it does not
reconstruct:

- prior file reads or writes
- prior network flows
- prior labels
- prior temporal gates
- prior ancestry outside the attached tree

For strict history from process start, use:

```bash
sudo -E actplane run -- <cmd>
```

or project MCP auto-attach:

```bash
actplane init --with-mcp
```

## Recommended Rollout

1. Generate or choose a policy.

   ```bash
   actplane init --list-templates
   actplane init --template no-git-branch --out actplane.yaml
   ```

2. Review support and limits.

   ```bash
   actplane compile --explain --report-out docs/actplane-review.txt
   actplane doctor
   ```

3. Start with `notify` or a narrow `kill` policy when possible.

4. Move to `block` only after `compile --explain` confirms BPF-LSM support for
   the relevant clauses.

5. For long-running parent/child sessions, reserve the hook profile needed by
   expected child deltas before the engine starts.

6. Keep `.actplane/last-violation.txt` with incident artifacts when a policy
   fires; it is the human-readable record of the corrective reason.
