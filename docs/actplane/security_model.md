# Security Model

ActPlane's reusable engine has two layers:

```text
IFC core:       object + label + event + rule
Runtime policy: domain + binding + delta + authority
```

Agent, subagent, MCP, hooks, prompts, and workspaces are integrations above this
model. They are not part of the engine security model.

This is the security model implemented by the current ActPlane engine. Policies
compile to a fixed kernel ABI, the eBPF/BPF-LSM engine evaluates them below the
tool layer, and the runtime records the policy provenance needed to audit each
decision.

## Core Entities

An IFC rule is pure system policy:

```text
rule = condition + effect + reason
```

A domain is the runtime policy boundary for a process tree:

```text
pid -> domain
domain -> effective policy
event(pid) checks only pid's domain
```

Files, sockets, and stdio are not domains. They are IFC objects that carry
labels and participate in flow rules.

A binding attaches a rule to a domain:

```text
binding = domain + rule
```

All bindings are mandatory and monotonic: once a rule is bound to a domain, it
cannot be removed or disabled by the domain or its children.

## Two Logical YAMLs

It is useful to keep two logical files, even if a CLI later allows them in one
physical YAML.

Rule catalog:

```yaml
version: 1

rules:
  no-git-branch:
    ifc: |
      rule no-git-branch:
        kill exec "git" "branch"
        because "do not create git branches"

  no-network:
    ifc: |
      rule no-network:
        block connect any
        because "network is disabled by default"

  readonly:
    ifc: |
      rule readonly:
        block write file any
        because "this domain is read-only"
```

Domain policy:

```yaml
version: 1

domains:
  session:
    bind:
      - no-git-branch
      - no-network
```

The same rule can be bound by different domains:

```yaml
domains:
  review:
    parent: session
    bind:
      - readonly

  build:
    parent: session
    bind:
      - readonly
```

## Effective Policy

For a domain `D`:

```text
policy(D) = policy(parent(D)) + local(D)
```

The security invariant is monotonic tightening:

```text
policy(child) >= policy(parent)
```

Here `>=` means "at least as restrictive". A child domain inherits all parent
rules and may only add more rules. It cannot remove, disable, or weaken any
inherited rule.

## Child Updates

A child domain may update its own domain policy if its authority allows it.

Allowed updates:

```text
add local bindings
add labels
add gates
narrow scope
create child domains with no more authority than delegated
```

Rejected updates:

```text
remove inherited bindings
modify parent domain state
modify sibling domain state
widen scope
remove labels or gates
increase delegated authority
mutate an existing rule definition
enable hook classes or path-matcher classes that were not reserved when the engine loaded
```

## Examples

### 1. Child Adds a Rule for Its Own Children

Child domain:

```yaml
domains:
  review:
    parent: session
    bind:
      - readonly
```

Grandchild domain:

```yaml
domains:
  review-helper:
    parent: review
```

Result:

```text
review-helper inherits no-git-branch, no-network from session
review-helper inherits readonly from review
review-helper cannot remove any of these rules
```

### 2. Runtime Rule Addition

Rules can be added at runtime if they are submitted as compiled policy deltas.
The kernel should not parse YAML or DSL in the admission path.

Current CLI shape:

```bash
actplane control bind-child --pid 1234 --child-id 1234
actplane control delta add --domain-id 1234 --delta rules/no-curl.dsl
actplane control launch-child --delta rules/no-curl.dsl -- codex --cd /work
```

Semantics:

```text
userspace compiles rule DSL -> fixed taint_config entries
userspace submits append-rule/update delta over cap_req
kernel checks caller authority, target domain, loaded engine profile, and monotonicity
kernel installs accepted entries into the target domain's effective policy mask
```

MCP exposes the same path as `bind_child_domain`, `launch_child`, and
`append_policy_delta`. The kernel does not parse YAML or DSL.

## Runtime Delta Admission

User space does not directly mutate effective policy state. It submits deltas.

Delta classes:

```text
create_domain
bind_rule
add_rule_ir
add_label
add_gate
narrow_scope
append compiled update
append compiled rule
```

A delta contains only precompiled IDs and masks:

```text
caller_pid
domain_id
required_mask
add_label_mask
add_restrict_mask
add_gate_mask
new_scope_id
compiled update/rule entries
```

The kernel admits a delta only if:

```text
caller_pid is bound to a domain
caller may affect the target domain
caller has the required authority bits
scope only narrows
labels/gates/restrictions only add
new bindings do not weaken inherited policy
compiled entries fit the hook and matcher classes enabled in the loaded engine profile
```

Accepted deltas are merged into the domain's already-computed effective state.
The syscall fast path should not walk the domain tree.

## Rule Identity

Runtime-added rules are tracked by the compiled rule metadata and audit hash:

```text
rule_id = lowered kernel rule id
source_hash = hash(rule source text)
clause_hash = hash(lowered clause text)
```

Names such as `no-network` are user-facing aliases. Kernel events use numeric
lowered `rule_id` values. Userspace audit records bind those ids back to source
and clause hashes so policy deltas remain attributable.

## Current Implementation Mapping

The current low-level ABI still uses `target_id` in some structs. In this model,
that id is a domain id:

```text
cap_task[pid] -> domain id
cap_state[domain id] -> effective domain state
```

Current implemented fields:

```text
parent
scope_id
labels
authority_mask
target_mask
restrict_mask
gate_mask
label_mask
```

Implemented today:

```text
rule catalog in policy YAML
domain bindings in policy YAML
domain selection for policies that use domains
actplane compile --domains effective-policy view
actplane compile selected-domain summary and --json/--explain reports
starter actplane.yaml generated as a flat policy
binding resolution at compile time for domain policies
valid and invalid domain policy corpus tests
CLI UX tests for domain selection/errors
user ringbuf request path
cap_state runtime domain map
cap_task pid-to-domain binding
cap_policy per-domain rule mask
mask-based authority checks
monotonic labels/restrictions/gates/scope update
runtime bind-child and launch-child control paths
append-only compiled update/rule deltas
static metadata approval gate for append_policy_delta
```
