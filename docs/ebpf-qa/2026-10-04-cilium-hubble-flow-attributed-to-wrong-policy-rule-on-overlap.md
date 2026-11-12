# Why does Hubble attribute a flow to the wrong policy rule when two overlapping L3/L4 policy entries match, even though the BPF datapath enforces the correct one?

Enforcement is not affected — the BPF datapath picks the correct entry. The misattribution lives in the userspace copy of the policy decision: `pkg/policy/mapstate.go` is a Go re-implementation of the same entry selection the C datapath makes, and in v1.20 its LPM prefix comparison was inverted when the lookup was refactored. `mapState.lookup()` returns the *other* entry for two same-precedence overlapping rules, so Hubble credits the flow to the wrong rule in `egress_allowed_by` while the recorded `policy_match_type` describes the entry the datapath actually matched. The record contradicts itself.

## The mechanism

Cilium stores L3/L4 policy in one LPM-trie keyed map. For a flow, the datapath performs two lookups: one under the specific remote identity and one under that identity's aggregate (for example `aggregate-cluster` for in-cluster pods, or 0 for world traffic). When both entries exist, `bpf/lib/policy.h` applies a fixed precedence ladder:

1. A specific-identity entry at the maximum precedence (a priority-0 deny) is taken without looking at the aggregate entry.
2. Otherwise the higher-precedence entry wins.
3. At equal precedence, the entry with the longer `lpm_prefix_length` wins — that is, the more specific L4. A wildcard-port entry has a short prefix (protocol-only), a fixed-port entry the full protocol-plus-port length, so a port-80 rule outranks an any-port rule at the same precedence.
4. A tie in prefix length selects the specific-identity entry.

The per-entry byte and packet counters that policy accounting maintains are what make this observable: whichever entry the datapath matched is the one whose counter moves.

The Go copy in v1.20 inverts step 3 on the allow path:

```go
if idKey.PrefixLength() > aggKey.PrefixLength() {
    return authOverride(aggEntry, idEntry), true
}
```

Read against the datapath, the condition is backwards: it returns the aggregate entry when the specific entry has the longer prefix, and falls through to the specific entry when the aggregate has the longer prefix. Only the tie case coincides. The inversion was introduced when the lookup was refactored; before that refactor the comparison matched the datapath.

The deny path had the same flaw in another form: a same-precedence deny pair always returned the specific entry, whereas the datapath does that only for a priority-0 deny and otherwise still compares prefix lengths. The verdict is the same on both sides, so enforcement is unaffected either way — only which rule is credited differs.

## Verification and debugging path

The upstream reproducer (in the issue below) shows the full public path:

1. Create two same-precedence overlapping entries for the same flow, with different L4 specificity. For example, a specific-identity allow with a wildcard port next to an aggregate allow (`toEntities: all`) with a fixed port, so the datapath must pick the port entry.
2. Read the datapath truth: `cilium-dbg bpf policy get <endpoint-id>` lists every policy-map entry with its per-entry BYTES/PACKETS counters (policy accounting is on by default). After sending the traffic, the counters on the longer-prefix entry move and the other entry's counters stay put.
3. Read the userspace credit: `hubble observe` in JSON, and compare `egress_allowed_by` (which rule Hubble attributes the flow to) against `policy_match_type` (which entry's character it reports).
4. The signature of this bug is the mismatch between the two: the counters say one entry was matched, the attribution says the other rule produced the verdict, and the match type describes a rule shape that the credited rule does not have (for example an L4-only match type credited to a portless L3-only rule). The datapath verdict — allow or deny — is correct; only the correlation is off.

## The limitation

- Packet enforcement was never affected, in any direction. The C datapath is correct; the defect is confined to the userspace copy, so everything that reads it — Hubble's policy correlation, and the Go-side tests that use the same lookup — inherits the wrong rule attribution, never the wrong verdict.
- As of this run date, the stable v1.20 line (through v1.20.2) still carries the inverted comparison. The fix landed on `main` on 2026-09-29 as two commits pushed by the issue reporter — the allow-path correction, plus a follow-up that routes same-precedence denies through the prefix comparison instead of returning the specific entry unconditionally — and it first ships in the v1.21.0 pre-release line (confirmed present in v1.21.0-pre.3). No 1.20.x backport was in flight at the run date.
- The upstream issue was closed citing a flood of automated reports and, at that point, no confirmed real-user complaint; the reproducer above does demonstrate a concrete user-visible impact on Hubble output. Until you are on a release carrying the fix, do not rely on Hubble's rule attribution when same-precedence rules with different port specificity overlap — cross-check it against the per-entry counters, or plan the upgrade to the 1.21 line.

## References

- [cilium/cilium issue 48945](https://github.com/cilium/cilium/issues/48945) — the upstream report: the inverted comparison, the C datapath rule it contradicts, and a kind-cluster reproducer with counter and Hubble output.
- [cilium/cilium pull request 49062](https://github.com/cilium/cilium/pull/49062) — a community fix PR that was declined and closed unmerged; its review comments independently flag the deny-path variant of the same symptom.
- [Cilium v1.20.2 — bpf/lib/policy.h](https://github.com/cilium/cilium/blob/v1.20.2/bpf/lib/policy.h) — the datapath precedence ladder, the LPM prefix-length constants, and the per-entry policy accounting.
- [Cilium main — pkg/policy/mapstate.go](https://github.com/cilium/cilium/blob/main/pkg/policy/mapstate.go) — the corrected userspace lookup, with the fixed comparison.
- [Cilium — Hubble observability](https://docs.cilium.io/en/stable/observability/hubble/) — how flow records carry policy correlation.

## Community discussion today

The selected question came from an opt-in archive: a thread reporting that a flow matching two overlapping policy rules is attributed by Hubble to the wrong one, answered in-thread by pointing at the already-tracked upstream issue and its in-review fix — datapath correct, only the Go-side copy off, with the note that the helper that wrapped the result was already gone on main while the comparison itself was still wrong. The thread was first sighted the previous day and deferred, because the fix was then still under review; by this run the fix had landed on main and the upstream issue was closed, which made it publishable as a complete, verified answer.

Other threads this day: a continued Cilium LoadBalancer shared-VIP frontend-ownership thread (a re-post of the question published two days ago, now with a maintainer pointing to a tracked upstream issue and a fix in review), a GnuTLS HTTP/2 per-connection HPACK decoder thread (published as a question a day earlier), a high-frequency socket-layer drop-latency versus user-space context-switch benchmark request that was too thin to publish, an OTEL Kubernetes cache environment-variable thread (published a day earlier), and an OpenTelemetry GenAI semantic-conventions thread about recording which skills were available to an agent at invocation start (a semantic-conventions pull request, out of scope for this page).

Channel coverage for this run: the two opt-in archives provided eleven messages, all covered above. The visible-browser-only sources (Discord, the eunomia-bpf and sched-ext communities, the bpf mailing list, and r/eBPF) could not be reviewed in this run — no visible-browser session was available — so they are marked uncovered, not quiet.
