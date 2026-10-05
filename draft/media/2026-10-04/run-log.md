# QA publication run log — 2026-10-04

## Selected candidate

- Slug: `cilium-hubble-flow-attributed-to-wrong-policy-rule-on-overlap`
- Question: why Hubble attributes a flow to the wrong policy rule when two
  overlapping L3/L4 policy entries match, when the BPF datapath enforces the
  correct entry.
- Source: opt-in archive thread sighting the Hubble misattribution, answered
  in-thread against cilium/cilium#48945 and the fix PR cilium/cilium#49062.
  The thread was first sighted on 10-03 and deferred (fix still in review);
  on 10-04 the fix had landed on main (two commits, 2026-09-29) and the
  upstream issue was closed, making it publishable.
- Other 10-04 archive threads: the LoadBalancer shared-VIP frontend thread
  (re-post of the 10-02 published question), the GnuTLS HPACK thread
  (10-01 published), the OTEL env-var thread (10-03 published), a
  high-frequency socket-layer benchmark request (too thin to publish), and
  an OTel GenAI semantic-conventions PR request (out of scope). All covered
  in the page's "Community discussion today".

## Verification against public primary sources

- `bpf/lib/policy.h` @ v1.20.2: datapath precedence ladder — specific entry
  at `MAX_PRECEDENCE` short-circuits; otherwise higher precedence wins; at
  equal precedence the longer `lpm_prefix_length` entry wins; tie selects the
  specific-identity entry. Per-entry BYTES/PACKETS counters via policy
  accounting.
- `pkg/policy/mapstate.go` @ v1.20.2: inverted comparison on the allow path
  (`idKey.PrefixLength() > aggKey.PrefixLength()` returns the aggregate
  entry) — the opposite of the datapath; deny path unconditionally returns
  the specific entry at equal precedence.
- `main`: corrected by two commits on 2026-09-29 (issue reporter): the
  allow-path comparison flip plus a deny-path follow-up that routes
  same-precedence denies through the prefix comparison. Confirmed present
  in `v1.21.0-pre.3`; absent from stable v1.20.x. PR #49062 (AI-generated,
  declined, closed unmerged) independently flagged the deny-path variant in
  review.

## Privacy + content

- No names/handles/Slack-Discord URLs/exact timestamps/IPs/credentials in
  either page; anonymized summary only. Public primary-source links in
  `## References` only.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, `## References`, community
  discussion. Mobile-clean rendering via short inline code tokens; the Go
  comparison sits in a fenced block (internal scroll, no horizontal page
  overflow at 390px).

## Coverage disclosure

Two opt-in archive channels covered (11 messages in
`snapshot-2026-10-04.txt`). Visible-browser-only sources (Discord,
eunomia-bpf and sched-ext communities, bpf mailing list, r/eBPF) were not
reviewed this run (no visible-browser session) — marked uncovered-not-quiet
on the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-04-cilium-hubble-flow-attributed-to-wrong-policy-rule-on-overlap.md`
- ZH: `docs/ebpf-qa/2026-10-04-cilium-hubble-flow-attributed-to-wrong-policy-rule-on-overlap.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest Answers`) and
  `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `bc19e1c92`
  (`docs(ebpf-qa): cilium-hubble-flow-attributed-to-wrong-policy-rule-on-overlap (2026-10-04)`).
- This run-log commits separately as the 10-04 same-day artifact.
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-04.json`
  (written by the validator only).
