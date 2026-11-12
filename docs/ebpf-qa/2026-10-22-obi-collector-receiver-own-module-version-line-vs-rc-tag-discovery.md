# Why does the OBI Collector receiver keep its own module and version line, and why does a plain go get update not discover the receiver at an RC tag?

Two independent barriers. OBI split its Collector receiver into a nested Go module - the `collector/` subdirectory carries its own `go.mod`, and the import path stays the same - and that nested module is released on its own v0 line, tagged `collector/v0.15.2`, while the core module tags `v0.15.2` and `v1.0.0-rc.2` without a prefix. A plain `go get -u` never reaches the RC for two separate reasons: Go's `latest` and `upgrade` queries prefer release versions over pre-releases, and `v1.0.0-rc.2` is a pre-release; and the receiver module has no tag on the v1 line at all - `collector/v1.0.0-rc.2` does not exist - so on the receiver side the highest selectable version is on the v0 line. Testing the receiver against the RC requires selecting both module versions explicitly in one update.

The split itself is the fix for the v0.15.x release saga: when a commit carried both a core tag and a Collector tag, `git describe` substituted the Collector version into core release artifact paths, and the module split (#3748) plus the pipeline fix (#3770) is what let `v0.15.2` ship the correction.

## The mechanism

1. **The receiver is a nested module on its own release line.** As of `v0.15.2`, the `collector/` subdirectory contains a `go.mod` declaring the receiver module `go.opentelemetry.io/obi/collector`; the receiver packages keep their existing import path, and the nested module requires the core at `v0.15.2`. Go's rule for modules in subdirectories: every version tag is prefixed with the module subdirectory - here `collector/v0.15.2` - while a root module tags plain `v0.15.2` and `v1.0.0-rc.2`. Because Go does not require a major version suffix below v2, the receiver can keep an independent v0 line while the core walks the v1 RCs; that is the "own module and version line" from the release notes.

2. **Why the split happened.** `v0.15.0` was blocked by BPF verifier failures; `v0.15.1` shipped the correction, but its artifact build failed because version discovery ran `git describe` on a commit that carried both a core tag and a Collector tag, and substituted the Collector version into core artifact paths. The pipeline fix pins artifact paths (binary archives, generated sources, SBOMs, checksums) to the selected core tag, and `v0.15.2` "supersedes the unpublished v0.15.0 and v0.15.1" by carrying the correction and the module split together.

3. **Why `go get -u` cannot discover the RC.** Go's version queries `latest`, `upgrade`, and `patch` prefer release versions over pre-release versions: if `v1.2.2` and `v1.2.3-pre` are both available, `latest` selects `v1.2.2` even though the pre-release sorts higher. Two consequences for OBI. On the core module, `v1.0.0-rc.2` is a pre-release, so a plain update resolves to the highest non-pre-release - for the core that is `v0.15.2` (a v0 release is still a "release" in query terms; it simply carries no compatibility guarantee). On the receiver module, the v1 line has no tag: `collector/v0.15.2` is its only tag, and there is no `collector/v1.0.0-rc.2` to update toward. The RC2 release notes state this directly: "RC tags alone do not make the split receiver discoverable through a plain `go get -u`."

4. **The supported pairing is an explicit two-module selection.** One `go get` names both module versions: the receiver module at `v0.15.2` and the core module at `v1.0.0-rc.2`. Both then appear as separate `require` entries in `go.mod`, and the receiver's minimum requirement on the core is satisfied by the explicitly selected RC version.

## Verification and debugging path

1. **Read the two tag lines directly.** `git ls-remote --tags` on the OBI repository shows the core line (`v0.15.2`, `v1.0.0-rc.2`) and the subdirectory-prefixed receiver line (`collector/v0.15.2`); the absence of a `collector/v1.0.0-rc.2` entry is the concrete fact no update query can discover.
2. **Confirm what the query selects.** `go list -m -versions` lists tagged versions including RC tags, but `go get -u` resolves to the highest non-pre-release - so a plain upgrade leaves the core on the v0 line and the receiver at `collector/v0.15.2`.
3. **Pin the RC pairing explicitly.** The pairing from the RC2 notes: one `go get` selecting the receiver module at `v0.15.2` and the core module at `v1.0.0-rc.2`. After the update, `go.mod` carries both `require` entries and `go.sum` has entries for both module lines.

## The limitation

- The published telemetry schema in RC2 is still `0.15.2`; the `1.0.0` schema cut happens at the final release, so RC2 output is not the final v1 contract.
- Telemetry names and labels changed between RCs, so existing dashboards and queries need remapping on upgrade.
- The receiver's `v0.15.2` line is a v0 line: no compatibility guarantee, and it can move independently of - or lag - the core.
- The release preference of `go get -u` is a property of Go's module queries in general, not an OBI quirk: no plain update moves any module to a pre-release tag.
- This answer describes the `v1.0.0-rc.2` candidate; the final `v1.0.0` release may add a receiver tag on the v1 line and close the asymmetry.

## References

- [OBI release v1.0.0-rc.2](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/releases/tag/v1.0.0-rc.2) - release-impacting changes: the receiver as the separate module `go.opentelemetry.io/obi/collector`, independent v0 release line, no `collector/v1.0.0-rc.2` tag, the explicit two-version `go get` pairing, Config v2 as the proposed v1 config contract with Config v1 frozen, and the telemetry schema staying at `0.15.2` until the final release.
- [OBI release v0.15.2](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/releases/tag/v0.15.2) - why `v0.15.2` supersedes the unpublished v0.15.0 and v0.15.1: BPF verifier failures that blocked v0.15.0, the `git describe` Collector-tag substitution into artifact paths (fixed by #3770), the verifier loop widening fix (#3759), and the receiver split (#3748).
- [Go Modules Reference](https://go.dev/ref/mod) - "Version queries": release versions preferred over pre-release versions for `latest`, `upgrade`, and `patch`; "Module directories within a repository": subdirectory modules use subdirectory-prefixed tags; "Mapping versions to commits": root modules use plain version tags.
- [OBI v1.0.0 tracking issue](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/1133) - the RC-to-final path and the schema cut.

## Community discussion today

The monitored window for this run is a 7-day archive window across the two opt-in archive sources, returning five messages, all on one thread: the OBI v0.15.x to v1.0.0-rc.2 release sequence. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.

**The OBI v0.15.x release saga.** The thread opened with `v0.15.0` blocked by BPF verifier failures, the resolution already on main and a v0.15.1 promised; a follow-up reported that v0.15.1's artifact build had failed; and finally `v0.15.2` was announced as containing the separate module and the fixes needed, with v0.15.2 confirmed working. The practical step for operators on the v0 line: use `collector/v0.15.2` for the receiver. The artifact-path correction is a pipeline-side change, not something users must work around.

**OBI v1.0.0-rc.2 community testing.** The window closes with the announcement pushing `v1.0.0-rc.2` for community testing: Go Auto SDK span capture, expanded .NET instrumentation and runtime metrics, AWS Route53 name resolution, and fixes for BPF verification, trace context propagation, and metrics. Upgraders from RC1 face the Config v2 migration and the telemetry compatibility changes: telemetry names and labels changed, and the Collector receiver now has its own module and version line, which is what this page answers. Unresolved boundary: the schema is still `0.15.2`, and the receiver has no v1-line tag.

**A new contributor's instrumentation proposal.** A new contributor requested review of a proposed sentence-transformers instrumentation in the Python GenAI instrumentation repository (public issue #910 and PR #911 in the opentelemetry-python-genai project). The request is in the review queue; there is no community answer to summarize yet.
