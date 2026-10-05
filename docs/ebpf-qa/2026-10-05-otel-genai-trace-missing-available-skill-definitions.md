# Why can a trace not show which skills an agent could choose from, and how does the new skill definitions attribute make that list visible?

The conventions today record which skill actually ran, not which were on the table. The merged `gen_ai.skill.name`, `gen_ai.skill.description`, `gen_ai.skill.source.uri`, and `gen_ai.skill.resource.name` attributes sit on `execute_tool` spans, so a trace shows what the agent did with the skill it picked. What it cannot show is what was never picked and why: when an agent skips a relevant skill or grabs a similar wrong one, the trace cannot say whether the skill was never offered, was offered with a poor description, or simply lost to a better-matching neighbor. The new `gen_ai.skill.definitions` attribute (open pull request 557 in the GenAI semantic-conventions repo) puts the offered list itself on the internal `invoke_agent` span, so the same trace can answer "what could the agent have chosen from?"

## The mechanism

Skill telemetry was deliberately keyed on attributes rather than on a span type, because the tool names differ per framework: one framework offers `load_skill`, `load_skill_resource`, and `run_skill_script` as separate tools; another names the same lifecycle stages differently. What got recorded is the resolved skill — its `name`, `description`, `source_uri`, and the skill-relative resource that was touched — on whichever `execute_tool` span performed the load. The blind spot that this leaves is exactly the "which ones were offered" question: only the chosen skill is visible.

The new attribute fills that half. `gen_ai.skill.definitions` is an opt-in, development-stability attribute on the internal `invoke_agent` span, recorded at invocation start. Its value is an array of skill definitions. Each definition follows the Agent Skills specification — a skill is a folder whose `SKILL.md` frontmatter makes `name` (1 to 64 characters, lowercase alphanumeric and hyphens) and `description` (1 to 1024 characters) required — so a definition carries exactly that pair, plus optional `source_uri` (mirroring `gen_ai.skill.source.uri`), `compatibility`, `license`, `metadata`, and the experimental `allowed-tools` field.

The obvious alternative, reusing `gen_ai.tool.definitions`, does not fit. That attribute describes function-call tools, whose identity includes a parameter schema; a skill has no parameter schema. It is a folder of instructions and optional bundled resources, and forcing it into the tool shape would invent a schema that does not exist.

The data the attribute standardizes already exists: coding agents write the list of skills they offer to the model into the session transcript on disk (one transcript has a `skill_listing` entry with the skill names and descriptions; another carries an "Available skills" block in its first developer message). The attribute gives that list a standard place in a trace instead of making every trace UI scrape per-agent transcripts.

## Verification and debugging path

1. In a trace, look at the agent's internal `invoke_agent` span for `gen_ai.skill.definitions`. Absence means the instrumentation does not emit it or the opt-in is off — it is not evidence that no skills were available.
2. Look at the `execute_tool` spans named `load_skill` for the `gen_ai.skill.*` attributes: that is the "what actually ran" half, including where the skill came from via its `source.uri`.
3. The diagnosis is the comparison. If the skill you expected is not in the definitions list, it was never offered. If it is in the list, the failure is description quality or similarity ambiguity, which the recorded `description` values let you check.
4. When present, validate the value against the pull request's JSON schema: an array; every entry has a `name` matching `^[a-z0-9]+(-[a-z0-9]+)*$` and a non-empty `description`. Remember the semantics: the list is a snapshot at invocation start, so skills added or removed mid-run are not reflected.

## The limitation

- The pull request is open as of this run; the attribute is not part of a released convention, and development stability means the shape can still move before release.
- It is opt-in, so most instrumentations will not emit it unless enabled. A missing attribute is not evidence that no skills were available.
- The registry flags the attribute as possibly sensitive: skill names and descriptions can describe internal processes and data. Decide deliberately what you send to a collector.
- Optional properties are off by default because the list can be large; `source_uri` and `compatibility` require enabling the instrumentation's option.
- The attribute records the offered set, not the correct set: it turns "was it ever offered?" from a guess into a fact, but a vague description only becomes a diagnosable problem once you read the recorded descriptions.

## References

- [semantic-conventions-genai pull request 557](https://github.com/open-telemetry/semantic-conventions-genai/pull/557) — adds `gen_ai.skill.definitions` to the internal `invoke_agent` span, with the schema, reference scenarios, and the motivation.
- [semantic-conventions-genai pull request 498](https://github.com/open-telemetry/semantic-conventions-genai/pull/498) — merged 2026-09-29: the `gen_ai.skill.name`, `gen_ai.skill.description`, `gen_ai.skill.source.uri`, and `gen_ai.skill.resource.name` attributes on `execute_tool` spans.
- [semantic-conventions-genai — GenAI attributes registry](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md) — the published `gen_ai.skill.*` and `gen_ai.tool.definitions` definitions the new attribute contrasts with.
- [semantic-conventions-genai pull request 557 — skill-definitions JSON schema](https://github.com/open-telemetry/semantic-conventions-genai/blob/0e098d9bef6118b36819f83ccea438fa8dba09b0/model/gen-ai/gen-ai-skill-definitions.json) — the `SkillDefinition` shape: required `name` and `description`, optional `source_uri`, `compatibility`, `license`, `metadata`, and `allowed-tools`.
- [Agent Skills specification](https://agentskills.io/specification) — a skill is a folder with a `SKILL.md` whose `name` and `description` frontmatter fields are required.

## Community discussion today

The selected question came from an opt-in archive: a thread announcing a pull request that records which Agent Skills were available to an agent when an invocation started, so a trace can distinguish "the skill was never offered" from "it was offered but a poor or similar match won" — the ask is review of the new attribute, its schema, and its placement on the agent-invocation span. The thread also noted that coding agents already write the offered-skill list into session transcripts, which is what makes a standard trace attribute feasible without new data collection.

Other threads this day: an OpenTelemetry eBPF instrumentation thread on the Kubernetes cache address environment variable being ignored when the Helm chart renders a Config v2 document — a re-post of the question published two days ago, now carrying a helm-charts pull request, in-thread confirmation that it is a real issue, and a manual workaround that wires the cache address into the Kubernetes enricher config; not republished because the published answer already covers it. A second thread asked for kernel-space drop latency versus user-space context-switch benchmarks under thousands of concurrent socket-layer retries per second with heavy ring-buffer load; no public primary source or decisive boundary was available, so it stayed unpublished.

Channel coverage for this run: the two opt-in archives provided eight messages, all covered above. The visible-browser-only sources (Discord, the eunomia-bpf and sched-ext communities, the bpf mailing list, and r/eBPF) could not be reviewed in this run — no visible-browser session was available — so they are marked uncovered, not quiet.
