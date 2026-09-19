# Why does the OBI config migration from v1 to v2 refuse to write a file when some fields have no mapping, and what is the middle ground?

**Short answer:** because `obi config migrate` is a behavior-preserving conversion, not a translation pass. If any Config v1 field has no Config v2 equivalent, the command aborts with `migration failed: fields are outside the supported v1-to-v2 migration contract: <fields>` and writes nothing (exit 1) — it deliberately will not emit a "mostly right" v2 file, because a silently dropped field would produce a config that validates but behaves differently. The middle ground the maintainers asked for now ships as an explicit flag: `obi config migrate --allow-partial <file>` writes a valid v2 config for every field it *can* carry, reports the unmapped fields, and signals the gap with a distinct exit code 3 (`ExitPartial`) instead of 0 or 1. Partial migration is opt-in, not the default, and it does not comment out the original v1 fields — it leaves them out of the v2 output and lists them so you can map each one by hand.

## The default is deliberately all-or-nothing

The migration command reads only the v1 file you hand it, converts it to Config v2, round-trips the result through the v2 validator, and checks that every input field it was supposed to preserve actually survives. In the source (`cmd/obi/internal/configcmd/configcmd.go`), the unmappable fields are accumulated into an `unsupported` list; the command computes `partial := len(unsupported) != 0`, and then:

```
if partial && !options.allowPartial {
    return nil, "", false, fmt.Errorf(
        "fields are outside the supported v1-to-v2 migration contract: %s",
        strings.Join(unsupported, ", "),
    )
}
```

So with no flag, a single unmappable field fails the whole run and no v2 file is written. The reasoning is that `migrate` promises a behavior-preserving conversion: silently dropping a field could yield a v2 config that looks valid but changes what OBI captures. The docs state the same contract — "If the migration command cannot preserve a setting, it fails and identifies the Config v1 field in the error message. Replace or retire the unsupported behavior before you rerun the command." This is why, for a config that mixes mappable and unmappable fields, the only options before the flag existed were to strip the offending fields, migrate, and re-add the stripped ones by hand.

## The middle ground now ships as an explicit flag

The exact "middle ground" a user was asking about — migrate what you can, report the rest — is now a first-class option rather than a proposal. In `configcmd.go`:

```
allowPartial := flags.Bool("allow-partial", false,
    "write valid v2 output when some v1 fields cannot be preserved")
...
if partial {
    return output, partialMigrationReport(replaced, unsupported), true, nil
}
```

and `runMigrate` returns `ExitPartial` (3) when a partial run produced a file:

```
ExitSuccess = 0   // fully migrated, no unmapped fields
ExitError   = 1   // parse/validation/migration failed (including all-or-nothing hit)
ExitUsage   = 2   // bad arguments
ExitPartial = 3   // v2 file written, but some v1 fields were not carried over
```

The important distinctions:
- `--allow-partial` writes a **valid** v2 config for every preservable field — it is not a lossy dump.
- It **does not** comment out the original v1 fields. Many v1 fields have no meaningful location in the v2 structure, so there is nowhere to re-nest them; the command omits them and names them in the partial report instead.
- The exit code 3 is the hook for automation: branch on it, read the reported field list, and hand-map each reported field to a supported Config v2 or Collector behavior. That is the intended workflow, not a workaround.

## How to run a migration without losing fields

1. Save the current v1 file and record the OBI binary version (the command only reads the file you give it; env-var, flag, or Helm-supplied settings are not migrated).
2. Run `obi config migrate <v1.yaml>` first. A clean run exits 0 and prints `migrated v1 config to OBI config v2`. A hit on an unmappable field exits 1 with the field list.
3. For the mixed case, run `obi config migrate --allow-partial <v1.yaml>`. You get a generated v2 file for the mappable subset; the exit code is 3 and the partial report lists every field that did not carry over.
4. `obi config validate <generated-v2.yaml>` to confirm the output is well-formed, then in a canary deployment verify behavior. For every reported field, pick a supported v2/Collector equivalent (the migration guide's "Handle settings that need manual changes" table maps the common ones, e.g. `service_name` → top-level `resource` attributes) and apply it in the v2 file.
5. Note a behavior-change to preserve: a v1 file with no selection fields disables application capture, whereas v2 *includes* workloads by default — so the migration sets `default_action: exclude` when the source did not select a workload. Reintroducing v2 default inclusion after migration would silently capture more than the v1 config did.

## The second boundary: the Helm chart still emits a v1-only field

A separate, compounding gap is on the chart side, not the migration code. The `opentelemetry-ebpf-instrumentation` Helm chart renders a v1-style field into the config it injects (from its `_helpers.tpl`), and that field is not part of the Config v2 schema — so a v2-migrated config rendered through an un-updated chart fails validation even when the migration itself is clean. That is a chart defect, not an OBI migration-code defect: the fix is to update the chart so it stops injecting the v1-only field (or add v2-aware rendering), and it belongs in the `opentelemetry-helm-charts` repo, reported with the chart and OBI versions, the relevant values, the rendered config, and the validation error.

## The limitation that decides it

The decisive boundary is that migration is a **validation-gated, all-or-nothing conversion by default**, and its escape hatch is an **explicit, report-producing partial mode with its own exit code**, not a silent best-effort. The reason the default is not to emit a partial file and to not comment out the un-mappable fields is that a "valid-looking" v2 config with a dropped or re-homed field can validate and run while changing what OBI observes. So the practical rule is: run `--allow-partial` only when you will actually read the partial report and hand-map the listed fields; keep the plain `migrate` as the gate for configs that must migrate losslessly. And keep the chart gap separate — a clean v2 migration can still fail at validation until the chart stops injecting a v1-only field.

## References

- [OBI docs: Migrate from OBI Config v1 to Config v2 (exit-status table; "If the migration command cannot preserve a setting, it fails and identifies the Config v1 field"; rejects unknown v1 fields; `default_action: exclude` when the source selects no workload)](https://opentelemetry.io/docs/zero-code/obi/configure/migrate-to-config-v2/)
- [OBI docs: Config v2 reference (`capture.policy` / `capture.rules`, `default_action`, how omission changes capture)](https://opentelemetry.io/docs/zero-code/obi/configure/config-v2/)
- [OBI source: `cmd/obi/internal/configcmd/configcmd.go` (`obi config migrate` — `--allow-partial` flag, `fields are outside the supported v1-to-v2 migration contract` error, `ExitPartial = 3`)](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/cmd/obi/internal/configcmd/configcmd.go)
- [OBI Helm chart: `opentelemetry-ebpf-instrumentation/templates/_helpers.tpl` (the v1-only field the chart injects that is not in the v2 schema)](https://github.com/open-telemetry/opentelemetry-helm-charts/blob/main/charts/opentelemetry-ebpf-instrumentation/templates/_helpers.tpl)
- [opentelemetry-helm-charts issues (where the chart v2 support gap is tracked)](https://github.com/open-telemetry/opentelemetry-helm-charts/issues)

## Community discussion today

The monitored window is a rolling week across the two allowlisted Slack archives that opted into the read-only archive (both OpenTelemetry instrumentation channels); the two Discord workspaces and the public mailing-list and subreddit surfaces were not reviewed in this run, and that gap is noted rather than treated as quiet. Several themes recur from the prior days' windows; the one with a new, source-confirmed boundary is the config migration.

**OBI config v1→v2 migration and the partial-migration boundary (the question above).** A user whose config tripped `fields are outside the supported v1-to-v2 migration contract` had to strip fields, migrate, and re-add them, because no v2 was written and no partial path existed yet; they asked for a middle ground. The maintainer's position is now reflected in the source: the all-or-nothing default is intentional (a silently dropped field would produce a valid-looking but materially different config), the escape hatch is an explicit `--allow-partial` that writes what it can and reports the rest, and commenting out the original fields is ruled out because many v1 fields have no v2 home. The unresolved boundary is operational rather than technical: `--allow-partial` ships with a distinct exit code, but the follow-through (reading the partial report and hand-mapping each listed field) is still manual, and the common case of "which of my fields are the reported ones" has no guidance yet.

**OBI Helm chart v2 gap (a separate boundary).** A user found that the chart still injects a v1-only field into the rendered config, so even a clean v2 migration fails validation through an un-updated chart; maintainers directed that it be filed as its own issue in the `opentelemetry-helm-charts` repo with the chart/OBI versions, the relevant values, the rendered config, and the validation error. Until that lands, the practical workaround is to hand-edit the rendered configmap to drop the v1-only key before validation.

**Node.js sentinel cost attribution (recurring).** The Node.js cost thread continues with a maintainer-confirmed read: client-span parenting comes from the fd-pair map, while the per-callback `async_hooks` sentinel keeps the trace-context map aligned with the active request — and a third consumer is external trace/profile correlation, so gating the sentinel only on manual spans plus log enrichment can silently break that integration. The direction is to separate per-callback context refresh from fd-pair correlation and gate per consumer; an in-flight OBI pull request targets the sentinel's performance and an agent-uninstall path.

**Managed agent harnesses and observability lock-in (recurring).** On the hosted agent harness: when the loop lives in a managed service, the inner model/tool span tree only exists if the harness exports it, and a public beta that exposes no tracing configuration or external trace exporters leaves generic HTTP-client instrumentation with only the outbound crossings to reconstruct.
