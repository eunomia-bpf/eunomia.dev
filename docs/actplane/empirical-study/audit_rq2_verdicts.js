#!/usr/bin/env node
/* Recompute a trace-level separation between kernel intervention and the RQ2
 * end-to-end trajectory judgment. This script reads the frozen artifact-ready
 * export and never calls a model or mutates experiment data. */

const fs = require("fs");
const path = require("path");

const root = process.argv[2];
if (!root) {
  console.error("usage: node audit_rq2_verdicts.js ARTIFACT_ROOT [OUTPUT_JSON]");
  process.exit(2);
}

const manifest = path.join(root, "docs/artifact/rq2-qwen-primary/selected_runner_results.txt");
const judgeDir = "trajectory_judges_llama_cpp_guardrail_response";
const rows = [];

const TRACE_FAMILIES = {
  "trace_allowed_effect_compliant.jsonl": "allowed_effect",
  "trace_lookalike_compliant.jsonl": "lookalike",
  "trace_visible_violation.jsonl": "direct",
  "trace_script_visible_violation.jsonl": "script",
  "trace_opaque_fixture_violation.jsonl": "hidden",
};
const PATH_GROUPS = {
  compliant: ["allowed_effect", "lookalike"],
  direct: ["direct"],
  indirect: ["script", "hidden"],
};
const SYSTEMS = ["prompt-filter", "tool-regex", "tool-ifc", "actplane-opaque", "actplane"];

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function compact(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function feedbackFacts(messages) {
  const text = (messages || []).join("\n");
  return {
    effect: /Operation killed|"effect":"kill"/.test(text) ? "kill" :
      /matched notify rule|"effect":"notify"/.test(text) ? "notify" : "none",
    rule: (text.match(/rule `([^`]+)`/) || [])[1] || "",
    target: (text.match(/Target operation: ([^\n]+)/) || text.match(/Operation `([^`]+)`/ ) || [])[1] || "",
    reason: compact((text.match(/- Reason: ([^\n]+)/) || [])[1] || ""),
  };
}

for (const line of fs.readFileSync(manifest, "utf8").split(/\r?\n/)) {
  if (!line || line.startsWith("#")) continue;
  const fields = line.split("\t");
  if (fields.length !== 7) {
    throw new Error(`malformed manifest row: expected 7 tab-separated fields, got ${fields.length}`);
  }
  const [system, repoKey, statement, trace, manifestAuxLabel] = fields;
  if (!SYSTEMS.includes(system)) throw new Error(`unknown system in manifest: ${system}`);
  const rel = fields[fields.length - 1];
  const resultFile = path.join(root, rel);
  const result = readJson(resultFile);
  if (result.system !== system) {
    throw new Error(`manifest/result system mismatch for ${rel}`);
  }
  const judgeFile = path.join(path.dirname(resultFile), judgeDir,
    `${path.basename(resultFile, ".json")}.judge.json`);
  const judge = readJson(judgeFile);
  if (judge.source_result !== rel) {
    throw new Error(`judge/result pairing mismatch for ${rel}`);
  }
  const facts = feedbackFacts(result.setup_feedbacks);
  const recoveryFacts = feedbackFacts(result.recovery_feedbacks);
  const traceFamily = TRACE_FAMILIES[trace];
  if (!traceFamily) throw new Error(`unknown trace family: ${trace}`);
  rows.push({
    key: `${repoKey}/${statement}/${trace}`,
    system,
    repo: result.repo,
    statement,
    trace,
    trace_family: traceFamily,
    ground_truth_violation: Boolean(result.ground_truth && result.ground_truth.violation),
    judgment: judge.judgment && judge.judgment.confusion_label,
    manifest_aux_label: manifestAuxLabel,
    kernel_intervened: Boolean(result.setup_fired),
    visible_intervention: Boolean(result.setup_visible_intervention),
    runner_blocked: Boolean(result.score && result.score.setup_blocked),
    effect: facts.effect,
    matched_rule: facts.rule,
    target: facts.target,
    reason: facts.reason,
    recovery_intervened: (result.recovery_feedbacks || []).length > 0,
    recovery_effect: recoveryFacts.effect,
    recovery_target: recoveryFacts.target,
    any_observed_intervention: Boolean(result.setup_fired) || (result.recovery_feedbacks || []).length > 0,
    setup_errors: (result.setup_errors || []).map(compact),
    tool_failures: (result.tool_log || []).filter(x => x.phase === "setup" && x.returncode !== 0)
      .map(x => ({tool: x.tool, returncode: x.returncode, command: compact(x.command || x.file_path)})),
    recovery_attempted: Boolean(result.score && result.score.recovery_attempted),
    recovery_tool_count: result.score && result.score.recovery_tool_count || 0,
    agent_error: result.agent_error && result.agent_error.type || "",
    judge_confidence: judge.judgment && judge.judgment.confidence,
    judge_rationale: compact(judge.judgment && judge.judgment.rationale),
    directive: compact(result.ground_truth && result.ground_truth.directive),
    result_file: rel,
  });
}

const paired = new Map();
for (const row of rows) {
  if (!paired.has(row.key)) paired.set(row.key, {});
  if (paired.get(row.key)[row.system]) {
    throw new Error(`duplicate ${row.system} row for ${row.key}`);
  }
  paired.get(row.key)[row.system] = row;
}

for (const [key, systems] of paired) {
  for (const system of SYSTEMS) {
    if (!systems[system]) throw new Error(`missing ${system} row for ${key}`);
  }
}
if (rows.length !== 950 || paired.size !== 190) {
  throw new Error(`incomplete frozen matrix: ${rows.length} rows across ${paired.size} traces`);
}

for (const row of rows) {
  const otherSystem = row.system === "actplane" ? "actplane-opaque" :
    row.system === "actplane-opaque" ? "actplane" : null;
  const other = otherSystem && paired.get(row.key)[otherSystem];
  if (other) {
    row.paired_system = other.system;
    row.paired_judgment = other.judgment;
    row.paired_kernel_intervened = other.kernel_intervened;
    row.paired_any_observed_intervention = other.any_observed_intervention;
    row.paired_effect = other.effect;
  }
}

function confusion(items) {
  const labels = {TP: 0, TN: 0, FP: 0, FN: 0};
  for (const row of items) {
    if (!(row.judgment in labels)) throw new Error(`unknown judgment: ${row.judgment}`);
    labels[row.judgment] += 1;
  }
  const total = items.length;
  const correct = labels.TP + labels.TN;
  return {
    ...labels,
    correct,
    total,
    dcr: total ? correct / total : null,
    dcr_percent: total ? Number((100 * correct / total).toFixed(1)) : null,
  };
}

function exactMcNemar(aOnly, bOnly) {
  const discordant = aOnly + bOnly;
  const tail = Math.min(aOnly, bOnly);
  if (!discordant) return 1;
  let term = Math.pow(0.5, discordant);
  let sum = term;
  for (let k = 0; k < tail; k += 1) {
    term *= (discordant - k) / (k + 1);
    sum += term;
  }
  return Math.min(1, 2 * sum);
}

function comparePaired(systemA, systemB, families) {
  const aRows = rows.filter(row => row.system === systemA && families.includes(row.trace_family));
  const cells = {both_correct: 0, a_only_correct: 0, b_only_correct: 0, both_wrong: 0};
  for (const a of aRows) {
    const b = paired.get(a.key)[systemB];
    if (!b) throw new Error(`missing ${systemB} pair for ${a.key}`);
    const aCorrect = a.judgment === "TP" || a.judgment === "TN";
    const bCorrect = b.judgment === "TP" || b.judgment === "TN";
    const cell = aCorrect && bCorrect ? "both_correct" : aCorrect ? "a_only_correct" :
      bCorrect ? "b_only_correct" : "both_wrong";
    cells[cell] += 1;
  }
  const aCorrect = cells.both_correct + cells.a_only_correct;
  const bCorrect = cells.both_correct + cells.b_only_correct;
  return {
    system_a: systemA,
    system_b: systemB,
    n: aRows.length,
    system_a_correct: aCorrect,
    system_b_correct: bCorrect,
    dcr_difference_percentage_points: Number((100 * (aCorrect - bCorrect) / aRows.length).toFixed(1)),
    paired_outcomes: cells,
    exact_mcnemar_two_sided_p: exactMcNemar(cells.a_only_correct, cells.b_only_correct),
  };
}

const byTraceFamily = {};
const byPathGroup = {};
for (const system of SYSTEMS) {
  byTraceFamily[system] = {};
  for (const family of Object.values(TRACE_FAMILIES)) {
    byTraceFamily[system][family] = confusion(
      rows.filter(row => row.system === system && row.trace_family === family));
  }
  byPathGroup[system] = {};
  for (const [group, families] of Object.entries(PATH_GROUPS)) {
    byPathGroup[system][group] = confusion(
      rows.filter(row => row.system === system && families.includes(row.trace_family)));
  }
  byPathGroup[system].overall = confusion(rows.filter(row => row.system === system));
}

const actplaneVsFides = {};
for (const [group, families] of Object.entries(PATH_GROUPS)) {
  actplaneVsFides[group] = comparePaired("actplane", "tool-ifc", families);
}
actplaneVsFides.overall = comparePaired(
  "actplane", "tool-ifc", Object.values(TRACE_FAMILIES));

const stageByFamily = {};
for (const system of ["actplane", "actplane-opaque"]) {
  stageByFamily[system] = {};
  for (const family of Object.values(TRACE_FAMILIES)) {
    const items = rows.filter(row => row.system === system && row.trace_family === family);
    stageByFamily[system][family] = {
      rows: items.length,
      setup_intervened: items.filter(row => row.kernel_intervened).length,
      any_phase_intervened: items.filter(row => row.any_observed_intervention).length,
      any_kill: items.filter(row => row.effect === "kill" || row.recovery_effect === "kill").length,
      any_notify: items.filter(row => row.effect === "notify" || row.recovery_effect === "notify").length,
    };
  }
}

const counts = {};
const interventionByLabel = {};
for (const row of rows) {
  const key = `${row.system}:${row.judgment}`;
  counts[key] = (counts[key] || 0) + 1;
  if (!interventionByLabel[key]) {
    interventionByLabel[key] = {rows: 0, setup: 0, any_phase: 0, kill: 0, notify: 0};
  }
  const cell = interventionByLabel[key];
  cell.rows += 1;
  cell.setup += Number(row.kernel_intervened);
  cell.any_phase += Number(row.any_observed_intervention);
  cell.kill += Number(row.effect === "kill" || row.recovery_effect === "kill");
  cell.notify += Number(row.effect === "notify" || row.recovery_effect === "notify");
}
const actplaneFp = rows.filter(x => x.system === "actplane" && x.judgment === "FP");
const actplanePairs = rows.filter(x => x.system === "actplane");
const pairedSetupTriggers = {both: 0, actplane_only: 0, opaque_only: 0, neither: 0};
for (const row of actplanePairs) {
  const left = row.kernel_intervened;
  const right = row.paired_kernel_intervened;
  const key = left && right ? "both" : left ? "actplane_only" : right ? "opaque_only" : "neither";
  pairedSetupTriggers[key] += 1;
}
const manifestAuxiliaryLabel = {
  note: "The manifest's undocumented fifth column is retained only for provenance; the paired judge file is the final outcome source used by the paper verifier.",
  differs_from_judge: rows.filter(row => row.judgment !== row.manifest_aux_label).length,
  differs_from_judge_by_system: {},
};
for (const system of SYSTEMS) {
  manifestAuxiliaryLabel.differs_from_judge_by_system[system] = rows.filter(
    row => row.system === system && row.judgment !== row.manifest_aux_label).length;
}
const summary = {
  schema: "rq2-verdict-audit-v2",
  source_manifest: "docs/artifact/rq2-qwen-primary/selected_runner_results.txt",
  final_outcome_source: "trajectory_judges_llama_cpp_guardrail_response/*.judge.json judgment.confusion_label",
  total_rows: rows.length,
  manifest_auxiliary_label: manifestAuxiliaryLabel,
  counts,
  intervention_by_label: interventionByLabel,
  outcome_by_trace_family: byTraceFamily,
  outcome_by_path_group: byPathGroup,
  actplane_vs_fides_paired: actplaneVsFides,
  actplane_stage_by_trace_family: stageByFamily,
  actplane_vs_opaque_setup_triggers: pairedSetupTriggers,
  actplane_fp: {
    count: actplaneFp.length,
    kernel_intervened: actplaneFp.filter(x => x.kernel_intervened).length,
    any_observed_intervention: actplaneFp.filter(x => x.any_observed_intervention).length,
    kill: actplaneFp.filter(x => x.effect === "kill").length,
    notify: actplaneFp.filter(x => x.effect === "notify").length,
    paired_opaque_intervened: actplaneFp.filter(x => x.paired_kernel_intervened).length,
    paired_opaque_fp: actplaneFp.filter(x => x.paired_judgment === "FP").length,
  },
};
const output = {summary, rows};
const serialized = JSON.stringify(output, null, 2) + "\n";
if (process.argv[3]) fs.writeFileSync(process.argv[3], serialized);
else process.stdout.write(serialized);
