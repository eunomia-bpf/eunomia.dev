#!/usr/bin/env python3
"""Verify paper-facing ActPlane artifact summaries.

The default artifact targets do not regenerate policies. They read frozen
inputs/results committed on the artifact branch and recompute the tables that
the paper cites.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RQ1_DIR = ROOT / "docs/eval_runs/rq1-expressiveness/full-607-subagents"
RQ2_DIR = ROOT / "docs/eval_runs/full/deepseek_rq1_20260607T193612Z_v4_pro"
RQ3_MICRO = ROOT / "docs/rq2-performance/results/rq2-micro-2026-06-02T-osdi"
RQ3_MACRO = ROOT / "docs/rq2-performance/results/rq2-macro-2026-06-02T-osdi-v2"
RQ2_JUDGE_DIR = "trajectory_judges_deepseek_deepseek_v4_pro_guardrail_response"


def require(path: Path, hint: str) -> None:
    if not path.exists():
        rel = path.relative_to(ROOT)
        raise SystemExit(
            f"missing {rel}\n"
            f"{hint}\n"
            "If you are on master, switch to the artifact-ready branch or fetch "
            "the raw backup listed in docs/ARTIFACT.md."
        )


def load_json(path: Path) -> Any:
    require(path, "required artifact file is absent")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_rq1() -> int:
    summary = load_json(RQ1_DIR / "summary.json")
    coverage = summary["coverage"]
    retry = summary["retry"]

    expected = {
        "all": (607, 607),
        "per_event": (392, 392),
        "cross_event": (215, 215),
    }
    for key, (total, compiled) in expected.items():
        got = coverage[key]
        if got["total"] != total or got["compiled"] != compiled:
            raise SystemExit(f"RQ1 {key} mismatch: expected {compiled}/{total}, got {got}")

    print("RQ1 expressiveness")
    print(f"- all directives compiled: {coverage['all']['compiled']}/{coverage['all']['total']}")
    print(
        f"- per-event: {coverage['per_event']['compiled']}/{coverage['per_event']['total']}; "
        f"cross-event: {coverage['cross_event']['compiled']}/{coverage['cross_event']['total']}"
    )
    print(f"- retry rate: {100 * retry['retry_rate']:.1f}%")
    print(f"- source: {RQ1_DIR.relative_to(ROOT)}")
    return 0


def load_summarizer_module():
    path = ROOT / "docs/eval_scripts/summarize_agent_sdk_results.py"
    require(path, "RQ2 summarizer is missing")
    spec = importlib.util.spec_from_file_location("actplane_rq2_summarizer", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_rq2() -> int:
    selected = RQ2_DIR / "selected_runner_results.txt"
    require(selected, "RQ2 selected runner list is missing")
    require(RQ2_DIR / "rq2_data_summary.md", "RQ2 data summary is missing")

    summarizer = load_summarizer_module()
    paths = []
    for line in selected.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        paths.append(path if path.is_absolute() else ROOT / path)
    results = []
    for path in summarizer.iter_result_files(paths):
        item = summarizer.load_json(path)
        if item and item.get("system") in summarizer.SYSTEMS:
            results.append(item)
    if not results:
        raise SystemExit("RQ2 selected list did not resolve to runner results")

    results = summarizer.select_latest(results)
    results = [item for item in results if summarizer.is_scorable_result(item)]
    rows, missing = summarizer.load_judged_rows(results, judge_dir_name=RQ2_JUDGE_DIR)
    if missing:
        raise SystemExit(f"RQ2 missing {len(missing)} judge files; first missing: {missing[0]}")

    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[str(row["result"].get("system") or "unknown")].append(row)
    summary = {
        system: summarizer.summarize_system(items)
        for system, items in by_system.items()
        if items
    }

    expected = {
        "prompt-filter": (93, 177, 22, 71, 5, 79, 13, 190),
        "tool-regex": (80, 183, 32, 48, 28, 75, 7, 190),
        "tool-ifc": (87, 189, 38, 49, 27, 75, 1, 190),
        "actplane": (144, 186, 82, 62, 14, 28, 4, 190),
        "actplane-opaque": (108, 175, 34, 74, 1, 66, 15, 190),
    }
    for system, expected_tuple in expected.items():
        item = summary.get(system)
        if item is None:
            raise SystemExit(f"RQ2 missing system {system}")
        got = (
            item["correct"],
            item["scored"],
            item["tp"],
            item["tn"],
            item["fp"],
            item["fn"],
            item["unclear"],
            item["judged"],
        )
        if got != expected_tuple:
            raise SystemExit(f"RQ2 {system} mismatch: expected {expected_tuple}, got {got}")

    print("RQ2 decision compliance")
    for system in summarizer.SYSTEMS:
        item = summary[system]
        display = summarizer.DISPLAY_NAMES.get(system, system)
        print(
            f"- {display}: {item['correct']}/{item['scored']} "
            f"({100 * item['correct'] / item['scored']:.1f}%), "
            f"TP={item['tp']} TN={item['tn']} FP={item['fp']} FN={item['fn']} "
            f"unclear={item['unclear']}"
        )
    print(f"- source: {RQ2_DIR.relative_to(ROOT)}")
    return 0


def verify_rq3() -> int:
    micro = load_json(RQ3_MICRO / "aggregate.json")
    macro = load_json(RQ3_MACRO / "aggregate.json")
    require(RQ3_MICRO / "metadata.json", "RQ3 micro metadata is missing")
    require(RQ3_MACRO / "metadata.json", "RQ3 macro metadata is missing")

    by_micro = {(row["config"], row["op"]): row for row in micro}
    by_macro = {(row["config"], row["workload"]): row for row in macro}
    for key in [("ap-32", "open"), ("ap-32", "write"), ("ap-32", "connect"), ("ap-32", "fork"), ("ap-32", "exec")]:
        if key not in by_micro:
            raise SystemExit(f"RQ3 micro missing {key}")
    for key in [("ap-32", "agent-trace"), ("ap-32", "linux-build")]:
        if key not in by_macro:
            raise SystemExit(f"RQ3 macro missing {key}")

    agent_trace = by_macro[("ap-32", "agent-trace")]
    linux_build = by_macro[("ap-32", "linux-build")]

    print("RQ3 performance")
    print("- ap-32 microbenchmark p50 overheads:")
    for op in ["open", "write", "connect", "fork", "exec"]:
        row = by_micro[("ap-32", op)]
        print(f"  - {op}: {row['overhead_p50_ns_pct']:.2f}%")
    print(
        f"- agent-trace elapsed overhead: {agent_trace['elapsed_overhead_pct']:.1f}% "
        f"(median {agent_trace['median_elapsed_s']}s)"
    )
    print(
        f"- linux-build elapsed overhead: {linux_build['elapsed_overhead_pct']:.1f}% "
        f"(median {linux_build['median_elapsed_s']}s)"
    )
    print(f"- micro source: {RQ3_MICRO.relative_to(ROOT)}")
    print(f"- macro source: {RQ3_MACRO.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rq", choices=["rq1", "rq2", "rq3", "all"])
    args = parser.parse_args(argv)

    if args.rq in {"rq1", "all"}:
        verify_rq1()
    if args.rq in {"rq2", "all"}:
        verify_rq2()
    if args.rq in {"rq3", "all"}:
        verify_rq3()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
