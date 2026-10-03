#!/usr/bin/env python3
"""Audit the frozen OpenAgentSafety policy inventory without outcome claims.

This script deliberately reports syntax compilation and observable policy
structure.  It does not reconstruct per-task outcomes or grade policy meaning.
Official-task availability is a local file-presence count in an operator-
supplied directory; provenance from the frozen benchmark commit is recorded,
never verified.  The summary carries content-relative identifiers only, so
the same logical inputs hash identically across checkouts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ACTION_RE = re.compile(r"^\s+(?:kill|notify)\s+(exec|open|read|write|unlink|connect)\b")
COMPILE_RE = re.compile(r"compiled (\d+) rule\(s\)")
SERVICE_MARKERS = ("gitlab", "owncloud", "plane")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen OpenAgentSafety policies and compile them with ActPlane."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Root containing docs/OpenAgentSafety and docs/artifact from artifact-ready",
    )
    parser.add_argument(
        "--compiler", type=Path, required=True, help="ActPlane CLI binary to use"
    )
    parser.add_argument(
        "--official-task-root",
        type=Path,
        help="Optional flat directory containing <task-id>.md from the frozen OAS commit",
    )
    parser.add_argument(
        "--benchmark-commit",
        help=(
            "40-hex SHA of the frozen OpenAgentSafety benchmark commit to "
            "record. When supplied, it is checked against the expected "
            "submodule commit recorded in the frozen artifact README."
        ),
    )
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=361)
    parser.add_argument("--expected-description", type=int, default=311)
    parser.add_argument("--expected-final", type=int, default=50)
    parser.add_argument("--expected-noop", type=int, default=58)
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_content_digest(root: Path) -> tuple[str, int]:
    """Return ``(sha256, file_count)`` over ``"<sha256>  <relpath>"`` lines.

    Files are visited in sorted relative order, so the digest is stable across
    checkouts: it depends only on file names and bytes, never on where the
    tree is mounted.
    """
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(f"{sha256(path)}  {rel}\n".encode("utf-8"))
        count += 1
    return (digest.hexdigest(), count)


def benchmark_commit_expected(artifact_root: Path) -> str | None:
    """Read the OAS submodule commit recorded in the frozen artifact README.

    The frozen artifact tree records (but does not commit) the expected commit
    of the official OpenAgentSafety benchmark submodule.  Recording it keeps
    the provenance claim tied to a frozen input instead of an operator's
    unverified fetch.
    """
    readme = artifact_root / "docs" / "OpenAgentSafety" / "README.md"
    if not readme.is_file():
        return None
    match = re.search(
        r"Expected submodule commit:\s*`?([0-9a-f]{40})`?",
        readme.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def service_membership(batch_root: Path) -> dict[str, set[str]]:
    membership: dict[str, set[str]] = defaultdict(set)
    for path in sorted(batch_root.glob("*.json")):
        # Match only the manifest basename.  Parent paths such as
        # "actplane-research" must not accidentally classify every task as Plane.
        markers = {marker for marker in SERVICE_MARKERS if marker in path.name.lower()}
        if not markers:
            continue
        payload = load_json(path)
        assert isinstance(payload, dict)
        for case in payload.get("cases", []):
            membership[case["task_id"]].update(markers)
    return membership


def policy_actions(path: Path) -> Counter[str]:
    actions: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = ACTION_RE.match(line)
            if match:
                actions[match.group(1)] += 1
    return actions


def lowered_rule_sum(rows: list[dict[str, object]]) -> int:
    """Sum ``lowered_rules`` over rows that carry a value.

    Failed compiles store ``""`` in the row, so they are skipped here and
    reported through the collected errors instead of crashing the summary.
    """
    return sum(row["lowered_rules"] for row in rows if isinstance(row["lowered_rules"], int))


def compile_policy(compiler: Path, policy: Path, output: Path) -> tuple[int, int | None, str]:
    completed = subprocess.run(
        [str(compiler), "--policy", str(policy), "compile", "--out", str(output), "--force"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    match = COMPILE_RE.search(completed.stdout)
    return completed.returncode, int(match.group(1)) if match else None, completed.stdout.strip()


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    compiler = args.compiler.resolve()
    oas_root = artifact_root / "docs" / "OpenAgentSafety"
    manifest_path = oas_root / "data" / "remaining_attempt0_description_manifest.json"
    ledger_path = artifact_root / "docs" / "artifact" / "rq5_openagentsafety_ledger.json"
    batch_root = oas_root / "data" / "remaining_attempt0_batches"
    final_root = oas_root / "policies" / "actplane"
    description_root = oas_root / "policies" / "remaining_attempts" / "attempt0-description"

    manifest = load_json(manifest_path)
    ledger = load_json(ledger_path)
    assert isinstance(manifest, dict) and isinstance(ledger, dict)
    manifest_cases = manifest["cases"]
    ledger_rows_list = ledger["rows"]
    description_cases = {case["task_id"]: case for case in manifest_cases}
    ledger_rows = {row["task_id"]: row for row in ledger_rows_list}
    services = service_membership(batch_root)

    # Stable, content-relative identifiers for the hashed summary.  These
    # replace machine-specific absolute paths so the same logical inputs
    # hash identically across checkouts.
    oas_content_sha256, oas_file_count = directory_content_digest(oas_root)
    if args.official_task_root:
        task_root_content_sha256, task_root_file_count = directory_content_digest(
            args.official_task_root.resolve()
        )
    else:
        task_root_content_sha256 = None
        task_root_file_count = None
    version_probe = subprocess.run(
        [str(compiler), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    compiler_version = (
        version_probe.stdout.decode("utf-8", "replace").splitlines()[0].strip()
        if version_probe.returncode == 0 and version_probe.stdout
        else None
    )

    # Provenance: the frozen artifact tree records the expected commit of the
    # official OpenAgentSafety benchmark submodule.  The script records it,
    # optionally cross-checks an operator-supplied commit, and never claims
    # the operator's task files were verified against it.
    benchmark_commit_expected_value = benchmark_commit_expected(artifact_root)

    final_policies = sorted(final_root.glob("*.yaml"))
    description_policies = sorted(description_root.glob("*.yaml"))
    policy_specs: list[tuple[str, Path, bool]] = []
    policy_specs.extend(("final50", path, False) for path in final_policies)
    for path in description_policies:
        task_id = path.stem
        policy_specs.append(
            ("description311", path, bool(description_cases[task_id]["is_noop"]))
        )

    errors: list[str] = []
    manifest_case_ids = [case["task_id"] for case in manifest_cases]
    ledger_row_ids = [row["task_id"] for row in ledger_rows_list]
    duplicate_case_ids = sorted({k for k, v in Counter(manifest_case_ids).items() if v > 1})
    duplicate_row_ids = sorted({k for k, v in Counter(ledger_row_ids).items() if v > 1})
    if duplicate_case_ids:
        errors.append(f"description manifest has duplicate task IDs: {duplicate_case_ids}")
    if duplicate_row_ids:
        errors.append(f"ledger has duplicate task IDs: {duplicate_row_ids}")
    expected = {
        "raw ledger rows": (len(ledger_rows_list), args.expected_total),
        "raw description manifest cases": (len(manifest_cases), args.expected_description),
        "all policies": (len(policy_specs), args.expected_total),
        "description policies": (len(description_policies), args.expected_description),
        "final policies": (len(final_policies), args.expected_final),
        "description no-op policies": (
            sum(bool(case["is_noop"]) for case in manifest_cases),
            args.expected_noop,
        ),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            errors.append(f"{label}: expected {wanted}, found {actual}")
    if set(ledger_rows) != {path.stem for _, path, _ in policy_specs}:
        errors.append("ledger task IDs do not exactly match policy filenames")
    label_mismatches = []
    for case in manifest_cases:
        task_id = case["task_id"]
        ledger_row = ledger_rows.get(task_id)
        if ledger_row is None:
            continue
        manifest_noop = bool(case["is_noop"])
        ledger_noop = bool(ledger_row["is_noop"])
        if manifest_noop != ledger_noop:
            label_mismatches.append(
                f"{task_id} (manifest is_noop={manifest_noop}, ledger is_noop={ledger_noop})"
            )
        if (
            "status" in case
            and "status" in ledger_row
            and case["status"] != ledger_row["status"]
        ):
            label_mismatches.append(
                f"{task_id} (manifest status={case['status']!r}, "
                f"ledger status={ledger_row['status']!r})"
            )
    if label_mismatches:
        errors.append(
            "description manifest and ledger no-op/status labels disagree: "
            + ", ".join(label_mismatches)
        )
    benchmark_commit_supplied = args.benchmark_commit
    if (
        benchmark_commit_supplied
        and benchmark_commit_expected_value
        and benchmark_commit_supplied != benchmark_commit_expected_value
    ):
        errors.append(
            f"benchmark commit {benchmark_commit_supplied} does not match the "
            f"frozen expected submodule commit {benchmark_commit_expected_value}"
        )

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="actplane-oas-audit-") as temp_dir:
        output = Path(temp_dir) / "policy.bin"
        for group, policy, is_noop in policy_specs:
            actions = policy_actions(policy)
            task_id = policy.stem
            rc, lowered_rules, compile_output = compile_policy(compiler, policy, output)
            if rc != 0:
                errors.append(f"compile failed for {task_id}: {compile_output}")
            task_path = (
                args.official_task_root / f"{task_id}.md"
                if args.official_task_root
                else None
            )
            rows.append(
                {
                    "task_id": task_id,
                    "policy_group": group,
                    "is_noop": int(is_noop),
                    "service_markers": ",".join(sorted(services.get(task_id, set()))),
                    "official_task_available": int(bool(task_path and task_path.is_file())),
                    "compile_rc": rc,
                    "lowered_rules": lowered_rules if lowered_rules is not None else "",
                    "exec_rules": actions["exec"],
                    "connect_rules": actions["connect"],
                    "open_or_read_rules": actions["open"] + actions["read"],
                    "write_rules": actions["write"],
                    "unlink_rules": actions["unlink"],
                    "policy_sha256": sha256(policy),
                    "official_task_sha256": sha256(task_path)
                    if task_path and task_path.is_file()
                    else "",
                }
            )

    service_rows = [row for row in rows if row["service_markers"]]
    description_rows = [row for row in rows if row["policy_group"] == "description311"]
    nontrivial_description = [row for row in description_rows if not row["is_noop"]]
    official_available = sum(row["official_task_available"] for row in rows)
    summary = {
        "claim_boundary": {
            "does_establish": [
                "frozen policy inventory counts",
                "syntax compilation with the identified ActPlane binary",
                "syntactic policy action coverage",
            ],
            "does_not_establish": [
                "semantic policy correctness",
                "per-task prevention or end-to-end outcome",
                "held-out generalization",
                "an independent baseline comparison",
            ],
        },
        "inputs": {
            # Stable, content-relative identifiers only: no machine-specific
            # absolute paths, so the same logical inputs hash identically
            # across checkouts.
            "artifact_root_identifier": {
                "content_sha256": oas_content_sha256,
                "file_count": oas_file_count,
            },
            "compiler_sha256": sha256(compiler),
            "compiler_version": compiler_version,
            "manifest_sha256": sha256(manifest_path),
            "ledger_sha256": sha256(ledger_path),
            "official_task_root_identifier": {
                "content_sha256": task_root_content_sha256,
                "file_count": task_root_file_count,
            }
            if task_root_content_sha256 is not None
            else None,
        },
        "inventory": {
            "total_policies": len(rows),
            "final_policies": len(final_policies),
            "description_only_policies": len(description_rows),
            "description_only_nontrivial": len(nontrivial_description),
            "description_only_noop": len(description_rows) - len(nontrivial_description),
            "official_task_descriptions_available": official_available,
            "official_task_descriptions_missing": len(rows) - official_available
            if args.official_task_root
            else None,
            # The availability count above is local file presence in an
            # operator-supplied directory.  It is not provenance from the
            # frozen benchmark commit; see "provenance".
            "official_task_source": "local file presence",
        },
        "provenance": {
            "benchmark_commit_expected": benchmark_commit_expected_value,
            "benchmark_commit_supplied": benchmark_commit_supplied,
            "benchmark_commit_consistent": (
                benchmark_commit_supplied == benchmark_commit_expected_value
                if benchmark_commit_supplied and benchmark_commit_expected_value
                else None
            ),
        },
        "compilation": {
            "success": sum(row["compile_rc"] == 0 for row in rows),
            "failure": sum(row["compile_rc"] != 0 for row in rows),
            "lowered_rules_nontrivial_description": lowered_rule_sum(
                nontrivial_description
            ),
            "lowered_rules_noop_description": lowered_rule_sum(
                [row for row in description_rows if row["is_noop"]]
            ),
        },
        "service_manifest_subset": {
            "definition": (
                "task ID appears in a frozen batch manifest whose basename contains "
                "gitlab, owncloud, or plane"
            ),
            "tasks": len(service_rows),
            "with_connect_rule": sum(row["connect_rules"] > 0 for row in service_rows),
            "with_exec_rule": sum(row["exec_rules"] > 0 for row in service_rows),
            "with_neither_connect_nor_exec": sum(
                row["connect_rules"] == 0 and row["exec_rules"] == 0
                for row in service_rows
            ),
            "action_rule_totals": {
                name: sum(int(row[name]) for row in service_rows)
                for name in (
                    "connect_rules",
                    "exec_rules",
                    "open_or_read_rules",
                    "write_rules",
                    "unlink_rules",
                )
            },
        },
        "errors": errors,
    }

    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
