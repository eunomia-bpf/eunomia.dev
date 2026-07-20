#!/usr/bin/env python3
"""Check source coverage across per-platform media publishing ledgers."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SOURCES = SCRIPT_DIR / "sources.json"
DEFAULT_PLATFORMS_DIR = SCRIPT_DIR / "platforms"
CONFIRMED_STATUS = "confirmed"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def normalize_pattern(pattern: str) -> str:
    return pattern.replace("\\", "/")


def matches_any(path: str, patterns: list[str]) -> bool:
    path = normalize_pattern(path)
    return any(fnmatch.fnmatch(path, normalize_pattern(pattern)) for pattern in patterns)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return data


def load_platforms(platforms_dir: Path) -> dict[str, dict[str, Any]]:
    platforms: dict[str, dict[str, Any]] = {}
    if not platforms_dir.is_dir():
        raise SystemExit(f"Platform ledger directory does not exist: {platforms_dir}")
    for path in sorted(platforms_dir.glob("*.json")):
        platform = load_json(path)
        platform_id = platform.get("platform_id")
        if not platform_id:
            raise SystemExit(f"Platform JSON missing platform_id: {path}")
        if platform_id in platforms:
            raise SystemExit(f"Duplicate platform_id {platform_id}: {path}")
        if path.stem != platform_id:
            raise SystemExit(f"Platform filename must match platform_id: {path}")
        platforms[platform_id] = platform
    if not platforms:
        raise SystemExit(f"No platform JSON files found in {platforms_dir}")
    return platforms


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_frontmatter(lines: list[str]) -> dict[str, str]:
    if not lines or lines[0].strip() != "---":
        return {}

    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key in {"title", "date", "description"}:
            meta[key] = strip_quotes(value)
    return meta


def first_h1(lines: list[str]) -> str | None:
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return None


def extract_source_metadata(path: Path) -> dict[str, str | None]:
    text = read_text(path)
    lines = text.splitlines()
    frontmatter = parse_frontmatter(lines)
    title = frontmatter.get("title") or first_h1(lines) or path.stem
    return {
        "title": title,
        "date": frontmatter.get("date"),
        "description": frontmatter.get("description"),
    }


def validate_sources_config(config: dict[str, Any]) -> tuple[list[str], set[str], list[str]]:
    errors: list[str] = []
    source_sets = config.get("source_sets")
    if not isinstance(source_sets, list) or not source_sets:
        return ["source_sets must be a non-empty list"], set(), []

    source_set_ids: set[str] = set()
    ordered_source_set_ids: list[str] = []
    for source_set in source_sets:
        source_id = source_set.get("id") if isinstance(source_set, dict) else None
        if not source_id:
            errors.append("every source_set needs an id")
            continue
        if source_id in source_set_ids:
            errors.append(f"duplicate source_set id: {source_id}")
        source_set_ids.add(source_id)
        ordered_source_set_ids.append(source_id)
        globs = source_set.get("globs")
        if not isinstance(globs, list) or not globs:
            errors.append(f"source_set {source_id} needs a non-empty globs list")

    return errors, source_set_ids, ordered_source_set_ids


def validate_platforms(
    platforms: dict[str, dict[str, Any]], source_set_ids: set[str], repo_root: Path
) -> list[str]:
    errors: list[str] = []
    for platform_id, platform in platforms.items():
        for target_id in platform.get("target_source_sets", []):
            if target_id not in source_set_ids:
                errors.append(f"platform {platform_id} references unknown source_set {target_id}")

        entry_ids: set[str] = set()
        for entry in platform.get("published", []):
            entry_id = entry.get("id")
            if not entry_id:
                errors.append(f"platform {platform_id} has a published entry without id")
            elif entry_id in entry_ids:
                errors.append(f"platform {platform_id} has duplicate entry id {entry_id}")
            entry_ids.add(entry_id)

            if entry.get("status") == CONFIRMED_STATUS and not (entry.get("url") or entry.get("evidence_url")):
                errors.append(f"confirmed entry {platform_id}/{entry_id} needs a url or evidence_url")
            equivalent_source_paths = entry.get("equivalent_source_paths", [])
            if not isinstance(equivalent_source_paths, list):
                errors.append(
                    f"entry {platform_id}/{entry_id} equivalent_source_paths must be a list"
                )
                equivalent_source_paths = []
            for source_path in [entry.get("source_path"), *equivalent_source_paths]:
                if source_path and not (repo_root / source_path).is_file():
                    errors.append(
                        f"entry {platform_id}/{entry_id} source path does not exist: {source_path}"
                    )

        confirmed_source_urls = platform.get("confirmed_source_urls", {})
        if not isinstance(confirmed_source_urls, dict):
            errors.append(f"platform {platform_id} confirmed_source_urls must be an object")
            confirmed_source_urls = {}
        for source_path, url in confirmed_source_urls.items():
            if not (repo_root / source_path).is_file():
                errors.append(
                    f"platform {platform_id} compact source path does not exist: {source_path}"
                )
            if not isinstance(url, str) or not url:
                errors.append(
                    f"platform {platform_id} compact source path needs a URL: {source_path}"
                )

    return errors


def scan_sources(config: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source_set in config.get("source_sets", []):
        source_set_id = source_set["id"]
        exclude_globs = source_set.get("exclude_globs", [])
        for pattern in source_set.get("globs", []):
            for path in sorted(repo_root.glob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(repo_root).as_posix()
                if matches_any(rel, exclude_globs):
                    continue
                key = (source_set_id, rel)
                if key in seen:
                    continue
                seen.add(key)
                meta = extract_source_metadata(path)
                sources.append(
                    {
                        "path": rel,
                        "source_set": source_set_id,
                        "language": source_set.get("language"),
                        "title": meta["title"],
                        "date": meta["date"],
                    }
                )
    return sources


def confirmed_source_paths(platform: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for entry in platform.get("published", []):
        if entry.get("status") != CONFIRMED_STATUS:
            continue
        if entry.get("source_path"):
            paths.add(entry["source_path"])
        paths.update(entry.get("equivalent_source_paths", []))
    paths.update(platform.get("confirmed_source_urls", {}).keys())
    return paths


def build_report(
    sources_config: dict[str, Any],
    platforms: dict[str, dict[str, Any]],
    sources: list[dict[str, Any]],
    platform_filter: set[str] | None,
) -> dict[str, Any]:
    source_set_counts = Counter(source["source_set"] for source in sources)
    source_set_ids = [source_set["id"] for source_set in sources_config.get("source_sets", [])]
    platforms_report: dict[str, Any] = {}

    for platform_id, platform in platforms.items():
        if platform_filter and platform_id not in platform_filter:
            continue

        target_source_sets = platform.get("target_source_sets") or source_set_ids
        target_source_set_lookup = set(target_source_sets)
        target_sources = [
            source for source in sources if source["source_set"] in target_source_set_lookup
        ]
        covered_paths = confirmed_source_paths(platform)
        covered_target_paths = {
            source["path"] for source in target_sources if source["path"] in covered_paths
        }
        missing_sources = [
            source for source in target_sources if source["path"] not in covered_paths
        ]
        confirmed_entries = [
            entry for entry in platform.get("published", []) if entry.get("status") == CONFIRMED_STATUS
        ]
        compact_mapping_count = len(platform.get("confirmed_source_urls", {}))
        confirmed_without_source_path = [
            entry for entry in confirmed_entries if not entry.get("source_path")
        ]

        platforms_report[platform_id] = {
            "label": platform.get("label", platform_id),
            "last_checked": platform.get("last_checked"),
            "target_source_sets": target_source_sets,
            "target_source_count": len(target_sources),
            "published_source_count": len(covered_target_paths),
            "not_published_source_count": len(missing_sources),
            "confirmed_entry_count": len(confirmed_entries) + compact_mapping_count,
            "confirmed_compact_mapping_count": compact_mapping_count,
            "confirmed_entries_without_source_path": len(confirmed_without_source_path),
            "third_party_count": len(platform.get("third_party_mentions", [])),
            "reference_count": len(platform.get("references", [])),
            "pending_verification_count": len(platform.get("pending_verification", [])),
            "missing_sources": missing_sources,
        }

    return {
        "ledger_name": sources_config.get("ledger_name"),
        "last_checked": sources_config.get("last_checked"),
        "source_count": len(sources),
        "source_set_counts": {source_set_id: source_set_counts[source_set_id] for source_set_id in source_set_ids},
        "platforms": platforms_report,
    }


def print_text_report(report: dict[str, Any], show_missing: bool, limit: int) -> None:
    print(f"Media ledger: {report.get('ledger_name')}")
    print(f"Last checked: {report.get('last_checked')}")
    print(f"Sources scanned: {report['source_count']}")
    for source_set_id, count in report["source_set_counts"].items():
        print(f"  {source_set_id}: {count}")

    print()
    print("Platform coverage:")
    for platform_id, platform in report["platforms"].items():
        print(
            "- {label} ({platform_id}): {published}/{target} source files mapped, "
            "{missing} not published; {entries} confirmed entries, "
            "{unmapped} confirmed without source_path".format(
                label=platform["label"],
                platform_id=platform_id,
                published=platform["published_source_count"],
                target=platform["target_source_count"],
                missing=platform["not_published_source_count"],
                entries=platform["confirmed_entry_count"],
                unmapped=platform["confirmed_entries_without_source_path"],
            )
        )
        if show_missing:
            missing_sources = platform["missing_sources"]
            shown = missing_sources[:limit]
            for source in shown:
                print(f"    - {source['path']} :: {source['title']}")
            if len(missing_sources) > len(shown):
                print(f"    ... {len(missing_sources) - len(shown)} more")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        default=DEFAULT_SOURCES,
        help=f"Path to source-set JSON. Defaults to {DEFAULT_SOURCES}",
    )
    parser.add_argument(
        "--platforms-dir",
        type=Path,
        default=DEFAULT_PLATFORMS_DIR,
        help=f"Directory containing one JSON file per platform. Defaults to {DEFAULT_PLATFORMS_DIR}",
    )
    parser.add_argument(
        "--platform",
        action="append",
        help="Limit output to one platform id. Can be repeated.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print missing source paths under each platform.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum missing source rows shown per platform in text mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    sources_config = load_json(args.sources.resolve())
    platforms = load_platforms(args.platforms_dir.resolve())

    source_errors, source_set_ids, _ = validate_sources_config(sources_config)
    platform_errors = validate_platforms(platforms, source_set_ids, REPO_ROOT)
    errors = source_errors + platform_errors
    platform_filter = set(args.platform) if args.platform else None
    if platform_filter:
        unknown = sorted(platform_filter - set(platforms))
        if unknown:
            errors.append(f"unknown platform id(s): {', '.join(unknown)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    sources = scan_sources(sources_config, REPO_ROOT)
    report = build_report(sources_config, platforms, sources, platform_filter)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text_report(report, args.show_missing, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
