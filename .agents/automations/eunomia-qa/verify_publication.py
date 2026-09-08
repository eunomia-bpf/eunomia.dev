#!/usr/bin/env python3
"""Validate and publish one dated bilingual eBPF Q&A candidate.

The caller chooses a candidate date. This tool validates that pair and its
index links, builds and renders the site, commits only those owned paths while
preserving all unrelated worktree and index state, pushes, and verifies both
articles and both public indexes.

Usage:

    python verify_publication.py --receipt PRIVATE_RECEIPT_PATH --date YYYY-MM-DD

On success it exits 0 and writes a truthful receipt. On any failure it exits
nonzero and still writes a truthful receipt (useful when push succeeded but the
public check later failed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
TEST = REPO / "test"
DOCS = REPO / "docs" / "ebpf-qa"
BUILTIN_NODE = "node"

DESKTOP = {"width": 1440, "height": 900}
MOBILE = {"width": 390, "height": 844}
PUBLIC_BASE = "https://eunomia.dev"

# Privacy hazards that must never appear in a public Q&A page.
SLACK_DISCORD_LINK = re.compile(
    r"https?://([a-z0-9-]+\.)?(slack\.com|discord\.(com|gg|me))/\S+", re.I
)
SLACK_HASHMENTION = re.compile(r"<@[!&]?\d+>", re.I)
DISCORD_MENTION = re.compile(r"<@!?(\d{17,20})>", re.I)
DISCORD_CHANNEL = re.compile(r"<#(\d{17,20})>", re.I)


class Failure(Exception):
    """Raised to abort the run with an aggregate, human-readable reason."""


def die(reason: str) -> None:
    raise Failure(reason)


def run_captured(
    cmd: list[str], cwd: Path, log_path: Path, *, check: bool = True
) -> int:
    """Run a command, capturing full stdout/stderr to log_path (mode 0600)."""
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as out:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
        log_path.chmod(0o600)
        return proc.returncode
    except FileNotFoundError as exc:
        die(f"missing executable for {' '.join(cmd[:1])}: {exc}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise Failure(
            f"git {args[0]} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def first_h1(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def read_frontmatter(text: str) -> dict[str, str]:
    """Parse a minimal leading YAML frontmatter block (key: value)."""
    data: dict[str, str] = {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return data
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, _, value = line.partition(":")
            data[key.strip()] = value.strip().strip("\"'")
    return data


def check_branch() -> None:
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        die(f"not on main (branch={branch})")


def find_qa_pair(date: str) -> tuple[Path, Path]:
    """Locate the dated pair for `date`; enforce exactly one slug pair."""
    pattern = re.compile(
        rf"^{re.escape(date)}-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)\.md$"
    )
    english: dict[str, Path] = {}
    zh: dict[str, Path] = {}
    for entry in DOCS.iterdir():
        if not entry.is_file():
            continue
        m = pattern.match(entry.name)
        if m:
            english[m.group("slug")] = entry
        mz = pattern.match(entry.name.replace(".zh.md", ".md"))
        if mz and entry.name.endswith(".zh.md"):
            zh[mz.group("slug")] = entry

    if not english:
        die(f"no English Q&A file for {date} in docs/ebpf-qa")
    if len(english) > 1:
        die(f"multiple English Q&A pairs for {date}: {sorted(english)}")
    slug = next(iter(english))
    if slug not in zh:
        die(f"missing Chinese counterpart for {slug}")
    return english[slug], zh[slug]


def check_candidate_paths(eng: Path, zhd: Path, slug: str) -> list[Path]:
    index = DOCS / "index.md"
    index_zh = DOCS / "index.zh.md"
    paths = [eng, zhd, index, index_zh]
    for path in paths:
        if not path.is_file():
            die(f"candidate path missing: {path.relative_to(REPO)}")
    expected_links = {
        index: f"/ebpf-qa/{eng.stem}/",
        index_zh: f"/zh/ebpf-qa/{eng.stem}/",
    }
    for idx, link in expected_links.items():
        if link not in idx.read_text(encoding="utf-8"):
            die(f"{idx.name} does not link the new {link} entry")
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", *[str(p.relative_to(REPO)) for p in paths]],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        die(f"git status failed for candidate paths: {proc.stderr.strip()}")
    if not proc.stdout.strip():
        die("candidate paths have no unpublished changes")
    return paths


def scan_privacy_hazards(eng: Path, zhd: Path) -> None:
    for path in (eng, zhd):
        text = path.read_text(encoding="utf-8")
        if SLACK_DISCORD_LINK.search(text):
            die(f"{path.name} contains a Slack/Discord message link")
        for name, rx in (
            ("Slack mention", SLACK_HASHMENTION),
            ("Discord user mention", DISCORD_MENTION),
            ("Discord channel mention", DISCORD_CHANNEL),
        ):
            if rx.search(text):
                die(f"{path.name} contains a {name}")
    # Title must be a genuine technical question, not a placeholder.
    title = first_h1(eng.read_text(encoding="utf-8"))
    if len(title) < 12 or "TODO" in title or "placeholder" in title.lower():
        die(f"English H1 does not look like a real technical question: {title!r}")
    if not any(ch in title for ch in ("?", "why", "how", "can", "does", "should")):
        die(f"English H1 is not phrased as a question: {title!r}")


def npm_present() -> bool:
    return shutil.which(BUILTIN_NODE) is not None and shutil.which("npm") is not None


def run_content_test(logs: Path) -> None:
    log = logs / "content-test.log"
    rc = run_captured(
        ["npm", "--prefix", "app", "run", "test:content"],
        cwd=REPO,
        log_path=log,
    )
    if rc != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        die(f"npm test:content failed (rc={rc}); tail: {' | '.join(tail)}")


def run_build(logs: Path) -> None:
    log = logs / "build.log"
    env = os.environ.copy()
    env["NEXT_PUBLIC_SITE_URL"] = "https://eunomia.dev"
    with log.open("w", encoding="utf-8", errors="replace") as out:
        proc = subprocess.run(
            ["npm", "--prefix", "app", "run", "build"],
            cwd=REPO,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    log.chmod(0o600)
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        die(f"npm build failed (rc={proc.returncode}); tail: {' | '.join(tail)}")


def locate_out_dir() -> Path:
    for candidate in (APP / "out", APP / ".static-builds" / "export"):
        if (candidate / "index.html").is_file():
            return candidate
    die("no static export found (looked for app/out or app/.static-builds/export)")


def built_route_path(out_dir: Path, route: str) -> Path:
    """Map an absolute public route to its file below the static export."""
    return out_dir / route.lstrip("/") / "index.html"


def render_route(
    out_dir: Path, route: str, expected_h1: str, shots: Path, label: str
) -> dict[str, str]:
    """Render one built route with real Chromium; return hashes + checks."""
    html = built_route_path(out_dir, route)
    if not html.is_file():
        die(f"built route missing: {html.relative_to(APP.parent)}")
    # Confirm the generated HTML actually contains the expected H1 text.
    blob = html.read_text(encoding="utf-8", errors="replace")
    if expected_h1 not in blob:
        die(f"generated HTML for {route} missing expected H1 {expected_h1!r}")

    shots_dir = shots / label
    shots_dir.mkdir(parents=True, exist_ok=True)
    script = (
        f"import pw from '{TEST / 'node_modules/playwright/index.mjs'}';\n"
        f"import {{ startStaticServer, stopStaticServer }} from "
        f"'{TEST / 'scripts/lib/static-server.mjs'}';\n"
        f"const out = {str(out_dir)!r};\n"
        f"const route = {route!r};\n"
        f"const expectedH1 = {expected_h1!r};\n"
        f"const shotsDir = {str(shots_dir)!r};\n"
        f"const views = {{desktop: {DESKTOP!r}, mobile: {MOBILE!r}}};\n"
        f"const fs = await import('node:fs');\n"
        f"let results = [];\n"
        f"const server = await startStaticServer({{ rootDir: out, port: 0, host: '127.0.0.1' }});\n"
        f"try {{\n"
        f"  const browser = await pw.chromium.launch();\n"
        f"  for (const [name, vp] of Object.entries(views)) {{\n"
        f"    const page = await browser.newPage({{ viewport: vp }});\n"
        f"    const pageErrors = [];\n"
        f"    page.on('pageerror', (e) => pageErrors.push(String(e)));\n"
        f"    const port = server.server.address().port;\n"
        f"    const res = await page.goto(`http://127.0.0.1:${{port}}${{route}}`,\n"
        f"      {{ waitUntil: 'networkidle', timeout: 30000 }});\n"
        f"    if (res.status() !== 200) throw new Error('HTTP ' + res.status());\n"
        f"    const h1 = await page.$eval('h1', (el) => el.textContent.trim());\n"
        f"    if (!h1 || h1.length < 4) throw new Error('empty h1');\n"
        f"    if (h1 !== expectedH1) throw new Error('h1 mismatch: ' + h1);\n"
        f"    const overflow = await page.evaluate(() =>\n"
        f"      document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);\n"
        f"    if (overflow) throw new Error('horizontal overflow');\n"
        f"    if (pageErrors.length) throw new Error('page errors: ' + pageErrors.join('; '));\n"
        f"    const shot = `${{shotsDir}}/${{name}}.png`;\n"
        f"    await page.screenshot({{ path: shot, fullPage: true }});\n"
        f"    results.push({{ route, name, h1, sha256: null }});\n"
        f"    await page.close();\n"
        f"  }}\n"
        f"  await browser.close();\n"
        f"}} finally {{\n"
        f"  await stopStaticServer(server);\n"
        f"}}\n"
        f"for (const r of results) {{\n"
        f"  r.sha256 = await (async () => {{\n"
        f"    const crypto = await import('node:crypto');\n"
        f"    const buf = fs.readFileSync(`${{shotsDir}}/${{r.name}}.png`);\n"
        f"    return crypto.createHash('sha256').update(buf).digest('hex');\n"
        f"  }})();\n"
        f"}}\n"
        f"console.log(JSON.stringify(results));\n"
    )
    mod = shots_dir / f"render_{label}.mjs"
    mod.write_text(script, encoding="utf-8")
    rc = run_captured(
        ["node", str(mod)],
        cwd=REPO,
        log_path=shots_dir / "render.log",
    )
    if rc != 0:
        logtext = (shots_dir / "render.log").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-15:]
        die(f"render failed for {route}: {' | '.join(logtext)}")
    # Parse the last JSON line.
    logtext = (shots_dir / "render.log").read_text(encoding="utf-8", errors="replace")
    last = [ln for ln in logtext.splitlines() if ln.startswith("[")][-1]
    entries = json.loads(last)
    return {
        "route": route,
        "expected_h1": expected_h1,
        "screenshots": {
            e["name"]: e["sha256"] for e in entries if e.get("sha256")
        },
        "h1_seen": entries[0]["h1"] if entries else "",
    }


def check_public(route: str, expected_text: str) -> None:
    """Bounded per-request retries against the live public URL."""
    url = f"{PUBLIC_BASE}{route}"
    last = ""
    for _ in range(5):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "eunomia-qa-verifier"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    last = f"HTTP {resp.status}"
                else:
                    body = resp.read().decode("utf-8", errors="replace")
                    if expected_text not in body:
                        last = f"expected content not found for {url}"
                    else:
                        return
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001 - aggregate any transport error
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(8)
    die(f"public check failed for {url}: {last}")


def write_receipt(
    receipt: Path,
    payload: dict,
    status: str,
    reason: str | None,
    commit: str | None,
    public_urls: list[str],
    checks: dict,
    screenshots: dict,
) -> None:
    receipt.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "status": status,
        "reason": reason,
        "commit": commit,
        "public_urls": public_urls,
        "checks": checks,
        "screenshots": screenshots,
    }
    receipt.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    receipt.chmod(0o600)


def commit_and_push(paths: list[Path], slug: str, date: str) -> str:
    relative = [str(p.relative_to(REPO)) for p in paths]
    try:
        for path in relative:
            subprocess.run(
                ["git", "add", path],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            )
        subprocess.run(
            [
                "git", "commit", "--only", "-m",
                f"docs(ebpf-qa): {slug} ({date})", "--", *relative,
            ],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise Failure(f"scoped git commit failed: {detail[:500]}") from exc
    new_head = git("rev-parse", "HEAD")
    proc = subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise Failure(
            f"git push origin main failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return new_head


def verify_remote_contains(new_head: str) -> None:
    proc = subprocess.run(
        ["git", "fetch", "origin", "main"],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise Failure(f"git fetch origin main failed: {proc.stderr.strip()}")
    contained = subprocess.run(
        ["git", "merge-base", "--is-ancestor", new_head, "origin/main"],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    if contained.returncode != 0:
        remote_sha = git("rev-parse", "origin/main")
        raise Failure(
            f"publication commit {new_head[:12]} is not contained in remote main "
            f"{remote_sha[:12]}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--before",
        help="deprecated compatibility argument; concurrent HEAD movement is allowed",
    )
    ap.add_argument("--receipt", required=True, help="private receipt.json path")
    ap.add_argument("--date", required=True, help="run date YYYY-MM-DD")
    args = ap.parse_args()

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        print("INVALID DATE", file=sys.stderr)
        return 2

    receipt = Path(args.receipt).resolve()
    logs = receipt.parent / "logs"
    screenshots = receipt.parent / "screenshots"
    logs.mkdir(parents=True, exist_ok=True)

    status = "failed"
    reason: str | None = None
    commit: str | None = None
    public_urls: list[str] = []
    checks: dict[str, str] = {}
    screenshots_map: dict[str, dict[str, str]] = {}

    try:
        if not npm_present():
            die("npm/node not found; coordinator must install app & test dependencies")
        if not (TEST / "node_modules" / "playwright" / "index.mjs").exists():
            die("Playwright module not installed in test/node_modules")

        check_branch()
        checks["branch"] = "ok"

        eng, zhd = find_qa_pair(args.date)
        slug = eng.stem.replace(f"{args.date}-", "", 1)
        paths = check_candidate_paths(eng, zhd, slug)
        checks["candidate_paths"] = "ok"

        checks["index_links"] = "ok"
        scan_privacy_hazards(eng, zhd)
        checks["privacy"] = "ok"

        run_content_test(logs)
        checks["content_test"] = "ok"

        run_build(logs)
        checks["build"] = "ok"

        out_dir = locate_out_dir()
        expected_h1 = first_h1(eng.read_text(encoding="utf-8"))
        zh_h1 = first_h1(zhd.read_text(encoding="utf-8"))
        en_route = f"/ebpf-qa/{eng.stem}/"
        zh_route = f"/zh/ebpf-qa/{eng.stem}/"
        screenshots_map[en_route] = render_route(
            out_dir, en_route, expected_h1, screenshots, "en"
        )
        screenshots_map[zh_route] = render_route(
            out_dir, zh_route, zh_h1, screenshots, "zh"
        )
        checks["render"] = "ok"

        commit = commit_and_push(paths, slug, args.date)
        checks["commit_push"] = "ok"
        verify_remote_contains(commit)
        checks["remote_contains_commit"] = "ok"

        en_index_route = "/ebpf-qa/"
        zh_index_route = "/zh/ebpf-qa/"
        public_urls = [
            f"{PUBLIC_BASE}{en_route}",
            f"{PUBLIC_BASE}{zh_route}",
            f"{PUBLIC_BASE}{en_index_route}",
            f"{PUBLIC_BASE}{zh_index_route}",
        ]
        check_public(en_route, expected_h1)
        check_public(zh_route, zh_h1)
        check_public(en_index_route, f"/{eng.stem}/")
        check_public(zh_index_route, f"/zh/ebpf-qa/{eng.stem}/")
        checks["public"] = "ok"

        status = "published"
        reason = None
    except Failure as exc:
        reason = str(exc)
    except Exception as exc:  # noqa: BLE001 - receipt must record execution failures
        reason = f"unexpected {type(exc).__name__}: {str(exc)[:500]}"

    write_receipt(
        receipt,
        payload={},
        status=status,
        reason=reason,
        commit=commit,
        public_urls=public_urls,
        checks=checks,
        screenshots=screenshots_map,
    )

    if status == "published":
        print(f"PUBLISHED {slug} {commit[:12]}")
        return 0
    print(f"FAILED {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
