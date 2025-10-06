#!/usr/bin/env python3
"""
Generate GitHub organization activity report.
Collects metrics on stars, repositories, PRs, and commits for a given date range.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any


def run_gh_api(endpoint: str, params: Dict[str, str] = None, paginate: bool = False) -> List[Dict[str, Any]]:
    """Run gh api command and return parsed JSON results."""
    cmd = ["gh", "api"]
    if paginate:
        cmd.append("--paginate")

    if params:
        for key, value in params.items():
            cmd.extend(["-f", f"{key}={value}"])

    cmd.append(endpoint)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running gh api command: {' '.join(cmd)}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise

    # Handle paginated results (multiple JSON objects)
    if paginate:
        lines = result.stdout.strip().split('\n')
        all_items = []
        for line in lines:
            if line:
                data = json.loads(line)
                if isinstance(data, list):
                    all_items.extend(data)
                else:
                    all_items.append(data)
        return all_items
    else:
        return json.loads(result.stdout)


def run_gh_api_with_header(endpoint: str, headers: List[str], params: Dict[str, str] = None, paginate: bool = False) -> Dict[str, Any]:
    """Run gh api command with custom headers."""
    cmd = ["gh", "api"]

    for header in headers:
        cmd.extend(["-H", header])

    if paginate:
        cmd.append("--paginate")

    if params:
        for key, value in params.items():
            cmd.extend(["-f", f"{key}={value}"])

    cmd.append(endpoint)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running gh api command: {' '.join(cmd)}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise

    return json.loads(result.stdout)


def count_new_stars(org: str, start_z: str, end_z: str) -> int:
    """Count new stars across all repos in the organization."""
    total_stars = 0

    try:
        # Get all repos in org
        repos = run_gh_api(f"/orgs/{org}/repos", paginate=True)
        repo_names = [repo["name"] for repo in repos]

        for repo_name in repo_names:
            try:
                # Get stargazers with timestamps
                stargazers = run_gh_api_with_header(
                    f"/repos/{org}/{repo_name}/stargazers",
                    ["Accept: application/vnd.github.star+json"],
                    paginate=True
                )

                # Count stars in date range
                if isinstance(stargazers, list):
                    count = sum(1 for s in stargazers
                               if s.get("starred_at", "") >= start_z and s.get("starred_at", "") <= end_z)
                    total_stars += count
            except Exception as e:
                print(f"Warning: Failed to get stars for {org}/{repo_name}: {e}", file=sys.stderr)
                continue

        return total_stars
    except Exception as e:
        print(f"Warning: Failed to count stars: {e}", file=sys.stderr)
        return 0


def get_new_repos(org: str, start_z: str, end_z: str) -> List[str]:
    """Get repositories created in the date range."""
    try:
        repos = run_gh_api(f"/orgs/{org}/repos", paginate=True)

        new_repos = []
        for repo in repos:
            if repo.get("created_at", "") >= start_z and repo.get("created_at", "") <= end_z:
                new_repos.append(f"- [{repo['full_name']}]({repo['html_url']}) — created {repo['created_at']}")

        return new_repos
    except Exception as e:
        print(f"Warning: Failed to get new repos: {e}", file=sys.stderr)
        return []


def get_prs_opened(org: str, start: str, end: str) -> List[str]:
    """Get PRs opened in the date range."""
    try:
        result = run_gh_api_with_header(
            "/search/issues",
            ["Accept: application/vnd.github+json"],
            params={
                "q": f"org:{org} is:pr created:{start}..{end}",
                "per_page": "100"
            },
            paginate=False  # Search API pagination works differently
        )

        items = result.get("items", []) if isinstance(result, dict) else []

        prs = []
        for item in items:
            repo_name = item["repository_url"].split("/repos/")[1] if "/repos/" in item["repository_url"] else "unknown"
            prs.append(f"- [{item['title']}]({item['html_url']}) — {repo_name} #{item['number']}")

        return prs
    except Exception as e:
        print(f"Warning: Failed to get PRs opened: {e}", file=sys.stderr)
        return []


def get_prs_merged(org: str, start: str, end: str) -> List[str]:
    """Get PRs merged in the date range."""
    try:
        result = run_gh_api_with_header(
            "/search/issues",
            ["Accept: application/vnd.github+json"],
            params={
                "q": f"org:{org} is:pr is:merged merged:{start}..{end}",
                "per_page": "100"
            },
            paginate=False  # Search API pagination works differently
        )

        items = result.get("items", []) if isinstance(result, dict) else []

        prs = []
        for item in items:
            repo_name = item["repository_url"].split("/repos/")[1] if "/repos/" in item["repository_url"] else "unknown"
            prs.append(f"- [{item['title']}]({item['html_url']}) — {repo_name} #{item['number']}")

        return prs
    except Exception as e:
        print(f"Warning: Failed to get PRs merged: {e}", file=sys.stderr)
        return []


def get_commits(org: str, start: str, end: str) -> List[str]:
    """Get commits in the date range."""
    try:
        result = run_gh_api_with_header(
            "/search/commits",
            ["Accept: application/vnd.github+json"],
            params={
                "q": f"org:{org} committer-date:{start}..{end}",
                "per_page": "100"
            },
            paginate=False  # Search API pagination works differently
        )

        items = result.get("items", []) if isinstance(result, dict) else []

        commits = []
        for item in items:
            message = item["commit"]["message"].split("\n")[0]
            sha_short = item["sha"][:7]
            commits.append(
                f"- [{message}]({item['html_url']}) — {item['repository']['full_name']} "
                f"@{sha_short} ({item['commit']['committer']['date']})"
            )

        return commits
    except Exception as e:
        print(f"Warning: Failed to get commits: {e}", file=sys.stderr)
        return []


def append_org_activity(org: str, start: str, end: str, output_file: str = "issue_metrics.md"):
    """Append organization activity addendum to the metrics file."""
    start_z = f"{start}T00:00:00Z"
    end_z = f"{end}T23:59:59Z"

    with open(output_file, "a") as f:
        f.write(f"\n## Org Activity Addendum ({org}, {start}..{end})\n")

        # New stars
        f.write("### New Stars (count)\n")
        total_stars = count_new_stars(org, start_z, end_z)
        f.write(f"- Total new stars: **{total_stars}**\n")

        # New repositories
        f.write("\n### New Repositories\n")
        new_repos = get_new_repos(org, start_z, end_z)
        if new_repos:
            f.write("\n".join(new_repos) + "\n")
        else:
            f.write("- (none)\n")

        # PRs opened
        f.write("\n### Pull Requests Opened\n")
        prs_opened = get_prs_opened(org, start, end)
        if prs_opened:
            f.write("\n".join(prs_opened) + "\n")
        else:
            f.write("- (none)\n")

        # PRs merged
        f.write("\n### Pull Requests Merged\n")
        prs_merged = get_prs_merged(org, start, end)
        if prs_merged:
            f.write("\n".join(prs_merged) + "\n")
        else:
            f.write("- (none)\n")

        # Commits
        f.write("\n### Commits\n")
        commits = get_commits(org, start, end)
        if commits:
            f.write("\n".join(commits) + "\n")
        else:
            f.write("- (none)\n")


def main():
    """Main entry point."""
    # Get parameters from environment variables
    org = os.getenv("ORG")
    date_range = os.getenv("RANGE")

    if not org or not date_range:
        print("Error: ORG and RANGE environment variables must be set", file=sys.stderr)
        sys.exit(1)

    # Parse date range
    try:
        start, end = date_range.split("..")
    except ValueError:
        print(f"Error: Invalid RANGE format '{date_range}'. Expected 'YYYY-MM-DD..YYYY-MM-DD'", file=sys.stderr)
        sys.exit(1)

    print(f"Generating org activity report for {org} from {start} to {end}")
    append_org_activity(org, start, end)
    print("Report appended to issue_metrics.md")


if __name__ == "__main__":
    main()
