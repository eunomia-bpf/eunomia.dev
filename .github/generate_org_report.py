#!/usr/bin/env python3
"""
Generate GitHub organization activity report.
Collects metrics on stars, repositories, PRs, and commits for a given date range.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_REPORT_ITEM_LIMIT = 100


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


def parse_github_timestamp(value: str):
    """Parse a GitHub timestamp, ignoring the sentinel used for open PRs."""
    if not value or value.startswith("0001-01-01"):
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def is_closed_item(item: Dict[str, Any]) -> bool:
    """Return true when a search result has a usable closedAt timestamp."""
    created = parse_github_timestamp(item.get("createdAt", ""))
    closed = parse_github_timestamp(item.get("closedAt", ""))
    return bool(created and closed and closed >= created)


def report_label_from_path(report_path: Path) -> str:
    """Build a display label for a report archive file."""
    stem = report_path.stem
    try:
        return datetime.strptime(stem, "%Y-%m").strftime("%B %Y")
    except ValueError:
        return stem


def refresh_reports_index(index_file: str, monthly_reports_dir: str):
    """Refresh the website-facing reports index."""
    index_path = Path(index_file)
    monthly_dir = Path(monthly_reports_dir)
    monthly_reports = sorted(monthly_dir.glob("*.md"), reverse=True) if monthly_dir.exists() else []

    lines = [
        "# Activity Reports",
        "",
        "Monthly public reports generated from GitHub organization activity.",
        "",
        "## Monthly Org Reports",
        ""
    ]

    if monthly_reports:
        for report_path in monthly_reports:
            label = report_label_from_path(report_path)
            href = f"/reports/org/monthly/{report_path.stem}/"
            lines.append(f"- [{label}]({href})")
    else:
        lines.append("- No monthly reports have been committed yet.")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_list(f, items: List[str], empty_text: str = "- (none)", limit: int = None):
    """Write a capped markdown list and include the omitted count."""
    if not items:
        f.write(f"{empty_text}\n")
        return

    effective_limit = limit or DEFAULT_REPORT_ITEM_LIMIT
    visible_items = items[:effective_limit]
    f.write("\n".join(visible_items) + "\n")
    omitted = len(items) - len(visible_items)
    if omitted > 0:
        f.write(f"- ... {omitted} more omitted from this website archive\n")


def get_new_stars(org: str, start_z: str, end_z: str) -> tuple[int, List[str]]:
    """Get new stars across all repos in the organization, returning total and per-repo breakdown."""
    total_stars = 0
    repo_stars = []

    try:
        # Get all repos in org
        repos = run_gh_api(f"/orgs/{org}/repos", paginate=True)

        for repo in repos:
            repo_name = repo["name"]
            repo_full_name = repo["full_name"]
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
                    if count > 0:
                        repo_stars.append(f"  - [{repo_full_name}]({repo['html_url']}): **{count}** new stars")
                    total_stars += count
            except Exception as e:
                print(f"Warning: Failed to get stars for {org}/{repo_name}: {e}", file=sys.stderr)
                continue

        return total_stars, repo_stars
    except Exception as e:
        print(f"Warning: Failed to count stars: {e}", file=sys.stderr)
        return 0, []


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
        # Use gh search prs command instead of gh api
        cmd = ["gh", "search", "prs", f"org:{org}", f"created:{start}..{end}", "--json", "title,url,repository,number", "--limit", "1000"]
        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        items = json.loads(result.stdout)
        print(f"Found {len(items)} PRs opened", file=sys.stderr)

        prs = []
        for item in items:
            repo_name = item["repository"]["nameWithOwner"]
            prs.append(f"- [{item['title']}]({item['url']}) — {repo_name} #{item['number']}")

        return prs
    except subprocess.CalledProcessError as e:
        print(f"Error running gh search prs: {' '.join(cmd)}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Warning: Failed to get PRs opened: {e}", file=sys.stderr)
        return []


def get_prs_merged(org: str, start: str, end: str) -> List[str]:
    """Get PRs merged in the date range."""
    try:
        # Use gh search prs command instead of gh api
        cmd = ["gh", "search", "prs", f"org:{org}", "is:merged", f"merged:{start}..{end}", "--json", "title,url,repository,number", "--limit", "1000"]
        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        items = json.loads(result.stdout)
        print(f"Found {len(items)} PRs merged", file=sys.stderr)

        prs = []
        for item in items:
            repo_name = item["repository"]["nameWithOwner"]
            prs.append(f"- [{item['title']}]({item['url']}) — {repo_name} #{item['number']}")

        return prs
    except subprocess.CalledProcessError as e:
        print(f"Error running gh search prs: {' '.join(cmd)}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Warning: Failed to get PRs merged: {e}", file=sys.stderr)
        return []


def get_commits(org: str, start: str, end: str) -> List[Dict[str, str]]:
    """Get commits in the date range."""
    try:
        # Use gh search commits command with proper date range format
        cmd = ["gh", "search", "commits",
               f"org:{org}",
               f"committer-date:{start}..{end}",
               "--json", "commit,sha,url,repository,committer",
               "--limit", "1000"]

        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not result.stdout.strip():
            print("Warning: No commits found (empty response)", file=sys.stderr)
            return []

        items = json.loads(result.stdout)
        print(f"Found {len(items)} commits", file=sys.stderr)

        commits = []
        for item in items:
            message = item["commit"]["message"].split("\n")[0]
            sha_short = item["sha"][:7]
            # The repository structure might not have nameWithOwner, construct it from name and owner
            repo_info = item.get("repository", {})
            if "nameWithOwner" in repo_info:
                repo_name = repo_info["nameWithOwner"]
            elif "name" in repo_info and "owner" in repo_info:
                owner = repo_info["owner"].get("login", "unknown")
                repo_name = f"{owner}/{repo_info['name']}"
            else:
                repo_name = repo_info.get("name", "unknown")

            # Get committer date from the committer object
            committer_date = item.get("committer", {}).get("date", item["commit"].get("committer", {}).get("date", ""))
            commits.append({
                "repo": repo_name,
                "line": f"- [{message}]({item['url']}) — @{sha_short} ({committer_date})"
            })

        return commits
    except subprocess.CalledProcessError as e:
        print(f"Error running gh search commits: {' '.join(cmd)}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Warning: Failed to get commits: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []


def write_commits_by_repo(f, commits: List[Dict[str, str]], per_repo_limit: int = 10):
    """Write commits grouped by repository with a per-repository detail cap."""
    if not commits:
        f.write("- (none)\n")
        return

    commits_by_repo: Dict[str, List[str]] = {}
    for commit in commits:
        repo = commit.get("repo", "unknown")
        commits_by_repo.setdefault(repo, []).append(commit.get("line", ""))

    for repo in sorted(commits_by_repo):
        repo_commits = commits_by_repo[repo]
        f.write(f"\n#### {repo}\n")
        f.write(f"- Total commits: **{len(repo_commits)}**\n\n")
        for line in repo_commits[:per_repo_limit]:
            if line:
                f.write(f"{line}\n")
        omitted = len(repo_commits) - per_repo_limit
        if omitted > 0:
            f.write(f"- ... {omitted} more commits merged into this repository summary\n")


def get_issue_metrics(query: str, start: str, end: str) -> Dict[str, Any]:
    """Get issue metrics for the date range."""
    try:
        # Search for issues created in the date range
        cmd = [
            "gh", "search", "issues",
            query,
            f"created:{start}..{end}",
            "--include-prs",
            "--json", "title,url,repository,number,createdAt,closedAt,commentsCount",
            "--limit", "1000"
        ]
        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        items = json.loads(result.stdout)
        print(f"Found {len(items)} issues and PRs", file=sys.stderr)

        total_items = len(items)
        closed_items = sum(1 for item in items if is_closed_item(item))

        # Calculate average time to first response and time to close
        # Note: This is a simplified version - full version would need to fetch comments
        time_to_close_sum = 0
        closed_count = 0

        for item in items:
            created = parse_github_timestamp(item.get("createdAt", ""))
            closed = parse_github_timestamp(item.get("closedAt", ""))
            if created and closed and closed >= created:
                time_to_close_sum += (closed - created).total_seconds()
                closed_count += 1

        if closed_count > 0:
            avg_time_to_close = time_to_close_sum / closed_count
            days = int(avg_time_to_close // 86400)
            hours = int((avg_time_to_close % 86400) // 3600)
            minutes = int((avg_time_to_close % 3600) // 60)
            seconds = int(avg_time_to_close % 60)
            avg_time_str = f"{days} day{'s' if days != 1 else ''}, {hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            avg_time_str = "N/A"

        return {
            "total_items": total_items,
            "closed_items": closed_items,
            "avg_time_to_close": avg_time_str,
            "issues": items
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running gh search issues: {' '.join(cmd)}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return {"total_items": 0, "closed_items": 0, "avg_time_to_close": "N/A", "issues": []}
    except Exception as e:
        print(f"Warning: Failed to get issue metrics: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"total_items": 0, "closed_items": 0, "avg_time_to_close": "N/A", "issues": []}


def generate_complete_report(
    org: str,
    start: str,
    end: str,
    output_file: str = "issue_metrics.md",
    query: str = None
):
    """Generate complete report including issue metrics and org activity."""
    issue_query = query or f"org:{org}"

    # Get issue metrics
    metrics = get_issue_metrics(issue_query, start, end)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Write issue metrics section
        f.write(f"# Org Activity Report ({start}..{end})\n\n")
        f.write(f"- Search query: `{issue_query}`\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total issues/PRs: **{metrics['total_items']}**\n")
        f.write(f"- Closed issues/PRs: **{metrics['closed_items']}**\n")
        f.write(f"- Average time to close: **{metrics['avg_time_to_close']}**\n\n")

        if metrics['issues']:
            f.write(f"## Issues and Pull Requests\n\n")
            for item in metrics['issues']:
                repo_name = item.get('repository', {}).get('nameWithOwner', 'unknown')
                status = "Closed" if is_closed_item(item) else "Open"
                f.write(f"- [{item['title']}]({item['url']}) — {repo_name} #{item['number']} ({status})\n")
            f.write("\n")

    append_org_activity(org, start, end, output_file)


def append_org_activity(org: str, start: str, end: str, output_file: str = "issue_metrics.md"):
    """Append organization activity addendum to the metrics file."""
    start_z = f"{start}T00:00:00Z"
    end_z = f"{end}T23:59:59Z"

    with open(output_file, "a") as f:
        f.write(f"\n## Org Activity Addendum ({org}, {start}..{end})\n")

        # New stars
        f.write("### New Stars\n")
        total_stars, repo_stars = get_new_stars(org, start_z, end_z)
        f.write(f"- Total new stars: **{total_stars}**\n")
        if repo_stars:
            f.write("\n**By Repository:**\n")
            f.write("\n".join(repo_stars) + "\n")
        else:
            f.write("- (no new stars in individual repos)\n")

        # New repositories
        f.write("\n### New Repositories\n")
        new_repos = get_new_repos(org, start_z, end_z)
        write_markdown_list(f, new_repos)

        # PRs opened
        f.write("\n### Pull Requests Opened\n")
        prs_opened = get_prs_opened(org, start, end)
        f.write(f"- Total: **{len(prs_opened)}**\n\n")
        write_markdown_list(f, prs_opened)

        # PRs merged
        f.write("\n### Pull Requests Merged\n")
        prs_merged = get_prs_merged(org, start, end)
        f.write(f"- Total: **{len(prs_merged)}**\n\n")
        write_markdown_list(f, prs_merged)

        # Commits
        f.write("\n### Commits\n")
        commits = get_commits(org, start, end)
        f.write(f"- Total: **{len(commits)}**\n\n")
        write_commits_by_repo(f, commits)


def main():
    """Main entry point."""
    # Get parameters from environment variables
    org = os.getenv("ORG")
    date_range = os.getenv("RANGE")
    query = os.getenv("QUERY")
    output_file = os.getenv("REPORT_OUTPUT_FILE", "issue_metrics.md")
    reports_index_file = os.getenv("REPORT_INDEX_FILE")
    monthly_reports_dir = os.getenv("MONTHLY_REPORTS_DIR", "docs/reports/org/monthly")

    if not org or not date_range:
        print("Error: ORG and RANGE environment variables must be set", file=sys.stderr)
        sys.exit(1)

    # Parse date range
    try:
        start, end = date_range.split("..")
    except ValueError:
        print(f"Error: Invalid RANGE format '{date_range}'. Expected 'YYYY-MM-DD..YYYY-MM-DD'", file=sys.stderr)
        sys.exit(1)

    print(f"Generating complete org report for {org} from {start} to {end}")
    generate_complete_report(org, start, end, output_file=output_file, query=query)
    print(f"Report generated to {output_file}")

    if reports_index_file:
        refresh_reports_index(reports_index_file, monthly_reports_dir)
        print(f"Reports index refreshed at {reports_index_file}")


if __name__ == "__main__":
    main()
