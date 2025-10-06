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


def get_commits(org: str, start: str, end: str) -> List[str]:
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
            commits.append(
                f"- [{message}]({item['url']}) — {repo_name} "
                f"@{sha_short} ({committer_date})"
            )

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


def get_issue_metrics(org: str, start: str, end: str) -> Dict[str, Any]:
    """Get issue metrics for the date range."""
    try:
        # Search for issues created in the date range
        cmd = ["gh", "search", "issues", f"org:{org}", f"created:{start}..{end}",
               "--json", "title,url,repository,number,createdAt,closedAt,commentsCount",
               "--limit", "1000"]
        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        items = json.loads(result.stdout)
        print(f"Found {len(items)} issues", file=sys.stderr)

        total_items = len(items)
        closed_items = sum(1 for item in items if item.get("closedAt"))

        # Calculate average time to first response and time to close
        # Note: This is a simplified version - full version would need to fetch comments
        from datetime import datetime

        time_to_close_sum = 0
        closed_count = 0

        for item in items:
            if item.get("closedAt"):
                created = datetime.fromisoformat(item["createdAt"].replace("Z", "+00:00"))
                closed = datetime.fromisoformat(item["closedAt"].replace("Z", "+00:00"))
                time_to_close_sum += (closed - created).total_seconds()
                closed_count += 1

        avg_time_to_close = time_to_close_sum / closed_count if closed_count > 0 else 0

        # Format average time
        days = int(avg_time_to_close // 86400)
        hours = int((avg_time_to_close % 86400) // 3600)
        minutes = int((avg_time_to_close % 3600) // 60)
        avg_time_str = f"{days} day{'s' if days != 1 else ''}, {hours:02d}:{minutes:02d}:{int(avg_time_to_close % 60):02d}"

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


def generate_complete_report(org: str, start: str, end: str, output_file: str = "issue_metrics.md"):
    """Generate complete report including issue metrics and org activity."""
    start_z = f"{start}T00:00:00Z"
    end_z = f"{end}T23:59:59Z"

    # Get issue metrics
    metrics = get_issue_metrics(org, start, end)

    with open(output_file, "w") as f:
        # Write issue metrics section
        f.write(f"# Issue Metrics ({start}..{end})\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total issues/PRs: **{metrics['total_items']}**\n")
        f.write(f"- Closed issues/PRs: **{metrics['closed_items']}**\n")
        f.write(f"- Average time to close: **{metrics['avg_time_to_close']}**\n\n")

        if metrics['issues']:
            f.write(f"## Issues and Pull Requests\n\n")
            for item in metrics['issues']:
                repo_name = item.get('repository', {}).get('nameWithOwner', 'unknown')
                status = "✅ Closed" if item.get("closedAt") else "🔄 Open"
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

    print(f"Generating complete org report for {org} from {start} to {end}")
    generate_complete_report(org, start, end)
    print("Report generated to issue_metrics.md")


if __name__ == "__main__":
    main()
