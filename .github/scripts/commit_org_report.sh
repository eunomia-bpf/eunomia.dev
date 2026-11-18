#!/usr/bin/env bash
# Commit a generated org report and push it, rebasing onto whatever landed on
# the branch while the report was being generated.
#
# The scheduled job takes minutes to gather stats; any other push to `main` in
# that window makes a plain `git push` non-fast-forward and the whole report
# run fails (run 43853713419, 2026-11-16: "! [rejected] HEAD -> main (fetch
# first)"). After a rejection the generator runs again on the new base, so the
# retry never re-applies a stale diff and a generated index file is rebuilt
# rather than conflicted.
#
# Usage: commit_org_report.sh "<commit message>" <generated file> [<file>...]
set -euo pipefail

branch="${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"
message="$1"
shift

for attempt in 1 2 3 4 5; do
  git add "$@"
  if git diff --staged --quiet; then
    echo "Report unchanged on attempt ${attempt}; nothing to commit."
    exit 0
  fi
  git commit -m "$message"
  if git push origin "HEAD:${branch}"; then
    exit 0
  fi
  echo "Push to ${branch} rejected on attempt ${attempt}; regenerating on top of origin/${branch}."
  git fetch origin "${branch}"
  git reset --hard FETCH_HEAD
  python3 .github/generate_org_report.py
  sleep $((attempt * 5))
done

echo "::error::could not push the report to ${branch} after 5 attempts"
exit 1