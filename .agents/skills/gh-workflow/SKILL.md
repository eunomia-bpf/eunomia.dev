---
name: gh-workflow
description: Use GitHub CLI (`gh`) for local GitHub workflows such as installing or locating gh on Windows, checking authentication, creating or updating pull requests, inspecting PR reviews/comments/checks, monitoring CI, and falling back from GitHub connector failures such as "must be a collaborator" when a pushed branch can be published with gh.
---

# GitHub CLI Workflow

Use `gh` when a task needs local GitHub operations from the checkout: PR
creation, PR updates, checks, reviews, comments, Actions runs, issues, or repo
metadata. Prefer `gh` over hand-written REST calls when the user explicitly asks
for `gh` or when a connector cannot perform a write that local credentials can.

## Find Or Install `gh`

1. Check the current shell:

   ```powershell
   gh --version
   Get-Command gh -All -ErrorAction SilentlyContinue
   ```

2. On Windows, also check the installed absolute path because PATH may be stale:

   ```powershell
   Test-Path "$env:ProgramFiles\GitHub CLI\gh.exe"
   & "$env:ProgramFiles\GitHub CLI\gh.exe" --version
   ```

3. If missing and `winget` is available, install GitHub CLI:

   ```powershell
   winget show --id GitHub.cli -e --source winget
   winget install --id GitHub.cli -e --source winget --accept-package-agreements --accept-source-agreements --silent
   ```

4. After install, use the absolute path if the current shell still cannot find
   `gh` through PATH.

## Authentication

Check auth before writes:

```powershell
& "$env:ProgramFiles\GitHub CLI\gh.exe" auth status
```

If not authenticated, ask the user to complete `gh auth login` in a visible
browser flow. Never print tokens. Avoid `gh auth token` unless piping it
directly into another command without echoing it.

## Publish A PR

1. Inspect state before branch, commit, rebase, push, or PR creation:

   ```powershell
   git status --short --branch
   git branch --show-current
   git log --oneline --decorate --max-count=5
   ```

2. Preserve unrelated dirty files. Stage explicit pathspecs, commit intended
   changes, and push the current branch:

   ```powershell
   git push -u origin "$(git branch --show-current)"
   ```

3. Discover repo and default branch:

   ```powershell
   gh repo view --json nameWithOwner,defaultBranchRef,url
   ```

4. Check for an existing PR before creating a new one:

   ```powershell
   gh pr list --head "$(git branch --show-current)" --state all --json number,url,state,title,isDraft,headRefName,baseRefName
   ```

5. Create a ready PR by default unless the user asks for draft. Use a temp body
   file so Markdown renders correctly:

   ```powershell
   $body = New-TemporaryFile
   Set-Content -Encoding UTF8 $body @'
   ## Summary

   - ...

   ## Validation

   - ...
   '@
   gh pr create --base main --head "$(git branch --show-current)" --title "Add content publishing workflow" --body-file $body
   ```

Use `--draft` only when requested. Do not use `--fill` if commit messages would
produce a noisy or incomplete PR body.

## Inspect PRs, Comments, And CI

Use these after every push:

```powershell
gh pr view <number> --json number,url,state,isDraft,headRefName,baseRefName,mergeStateStatus,statusCheckRollup,reviews,latestReviews,comments
gh api repos/<owner>/<repo>/pulls/<number>/comments
gh pr checks <number> --watch --interval 10
```

For Copilot or inline-review gates, inspect both review submissions and pull
request review comments. Treat connector or web UI summaries as insufficient
when the repository workflow requires explicit comment/thread checks.

## Common Failures

- `gh` not recognized after install: call
  `C:\Program Files\GitHub CLI\gh.exe` directly or start a fresh shell.
- Connector PR creation returns `must be a collaborator`: use `gh pr create`
  with the already-pushed branch and authenticated local account.
- Existing merged PR used the same old head branch: create and push a fresh
  branch name from the current commit, then run `gh pr create` with that head.
- Dirty worktree blocks rebase/switch: inspect the diff; commit intended changes
  or leave unrelated changes untouched and choose a non-destructive path.
