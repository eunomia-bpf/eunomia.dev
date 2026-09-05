# eunomia-qa runner

Single-run automation that drafts one bilingual eBPF Q&A pair from a read-only
Slack archive snapshot, then hands off to `verify_publication.py` for all
mechanical validation and publication. No model review, no fallbacks, no
schedulers.

## Usage

```bash
# Environment preflight (tools, venv, env keys, model allowlist)
run.sh --check

# Read-only archive probe (no writes; exits 1 if any source is inaccessible)
run.sh --probe

# Full run: clean-worktree gate -> flock -> 12 h timeout -> --locked
run.sh --run
```

`--locked` is the internal entrypoint invoked by `--run` under `flock` and
`timeout`. Do not call it directly.

## Required environment

| Variable | Purpose |
|---|---|
| `OPENCODE_CONFIG` | Path to the opencode config file |
| `LITELLM_API_KEY` | API key for the model gateway |
| `EUNOMIA_QA_ARCHIVE_DSN` | PostgreSQL DSN for the Slack archive (read-only role) |
| `EUNOMIA_QA_MODEL` | (optional) model ID; must be in the allowlist |
| `EUNOMIA_QA_DIR` | (optional) directory override; defaults to the script's own dir |

The venv Python lives at `/workspaces/.agent-state/eunomia-qa/venv/bin/python`.
Create it with:

```bash
python3 -m venv /workspaces/.agent-state/eunomia-qa/venv
/workspaces/.agent-state/eunomia-qa/venv/bin/pip install -r requirements.txt
```

## Flow

1. `cmd_run` verifies the repo is on `main` with a clean worktree
   (including untracked files). Refuses otherwise.
2. Acquires an exclusive `flock` on the state-dir lock file; blocks if another
   run is in progress.
3. Wraps `--locked` in `timeout -k 30 43200` (12 h hard cap, 30 s kill grace).
4. `cmd_locked` creates a private `mktemp -d` workdir (umask 077), runs
   `archive_reader.py probe` then `archive_reader.py snapshot`, and enforces a
   120 KB snapshot size limit.
5. Invokes `opencode run --auto -m <model>` in a new session (`setsid`), with
   XDG dirs redirected into the private workdir. The model drafts exactly four
   uncommitted files under `docs/ebpf-qa/` and stops.
6. On model success, runs `verify_publication.py` which owns the content tests,
   static build, Chromium render, commit/push, remote-HEAD check, and public
   page check.
7. `cleanup` (EXIT trap) kills the model process group and removes the private
   workdir on every exit path, including signals.

## Security invariants

- All intermediate files live in a private `mktemp -d` directory (0700).
- `archive_reader.py` enforces read-only transactions, rejects privileged
  roles, rejects any write grant, and bounds stdout to counts/reason codes
  (never raw message text, identities, the DSN, SQL, or exception details).
- Snapshot output is mode 0600 and never exceeds 120 KB.
- The model never commits, pushes, builds, tests, or runs the verifier.
- The receipt is written by the verifier, not the model.
- Signal traps (`HUP`/`INT`/`TERM`) trigger cleanup with the correct exit
  codes (129/130/143).
