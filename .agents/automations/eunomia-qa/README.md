# eunomia-qa runner

Single-run automation that drafts one bilingual eBPF Q&A pair from a read-only
Slack archive snapshot, then hands off to `verify_publication.py` for all
mechanical validation and publication. No model review, no fallbacks, no
schedulers.

## Usage

```bash
# Environment preflight (tools, venv, env keys, model allowlist)
run.sh --check

# Full run: clean-worktree gate -> flock -> 12 h timeout -> --locked
run.sh --run
```

`--locked` is the internal entrypoint invoked by `--run` under `flock` and
`timeout`. Do not call it directly.

## Required environment

| Variable | Purpose |
|---|---|
| `OPENCODE_CONFIG` | Path to the opencode config file |
| `EUNOMIA_QA_OPENCODE_CONFIG_CONTENT` | Exported as `OPENCODE_CONFIG_CONTENT` immediately before invoking OpenCode |
| `LITELLM_API_KEY` | API key for the model gateway |
| `EUNOMIA_QA_ARCHIVE_DSN` | PostgreSQL DSN for the Slack archive (read-only role) |
| `EUNOMIA_QA_MODEL` | (optional) model ID; must be in the allowlist |
| `EUNOMIA_QA_DIR` | (optional) directory override; defaults to the script's own dir |

OpenCode is resolved via `command -v opencode` when available, otherwise the
durable path `/workspaces/.agent-state/agent-cli-home/.opencode/bin/opencode`;
the run fails clearly if neither is executable.

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
4. `cmd_locked` creates a private `mktemp -d` workdir (umask 077) holding the
   private snapshot path, and injects one exact shell-quoted
   `archive_reader.py snapshot --output <snapshot>` command into the prompt.
   The runner performs no archive reads of its own.
5. Invokes `opencode run --auto -m <model>` in a new session (`setsid`), with
   XDG dirs redirected into the private workdir. The model itself runs the
   injected snapshot command once, reads the private snapshot, and drafts
   exactly four uncommitted files under `docs/ebpf-qa/` — or drafts nothing
   when the reader fails or the evidence/coverage is insufficient — then stops.
6. After the model exits, the runner fails closed when the private snapshot is
   missing, empty, or above 120 KB, and otherwise always invokes
   `verify_publication.py`, which owns the content tests, static build,
   Chromium render, commit/push, remote-HEAD check, and public page check.
   Missing or insufficient evidence therefore fails closed after the model
   call.
7. `cleanup` (EXIT trap) kills the model process group and removes the private
   workdir on every exit path, including signals.

## Security invariants

- All intermediate files live in a private `mktemp -d` directory (0700).
- `archive_reader.py` enforces read-only transactions, rejects privileged
  roles, rejects any write grant, and bounds stdout to counts/reason codes
  (never raw message text, identities, the DSN, SQL, or exception details).
- Snapshot output is mode 0600 and never exceeds 120 KB.
- No archive read happens before the model call; after the model exits the
  runner only re-checks the private snapshot's existence and size.
- The model never commits, pushes, builds, tests, or runs the verifier.
- The receipt is written by the verifier, not the model.
- Signal traps (`HUP`/`INT`/`TERM`) trigger cleanup with the correct exit
  codes (129/130/143).
