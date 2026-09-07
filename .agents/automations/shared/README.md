# Shared Workspace Agent Runner

`agent-runner.sh` is the single generic runner for scheduled Workspace agent
tasks. It runs one caller-provided primary agent command and, only if that
attempt fails, one caller-provided fallback agent command. It owns no task
logic: prompts, models, routes, session semantics, and any time bounds belong
to the caller, not to this file.

## Invocation

```
RUNNER_STATE_DIR=/path/to/state \
  agent-runner.sh --prompt PROMPT_FILE --events EVENTS_FILE --report REPORT_FILE \
      --primary 'AGENT COMMAND' [--fallback 'AGENT COMMAND']
```

- `--prompt FILE`: fed to every command on stdin; commands may also read
  `$RUNNER_PROMPT_FILE` directly.
- `--events FILE`: the command's stdout appends here; stderr appends to
  `<events>.stderr`. The runner's own `runner_route=<route> exit=<code>` lines
  are also written here by callers that redirect runner output to the file.
- `--report FILE`: each command must write its final report here. An attempt
  succeeds only when the command exits zero **and** the report is non-empty.
- `--primary CMD` and `--fallback CMD` are complete agent invocations, each
  run with `bash -c` in the caller's environment.
- Argument handling is strict: every flag requires a value, unknown or
  trailing flags (including `--help`/positional extras) exit 2 with usage, and
  `--prompt`, `--events`, `--report`, `--primary` are all required.

## Environment

| Variable | Meaning |
| --- | --- |
| `RUNNER_STATE_DIR` (required) | State directory, created 0700 if missing. |
| `RUNNER_SESSION_FILE` | Persisted session id, default `<state>/session`. |
| `RUNNER_SESSION_EXTRACT` | Optional `jq` filter applied to every new event line produced by a successful attempt; the first match is atomically persisted to the session file. |

Exported to every command: `RUNNER_SESSION_ID` (re-read from the persisted
session file at the start of each attempt, empty when absent),
`RUNNER_PROMPT_FILE`, `RUNNER_REPORT_FILE`, `RUNNER_EVENT_FILE`.

## Failure, fallback, and sessions

- Primary failure with no `--fallback`: exits with the failed attempt's exit
  code, or 1 when the attempt exited zero but produced an empty report.
- With a `--fallback`: the persisted session file is deleted, the report is
  cleared, and `RUNNER_SESSION_ID` is re-derived from the now-absent file
  (empty) before the fallback runs, so the fallback cannot inherit the
  primary's session. If the fallback attempt succeeds and
  `RUNNER_SESSION_EXTRACT` matches events, the new session id is persisted.
- Both routes failing exits with the fallback attempt's exit code (1 when
  that attempt exited zero with an empty report). Overall failure is always
  nonzero.

## Known consumers

`eunomia-community-patrol/run.sh` uses the Codex thread id as its session
(`$STATE/codex-thread-id`, extracted from `codex exec --json` events) with a
local OpenCode route as fallback. Other automation tasks (for example QA) may
adopt the same contract without changing their business prompts.
