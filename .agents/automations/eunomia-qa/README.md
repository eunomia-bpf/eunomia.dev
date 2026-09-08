# eunomia-qa runner

Repository-owned QA automation consists of the task prompt, archive reader,
and publication validator in this directory. Workspace scheduling and agent
selection are external to the repository. The active agent owns the
publication end to end and may use `verify_publication.py` for deterministic
privacy, content, build, render, scoped-commit, push, and live-page checks.

No repository launcher, model allowlist, fixed model role, mandatory local
model, or preflight choreography is required. Do not reintroduce those through
a wrapper or hidden helper.

## Publication policy

Coverage, freshness, and archive-access gaps never stop a run: the prompt and
the `eunomia-community-radar` skill instruct the agent to continue with
existing readable archive data, past real unresolved questions, permitted
public community pages, and public primary documentation, and to report the
gaps honestly inside the page and the run report. A retry continues the same
candidate: the four files for the date are finished and improved, never
duplicated or discarded. The run still fails truthfully on real validation
failures — anonymization/privacy checks, content tests, build, render,
commit/push, remote-containment, or public-page checks — never silently.

## Usage

Resolve the placeholders in `prompt.md` with a private snapshot path, private
receipt path, the current date, and exactly one invocation of
`archive_reader.py snapshot`. Then give the resolved prompt to the active
Workspace agent.

## Required environment (mounted, unchanged)

| Variable | Purpose |
|---|---|
| `EUNOMIA_QA_ARCHIVE_DSN` | PostgreSQL DSN for the Slack archive (read-only role) |
| `EUNOMIA_QA_STATE_DIR` | (optional) state dir; default `/workspaces/.agent-state/eunomia-qa` |

## Flow

1. The active agent reads the resolved prompt and routed radar skill.
2. It runs the injected snapshot command once and uses whatever permitted
   coverage is available.
3. It resumes an unpublished candidate before creating a new one, regardless
   of calendar rollover.
4. It repairs and validates the content, then runs `verify_publication.py`
   with the candidate date.
5. The validator commits only the candidate pair and indexes, preserves other
   worktree/index state, pushes, and verifies both articles and both indexes.
6. Any failure is reported as incomplete and repaired in the same work chain;
   only a successful live verification is publication success.

## Security invariants

- All intermediate files live in the private temporary directory (0700);
  the runner forces the report/events files to 0600.
- `archive_reader.py` enforces read-only transactions, rejects privileged
  roles, rejects any write grant, bounds the snapshot to 120 KB, and bounds
  stdout to counts/reason codes (never raw message text, identities, the DSN,
  SQL, or exception details).
- The raw snapshot lives only in the private workdir and is removed when the
  run exits; transcripts, logs, prompts, sessions, and raw snapshot text are
  never stored in Git or persistent state.
- The receipt is written only by `verify_publication.py`, never by the agent
  or the entry.
- Private receipt and validation artifacts are mode `0600` when written by the
  validator. Workspace-owned retention is outside this repository's contract.
