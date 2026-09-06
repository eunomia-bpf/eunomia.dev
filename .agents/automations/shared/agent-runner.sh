#!/usr/bin/env bash
# Generic Workspace agent runner: runs one primary agent command under a
# timeout and, only on failure, one fallback command. Prompts, models, and
# routes are caller-supplied; this file owns no task logic.
# Env: RUNNER_STATE_DIR (required), RUNNER_TIMEOUT (required positive int,
# seconds per attempt), RUNNER_SESSION_EXTRACT (optional jq filter applied
# to each event JSON line), RUNNER_SESSION_FILE (default $RUNNER_STATE_DIR/
# session). Usage: agent-runner.sh --prompt FILE --events FILE --report
# FILE --primary CMD [--fallback CMD]. The command gets the prompt on
# stdin, stdout appended to the events file, stderr to <events>.stderr,
# RUNNER_SESSION_ID from the persisted session, and RUNNER_PROMPT_FILE /
# RUNNER_REPORT_FILE / RUNNER_EVENT_FILE paths. Success is a zero exit with
# a non-empty report; a fallback run clears the earlier session first, and
# failure exits with the final failed attempt's code.
set -euo pipefail
usage() { printf 'usage: agent-runner.sh --prompt FILE --events FILE --report FILE --primary CMD [--fallback CMD]\n' >&2; exit 2; }
prompt="" events="" report="" primary="" fallback=""
while [[ $# -ge 2 ]]; do
  case "$1" in --prompt) prompt="$2";; --events) events="$2";; --report) report="$2";; --primary) primary="$2";; --fallback) fallback="$2";; *) usage;; esac
  shift 2
done
[[ -n "$prompt" && -n "$events" && -n "$report" && -n "$primary" ]] || usage
: "${RUNNER_STATE_DIR:?RUNNER_STATE_DIR is required}"; : "${RUNNER_TIMEOUT:?RUNNER_TIMEOUT is required}"
[[ "$RUNNER_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { printf 'RUNNER_TIMEOUT must be a positive integer\n' >&2; exit 2; }
[[ -s "$prompt" ]] || { printf 'runner prompt is missing or empty: %s\n' "$prompt" >&2; exit 2; }
session_file="${RUNNER_SESSION_FILE:-$RUNNER_STATE_DIR/session}"
install -d -m 0700 "$RUNNER_STATE_DIR" "$(dirname -- "$events")" "$(dirname -- "$report")"
: >"$report"; : >>"$events"; : >>"${events}.stderr"; chmod 0600 "$report" "$events" "${events}.stderr"
export RUNNER_SESSION_ID="" RUNNER_PROMPT_FILE="$prompt" RUNNER_REPORT_FILE="$report" RUNNER_EVENT_FILE="$events"
[[ -s "$session_file" ]] && export RUNNER_SESSION_ID="$(<"$session_file")"; attempt_rc=0
attempt() {
  local rc=0 offset new_session tmp
  offset="$(wc -c <"$events")"
  timeout --foreground "$RUNNER_TIMEOUT" bash -c "$2" <"$prompt" >>"$events" 2>>"${events}.stderr" || rc=$?
  attempt_rc="$rc"
  printf 'runner_route=%s exit=%s timeout=%s\n' "$1" "$rc" "$RUNNER_TIMEOUT" >&2
  [[ "$rc" -eq 0 ]] && [[ -s "$report" ]] || return 1
  if [[ -n "${RUNNER_SESSION_EXTRACT:-}" ]]; then
    new_session="$(tail -c +$((offset + 1)) -- "$events" | jq -Rr "fromjson? | $RUNNER_SESSION_EXTRACT" 2>/dev/null | head -n 1 || true)"
    [[ -n "$new_session" ]] || printf 'runner: no session id matched RUNNER_SESSION_EXTRACT\n' >&2
    if [[ -n "$new_session" ]]; then tmp="$(mktemp "${session_file}.XXXXXX")"; printf '%s\n' "$new_session" >"$tmp"; chmod 0600 "$tmp"; mv -f -- "$tmp" "$session_file"; fi
  fi
  return 0
}
if ! attempt primary "$primary"; then
  if [[ -z "$fallback" ]]; then printf 'runner: primary route failed and no fallback was provided\n' >&2; exit "$((attempt_rc == 0 ? 1 : attempt_rc))"; fi
  rm -f -- "$session_file"; : >"$report"
  if ! attempt fallback "$fallback"; then printf 'runner: fallback route failed; see %s and %s.stderr\n' "$events" "$events" >&2; exit "$((attempt_rc == 0 ? 1 : attempt_rc))"; fi
fi
