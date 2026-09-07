#!/usr/bin/env bash
# Generic Workspace agent runner: runs one caller-provided primary agent
# command and, only if that attempt fails, one caller-provided fallback
# command. No task logic, models, routes, or time bounds live here; see
# README.md for the full invocation contract.
set -euo pipefail
usage() { printf 'usage: agent-runner.sh --prompt FILE --events FILE --report FILE --primary CMD [--fallback CMD]\n' >&2; exit 2; }
prompt="" events="" report="" primary="" fallback=""
while (($#)); do
  case "$1" in
    --prompt|--events|--report|--primary|--fallback)
      [[ $# -ge 2 ]] || { printf 'agent-runner.sh: %s requires a value\n' "$1" >&2; exit 2; }
      case "$1" in --prompt) prompt="$2";; --events) events="$2";; --report) report="$2";; --primary) primary="$2";; --fallback) fallback="$2";; esac
      shift 2;;
    *) printf 'agent-runner.sh: unknown argument: %s\n' "$1" >&2; usage;;
  esac
done
[[ -n "$prompt" && -n "$events" && -n "$report" && -n "$primary" ]] || usage
: "${RUNNER_STATE_DIR:?RUNNER_STATE_DIR is required}"
session_file="${RUNNER_SESSION_FILE:-$RUNNER_STATE_DIR/session}"
install -d -m 0700 "$RUNNER_STATE_DIR" "$(dirname -- "$events")" "$(dirname -- "$report")"
: >"$report"; : >>"$events"; : >>"${events}.stderr"; chmod 0600 "$report" "$events" "${events}.stderr"
export RUNNER_PROMPT_FILE="$prompt" RUNNER_REPORT_FILE="$report" RUNNER_EVENT_FILE="$events"
attempt_rc=0
fail() { printf 'agent-runner.sh: %s\n' "$1" >&2; exit "$((attempt_rc == 0 ? 1 : attempt_rc))"; }
attempt() {
  local route="$1" command="$2" rc=0 offset new_session tmp
  if [[ -s "$session_file" ]]; then RUNNER_SESSION_ID="$(<"$session_file")"; else RUNNER_SESSION_ID=""; fi
  export RUNNER_SESSION_ID
  offset="$(wc -c <"$events")"
  bash -c "$command" <"$prompt" >>"$events" 2>>"${events}.stderr" || rc=$?
  attempt_rc="$rc"
  printf 'runner_route=%s exit=%s\n' "$route" "$rc"
  [[ "$rc" -eq 0 && -s "$report" ]] || return 1
  if [[ -n "${RUNNER_SESSION_EXTRACT:-}" ]]; then
    new_session="$(tail -c +$((offset + 1)) -- "$events" | jq -Rr "fromjson? | $RUNNER_SESSION_EXTRACT // empty" 2>/dev/null | head -n 1 || true)"
    if [[ -n "$new_session" ]]; then
      tmp="$(mktemp "${session_file}.XXXXXX")"; printf '%s\n' "$new_session" >"$tmp"; chmod 0600 "$tmp"; mv -f -- "$tmp" "$session_file"
    else
      printf 'agent-runner.sh: no session id matched RUNNER_SESSION_EXTRACT\n' >&2
    fi
  fi
}
if ! attempt primary "$primary"; then
  if [[ -z "$fallback" ]]; then fail "primary route failed and no fallback was provided"; fi
  rm -f -- "$session_file"; : >"$report"
  attempt fallback "$fallback" || fail "fallback route failed; see $events and $events.stderr"
fi
