#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR/../../.." rev-parse --show-toplevel)"
# shellcheck source=task.env
. "$SCRIPT_DIR/task.env"

: "${EUNOMIA_PATROL_AGENT:?missing Agent selection}"
: "${EUNOMIA_PATROL_MODEL:?missing model selection}"
: "${EUNOMIA_PATROL_REASONING_EFFORT:?missing reasoning effort}"
: "${EUNOMIA_PATROL_TIMEZONE:?missing timezone}"

export TZ="$EUNOMIA_PATROL_TIMEZONE"
STATE_ROOT="${EUNOMIA_PATROL_STATE_ROOT:-/workspaces/.agent-state/eunomia-community-patrol}"
CHECKOUT_ROOT="${EUNOMIA_PATROL_CHECKOUT_ROOT:-/workspaces/patrol-repositories}"
MEMORY_PATH="${EUNOMIA_PATROL_MEMORY_PATH:-$STATE_ROOT/memory.md}"
TRIAGE_SKILL="${EUNOMIA_PATROL_TRIAGE_SKILL:-$REPO_ROOT/.agents/skills/oss-issue-triage/SKILL.md}"
CHANGE_SKILL="${EUNOMIA_PATROL_CHANGE_SKILL:-$REPO_ROOT/.agents/skills/oss-change-workflow/SKILL.md}"
REPORT_DIR="$STATE_ROOT/reports"
LOG_DIR="$STATE_ROOT/logs"
THREAD_FILE="$STATE_ROOT/codex-thread-id"
CLAUDE_SESSION_FILE="$STATE_ROOT/claude-session-id"
PATROL_SKILL="$REPO_ROOT/.agents/skills/eunomia-community-patrol/SKILL.md"
PROMPT_SOURCE="$SCRIPT_DIR/prompt.md"

install -d -m 0700 "$STATE_ROOT" "$CHECKOUT_ROOT" "$REPORT_DIR" "$LOG_DIR"

require_file() {
  if [[ ! -s "$1" ]]; then
    printf 'required file is missing or empty: %s\n' "$1" >&2
    return 1
  fi
}

prepare_skills() {
  if [[ ! -x "$REPO_ROOT/scripts/sync-agent-skills.sh" ]]; then
    printf 'Skill bridge initializer is unavailable\n' >&2
    return 1
  fi
  "$REPO_ROOT/scripts/sync-agent-skills.sh"
}

check_runtime() {
  local command_name
  for command_name in git gh jq flock sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 || {
      printf 'required command is unavailable: %s\n' "$command_name" >&2
      return 1
    }
  done
  require_file "$PATROL_SKILL"
  require_file "$MEMORY_PATH"
  require_file "$TRIAGE_SKILL"
  require_file "$CHANGE_SKILL"
  require_file "$PROMPT_SOURCE"
  command -v "$EUNOMIA_PATROL_AGENT" >/dev/null 2>&1 || {
    printf 'selected Agent is unavailable: %s\n' "$EUNOMIA_PATROL_AGENT" >&2
    return 1
  }
}

render_prompt() {
  cat "$PROMPT_SOURCE"
  printf '\nRuntime paths for this invocation:\n'
  printf -- '- repository root: %s\n' "$REPO_ROOT"
  printf -- '- patrol memory: %s\n' "$MEMORY_PATH"
  printf -- '- repository checkout pool: %s\n' "$CHECKOUT_ROOT"
  printf -- '- oss-issue-triage Skill: %s\n' "$TRIAGE_SKILL"
  printf -- '- oss-change-workflow Skill: %s\n' "$CHANGE_SKILL"
}

probe_model() {
  local probe_file
  probe_file="$(mktemp "$STATE_ROOT/model-probe.XXXXXX")"
  case "$EUNOMIA_PATROL_AGENT" in
    codex)
      codex exec --ephemeral \
        --model "$EUNOMIA_PATROL_MODEL" \
        --config "model_reasoning_effort=\"$EUNOMIA_PATROL_REASONING_EFFORT\"" \
        --dangerously-bypass-approvals-and-sandbox \
        --output-last-message "$probe_file" \
        "Reply with exactly EUNOMIA_PATROL_MODEL_READY."
      ;;
    claude)
      local probe_json
      probe_json="$(mktemp "$STATE_ROOT/model-probe-json.XXXXXX")"
      claude -p \
        --model "$EUNOMIA_PATROL_MODEL" \
        --effort "$EUNOMIA_PATROL_REASONING_EFFORT" \
        --permission-mode bypassPermissions \
        --output-format json \
        --no-session-persistence \
        "Reply with exactly EUNOMIA_PATROL_MODEL_READY." >"$probe_json"
      jq -er '.result // empty' "$probe_json" >"$probe_file"
      rm -f "$probe_json"
      ;;
    *)
      printf 'unsupported Agent adapter: %s\n' "$EUNOMIA_PATROL_AGENT" >&2
      rm -f "$probe_file"
      return 2
      ;;
  esac
  grep -Fxq 'EUNOMIA_PATROL_MODEL_READY' "$probe_file"
  rm -f "$probe_file"
  printf 'model_probe=ready agent=%s model=%s effort=%s\n' \
    "$EUNOMIA_PATROL_AGENT" "$EUNOMIA_PATROL_MODEL" \
    "$EUNOMIA_PATROL_REASONING_EFFORT"
}

run_codex() {
  local prompt_file="$1" report_file="$2" event_file="$3" thread_id=""
  (
    cd "$REPO_ROOT"
  if [[ -s "$THREAD_FILE" ]]; then
    thread_id="$(<"$THREAD_FILE")"
    codex exec resume \
      --model "$EUNOMIA_PATROL_MODEL" \
      --config "model_reasoning_effort=\"$EUNOMIA_PATROL_REASONING_EFFORT\"" \
      --dangerously-bypass-approvals-and-sandbox \
      --json \
      --output-last-message "$report_file" \
      "$thread_id" - <"$prompt_file" >"$event_file" 2>&1
  else
    (
      cd "$REPO_ROOT"
      codex exec \
        --model "$EUNOMIA_PATROL_MODEL" \
        --config "model_reasoning_effort=\"$EUNOMIA_PATROL_REASONING_EFFORT\"" \
        --dangerously-bypass-approvals-and-sandbox \
        --json \
        --output-last-message "$report_file" \
        - <"$prompt_file" >"$event_file" 2>&1
    )
  fi
  )

  thread_id="$(jq -Rr 'fromjson? | select(.type == "thread.started") | .thread_id' "$event_file" | head -n 1)"
  if [[ -n "$thread_id" ]]; then
    printf '%s\n' "$thread_id" >"$THREAD_FILE"
    chmod 0600 "$THREAD_FILE"
  elif [[ ! -s "$THREAD_FILE" ]]; then
    printf 'Codex did not report a persistent thread id\n' >&2
    return 1
  fi
}

run_claude() {
  local prompt_file="$1" report_file="$2" event_file="$3" session_id=""
  local -a args=(
    -p
    --model "$EUNOMIA_PATROL_MODEL"
    --effort "$EUNOMIA_PATROL_REASONING_EFFORT"
    --permission-mode bypassPermissions
    --output-format json
  )
  if [[ -s "$CLAUDE_SESSION_FILE" ]]; then
    args+=(--resume "$(<"$CLAUDE_SESSION_FILE")")
  fi
  (
    cd "$REPO_ROOT"
    claude "${args[@]}" "$(cat "$prompt_file")" >"$event_file"
  )
  jq -er '.result // empty' "$event_file" >"$report_file"
  session_id="$(jq -r '.session_id // empty' "$event_file")"
  if [[ -z "$session_id" ]]; then
    printf 'Claude did not report a persistent session id\n' >&2
    return 1
  fi
  printf '%s\n' "$session_id" >"$CLAUDE_SESSION_FILE"
  chmod 0600 "$CLAUDE_SESSION_FILE"
}

run_patrol() {
  local timestamp prompt_file report_tmp report_file event_file

  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  prompt_file="$(mktemp "$STATE_ROOT/prompt.XXXXXX")"
  report_tmp="$(mktemp "$REPORT_DIR/report.XXXXXX")"
  report_file="$REPORT_DIR/$timestamp.md"
  event_file="$LOG_DIR/$timestamp.ndjson"
  trap "rm -f -- $(printf '%q' "$prompt_file") $(printf '%q' "$report_tmp")" EXIT
  : >"$event_file"
  render_prompt >"$prompt_file"
  chmod 0600 "$prompt_file" "$report_tmp" "$event_file"

  case "$EUNOMIA_PATROL_AGENT" in
    codex) run_codex "$prompt_file" "$report_tmp" "$event_file" ;;
    claude) run_claude "$prompt_file" "$report_tmp" "$event_file" ;;
    *)
      printf 'unsupported Agent adapter: %s\n' "$EUNOMIA_PATROL_AGENT" >&2
      return 2
      ;;
  esac

  require_file "$report_tmp"
  mv "$report_tmp" "$report_file"
  rm -f "$prompt_file"
  trap - EXIT
  chmod 0600 "$report_file" "$event_file"
  ln -sfn "$report_file" "$STATE_ROOT/latest-report.md"
  cat "$report_file"
}

mode="${1:---run}"
if [[ "$mode" == "--run" ]]; then
  exec flock --close --nonblock --conflict-exit-code 75 \
    "$STATE_ROOT/patrol.lock" "$0" --run-locked
fi

case "$mode" in
  --check)
    prepare_skills
    check_runtime
    gh api user --jq '"github_login=" + .login + " github_id=" + (.id | tostring)'
    "$EUNOMIA_PATROL_AGENT" --version
    printf 'memory_bytes=%s memory_sha256=%s\n' \
      "$(wc -c <"$MEMORY_PATH")" \
      "$(sha256sum "$MEMORY_PATH" | cut -d' ' -f1)"
    printf 'runtime_check=ready\n'
    ;;
  --probe)
    prepare_skills
    check_runtime
    probe_model
    ;;
  --run-locked)
    prepare_skills
    check_runtime
    run_patrol
    ;;
  *)
    printf 'usage: %s [--check|--probe|--run]\n' "$0" >&2
    exit 2
    ;;
esac
