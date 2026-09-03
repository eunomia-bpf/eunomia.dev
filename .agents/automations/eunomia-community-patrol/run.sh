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
OPENAI_THREAD_FILE="$STATE_ROOT/codex-thread-id"
LOCAL_THREAD_FILE="$STATE_ROOT/codex-thread-id-${EUNOMIA_PATROL_LOCAL_PROVIDER:-local}"
CLAUDE_SESSION_FILE="$STATE_ROOT/claude-session-id"
PATROL_SKILL="$REPO_ROOT/.agents/skills/eunomia-community-patrol/SKILL.md"
PROMPT_SOURCE="$SCRIPT_DIR/prompt.md"
ACTIVE_CODEX_MODEL="$EUNOMIA_PATROL_MODEL"
ACTIVE_CODEX_ROUTE="openai-fallback"
ACTIVE_CODEX_THREAD_FILE="$OPENAI_THREAD_FILE"
ACTIVE_CODEX_CONFIG_ARGS=()

install -d -m 0700 "$STATE_ROOT" "$CHECKOUT_ROOT" "$REPORT_DIR" "$LOG_DIR"

require_file() {
  if [[ ! -s "$1" ]]; then
    printf 'required file is missing or empty: %s\n' "$1" >&2
    return 1
  fi
}
write_state_id_atomic() {
  local target="$1" value="$2" temporary
  temporary="$(mktemp "${target}.XXXXXX")"
  printf '%s\n' "$value" >"$temporary"
  chmod 0600 "$temporary"
  mv -f -- "$temporary" "$target"
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
  for command_name in git gh jq flock sha256sum timeout; do
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
  if [[ "$EUNOMIA_PATROL_AGENT" == "codex" ]]; then
    printf -- '- selected model route: %s / %s\n' \
      "$ACTIVE_CODEX_ROUTE" "$ACTIVE_CODEX_MODEL"
  fi
}

configure_local_codex_route() {
  local model="$1"
  ACTIVE_CODEX_MODEL="$model"
  ACTIVE_CODEX_ROUTE="local:${EUNOMIA_PATROL_LOCAL_PROVIDER}"
  ACTIVE_CODEX_THREAD_FILE="$LOCAL_THREAD_FILE"
  ACTIVE_CODEX_CONFIG_ARGS=(
    --config "model_provider=\"${EUNOMIA_PATROL_LOCAL_PROVIDER}\""
    --config "model_providers.${EUNOMIA_PATROL_LOCAL_PROVIDER}.name=\"Spark LiteLLM gateway\""
    --config "model_providers.${EUNOMIA_PATROL_LOCAL_PROVIDER}.base_url=\"${EUNOMIA_PATROL_LOCAL_BASE_URL}\""
    --config "model_providers.${EUNOMIA_PATROL_LOCAL_PROVIDER}.env_key=\"LITELLM_API_KEY\""
    --config "model_providers.${EUNOMIA_PATROL_LOCAL_PROVIDER}.wire_api=\"responses\""
  )
}

configure_openai_codex_route() {
  ACTIVE_CODEX_MODEL="$EUNOMIA_PATROL_MODEL"
  ACTIVE_CODEX_ROUTE="openai-fallback"
  ACTIVE_CODEX_THREAD_FILE="$OPENAI_THREAD_FILE"
  ACTIVE_CODEX_CONFIG_ARGS=()
}

probe_active_codex_route() {
  local probe_file probe_log
  probe_file="$(mktemp "$STATE_ROOT/model-probe.XXXXXX")"
  probe_log="$(mktemp "$STATE_ROOT/model-probe-log.XXXXXX")"
  if ! timeout --foreground 180 codex exec --ephemeral \
      --model "$ACTIVE_CODEX_MODEL" \
      "${ACTIVE_CODEX_CONFIG_ARGS[@]}" \
      --config "model_reasoning_effort=\"$EUNOMIA_PATROL_REASONING_EFFORT\"" \
      --dangerously-bypass-approvals-and-sandbox \
      --output-last-message "$probe_file" \
      "Reply with exactly EUNOMIA_PATROL_MODEL_READY." \
      >"$probe_log" 2>&1; then
    rm -f "$probe_file" "$probe_log"
    return 1
  fi
  if ! grep -Fxq 'EUNOMIA_PATROL_MODEL_READY' "$probe_file"; then
    rm -f "$probe_file" "$probe_log"
    return 1
  fi
  rm -f "$probe_file" "$probe_log"
  printf 'model_probe=ready agent=codex route=%s model=%s effort=%s\n' \
    "$ACTIVE_CODEX_ROUTE" "$ACTIVE_CODEX_MODEL" \
    "$EUNOMIA_PATROL_REASONING_EFFORT"
}

select_codex_route() {
  local model
  if [[ -n "${LITELLM_API_KEY:-}" ]]; then
    for model in ${EUNOMIA_PATROL_LOCAL_MODELS:-}; do
      configure_local_codex_route "$model"
      if probe_active_codex_route; then
        return 0
      fi
      printf 'model_probe=unavailable agent=codex route=%s model=%s\n' \
        "$ACTIVE_CODEX_ROUTE" "$ACTIVE_CODEX_MODEL" >&2
    done
  else
    printf 'model_probe=skipped agent=codex route=local reason=LITELLM_API_KEY_missing\n' >&2
  fi

  configure_openai_codex_route
  probe_active_codex_route
}

probe_model() {
  case "$EUNOMIA_PATROL_AGENT" in
    codex)
      select_codex_route
      ;;
    claude)
      local probe_file probe_json
      probe_file="$(mktemp "$STATE_ROOT/model-probe.XXXXXX")"
      probe_json="$(mktemp "$STATE_ROOT/model-probe-json.XXXXXX")"
      if ! claude -p \
          --model "$EUNOMIA_PATROL_MODEL" \
          --effort "$EUNOMIA_PATROL_REASONING_EFFORT" \
          --permission-mode bypassPermissions \
          --output-format json \
          --no-session-persistence \
          "Reply with exactly EUNOMIA_PATROL_MODEL_READY." >"$probe_json"; then
        rm -f "$probe_file" "$probe_json"
        return 1
      fi
      if ! jq -er '.result // empty' "$probe_json" >"$probe_file"; then
        rm -f "$probe_file" "$probe_json"
        return 1
      fi
      rm -f "$probe_json"
      if ! grep -Fxq 'EUNOMIA_PATROL_MODEL_READY' "$probe_file"; then
        rm -f "$probe_file"
        return 1
      fi
      rm -f "$probe_file"
      printf 'model_probe=ready agent=claude model=%s effort=%s\n' \
        "$EUNOMIA_PATROL_MODEL" "$EUNOMIA_PATROL_REASONING_EFFORT"
      ;;
    *)
      printf 'unsupported Agent adapter: %s\n' "$EUNOMIA_PATROL_AGENT" >&2
      return 2
      ;;
  esac
}

run_codex() {
  local prompt_file="$1" report_file="$2" event_file="$3" thread_id=""
  if [[ -e "$ACTIVE_CODEX_THREAD_FILE" && ! -s "$ACTIVE_CODEX_THREAD_FILE" ]]; then
    printf 'Codex continuity file exists but is empty: %s\n' "$ACTIVE_CODEX_THREAD_FILE" >&2
    return 1
  fi
  if [[ ! -e "$ACTIVE_CODEX_THREAD_FILE" ]]; then
    : >"$ACTIVE_CODEX_THREAD_FILE"
    chmod 0600 "$ACTIVE_CODEX_THREAD_FILE"
  fi

  (
    cd "$REPO_ROOT"
  if [[ -s "$ACTIVE_CODEX_THREAD_FILE" ]]; then
    thread_id="$(<"$ACTIVE_CODEX_THREAD_FILE")"
    codex exec resume \
      --model "$ACTIVE_CODEX_MODEL" \
      "${ACTIVE_CODEX_CONFIG_ARGS[@]}" \
      --config "model_reasoning_effort=\"$EUNOMIA_PATROL_REASONING_EFFORT\"" \
      --dangerously-bypass-approvals-and-sandbox \
      --json \
      --output-last-message "$report_file" \
      "$thread_id" - <"$prompt_file" >"$event_file" 2>&1
  else
    (
      cd "$REPO_ROOT"
      codex exec \
        --model "$ACTIVE_CODEX_MODEL" \
        "${ACTIVE_CODEX_CONFIG_ARGS[@]}" \
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
    write_state_id_atomic "$ACTIVE_CODEX_THREAD_FILE" "$thread_id"
  elif [[ ! -s "$ACTIVE_CODEX_THREAD_FILE" ]]; then
    printf 'Codex did not report a persistent thread id\n' >&2
    return 1
  fi
}

run_claude() {
  local prompt_file="$1" report_file="$2" event_file="$3" session_id="" error_file=""
  local -a args=(
    -p
    --model "$EUNOMIA_PATROL_MODEL"
    --effort "$EUNOMIA_PATROL_REASONING_EFFORT"
    --permission-mode bypassPermissions
    --output-format json
  )
  if [[ -e "$CLAUDE_SESSION_FILE" && ! -s "$CLAUDE_SESSION_FILE" ]]; then
    printf 'Claude continuity file exists but is empty: %s\n' "$CLAUDE_SESSION_FILE" >&2
    return 1
  fi
  if [[ -s "$CLAUDE_SESSION_FILE" ]]; then
    args+=(--resume "$(<"$CLAUDE_SESSION_FILE")")
  else
    : >"$CLAUDE_SESSION_FILE"
    chmod 0600 "$CLAUDE_SESSION_FILE"
  fi
  error_file="${event_file%.ndjson}.stderr.log"
  : >"$error_file"
  chmod 0600 "$error_file"
  (
    cd "$REPO_ROOT"
    claude "${args[@]}" "$(cat "$prompt_file")" >"$event_file" 2>"$error_file"
  )
  jq -er '.result // empty' "$event_file" >"$report_file"
  session_id="$(jq -r '.session_id // empty' "$event_file")"
  if [[ -z "$session_id" ]]; then
    printf 'Claude did not report a persistent session id\n' >&2
    return 1
  fi
  write_state_id_atomic "$CLAUDE_SESSION_FILE" "$session_id"
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
    probe_model
    run_patrol
    ;;
  *)
    printf 'usage: %s [--check|--probe|--run]\n' "$0" >&2
    exit 2
    ;;
esac
