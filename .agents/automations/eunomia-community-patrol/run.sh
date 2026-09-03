#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR/../../.." rev-parse --show-toplevel)"
# shellcheck source=task.env
. "$SCRIPT_DIR/task.env"

: "${EUNOMIA_PATROL_COORDINATOR_AGENT:?missing coordinator Agent selection}"
: "${EUNOMIA_PATROL_COORDINATOR_MODEL:?missing coordinator model selection}"
: "${EUNOMIA_PATROL_COORDINATOR_REASONING_EFFORT:?missing coordinator reasoning effort}"
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
PATROL_SKILL="$REPO_ROOT/.agents/skills/eunomia-community-patrol/SKILL.md"
PROMPT_SOURCE="$SCRIPT_DIR/prompt.md"
WORKER_LANES=(partition-1 partition-2 partition-3)
read -r -a WORKER_MODELS <<<"${EUNOMIA_PATROL_WORKER_MODELS:-}"

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
  if [[ "$EUNOMIA_PATROL_COORDINATOR_AGENT" != "codex" ]]; then
    printf 'the patrol reconciliation Agent must be codex, got: %s\n' "$EUNOMIA_PATROL_COORDINATOR_AGENT" >&2
    return 1
  fi
  command -v codex >/dev/null 2>&1 || {
    printf 'reconciliation Agent is unavailable: codex\n' >&2
    return 1
  }
  command -v opencode >/dev/null 2>&1 || {
    printf 'execution worker is unavailable: opencode\n' >&2
    return 1
  }
  if [[ "${#WORKER_MODELS[@]}" -ne "${#WORKER_LANES[@]}" ]]; then
    printf 'expected %s worker models, got %s\n' \
      "${#WORKER_LANES[@]}" "${#WORKER_MODELS[@]}" >&2
    return 1
  fi
}

render_prompt() {
  local worker_dir="$1" lane
  cat "$PROMPT_SOURCE"
  printf '\nRuntime paths for this invocation:\n'
  printf -- '- repository root: %s\n' "$REPO_ROOT"
  printf -- '- patrol memory: %s\n' "$MEMORY_PATH"
  printf -- '- repository checkout pool: %s\n' "$CHECKOUT_ROOT"
  printf -- '- oss-issue-triage Skill: %s\n' "$TRIAGE_SKILL"
  printf -- '- oss-change-workflow Skill: %s\n' "$CHANGE_SKILL"
  printf '\nCompleted peer-worker reports for this invocation:\n'
  for lane in "${WORKER_LANES[@]}"; do
    printf -- '- %s: report=%s/%s.md status=%s/%s.status\n' \
      "$lane" "$worker_dir" "$lane" "$worker_dir" "$lane"
  done
  printf '%s\n' \
    'Three fully enabled peer workers ran first on disjoint open-item partitions.' \
    'They were authorized to inspect, execute commands, edit source, validate, commit, push, open pull requests, and send the public GitHub replies allowed by the patrol Skill.' \
    'Read every ready report and verify its claimed external state before relying on it. Do not repeat an action already completed by a worker.' \
    'You are the reconciliation worker, not the sole writer. Complete the organization-wide inventory, handle eligible items left unfinished, update shared patrol memory, and produce the final report. An unavailable worker must not block the remaining patrol.'
}

probe_model() {
  local probe_file probe_log
  probe_file="$(mktemp "$STATE_ROOT/model-probe.XXXXXX")"
  probe_log="$(mktemp "$STATE_ROOT/model-probe-log.XXXXXX")"
  if ! timeout --foreground 180 codex exec --ephemeral \
      --model "$EUNOMIA_PATROL_COORDINATOR_MODEL" \
      --config "model_reasoning_effort=\"$EUNOMIA_PATROL_COORDINATOR_REASONING_EFFORT\"" \
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
  printf 'model_probe=ready agent=codex route=original model=%s effort=%s\n' \
    "$EUNOMIA_PATROL_COORDINATOR_MODEL" \
    "$EUNOMIA_PATROL_COORDINATOR_REASONING_EFFORT"
}

render_worker_config() {
  local model="$1"
  jq -cn \
    --arg provider "$EUNOMIA_PATROL_WORKER_PROVIDER" \
    --arg base_url "$EUNOMIA_PATROL_WORKER_BASE_URL" \
    --arg model "$model" \
    '{
      "$schema": "https://opencode.ai/config.json",
      permission: "allow",
      provider: {
        ($provider): {
          npm: "@ai-sdk/openai-compatible",
          name: "Spark LiteLLM gateway",
          options: {
            baseURL: $base_url,
            apiKey: "{env:LITELLM_API_KEY}"
          },
          models: {
            ($model): {name: $model}
          }
        }
      },
      agent: {
        "patrol-worker": {
          description: "Fully enabled Eunomia patrol execution worker",
          mode: "primary",
          permission: "allow"
        }
      }
    }'
}

write_worker_prompt() {
  local lane="$1" snapshot="$2" target="$3" checkout_dir="$4"
  {
    printf '%s\n' \
      'You are a fully enabled execution worker for the Eunomia community patrol.' \
      'Read the patrol Skill and runtime memory completely, then use all tools needed to complete real work for every actionable item in your assigned snapshot.' \
      'Within the Skill authorization you may inspect GitHub, run commands, clone repositories, edit and test code, commit, push, open pull requests, and send public GitHub replies. Do not stop at suggestions when an authorized action can be completed.' \
      'Your snapshot is a disjoint partition. Work only on those assigned items so parallel workers do not duplicate or conflict with one another.' \
      'Use the isolated checkout pool below for repository changes. Do not edit the automation source checkout.' \
      'Do not update the shared memory file concurrently; record every action, URL, commit, pull request, blocker, and required memory update in your final Chinese report for the reconciliation worker.'
    printf 'Worker lane: %s\nPatrol skill: %s\nPatrol memory: %s\nAssigned open-item snapshot: %s\nIsolated checkout pool: %s\n' \
      "$lane" "$PATROL_SKILL" "$MEMORY_PATH" "$snapshot" "$checkout_dir"
  } >"$target"
  chmod 0600 "$target"
}

collect_open_item_snapshot() {
  local target="$1" temporary
  temporary="$(mktemp "${target}.XXXXXX")"
  if gh search issues --owner eunomia-bpf --state open --limit 1000 \
      --json author,commentsCount,createdAt,isPullRequest,labels,number,repository,state,title,updatedAt,url \
      >"$temporary" 2>/dev/null; then
    mv -f -- "$temporary" "$target"
  else
    printf '[]\n' >"$target"
    rm -f -- "$temporary"
  fi
  chmod 0600 "$target"
}

partition_open_item_snapshot() {
  local snapshot="$1" index="$2" count="$3" target="$4" temporary
  temporary="$(mktemp "${target}.XXXXXX")"
  jq --argjson index "$index" --argjson count "$count" \
    '[to_entries[] | select((.key % $count) == $index) | .value]' \
    "$snapshot" >"$temporary"
  mv -f -- "$temporary" "$target"
  chmod 0600 "$target"
}

run_worker() {
  local lane="$1" model="$2" prompt_file="$3" report_file="$4" status_file="$5" checkout_dir="$6"
  local config temporary_report temporary_status stderr_file
  temporary_report="$(mktemp "${report_file}.XXXXXX")"
  temporary_status="$(mktemp "${status_file}.XXXXXX")"
  stderr_file="${report_file%.md}.stderr.log"
  : >"$stderr_file"
  chmod 0600 "$temporary_report" "$temporary_status" "$stderr_file"

  if [[ -z "${LITELLM_API_KEY:-}" ]]; then
    printf 'unavailable: LITELLM_API_KEY is missing\n' >"$temporary_status"
  else
    config="$(render_worker_config "$model")"
    if env OPENCODE_CONFIG_CONTENT="$config" \
        opencode run --pure \
          --agent patrol-worker \
          --model "${EUNOMIA_PATROL_WORKER_PROVIDER}/${model}" \
          --format default \
          --dir "$checkout_dir" \
          "$(cat "$prompt_file")" \
          >"$temporary_report" 2>"$stderr_file" && \
        [[ -s "$temporary_report" ]]; then
      mv -f -- "$temporary_report" "$report_file"
      printf 'ready\n' >"$temporary_status"
    else
      rm -f -- "$temporary_report"
      printf 'unavailable: model call failed\n' >"$temporary_status"
    fi
  fi
  mv -f -- "$temporary_status" "$status_file"
  printf 'worker_status=%s model=%s lane=%s\n' \
    "$(cut -d: -f1 <"$status_file")" "$model" "$lane"
}

probe_workers() {
  local worker_dir lane model index pid status_file checkout_dir
  local -a pids=()
  worker_dir="$(mktemp -d "$STATE_ROOT/worker-probe.XXXXXX")"
  chmod 0700 "$worker_dir"
  for index in "${!WORKER_LANES[@]}"; do
    lane="${WORKER_LANES[$index]}"
    model="${WORKER_MODELS[$index]}"
    checkout_dir="$worker_dir/$lane-checkout"
    install -d -m 0700 "$checkout_dir"
    printf 'Reply with exactly EUNOMIA_PATROL_WORKER_READY.\n' >"$worker_dir/$lane.prompt"
    run_worker "$lane" "$model" "$worker_dir/$lane.prompt" \
      "$worker_dir/$lane.md" "$worker_dir/$lane.status" "$checkout_dir" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
  for lane in "${WORKER_LANES[@]}"; do
    status_file="$worker_dir/$lane.status"
    [[ -s "$status_file" ]] || printf 'unavailable: no status\n' >"$status_file"
    printf 'worker_probe=%s lane=%s\n' "$(cut -d: -f1 <"$status_file")" "$lane"
  done
  rm -rf -- "$worker_dir"
}

run_codex() {
  local prompt_file="$1" report_file="$2" event_file="$3" thread_id=""
  if [[ -e "$THREAD_FILE" && ! -s "$THREAD_FILE" ]]; then
    printf 'Codex continuity file exists but is empty: %s\n' "$THREAD_FILE" >&2
    return 1
  fi
  if [[ ! -e "$THREAD_FILE" ]]; then
    : >"$THREAD_FILE"
    chmod 0600 "$THREAD_FILE"
  fi

  (
    cd "$REPO_ROOT"
  if [[ -s "$THREAD_FILE" ]]; then
    thread_id="$(<"$THREAD_FILE")"
    codex exec resume \
      --model "$EUNOMIA_PATROL_COORDINATOR_MODEL" \
      --config "model_reasoning_effort=\"$EUNOMIA_PATROL_COORDINATOR_REASONING_EFFORT\"" \
      --dangerously-bypass-approvals-and-sandbox \
      --json \
      --output-last-message "$report_file" \
      "$thread_id" - <"$prompt_file" >"$event_file" 2>&1
  else
    (
      cd "$REPO_ROOT"
      codex exec \
        --model "$EUNOMIA_PATROL_COORDINATOR_MODEL" \
        --config "model_reasoning_effort=\"$EUNOMIA_PATROL_COORDINATOR_REASONING_EFFORT\"" \
        --dangerously-bypass-approvals-and-sandbox \
        --json \
        --output-last-message "$report_file" \
        - <"$prompt_file" >"$event_file" 2>&1
    )
  fi
  )

  thread_id="$(jq -Rr 'fromjson? | select(.type == "thread.started") | .thread_id' "$event_file" | head -n 1)"
  if [[ -n "$thread_id" ]]; then
    write_state_id_atomic "$THREAD_FILE" "$thread_id"
  elif [[ ! -s "$THREAD_FILE" ]]; then
    printf 'Codex did not report a persistent thread id\n' >&2
    return 1
  fi
}

run_patrol() {
  local timestamp prompt_file report_tmp report_file event_file worker_dir snapshot partition checkout_dir
  local lane model index pid
  local -a worker_pids=()

  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  prompt_file="$(mktemp "$STATE_ROOT/prompt.XXXXXX")"
  report_tmp="$(mktemp "$REPORT_DIR/report.XXXXXX")"
  report_file="$REPORT_DIR/$timestamp.md"
  event_file="$LOG_DIR/$timestamp.ndjson"
  worker_dir="$(mktemp -d "$STATE_ROOT/workers.XXXXXX")"
  snapshot="$worker_dir/open-items.json"
  trap "rm -f -- $(printf '%q' "$prompt_file") $(printf '%q' "$report_tmp"); rm -rf -- $(printf '%q' "$worker_dir")" EXIT
  : >"$event_file"
  collect_open_item_snapshot "$snapshot"
  for index in "${!WORKER_LANES[@]}"; do
    lane="${WORKER_LANES[$index]}"
    model="${WORKER_MODELS[$index]}"
    partition="$worker_dir/$lane-items.json"
    checkout_dir="$CHECKOUT_ROOT/$lane"
    install -d -m 0700 "$checkout_dir"
    partition_open_item_snapshot "$snapshot" "$index" "${#WORKER_LANES[@]}" "$partition"
    write_worker_prompt "$lane" "$partition" "$worker_dir/$lane.prompt" "$checkout_dir"
    run_worker "$lane" "$model" "$worker_dir/$lane.prompt" \
      "$worker_dir/$lane.md" "$worker_dir/$lane.status" "$checkout_dir" &
    worker_pids+=("$!")
  done

  for pid in "${worker_pids[@]}"; do
    wait "$pid" || true
  done
  for lane in "${WORKER_LANES[@]}"; do
    [[ -s "$worker_dir/$lane.status" ]] || printf 'unavailable: no status\n' >"$worker_dir/$lane.status"
  done

  render_prompt "$worker_dir" >"$prompt_file"
  chmod 0600 "$prompt_file" "$report_tmp" "$event_file"
  run_codex "$prompt_file" "$report_tmp" "$event_file"

  require_file "$report_tmp"
  mv "$report_tmp" "$report_file"
  rm -f "$prompt_file"
  trap - EXIT
  chmod 0600 "$report_file" "$event_file"
  ln -sfn "$report_file" "$STATE_ROOT/latest-report.md"
  rm -rf -- "$worker_dir"
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
    "$EUNOMIA_PATROL_COORDINATOR_AGENT" --version
    printf 'memory_bytes=%s memory_sha256=%s\n' \
      "$(wc -c <"$MEMORY_PATH")" \
      "$(sha256sum "$MEMORY_PATH" | cut -d' ' -f1)"
    printf 'runtime_check=ready\n'
    ;;
  --probe)
    prepare_skills
    check_runtime
    probe_model
    probe_workers
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
