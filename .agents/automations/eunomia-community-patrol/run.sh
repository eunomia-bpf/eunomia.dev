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
read -r -a COORDINATOR_FALLBACK_MODELS <<<"${EUNOMIA_PATROL_COORDINATOR_FALLBACK_MODELS:-}"
MODEL_PROBE_TIMEOUT_SECONDS="${EUNOMIA_PATROL_MODEL_PROBE_TIMEOUT_SECONDS:-90}"
WORKER_TIMEOUT_SECONDS="${EUNOMIA_PATROL_WORKER_TIMEOUT_SECONDS:-900}"
COORDINATOR_TIMEOUT_SECONDS="${EUNOMIA_PATROL_COORDINATOR_TIMEOUT_SECONDS:-1800}"
FALLBACK_COORDINATOR_BUDGET_SECONDS="${EUNOMIA_PATROL_FALLBACK_COORDINATOR_BUDGET_SECONDS:-1800}"

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
  if [[ "${#COORDINATOR_FALLBACK_MODELS[@]}" -eq 0 ]]; then
    printf 'at least one OpenCode coordinator fallback model is required\n' >&2
    return 1
  fi
  if [[ ! "$MODEL_PROBE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
      [[ ! "$WORKER_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
      [[ ! "$COORDINATOR_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
      [[ ! "$FALLBACK_COORDINATOR_BUDGET_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'probe, worker, coordinator, and fallback budgets must be positive integers\n' >&2
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
    printf -- '- %s: report=%s/%s.md status=%s/%s.status stderr=%s/%s.stderr.log\n' \
      "$lane" "$worker_dir" "$lane" "$worker_dir" "$lane" "$worker_dir" "$lane"
  done
  printf '%s\n' \
    'Three fully enabled OpenCode workers using different local models ran first on disjoint open-item partitions.' \
    'They were authorized to inspect, execute commands, perform actual source development, validate, commit, push, and open pull requests within the patrol Skill. Public maintainer replies are reserved for the coordinator so contributors receive one coherent response.' \
    'Read every worker status and available report. For a partial or unavailable worker, inspect only the relevant action lines in its private stderr log, then verify branches, pull requests, comments, reviews, and commits against live GitHub state before writing. Do not repeat an action already completed by a worker.' \
    'You are the reconciliation and public-response coordinator, not the implementation worker. Do not personally edit implementation source or take over builds and tests. When more code work is required, dispatch it through OpenCode to the available Qwen Next, GLM Next, and Qwen 27B local routes, using more than one model when independent work or review exists.' \
    'Treat the local model context windows as approximately 200k tokens. Keep headroom by passing focused files, issue evidence, and compact summaries instead of the full organization history or large raw logs.' \
    'Complete the organization-wide inventory, send eligible maintainer replies, update shared patrol memory, and produce the final report. An unavailable worker or exhausted Codex route must not block the patrol; the runner can transfer coordination to OpenCode, whose coordinator must recheck external state before continuing.'
}

probe_model() {
  local probe_file probe_log
  probe_file="$(mktemp "$STATE_ROOT/model-probe.XXXXXX")"
  probe_log="$(mktemp "$STATE_ROOT/model-probe-log.XXXXXX")"
  if ! timeout --foreground "$MODEL_PROBE_TIMEOUT_SECONDS" codex exec --ephemeral \
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
        },
        "patrol-coordinator": {
          description: "Fallback Eunomia patrol coordinator and public-response worker",
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
      'You are a fully enabled OpenCode execution worker using one of the Eunomia patrol local models.' \
      'Read the patrol Skill and runtime memory completely, then use all tools needed to complete real implementation work for every actionable item in your assigned snapshot.' \
      'Within the Skill authorization you may inspect GitHub, run commands, clone repositories, edit and test code, commit, push, and open pull requests. Do not stop at suggestions when an authorized development action can be completed.' \
      'Do not send issue comments, pull-request comments, or reviews; record the exact evidence and proposed response for the coordinator, which owns coherent public replies. This is task coordination, not a tool-permission restriction.' \
      'Your snapshot is a disjoint partition. Work only on those assigned items so workers using Qwen Next, GLM Next, and Qwen 27B do not duplicate or conflict with one another.' \
      'Treat your context window as approximately 200k tokens. Keep enough headroom for implementation and validation by reading focused files and compact evidence instead of loading whole repositories, histories, or raw logs.' \
      'Use the isolated checkout pool below for repository changes. Do not edit the automation source checkout. Keep build concurrency memory-bounded.' \
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
  local config temporary_report temporary_status stderr_file worker_rc
  temporary_report="$(mktemp "${report_file}.XXXXXX")"
  temporary_status="$(mktemp "${status_file}.XXXXXX")"
  stderr_file="${report_file%.md}.stderr.log"
  : >"$stderr_file"
  chmod 0600 "$temporary_report" "$temporary_status" "$stderr_file"

  if [[ -z "${LITELLM_API_KEY:-}" ]]; then
    printf 'unavailable: LITELLM_API_KEY is missing\n' >"$temporary_status"
  else
    config="$(render_worker_config "$model")"
    worker_rc=0
    env OPENCODE_CONFIG_CONTENT="$config" \
        MAKEFLAGS="-j1" \
        CMAKE_BUILD_PARALLEL_LEVEL=1 \
        CARGO_BUILD_JOBS=1 \
        GOMAXPROCS=2 \
      timeout --foreground "$WORKER_TIMEOUT_SECONDS" \
        opencode run --pure \
          --agent patrol-worker \
          --model "${EUNOMIA_PATROL_WORKER_PROVIDER}/${model}" \
          --format default \
          --dir "$checkout_dir" \
          "$(cat "$prompt_file")" \
          >"$temporary_report" 2>"$stderr_file" || worker_rc=$?
    if [[ "$worker_rc" -eq 0 ]] && [[ -s "$temporary_report" ]]; then
      mv -f -- "$temporary_report" "$report_file"
      printf 'ready\n' >"$temporary_status"
    else
      if [[ -s "$temporary_report" ]]; then
        mv -f -- "$temporary_report" "$report_file"
        if [[ "$worker_rc" -eq 124 ]]; then
          printf 'partial: worker exceeded %ss bound; reconcile report and stderr against live state\n' \
            "$WORKER_TIMEOUT_SECONDS" >"$temporary_status"
        else
          printf 'partial: model call failed with exit %s; reconcile report and stderr against live state\n' \
            "$worker_rc" >"$temporary_status"
        fi
      else
        rm -f -- "$temporary_report"
        if [[ "$worker_rc" -eq 124 ]]; then
          printf 'unavailable: worker exceeded %ss bound; inspect stderr and live state\n' \
            "$WORKER_TIMEOUT_SECONDS" >"$temporary_status"
        else
          printf 'unavailable: model call failed with exit %s; inspect stderr and live state\n' \
            "$worker_rc" >"$temporary_status"
        fi
      fi
    fi
  fi
  mv -f -- "$temporary_status" "$status_file"
  printf 'worker_status=%s model=%s lane=%s\n' \
    "$(cut -d: -f1 <"$status_file")" "$model" "$lane"
}

probe_workers() {
  local worker_dir lane model index status_file checkout_dir proof_file
  worker_dir="$(mktemp -d "$STATE_ROOT/worker-probe.XXXXXX")"
  chmod 0700 "$worker_dir"
  for index in "${!WORKER_LANES[@]}"; do
    lane="${WORKER_LANES[$index]}"
    model="${WORKER_MODELS[$index]}"
    checkout_dir="$worker_dir/$lane-checkout"
    proof_file="$checkout_dir/tool-proof.txt"
    install -d -m 0700 "$checkout_dir"
    printf 'Use your shell tool to write exactly EUNOMIA_PATROL_TOOL_READY to %s, read the file back, then reply with exactly EUNOMIA_PATROL_WORKER_READY.\n' \
      "$proof_file" >"$worker_dir/$lane.prompt"
    run_worker "$lane" "$model" "$worker_dir/$lane.prompt" \
      "$worker_dir/$lane.md" "$worker_dir/$lane.status" "$checkout_dir"
  done
  for lane in "${WORKER_LANES[@]}"; do
    status_file="$worker_dir/$lane.status"
    proof_file="$worker_dir/$lane-checkout/tool-proof.txt"
    [[ -s "$status_file" ]] || printf 'unavailable: no status\n' >"$status_file"
    if [[ "$(cut -d: -f1 <"$status_file")" == "ready" ]] && \
        ! grep -Fxq 'EUNOMIA_PATROL_TOOL_READY' "$proof_file" 2>/dev/null; then
      printf 'unavailable: tool write proof missing\n' >"$status_file"
    fi
    printf 'worker_probe=%s lane=%s\n' "$(cut -d: -f1 <"$status_file")" "$lane"
  done
  rm -rf -- "$worker_dir"
}

run_codex() {
  local prompt_file="$1" report_file="$2" event_file="$3" thread_id="" codex_rc=0
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
      timeout --foreground "$COORDINATOR_TIMEOUT_SECONDS" \
        codex exec resume \
          --model "$EUNOMIA_PATROL_COORDINATOR_MODEL" \
          --config "model_reasoning_effort=\"$EUNOMIA_PATROL_COORDINATOR_REASONING_EFFORT\"" \
          --dangerously-bypass-approvals-and-sandbox \
          --json \
          --output-last-message "$report_file" \
          "$thread_id" - <"$prompt_file" >"$event_file" 2>&1
    else
      timeout --foreground "$COORDINATOR_TIMEOUT_SECONDS" \
        codex exec \
          --model "$EUNOMIA_PATROL_COORDINATOR_MODEL" \
          --config "model_reasoning_effort=\"$EUNOMIA_PATROL_COORDINATOR_REASONING_EFFORT\"" \
          --dangerously-bypass-approvals-and-sandbox \
          --json \
          --output-last-message "$report_file" \
          - <"$prompt_file" >"$event_file" 2>&1
    fi
  ) || codex_rc=$?

  thread_id="$(jq -Rr 'fromjson? | select(.type == "thread.started") | .thread_id' "$event_file" | head -n 1)"
  if [[ -n "$thread_id" ]]; then
    write_state_id_atomic "$THREAD_FILE" "$thread_id"
  elif [[ ! -s "$THREAD_FILE" ]]; then
    printf 'Codex did not report a persistent thread id\n' >&2
  fi
  return "$codex_rc"
}

append_retry_handoff() {
  local prompt_file="$1" event_file="$2" previous_route="$3"
  {
    printf '\nPrevious coordinator attempt did not finish.\n'
    printf -- '- previous route: %s\n' "$previous_route"
    printf -- '- private attempt event log: %s\n' "$event_file"
    printf '%s\n' \
      'Before any GitHub write, search that private log for completed tool calls and action identifiers without loading it wholesale into context.' \
      'Reconcile every relevant branch, commit, pull request, comment, review, and memory update against live GitHub and filesystem state. Continue only with the next missing action; never repeat a completed write.'
  } >>"$prompt_file"
}

run_opencode_coordinator() {
  local prompt_file="$1" report_file="$2" event_file="$3"
  local model config attempt_report attempt_log coordinator_rc remaining_seconds
  local fallback_deadline=$((SECONDS + FALLBACK_COORDINATOR_BUDGET_SECONDS))
  for model in "${COORDINATOR_FALLBACK_MODELS[@]}"; do
    remaining_seconds=$((fallback_deadline - SECONDS))
    if [[ "$remaining_seconds" -le 0 ]]; then
      break
    fi
    attempt_report="$(mktemp "${report_file}.opencode.XXXXXX")"
    attempt_log="$(mktemp "${event_file}.opencode.XXXXXX")"
    config="$(render_worker_config "$model")"
    coordinator_rc=0
    env OPENCODE_CONFIG_CONTENT="$config" \
        MAKEFLAGS="-j1" \
        CMAKE_BUILD_PARALLEL_LEVEL=1 \
        CARGO_BUILD_JOBS=1 \
        GOMAXPROCS=2 \
      timeout --foreground "$remaining_seconds" \
        opencode run --pure \
          --agent patrol-coordinator \
          --model "${EUNOMIA_PATROL_WORKER_PROVIDER}/${model}" \
          --format default \
          --dir "$REPO_ROOT" \
          "$(cat "$prompt_file")" \
          >"$attempt_report" 2>"$attempt_log" || coordinator_rc=$?
    {
      printf 'coordinator_attempt=opencode model=%s exit=%s\n' "$model" "$coordinator_rc"
      cat "$attempt_log"
      if [[ "$coordinator_rc" -ne 0 ]] && [[ -s "$attempt_report" ]]; then
        printf '\npartial_coordinator_output model=%s\n' "$model"
        cat "$attempt_report"
      fi
    } >>"$event_file"
    rm -f -- "$attempt_log"
    if [[ "$coordinator_rc" -eq 0 ]] && [[ -s "$attempt_report" ]]; then
      mv -f -- "$attempt_report" "$report_file"
      printf 'coordinator_route=opencode model=%s\n' "$model"
      return 0
    fi
    rm -f -- "$attempt_report"
    append_retry_handoff "$prompt_file" "$event_file" "OpenCode/$model"
  done
  printf 'all OpenCode coordinator fallback models failed\n' >&2
  return 1
}

run_coordinator() {
  local preferred_route="$1" prompt_file="$2" report_file="$3" event_file="$4"
  if [[ "$preferred_route" == "codex" ]]; then
    if run_codex "$prompt_file" "$report_file" "$event_file" && [[ -s "$report_file" ]]; then
      printf 'coordinator_route=codex model=%s\n' "$EUNOMIA_PATROL_COORDINATOR_MODEL"
      return 0
    fi
    printf 'coordinator_fallback=opencode reason=codex_unavailable_or_incomplete\n'
    printf '\ncoordinator_fallback=opencode reason=codex_unavailable_or_incomplete\n' >>"$event_file"
    append_retry_handoff "$prompt_file" "$event_file" "Codex/$EUNOMIA_PATROL_COORDINATOR_MODEL"
    : >"$report_file"
  fi
  run_opencode_coordinator "$prompt_file" "$report_file" "$event_file"
}

run_patrol() {
  local timestamp prompt_file report_tmp report_file event_file worker_dir snapshot partition checkout_dir
  local lane model index coordinator_route="opencode" failed_worker_dir

  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  prompt_file="$(mktemp "$STATE_ROOT/prompt.XXXXXX")"
  report_tmp="$(mktemp "$REPORT_DIR/report.XXXXXX")"
  report_file="$REPORT_DIR/$timestamp.md"
  event_file="$LOG_DIR/$timestamp.ndjson"
  worker_dir="$(mktemp -d "$STATE_ROOT/workers.XXXXXX")"
  snapshot="$worker_dir/open-items.json"
  trap "rm -f -- $(printf '%q' "$prompt_file") $(printf '%q' "$report_tmp"); rm -rf -- $(printf '%q' "$worker_dir")" EXIT
  : >"$event_file"
  if probe_model; then
    coordinator_route="codex"
  else
    printf 'coordinator_probe=unavailable fallback=opencode\n'
  fi
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
      "$worker_dir/$lane.md" "$worker_dir/$lane.status" "$checkout_dir"
  done

  for lane in "${WORKER_LANES[@]}"; do
    [[ -s "$worker_dir/$lane.status" ]] || printf 'unavailable: no status\n' >"$worker_dir/$lane.status"
  done

  render_prompt "$worker_dir" >"$prompt_file"
  chmod 0600 "$prompt_file" "$report_tmp" "$event_file"
  if ! run_coordinator "$coordinator_route" "$prompt_file" "$report_tmp" "$event_file"; then
    failed_worker_dir="$LOG_DIR/$timestamp-workers"
    mv -- "$worker_dir" "$failed_worker_dir"
    chmod -R go-rwx "$failed_worker_dir"
    printf 'coordinator_failed worker_evidence=%s\n' "$failed_worker_dir" >>"$event_file"
    return 1
  fi

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
    run_patrol
    ;;
  *)
    printf 'usage: %s [--check|--probe|--run]\n' "$0" >&2
    exit 2
    ;;
esac
