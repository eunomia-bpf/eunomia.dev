#!/usr/bin/env bash
set -euo pipefail; umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/task.env"
REPO_ROOT="$(git -C "$SCRIPT_DIR/../../.." rev-parse --show-toplevel)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
STATE_ROOT="${EUNOMIA_PATROL_STATE_ROOT:-/workspaces/.agent-state/eunomia-community-patrol}"
install -d -m 0700 "$STATE_ROOT/logs" "$STATE_ROOT/reports"
prompt="$(mktemp "$STATE_ROOT/prompt.XXXXXX")"
sed -e "s|__REPO_ROOT__|$REPO_ROOT|g" -e "s|__STATE_ROOT__|$STATE_ROOT|g" -e "s|__LOCAL_MODELS__|$EUNOMIA_PATROL_LOCAL_MODELS|g" "$SCRIPT_DIR/prompt.md" >"$prompt"
CF="--model $EUNOMIA_PATROL_AGENT_MODEL --config model_reasoning_effort=$EUNOMIA_PATROL_AGENT_EFFORT --dangerously-bypass-approvals-and-sandbox"
PRIMARY="cd $REPO_ROOT && if [[ -n \$RUNNER_SESSION_ID ]]; then exec codex exec resume $CF --json --output-last-message \"\$RUNNER_REPORT_FILE\" \"\$RUNNER_SESSION_ID\" -; else exec codex exec $CF --json --output-last-message \"\$RUNNER_REPORT_FILE\" -; fi"
FALLBACK='o="$(mktemp "$RUNNER_STATE_DIR/oc-events.XXXXXX")"; opencode run --auto --format json --dir '"$REPO_ROOT"' --model spark-gateway/'"$EUNOMIA_PATROL_FALLBACK_MODEL"' "$(cat "$RUNNER_PROMPT_FILE")" | tee "$o"; r="$(jq -r "select(.type==\"step_finish\") | .part.reason // \"unknown\"" "$o" | tail -n 1)"; jq -rs "map(select(.type==\"text\" and .part.time?.end) | .part.text) | last // empty" "$o" >"$RUNNER_REPORT_FILE"; [[ "$r" == "stop" && -s "$RUNNER_REPORT_FILE" ]]'
export TZ="$EUNOMIA_PATROL_TIMEZONE" RUNNER_STATE_DIR="$STATE_ROOT" RUNNER_SESSION_FILE="$STATE_ROOT/codex-thread-id" RUNNER_SESSION_EXTRACT='select(.type == "thread.started") | .thread_id // empty'
exec flock --close --nonblock --conflict-exit-code 75 "$STATE_ROOT/patrol.lock" >>"$STATE_ROOT/logs/$timestamp.ndjson" 2>&1 \
  "$REPO_ROOT/.agents/automations/shared/agent-runner.sh" \
  --prompt "$prompt" --events "$STATE_ROOT/logs/$timestamp.ndjson" --report "$STATE_ROOT/reports/$timestamp.md" \
  --primary "$PRIMARY" --fallback "$FALLBACK"
