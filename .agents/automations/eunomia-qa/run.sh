#!/usr/bin/env bash
# eunomia-qa runner: --check | --probe | --run
set -u

QA_DIR="${EUNOMIA_QA_DIR:-$(cd "$(dirname "$0")" && pwd)}"
STATE_DIR="/workspaces/.agent-state/eunomia-qa"
VENV_PY="$STATE_DIR/venv/bin/python"
LOCK_FILE="$STATE_DIR/run.lock"
TOTAL_SECS=43200
TERM_GRACE_SECS=30
DEFAULT_MODEL="spark-gateway/qwen3.8-27b-nvfp4-200k"
ALLOWED_MODELS=(
  "spark-gateway/qwen3.8-27b-nvfp4-200k"
  "spark-gateway/qwen3.8-flash-next-nvfp4-220k"
  "spark-gateway/glm-5.3-flash-nvfp4-dflash2-200k"
)

log() { printf '[eunomia-qa] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 1; }

check_env_key() {
  [ -n "${!1:-}" ]
}

pick_model() {
  local want="${EUNOMIA_QA_MODEL:-$DEFAULT_MODEL}" m
  for m in "${ALLOWED_MODELS[@]}"; do
    [ "$want" = "$m" ] && { printf '%s' "$m"; return 0; }
  done
  return 1
}

cmd_check() {
  local fail=0 have
  for have in bash git flock timeout setsid opencode; do
    if command -v "$have" >/dev/null 2>&1; then
      log "tool ok: $have"
    else
      log "tool MISSING: $have"; fail=1
    fi
  done
  if [ -x "$VENV_PY" ]; then
    log "venv ok: $VENV_PY"
  else
    log "venv MISSING: $VENV_PY"; fail=1
  fi
  local key
  for key in OPENCODE_CONFIG LITELLM_API_KEY EUNOMIA_QA_ARCHIVE_DSN; do
    if check_env_key "$key"; then
      log "env ok: $key"
    else
      log "env MISSING: $key"; fail=1
    fi
  done
  [ -f "$QA_DIR/prompt.md" ] || { log "prompt template MISSING: $QA_DIR/prompt.md"; fail=1; }
  [ -f "$QA_DIR/verify_publication.py" ] || { log "verifier MISSING: $QA_DIR/verify_publication.py"; fail=1; }
  if pick_model >/dev/null; then
    log "model ok: ${EUNOMIA_QA_MODEL:-$DEFAULT_MODEL}"
  else
    log "model not allowed: ${EUNOMIA_QA_MODEL:-<unset>}"; fail=1
  fi
  [ "$fail" -eq 0 ] && log "check: OK" || log "check: FAILED"
  return "$fail"
}

cmd_probe() {
  cd "$QA_DIR" || die "cannot cd $QA_DIR"
  "$VENV_PY" archive_reader.py probe
}

# Internal mode: runs under flock + timeout, preconditions already verified.
cmd_locked() {
  umask 077
  local work snapshot receipt prompt_resolved model_out model_err
  local model start_sha run_date rc snapsize
  local modpid=""
  work="$(mktemp -d /tmp/eunomia-qa-run-XXXXXX)" || die "mktemp failed"
  snapshot="$work/snapshot.txt"
  receipt="$work/receipt.json"
  prompt_resolved="$work/prompt.md"
  model_out="$work/model.out"
  model_err="$work/model.err"

  kill_model_group() {
    [ -n "${modpid:-}" ] || return 0
    local pg
    pg="$modpid"
    kill -0 "$pg" 2>/dev/null || return 0
    kill -TERM -- "-$pg" 2>/dev/null || true
    local i
    for i in 1 2 3 4 5 6 7 8 9 10; do
      if ! kill -0 "$pg" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pg" 2>/dev/null; then
      kill -KILL -- "-$pg" 2>/dev/null || true
    fi
    wait "$modpid" 2>/dev/null || true
    return 0
  }

  cleanup() {
    trap - EXIT HUP INT TERM
    kill_model_group
    rm -rf "$work"
  }
  trap cleanup EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM

  cd /workspaces/repository || exit 1

  git pull --ff-only || exit 1
  start_sha="$(git rev-parse HEAD)" || exit 1
  run_date="$(date +%F)"
  model="$(pick_model)" || { log "model not allowed"; exit 2; }

  log "probe: archive_reader.py probe"
  ( cd "$QA_DIR" && "$VENV_PY" archive_reader.py probe ) || exit 1
  log "snapshot: archive_reader.py snapshot"
  ( cd "$QA_DIR" && "$VENV_PY" archive_reader.py snapshot --output "$snapshot" ) || exit 1
  snapsize="$(wc -c <"$snapshot")" || exit 1
  if [ "$snapsize" -gt 120000 ]; then
    log "snapshot too large ($snapsize > 120000 bytes)"; exit 1
  fi

  [ -n "${OPENCODE_CONFIG:-}" ] || { log "OPENCODE_CONFIG unset"; exit 1; }
  [ -f "$OPENCODE_CONFIG" ] || { log "OPENCODE_CONFIG file missing"; exit 1; }

  export XDG_DATA_HOME="$work/xdg/data" XDG_STATE_HOME="$work/xdg/state"
  export XDG_CONFIG_HOME="$work/xdg/config" XDG_CACHE_HOME="$work/xdg/cache"
  mkdir -p "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
  export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$STATE_DIR/browsers}"

  sed -e "s|__SNAPSHOT_PATH__|$snapshot|g" \
      -e "s|__RECEIPT_PATH__|$receipt|g" \
      -e "s|__RUN_DATE__|$run_date|g" \
      "$QA_DIR/prompt.md" >"$prompt_resolved" || exit 1

  log "model: $model (12h max, private tmpdir)"
  setsid opencode run --auto -m "$model" "Read and follow $prompt_resolved" \
    >"$model_out" 2>"$model_err" &
  modpid=$!
  wait "$modpid"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "model failed rc=$rc (raw output kept in private tmpdir only)"; exit "$rc"
  fi
  [ -f "$QA_DIR/verify_publication.py" ] || { log "verifier MISSING, not bypassing"; exit 1; }
  log "verify: verify_publication.py"
  "$VENV_PY" "$QA_DIR/verify_publication.py" \
    --before "$start_sha" --receipt "$receipt" --date "$run_date" || exit 1
  log "run complete: $run_date (base $start_sha)"
}

cmd_run() {
  cd /workspaces/repository || die "repo missing"
  local branch dirty
  branch="$(git rev-parse --abbrev-ref HEAD)"
  [ "$branch" = "main" ] || die "not on main ($branch)"
  dirty="$(git status --porcelain)"
  [ -z "$dirty" ] || die "worktree not clean:"
  mkdir -p "$STATE_DIR" || die "state dir missing"
  log "acquiring lock $LOCK_FILE; full run capped at ${TOTAL_SECS}s"
  exec flock -x "$LOCK_FILE" \
    timeout -k "$TERM_GRACE_SECS" "$TOTAL_SECS" \
    "$0" "--locked"
}

case "${1:---help}" in
  --check)  cmd_check ;;
  --probe)  cmd_probe ;;
  --run)    cmd_run ;;
  --locked) cmd_locked ;;
  *) log "usage: run.sh --check | --probe | --run"; exit 2 ;;
esac
