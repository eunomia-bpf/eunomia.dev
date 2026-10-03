#!/bin/bash
# Measure persistent SECRET-label intervention burden over a process lineage.
# Run as root after building target/release/actplane and bpf/process.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ACT="${ACTPLANE_BIN:-$ROOT/target/release/actplane}"
PROC="${ACTPLANE_PROCESS_BIN:-$ROOT/bpf/process}"
OUT="${1:-$ROOT/docs/empirical-study/results/long-session-overtaint}"
WORK="$(mktemp -d /tmp/actplane-long-session.XXXXXX)"
READY_TRIES=2400
RUN_TRIES=1000

cleanup() {
  if [ -n "${loader_pid:-}" ]; then kill "$loader_pid" 2>/dev/null || true; fi
  if [ -n "${trigger_pid:-}" ]; then kill -CONT "$trigger_pid" 2>/dev/null || true; kill "$trigger_pid" 2>/dev/null || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 2; }
[ -x "$ACT" ] || { echo "missing $ACT" >&2; exit 2; }
[ -x "$PROC" ] || { echo "missing $PROC" >&2; exit 2; }
mkdir -p "$OUT/raw"

secret="$WORK/session.env"
printf '%s\n' 'TOKEN=frozen-experiment-secret' > "$secret"
policy="source SECRET = file \"$secret\"
rule long-session-no-egress:
  notify connect endpoint \"*\" if SECRET
  because \"A process lineage that has read a secret remains in sensitive context\""
"$ACT" --rule "$policy" compile --out "$WORK/config.bin" --force >"$OUT/compile.stdout" 2>"$OUT/compile.stderr"

wait_stopped() {
  local pid="$1" stat state
  for _ in $(seq 1 200); do
    stat="$(cat "/proc/$pid/stat" 2>/dev/null)" || return 1
    stat="${stat##*) }"; state="${stat%% *}"
    case "$state" in T|t) return 0;; Z) return 1;; esac
    sleep 0.01
  done
  return 1
}

wait_ready() {
  local pid="$1" file="$2"
  for _ in $(seq 1 "$READY_TRIES"); do
    grep -q "ActPlane: ready" "$file" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 0.025
  done
  return 1
}

wait_done() {
  local pid="$1" stat state
  for _ in $(seq 1 "$RUN_TRIES"); do
    stat="$(cat "/proc/$pid/stat" 2>/dev/null)" || return 0
    stat="${stat##*) }"; state="${stat%% *}"
    [ "$state" = Z ] && return 0
    sleep 0.01
  done
  return 1
}

connect_loop='for i in $(seq 1 "$N"); do timeout 0.2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$((31000+i))" 2>/dev/null || true; done'

run_case() {
  local name="$1" expected="$2" body="$3" result err got
  result="$OUT/raw/$name.jsonl"
  err="$OUT/raw/$name.stderr"
  : > "$result"; : > "$err"
  SECRET_PATH="$secret" CONNECT_LOOP="$connect_loop" /bin/bash -c 'kill -STOP $$; eval "$1"' actplane-long-session "$body" >/dev/null 2>&1 &
  trigger_pid=$!
  wait_stopped "$trigger_pid" || { echo "$name trigger-not-stopped" >&2; return 1; }
  "$PROC" --config "$WORK/config.bin" --seed-pid "$trigger_pid" >"$result" 2>"$err" &
  loader_pid=$!
  wait_ready "$loader_pid" "$err" || { echo "$name loader-not-ready" >&2; return 1; }
  kill -CONT "$trigger_pid"
  wait_done "$trigger_pid" || kill "$trigger_pid" 2>/dev/null || true
  wait "$trigger_pid" 2>/dev/null || true
  sleep 0.2
  kill "$loader_pid" 2>/dev/null || true
  wait "$loader_pid" 2>/dev/null || true
  loader_pid=""; trigger_pid=""
  got="$(grep -c '"rule":"long-session-no-egress"' "$result" || true)"
  printf '%s\t%s\t%s\n' "$name" "$expected" "$got" >> "$OUT/counts.tsv"
  [ "$got" = "$expected" ]
}

: > "$OUT/counts.tsv"
printf '%s\t%s\t%s\n' case expected_interventions observed_interventions >> "$OUT/counts.tsv"
failures=0
run_case clean_pre_read 0 'N=5; eval "$CONNECT_LOOP"' || failures=$((failures + 1))
run_case same_lineage_1 1 'read -r x < "$SECRET_PATH"; N=1; eval "$CONNECT_LOOP"' || failures=$((failures + 1))
run_case same_lineage_5 5 'read -r x < "$SECRET_PATH"; N=5; eval "$CONNECT_LOOP"' || failures=$((failures + 1))
run_case same_lineage_20 20 'read -r x < "$SECRET_PATH"; N=20; eval "$CONNECT_LOOP"' || failures=$((failures + 1))
run_case sibling_after_reader_exit 0 '(read -r x < "$SECRET_PATH"); N=5; eval "$CONNECT_LOOP"' || failures=$((failures + 1))
run_case descendant_after_read_5 5 'read -r x < "$SECRET_PATH"; export N=5; bash -c "$CONNECT_LOOP"' || failures=$((failures + 1))

{
  printf 'timestamp_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'git_commit\t%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
  printf 'kernel\t%s\n' "$(uname -r)"
  printf 'policy_sha256\t%s\n' "$(printf '%s' "$policy" | sha256sum | cut -d' ' -f1)"
  printf 'actplane_bin\t%s\n' "$ACT"
  printf 'process_bin\t%s\n' "$PROC"
} > "$OUT/metadata.tsv"
printf '%s\n' "$policy" > "$OUT/policy.dsl"
if [ "$failures" -ne 0 ]; then
  echo "wrote failed run evidence to $OUT ($failures case failures)" >&2
  exit 1
fi
echo "wrote successful run to $OUT"
