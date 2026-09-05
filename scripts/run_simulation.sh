#!/usr/bin/env bash

# Wrapper to run simulation/run_simulation.py and restart it when
# the experiment directory shows no updates for a configured timeout.

set -u

CHECK_INTERVAL=${CHECK_INTERVAL:-60}      # seconds between checks
STALE_TIMEOUT=${STALE_TIMEOUT:-3000}       # seconds of no updates before restart
PYTHON=${PYTHON:-python}
SCRIPT="${SCRIPT:-$(dirname "$0")/../simulation/run_simulation.py}"

EXP_DIR="
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp_dir)
      EXP_DIR="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --start_from)
      EXP_DIR="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$EXP_DIR" ]]; then
  echo "Error: --exp_dir or --start_from must be provided to monitor updates."
  echo "Usage: $0 --exp_dir /path/to/exp [other args passed to run_simulation.py]"
  exit 2
fi

EXP_DIR="$(realpath -m "$EXP_DIR")"
mkdir -p "$EXP_DIR"

child=""

# 하위 프로세스 그룹까지 모두 종료하는 함수
kill_tree() {
  local pid=$1
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    # 자식 프로세스들까지 포함하여 SIGTERM 전송
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
    sleep 3
    if kill -0 "$pid" 2>/dev/null; then
      pkill -9 -P "$pid" 2>/dev/null || true
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
}

cleanup() {
  local sig="$1"
  echo "Stopping monitor (received ${sig})."
  kill_tree "$child"
  case "$sig" in
    INT) exit 130 ;;
    TERM) exit 143 ;;
    *) exit 0 ;;
  esac
}

trap 'cleanup INT' INT
trap 'cleanup TERM' TERM

get_latest_mtime() {
  if find "$EXP_DIR" -type f -print -quit | grep -q .; then
    find "$EXP_DIR" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -n1 | cut -d. -f1
  else
    stat -c '%Y' "$EXP_DIR" 2>/dev/null || echo 0
  fi
}

echo "Monitoring EXP_DIR=$EXP_DIR; CHECK_INTERVAL=${CHECK_INTERVAL}s; STALE_TIMEOUT=${STALE_TIMEOUT}s"

while true; do
  echo "Starting: $PYTHON $SCRIPT ${ARGS[*]}"
  $PYTHON "$SCRIPT" "${ARGS[@]}" &
  child=$!
  echo "Launched PID=$child at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  start_time=$(date +%s)
  file_mtime=$(get_latest_mtime)
  
  # 핵심 수정 1: 프로세스 시작 시점과 기존 파일 mtime 중 더 최근 시간을 기준점으로 설정
  if [[ $file_mtime -gt $start_time ]]; then
    last_seen=$file_mtime
  else
    last_seen=$start_time
  fi

  was_killed=0

  while kill -0 "$child" 2>/dev/null; do
    sleep "$CHECK_INTERVAL"
    now=$(date +%s)
    current_mtime=$(get_latest_mtime)
    
    if [[ -z "$current_mtime" ]]; then
      current_mtime=0
    fi

    if [[ $current_mtime -gt $last_seen ]]; then
      last_seen=$current_mtime
      echo "Update detected at $(date -d @${last_seen} -u +%Y-%m-%dT%H:%M:%SZ)"
      continue
    fi

    age=$(( now - last_seen ))
    if [[ $age -ge $STALE_TIMEOUT ]]; then
      echo "No updates for ${age}s (>= ${STALE_TIMEOUT}s). Restarting process PID=$child"
      was_killed=1
      kill_tree "$child"
      break
    fi
  done

  # exit code 확인
  wait "$child" 2>/dev/null
  rc=$?
  echo "Process PID=$child exited with code $rc at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  # 핵심 수정 2: 타임아웃으로 죽인 것이 아니고 정상 완료(0)된 경우 루프 종료
  if [[ $was_killed -eq 0 ]]; then
    if [[ $rc -eq 0 ]]; then
      echo "Simulation completed successfully."
      break
    else
      echo "Simulation crashed with exit code $rc."
      # 필요 시 무한 재시작 대신 에러 발생 시 종료하려면 아래 break 주석 해제
      # break
    fi
  fi

  echo "Restarting in 5s..."
  sleep 5
done