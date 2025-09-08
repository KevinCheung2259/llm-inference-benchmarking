#!/bin/bash
set -euo pipefail

# --- Configuration ---
INPUT_LOG="/home/ubuntu/replay-logs-origin.log"   # Change to your log file path
API_BASE="http://localhost:8888/v1"                   # Change to your API base URL
MODEL_NAME="Qwen/QwQ-32B"         # Change to your model
API_KEY="your_api_key_here"                           # Change to your API key
MAX_ROUNDS=5
MAX_TOKENS=200
USE_CHAT=false     # Set to true if using chat endpoint
VERBOSE=false      # Set to true for detailed logging

# Replay configuration
DATASET="flowgpt"
TARGET_QPS=10                  # only used when REPLAY_MODE="qps"
REPLAY_MODE="timestamp"        # "timestamp" | "qps"
LOWER_BOUND=0.0                # only used when REPLAY_MODE="timestamp"
UPPER_BOUND=0.12                # only used when REPLAY_MODE="timestamp"

# SLO
E2E_SLO=5
TTFT_SLO=1000
TPOT_SLO=50

# EXP Scheduler Config
ENGINE=sla
ROUTER=qps

# Output (per-process会自动加后缀，避免覆盖)
OUT_DIR_ROOT="exp_qwen32b/${DATASET}_${REPLAY_MODE}_${ENGINE}_${ROUTER}_${LOWER_BOUND}_${UPPER_BOUND}"
OUT_PREFIX="${OUT_DIR_ROOT}/${DATASET}_${REPLAY_MODE}_${ENGINE}_${ROUTER}"

# --- Common Arguments ---
COMMON_ARGS="--input $INPUT_LOG --api-base $API_BASE --model $MODEL_NAME \
--api-key $API_KEY --max-tokens $MAX_TOKENS --max-rounds $MAX_ROUNDS \
--e2e-slo $E2E_SLO --ttft-slo $TTFT_SLO --tpot-slo $TPOT_SLO --preload-time 10"

if $USE_CHAT; then
  COMMON_ARGS="$COMMON_ARGS --use-chat"
fi

if $VERBOSE; then
  COMMON_ARGS="$COMMON_ARGS --verbose"
fi

# mode-specific common args
if [ "$REPLAY_MODE" = "qps" ]; then
  COMMON_ARGS="$COMMON_ARGS --replay-mode qps"
elif [ "$REPLAY_MODE" = "timestamp" ]; then
  COMMON_ARGS="$COMMON_ARGS --replay-mode timestamp"
else
  echo "Unsupported REPLAY_MODE: $REPLAY_MODE"
  exit 1
fi

# --- Helpers ---
PIDS=()

start_process_timestamp() {
  local start="$1"
  local end="$2"
  local tag
  tag=$(printf "s%.1f_%.1f" "$start" "$end")
  local json_out="${OUT_PREFIX}_${tag}.json"
  local csv_out="${OUT_PREFIX}_${tag}.csv"
  mkdir -p "$(dirname "$json_out")"

  echo "[timestamp] Launching sample-range ${start} ${end} ..."
  set -x
  python3 online_replay.py \
    $COMMON_ARGS \
    --json-output "$json_out" \
    --detailed-logs "$csv_out" \
    --sample-range "$start" "$end" &
  set +x
  PIDS+=("$!")
}

start_process_qps() {
  local qps="$1"
  local idx="$2"
  local tag="qps${qps}_i${idx}"
  local json_out="${OUT_PREFIX}_${tag}.json"
  local csv_out="${OUT_PREFIX}_${tag}.csv"
  mkdir -p "$(dirname "$json_out")"

  echo "[qps] Launching target-qps ${qps} (proc #${idx}) ..."
  set -x
  python3 online_replay.py \
    $COMMON_ARGS \
    --json-output "$json_out" \
    --detailed-logs "$csv_out" \
    --target-qps "$qps" &
  set +x
  PIDS+=("$!")
}

cleanup() {
  # echo "Stopping all started processes..."
  # 向当前进程组内所有子进程发送SIGTERM
  kill 0 >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

# --- Launch ---
echo "Mode: $REPLAY_MODE"
echo "Output dir: $OUT_DIR_ROOT"

if [ "$REPLAY_MODE" = "timestamp" ]; then
  # 每 0.1 的范围启动一个进程；相邻进程相隔 10s
  # 例如 [0.0, 0.5) -> [0.0,0.1], [0.1,0.2], ..., [0.4,0.5]
  # 使用 seq 生成起点序列（不包含 UPPER_BOUND）
  # 防止浮点误差，end 边界向下偏移一个极小量
  END_FOR_SEQ=$(awk -v ub="$UPPER_BOUND" 'BEGIN{printf "%.1f", ub-0.0001}')
  for s in $(seq -f "%.1f" "$LOWER_BOUND" 0.1 "$END_FOR_SEQ"); do
    e=$(awk -v ss="$s" -v ub="$UPPER_BOUND" 'BEGIN{e=ss+0.1; if (e>ub) e=ub; printf "%.1f", e}')
    start_process_timestamp "$s" "$e"
    sleep 10
  done
elif [ "$REPLAY_MODE" = "qps" ]; then
  # 根据每 5 qps 启动一个进程；相隔 10s
  # 例如 TARGET_QPS=23 -> 启动: 5,5,5,5,3
  if [ "$TARGET_QPS" -le 0 ]; then
    echo "TARGET_QPS must be > 0 for qps mode."
    exit 1
  fi
  step=5
  remaining="$TARGET_QPS"
  idx=1
  while [ "$remaining" -gt 0 ]; do
    qps_chunk=$step
    if [ "$remaining" -lt "$step" ]; then
      qps_chunk="$remaining"
    fi
    start_process_qps "$qps_chunk" "$idx"
    remaining=$((remaining - qps_chunk))
    idx=$((idx + 1))
    [ "$remaining" -gt 0 ] && sleep 10
  done
fi

echo "Waiting for processes to complete... (Ctrl+C to stop)"
# 等待所有子进程
if [ "${#PIDS[@]}" -gt 0 ]; then
  wait "${PIDS[@]}"
fi

echo "All processes completed."
