#!/bin/bash

# --- Configuration ---
INPUT_LOG="/mnt/shared/data/replay-logs-origin.log" # Change to your log file path
API_BASE="http://localhost:8769/v1"           # Change to your API base URL
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" # Change to your model
API_KEY="your_api_key_here"                     # Change to your API key
MAX_ROUNDS=10
MAX_TOKENS=200
USE_CHAT=false      # Set to true if using chat endpoint
VERBOSE=false        # Set to true for detailed logging

# Target QPS per client (adjust if using qps mode)
TARGET_QPS=10
# Replay mode: timestamp, qps
REPLAY_MODE="timestamp"

# SLO
E2E_SLO=5
TTFT_SLO=1000
TPOT_SLO=50

# Output
OUTPUT_DIR="exp/sla_elrar_30.json"

# --- Common Arguments ---
# Construct common arguments, adjust based on your needs and the replay mode
COMMON_ARGS="--input $INPUT_LOG --api-base $API_BASE --model $MODEL_NAME 
--api-key $API_KEY --max-tokens $MAX_TOKENS --max-rounds $MAX_ROUNDS 
--e2e-slo $E2E_SLO --ttft-slo $TTFT_SLO --tpot-slo $TPOT_SLO --preload-time 10 --json-output $OUTPUT_DIR"

if $USE_CHAT; then
  COMMON_ARGS="$COMMON_ARGS --use-chat"
fi

if $VERBOSE; then
  COMMON_ARGS="$COMMON_ARGS --verbose"
fi

# Add mode-specific arguments
if [ "$REPLAY_MODE" == "qps" ]; then
  COMMON_ARGS="$COMMON_ARGS --replay-mode qps --target-qps $TARGET_QPS"
elif [ "$REPLAY_MODE" == "timestamp" ]; then
  COMMON_ARGS="$COMMON_ARGS --replay-mode timestamp"
fi

# --- Launch Processes ---

# 设置进程组，这样Ctrl+C会终止整个组
set -m

# --- 启动进程 ---
# 冷启动阶段
echo "Starting process for sample..."
python3 online_replay.py \
  $COMMON_ARGS \
  --sample-range 0.0 0.20 & 
P1_PID=$!
echo "Process 1 started with PID: $P1_PID"

sleep 60

echo "Starting process for sample..."
python3 online_replay.py \
  $COMMON_ARGS \
  --sample-range 0.20 0.40 & 
P2_PID=$!
echo "Process 2 started with PID: $P2_PID"

sleep 60

echo "Starting process for sample..."
python3 online_replay.py \
  $COMMON_ARGS \
  --sample-range 0.40 0.60 & 
P3_PID=$!
echo "Process 3 started with PID: $P3_PID"

# 将后台进程放入新的进程组
pgid1=$(ps -o pgid= $P1_PID | grep -o [0-9]*)
pgid2=$(ps -o pgid= $P2_PID | grep -o [0-9]*)
pgid3=$(ps -o pgid= $P3_PID | grep -o [0-9]*)
pgid4=$(ps -o pgid= $P4_PID | grep -o [0-9]*)


# --- 等待进程完成 ---
echo "Waiting for processes to complete... (Press Ctrl+C to stop)"
trap "kill -- -$pgid1 $pgid2 $pgid3 $pgid4" EXIT  # 当脚本退出时杀死整个进程组
wait $P1_PID $P2_PID $P3_PID $P4_PID

echo "All processes completed."

