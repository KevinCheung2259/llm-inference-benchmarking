#!/bin/bash

# --- Configuration ---
INPUT_LOG="/mnt/shared/data/replay-logs-origin.log" # Change to your log file path
API_BASE="http://192.168.49.2:32108/v1"           # Change to your API base URL
MODEL_NAME="Nitral-AI/Captain-Eris_Violet-V0.420-12B" # Change to your model
API_KEY="your_api_key_here"                     # Change to your API key
MAX_TOKENS=180
USE_CHAT=false      # Set to true if using chat endpoint
VERBOSE=false        # Set to true for detailed logging

# Target QPS per client (adjust if using qps mode)
TARGET_QPS=13
# Replay mode: timestamp, qps
REPLAY_MODE="qps"

# --- Common Arguments ---
# Construct common arguments, adjust based on your needs and the replay mode
COMMON_ARGS="--input $INPUT_LOG --api-base $API_BASE --model $MODEL_NAME --api-key $API_KEY --max-tokens $MAX_TOKENS"

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
echo "Starting process for sample..."
python3 online_replay.py \
  $COMMON_ARGS \
  --sample-range 0.0 0.25 & 
P1_PID=$!
echo "Process 1 started with PID: $P1_PID"

sleep 1

echo "Starting process for sample..."
python3 online_replay.py \
  $COMMON_ARGS \
  --sample-range 0.5 0.75 & 
P2_PID=$!
echo "Process 2 started with PID: $P2_PID"

# 将后台进程放入新的进程组
pgid=$(ps -o pgid= $P1_PID | grep -o [0-9]*)

# --- 等待进程完成 ---
echo "Waiting for processes to complete... (Press Ctrl+C to stop)"
trap "kill -- -$pgid" EXIT  # 当脚本退出时杀死整个进程组
wait $P1_PID $P2_PID

echo "All processes completed."

