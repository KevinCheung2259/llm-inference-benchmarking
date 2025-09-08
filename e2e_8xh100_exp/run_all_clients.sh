#!/usr/bin/env bash
set -o pipefail

base="/home/ubuntu/zhangy/llm-inference-benchmarking"
scripts=(
  "run_client_split_.sh"
  "run_client_split.sh"
  "run_client_split_1.sh"
  "run_client_split_2.sh"
  "run_client_split_3.sh"
  "run_client_split_4_.sh"
  "run_client_split_4.sh"
  "run_client_split_5.sh"
  "run_client_split_6.sh"
  "run_client_split_7.sh"
)

ts(){ date '+%F %T'; }

run_one() {
  local s="$1"
  echo "[$(ts)] START $s"
  # 若子脚本可能“常驻不退出”，给它加个超时（示例 2 小时），不需要可去掉 timeout
  timeout 2h bash "$base/$s"
  local rc=$?
  echo "[$(ts)] END   $s (rc=$rc)"
  return $rc
}

for s in "${scripts[@]}"; do
  if [ ! -f "$base/$s" ]; then
    echo "WARN: missing $base/$s，跳过"
    continue
  fi

  # 即便前一个失败也继续
  run_one "$s" || echo "WARN: $s 非零退出，继续下一个"

  # 间隔 30s
  sleep 30
done

echo "[$(ts)] ALL DONE"

