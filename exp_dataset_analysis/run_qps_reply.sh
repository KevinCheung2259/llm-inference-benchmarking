
cv=0.0
target_qps=2
preload_time=20 # 等待数据加载时间，根据各服务器性能而异
# dataset=shibing624/sharegpt_gpt4
# dataset=ajibawa-2023/Python-Code-23k-ShareGPT
dataset=simplescaling/s1K

python online_replay_sharedgpt.py \
  --replay-mode qps \
  --target-qps $target_qps \
  --sample-range 0.0 0.1 \
  --api-base http://localhost:8080/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --round-duration 5 \
  --cv $cv \
  --preload-time $preload_time \
  --dataset $dataset \
  --max-tokens 2000 \

