# LLM Inference Benchmarking for Chat 

### Set up vllm server 

```
docker run -p 8080:8080 --gpus all vllm/vllm-openai --model Nitral-AI/Captain-Eris_Violet-V0.420-12B --max-model-len 10000 --swap-space 4 --dtype auto --enable-chunked-prefill --disable-log-requests --enable-prefix-caching --port 8080 --root-path /api --served-model-name Nitral-AI/Captain-Eris_Violet-V0.420-12B --max-num-seqs 72 --quantization fp8 --max-num-batched-tokens 1024
```

----------------------

### Set up synthetic benchmark 

Gives you the option to use a source prompt at the start, and add some random text after it to control the prompt cache percentage. 

```
git clone https://github.com/FlowGPT/llm-inference-benchmarking

cd llm-inference-bench-char

pip install -r requirements.txt

python run.py --rounds 1 -q 0.5 --api-base http://localhost:8080/api/v1 --model mn-starcannon-13b  --max-tokens=250 --prompt-file prompt-1k.txt --random-tokens 3000 --use-chat
```
This benchmark runs 0.5 rps on a 13b model with an input of 4.5k tokens and an output of 250 tokens, prefix cache rate 20%. 

source: https://github.com/leptonai/leptonai/blob/main/misc/benchmark/run.py

-----------------------

### Online replay

You can use the online data to replay the production environments.

The online_replay.py script allows you to replay requests from log file with two different modes:

1. Timestamp-based replay (maintains original request timing):
```bash
python online_replay.py --input replay-logs-origin.log --replay-mode timestamp --sample-rate 0.1 \
    --api-base http://localhost:8080/api/v1 --model Nitral-AI/Captain-Eris_Violet-V0.420-12B --round-duration 60
```

2. QPS-based replay (controls request rate):
```bash
python online_replay.py --input replay-logs-origin.log --replay-mode qps --target-qps 5 --sample-rate 0.1 \
    --api-base http://localhost:8080/api/v1 --model Nitral-AI/Captain-Eris_Violet-V0.420-12B --round-duration 60
```

- How to choose between these two modes?

    In multi-instance scenarios, the timestamp mode is recommended. In single-instance scenarios, the qps mode is sufficient. This is because in production environments, requests are routed among multiple instances, making the request arrival pattern appear uniform for each individual instance.

- Despite similar final QPS, why does timestamp mode show higher latency?

    This is caused by non-uniform request arrival patterns. According to queueing theory, when requests arrive non-uniformly (e.g., Poisson process with high variance), bursty requests can lead to temporary queue buildup, significantly increasing the average queuing time.

Key parameters:
- `--input`: Input log file path
- `--replay-mode`: Replay mode (timestamp/qps)
- `--sample-rate`: Sampling rate (0.0-1.0), controls the proportion of requests to replay
- `--round-duration`: Performance statistics collection period (seconds)
- `--api-base`: API service endpoint
- `--model`: Model name
- `--use-chat`: Whether to use chat interface
- `--json-output`: Output performance metrics in JSON format
- `--verbose`: Enable detailed logging output (default: False, only show statistics)


Performance metrics:
- Latency statistics
- Throughput
- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- Input/Output Tokens per Minute
- Success Rate

