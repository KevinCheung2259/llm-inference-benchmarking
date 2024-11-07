# LLM Inference Benchmarking for Chat 

Gives you the option to use a source prompt at the start, and add some random text after it to control the prompt cache percentage. 


### Set up server 

```
docker run -p 8080:8080 --gpus all vllm/vllm-openai --model nothingiisreal/MN-12B-Starcannon-v3 --max-model-len 10000 --swap-space 4 --dtype auto --enable-chunked-prefill --disable-log-requests --enable-prefix-caching --port 8080 --root-path /api --served-model-name mn-starcannon-13b --max-num-seqs 24
```

### Set up benchmark 

```
git clone https://github.com/AlexXi19/llm-inference-bench-char
cd llm-inference-bench-char
pip install -r requirements.txt
python run.py --rounds 1 -q 0.5 --api-base http://localhost:8080/api/v1 --model mn-starcannon-13b  --max-tokens=250 --prompt-file prompt-1k.txt --random-tokens 3000 --use-chat
```
This benchmark runs 0.5 rps on a 13b model with an input of 4.5k tokens and an output of 250 tokens, prefix cache rate 20%. 

source: https://github.com/leptonai/leptonai/blob/main/misc/benchmark/run.py

