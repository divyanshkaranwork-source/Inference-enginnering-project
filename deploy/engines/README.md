# Serving Llama-2-7B with real engines (InferLab gateway in front)

Each compose file starts one production engine with **Llama-2-7B-chat**. It also starts the InferLab
gateway in proxy mode, so you keep the same auth, rate limiting, metrics and UI. The engine is never
exposed directly; only the gateway publishes a port, on loopback.

| File | Engine | Hardware | Weights |
|------|--------|----------|---------|
| `llamacpp.yml` | llama.cpp `llama-server` | CPU / Apple / any GPU | GGUF `llama-2-7b-chat.Q4_K_M.gguf` (~4 GB) |
| `vllm.yml` | vLLM | NVIDIA ≥ 16 GB (or AMD ROCm image) | HF `meta-llama/Llama-2-7b-chat-hf` (13.5 GB fp16) |
| `sglang.yml` | SGLang | NVIDIA ≥ 16 GB | same HF checkpoint |
| `tensorrt-llm.yml` | TensorRT-LLM `trtllm-serve` | NVIDIA ≥ 16 GB | same HF checkpoint |

```bash
make secret                                  # gateway API key -> secrets/api_keys.txt
echo "hf_xxx" > secrets/hf_token.txt         # Llama-2 is gated: accept the licence on huggingface.co first
docker compose -f deploy/engines/vllm.yml up
curl http://127.0.0.1:9000/v1/chat/completions -H "Authorization: Bearer $(cat secrets/api_keys.txt)" \
  -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Things to know:

- **Pin images.** The tags are defaults you can override (`VLLM_IMAGE=...`). For production, pin
  them by digest (`image@sha256:...`) after checking each project's release notes.
- **Secrets** are passed as compose secrets (files), not environment variables, and they are not
  baked into images. The HF token is only needed to download gated weights.
- **Model cache.** `./hf-cache` is mounted so weights download only once.
- Compare the engines with `python benchmarks/load_test.py --url http://127.0.0.1:9000 --concurrency 16`.
