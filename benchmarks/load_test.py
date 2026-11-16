"""Load generator for any OpenAI-compatible endpoint (InferLab, vLLM, SGLang, llama.cpp, TRT-LLM).

    export INFERLAB_API_KEY=...            # read from the environment, never from argv (shell history)
    python benchmarks/load_test.py --url http://127.0.0.1:8000 --requests 64 --concurrency 8 --max-tokens 64

Reports TTFT / TPOT / end-to-end percentiles and throughput - the numbers you
compare engines and configurations by.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from urllib.parse import urlsplit

import httpx
import numpy as np

PROMPTS = [
    "Once upon a time, there was a little dog named Max.",
    "Write a short story about a robot who learns to paint.",
    "Lily found a shiny stone in the garden. She",
    "Explain why the sky is blue to a five-year-old.",
]


async def one(client: httpx.AsyncClient, args, i: int) -> dict:
    body = {"prompt": PROMPTS[i % len(PROMPTS)], "max_tokens": args.max_tokens, "temperature": args.temperature,
            "stream": True}
    if args.model:
        body["model"] = args.model
    t0 = time.perf_counter()
    ttft, n, last = None, 0, t0
    gaps = []
    async with client.stream("POST", "/v1/completions", json=body) as r:
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code}
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if not ev.get("choices") or not ev["choices"][0].get("text"):
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            else:
                gaps.append(now - last)
            last = now
            n += 1
    return {"ok": True, "ttft": ttft, "tpot": float(np.mean(gaps)) if gaps else None, "e2e": last - t0, "tokens": n}


async def main(args) -> None:
    parts = urlsplit(args.url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SystemExit("--url must be http(s)://host:port")
    headers = {}
    key = os.environ.get("INFERLAB_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=httpx.Timeout(300, connect=10)) as client:
        async def guarded(i):
            async with sem:
                return await one(client, args, i)

        t0 = time.perf_counter()
        res = await asyncio.gather(*(guarded(i) for i in range(args.requests)))
        wall = time.perf_counter() - t0
    ok = [r for r in res if r["ok"]]
    print(f"requests: {len(ok)}/{len(res)} ok, concurrency {args.concurrency}, wall {wall:.1f}s")
    if len(ok) < len(res):
        print("  failures by status:", {r["status"] for r in res if not r["ok"]})
    if not ok:
        return
    toks = sum(r["tokens"] for r in ok)
    print(f"throughput: {toks / wall:.1f} output tok/s, {len(ok) / wall:.2f} req/s")
    for key_ in ("ttft", "tpot", "e2e"):
        v = np.array([r[key_] for r in ok if r[key_] is not None]) * 1e3
        if v.size:
            print(f"{key_:5} ms  p50 {np.percentile(v, 50):8.1f}  p90 {np.percentile(v, 90):8.1f}  "
                  f"p99 {np.percentile(v, 99):8.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    asyncio.run(main(ap.parse_args()))
