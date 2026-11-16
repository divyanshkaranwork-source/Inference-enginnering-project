"""Command-line interface:  ``inferlab <command>``  (or ``python -m inferlab``)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=float))


def cmd_serve(a) -> None:
    import logging

    import uvicorn

    from inferlab.server.app import create_app
    from inferlab.server.settings import Settings

    for flag, env in (("host", "INFERLAB_HOST"), ("port", "INFERLAB_PORT"), ("model", "INFERLAB_MODEL"),
                      ("backend", "INFERLAB_BACKEND"), ("kernels", "INFERLAB_KERNELS"),
                      ("weight_dtype", "INFERLAB_WEIGHT_DTYPE"), ("upstream_url", "INFERLAB_UPSTREAM_URL")):
        v = getattr(a, flag)
        if v is not None:
            os.environ[env] = str(v)
    if a.no_auth:
        os.environ["INFERLAB_AUTH_DISABLED"] = "true"
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    app = create_app(settings)
    # access_log=False: our RequestLogMiddleware logs method/path/status/latency + request id
    # (never query strings, bodies or headers, so prompts and keys stay out of logs)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=a.log_level, server_header=False,
                proxy_headers=False, access_log=False, timeout_graceful_shutdown=10)


def cmd_download(a) -> None:
    from inferlab import models

    names = list(models.ARTIFACTS) if a.all else (a.names or ["tokenizer.bin", "stories15M.bin"])
    failed = False
    for n in names:
        def progress(done, total, n=n):
            if total:
                sys.stderr.write(f"\r  {n}: {done / 2**20:7.1f} / {total / 2**20:.1f} MiB")
        print(f"downloading {n} ...", file=sys.stderr)
        try:
            p = models.download(n, Path(a.models_dir), force=a.force, progress=progress)
        except models.DownloadError as exc:
            print(f"\n  failed: {exc}", file=sys.stderr)
            failed = True
            continue
        sys.stderr.write("\n")
        print(f"  ok -> {p}")
    if failed:
        raise SystemExit(1)


def cmd_generate(a) -> None:
    from inferlab import models
    from inferlab.foundations.generate import GenerationStats, generate
    from inferlab.foundations.sampling import SamplingParams
    from inferlab.kernels import get_kernels

    model, tok = models.load(a.model, a.tokenizer, a.weight_dtype, get_kernels(a.kernels), Path(a.models_dir))
    sp = SamplingParams(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, max_tokens=a.max_tokens,
                        seed=a.seed)
    stats = GenerationStats()
    print(f"[{model.cfg.name}, {model.nbytes / 2**20:.1f} MiB, kernels={a.kernels}]", file=sys.stderr)
    sys.stdout.write(a.prompt)
    for ev in generate(model, tok, a.prompt, sp, use_kv_cache=not a.no_cache, stats=stats):
        sys.stdout.write(ev.text)
        sys.stdout.flush()
    print()
    d = stats.to_dict()
    print(f"\nTTFT {d['ttft_ms']:.1f} ms | TPOT {d['tpot_ms']:.2f} ms | {d['tokens_per_s']:.1f} tok/s | "
          f"{d['completion_tokens']} tokens ({d['finish_reason']})", file=sys.stderr)


def cmd_bench(a) -> None:
    from inferlab.kernels import bench

    res = bench.BENCHMARKS[a.op]()
    print(f"{res['op']} {res['shape']}  {res['cpp']}")
    for r in res["results"]:
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


def cmd_plan(a) -> None:
    from inferlab.parallel import Plan, evaluate, recommend

    if a.recommend:
        for r in recommend(a.model, a.gpu, a.gpus, a.batch, a.context):
            p, d = r["plan"], r["decode"]
            print(f"dp={p['dp']} tp={p['tp']} pp={p['pp']} ep={p['ep']}: {d['tokens_per_s_total']:.0f} tok/s total, "
                  f"{d['token_latency_ms']:.1f} ms/token, {r['memory_per_gpu_gb']['total']:.1f} GB/GPU")
        return
    _print_json(evaluate(a.model, a.gpu, Plan(a.dp, a.tp, a.pp, a.cp, a.ep), batch_per_replica=a.batch,
                         context_len=a.context))


def cmd_info(a) -> None:
    from inferlab.config import get_model_config
    from inferlab.foundations import perfmodel as pm

    cfg = get_model_config(a.model)
    print(f"{cfg.name}: {cfg.n_params / 1e9:.2f}B params")
    for k, v in cfg.param_breakdown().items():
        print(f"  {k:10s} {v / 1e6:10.1f} M")
    _print_json(pm.memory_report(cfg, a.dtype, a.dtype, a.batch, a.context, a.gpu))
    d = pm.decode_step(cfg, a.gpu, a.batch, a.context, a.dtype)
    print(f"decode @ batch {a.batch}: {d['step_time_ms']:.2f} ms/step, {d['tokens_per_s']:.0f} tok/s, "
          f"AI={d['arithmetic_intensity']:.1f} FLOP/B -> {d['bound']}-bound")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="inferlab", description="Inference engineering lab")
    p.add_argument("--models-dir", default=os.environ.get("INFERLAB_MODELS_DIR", "models"))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the OpenAI-compatible server + web UI")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--model")
    s.add_argument("--backend", choices=["native", "llamacpp", "vllm", "sglang", "tensorrt_llm"])
    s.add_argument("--upstream-url")
    s.add_argument("--kernels", choices=["numpy", "cpp"])
    s.add_argument("--weight-dtype", choices=["fp32", "fp16", "bf16", "int8", "q8_0", "q4_0"])
    s.add_argument("--no-auth", action="store_true", help="disable auth (only allowed on 127.0.0.1)")
    s.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("download", help="download model checkpoints (HTTPS + checksums)")
    d.add_argument("names", nargs="*")
    d.add_argument("--all", action="store_true")
    d.add_argument("--force", action="store_true")
    d.set_defaults(fn=cmd_download)

    g = sub.add_parser("generate", help="generate text with the NumPy engine")
    g.add_argument("prompt")
    g.add_argument("--model", default="auto")
    g.add_argument("--tokenizer")
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed", type=int)
    g.add_argument("--no-cache", action="store_true", help="disable the KV cache (see why it matters)")
    g.add_argument("--kernels", default="numpy", choices=["numpy", "cpp"])
    g.add_argument("--weight-dtype", default="fp32", choices=["fp32", "fp16", "bf16", "int8", "q8_0", "q4_0"])
    g.set_defaults(fn=cmd_generate)

    b = sub.add_parser("bench", help="kernel micro-benchmarks")
    b.add_argument("op", choices=["matmul", "gemv", "attention", "softmax"])
    b.set_defaults(fn=cmd_bench)

    pl = sub.add_parser("plan", help="5D parallelism planner")
    pl.add_argument("--model", default="llama-2-7b")
    pl.add_argument("--gpu", default="A100-SXM-80GB")
    pl.add_argument("--gpus", type=int, default=8)
    for k in ("dp", "tp", "pp", "cp", "ep"):
        pl.add_argument(f"--{k}", type=int, default=1)
    pl.add_argument("--batch", type=int, default=32)
    pl.add_argument("--context", type=int, default=4096)
    pl.add_argument("--recommend", action="store_true")
    pl.set_defaults(fn=cmd_plan)

    i = sub.add_parser("info", help="memory / FLOP math for a reference model")
    i.add_argument("--model", default="llama-2-7b")
    i.add_argument("--gpu", default="A100-SXM-80GB")
    i.add_argument("--dtype", default="fp16")
    i.add_argument("--batch", type=int, default=1)
    i.add_argument("--context", type=int, default=4096)
    i.set_defaults(fn=cmd_info)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
