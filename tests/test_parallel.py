import numpy as np
import pytest

from inferlab.foundations import ops
from inferlab.foundations.sampling import SamplingParams
from inferlab.parallel import (
    MoELayer,
    Plan,
    SimCluster,
    ep_forward,
    evaluate,
    pp_forward,
    recommend,
    ring_attention,
    simulate_dp,
    tp_forward,
    ulysses_attention,
)
from inferlab.parallel.data_parallel import make_workload
from inferlab.parallel.pipeline_parallel import split_layers

F32 = np.float32


def test_collective_costs():
    cl = SimCluster(4)
    xs = [np.full(256, i, F32) for i in range(4)]  # 1 KiB each
    out = cl.all_reduce(xs)
    assert np.all(out[2] == 6)
    s = cl.stats()["by_op"]["all_reduce"]
    assert s["bytes_per_rank"] == pytest.approx(2 * 3 / 4 * 1024)
    rs = cl.reduce_scatter(xs)
    assert len(rs) == 4 and rs[0].shape == (64,)


@pytest.mark.parametrize("tp,sp", [(1, False), (2, False), (2, True)])
def test_tensor_parallel_exact(tiny_model, tp, sp):
    toks = list(range(1, 25))
    out, cl = tp_forward(tiny_model, toks, tp, sequence_parallel=sp)
    np.testing.assert_allclose(out, tiny_model.logits_full(toks), atol=1e-3)
    if tp > 1:
        ops_used = cl.stats()["by_op"]
        assert ("reduce_scatter" in ops_used) == sp
        assert ops_used["all_gather"]["count"] >= 1


def test_tensor_parallel_rejects_bad_degree(tiny_model):
    with pytest.raises(ValueError):
        tp_forward(tiny_model, [1, 2, 3], 3)


def test_pipeline_parallel_exact(tiny_model, byte_tok):
    seqs = [byte_tok.encode(s) for s in ("hello", "pipeline parallel", "a", "four stages")]
    outs, sched, cl = pp_forward(tiny_model, seqs, pp=4, num_microbatches=4)
    for o, s in zip(outs, seqs, strict=True):
        np.testing.assert_allclose(o, tiny_model.logits_full(s), atol=1e-4)
    assert len(sched["events"]) == 16 and sched["ideal_bubble_fraction"] == pytest.approx(3 / 7)
    assert cl.stats()["by_op"]["send_recv"]["count"] == 3 * 4
    assert [list(r) for r in split_layers(7, 3)] == [[0, 1, 2], [3, 4], [5, 6]]


@pytest.mark.parametrize("scheme", ["contiguous", "zigzag"])
def test_ring_attention_exact(rng, scheme):
    q = rng.standard_normal((48, 4, 16), dtype=F32)
    k = rng.standard_normal((48, 2, 16), dtype=F32)
    v = rng.standard_normal((48, 2, 16), dtype=F32)
    out, st = ring_attention(q, k, v, 4, scheme)
    np.testing.assert_allclose(out, ops.attention(q, k, v, 0, True), atol=1e-5)
    if scheme == "zigzag":
        assert st["load_imbalance"] < 1.1
    else:
        assert st["load_imbalance"] > 1.3


def test_ulysses_exact(rng):
    q = rng.standard_normal((32, 4, 8), dtype=F32)
    k = rng.standard_normal((32, 2, 8), dtype=F32)
    v = rng.standard_normal((32, 2, 8), dtype=F32)
    out, st = ulysses_attention(q, k, v, 2)
    np.testing.assert_allclose(out, ops.attention(q, k, v, 0, True), atol=1e-5)
    assert st["comm"]["by_op"]["all_to_all"]["count"] == 4


def test_expert_parallel_exact_and_capacity(rng):
    layer = MoELayer.random(d=32, f=64, n_experts=8, top_k=2)
    x = rng.standard_normal((64, 32), dtype=F32)
    out, st = ep_forward(layer, x, 4)
    np.testing.assert_allclose(out, layer.forward(x), atol=1e-5)
    assert sum(st["tokens_per_expert"]) == 64 * 2
    out2, st2 = ep_forward(layer, x, 4, capacity_factor=0.5)
    assert st2["dropped_assignments"] > 0 and not np.allclose(out2, layer.forward(x))


def test_data_parallel_prefix_aware_routing(tiny_model, byte_tok):
    w = make_workload(3, 4)
    sp = SamplingParams(temperature=0, max_tokens=4)
    rr = simulate_dp(tiny_model, byte_tok, w, 2, "round_robin", sp)
    pa = simulate_dp(tiny_model, byte_tok, w, 2, "prefix_aware", sp)
    assert pa["prefix_hit_tokens"] >= rr["prefix_hit_tokens"]
    assert sum(r["requests"] for r in pa["replicas"]) == len(w)


def test_planner():
    one = evaluate("llama-2-7b", "A100-SXM-80GB", Plan(), batch_per_replica=8)
    tp2 = evaluate("llama-2-7b", "A100-SXM-80GB", Plan(tp=2), batch_per_replica=8)
    assert one["fits"] and tp2["memory_per_gpu_gb"]["weights"] == pytest.approx(one["memory_per_gpu_gb"]["weights"] / 2)
    assert tp2["decode"]["token_latency_ms"] < one["decode"]["token_latency_ms"]
    big = evaluate("llama-2-70b", "A100-SXM-80GB", Plan())
    assert not big["fits"]
    bad = evaluate("llama-2-7b", "A100-SXM-80GB", Plan(tp=3))
    assert not bad["valid"]
    assert not evaluate("llama-2-7b", "A100-SXM-80GB", Plan(ep=2))["valid"]  # dense model
    recs = recommend("llama-2-70b", "H100-SXM-80GB", 8)
    assert recs and all(r["fits"] for r in recs)
    tp8 = evaluate("llama-2-70b", "H100-SXM-80GB", Plan(tp=8))
    pp8 = evaluate("llama-2-70b", "H100-SXM-80GB", Plan(pp=8))
    assert tp8["decode"]["token_latency_ms"] < pp8["decode"]["token_latency_ms"]
