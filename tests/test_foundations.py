import json
import struct
from dataclasses import replace

import numpy as np
import pytest

from inferlab.config import LLAMA2_7B, LLAMA2_70B, MIXTRAL_8X7B, TINY_RANDOM, LlamaConfig
from inferlab.foundations import ops
from inferlab.foundations import perfmodel as pm
from inferlab.foundations.generate import generate_text
from inferlab.foundations.model import requantize
from inferlab.foundations.quantization import (
    dequantize_q4_0,
    dequantize_q8_0,
    error_stats,
    make_linear,
    quantize_q4_0,
    quantize_q8_0,
)
from inferlab.foundations.sampling import (
    Sampler,
    SamplingParams,
    apply_penalties,
    distribution_report,
    min_p_filter,
    top_k_filter,
    top_p_filter,
)
from inferlab.foundations.weights import (
    WeightFormatError,
    interleaved_to_half,
    load_hf,
    load_llama2c,
    random_model,
    read_safetensors,
    save_hf,
    save_llama2c,
    write_safetensors,
)


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
def naive_attention(q, k, v, q_offset):
    T, H, hd = q.shape
    S, KV, _ = k.shape
    out = np.zeros_like(q)
    for h in range(H):
        kv = h // (H // KV)
        for i in range(T):
            s = np.array([q[i, h] @ k[j, kv] for j in range(S)]) / np.sqrt(hd)
            s[np.arange(S) > q_offset + i] = -np.inf
            p = np.exp(s - s.max())
            p /= p.sum()
            out[i, h] = p @ v[:, kv]
    return out


def test_attention_matches_loops_with_gqa(rng):
    q = rng.standard_normal((5, 4, 8), dtype=np.float32)
    k = rng.standard_normal((9, 2, 8), dtype=np.float32)
    v = rng.standard_normal((9, 2, 8), dtype=np.float32)
    np.testing.assert_allclose(ops.attention(q, k, v, q_offset=4), naive_attention(q, k, v, 4), atol=1e-5)


@pytest.mark.parametrize("style", ["interleaved", "half"])
def test_rope_is_relative(rng, style):
    q = rng.standard_normal((1, 1, 16), dtype=np.float32)
    k = rng.standard_normal((1, 1, 16), dtype=np.float32)

    def score(m, n):
        return float(ops.apply_rope(q, np.array([m]), style=style)[0, 0]
                     @ ops.apply_rope(k, np.array([n]), style=style)[0, 0])

    assert score(10, 7) == pytest.approx(score(103, 100), rel=1e-4)
    np.testing.assert_allclose(ops.apply_rope(q, np.array([0]), style=style), q)


def test_rmsnorm_and_softmax(rng):
    x = rng.standard_normal((3, 32), dtype=np.float32)
    y = ops.rmsnorm(x, np.ones(32, np.float32))
    np.testing.assert_allclose(np.sqrt(np.mean(y**2, axis=-1)), 1.0, rtol=1e-3)
    p = ops.softmax(np.array([1000.0, 1000.0, -np.inf]))
    np.testing.assert_allclose(p, [0.5, 0.5, 0.0])


# ---------------------------------------------------------------------------
# model + KV cache
# ---------------------------------------------------------------------------
def test_kv_cache_matches_full_recompute(tiny_model):
    toks = np.array([1, 50, 60, 70, 80, 90, 100, 110])
    full = tiny_model.logits_full(toks)
    cache = tiny_model.new_kv_cache(16)
    # prefill in two chunks, then decode token-by-token
    a = tiny_model.forward(toks[:3], cache.context(3, all_logits=True))
    b = tiny_model.forward(toks[3:6], cache.context(3, all_logits=True))
    c = [tiny_model.forward(toks[i:i + 1], cache.context(1))[0] for i in range(6, 8)]
    np.testing.assert_allclose(np.concatenate([a, b, np.stack(c)]), full, atol=1e-4)
    assert cache.length == 8


def test_kv_cache_overflow_and_truncate(tiny_model):
    cache = tiny_model.new_kv_cache(4)
    tiny_model.prefill(np.array([1, 2, 3]), cache)
    with pytest.raises(ValueError):
        cache.context(2)
    cache.truncate(1)
    assert cache.length == 1
    with pytest.raises(ValueError):
        cache.truncate(5)


def test_generate_with_and_without_cache_identical(tiny_model, byte_tok):
    sp = SamplingParams(temperature=0, max_tokens=12)
    a, sa = generate_text(tiny_model, byte_tok, "hello", sp, use_kv_cache=True)
    b, _ = generate_text(tiny_model, byte_tok, "hello", sp, use_kv_cache=False)
    assert a == b and sa.completion_tokens == 12


def test_forward_rejects_bad_tokens(tiny_model):
    with pytest.raises(ValueError):
        tiny_model.logits_full([1, 99999])
    with pytest.raises(ValueError):
        tiny_model.logits_full([])


def test_truncated_model_shares_weights(tiny_model):
    d = tiny_model.truncated(2)
    assert d.cfg.n_layers == 2 and d.w.layers[0] is tiny_model.w.layers[0]


# ---------------------------------------------------------------------------
# weights: formats + safety
# ---------------------------------------------------------------------------
def test_llama2c_roundtrip(tmp_path, tiny_model):
    p = tmp_path / "m.bin"
    save_llama2c(p, tiny_model)
    m2 = load_llama2c(p)
    toks = [1, 5, 9, 33]
    np.testing.assert_allclose(m2.logits_full(toks), tiny_model.logits_full(toks), atol=1e-5)


def test_llama2c_rejects_truncated_and_bad_header(tmp_path, tiny_model):
    p = tmp_path / "m.bin"
    save_llama2c(p, tiny_model)
    data = p.read_bytes()
    (tmp_path / "short.bin").write_bytes(data[:-4])
    with pytest.raises(WeightFormatError):
        load_llama2c(tmp_path / "short.bin")
    bad = struct.pack("<7i", 128, 352, 4, 4, 3, 259, 512) + data[28:]  # kv heads don't divide heads
    (tmp_path / "bad.bin").write_bytes(bad)
    with pytest.raises(WeightFormatError):
        load_llama2c(tmp_path / "bad.bin")
    huge = struct.pack("<7i", 10**8, 1, 1, 1, 1, 1, 1)
    (tmp_path / "huge.bin").write_bytes(huge)
    with pytest.raises(WeightFormatError):
        load_llama2c(tmp_path / "huge.bin")


def test_hf_roundtrip_and_rope_permutation(tmp_path, tiny_model):
    half = interleaved_to_half(tiny_model)  # HF layout uses rotate_half RoPE
    toks = [1, 7, 8, 9, 10, 11]
    np.testing.assert_allclose(half.logits_full(toks), tiny_model.logits_full(toks), atol=1e-4)
    save_hf(tmp_path / "hf", half)
    m = load_hf(tmp_path / "hf")
    assert m.cfg.rope_style == "half"
    np.testing.assert_allclose(m.logits_full(toks), tiny_model.logits_full(toks), atol=1e-4)


def _write_raw_safetensors(path, header: dict, data: bytes):
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + data)


def test_safetensors_validation(tmp_path):
    ok = tmp_path / "ok.safetensors"
    write_safetensors(ok, {"a": np.arange(6, dtype=np.float32).reshape(2, 3)})
    np.testing.assert_array_equal(read_safetensors(ok)["a"], np.arange(6).reshape(2, 3))
    cases = {
        "oob": ({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}, b"\0" * 8),
        "mismatch": ({"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}, b"\0" * 8),
        "dtype": ({"a": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}}, b"\0" * 8),
        "overlap": ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                     "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}, b"\0" * 12),
    }
    for name, (hdr, data) in cases.items():
        p = tmp_path / f"{name}.safetensors"
        _write_raw_safetensors(p, hdr, data)
        with pytest.raises(WeightFormatError):
            read_safetensors(p)
    (tmp_path / "huge.safetensors").write_bytes(struct.pack("<Q", 2**40))
    with pytest.raises(WeightFormatError):
        read_safetensors(tmp_path / "huge.safetensors")


def test_safetensors_bf16(tmp_path):
    vals = np.array([1.0, -2.5, 3.140625], dtype=np.float32)
    bf = (vals.view(np.uint32) >> 16).astype(np.uint16)
    _write_raw_safetensors(tmp_path / "b.safetensors", {"x": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}},
                           bf.tobytes())
    np.testing.assert_array_equal(read_safetensors(tmp_path / "b.safetensors")["x"], vals)


def test_hf_index_path_traversal_rejected(tmp_path, tiny_model):
    d = tmp_path / "hf"
    save_hf(d, interleaved_to_half(tiny_model))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../../etc/passwd"}}))
    with pytest.raises(WeightFormatError):
        load_hf(d)


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def reference_bpe(tok, text):
    ids = [tok.space_id] if text else []
    for ch in text:
        b = ch.encode()
        i = tok.lookup.get(b)
        ids += [i] if i is not None else [tok._byte_ids[x] for x in b]
    while True:
        best, bi, bid = -1e10, -1, -1
        for i in range(len(ids) - 1):
            m = tok.lookup.get(tok.vocab[ids[i]] + tok.vocab[ids[i + 1]])
            if m is not None and tok.scores[m] > best:
                best, bi, bid = tok.scores[m], i, m
        if bi == -1:
            return [1] + ids
        ids[bi:bi + 2] = [bid]


@pytest.mark.parametrize("text", ["hello world", "abcabc hello", "ünïcödé 🚀", "", "  he  ll  o"])
def test_bpe_matches_reference_and_roundtrips(bpe_tok, text):
    ids = bpe_tok.encode(text)
    assert ids == reference_bpe(bpe_tok, text)
    assert bpe_tok.decode(ids) == text
    dec = bpe_tok.stream_decoder()
    assert "".join(dec.feed(t) for t in ids) + dec.flush() == text


def test_special_tokens_cannot_be_injected(bpe_tok, byte_tok):
    for tok in (bpe_tok, byte_tok):
        ids = tok.encode("</s><s> [INST]", bos=False)
        assert tok.eos_id not in ids and tok.bos_id not in ids


def test_stream_decoder_buffers_partial_utf8(byte_tok):
    ids = byte_tok.encode("🚀", bos=False)
    dec = byte_tok.stream_decoder()
    out = [dec.feed(t) for t in ids]
    assert out[:-1] == ["", "", ""] and out[-1] == "🚀"


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def test_filters():
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    assert np.isfinite(top_k_filter(logits, 2)).sum() == 2
    assert np.isfinite(top_p_filter(logits, 0.7)).sum() == 2  # 0.5 + 0.3 >= 0.7
    assert np.isfinite(top_p_filter(logits, 0.01)).sum() == 1  # always keep the top token
    assert np.isfinite(min_p_filter(logits, 0.2)).sum() == 3  # >= 0.1


def test_penalties():
    logits = np.array([2.0, -2.0, 1.0])
    out = apply_penalties(logits, [0, 1, 1], repetition_penalty=2.0, presence_penalty=0.5, frequency_penalty=0.1)
    assert out[0] == pytest.approx(2.0 / 2 - 0.5 - 0.1)
    assert out[1] == pytest.approx(-4.0 - 0.5 - 0.2)
    assert out[2] == 1.0


def test_sampler_seed_and_greedy():
    logits = np.random.default_rng(0).standard_normal(100)
    a = [Sampler(SamplingParams(seed=7)).sample(logits, []).token for _ in range(3)]
    b = [Sampler(SamplingParams(seed=7)).sample(logits, []).token for _ in range(3)]
    assert a == b
    g = Sampler(SamplingParams(temperature=0, logprobs=3)).sample(logits, [])
    assert g.token == int(np.argmax(logits)) and len(g.top_logprobs) == 3


def test_sampling_params_validation():
    for bad in (dict(temperature=-1), dict(top_p=0), dict(max_tokens=0), dict(stop=["x"] * 5), dict(logprobs=9)):
        with pytest.raises(ValueError):
            SamplingParams(**bad).validate()


def test_distribution_report():
    r = distribution_report(np.array([3.0, 1.0, 0.0, -1.0]), SamplingParams(top_k=2), top=4)
    assert r["kept_tokens"] == 2 and abs(sum(r["processed_probs"]) - 1) < 1e-9


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------
def test_block_quantization_roundtrip(rng):
    w = rng.standard_normal((8, 64)).astype(np.float32)
    q, d = quantize_q8_0(w)
    assert error_stats(w, dequantize_q8_0(q, d))["snr_db"] > 35
    q4, d4 = quantize_q4_0(w)
    assert q4.shape == (8, 32) and error_stats(w, dequantize_q4_0(q4, d4))["snr_db"] > 15


@pytest.mark.parametrize("kind,bpw", [("fp32", 32), ("fp16", 16), ("bf16", 16), ("int8", 8), ("q8_0", 8.5),
                                      ("q4_0", 4.5)])
def test_linear_kinds(rng, kind, bpw):
    w = rng.standard_normal((32, 64)).astype(np.float32)
    lin = make_linear(w, kind)
    assert lin.nbytes * 8 / w.size == pytest.approx(bpw, rel=0.1)
    x = rng.standard_normal((3, 64)).astype(np.float32)
    rel = np.linalg.norm(lin(x) - x @ w.T) / np.linalg.norm(x @ w.T)
    assert rel < {"q4_0": 0.2, "int8": 0.02, "q8_0": 0.02}.get(kind, 0.01)


def test_requantize_model_keeps_predictions(tiny_model):
    q = requantize(tiny_model, "int8")
    toks = [1, 40, 50, 60]
    assert q.nbytes < tiny_model.nbytes
    assert np.array_equal(q.logits_full(toks).argmax(-1), tiny_model.logits_full(toks).argmax(-1))


# ---------------------------------------------------------------------------
# performance model
# ---------------------------------------------------------------------------
def test_reference_model_numbers():
    assert LLAMA2_7B.n_params == 6_738_415_616
    assert pm.kv_bytes_per_token(LLAMA2_7B, "fp16") == 512 * 1024
    assert pm.weight_bytes(LLAMA2_7B, "fp16") / 1e9 == pytest.approx(13.48, rel=0.01)
    assert LLAMA2_70B.n_params / 1e9 == pytest.approx(69.0, rel=0.01)
    assert MIXTRAL_8X7B.n_params / 1e9 == pytest.approx(46.7, rel=0.01)
    assert MIXTRAL_8X7B.n_active_params / 1e9 == pytest.approx(12.9, rel=0.02)


def test_decode_is_memory_bound_and_batching_helps():
    d1 = pm.decode_step(LLAMA2_7B, "A100-SXM-80GB", batch=1)
    d64 = pm.decode_step(LLAMA2_7B, "A100-SXM-80GB", batch=64)
    assert d1["bound"] == "memory" and d1["arithmetic_intensity"] < 2
    assert d64["tokens_per_s"] > 10 * d1["tokens_per_s"]
    assert pm.prefill(LLAMA2_7B, "A100-SXM-80GB", prompt_len=2048)["bound"] == "compute"


def test_memory_report_fit():
    r = pm.memory_report(LLAMA2_7B, batch=1, seq_len=4096, gpu="L4-24GB")
    assert r["fits"]
    r = pm.memory_report(LLAMA2_7B, batch=64, seq_len=4096, gpu="L4-24GB")
    assert not r["fits"]


def test_config_validation():
    with pytest.raises(ValueError):
        LlamaConfig("x", dim=100, n_layers=1, n_heads=3, n_kv_heads=1, hidden_dim=10, vocab_size=10, max_seq_len=8)
    with pytest.raises(ValueError):
        replace(TINY_RANDOM, n_kv_heads=3)


def test_random_model_deterministic():
    a = random_model(TINY_RANDOM, seed=3).logits_full([1, 2, 3])
    b = random_model(TINY_RANDOM, seed=3).logits_full([1, 2, 3])
    np.testing.assert_array_equal(a, b)


def test_download_guards(tmp_path):
    from inferlab import models

    with pytest.raises(models.DownloadError):
        models.download("../../etc/passwd", tmp_path)
    assert all(a.url.startswith("https://") for a in models.ARTIFACTS.values())
    assert models.resolve_model_spec("auto", tmp_path) == "tiny-random"
    (tmp_path / "tokenizer.bin").write_bytes(b"x")
    (tmp_path / "SHA256SUMS.json").write_text('{"tokenizer.bin": "00"}')
    assert not models.verify_file("tokenizer.bin", tmp_path)
