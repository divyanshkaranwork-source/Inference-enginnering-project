import asyncio
import json

import httpx
import numpy as np
import pytest

from inferlab.foundations.generate import generate_text
from inferlab.foundations.sampling import SamplingParams
from inferlab.runtime.async_engine import AsyncEngine
from inferlab.runtime.backends import GenRequest, OpenAICompatBackend
from inferlab.runtime.backends.base import BackendBadRequest, BackendOverloaded
from inferlab.runtime.backends.openai_compat import validate_upstream_url
from inferlab.runtime.block_manager import BlockManager, hash_block
from inferlab.runtime.chat import apply_chat_template
from inferlab.runtime.engine import EngineConfig, LLMEngine
from inferlab.runtime.radix_cache import RadixCache
from inferlab.runtime.sequence import Sequence
from inferlab.runtime.speculative import ngram_propose, speculative_generate, verify_draft

PROMPTS = ["Once upon a time there was", "The quick brown fox jumps over the lazy dog. " * 3, "Hi",
           "Shared prefix: the system prompt says be helpful. Question 1",
           "Shared prefix: the system prompt says be helpful. Question 2"]
GREEDY = SamplingParams(temperature=0, max_tokens=24)


@pytest.fixture(scope="module")
def reference(tiny_model, byte_tok):
    return [generate_text(tiny_model, byte_tok, p, GREEDY)[0] for p in PROMPTS]


@pytest.mark.parametrize("cfg", [
    EngineConfig(),
    EngineConfig(num_kv_blocks=24, block_size=8, max_num_batched_tokens=16),  # forces chunking + preemption
    EngineConfig(policy="static", max_num_seqs=2),
    EngineConfig(enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=512),
    EngineConfig(block_size=1, max_num_seqs=1),
])
def test_engine_matches_single_sequence_reference(tiny_model, byte_tok, reference, cfg):
    eng = LLMEngine(tiny_model, byte_tok, cfg)
    res = eng.generate(PROMPTS, GREEDY)
    assert [r["text"] for r in res] == reference
    assert eng.bm.num_free_blocks == cfg.num_kv_blocks  # no leaked blocks
    assert all(rc == 0 for rc in eng.bm.ref_count)


def test_preemption_happens_under_memory_pressure(tiny_model, byte_tok, reference):
    eng = LLMEngine(tiny_model, byte_tok, EngineConfig(num_kv_blocks=24, block_size=8, max_num_batched_tokens=16))
    eng.generate(PROMPTS, GREEDY)
    assert eng.totals["preemptions"] > 0


def test_prefix_cache_reuse_and_salt_isolation(tiny_model, byte_tok, reference):
    eng = LLMEngine(tiny_model, byte_tok, EngineConfig(block_size=8))
    eng.generate([PROMPTS[3]], GREEDY)
    s = eng.add_request(PROMPTS[4], GREEDY)
    while eng.has_unfinished():
        eng.step()
    assert s.metrics.num_cached_prompt_tokens >= 40
    assert byte_tok.decode(s.output_tokens) == reference[4]
    other = eng.add_request(PROMPTS[4], GREEDY, cache_salt="tenant-b")
    while eng.has_unfinished():
        eng.step()
    assert other.metrics.num_cached_prompt_tokens == 0  # different tenant: no sharing


def test_hash_chain_depends_on_prefix_and_salt():
    a = hash_block(None, [1, 2, 3])
    assert hash_block(a, [4]) != hash_block(hash_block(None, [9, 2, 3]), [4])
    assert hash_block(None, [1, 2, 3], "x") != a


def test_abort_frees_blocks(tiny_model, byte_tok):
    eng = LLMEngine(tiny_model, byte_tok, EngineConfig(num_kv_blocks=32, block_size=8))
    s = eng.add_request("a long prompt " * 4, SamplingParams(max_tokens=50))
    eng.step()
    assert eng.bm.num_free_blocks < 32
    assert eng.abort(s.seq_id)
    assert eng.bm.num_free_blocks == 32 and not eng.has_unfinished()


def test_stop_strings_hold_back_partial_matches(tiny_model, byte_tok, reference):
    target = reference[0]
    stop = target[6:9]
    eng = LLMEngine(tiny_model, byte_tok)
    seq = eng.add_request(PROMPTS[0], SamplingParams(temperature=0, max_tokens=24, stop=[stop]))
    streamed = ""
    while eng.has_unfinished():
        for out in eng.step():
            streamed += out.text
    assert streamed == target[: target.find(stop)]
    assert seq.status.value == "stop"


def test_request_validation(tiny_model, byte_tok):
    eng = LLMEngine(tiny_model, byte_tok, EngineConfig(num_kv_blocks=4, block_size=4))
    with pytest.raises(ValueError):
        eng.add_request("x" * 600)  # longer than context
    with pytest.raises(ValueError):
        eng.add_request("hello", SamplingParams(max_tokens=100))  # can never fit in 16 KV slots
    with pytest.raises(ValueError):
        eng.add_request([1, 99999])


def test_guided_choice(tiny_model, byte_tok):
    eng = LLMEngine(tiny_model, byte_tok)
    res = []
    for seed in range(3):
        s = eng.add_request("Answer:", SamplingParams(temperature=1.0, max_tokens=20, seed=seed),
                            guided_choice=["yes", "no", "maybe"])
        while eng.has_unfinished():
            eng.step()
        res.append(s.output_text)
    assert all(r in ("yes", "no", "maybe") for r in res)


def test_block_manager_lru_eviction():
    bm = BlockManager(num_blocks=4, block_size=2)

    class S:  # minimal sequence stand-in
        def __init__(self, toks):
            self.all_tokens, self.prompt_tokens = toks, toks
            self.num_computed_tokens, self.block_table, self.block_hashes, self.cache_salt = 0, [], [], None

    a = S([1, 2, 3, 4, 5])
    assert bm.allocate_slots(a, 5)
    a.num_computed_tokens = 5
    bm.cache_full_blocks(a)
    bm.free(a)
    assert len(bm.cached) == 2 and bm.num_free_blocks == 4
    b = S([1, 2, 3, 4, 9])
    blocks, n = bm.get_computed_blocks(b)
    assert n == 4 and len(blocks) == 2  # both full blocks of the shared prefix are reused
    c = S(list(range(100, 108)))
    assert bm.allocate_slots(c, 8)  # needs all 4 blocks -> evicts cached ones
    assert bm.evictions == 2 and not bm.cached


# ---------------------------------------------------------------------------
def test_radix_cache():
    rc = RadixCache(capacity_tokens=12)
    assert rc.insert([1, 2, 3, 4]) == 0
    assert rc.insert([1, 2, 3, 9]) == 3
    assert rc.match_prefix([1, 2, 3, 4, 5])[0] == 4
    rc.insert([7, 7, 7, 7, 7, 7, 7, 7])  # 4 + 1 + 8 = 13 > 12 -> LRU eviction
    assert rc.total_tokens <= 12 and rc.evicted_tokens > 0


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method,draft_layers", [("ngram", None), ("draft_model", 2), ("draft_model", 4)])
def test_speculative_greedy_is_exact(tiny_model, byte_tok, method, draft_layers):
    sp = SamplingParams(temperature=0, max_tokens=40)
    for prompt in ("abcabcabcabc", "Once upon a time"):
        ref, _ = generate_text(tiny_model, byte_tok, prompt, sp)
        draft = tiny_model.truncated(draft_layers) if draft_layers else None
        out, st = speculative_generate(tiny_model, byte_tok, prompt, sp, method, draft, k=4)
        assert byte_tok.decode(out) == ref
        assert st.target_forwards <= len(out)


def test_rejection_sampling_preserves_distribution():
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
    q = np.array([0.1, 0.4, 0.1, 0.3, 0.1])
    counts = np.zeros(5)
    for _ in range(30000):
        d = int(rng.choice(5, p=q))
        n, nxt = verify_draft([d], [q], np.stack([p, p]), rng)
        counts[d if n == 1 else nxt] += 1
    np.testing.assert_allclose(counts / counts.sum(), p, atol=0.01)


def test_ngram_propose():
    assert ngram_propose([1, 2, 3, 4, 1, 2], k=2) == [3, 4]
    assert ngram_propose([5, 6, 7], k=3) == []


# ---------------------------------------------------------------------------
def test_chat_template(bpe_tok, byte_tok):
    msgs = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hello"}]
    ids = apply_chat_template(bpe_tok, msgs, "llama2")
    assert ids[0] == bpe_tok.bos_id and "<<SYS>>" in bpe_tok.decode(ids)
    with pytest.raises(ValueError):
        apply_chat_template(byte_tok, [{"role": "assistant", "content": "x"}], "llama2")
    with pytest.raises(ValueError):
        apply_chat_template(byte_tok, [{"role": "root", "content": "x"}], "plain")


def test_async_engine_concurrent_streams(tiny_model, byte_tok, reference):
    async def main():
        ae = AsyncEngine(LLMEngine(tiny_model, byte_tok), max_waiting=8)
        ae.start()
        try:
            async def one(p):
                return "".join([o.text async for o in ae.generate(p, GREEDY)])
            return await asyncio.gather(*(one(p) for p in PROMPTS))
        finally:
            ae.shutdown()

    assert asyncio.run(main()) == reference


def test_async_engine_cancellation_frees_memory(tiny_model, byte_tok):
    async def main():
        eng = LLMEngine(tiny_model, byte_tok, EngineConfig(num_kv_blocks=64))
        ae = AsyncEngine(eng)
        ae.start()
        gen = ae.generate("hello there", SamplingParams(max_tokens=200, ignore_eos=True))
        await gen.__anext__()
        await gen.aclose()  # client disconnects
        for _ in range(50):
            await asyncio.sleep(0.02)
            if not eng.has_unfinished():
                break
        ae.shutdown()
        return eng.bm.num_free_blocks

    assert asyncio.run(main()) == 64


# ---------------------------------------------------------------------------
def _sse(events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def test_openai_compat_backend_parses_stream():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        body = _sse([{"choices": [{"text": "Hel"}]}, {"choices": [{"text": "lo", "finish_reason": "length"}]},
                     {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}])
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    b = OpenAICompatBackend("vllm", "http://upstream:8000", "llama", api_key="secret-key",
                            transport=httpx.MockTransport(handler))

    async def main():
        req = GenRequest("r1", SamplingParams(max_tokens=2, top_k=5), prompt="Hi")
        return [c async for c in b.stream(req)]

    chunks = asyncio.run(main())
    assert "".join(c.text for c in chunks) == "Hello"
    assert any(c.finish_reason == "length" for c in chunks) and chunks[-1].finished
    assert seen["body"]["top_k"] == 5 and seen["body"]["prompt"] == "Hi" and seen["body"]["stream"] is True
    assert seen["auth"] == "Bearer secret-key"


@pytest.mark.parametrize("status,exc", [(429, BackendOverloaded), (400, BackendBadRequest)])
def test_openai_compat_backend_errors(status, exc):
    b = OpenAICompatBackend("sglang", "http://upstream:30000",
                            transport=httpx.MockTransport(lambda r: httpx.Response(status, text="no")))

    async def main():
        return [c async for c in b.stream(GenRequest("r", SamplingParams(), prompt="x"))]

    with pytest.raises(exc):
        asyncio.run(main())


@pytest.mark.parametrize("url", ["ftp://x", "http://user:pw@host", "http://host/?q=1", "nohost"])
def test_upstream_url_validation(url):
    with pytest.raises(ValueError):
        validate_upstream_url(url)


def test_sequence_requires_tokens(byte_tok):
    with pytest.raises(ValueError):
        Sequence("x", [], SamplingParams(), byte_tok.stream_decoder())
