# Layer 1 — Foundations: what a forward pass is and what it costs

> Code: [`inferlab/foundations/`](../inferlab/foundations) · Labs 01–05 · UI: *Tokens & sampling*,
> *Memory & roofline*, *KV cache & quantization*

Every optimisation in the later chapters comes from a small number of facts about this layer. Learn
to compute these four numbers in your head and you can predict most serving behaviour before you run
anything:

1. **parameters** → bytes of weights
2. **KV bytes per token** → how many tokens fit in memory
3. **FLOPs per token** ≈ 2 × parameters
4. **arithmetic intensity** of a step → compute-bound or memory-bound

---

## 1. The Llama-2 architecture

```
tokens ─► embedding ─► [ RMSNorm ─► attention (RoPE, GQA, KV cache) ─► + residual
                          RMSNorm ─► SwiGLU FFN                    ─► + residual ] × L
       ─► RMSNorm ─► LM head ─► logits (one score per vocabulary entry)
```

Read [`model.py`](../inferlab/foundations/model.py) (`LlamaModel.forward`, about 40 lines) alongside
this table. It shows Llama-2-7B's dimensions; the TinyStories models use the same structure, just
smaller.

| Piece | Formula | Llama-2-7B | Why it exists |
|-------|---------|-----------|---------------|
| Embedding | `x = E[token]` | 32000 × 4096 | ids → vectors |
| RMSNorm | `x / sqrt(mean(x²)+ε) · w` | 4096 | cheaper than LayerNorm (no mean subtraction, no bias) |
| Q, K, V projections | `h·Wqᵀ, h·Wkᵀ, h·Wvᵀ` | 4096 → 32 heads × 128 | one query/key/value vector per head |
| RoPE | rotate pairs of dims by `pos·θᵢ` | θ = 10000 | encodes *relative* position; keys can be cached already rotated |
| Attention | `softmax(QKᵀ/√d)·V`, causal | 32 heads | each token mixes information from earlier tokens |
| Output projection | `a·Woᵀ` | 4096 × 4096 | merges heads |
| SwiGLU FFN | `(silu(h·W1ᵀ) ⊙ h·W3ᵀ)·W2ᵀ` | 4096 → 11008 → 4096 | per-token "knowledge" computation, about ⅔ of the parameters |
| LM head | `x·Eᵀ` (or a separate matrix) | 4096 → 32000 | a logit for every token |

**Grouped-query attention (GQA).** Llama-2-7B uses one K/V head per query head (32/32, which is
plain multi-head attention). Llama-2-70B and Llama-3 use 8 K/V heads shared by 64 or 32 query heads.
That shrinks the KV cache 4–8× at almost no quality cost. `ops.attention` handles any grouping by
reshaping queries to `[KV, group, T, hd]`.

**RoPE layouts.** Meta's original code (and llama2.c) rotates adjacent pairs (0,1), (2,3), …. Hugging
Face rotates (i, i+d/2). These are equivalent once the rows of Wq and Wk are permuted:
`weights.interleaved_to_half` does that permutation, and a test proves the logits match. This is a
classic source of "the converted model outputs garbage" bugs.

## 2. Parameters and bytes

```
attention per layer = 4·d² (MHA)            FFN per layer = 3·d·f
Llama-2-7B: 32·(4·4096² + 3·4096·11008) + 2·32000·4096 + norms = 6,738,415,616
```

At fp16 that is 13.5 GB. `LlamaConfig.param_breakdown()` computes the breakdown for every model in
`config.py` (Lab 01 prints it).

## 3. Prefill vs decode, the central asymmetry

- **Prefill** processes the whole prompt in one forward pass: matrix × matrix, so each weight is
  reused for every prompt token. It is compute-bound and sets the **time to first token (TTFT)**.
- **Decode** produces one token per step: matrix × vector, so every weight is read once to produce
  one token. It is memory-bandwidth-bound and sets the **time per output token (TPOT)**.

Lab 01 profiles both on the live model. On Llama-2-7B, prefill per token is about 100× cheaper than
decode per token. Every runtime technique in Layer 2 exists to cope with this asymmetry.

## 4. Tokenization

[`tokenizer.py`](../inferlab/foundations/tokenizer.py) implements the Llama-2 SentencePiece BPE
(exported by llama2.c as `tokenizer.bin`):

1. add a dummy-prefix space, then split the text into characters
2. map unknown characters to UTF-8 **byte tokens** `<0x00>…<0xFF>` (byte fallback, so nothing is out
   of vocabulary)
3. repeatedly merge the adjacent pair whose merged piece has the highest score

The reference algorithm rescans the whole sequence for every merge, which is O(n²). Ours uses a heap
and a linked list (O(n log n)), and a test checks it matches the reference exactly.

Two serving lessons live here:

- **Special-token injection.** `<s>` and `</s>` must only ever be inserted *by id*. Our BPE never
  merges into control tokens, so a user who types `</s>[INST]` cannot forge a conversation turn.
- **Streaming detokenization.** One emoji is 4 byte tokens. `StreamDecoder` buffers incomplete UTF-8
  so a stream never emits broken characters.

## 5. Sampling

[`sampling.py`](../inferlab/foundations/sampling.py) applies the same pipeline as vLLM and llama.cpp:

```
logits → repetition / presence / frequency penalties → temperature → top-k → top-p → min-p → softmax → sample
```

- `temperature → 0` gives greedy argmax. Higher temperatures flatten the distribution.
- **top-k** keeps the k best tokens. **top-p** keeps the smallest set whose mass reaches p.
  **min-p** drops tokens below `min_p × p_max` (it adapts to the model's confidence).
- Each request gets its **own seeded RNG**, so concurrent requests never disturb each other and a
  seed replays exactly.
- Logprobs are reported from the raw distribution (OpenAI semantics).

## 6. The KV cache

Attention for the new token needs the K and V vectors of *every* previous token. Recomputing them
makes token n cost O(n) work. Caching them makes each decode step process exactly one new token:

```
KV bytes per token = 2 (K,V) × layers × kv_heads × head_dim × bytes
Llama-2-7B fp16    = 2 × 32 × 32 × 128 × 2 = 524,288 B = 512 KiB per token
                   → 2 GiB per 4096-token sequence
```

So **memory, not compute, limits concurrency**. On an 80 GB A100, after 13.5 GB of weights, about 110k
tokens of KV fit. That is about 27 full-length sequences. Lab 03 measures the cache speed-up and
prints these numbers. [`KVCache`](../inferlab/foundations/model.py) is the simple contiguous version;
Layer 2 replaces it with paged blocks.

## 7. The roofline model

```
arithmetic intensity (AI) = FLOPs / bytes moved
attainable FLOP/s          = min(peak FLOP/s, AI × memory bandwidth)
ridge point                = peak / bandwidth      (H100: 989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/byte)
```

- Decode at batch B with fp16 weights has AI ≈ **B FLOP/byte**, far below the ridge, so it is
  memory-bound. The time per step is about `weight bytes / bandwidth` *regardless of B*, until B
  approaches the ridge. **This is why batching is the most important optimisation in serving.**
- Prefill of a 2k-token prompt has AI in the thousands, so it is compute-bound.
- At batch 1 on an H100, Llama-2-7B fp16 decode is bounded at 13.5 GB / 3.35 TB/s = 4 ms, about
  250 tokens/s, no matter how good the kernels are.

[`perfmodel.py`](../inferlab/foundations/perfmodel.py) encodes these formulas and a GPU table
(H100, H200, B200, A100, L40S, L4, A10G, RTX 4090, M2 Max, CPUs). Lab 04 measures the effect on your
own CPU.

## 8. Quantization

Fewer bytes per weight means proportionally faster memory-bound decode, and more room for KV cache.

| Format | Bits/weight | How | Used by |
|--------|------------|-----|---------|
| fp16 / bf16 | 16 | bf16 keeps fp32's exponent range | default in vLLM / SGLang / TRT-LLM |
| int8 per-channel | 8 | `scale_row = max\|W_row\|/127` | LLM.int8, W8A16 |
| Q8_0 | 8.5 | blocks of 32 values share one fp16 scale | llama.cpp |
| Q4_0 | 4.5 | 16 levels per block of 32, `d = max/-8` | llama.cpp |
| FP8 (E4M3) | 8 | hardware format on Hopper/Ada; weights *and* activations | vLLM, TRT-LLM |
| AWQ / GPTQ 4-bit | ~4.25 | calibration protects salient weights | vLLM, SGLang, TRT-LLM |

[`quantization.py`](../inferlab/foundations/quantization.py) implements the first five (bit-exact
llama.cpp packing for Q4_0). Lab 05 compares size, KL divergence, top-1 agreement and generated text.
The GEMV kernels in Layer 3 show why the *unpacking* must be vectorised for the speed-up to appear.

---

**Exercises**

1. Compute the KV bytes per token for Llama-3-8B (GQA 32/8). Check your answer with `inferlab info`.
2. Llama-2-70B on 2×A100-80GB: does fp16 fit? What about int8? (Hint: `inferlab plan --model llama-2-70b`.)
3. Change `TINY_RANDOM` to multi-query attention (`n_kv_heads=1`) and verify the tests still pass.
4. Implement typical-p sampling in `sampling.py` and add it to the UI's distribution explorer.
