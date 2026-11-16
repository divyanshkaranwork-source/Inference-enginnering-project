# 5D parallelism: when one GPU is not enough

> Code: [`inferlab/parallel/`](../inferlab/parallel) · Lab 10 · UI: *5D parallelism* (planner +
> simulator)

A model is split across GPUs for one of two reasons: it doesn't **fit** (weights + KV > one GPU's
memory), or one GPU isn't **fast enough** (latency or throughput). There are five independent axes
to split along, and real deployments combine them. The world size is
`DP × PP × TP × CP`, with EP reusing the DP×TP ranks for MoE layers.

Every strategy below runs on `SimCluster`, a list of NumPy arrays standing in for GPUs, with
NCCL-style collectives that count the bytes each rank sends. Every simulation is **checked against
the single-device result**: the tests assert the outputs match.

| Axis | What is split | Communication per layer | Cuts memory of | Helps | Use across |
|------|---------------|-------------------------|----------------|-------|-----------|
| **DP** data | requests (full replicas) | none | — | throughput | anything |
| **TP** tensor | every weight matrix (by heads / FFN columns) | 2 all-reduces of `[tokens, d]` | weights + KV | latency + capacity | NVLink domain (≤ 8 GPUs) |
| **PP** pipeline | layers | 1 send/recv of `[tokens, d]` per stage boundary | weights + KV | capacity | nodes |
| **CP / SP** sequence | tokens of a long prompt | ring send of K/V chunks, or 2 all-to-alls | activations + KV | long-context prefill | NVLink / IB |
| **EP** expert | MoE experts | 2 all-to-alls of routed tokens | expert weights | MoE capacity + throughput | NVLink / IB |

## 1. Data parallelism ([`data_parallel.py`](../inferlab/parallel/data_parallel.py))

For inference, DP means N independent replicas behind a router. There is no model communication, so
throughput scales linearly. The interesting part is routing:

- *round-robin*: simple, but spreads each shared prefix over every replica (more cache misses)
- *least-loaded*: balances the outstanding tokens
- *prefix-aware*: hash the prompt prefix to pick a replica, so each system prompt is cached once
  (the SGLang router, NVIDIA Dynamo, llm-d). The trade-off is possible load imbalance.

`simulate_dp` runs **real engines** with shared weights and separate KV pools, and measures
prefix-cache hits for each policy.

## 2. Tensor parallelism ([`tensor_parallel.py`](../inferlab/parallel/tensor_parallel.py))

This is Megatron-LM's scheme. For each layer, with `tp` GPUs:

```
attention:  Wq, Wk, Wv  column-parallel → each GPU owns n_heads/tp heads, attention is local
            Wo          row-parallel    → partial sums → ALL-REDUCE
FFN:        W1, W3      column-parallel → each GPU owns hidden/tp columns
            W2          row-parallel    → partial sums → ALL-REDUCE
embedding + LM head: vocab-parallel (each GPU owns vocab/tp rows; Megatron pads the vocabulary)
```

Weights *and* KV heads are divided by `tp`, so each GPU reads `1/tp` of the bytes per token. For
memory-bound decode this cuts **latency** almost linearly, until the two all-reduces per layer
(which sit on the critical path of every token) dominate. That is why TP stays inside an NVLink domain.

When `tp > n_kv_heads` (for example TP=16 on a GQA model with 8 KV heads), KV heads must be
replicated; the planner warns about this.

**Sequence parallelism (Megatron-SP)** keeps the residual stream sharded along the sequence and
replaces each all-reduce with reduce-scatter + all-gather. The bytes are the same, but norms and
activations are now split `tp` ways. Run `tp_forward(..., sequence_parallel=True)` to see the
different collectives.

## 3. Pipeline parallelism ([`pipeline_parallel.py`](../inferlab/parallel/pipeline_parallel.py))

Layers are split into contiguous stages, and only the `[tokens, d]` activation crosses each boundary.
That is cheap enough for inter-node links. A request still has to pass through the stages *one after
another*, so:

- per-token **latency does not improve** (it gets slightly worse from the extra hops)
- to keep every stage busy you need ≥ `pp` micro-batches in flight. The **bubble** is
  `(pp − 1)/(m + pp − 1)` (GPipe). The simulator draws the Gantt chart from measured stage times.
- for memory-bound decode, each micro-batch re-reads its stage's weights. PP therefore adds
  *capacity* (fits bigger models, more KV) more than speed. The planner models this, and it is why
  `recommend()` prefers TP within a node.

## 4. Context / sequence parallelism ([`sequence_parallel.py`](../inferlab/parallel/sequence_parallel.py))

For 128k–1M-token prompts, even a single sequence's activations and KV overflow one GPU.

**Ring Attention.** Each GPU keeps its query chunk. K/V chunks travel around a ring, and after `cp`
steps every query has seen every key. Partial results merge exactly with the online-softmax
(log-sum-exp) rule, the same maths as FlashAttention tiles but across GPUs, and the send/recv
overlaps with compute. With causal masking, contiguous chunks are unbalanced: GPU 0 has almost
nothing to do while the last GPU does the most. **Zig-zag** partitioning (GPU i gets chunks i and
2cp−1−i) balances the work. The simulator reports the measured imbalance: about 1.7× contiguous
versus 1.0× zig-zag.

**Ulysses (DeepSpeed).** An all-to-all switches the sharding from sequence to heads. Each GPU then
runs normal attention for `H/cp` heads over the full sequence, and a second all-to-all switches back.
It is simple and efficient, but `cp ≤ n_kv_heads`.

## 5. Expert parallelism ([`expert_parallel.py`](../inferlab/parallel/expert_parallel.py))

An MoE layer (Mixtral: 8 experts, top-2; DeepSeek-V3: 256 routed experts, top-8) has far more
parameters than it uses per token. Expert parallelism places different experts on different GPUs:

```
route (local top-k) → ALL-TO-ALL dispatch → local experts → ALL-TO-ALL combine → weighted sum
```

- **load balance**: a popular expert makes its GPU the straggler (try `skew > 0` in the UI).
  Training adds balancing losses; serving adds redundant copies of hot experts (EPLB).
- **capacity factor**: cap tokens per expert to bound latency. Overflowing tokens are dropped, and
  the simulator shows the resulting error.
- attention is dense, so MoE serving often uses **DP for attention + EP for experts** (SGLang
  DP-attention, DeepSeek's deployment).

## 6. Planning ([`planner.py`](../inferlab/parallel/planner.py))

`evaluate()` computes, for a model × GPU × layout, the memory per GPU (weights, KV, activations),
whether it fits, and decode latency and throughput, including TP all-reduce time, EP all-to-all time,
PP hops and micro-batch re-reads. `recommend()` enumerates layouts for N GPUs and ranks the ones that
fit.

```bash
inferlab plan --model llama-2-7b  --gpu A100-SXM-80GB --tp 1 --batch 32       # doesn't fit at 4k x 32!
inferlab plan --model llama-2-70b --gpu H100-SXM-80GB --gpus 8 --recommend
inferlab plan --model mixtral-8x7b --gpu H100-SXM-80GB --gpus 8 --recommend
```

Rules of thumb these numbers reproduce:

1. Fit the model with the fewest GPUs per replica (quantize first: FP8 halves the memory).
2. Inside a node use TP; across nodes use PP or, better, more DP replicas.
3. Long prompts: add CP for prefill only.
4. MoE: EP for experts, DP for attention.
5. Scale throughput with DP replicas and a cache-aware router.

## 7. Real-world mapping

| Framework | Flags |
|-----------|-------|
| vLLM | `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--enable-expert-parallel` |
| SGLang | `--tp`, `--dp`, `--enable-dp-attention`, `--ep-size` / DeepEP |
| TensorRT-LLM | `--tp_size`, `--pp_size`, `--ep_size`, context parallelism in the PyTorch backend |
| llama.cpp | `--split-mode layer` (≈ PP) / `row` (≈ TP), `--tensor-split` |

---

**Exercises**

1. Add an *interleaved 1F1B*-style schedule to `pp_forward` for decode (several sequences in flight)
   and compare the bubble with GPipe.
2. Implement 2D TP (split along both dimensions) and compare its communication volume with 1D TP.
3. Add an expert-load-balancing pass: replicate the hottest expert onto the least-loaded rank, and
   measure the imbalance with `skew=2`.
4. Combine TP=2 × CP=2 on four simulated ranks for one prefill and verify it against
   `logits_full`.
