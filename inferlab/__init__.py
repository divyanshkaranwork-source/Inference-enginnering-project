"""InferLab: learn inference engineering by building a full LLM serving stack.

The package is organised in the same four layers used throughout the docs:

* ``inferlab.foundations`` - the math: Llama architecture, tokenizer, sampling,
  KV cache, quantization, memory / FLOP / roofline calculators.
* ``inferlab.runtime``     - the engine: paged KV cache, continuous batching,
  prefix caching, speculative decoding, adapters for llama.cpp / vLLM /
  SGLang / TensorRT-LLM.
* ``inferlab.kernels``     - C++ CPU kernels (tiled GEMM, FlashAttention,
  PagedAttention, Q8/Q4 GEMV) and NumPy reference implementations.
* ``inferlab.parallel``    - 5D parallelism (data, tensor, pipeline,
  sequence/context, expert) simulated on a virtual cluster.
* ``inferlab.server``      - the production layer: an OpenAI-compatible,
  authenticated, rate-limited, observable HTTP gateway plus a web UI.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
