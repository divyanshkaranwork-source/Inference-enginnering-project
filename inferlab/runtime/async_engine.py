"""AsyncEngine - drive ``LLMEngine.step()`` from a background thread.

An HTTP server handles many concurrent connections on one asyncio event loop,
but the model forward pass is blocking CPU/GPU work. The standard design
(vLLM's AsyncLLM, SGLang's TokenizerManager <-> Scheduler split):

    HTTP handlers --(thread-safe queue)--> engine thread: loop { step() }
    HTTP handlers <--(per-request asyncio.Queue, call_soon_threadsafe)--

* Backpressure: ``max_waiting`` caps queued requests -> HTTP 503 instead of
  unbounded memory growth.
* Cancellation: if a client disconnects, its generator is closed and the
  request is aborted so its KV blocks are freed immediately.
NumPy releases the GIL inside BLAS, so the event loop stays responsive while
the engine thread computes.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from inferlab.foundations.sampling import SamplingParams
from inferlab.runtime.engine import LLMEngine, RequestOutput

logger = logging.getLogger(__name__)


class EngineOverloaded(RuntimeError):
    pass


@dataclass
class _NewRequest:
    request_id: str
    prompt: str | list[int]
    params: SamplingParams
    cache_salt: str | None
    guided_choice: list[str] | None
    loop: asyncio.AbstractEventLoop
    q: asyncio.Queue


class AsyncEngine:
    def __init__(self, engine: LLMEngine, max_waiting: int = 64):
        self.engine = engine
        self.max_waiting = max_waiting
        self._inbox: queue.Queue = queue.Queue()
        self._aborts: queue.Queue = queue.Queue()
        self._streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot: dict = engine.snapshot()
        self._pending = 0
        self._last_pub = 0.0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name="inferlab-engine", daemon=True)
        self._started = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._started:
            self._thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot

    @property
    def num_pending(self) -> int:
        with self._lock:
            return self._pending

    # ------------------------------------------------------------------
    async def generate(
        self,
        prompt: str | list[int],
        params: SamplingParams,
        request_id: str | None = None,
        cache_salt: str | None = None,
        guided_choice: list[str] | None = None,
    ) -> AsyncIterator[RequestOutput]:
        if not self._started:
            raise RuntimeError("engine not started")
        with self._lock:
            if self._pending >= self.max_waiting + self.engine.cfg.max_num_seqs:
                raise EngineOverloaded("too many requests in flight")
            self._pending += 1
        rid = request_id or f"cmpl-{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self._inbox.put(_NewRequest(rid, prompt, params, cache_salt, guided_choice, loop, q))
        self._wake.set()
        finished = False
        try:
            while True:
                item = await q.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    finished = True
                    return
        finally:
            with self._lock:
                self._pending -= 1
            if not finished:  # client went away or error -> free resources
                self._aborts.put(rid)
                self._wake.set()

    # ------------------------------------------------------------------
    def _deliver(self, rid: str, item) -> None:
        target = self._streams.get(rid)
        if target is None:
            return
        loop, q = target
        try:
            loop.call_soon_threadsafe(q.put_nowait, item)
        except RuntimeError:  # event loop closed
            self._streams.pop(rid, None)

    def _run(self) -> None:
        eng = self.engine
        while not self._stop.is_set():
            # 1) new requests
            while True:
                try:
                    req: _NewRequest = self._inbox.get_nowait()
                except queue.Empty:
                    break
                self._streams[req.request_id] = (req.loop, req.q)
                try:
                    eng.add_request(
                        req.prompt, req.params, req.request_id, req.cache_salt, req.guided_choice
                    )
                except ValueError as exc:
                    self._deliver(req.request_id, exc)
                    self._streams.pop(req.request_id, None)
            # 2) aborts
            while True:
                try:
                    rid = self._aborts.get_nowait()
                except queue.Empty:
                    break
                eng.abort(rid)
                self._streams.pop(rid, None)
            # 3) one engine iteration
            if eng.has_unfinished():
                try:
                    outputs = eng.step()
                except Exception as exc:  # never let the engine thread die silently
                    logger.exception("engine step failed; aborting in-flight requests")
                    self.errors += 1
                    err = RuntimeError("internal engine error")
                    err.__cause__ = exc
                    for rid in list(self._streams):
                        self._deliver(rid, err)
                        eng.abort(rid)
                    self._streams.clear()
                    outputs = []
                for out in outputs:
                    self._deliver(out.request_id, out)
                    if out.finished:
                        self._streams.pop(out.request_id, None)
                if time.monotonic() - self._last_pub > 0.1:  # throttle UI snapshots
                    self._publish()
            else:
                self._publish()
                self._wake.wait(timeout=0.25)
                self._wake.clear()

    def _publish(self) -> None:
        snap = self.engine.snapshot()
        snap["time"] = time.time()
        self._last_pub = time.monotonic()
        with self._lock:
            self._snapshot = snap
