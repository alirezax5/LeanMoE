from __future__ import annotations

import argparse
import statistics
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from leanmoe.native import (
    BRIDGE_API_VERSION, LM_KV_Q8_0, ContextConfig, ModelConfig, NativeBridge
)

N_GPU_LAYERS = 41
N_CPU_MOE = 33
N_CTX = 81920
N_BATCH = 1024
N_UBATCH = 1024
TARGET_POS = 50000
WARMUP = 8
MEASURE = 128
FILL_TEXT = " test"


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
        return len(data)
    def flush(self):
        for s in self.streams: s.flush()


def pct(v, q):
    v = sorted(v)
    x = (len(v)-1)*q
    lo = int(x); hi = min(lo+1, len(v)-1)
    return v[lo]*(1-(x-lo)) + v[hi]*(x-lo)


def decode(bridge, ctx, tokens, pos, label):
    t0 = time.perf_counter()
    status = bridge.decode_tokens(ctx, tokens, start_pos=pos)
    dt = time.perf_counter() - t0
    if status != 0:
        raise RuntimeError(f"{label}: llama_decode status={status} pos={pos}")
    return dt


def one_run(runtime: Path, model_path: Path, use_mmap: int):
    b = NativeBridge(runtime)
    initialized = False
    model = ctx = None
    try:
        b.load()
        if b.api_version() != BRIDGE_API_VERSION:
            raise RuntimeError(f"Bridge API mismatch: {b.api_version()}")

        b.init(); initialized = True

        mc = ModelConfig(
            n_gpu_layers=N_GPU_LAYERS,
            n_cpu_moe=N_CPU_MOE,
            use_mmap=use_mmap,
            use_mlock=0,
        )
        t0 = time.perf_counter()
        model = b.model_load(model_path, mc)
        model_load = time.perf_counter() - t0

        cc = ContextConfig(
            n_ctx=N_CTX, n_batch=N_BATCH, n_ubatch=N_UBATCH,
            type_k=LM_KV_Q8_0, type_v=LM_KV_Q8_0,
            flash_attn=1, offload_kqv=1,
        )
        t0 = time.perf_counter()
        ctx = b.context_create(model, cc)
        context_create = time.perf_counter() - t0

        actual = (
            b.context_n_ctx(ctx),
            b.context_n_batch(ctx),
            b.context_n_ubatch(ctx),
        )
        if actual != (N_CTX, N_BATCH, N_UBATCH):
            raise RuntimeError(f"Unexpected context config: {actual}")

        ids = b.tokenize(model, FILL_TEXT, add_special=False, parse_special=False)
        if not ids: raise RuntimeError("Fill tokenization failed")
        fill_token = ids[0]

        pos = 0
        t0 = time.perf_counter()
        while pos < TARGET_POS:
            n = min(N_BATCH, TARGET_POS-pos)
            decode(b, ctx, [fill_token]*n, pos, "fill")
            pos += n
            print(f"\r[FILL mmap={use_mmap}] {pos:>6}/{TARGET_POS}", end="", flush=True)
        fill_time = time.perf_counter()-t0
        print()

        token = b.argmax_token(model, ctx)
        for _ in range(WARMUP):
            decode(b, ctx, [token], pos, "warmup")
            pos += 1
            token = b.argmax_token(model, ctx)

        decode_times = []
        argmax_times = []
        wall0 = time.perf_counter()

        for i in range(MEASURE):
            t0 = time.perf_counter()
            token = b.argmax_token(model, ctx)
            argmax_times.append(time.perf_counter()-t0)

            decode_times.append(decode(b, ctx, [token], pos, "measure"))
            pos += 1

            if (i+1) % 16 == 0:
                print(f"[MEASURE mmap={use_mmap}] {i+1}/{MEASURE}")

        wall = time.perf_counter()-wall0
        ms = [x*1000 for x in decode_times]
        arg_ms = [x*1000 for x in argmax_times]

        return {
            "mmap": use_mmap,
            "model_load_s": model_load,
            "context_create_s": context_create,
            "fill_s": fill_time,
            "decode_tps": MEASURE/sum(decode_times),
            "wall_tps": MEASURE/wall,
            "decode_median_ms": statistics.median(ms),
            "decode_p95_ms": pct(ms, .95),
            "argmax_mean_ms": statistics.mean(arg_ms),
            "wall_s": wall,
        }
    finally:
        if ctx is not None: b.context_free(ctx)
        if model is not None: b.model_free(model)
        if initialized: b.shutdown()
        b.close()


def main():
    ap = argparse.ArgumentParser(description="LeanMoE Phase 2B.4 real mmap ON/OFF A/B")
    ap.add_argument("runtime", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("--log", type=Path)
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = (a.log or Path("logs")/f"phase2b4-mmap-ab-{stamp}.txt").resolve()
    log.parent.mkdir(parents=True, exist_ok=True)

    with log.open("w", encoding="utf-8", buffering=1) as f:
        out, err = Tee(sys.__stdout__, f), Tee(sys.__stderr__, f)
        with redirect_stdout(out), redirect_stderr(err):
            print(f"[LOG] {log}")
            print("=== LeanMoE Phase 2B.4 — REAL MMAP A/B ===")
            print("Frozen: 41 GPU layers / 33 CPU MoE / ctx 81920 / batch 1024 / Q8_0 KV / FA ON")
            print("Each arm uses a fresh model + context in the same process.")
            print()

            results = []
            for mmap in (1, 0):
                label = "ON" if mmap else "OFF"
                print("="*72)
                print(f"MMAP {label}")
                print("="*72)
                r = one_run(a.runtime.resolve(), a.model.resolve(), mmap)
                results.append(r)
                print(
                    f"[RESULT mmap={label}] load={r['model_load_s']:.3f}s | "
                    f"fill={r['fill_s']:.3f}s | decode={r['decode_tps']:.2f} tok/s | "
                    f"wall={r['wall_tps']:.2f} tok/s | median={r['decode_median_ms']:.3f}ms | "
                    f"p95={r['decode_p95_ms']:.3f}ms | argmax={r['argmax_mean_ms']:.3f}ms"
                )
                print()

            on, off = results
            def delta(new, old):
                return ((new/old)-1.0)*100.0 if old else float("nan")

            print("=== A/B SUMMARY ===")
            print(f"{'metric':<24} {'mmap ON':>12} {'mmap OFF':>12} {'OFF vs ON':>12}")
            print("-"*64)
            rows = [
                ("model load s", on["model_load_s"], off["model_load_s"], False),
                ("50K fill s", on["fill_s"], off["fill_s"], False),
                ("decode tok/s", on["decode_tps"], off["decode_tps"], True),
                ("wall tok/s", on["wall_tps"], off["wall_tps"], True),
                ("decode median ms", on["decode_median_ms"], off["decode_median_ms"], False),
                ("decode p95 ms", on["decode_p95_ms"], off["decode_p95_ms"], False),
                ("argmax mean ms", on["argmax_mean_ms"], off["argmax_mean_ms"], False),
            ]
            for name, x, y, _ in rows:
                print(f"{name:<24} {x:>12.3f} {y:>12.3f} {delta(y,x):>+11.2f}%")

            print()
            print("=== PHASE 2B.4 MMAP A/B PASS ===")
            print("Both arms completed with fresh model/context and decode status 0.")
            print("Interpret performance only after confirming runtime logs show mmap behavior changed.")
            print("[LOG] Final return code = 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
