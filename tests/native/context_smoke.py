import ctypes
import os
import sys
import time
from pathlib import Path


LM_OK = 0
LM_KV_Q8_0 = 1


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int32),
        ("n_cpu_moe", ctypes.c_int32),
        ("use_mmap", ctypes.c_uint8),
        ("use_mlock", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 6),
    ]


class ContextConfig(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("flash_attn", ctypes.c_uint8),
        ("offload_kqv", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 6),
    ]


def fail(lib, message):
    err = lib.lm_last_error()
    detail = err.decode("utf-8", errors="replace") if err else ""
    raise RuntimeError(f"{message}: {detail}")


if len(sys.argv) != 3:
    print(
        "Usage: context_smoke.py "
        "<runtime-dir> <model.gguf>"
    )
    sys.exit(2)


runtime_dir = Path(sys.argv[1]).resolve()
model_path = Path(sys.argv[2]).resolve()

if not runtime_dir.is_dir():
    raise RuntimeError(f"Runtime directory not found: {runtime_dir}")

if not model_path.is_file():
    raise RuntimeError(f"Model not found: {model_path}")


os.add_dll_directory(str(runtime_dir))

bridge_path = runtime_dir / "leanmoe_bridge.dll"
lib = ctypes.CDLL(str(bridge_path))


lib.lm_api_version.restype = ctypes.c_uint32

lib.lm_last_error.restype = ctypes.c_char_p

lib.lm_init.argtypes = []
lib.lm_init.restype = ctypes.c_int

lib.lm_shutdown.argtypes = []
lib.lm_shutdown.restype = None

lib.lm_model_load.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ModelConfig),
    ctypes.POINTER(ctypes.c_void_p),
]
lib.lm_model_load.restype = ctypes.c_int

lib.lm_model_free.argtypes = [ctypes.c_void_p]
lib.lm_model_free.restype = None

lib.lm_context_create.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ContextConfig),
    ctypes.POINTER(ctypes.c_void_p),
]
lib.lm_context_create.restype = ctypes.c_int

lib.lm_context_free.argtypes = [ctypes.c_void_p]
lib.lm_context_free.restype = None

lib.lm_context_n_ctx.argtypes = [ctypes.c_void_p]
lib.lm_context_n_ctx.restype = ctypes.c_uint32

lib.lm_context_n_batch.argtypes = [ctypes.c_void_p]
lib.lm_context_n_batch.restype = ctypes.c_uint32

lib.lm_context_n_ubatch.argtypes = [ctypes.c_void_p]
lib.lm_context_n_ubatch.restype = ctypes.c_uint32


api = lib.lm_api_version()
print(f"[INFO] Bridge API version = {api}")

if api != 2:
    raise RuntimeError(f"Expected Bridge API 2, got {api}")


model = ctypes.c_void_p()
ctx = ctypes.c_void_p()

initialized = False

try:
    rc = lib.lm_init()
    if rc != LM_OK:
        fail(lib, "lm_init() failed")

    initialized = True
    print("[PASS] Backend initialized")


    model_cfg = ModelConfig(
        n_gpu_layers=41,
        n_cpu_moe=33,
        use_mmap=0,
        use_mlock=0,
    )

    print()
    print("=== MODEL CONFIG ===")
    print("n_gpu_layers = 41")
    print("n_cpu_moe    = 33")

    t0 = time.perf_counter()

    rc = lib.lm_model_load(
        str(model_path).encode("utf-8"),
        ctypes.byref(model_cfg),
        ctypes.byref(model),
    )

    if rc != LM_OK:
        fail(lib, "lm_model_load() failed")

    print(
        f"[PASS] Model loaded in "
        f"{time.perf_counter() - t0:.2f}s"
    )


    ctx_cfg = ContextConfig(
        n_ctx=81920,
        n_batch=1024,
        n_ubatch=1024,
        type_k=LM_KV_Q8_0,
        type_v=LM_KV_Q8_0,
        flash_attn=1,
        offload_kqv=1,
    )

    print()
    print("=== REQUESTED CONTEXT ===")
    print("n_ctx       = 81920")
    print("n_batch     = 1024")
    print("n_ubatch    = 1024")
    print("K cache     = Q8_0")
    print("V cache     = Q8_0")
    print("Flash Attn  = ON")
    print("offload_kqv = ON")

    t0 = time.perf_counter()

    rc = lib.lm_context_create(
        model,
        ctypes.byref(ctx_cfg),
        ctypes.byref(ctx),
    )

    if rc != LM_OK:
        fail(lib, "lm_context_create() failed")

    elapsed = time.perf_counter() - t0

    actual_ctx = lib.lm_context_n_ctx(ctx)
    actual_batch = lib.lm_context_n_batch(ctx)
    actual_ubatch = lib.lm_context_n_ubatch(ctx)

    print()
    print(f"[PASS] Context created in {elapsed:.2f}s")
    print()
    print("=== ACTUAL CONTEXT ===")
    print(f"n_ctx    = {actual_ctx}")
    print(f"n_batch  = {actual_batch}")
    print(f"n_ubatch = {actual_ubatch}")

    if actual_ctx != 81920:
        raise RuntimeError(
            f"Expected n_ctx=81920, got {actual_ctx}"
        )

    if actual_batch != 1024:
        raise RuntimeError(
            f"Expected n_batch=1024, got {actual_batch}"
        )

    if actual_ubatch != 1024:
        raise RuntimeError(
            f"Expected n_ubatch=1024, got {actual_ubatch}"
        )

    print()
    print("=== PHASE 2A CONTEXT SMOKE PASS ===")

finally:
    if ctx:
        lib.lm_context_free(ctx)
        print("[PASS] Context freed")

    if model:
        lib.lm_model_free(model)
        print("[PASS] Model freed")

    if initialized:
        lib.lm_shutdown()
        print("[PASS] Backend shutdown")
