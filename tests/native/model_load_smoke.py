from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int32),
        ("n_cpu_moe", ctypes.c_int32),
        ("use_mmap", ctypes.c_uint8),
        ("use_mlock", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 6),
    ]


def last_error(bridge) -> str:
    value = bridge.lm_last_error()
    if not value:
        return "<no error>"
    return value.decode("utf-8", errors="replace")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: model_load_smoke.py <runtime-dir> <model.gguf>")
        return 2

    runtime_dir = Path(sys.argv[1]).resolve()
    model_path = Path(sys.argv[2]).resolve()

    if not model_path.is_file():
        print(f"[FAIL] Model not found: {model_path}")
        return 1

    bridge_path = runtime_dir / "leanmoe_bridge.dll"

    if not bridge_path.is_file():
        print(f"[FAIL] Bridge not found: {bridge_path}")
        return 1

    dll_dir = os.add_dll_directory(str(runtime_dir))

    try:
        bridge = ctypes.CDLL(str(bridge_path))

        bridge.lm_api_version.argtypes = []
        bridge.lm_api_version.restype = ctypes.c_uint32

        bridge.lm_init.argtypes = []
        bridge.lm_init.restype = ctypes.c_int32

        bridge.lm_shutdown.argtypes = []
        bridge.lm_shutdown.restype = None

        bridge.lm_last_error.argtypes = []
        bridge.lm_last_error.restype = ctypes.c_char_p

        bridge.lm_model_load.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ModelConfig),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        bridge.lm_model_load.restype = ctypes.c_int32

        bridge.lm_model_free.argtypes = [ctypes.c_void_p]
        bridge.lm_model_free.restype = None

        bridge.lm_model_size.argtypes = [ctypes.c_void_p]
        bridge.lm_model_size.restype = ctypes.c_uint64

        bridge.lm_model_n_params.argtypes = [ctypes.c_void_p]
        bridge.lm_model_n_params.restype = ctypes.c_uint64

        bridge.lm_model_description.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        bridge.lm_model_description.restype = ctypes.c_int32

        print("=== LeanMoE Phase 1B.2 ===")
        print(f"Bridge: {bridge_path}")
        print(f"Model:  {model_path}")
        print()

        version = bridge.lm_api_version()
        print(f"[PASS] Bridge API version = {version}")

        result = bridge.lm_init()
        if result != 0:
            print(f"[FAIL] lm_init(): {result}: {last_error(bridge)}")
            return 1

        print("[PASS] Backend initialized")

        # CPU-only Phase 1B validation.
        config = ModelConfig(
            n_gpu_layers=0,
            n_cpu_moe=-1,
            use_mmap=1,
            use_mlock=0,
        )

        model = ctypes.c_void_p()

        print("[INFO] Loading model...")
        start = time.perf_counter()

        result = bridge.lm_model_load(
            os.fsencode(model_path),
            ctypes.byref(config),
            ctypes.byref(model),
        )

        elapsed = time.perf_counter() - start

        if result != 0 or not model.value:
            print(
                f"[FAIL] lm_model_load(): "
                f"{result}: {last_error(bridge)}"
            )
            return 1

        print(f"[PASS] Model loaded in {elapsed:.2f}s")

        size = bridge.lm_model_size(model)
        params = bridge.lm_model_n_params(model)

        description_buffer = ctypes.create_string_buffer(1024)

        result = bridge.lm_model_description(
            model,
            description_buffer,
            len(description_buffer),
        )

        if result != 0:
            print(
                f"[FAIL] lm_model_description(): "
                f"{result}: {last_error(bridge)}"
            )
            return 1

        description = description_buffer.value.decode(
            "utf-8",
            errors="replace",
        )

        print()
        print("=== Model Metadata ===")
        print(f"Description : {description}")
        print(f"Parameters  : {params:,}")
        print(f"Tensor size : {size / (1024 ** 3):.3f} GiB")

        print()
        print("[INFO] Freeing model...")
        bridge.lm_model_free(model)
        model = ctypes.c_void_p()

        print("[PASS] Model freed")

        bridge.lm_shutdown()
        print("[PASS] Backend shutdown")

        print()
        print("=== PHASE 1B.2 PASS ===")
        return 0

    finally:
        dll_dir.close()


if __name__ == "__main__":
    raise SystemExit(main())
