from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


EXPECTED_API_VERSION = 1


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: bridge_smoke.py <runtime-dir>")
        return 2

    runtime_dir = Path(sys.argv[1]).resolve()
    bridge_path = runtime_dir / "leanmoe_bridge.dll"

    if not bridge_path.is_file():
        print(f"[FAIL] bridge not found: {bridge_path}")
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

        version = bridge.lm_api_version()

        if version != EXPECTED_API_VERSION:
            print(
                f"[FAIL] API version: expected "
                f"{EXPECTED_API_VERSION}, got {version}"
            )
            return 1

        print(f"[PASS] Bridge API version = {version}")

        result = bridge.lm_init()

        if result != 0:
            error = bridge.lm_last_error()
            message = (
                error.decode("utf-8", errors="replace")
                if error
                else "<no error>"
            )
            print(f"[FAIL] lm_init(): {result}: {message}")
            return 1

        print("[PASS] lm_init()")

        bridge.lm_shutdown()
        print("[PASS] lm_shutdown()")

        print()
        print("=== LEANMOE BRIDGE SMOKE PASS ===")
        return 0

    finally:
        dll_dir.close()


if __name__ == "__main__":
    raise SystemExit(main())
