from __future__ import annotations

import ctypes
from pathlib import Path


BRIDGE_API_VERSION = 1


class LeanMoENativeError(RuntimeError):
    pass


class NativeBridge:
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir.resolve()
        self._dll_directory = None
        self._dll = None

    def load(self) -> None:
        if self._dll is not None:
            return

        dll_path = self.runtime_dir / "leanmoe_bridge.dll"

        if not dll_path.is_file():
            raise LeanMoENativeError(
                f"LeanMoE bridge not found: {dll_path}"
            )

        self._dll_directory = ctypes.windll.kernel32.AddDllDirectory(
            str(self.runtime_dir)
        )

        self._dll = ctypes.CDLL(str(dll_path))

        self._dll.lm_api_version.argtypes = []
        self._dll.lm_api_version.restype = ctypes.c_uint32

        version = self._dll.lm_api_version()

        if version != BRIDGE_API_VERSION:
            raise LeanMoENativeError(
                f"Bridge API mismatch: expected "
                f"{BRIDGE_API_VERSION}, got {version}"
            )