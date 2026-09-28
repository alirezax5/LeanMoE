# Phase 3B CI v3

Fixes the runtime verification failure after the native build/export gates passed.

The previous ctypes check registered only the staged LeanMoE directory. On
Python 3.12/Windows, dependent DLL resolution also needs the CUDA runtime
directory registered explicitly.

v3:
- verifies `leanmoe_bridge.dll` physically exists in the stage;
- prints the staged file list;
- prints `dumpbin /dependents` for the bridge;
- registers both the staged runtime directory and `%CUDA_PATH%\bin` with
  `os.add_dll_directory`;
- uses `ctypes.WinDLL`;
- keeps all successful v2 MSVC/CUDA build steps unchanged.

This is intentionally a CI-only correction. No native bridge source changes.
