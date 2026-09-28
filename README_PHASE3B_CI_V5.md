# Phase 3B CI v5

v4 isolated the runtime loader failure to `ggml-cuda.dll`.

The log proves:
- `ggml-base.dll` loads.
- `ggml-cpu.dll` loads.
- `ggml-cuda.dll` fails.
- `ggml-cuda.dll` directly depends on `cublas64_13.dll`.
- the runner diagnostic could not resolve `cublas64_13.dll`.
- `ggml-base.dll` and `ggml-cpu.dll` also declare `VCOMP140.DLL`.

v5 leaves LeanMoE native source untouched. It locates and stages:
- `cublas64_13.dll` from the installed CUDA 13.4 tree.
- `vcomp140.dll` from the Visual Studio x64 redistributable tree.

The v4 per-DLL loader diagnostic remains enabled.
