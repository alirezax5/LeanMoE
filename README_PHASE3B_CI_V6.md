# Phase 3B CI v6

v5 successfully staged `cublas64_13.dll`, but dumpbin showed that this DLL
itself directly depends on `cublasLt64_13.dll`. The runtime loader still failed
at `ggml-cuda.dll`.

v6 stages both CUDA BLAS runtime DLLs from the installed CUDA 13.4 tree:
- cublas64_13.dll
- cublasLt64_13.dll

It also loads `vcomp140.dll`, `cublasLt64_13.dll`, and `cublas64_13.dll`
explicitly before `ggml-cuda.dll` in the diagnostic sequence.

No LeanMoE bridge/native source is changed.
