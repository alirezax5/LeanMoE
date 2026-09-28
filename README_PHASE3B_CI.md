# Phase 3B CI

Copy:

`.github/workflows/phase-3b-sampling.yml`

into the LeanMoE repository.

This workflow intentionally does not modify CMakeLists.txt. It:

1. checks out pinned llama.cpp `cea74625f`
2. uses llama.cpp's Windows CUDA 13.4 setup action
3. builds `leanmoe_bridge`
4. compiles the Phase 3B public-header gate
5. verifies the sampler symbols with `dumpbin`
6. loads the produced DLL and requires `lm_api_version() == 4`
7. uploads `leanmoe-phase-3b-sampling-win64`

Use branch `phase-3b-sampling`.
