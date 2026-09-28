# LeanMoE Phase 3B — Step 1

This package contains full replacement copies of the current bridge files:

- `native/bridge/bridge.h`
- `native/bridge/bridge.cpp`

and:

- `tests/native/phase3b_api_compile.cpp`
- `docs/PHASE3B_NATIVE_SAMPLING_V4.md`

## Apply

Back up the current bridge, then copy the files into the same paths in
`C:\LeanMoE`.

This step intentionally does **not** modify `src/leanmoe/native.py` yet.

## CI goal

Build the CUDA bridge against pinned llama.cpp commit `cea74625f`.

The build must prove the v4 sampler API compiles and links before Python is
allowed to depend on it.

Expected API after installing the new CI artifact:

`lm_api_version() == 4`
