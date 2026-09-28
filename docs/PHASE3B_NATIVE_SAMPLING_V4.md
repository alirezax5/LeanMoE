# LeanMoE Phase 3B — Native Sampling ABI v4

## Scope

This package is the first controlled step of Phase 3B. It changes only the
native bridge sampling surface and adds a compile/link gate.

Bridge API: **v4**

Existing model/context/tokenize/decode/argmax/token-to-piece APIs remain intact.

## New ABI

- `lm_sampler_t`
- `lm_sampler_config`
- `lm_sampler_create`
- `lm_sampler_free`
- `lm_sampler_sample`
- `lm_sampler_reset`

## Sampling chain

Non-greedy:

`top-k -> top-p -> min-p -> temperature -> dist(seed)`

Greedy:

`greedy`

Filters that are configured as no-ops are omitted.

## Important pinned llama.cpp behavior

This implementation targets exactly:

- repository: `ggml-org/llama.cpp`
- commit: `cea74625f`
- build: `11213`

At this commit, `llama_sampler_sample(sampler, ctx, -1)` selects and accepts
the sampled token internally. LeanMoE therefore MUST NOT call
`llama_sampler_accept()` again for the same sampled token.

## Validation order

1. Replace only `native/bridge/bridge.h` and `bridge.cpp`.
2. Run CUDA CI compile/link.
3. Confirm `lm_api_version() == 4`.
4. Only after CI is green, update Python ctypes binding.
5. Then add deterministic seed, greedy parity, and streaming runtime tests.

Do not change the frozen 41/30 placement or context baseline in this phase.
