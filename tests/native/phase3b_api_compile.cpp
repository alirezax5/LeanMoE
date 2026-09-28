#include "bridge.h"

#include <cstdint>
#include <type_traits>

static_assert(LM_BRIDGE_API_VERSION == 4u, "Phase 3B requires bridge API v4");
static_assert(std::is_same<lm_sampler_t, void *>::value, "sampler handle ABI changed");

int main() {
    lm_sampler_config cfg{};
    cfg.temperature = 0.8f;
    cfg.top_k = 40;
    cfg.top_p = 0.95f;
    cfg.min_p = 0.05f;
    cfg.seed = 1234u;
    cfg.greedy = 0;

    lm_sampler_t sampler = nullptr;

    // Compile/link surface only. Runtime creation requires lm_init().
    (void) cfg;
    (void) sampler;

    auto * create_fn = &lm_sampler_create;
    auto * free_fn   = &lm_sampler_free;
    auto * sample_fn = &lm_sampler_sample;
    auto * reset_fn  = &lm_sampler_reset;

    (void) create_fn;
    (void) free_fn;
    (void) sample_fn;
    (void) reset_fn;

    return 0;
}
