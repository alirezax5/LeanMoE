#define LEANMOE_BRIDGE_BUILD
#include "bridge.h"

#include "llama.h"
#include "chat.h"

#include <exception>
#include <cstring>
#include <utility>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;
bool g_initialized = false;

void set_error(const char * message) {
    g_last_error = message ? message : "Unknown LeanMoE bridge error";
}

void clear_error() {
    g_last_error.clear();
}

struct lm_model_wrapper {
    llama_model * model = nullptr;
};

struct lm_context_wrapper {
    llama_context * context = nullptr;
};

struct lm_sampler_wrapper {
    llama_sampler * sampler = nullptr;
};

} // namespace


extern "C" {


uint32_t lm_api_version(void) {
    return LM_BRIDGE_API_VERSION;
}


lm_result lm_init(void) {
    clear_error();

    if (g_initialized) {
        return LM_OK;
    }

    try {
        llama_backend_init();
        g_initialized = true;
        return LM_OK;
    }
    catch (const std::exception & exc) {
        set_error(exc.what());
        return LM_ERROR_BACKEND;
    }
    catch (...) {
        set_error("Unknown exception during llama_backend_init()");
        return LM_ERROR_BACKEND;
    }
}


void lm_shutdown(void) {
    clear_error();

    if (!g_initialized) {
        return;
    }

    try {
        llama_backend_free();
    }
    catch (...) {
        set_error("Exception during llama_backend_free()");
    }

    g_initialized = false;
}


const char * lm_last_error(void) {
    return g_last_error.c_str();
}


lm_result lm_model_load(
    const char * path_utf8,
    const lm_model_config * config,
    lm_model_t * out_model
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (path_utf8 == nullptr || path_utf8[0] == '\0') {
        set_error("Model path is empty");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    if (config == nullptr) {
        set_error("Model config is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    if (out_model == nullptr) {
        set_error("Output model pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    *out_model = nullptr;

    try {
        llama_model_params params = llama_model_default_params();

        /*
         * Phase 1B intentionally maps only fields verified against
         * llama.cpp commit cea74625f.
         */
        params.n_gpu_layers = config->n_gpu_layers;

        /*
         * Match llama.cpp --n-cpu-moe semantics:
         * keep expert tensors for the first N transformer blocks on CPU.
         *
         * The pattern matches upstream LLM_FFN_EXPS_REGEX:
         * \.ffn_(up|down|gate|gate_up)_(ch|)exps
         */
        std::vector<std::string> cpu_moe_patterns;
        std::vector<llama_model_tensor_buft_override> tensor_overrides;

        if (config->n_cpu_moe < -1) {
            set_error("n_cpu_moe must be -1 or greater");
            return LM_ERROR_INVALID_ARGUMENT;
        }

        if (config->n_cpu_moe > 0) {
            cpu_moe_patterns.reserve(
                static_cast<size_t>(config->n_cpu_moe)
            );

            tensor_overrides.reserve(
                static_cast<size_t>(config->n_cpu_moe) + 1
            );

            for (int32_t i = 0; i < config->n_cpu_moe; ++i) {
                cpu_moe_patterns.push_back(
                    "blk\\." + std::to_string(i) +
                    "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"
                );
            }

            for (const std::string & pattern : cpu_moe_patterns) {
                tensor_overrides.push_back({
                    pattern.c_str(),
                    ggml_backend_cpu_buffer_type()
                });
            }

            tensor_overrides.push_back({nullptr, nullptr});

            params.tensor_buft_overrides =
                tensor_overrides.data();
        }

        /*
         * Phase 2B.4: map the existing LeanMoE mmap/mlock controls to the
         * pinned llama.cpp load-mode API. The public LeanMoE ABI remains v3.
         */
        if (config->use_mmap != 0) {
            params.load_mode = config->use_mlock != 0
                ? LLAMA_LOAD_MODE_MMAP_MLOCK
                : LLAMA_LOAD_MODE_MMAP;
        } else {
            params.load_mode = config->use_mlock != 0
                ? LLAMA_LOAD_MODE_MLOCK
                : LLAMA_LOAD_MODE_NONE;
        }

        llama_model * model =
            llama_model_load_from_file(path_utf8, params);

        if (model == nullptr) {
            set_error("llama_model_load_from_file() returned null");
            return LM_ERROR_LOAD_MODEL;
        }

        lm_model_wrapper * wrapper = nullptr;

        try {
            wrapper = new lm_model_wrapper();
        }
        catch (const std::bad_alloc &) {
            llama_model_free(model);
            set_error("Failed to allocate LeanMoE model wrapper");
            return LM_ERROR_OUT_OF_MEMORY;
        }

        wrapper->model = model;
        *out_model = static_cast<lm_model_t>(wrapper);

        return LM_OK;
    }
    catch (const std::bad_alloc &) {
        set_error("Out of memory while loading model");
        return LM_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::exception & exc) {
        set_error(exc.what());
        return LM_ERROR_LOAD_MODEL;
    }
    catch (...) {
        set_error("Unknown exception while loading model");
        return LM_ERROR_LOAD_MODEL;
    }
}


void lm_model_free(lm_model_t model) {
    if (model == nullptr) {
        return;
    }

    lm_model_wrapper * wrapper =
        static_cast<lm_model_wrapper *>(model);

    if (wrapper->model != nullptr) {
        llama_model_free(wrapper->model);
        wrapper->model = nullptr;
    }

    delete wrapper;
}

lm_result lm_context_create(
    lm_model_t model,
    const lm_context_config * config,
    lm_context_t * out_context
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (model == nullptr) {
        set_error("Model handle is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    if (config == nullptr) {
        set_error("Context config is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    if (out_context == nullptr) {
        set_error("Output context pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    *out_context = nullptr;

    lm_model_wrapper * model_wrapper =
        static_cast<lm_model_wrapper *>(model);

    if (model_wrapper->model == nullptr) {
        set_error("Internal llama model pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    if (config->n_ctx == 0 ||
        config->n_batch == 0 ||
        config->n_ubatch == 0) {
        set_error("Context sizes must be greater than zero");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    if (config->n_ubatch > config->n_batch) {
        set_error("n_ubatch must not exceed n_batch");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    try {
        llama_context_params params =
            llama_context_default_params();

        params.n_ctx = config->n_ctx;
        params.n_batch = config->n_batch;
        params.n_ubatch = config->n_ubatch;

        switch (config->type_k) {
            case LM_KV_F16:
                params.type_k = GGML_TYPE_F16;
                break;

            case LM_KV_Q8_0:
                params.type_k = GGML_TYPE_Q8_0;
                break;

            default:
                set_error("Unsupported K cache type");
                return LM_ERROR_INVALID_ARGUMENT;
        }

        switch (config->type_v) {
            case LM_KV_F16:
                params.type_v = GGML_TYPE_F16;
                break;

            case LM_KV_Q8_0:
                params.type_v = GGML_TYPE_Q8_0;
                break;

            default:
                set_error("Unsupported V cache type");
                return LM_ERROR_INVALID_ARGUMENT;
        }

        params.flash_attn_type =
    config->flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;

        params.offload_kqv =
            config->offload_kqv != 0;

        llama_context * context =
            llama_init_from_model(
                model_wrapper->model,
                params
            );

        if (context == nullptr) {
            set_error("llama_init_from_model() returned null");
            return LM_ERROR_BACKEND;
        }

        lm_context_wrapper * wrapper = nullptr;

        try {
            wrapper = new lm_context_wrapper();
        }
        catch (const std::bad_alloc &) {
            llama_free(context);
            set_error("Failed to allocate LeanMoE context wrapper");
            return LM_ERROR_OUT_OF_MEMORY;
        }

        wrapper->context = context;

        *out_context =
            static_cast<lm_context_t>(wrapper);

        return LM_OK;
    }
    catch (const std::bad_alloc &) {
        set_error("Out of memory while creating context");
        return LM_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::exception & exc) {
        set_error(exc.what());
        return LM_ERROR_BACKEND;
    }
    catch (...) {
        set_error("Unknown exception while creating context");
        return LM_ERROR_BACKEND;
    }
}


void lm_context_free(lm_context_t context) {
    if (context == nullptr) {
        return;
    }

    lm_context_wrapper * wrapper =
        static_cast<lm_context_wrapper *>(context);

    if (wrapper->context != nullptr) {
        llama_free(wrapper->context);
        wrapper->context = nullptr;
    }

    delete wrapper;
}


uint32_t lm_context_n_ctx(lm_context_t context) {
    if (context == nullptr) {
        return 0;
    }

    const lm_context_wrapper * wrapper =
        static_cast<const lm_context_wrapper *>(context);

    if (wrapper->context == nullptr) {
        return 0;
    }

    return llama_n_ctx(wrapper->context);
}


uint32_t lm_context_n_batch(lm_context_t context) {
    if (context == nullptr) {
        return 0;
    }

    const lm_context_wrapper * wrapper =
        static_cast<const lm_context_wrapper *>(context);

    if (wrapper->context == nullptr) {
        return 0;
    }

    return llama_n_batch(wrapper->context);
}


uint32_t lm_context_n_ubatch(lm_context_t context) {
    if (context == nullptr) {
        return 0;
    }

    const lm_context_wrapper * wrapper =
        static_cast<const lm_context_wrapper *>(context);

    if (wrapper->context == nullptr) {
        return 0;
    }

    return llama_n_ubatch(wrapper->context);
}
int32_t lm_vocab_size(lm_model_t model) {
    if (model == nullptr) {
        return 0;
    }

    const lm_model_wrapper * wrapper =
        static_cast<const lm_model_wrapper *>(model);

    if (wrapper->model == nullptr) {
        return 0;
    }

    const llama_vocab * vocab =
        llama_model_get_vocab(wrapper->model);

    if (vocab == nullptr) {
        return 0;
    }

    return llama_vocab_n_tokens(vocab);
}


lm_result lm_token_is_eog(
    lm_model_t model,
    lm_token token,
    uint8_t * out_is_eog
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }
    if (model == nullptr || out_is_eog == nullptr) {
        set_error("Invalid EOG query argument");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    const lm_model_wrapper * wrapper =
        static_cast<const lm_model_wrapper *>(model);
    if (wrapper->model == nullptr) {
        set_error("Internal llama model pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(wrapper->model);
    if (vocab == nullptr) {
        set_error("llama_model_get_vocab() returned null");
        return LM_ERROR_INTERNAL;
    }

    *out_is_eog = llama_vocab_is_eog(vocab, static_cast<llama_token>(token)) ? 1 : 0;
    return LM_OK;
}


lm_result lm_tokenize(
    lm_model_t model,
    const char * text_utf8,
    uint8_t add_special,
    uint8_t parse_special,
    lm_token * tokens,
    int32_t token_capacity,
    int32_t * out_count
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (model == nullptr ||
        text_utf8 == nullptr ||
        out_count == nullptr ||
        token_capacity < 0) {
        set_error("Invalid tokenization argument");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    lm_model_wrapper * wrapper =
        static_cast<lm_model_wrapper *>(model);

    if (wrapper->model == nullptr) {
        set_error("Internal llama model pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab =
        llama_model_get_vocab(wrapper->model);

    if (vocab == nullptr) {
        set_error("llama_model_get_vocab() returned null");
        return LM_ERROR_INTERNAL;
    }

    const size_t text_size = std::char_traits<char>::length(text_utf8);

    if (text_size >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        set_error("Input text is too large to tokenize");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    const int32_t result = llama_tokenize(
        vocab,
        text_utf8,
        static_cast<int32_t>(text_size),
        reinterpret_cast<llama_token *>(tokens),
        token_capacity,
        add_special != 0,
        parse_special != 0
    );

    if (result == INT32_MIN) {
        set_error("llama_tokenize() overflow");
        return LM_ERROR_INTERNAL;
    }

    if (result < 0) {
        *out_count = -result;
        return LM_ERROR_BUFFER_TOO_SMALL;
    }

    *out_count = result;
    return LM_OK;
}


lm_result lm_decode_tokens(
    lm_context_t context,
    const lm_token * tokens,
    int32_t token_count,
    int32_t start_pos,
    int32_t * out_decode_status
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (context == nullptr ||
        tokens == nullptr ||
        token_count <= 0 ||
        start_pos < 0 ||
        out_decode_status == nullptr) {
        set_error("Invalid decode argument");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    lm_context_wrapper * wrapper =
        static_cast<lm_context_wrapper *>(context);

    if (wrapper->context == nullptr) {
        set_error("Internal llama context pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    llama_batch batch =
        llama_batch_init(token_count, 0, 1);

    if (batch.token == nullptr ||
        batch.pos == nullptr ||
        batch.n_seq_id == nullptr ||
        batch.seq_id == nullptr ||
        batch.logits == nullptr) {
        llama_batch_free(batch);
        set_error("llama_batch_init() failed");
        return LM_ERROR_OUT_OF_MEMORY;
    }

    batch.n_tokens = token_count;

    for (int32_t i = 0; i < token_count; ++i) {
        batch.token[i] =
            static_cast<llama_token>(tokens[i]);

        batch.pos[i] =
            static_cast<llama_pos>(start_pos + i);

        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;

        batch.logits[i] =
            (i == token_count - 1) ? 1 : 0;
    }

    const int32_t status =
        llama_decode(wrapper->context, batch);

    llama_batch_free(batch);

    *out_decode_status = status;

    /*
     * Preserve llama.cpp's decode status exactly.
     * A non-zero status is not automatically a bridge failure.
     */
    return LM_OK;
}


lm_result lm_argmax_token(
    lm_model_t model,
    lm_context_t context,
    lm_token * out_token
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (model == nullptr ||
        context == nullptr ||
        out_token == nullptr) {
        set_error("Invalid argmax argument");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    lm_model_wrapper * model_wrapper =
        static_cast<lm_model_wrapper *>(model);

    lm_context_wrapper * context_wrapper =
        static_cast<lm_context_wrapper *>(context);

    if (model_wrapper->model == nullptr ||
        context_wrapper->context == nullptr) {
        set_error("Internal model or context pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab =
        llama_model_get_vocab(model_wrapper->model);

    if (vocab == nullptr) {
        set_error("llama_model_get_vocab() returned null");
        return LM_ERROR_INTERNAL;
    }

    const int32_t n_vocab =
        llama_vocab_n_tokens(vocab);

    if (n_vocab <= 0) {
        set_error("Invalid vocabulary size");
        return LM_ERROR_INTERNAL;
    }

    float * logits =
        llama_get_logits_ith(context_wrapper->context, -1);

    if (logits == nullptr) {
        set_error("llama_get_logits_ith() returned null");
        return LM_ERROR_INTERNAL;
    }

    int32_t best = 0;

    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }

    *out_token = static_cast<lm_token>(best);

    return LM_OK;
}


lm_result lm_token_to_piece(
    lm_model_t model,
    lm_token token,
    uint8_t render_special,
    char * buffer,
    int32_t buffer_size,
    int32_t * out_size
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (model == nullptr ||
        buffer_size < 0 ||
        out_size == nullptr) {
        set_error("Invalid token-to-piece argument");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    lm_model_wrapper * wrapper =
        static_cast<lm_model_wrapper *>(model);

    if (wrapper->model == nullptr) {
        set_error("Internal llama model pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab =
        llama_model_get_vocab(wrapper->model);

    if (vocab == nullptr) {
        set_error("llama_model_get_vocab() returned null");
        return LM_ERROR_INTERNAL;
    }

    const int32_t result = llama_token_to_piece(
        vocab,
        static_cast<llama_token>(token),
        buffer,
        buffer_size > 0 ? buffer_size - 1 : 0,
        0,
        render_special != 0
    );

    if (result < 0) {
        *out_size = -result;
        return LM_ERROR_BUFFER_TOO_SMALL;
    }

    *out_size = result;

    if (buffer == nullptr || buffer_size <= result) {
        return LM_ERROR_BUFFER_TOO_SMALL;
    }

    buffer[result] = '\0';

    return LM_OK;
}


lm_result lm_sampler_create(
    const lm_sampler_config * config,
    lm_sampler_t * out_sampler
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (config == nullptr || out_sampler == nullptr) {
        set_error("Invalid sampler create argument");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    *out_sampler = nullptr;

    if (config->greedy == 0) {
        if (!(config->temperature > 0.0f)) {
            set_error("temperature must be > 0 for non-greedy sampling");
            return LM_ERROR_INVALID_ARGUMENT;
        }
        if (!(config->top_p > 0.0f && config->top_p <= 1.0f)) {
            set_error("top_p must be in (0, 1]");
            return LM_ERROR_INVALID_ARGUMENT;
        }
        if (!(config->min_p >= 0.0f && config->min_p <= 1.0f)) {
            set_error("min_p must be in [0, 1]");
            return LM_ERROR_INVALID_ARGUMENT;
        }
    }

    llama_sampler * chain = nullptr;

    try {
        chain = llama_sampler_chain_init(
            llama_sampler_chain_default_params()
        );

        if (chain == nullptr) {
            set_error("llama_sampler_chain_init() returned null");
            return LM_ERROR_OUT_OF_MEMORY;
        }

        if (config->greedy != 0) {
            llama_sampler * greedy = llama_sampler_init_greedy();
            if (greedy == nullptr) {
                llama_sampler_free(chain);
                set_error("llama_sampler_init_greedy() returned null");
                return LM_ERROR_OUT_OF_MEMORY;
            }
            llama_sampler_chain_add(chain, greedy);
        } else {
            if (config->top_k > 0) {
                llama_sampler * s = llama_sampler_init_top_k(config->top_k);
                if (s == nullptr) {
                    llama_sampler_free(chain);
                    set_error("llama_sampler_init_top_k() returned null");
                    return LM_ERROR_OUT_OF_MEMORY;
                }
                llama_sampler_chain_add(chain, s);
            }

            if (config->top_p < 1.0f) {
                llama_sampler * s = llama_sampler_init_top_p(config->top_p, 1);
                if (s == nullptr) {
                    llama_sampler_free(chain);
                    set_error("llama_sampler_init_top_p() returned null");
                    return LM_ERROR_OUT_OF_MEMORY;
                }
                llama_sampler_chain_add(chain, s);
            }

            if (config->min_p > 0.0f) {
                llama_sampler * s = llama_sampler_init_min_p(config->min_p, 1);
                if (s == nullptr) {
                    llama_sampler_free(chain);
                    set_error("llama_sampler_init_min_p() returned null");
                    return LM_ERROR_OUT_OF_MEMORY;
                }
                llama_sampler_chain_add(chain, s);
            }

            llama_sampler * temp =
                llama_sampler_init_temp(config->temperature);
            if (temp == nullptr) {
                llama_sampler_free(chain);
                set_error("llama_sampler_init_temp() returned null");
                return LM_ERROR_OUT_OF_MEMORY;
            }
            llama_sampler_chain_add(chain, temp);

            llama_sampler * dist =
                llama_sampler_init_dist(config->seed);
            if (dist == nullptr) {
                llama_sampler_free(chain);
                set_error("llama_sampler_init_dist() returned null");
                return LM_ERROR_OUT_OF_MEMORY;
            }
            llama_sampler_chain_add(chain, dist);
        }

        lm_sampler_wrapper * wrapper = nullptr;
        try {
            wrapper = new lm_sampler_wrapper();
        }
        catch (const std::bad_alloc &) {
            llama_sampler_free(chain);
            set_error("Failed to allocate LeanMoE sampler wrapper");
            return LM_ERROR_OUT_OF_MEMORY;
        }

        wrapper->sampler = chain;
        *out_sampler = static_cast<lm_sampler_t>(wrapper);
        return LM_OK;
    }
    catch (const std::bad_alloc &) {
        if (chain != nullptr) {
            llama_sampler_free(chain);
        }
        set_error("Out of memory while creating sampler");
        return LM_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::exception & exc) {
        if (chain != nullptr) {
            llama_sampler_free(chain);
        }
        set_error(exc.what());
        return LM_ERROR_BACKEND;
    }
    catch (...) {
        if (chain != nullptr) {
            llama_sampler_free(chain);
        }
        set_error("Unknown exception while creating sampler");
        return LM_ERROR_BACKEND;
    }
}


void lm_sampler_free(lm_sampler_t sampler) {
    if (sampler == nullptr) {
        return;
    }

    lm_sampler_wrapper * wrapper =
        static_cast<lm_sampler_wrapper *>(sampler);

    if (wrapper->sampler != nullptr) {
        llama_sampler_free(wrapper->sampler);
        wrapper->sampler = nullptr;
    }

    delete wrapper;
}


lm_result lm_sampler_sample(
    lm_sampler_t sampler,
    lm_context_t context,
    lm_token * out_token
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (sampler == nullptr ||
        context == nullptr ||
        out_token == nullptr) {
        set_error("Invalid sampler sample argument");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    lm_sampler_wrapper * sampler_wrapper =
        static_cast<lm_sampler_wrapper *>(sampler);

    lm_context_wrapper * context_wrapper =
        static_cast<lm_context_wrapper *>(context);

    if (sampler_wrapper->sampler == nullptr ||
        context_wrapper->context == nullptr) {
        set_error("Internal sampler or context pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    try {
        /*
         * In pinned llama.cpp cea74625f, llama_sampler_sample()
         * applies the chain, selects the token, and accepts it internally.
         */
        const llama_token token =
            llama_sampler_sample(
                sampler_wrapper->sampler,
                context_wrapper->context,
                -1
            );

        *out_token = static_cast<lm_token>(token);
        return LM_OK;
    }
    catch (const std::exception & exc) {
        set_error(exc.what());
        return LM_ERROR_BACKEND;
    }
    catch (...) {
        set_error("Unknown exception while sampling token");
        return LM_ERROR_BACKEND;
    }
}


lm_result lm_sampler_reset(lm_sampler_t sampler) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (sampler == nullptr) {
        set_error("Sampler handle is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    lm_sampler_wrapper * wrapper =
        static_cast<lm_sampler_wrapper *>(sampler);

    if (wrapper->sampler == nullptr) {
        set_error("Internal llama sampler pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    try {
        llama_sampler_reset(wrapper->sampler);
        return LM_OK;
    }
    catch (const std::exception & exc) {
        set_error(exc.what());
        return LM_ERROR_BACKEND;
    }
    catch (...) {
        set_error("Unknown exception while resetting sampler");
        return LM_ERROR_BACKEND;
    }
}



lm_result lm_chat_apply_template(
    lm_model_t model,
    const lm_chat_message * messages,
    int32_t message_count,
    uint8_t add_generation_prompt,
    uint8_t enable_thinking,
    char * buffer,
    int32_t buffer_size,
    int32_t * out_size
) {
    clear_error();

    if (!g_initialized) {
        set_error("LeanMoE bridge is not initialized");
        return LM_ERROR_NOT_INITIALIZED;
    }

    if (model == nullptr ||
        message_count <= 0 ||
        messages == nullptr ||
        buffer_size < 0 ||
        out_size == nullptr) {
        set_error("Invalid chat template argument");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    lm_model_wrapper * wrapper =
        static_cast<lm_model_wrapper *>(model);

    if (wrapper->model == nullptr) {
        set_error("Internal llama model pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::vector<common_chat_msg> chat_messages;
        chat_messages.reserve(static_cast<size_t>(message_count));

        for (int32_t i = 0; i < message_count; ++i) {
            if (messages[i].role_utf8 == nullptr ||
                messages[i].content_utf8 == nullptr) {
                set_error("Chat message role/content is null");
                return LM_ERROR_INVALID_ARGUMENT;
            }

            common_chat_msg msg;
            msg.role = messages[i].role_utf8;
            msg.content = messages[i].content_utf8;
            chat_messages.push_back(std::move(msg));
        }

        auto templates =
            common_chat_templates_init(wrapper->model, "");

        if (!templates) {
            set_error("common_chat_templates_init() returned null");
            return LM_ERROR_INTERNAL;
        }

        common_chat_templates_inputs inputs;
        inputs.messages = std::move(chat_messages);
        inputs.add_generation_prompt = add_generation_prompt != 0;
        inputs.use_jinja = true;
        inputs.enable_thinking = enable_thinking != 0;

        const common_chat_params params =
            common_chat_templates_apply(templates.get(), inputs);

        if (params.prompt.size() >
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            set_error("Formatted chat prompt is too large");
            return LM_ERROR_INTERNAL;
        }

        *out_size = static_cast<int32_t>(params.prompt.size());

        if (buffer == nullptr || buffer_size <= *out_size) {
            return LM_ERROR_BUFFER_TOO_SMALL;
        }

        if (*out_size > 0) {
            std::memcpy(buffer, params.prompt.data(), params.prompt.size());
        }
        buffer[*out_size] = '\0';
        return LM_OK;
    }
    catch (const std::bad_alloc &) {
        set_error("Out of memory while applying chat template");
        return LM_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::exception & exc) {
        set_error(exc.what());
        return LM_ERROR_BACKEND;
    }
    catch (...) {
        set_error("Unknown exception while applying chat template");
        return LM_ERROR_BACKEND;
    }
}


uint64_t lm_model_size(lm_model_t model) {
    if (model == nullptr) {
        return 0;
    }

    const lm_model_wrapper * wrapper =
        static_cast<const lm_model_wrapper *>(model);

    if (wrapper->model == nullptr) {
        return 0;
    }

    return llama_model_size(wrapper->model);
}


uint64_t lm_model_n_params(lm_model_t model) {
    if (model == nullptr) {
        return 0;
    }

    const lm_model_wrapper * wrapper =
        static_cast<const lm_model_wrapper *>(model);

    if (wrapper->model == nullptr) {
        return 0;
    }

    return llama_model_n_params(wrapper->model);
}


lm_result lm_model_description(
    lm_model_t model,
    char * buffer,
    size_t buffer_size
) {
    clear_error();

    if (model == nullptr) {
        set_error("Model handle is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    if (buffer == nullptr || buffer_size == 0) {
        set_error("Description buffer is invalid");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    const lm_model_wrapper * wrapper =
        static_cast<const lm_model_wrapper *>(model);

    if (wrapper->model == nullptr) {
        set_error("Internal llama model pointer is null");
        return LM_ERROR_INVALID_ARGUMENT;
    }

    const int32_t result =
        llama_model_desc(wrapper->model, buffer, buffer_size);

    if (result < 0) {
        set_error("llama_model_desc() failed");
        return LM_ERROR_INTERNAL;
    }

    return LM_OK;
}


} // extern "C"
