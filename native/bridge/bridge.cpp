#define LEANMOE_BRIDGE_BUILD
#include "bridge.h"

#include "llama.h"

#include <exception>
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
         * NOT mapped in Phase 1B:
         *
         * config->use_mmap
         * config->use_mlock
         *
         * These LeanMoE settings remain reserved until their exact
         * implementation path is defined and tested.
         */

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
