#define LEANMOE_BRIDGE_BUILD
#include "bridge.h"

#include "llama.h"

#include <exception>
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
