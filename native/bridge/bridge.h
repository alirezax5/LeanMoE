#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
    #if defined(LEANMOE_BRIDGE_BUILD)
        #define LM_API __declspec(dllexport)
    #else
        #define LM_API __declspec(dllimport)
    #endif
#else
    #define LM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LM_BRIDGE_API_VERSION 3u

typedef void * lm_model_t;

typedef enum lm_result {
    LM_OK                     = 0,
    LM_ERROR_INVALID_ARGUMENT = -1,
    LM_ERROR_NOT_INITIALIZED  = -2,
    LM_ERROR_LOAD_MODEL       = -3,
    LM_ERROR_OUT_OF_MEMORY    = -7,
    LM_ERROR_BACKEND          = -8,
    LM_ERROR_INTERNAL         = -100,
    LM_ERROR_BUFFER_TOO_SMALL = -101
} lm_result;

typedef struct lm_model_config {
    int32_t n_gpu_layers;

    /*
     * Reserved for LeanMoE MoE placement.
     *
     * IMPORTANT:
     * This is NOT mapped directly to llama_model_params in Phase 1B.
     * llama.cpp commit cea74625f does not expose n_cpu_moe as a
     * simple field in llama_model_params.
     */
    int32_t n_cpu_moe;

    uint8_t use_mmap;
    uint8_t use_mlock;

    uint8_t reserved[6];
} lm_model_config;


/* Runtime */

LM_API uint32_t lm_api_version(void);

LM_API lm_result lm_init(void);

LM_API void lm_shutdown(void);

LM_API const char * lm_last_error(void);


/* Model */

LM_API lm_result lm_model_load(
    const char * path_utf8,
    const lm_model_config * config,
    lm_model_t * out_model
);

LM_API void lm_model_free(
    lm_model_t model
);


/* Context */

typedef void * lm_context_t;

typedef enum lm_kv_type {
    LM_KV_F16  = 0,
    LM_KV_Q8_0 = 1
} lm_kv_type;

typedef struct lm_context_config {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;

    lm_kv_type type_k;
    lm_kv_type type_v;

    uint8_t flash_attn;
    uint8_t offload_kqv;

    uint8_t reserved[6];
} lm_context_config;

LM_API lm_result lm_context_create(
    lm_model_t model,
    const lm_context_config * config,
    lm_context_t * out_context
);

LM_API void lm_context_free(
    lm_context_t context
);

LM_API uint32_t lm_context_n_ctx(
    lm_context_t context
);

LM_API uint32_t lm_context_n_batch(
    lm_context_t context
);

LM_API uint32_t lm_context_n_ubatch(
    lm_context_t context
);

/* Inference */

typedef int32_t lm_token;

LM_API int32_t lm_vocab_size(
    lm_model_t model
);

/*
 * Tokenize UTF-8 text.
 *
 * On success:
 *   returns LM_OK and writes the token count to out_count.
 *
 * If tokens is NULL or token_capacity is too small:
 *   returns LM_ERROR_BUFFER_TOO_SMALL and writes the required
 *   token count to out_count.
 */
LM_API lm_result lm_tokenize(
    lm_model_t model,
    const char * text_utf8,
    uint8_t add_special,
    uint8_t parse_special,
    lm_token * tokens,
    int32_t token_capacity,
    int32_t * out_count
);

/*
 * Decode one single-sequence token batch.
 *
 * start_pos is the position assigned to tokens[0].
 * Logits are requested only for the final token.
 *
 * llama_decode() warning/error status is returned separately
 * through out_decode_status.
 */
LM_API lm_result lm_decode_tokens(
    lm_context_t context,
    const lm_token * tokens,
    int32_t token_count,
    int32_t start_pos,
    int32_t * out_decode_status
);

/*
 * Return the token with the largest logit from the most recent
 * decode output. This is a deterministic diagnostic primitive,
 * not LeanMoE's final sampling implementation.
 */
LM_API lm_result lm_argmax_token(
    lm_model_t model,
    lm_context_t context,
    lm_token * out_token
);

/*
 * Convert one token to its UTF-8 byte representation.
 *
 * out_size receives the number of bytes excluding the optional
 * NUL terminator added by the bridge.
 */
LM_API lm_result lm_token_to_piece(
    lm_model_t model,
    lm_token token,
    uint8_t render_special,
    char * buffer,
    int32_t buffer_size,
    int32_t * out_size
);

/* Metadata */

LM_API uint64_t lm_model_size(
    lm_model_t model
);

LM_API uint64_t lm_model_n_params(
    lm_model_t model
);

LM_API lm_result lm_model_description(
    lm_model_t model,
    char * buffer,
    size_t buffer_size
);


#ifdef __cplusplus
}
#endif
