#include "llama.h"

#include <cstdint>
#include <type_traits>

static_assert(sizeof(llama_token) == sizeof(int32_t));
static_assert(sizeof(llama_pos) == sizeof(int32_t));
static_assert(sizeof(llama_seq_id) == sizeof(int32_t));

int main() {
    const llama_vocab * vocab = nullptr;
    llama_context * ctx = nullptr;

    llama_token tokens[4] = {};

    (void) llama_tokenize(
        vocab,
        "test",
        4,
        tokens,
        4,
        false,
        false
    );

    llama_batch batch = llama_batch_init(4, 0, 1);

    batch.n_tokens = 1;
    batch.token[0] = tokens[0];
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;

    (void) llama_decode(ctx, batch);
    (void) llama_get_logits_ith(ctx, -1);

    char piece[256];
    (void) llama_token_to_piece(
        vocab,
        tokens[0],
        piece,
        sizeof(piece),
        0,
        false
    );

    llama_batch_free(batch);

    return 0;
}
