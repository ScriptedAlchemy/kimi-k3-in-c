#ifndef K3_RUNTIME_INTERNAL_H
#define K3_RUNTIME_INTERNAL_H

#include <stddef.h>

#include "k3_cache.h"
#include "k3_cfg.h"
#include "k3_forward.h"
#include "k3_generate.h"
#include "k3_runtime.h"
#include "k3_st.h"
#include "k3_tok.h"
#include "k3_trunk.h"

struct K3Runtime {
    K3Cfg cfg;
    int full_attn[128];
    K3St shards;
    K3Trunk trunk;
    K3Cache cache;
    K3Weights weights;
    Tok tokenizer;

    float *hidden;
    float *attn_res;
    float *recurrent;
    float *scratch;
    float *logits;
    int32_t *sequence;

    size_t hidden_floats;
    size_t attn_res_floats;
    size_t recurrent_floats;
    size_t scratch_floats;
    size_t kv_floats;
    size_t rope_floats;
    size_t sequence_capacity;
    uint32_t context_tokens;

    volatile int cancel;
    int busy;
    long expert_drops_before;
    int have_config;
    int have_shards;
    int have_trunk;
    int have_cache;
    int have_model;
    int have_tokenizer;
    int resident_layers_bound;
    char error[512];
};

K3Status k3_runtime_open_tokenizer_for_test(const char *tokenizer_dir,
                                            K3Runtime **runtime_out);

#endif
