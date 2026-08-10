#ifndef K3_FORWARD_H
#define K3_FORWARD_H

#include "k3.h"
#include "k3_bind.h"
#include "k3_cache.h"
#include "k3_trunk.h"

typedef struct {
    K3LayerBind *layers;
    K3ModelBind model;
    int n_bound;
    K3Trunk *trunk;
    float *kv_cache;
    float *rope_cache;
    int *mla_slot;
    int n_mla;
    int kv_capacity;
    int cached;
    int draft_mode;
} K3Weights;

int k3_forward(K3Weights *weights, const K3Cfg *cfg, K3Cache *cache,
               const int32_t *ids, int token_count, float *last_logits,
               float *scratch, float *hidden, float *attn_res,
               float *recurrent, int32_t *all_argmax);

#endif
