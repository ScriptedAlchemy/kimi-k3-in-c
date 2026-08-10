#include <stdio.h>
#include <string.h>

#include "k3_forward.h"

static int argmax_(const float *v, int n)
{ int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

/* One full forward over T tokens, writing logits for the LAST position only. Every
 * step rebuilds state from scratch, matching the path the oracle validates.
 *
 * Returns 0 on success and -1 if the forward could not be completed. The caller MUST
 * check: on failure logits_last is left untouched, and argmaxing an untouched buffer
 * yields a token drawn from uninitialised memory, printed as though it were output. */
/* arg_all: when non-NULL, receives argmax(logits) for EVERY position 0..T-1, which is
 * what batched greedy verification consumes. logits_last still gets the final position's
 * full vector either way. The extra cost is one lm_head matmul per additional position,
 * pure RAM-resident compute; measured, an extra verified position costs ~22% of a serial
 * token at streamed-trunk budgets, which is the entire economics of --spec. */
int k3_forward(K3Weights *weights, const K3Cfg *cfg, K3Cache *cache,
               const int32_t *ids, int token_count, float *last_logits,
               float *scratch, float *hidden, float *attn_res,
               float *recurrent, int32_t *all_argmax)
{
    const int E = cfg->hidden;
    const int maxb = cfg->n_layers / cfg->attn_res_block + 2;
    const int P = cfg->kda_heads * cfg->kda_head_dim;
    const size_t kper = (size_t)P * cfg->kda_head_dim +
                        (size_t)3 * P * (cfg->conv_k - 1);

    for (int t = 0; t < token_count; t++)
        k3_embed_row(hidden + (size_t)t * E, weights->model.embed,
                     weights->model.wdt, ids[t], E);

    memset(attn_res, 0, (size_t)token_count * maxb * E * sizeof(float));
    /* Incremental decode carries the KDA recurrent matrix and ShortConv history across
     * steps, so it must NOT be cleared here; the full-recompute path rebuilds from
     * scratch every step and must be. */
    if (!weights->kv_cache)
        memset(recurrent, 0, kper * (size_t)weights->n_bound * sizeof(float));
    int nb = 0;
    for (int L = 0; L < weights->n_bound; L++) {
        /* Streaming: bring this layer in, and hint the next one so its read overlaps
         * this layer's arithmetic. The order is fixed 0..92 every token, so the hint is
         * never wrong. */
        if (weights->trunk) {
            if (k3_trunk_bind(weights->trunk, cfg, L, &weights->layers[L]) != 0) {
                fprintf(stderr, "trunk bind failed at layer %d\n", L);
                return -1;
            }
            k3_trunk_prefetch(weights->trunk, L + 1);
        }
        /* Point this layer's MoE at the cache before use. Doing it here rather than at
         * bind time keeps K3LayerBind independent of any particular cache. */
        if (weights->layers[L].lay.moe) {
            weights->layers[L].moe.src = &cache->src;
            weights->layers[L].moe.layer = L;
            /* The draft routes only among resident experts, reading zero new expert bytes;
             * the exact model keeps true routing. This is what makes a draft step cheap. */
            weights->layers[L].moe.cache_only = weights->draft_mode;
        }
        if (weights->kv_cache && weights->mla_slot[L] >= 0) {
            const size_t kvper = (size_t)weights->kv_capacity * cfg->n_heads *
                                 (cfg->qk_nope + cfg->v_head);
            const size_t rpper = (size_t)weights->kv_capacity * cfg->qk_rope;
            const int mi = weights->mla_slot[L];
            k3_decoder_layer_inc(hidden, attn_res, &nb, &weights->layers[L].lay,
                                 cfg, L, token_count,
                                 recurrent + kper * (size_t)L, scratch,
                                 weights->kv_cache + kvper * (size_t)mi,
                                 weights->rope_cache + rpper * (size_t)mi,
                                 weights->cached, weights->kv_capacity);
        } else {
            k3_decoder_layer_inc(hidden, attn_res, &nb, &weights->layers[L].lay,
                                 cfg, L, token_count,
                                 recurrent + kper * (size_t)L, scratch,
                                 NULL, NULL, 0, 0);
        }
    }

    /* The model-level aggregator, beyond the two per layer. Exactly one pair exists in
     * the checkpoint; skipping it is silent. */
    if (weights->model.out_res_norm && weights->model.out_res_proj) {
        float *fold = scratch;
        float *src  = fold + E;
        for (int i = 0; i < E; i++)
            fold[i] = weights->model.out_res_norm[i] * weights->model.out_res_proj[i];
        for (int t = 0; t < token_count; t++) {
            for (int b = 0; b < nb; b++)
                memcpy(src + (size_t)b * E,
                       attn_res + ((size_t)t * maxb + b) * E,
                       (size_t)E * sizeof(float));
            memcpy(src + (size_t)nb * E, hidden + (size_t)t * E,
                   (size_t)E * sizeof(float));
            k3_attn_res(hidden + (size_t)t * E, src, fold, nb + 1, E,
                        cfg->rms_eps);
        }
    }

    float *nrm = scratch;
    if (all_argmax) {
        for (int t = 0; t < token_count; t++) {
            k3_rmsnorm(nrm, hidden + (size_t)t * E, weights->model.norm, E,
                       cfg->rms_eps);
            k3_mmw(last_logits, nrm, weights->model.lm_head,
                   weights->model.wdt, E, cfg->vocab);
            all_argmax[t] = argmax_(last_logits, cfg->vocab);
        }
        /* logits_last now holds the FINAL position's vector, same as the plain path. */
        return 0;
    }
    k3_rmsnorm(nrm, hidden + (size_t)(token_count - 1) * E,
               weights->model.norm, E, cfg->rms_eps);
    k3_mmw(last_logits, nrm, weights->model.lm_head, weights->model.wdt, E,
           cfg->vocab);
    return 0;
}
