#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "k3_runtime_internal.h"

static void k3_set_error(K3Runtime *runtime, const char *format, ...)
{
    if (!runtime) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(runtime->error, sizeof runtime->error, format, arguments);
    va_end(arguments);
}

static int k3_size_product(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int k3_readable_file(const char *directory, const char *name)
{
    char path[4096];
    const int n = snprintf(path, sizeof path, "%s/%s", directory, name);
    return n > 0 && (size_t)n < sizeof path && access(path, R_OK) == 0;
}

static K3Status k3_runtime_load_tokenizer(K3Runtime *runtime,
                                          const char *tokenizer_dir)
{
    if (!tokenizer_dir || !*tokenizer_dir) {
        k3_set_error(runtime, "tokenizer directory is empty");
        return K3_E_ARGUMENT;
    }
    if (!k3_readable_file(tokenizer_dir, "tiktoken.model") ||
        !k3_readable_file(tokenizer_dir, "tokenizer_config.json")) {
        k3_set_error(runtime, "tokenizer files are missing under %s", tokenizer_dir);
        return K3_E_IO;
    }
    k3_tok_load(&runtime->tokenizer, tokenizer_dir);
    runtime->have_tokenizer = 1;
    return K3_OK;
}

void k3_runtime_options_init(K3RuntimeOptions *options)
{
    if (!options) return;
    memset(options, 0, sizeof *options);
    options->trunk_budget_bytes = UINT64_C(110000000000);
    options->expert_cache_bytes = UINT64_C(13000000000);
    options->context_tokens = 512;
    options->incremental = 1;
}

void k3_generation_options_init(K3GenerationOptions *options)
{
    if (!options) return;
    memset(options, 0, sizeof *options);
    options->max_tokens = 8;
    options->temperature = 1.0;
    options->top_p = 0.95;
    options->greedy = 1;
}

static K3Status k3_runtime_allocate_buffers(K3Runtime *runtime, int incremental)
{
    const K3Cfg *cfg = &runtime->cfg;
    const size_t context = runtime->context_tokens;
    const size_t layers = (size_t)runtime->weights.n_bound;
    const size_t hidden = (size_t)cfg->hidden;
    const size_t max_blocks = (size_t)(cfg->n_layers / cfg->attn_res_block + 2);
    const size_t channels = (size_t)cfg->kda_heads * cfg->kda_head_dim;
    size_t per_layer = 0, temporary = 0;

    if (!k3_size_product(channels, (size_t)cfg->kda_head_dim, &per_layer) ||
        !k3_size_product((size_t)3 * channels, (size_t)(cfg->conv_k - 1), &temporary) ||
        per_layer > SIZE_MAX - temporary) {
        k3_set_error(runtime, "recurrent-state size overflows size_t");
        return K3_E_MEMORY;
    }
    per_layer += temporary;
    if (!k3_size_product(per_layer, layers, &runtime->recurrent_floats) ||
        !k3_size_product(context, hidden, &runtime->hidden_floats) ||
        !k3_size_product(context, max_blocks, &temporary) ||
        !k3_size_product(temporary, hidden, &runtime->attn_res_floats)) {
        k3_set_error(runtime, "request-buffer size overflows size_t");
        return K3_E_MEMORY;
    }

    runtime->scratch_floats = k3_layer_scratch(cfg, (int)context);
    if (incremental) {
        const size_t cached = k3_mla_scratch_cached(cfg, (int)context,
                                                    (int)context, 1);
        if (cached > runtime->scratch_floats) runtime->scratch_floats = cached;
    }

    runtime->weights.mla_slot = (int *)malloc(layers * sizeof(int));
    if (!runtime->weights.mla_slot) goto memory_error;
    runtime->weights.n_mla = 0;
    for (int layer = 0; layer < runtime->weights.n_bound; layer++) {
        runtime->weights.mla_slot[layer] =
            k3_is_mla(cfg, layer) ? runtime->weights.n_mla++ : -1;
    }

    if (incremental) {
        size_t per_position = 0;
        if (!k3_size_product((size_t)cfg->n_heads,
                             (size_t)(cfg->qk_nope + cfg->v_head),
                             &per_position) ||
            !k3_size_product(per_position, context, &temporary) ||
            !k3_size_product(temporary, (size_t)runtime->weights.n_mla,
                             &runtime->kv_floats) ||
            !k3_size_product((size_t)cfg->qk_rope, context, &temporary) ||
            !k3_size_product(temporary, (size_t)runtime->weights.n_mla,
                             &runtime->rope_floats)) {
            k3_set_error(runtime, "KV-cache size overflows size_t");
            return K3_E_MEMORY;
        }
        runtime->weights.kv_capacity = (int)context;
        runtime->weights.kv_cache =
            (float *)calloc(runtime->kv_floats ? runtime->kv_floats : 1,
                            sizeof(float));
        runtime->weights.rope_cache =
            (float *)calloc(runtime->rope_floats ? runtime->rope_floats : 1,
                            sizeof(float));
        if (!runtime->weights.kv_cache || !runtime->weights.rope_cache)
            goto memory_error;
    }

    runtime->hidden = (float *)malloc(runtime->hidden_floats * sizeof(float));
    runtime->attn_res = (float *)malloc(runtime->attn_res_floats * sizeof(float));
    runtime->recurrent =
        (float *)calloc(runtime->recurrent_floats, sizeof(float));
    runtime->scratch = (float *)malloc(runtime->scratch_floats * sizeof(float));
    runtime->logits = (float *)malloc((size_t)cfg->vocab * sizeof(float));
    runtime->sequence = (int32_t *)malloc(context * sizeof(int32_t));
    runtime->sequence_capacity = context;
    if (!runtime->hidden || !runtime->attn_res || !runtime->recurrent ||
        !runtime->scratch || !runtime->logits || !runtime->sequence)
        goto memory_error;
    return K3_OK;

memory_error:
    k3_set_error(runtime, "unable to allocate runtime buffers for %u tokens",
                 runtime->context_tokens);
    return K3_E_MEMORY;
}

K3Status k3_runtime_open(const char *model_dir,
                         const K3RuntimeOptions *options,
                         K3Runtime **runtime_out)
{
    K3RuntimeOptions defaults;
    struct stat model_stat;
    char config_path[4096];
    K3Runtime *runtime = NULL;
    K3Status status = K3_E_ENGINE;

    if (!runtime_out) return K3_E_ARGUMENT;
    *runtime_out = NULL;
    if (!model_dir || !*model_dir) return K3_E_ARGUMENT;
    if (stat(model_dir, &model_stat) != 0 || !S_ISDIR(model_stat.st_mode))
        return K3_E_IO;

    k3_runtime_options_init(&defaults);
    if (!options) options = &defaults;
    if (options->layers < 0 || options->context_tokens > (uint32_t)INT_MAX ||
        options->trunk_budget_bytes > (uint64_t)INT64_MAX ||
        options->expert_cache_bytes > (uint64_t)INT64_MAX)
        return K3_E_ARGUMENT;

    runtime = (K3Runtime *)calloc(1, sizeof *runtime);
    if (!runtime) return K3_E_MEMORY;
    runtime->trunk.fd = -1;
    runtime->context_tokens = options->context_tokens ? options->context_tokens : 512;

    if (options->config_path) {
        if (strlen(options->config_path) >= sizeof config_path) {
            status = K3_E_ARGUMENT;
            k3_set_error(runtime, "config path is too long");
            goto fail;
        }
        memcpy(config_path, options->config_path,
               strlen(options->config_path) + 1);
    } else {
        const int n = snprintf(config_path, sizeof config_path,
                               "%s/config.json", model_dir);
        if (n < 0 || (size_t)n >= sizeof config_path) {
            status = K3_E_ARGUMENT;
            k3_set_error(runtime, "model directory path is too long");
            goto fail;
        }
    }
    if (access(config_path, R_OK) != 0) {
        status = K3_E_IO;
        k3_set_error(runtime, "cannot read config %s", config_path);
        goto fail;
    }
    if (!k3_cfg_load_file(&runtime->cfg, runtime->full_attn,
                          (int)(sizeof runtime->full_attn /
                                sizeof runtime->full_attn[0]),
                          config_path)) {
        status = K3_E_FORMAT;
        k3_set_error(runtime, "model config is invalid: %s", config_path);
        goto fail;
    }
    runtime->have_config = 1;

    const int layer_count = options->layers ? options->layers : runtime->cfg.n_layers;
    if (layer_count <= 0 || layer_count > runtime->cfg.n_layers) {
        status = K3_E_ARGUMENT;
        k3_set_error(runtime, "requested %d layers from a %d-layer model",
                     layer_count, runtime->cfg.n_layers);
        goto fail;
    }

    if (k3_st_open(&runtime->shards, model_dir) != 0) {
        status = K3_E_IO;
        k3_set_error(runtime, "cannot open safetensors under %s", model_dir);
        goto fail;
    }
    runtime->have_shards = 1;

    runtime->weights.layers =
        (K3LayerBind *)calloc((size_t)layer_count, sizeof(K3LayerBind));
    if (!runtime->weights.layers) {
        status = K3_E_MEMORY;
        k3_set_error(runtime, "cannot allocate layer bindings");
        goto fail;
    }
    runtime->weights.n_bound = layer_count;

    for (int layer = 0; layer < layer_count; layer++) {
        if (k3_bind_layer_bytes(&runtime->shards, &runtime->cfg, layer) < 0) {
            status = K3_E_FORMAT;
            k3_set_error(runtime, "layer %d has an incomplete tensor set", layer);
            goto fail;
        }
        if (!k3_is_dense(&runtime->cfg, layer)) {
            K3ExpertRef expert;
            for (int index = 0; index < runtime->cfg.n_experts; index++) {
                if (k3_expert_ref(&runtime->shards, layer, index, &expert) != 0) {
                    status = K3_E_FORMAT;
                    k3_set_error(runtime,
                                 "layer %d expert %d has an incomplete MXFP4 tensor set",
                                 layer, index);
                    goto fail;
                }
            }
        }
    }

    if (options->trunk_dir) {
        if (k3_trunk_open(&runtime->trunk, options->trunk_dir, &runtime->cfg,
                          (int64_t)options->trunk_budget_bytes) != 0) {
            status = K3_E_IO;
            k3_set_error(runtime, "cannot open packed BF16 trunk at %s",
                         options->trunk_dir);
            goto fail;
        }
        runtime->have_trunk = 1;
        if (runtime->trunk.n_layers < layer_count) {
            status = K3_E_FORMAT;
            k3_set_error(runtime, "packed trunk has %d layers; %d required",
                         runtime->trunk.n_layers, layer_count);
            goto fail;
        }
        runtime->weights.trunk = &runtime->trunk;
    } else {
        for (int layer = 0; layer < layer_count; layer++) {
            if (k3_bind_layer(&runtime->shards, &runtime->cfg, layer,
                              &runtime->weights.layers[layer]) != 0) {
                status = K3_E_FORMAT;
                k3_set_error(runtime, "cannot bind layer %d", layer);
                goto fail;
            }
            runtime->resident_layers_bound = layer + 1;
        }
    }

    if (k3_bind_model(&runtime->shards, &runtime->cfg, 1,
                      &runtime->weights.model) != 0) {
        status = K3_E_FORMAT;
        k3_set_error(runtime, "model-level tensors are incomplete");
        goto fail;
    }
    runtime->have_model = 1;

    if (k3_cache_init(&runtime->cache, &runtime->shards, &runtime->cfg,
                      (int64_t)options->expert_cache_bytes) != 0) {
        status = K3_E_MEMORY;
        k3_set_error(runtime, "cannot initialize the native MXFP4 expert cache");
        goto fail;
    }
    runtime->have_cache = 1;

    if (options->tokenizer_dir) {
        status = k3_runtime_load_tokenizer(runtime, options->tokenizer_dir);
        if (status != K3_OK) goto fail;
    }

    status = k3_runtime_allocate_buffers(runtime, options->incremental != 0);
    if (status != K3_OK) goto fail;
    *runtime_out = runtime;
    return K3_OK;

fail:
    if (runtime->error[0]) fprintf(stderr, "k3_runtime: %s\n", runtime->error);
    k3_runtime_close(runtime);
    return status;
}

K3Status k3_runtime_open_tokenizer_for_test(const char *tokenizer_dir,
                                            K3Runtime **runtime_out)
{
    if (!runtime_out) return K3_E_ARGUMENT;
    *runtime_out = NULL;
    K3Runtime *runtime = (K3Runtime *)calloc(1, sizeof *runtime);
    if (!runtime) return K3_E_MEMORY;
    runtime->trunk.fd = -1;
    const K3Status status = k3_runtime_load_tokenizer(runtime, tokenizer_dir);
    if (status != K3_OK) {
        k3_runtime_close(runtime);
        return status;
    }
    *runtime_out = runtime;
    return K3_OK;
}

void k3_runtime_close(K3Runtime *runtime)
{
    if (!runtime) return;
    free(runtime->hidden);
    free(runtime->attn_res);
    free(runtime->recurrent);
    free(runtime->scratch);
    free(runtime->logits);
    free(runtime->sequence);
    free(runtime->weights.kv_cache);
    free(runtime->weights.rope_cache);
    free(runtime->weights.mla_slot);
    if (runtime->have_tokenizer) k3_tok_free(&runtime->tokenizer);
    if (runtime->have_cache) k3_cache_free(&runtime->cache);
    if (runtime->have_model) k3_bind_model_free(&runtime->weights.model);
    if (runtime->weights.layers) {
        for (int layer = 0; layer < runtime->resident_layers_bound; layer++)
            k3_bind_free(&runtime->weights.layers[layer]);
    }
    free(runtime->weights.layers);
    if (runtime->have_trunk) k3_trunk_close(&runtime->trunk);
    if (runtime->have_shards) k3_st_close(&runtime->shards);
    free(runtime);
}

K3Status k3_runtime_info(K3Runtime *runtime, K3RuntimeInfo *info)
{
    if (!runtime || !info) return K3_E_ARGUMENT;
    if (!runtime->have_config || !runtime->have_model) {
        k3_set_error(runtime, "model is not loaded");
        return K3_E_ENGINE;
    }
    memset(info, 0, sizeof *info);
    info->n_layers = (uint32_t)runtime->weights.n_bound;
    info->n_experts = (uint32_t)runtime->cfg.n_experts;
    info->top_k = (uint32_t)runtime->cfg.topk;
    info->hidden = (uint32_t)runtime->cfg.hidden;
    info->vocabulary = (uint32_t)runtime->cfg.vocab;
    info->context_tokens = runtime->context_tokens;
    info->model_family = "Kimi K3";
    info->weight_format = "official MXFP4 experts + BF16 trunk";
    return K3_OK;
}

K3Status k3_runtime_tokenize(K3Runtime *runtime, const char *text,
                             size_t text_bytes, int allow_special,
                             int32_t *ids, size_t capacity, size_t *needed)
{
    if (!runtime || !needed || (!text && text_bytes != 0) ||
        (!ids && capacity != 0) || text_bytes > (size_t)INT_MAX ||
        capacity > (size_t)INT_MAX)
        return K3_E_ARGUMENT;
    if (!runtime->have_tokenizer) {
        k3_set_error(runtime, "tokenizer is not loaded");
        return K3_E_ENGINE;
    }
    if (!text) text = "";
    const size_t temporary_capacity = text_bytes + 1;
    if (temporary_capacity > (size_t)INT_MAX ||
        temporary_capacity > SIZE_MAX / sizeof(int))
        return K3_E_MEMORY;
    int *temporary =
        (int *)malloc((temporary_capacity ? temporary_capacity : 1) * sizeof(int));
    if (!temporary) return K3_E_MEMORY;
    const int count = tok_encode_mode(&runtime->tokenizer, text, (int)text_bytes,
                                      temporary, (int)temporary_capacity,
                                      allow_special ? 1 : 0);
    *needed = (size_t)count;
    if (capacity < (size_t)count) {
        free(temporary);
        return ids ? K3_E_ARGUMENT : K3_OK;
    }
    for (int i = 0; i < count; i++) ids[i] = (int32_t)temporary[i];
    free(temporary);
    return K3_OK;
}

K3Status k3_runtime_decode(K3Runtime *runtime, const int32_t *ids, size_t count,
                           char *text, size_t capacity, size_t *needed)
{
    if (!runtime || !needed || (!ids && count != 0) ||
        (!text && capacity != 0) || count > (size_t)INT_MAX ||
        capacity > (size_t)INT_MAX)
        return K3_E_ARGUMENT;
    if (!runtime->have_tokenizer) {
        k3_set_error(runtime, "tokenizer is not loaded");
        return K3_E_ENGINE;
    }
    for (size_t i = 0; i < count; i++) {
        if (ids[i] < 0 || ids[i] >= runtime->tokenizer.n_ids ||
            !runtime->tokenizer.id2str[ids[i]]) {
            k3_set_error(runtime, "token id %d is outside the loaded vocabulary", ids[i]);
            return K3_E_ARGUMENT;
        }
    }
    if (count == 0) {
        *needed = 0;
        if (text && capacity) text[0] = 0;
        return K3_OK;
    }

    if (count > (SIZE_MAX - 1) / 8) return K3_E_MEMORY;
    size_t temporary_capacity = count * 8 + 1;
    if (temporary_capacity < 64) temporary_capacity = 64;
    char *temporary = NULL;
    int decoded = 0;
    for (;;) {
        if (temporary_capacity > (size_t)INT_MAX) return K3_E_MEMORY;
        char *grown = (char *)realloc(temporary, temporary_capacity);
        if (!grown) { free(temporary); return K3_E_MEMORY; }
        temporary = grown;
        decoded = tok_decode(&runtime->tokenizer, (const int *)ids, (int)count,
                             temporary, (int)temporary_capacity);
        if ((size_t)decoded < temporary_capacity) break;
        if (temporary_capacity > (size_t)INT_MAX / 2) {
            free(temporary); return K3_E_MEMORY;
        }
        temporary_capacity *= 2;
    }
    *needed = (size_t)decoded;
    if (capacity < (size_t)decoded) {
        free(temporary);
        return text ? K3_E_ARGUMENT : K3_OK;
    }
    if (decoded) memcpy(text, temporary, (size_t)decoded);
    if (capacity > (size_t)decoded) text[decoded] = 0;
    free(temporary);
    return K3_OK;
}

static K3Status k3_runtime_reset_state(K3Runtime *runtime)
{
    if (runtime->recurrent)
        memset(runtime->recurrent, 0,
               runtime->recurrent_floats * sizeof(float));
    if (runtime->weights.kv_cache)
        memset(runtime->weights.kv_cache, 0,
               runtime->kv_floats * sizeof(float));
    if (runtime->weights.rope_cache)
        memset(runtime->weights.rope_cache, 0,
               runtime->rope_floats * sizeof(float));
    runtime->weights.cached = 0;
    runtime->cancel = 0;
    runtime->error[0] = 0;
    if (runtime->have_cache) k3_cache_reset_stats(&runtime->cache);
    if (runtime->have_trunk) {
        runtime->trunk.hits = runtime->trunk.misses = 0;
        runtime->trunk.bytes_read = 0;
        runtime->trunk.load_seconds = 0.0;
    }
    return K3_OK;
}

K3Status k3_runtime_reset(K3Runtime *runtime)
{
    if (!runtime) return K3_E_ARGUMENT;
    if (__sync_val_compare_and_swap(&runtime->busy, 0, 0) != 0)
        return K3_E_BUSY;
    return k3_runtime_reset_state(runtime);
}

static K3Status k3_runtime_generation_prepare(void *opaque)
{
    K3Runtime *runtime = (K3Runtime *)opaque;
    const K3Status status = k3_runtime_reset_state(runtime);
    runtime->expert_drops_before = k3_expert_drops;
    return status;
}

static K3Status k3_runtime_generation_step(void *opaque,
                                           const int32_t *tokens, size_t count,
                                           float **logits, size_t *vocabulary)
{
    K3Runtime *runtime = (K3Runtime *)opaque;
    if (!tokens || count == 0 || count > (size_t)INT_MAX || !logits || !vocabulary)
        return K3_E_ARGUMENT;
    if (k3_forward(&runtime->weights, &runtime->cfg, &runtime->cache,
                   tokens, (int)count, runtime->logits, runtime->scratch,
                   runtime->hidden, runtime->attn_res, runtime->recurrent,
                   NULL) != 0) {
        k3_set_error(runtime, "forward pass failed");
        return K3_E_ENGINE;
    }
    if (runtime->weights.kv_cache)
        runtime->weights.cached += (int)count;
    *logits = runtime->logits;
    *vocabulary = (size_t)runtime->cfg.vocab;
    return K3_OK;
}

static K3Status k3_runtime_generation_decode(void *opaque,
                                             const int32_t *ids, size_t count,
                                             char *text, size_t capacity,
                                             size_t *needed)
{
    return k3_runtime_decode((K3Runtime *)opaque, ids, count,
                             text, capacity, needed);
}

static uint64_t k3_runtime_generation_bytes_read(void *opaque)
{
    const K3Runtime *runtime = (const K3Runtime *)opaque;
    return runtime->cache.bytes_read;
}

static K3Status k3_runtime_generation_finish(void *opaque, K3Status status,
                                             K3Usage *usage)
{
    K3Runtime *runtime = (K3Runtime *)opaque;
    usage->expert_hits = runtime->cache.hits;
    usage->expert_misses = runtime->cache.misses;
    usage->expert_bytes_read = runtime->cache.bytes_read;
    usage->seconds_io = runtime->cache.load_seconds +
                        (runtime->have_trunk ? runtime->trunk.load_seconds : 0.0);
    if (k3_expert_drops != runtime->expert_drops_before) {
        k3_set_error(runtime,
                     "request invalid: one or more routed experts failed to load");
        return K3_E_ENGINE;
    }
    return status;
}

K3Status k3_runtime_generate(K3Runtime *runtime, const int32_t *prompt,
                             size_t prompt_tokens,
                             const K3GenerationOptions *options,
                             K3TokenCallback callback, void *user,
                             K3Usage *usage)
{
    if (!runtime || !prompt || prompt_tokens == 0 || !callback || !usage)
        return K3_E_ARGUMENT;
    if (!runtime->have_model || !runtime->have_cache) {
        k3_set_error(runtime, "model is not loaded");
        return K3_E_ENGINE;
    }
    if (!runtime->have_tokenizer) {
        k3_set_error(runtime, "tokenizer is not loaded");
        return K3_E_ENGINE;
    }

    K3GenerateLoop loop;
    memset(&loop, 0, sizeof loop);
    loop.step = k3_runtime_generation_step;
    loop.decode = k3_runtime_generation_decode;
    loop.prepare = k3_runtime_generation_prepare;
    loop.finish = k3_runtime_generation_finish;
    loop.bytes_read = k3_runtime_generation_bytes_read;
    loop.context = runtime;
    loop.cancel = &runtime->cancel;
    loop.busy = &runtime->busy;
    loop.sequence = runtime->sequence;
    loop.sequence_capacity = runtime->sequence_capacity;
    loop.vocabulary = (size_t)runtime->cfg.vocab;
    loop.incremental = runtime->weights.kv_cache != NULL;
    return k3_generate_loop(&loop, prompt, prompt_tokens, options,
                            callback, user, usage);
}

K3Status k3_runtime_cancel(K3Runtime *runtime)
{
    if (!runtime) return K3_E_ARGUMENT;
    runtime->cancel = 1;
    return K3_OK;
}

const char *k3_runtime_last_error(const K3Runtime *runtime)
{
    if (!runtime) return "runtime is null";
    return runtime->error[0] ? runtime->error : "";
}

const char *k3_status_string(K3Status status)
{
    switch (status) {
        case K3_OK: return "ok";
        case K3_E_IO: return "I/O error";
        case K3_E_FORMAT: return "invalid model format";
        case K3_E_MEMORY: return "out of memory";
        case K3_E_ARGUMENT: return "invalid argument";
        case K3_E_BUSY: return "runtime is busy";
        case K3_E_CANCELLED: return "generation cancelled";
        case K3_E_ENGINE: return "engine error";
        default: return "unknown status";
    }
}
