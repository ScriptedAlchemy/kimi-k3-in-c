#ifndef K3_RUNTIME_H
#define K3_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct K3Runtime K3Runtime;

typedef enum {
    K3_OK = 0,
    K3_E_IO = -1,
    K3_E_FORMAT = -2,
    K3_E_MEMORY = -3,
    K3_E_ARGUMENT = -4,
    K3_E_BUSY = -5,
    K3_E_CANCELLED = -6,
    K3_E_ENGINE = -7
} K3Status;

typedef struct {
    const char *trunk_dir;
    const char *tokenizer_dir;
    const char *config_path;
    uint64_t trunk_budget_bytes;
    uint64_t expert_cache_bytes;
    uint32_t context_tokens;
    int32_t layers;
    int32_t incremental;
} K3RuntimeOptions;

typedef struct {
    uint32_t max_tokens;
    double temperature;
    double top_p;
    uint64_t seed;
    int32_t greedy;
    const int32_t *stop_token_ids;
    size_t stop_token_count;
} K3GenerationOptions;

typedef struct {
    uint32_t index;
    int32_t token_id;
    const char *piece;
    size_t piece_bytes;
    double seconds;
    uint64_t expert_bytes_read;
} K3Token;

typedef struct {
    uint64_t prompt_tokens;
    uint64_t completion_tokens;
    uint64_t expert_hits;
    uint64_t expert_misses;
    uint64_t expert_bytes_read;
    double seconds_total;
    double seconds_io;
} K3Usage;

typedef struct {
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t top_k;
    uint32_t hidden;
    uint32_t vocabulary;
    uint32_t context_tokens;
    const char *model_family;
    const char *weight_format;
} K3RuntimeInfo;

typedef int (*K3TokenCallback)(const K3Token *token, void *user);

void k3_runtime_options_init(K3RuntimeOptions *options);
void k3_generation_options_init(K3GenerationOptions *options);
K3Status k3_runtime_open(const char *model_dir,
                         const K3RuntimeOptions *options,
                         K3Runtime **runtime_out);
void k3_runtime_close(K3Runtime *runtime);
K3Status k3_runtime_info(K3Runtime *runtime, K3RuntimeInfo *info);
K3Status k3_runtime_tokenize(K3Runtime *runtime, const char *text,
                             size_t text_bytes, int allow_special,
                             int32_t *ids, size_t capacity, size_t *needed);
K3Status k3_runtime_decode(K3Runtime *runtime, const int32_t *ids, size_t count,
                           char *text, size_t capacity, size_t *needed);
K3Status k3_runtime_reset(K3Runtime *runtime);
K3Status k3_runtime_generate(K3Runtime *runtime, const int32_t *prompt,
                             size_t prompt_tokens,
                             const K3GenerationOptions *options,
                             K3TokenCallback callback, void *user,
                             K3Usage *usage);
K3Status k3_runtime_cancel(K3Runtime *runtime);
const char *k3_runtime_last_error(const K3Runtime *runtime);
const char *k3_status_string(K3Status status);

#ifdef __cplusplus
}
#endif

#endif
