#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "k3_runtime.h"

void k3_runtime_options_init(K3RuntimeOptions *options)
{
    if (options == NULL) return;
    memset(options, 0, sizeof(*options));
    options->incremental = 1;
}

void k3_generation_options_init(K3GenerationOptions *options)
{
    if (options == NULL) return;
    memset(options, 0, sizeof(*options));
    options->max_tokens = 8;
    options->temperature = 1.0;
    options->top_p = 0.95;
    options->greedy = 1;
}

K3Status k3_runtime_open(const char *model_dir,
                         const K3RuntimeOptions *options,
                         K3Runtime **runtime_out)
{
    if (runtime_out != NULL) *runtime_out = NULL;
    return K3_E_ENGINE;
}

void k3_runtime_close(K3Runtime *runtime) { }

K3Status k3_runtime_info(K3Runtime *runtime, K3RuntimeInfo *info)
{
    return K3_E_ENGINE;
}

K3Status k3_runtime_tokenize(K3Runtime *runtime, const char *text,
                             size_t text_bytes, int allow_special,
                             int32_t *ids, size_t capacity, size_t *needed)
{
    if (needed != NULL) *needed = 0;
    return K3_E_ENGINE;
}

K3Status k3_runtime_decode(K3Runtime *runtime, const int32_t *ids, size_t count,
                           char *text, size_t capacity, size_t *needed)
{
    if (needed != NULL) *needed = 0;
    return K3_E_ENGINE;
}

K3Status k3_runtime_reset(K3Runtime *runtime) { return K3_E_ENGINE; }

K3Status k3_runtime_generate(K3Runtime *runtime, const int32_t *prompt,
                             size_t prompt_tokens,
                             const K3GenerationOptions *options,
                             K3TokenCallback callback, void *user,
                             K3Usage *usage)
{
    return K3_E_ENGINE;
}

K3Status k3_runtime_cancel(K3Runtime *runtime) { return K3_E_ENGINE; }

const char *k3_runtime_last_error(const K3Runtime *runtime)
{
    return "runtime implementation pending";
}

const char *k3_status_string(K3Status status)
{
    return status == K3_OK ? "ok" : "engine error";
}

static int on_token(const K3Token *token, void *user)
{
    size_t *n = (size_t *)user;
    if (token->token_id >= 0) (*n)++;
    return 1;
}

int main(void)
{
    K3Runtime *runtime = NULL;
    K3RuntimeOptions options;
    K3GenerationOptions generation;
    K3Usage usage;
    K3RuntimeInfo info;
    size_t count = 0, needed = 0;
    int32_t ids[8];

    k3_runtime_options_init(&options);
    k3_generation_options_init(&generation);
    (void)k3_runtime_open("model", &options, &runtime);
    (void)k3_runtime_info(runtime, &info);
    (void)k3_runtime_tokenize(runtime, "x", 1, 0, ids, 8, &needed);
    (void)k3_runtime_decode(runtime, ids, needed, NULL, 0, &needed);
    (void)k3_runtime_reset(runtime);
    (void)k3_runtime_generate(runtime, ids, needed, &generation,
                              on_token, &count, &usage);
    (void)k3_runtime_cancel(runtime);
    (void)k3_runtime_last_error(runtime);
    k3_runtime_close(runtime);
    return count == SIZE_MAX;
}
