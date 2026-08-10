#include <stddef.h>
#include <stdint.h>

#include "k3_runtime.h"

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
