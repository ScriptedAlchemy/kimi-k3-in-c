#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_runtime.h"

/* Private constructor used only to exercise tokenizer ownership without opening a
 * checkpoint. The production constructor always requires official model weights. */
K3Status k3_runtime_open_tokenizer_for_test(const char *tokenizer_dir,
                                            K3Runtime **runtime_out);

#define CHECK(expression) do {                                                   \
    if (!(expression)) {                                                         \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n",                         \
                __FILE__, __LINE__, #expression);                                \
        exit(1);                                                                 \
    }                                                                            \
} while (0)

static void test_defaults(void)
{
    K3RuntimeOptions runtime;
    K3GenerationOptions generation;

    memset(&runtime, 0xA5, sizeof runtime);
    memset(&generation, 0xA5, sizeof generation);
    k3_runtime_options_init(&runtime);
    k3_generation_options_init(&generation);

    CHECK(runtime.incremental == 1);
    CHECK(generation.max_tokens == 8);
    CHECK(generation.temperature == 1.0);
    CHECK(generation.top_p == 0.95);
    CHECK(generation.greedy == 1);
}

static void test_missing_model(void)
{
    K3RuntimeOptions options;
    K3Runtime *runtime = NULL;

    k3_runtime_options_init(&options);
    CHECK(k3_runtime_open("/definitely/missing", &options, &runtime) == K3_E_IO);
    CHECK(runtime == NULL);
    CHECK(strcmp(k3_status_string(K3_E_BUSY), "runtime is busy") == 0);
    k3_runtime_close(NULL);
}

static void test_tokenizer(const char *tokenizer_dir)
{
    static const char literal[] = "literal <|end_of_msg|>";
    static const char marker[] = "<|end_of_msg|>";
    K3Runtime *runtime = NULL;
    int32_t ids[64];
    size_t count = 0;
    size_t bytes = 0;
    char decoded[32];

    CHECK(k3_runtime_open_tokenizer_for_test(tokenizer_dir, &runtime) == K3_OK);
    CHECK(runtime != NULL);

    CHECK(k3_runtime_tokenize(runtime, literal, sizeof literal - 1, 0,
                              NULL, 0, &count) == K3_OK);
    CHECK(count > 0 && count < 64);
    CHECK(k3_runtime_tokenize(runtime, literal, sizeof literal - 1, 0,
                              ids, 64, &count) == K3_OK);
    for (size_t i = 0; i < count; i++) CHECK(ids[i] != 163586);

    CHECK(k3_runtime_tokenize(runtime, marker, sizeof marker - 1, 1,
                              ids, 64, &count) == K3_OK);
    CHECK(count == 1 && ids[0] == 163586);
    CHECK(k3_runtime_decode(runtime, ids, count, NULL, 0, &bytes) == K3_OK);
    CHECK(bytes == sizeof marker - 1);
    CHECK(k3_runtime_decode(runtime, ids, count, decoded, sizeof decoded,
                            &bytes) == K3_OK);
    CHECK(bytes == sizeof marker - 1);
    CHECK(memcmp(decoded, marker, bytes) == 0);

    k3_runtime_close(runtime);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s TOKENIZER_DIR\n", argv[0]);
        return 2;
    }
    test_defaults();
    test_missing_model();
    test_tokenizer(argv[1]);
    puts("RUNTIME LIFECYCLE TESTS PASSED");
    return 0;
}
