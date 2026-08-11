#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_generate.h"

#define CHECK(expression) do {                                                   \
    if (!(expression)) {                                                         \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n",                         \
                __FILE__, __LINE__, #expression);                                \
        exit(1);                                                                 \
    }                                                                            \
} while (0)

enum { FAKE_VOCABULARY = 163587 };

typedef struct {
    int step;
    float logits[FAKE_VOCABULARY];
} FakeEngine;

typedef struct {
    int32_t ids[8];
    size_t count;
    int cancel_after_first;
} Collector;

static K3Status fake_step(void *opaque, const int32_t *tokens, size_t count,
                          float **logits, size_t *vocabulary)
{
    static const int output[] = {7, 8, 163586};
    FakeEngine *engine = (FakeEngine *)opaque;
    CHECK(tokens != NULL && count > 0);
    CHECK(engine->step < (int)(sizeof output / sizeof output[0]));
    for (int i = 0; i < FAKE_VOCABULARY; i++) engine->logits[i] = -1000.0f;
    engine->logits[output[engine->step++]] = 1000.0f;
    *logits = engine->logits;
    *vocabulary = FAKE_VOCABULARY;
    return K3_OK;
}

static K3Status fake_decode(void *opaque, const int32_t *ids, size_t count,
                            char *text, size_t capacity, size_t *needed)
{
    (void)opaque;
    if (!ids || count != 1 || !needed) return K3_E_ARGUMENT;
    char piece[32];
    const int n = snprintf(piece, sizeof piece, "%d", ids[0]);
    if (n < 0) return K3_E_ENGINE;
    *needed = (size_t)n;
    if (!text && capacity == 0) return K3_OK;
    if (!text || capacity < (size_t)n) return K3_E_ARGUMENT;
    memcpy(text, piece, (size_t)n);
    return K3_OK;
}

static int collect(const K3Token *token, void *opaque)
{
    Collector *collector = (Collector *)opaque;
    CHECK(token->piece != NULL && token->piece_bytes > 0);
    CHECK(collector->count < sizeof collector->ids / sizeof collector->ids[0]);
    collector->ids[collector->count++] = token->token_id;
    return !(collector->cancel_after_first && collector->count == 1);
}

static K3GenerateLoop make_loop(FakeEngine *engine, volatile int *cancel,
                                int *busy, int32_t *sequence, size_t capacity)
{
    K3GenerateLoop loop;
    memset(&loop, 0, sizeof loop);
    loop.step = fake_step;
    loop.decode = fake_decode;
    loop.context = engine;
    loop.cancel = cancel;
    loop.busy = busy;
    loop.sequence = sequence;
    loop.sequence_capacity = capacity;
    loop.vocabulary = FAKE_VOCABULARY;
    loop.incremental = 1;
    return loop;
}

static void test_stops_and_usage(void)
{
    const int32_t prompt[] = {1, 2};
    const int32_t stops[] = {163586};
    int32_t sequence[16];
    volatile int cancel = 0;
    int busy = 0;
    FakeEngine engine; memset(&engine, 0, sizeof engine);
    Collector collector; memset(&collector, 0, sizeof collector);
    K3GenerateLoop loop = make_loop(&engine, &cancel, &busy, sequence, 16);
    K3GenerationOptions options;
    K3Usage usage;
    k3_generation_options_init(&options);
    options.max_tokens = 8;
    options.stop_token_ids = stops;
    options.stop_token_count = 1;

    CHECK(k3_generate_loop(&loop, prompt, 2, &options,
                           collect, &collector, &usage) == K3_OK);
    CHECK(collector.count == 2);
    CHECK(collector.ids[0] == 7 && collector.ids[1] == 8);
    CHECK(usage.prompt_tokens == 2);
    CHECK(usage.completion_tokens == 2);
    CHECK(busy == 0);
}

static void test_cancel_and_busy(void)
{
    const int32_t prompt[] = {1};
    int32_t sequence[16];
    volatile int cancel = 0;
    int busy = 0;
    FakeEngine engine; memset(&engine, 0, sizeof engine);
    Collector collector; memset(&collector, 0, sizeof collector);
    collector.cancel_after_first = 1;
    K3GenerateLoop loop = make_loop(&engine, &cancel, &busy, sequence, 16);
    K3GenerationOptions options;
    K3Usage usage;
    k3_generation_options_init(&options);

    CHECK(k3_generate_loop(&loop, prompt, 1, &options,
                           collect, &collector, &usage) == K3_E_CANCELLED);
    CHECK(collector.count == 1);
    CHECK(usage.completion_tokens == 1);
    CHECK(busy == 0);

    busy = 1;
    CHECK(k3_generate_loop(&loop, prompt, 1, &options,
                           collect, &collector, &usage) == K3_E_BUSY);
    CHECK(busy == 1);
}

static void test_invalid_prompt_fails_before_forward(void)
{
    const int32_t prompt[] = {FAKE_VOCABULARY};
    int32_t sequence[16];
    volatile int cancel = 0;
    int busy = 0;
    FakeEngine engine; memset(&engine, 0, sizeof engine);
    Collector collector; memset(&collector, 0, sizeof collector);
    K3GenerateLoop loop = make_loop(&engine, &cancel, &busy, sequence, 16);
    K3GenerationOptions options;
    K3Usage usage;
    k3_generation_options_init(&options);

    CHECK(k3_generate_loop(&loop, prompt, 1, &options,
                           collect, &collector, &usage) == K3_E_ARGUMENT);
    CHECK(engine.step == 0);
    CHECK(busy == 0);
}

int main(void)
{
    test_stops_and_usage();
    test_cancel_and_busy();
    test_invalid_prompt_fails_before_forward();
    puts("GENERATION LOOP TESTS PASSED");
    return 0;
}
