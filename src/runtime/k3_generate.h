#ifndef K3_GENERATE_H
#define K3_GENERATE_H

#include <stddef.h>
#include <stdint.h>

#include "k3_runtime.h"

typedef K3Status (*K3GenerateStep)(void *context, const int32_t *tokens,
                                   size_t count, float **logits,
                                   size_t *vocabulary);

typedef K3Status (*K3GenerateDecode)(void *context, const int32_t *ids,
                                     size_t count, char *text,
                                     size_t capacity, size_t *needed);

typedef K3Status (*K3GeneratePrepare)(void *context);
typedef K3Status (*K3GenerateFinish)(void *context, K3Status status,
                                     K3Usage *usage);
typedef uint64_t (*K3GenerateBytesRead)(void *context);

typedef struct {
    K3GenerateStep step;
    K3GenerateDecode decode;
    K3GeneratePrepare prepare;
    K3GenerateFinish finish;
    K3GenerateBytesRead bytes_read;
    void *context;
    volatile int *cancel;
    int *busy;
    int32_t *sequence;
    size_t sequence_capacity;
    size_t vocabulary;
    int incremental;
} K3GenerateLoop;

K3Status k3_generate_loop(K3GenerateLoop *loop,
                          const int32_t *prompt, size_t prompt_tokens,
                          const K3GenerationOptions *options,
                          K3TokenCallback callback, void *user,
                          K3Usage *usage);

#endif
