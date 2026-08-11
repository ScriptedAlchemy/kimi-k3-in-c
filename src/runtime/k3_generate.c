#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3_generate.h"
#include "k3_sampler.h"

static double k3_generate_now(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static int k3_generate_is_stop(const K3GenerationOptions *options,
                               int32_t token)
{
    for (size_t i = 0; i < options->stop_token_count; i++)
        if (options->stop_token_ids[i] == token) return 1;
    return 0;
}

K3Status k3_generate_loop(K3GenerateLoop *loop,
                          const int32_t *prompt, size_t prompt_tokens,
                          const K3GenerationOptions *options,
                          K3TokenCallback callback, void *user,
                          K3Usage *usage)
{
    K3GenerationOptions defaults;
    K3Sampler sampler;
    K3Status status = K3_OK;
    int sampler_ready = 0;
    size_t sequence_count = prompt_tokens;
    const double started = k3_generate_now();

    if (usage) memset(usage, 0, sizeof *usage);
    if (!loop || !loop->step || !loop->decode || !loop->cancel || !loop->busy ||
        !loop->sequence || !prompt || prompt_tokens == 0 || !callback || !usage)
        return K3_E_ARGUMENT;
    k3_generation_options_init(&defaults);
    if (!options) options = &defaults;
    if (options->max_tokens == 0 || !(options->temperature > 0.0) ||
        !(options->top_p > 0.0) || options->top_p > 1.0 ||
        (options->stop_token_count != 0 && !options->stop_token_ids) ||
        loop->vocabulary == 0 || loop->vocabulary > (size_t)INT_MAX ||
        prompt_tokens > loop->sequence_capacity ||
        options->max_tokens > loop->sequence_capacity - prompt_tokens)
        return K3_E_ARGUMENT;
    for (size_t i = 0; i < prompt_tokens; i++)
        if (prompt[i] < 0 || (size_t)prompt[i] >= loop->vocabulary)
            return K3_E_ARGUMENT;
    for (size_t i = 0; i < options->stop_token_count; i++)
        if (options->stop_token_ids[i] < 0 ||
            (size_t)options->stop_token_ids[i] >= loop->vocabulary)
            return K3_E_ARGUMENT;

    if (__sync_lock_test_and_set(loop->busy, 1)) return K3_E_BUSY;
    *loop->cancel = 0;
    usage->prompt_tokens = prompt_tokens;
    memcpy(loop->sequence, prompt, prompt_tokens * sizeof(*prompt));

    if (loop->prepare) {
        status = loop->prepare(loop->context);
        if (status != K3_OK) goto cleanup;
    }
    k3_sampler_init(&sampler, options->temperature, options->top_p,
                    options->seed, 0);
    sampler_ready = 1;

    for (uint32_t index = 0; index < options->max_tokens; index++) {
        if (*loop->cancel) { status = K3_E_CANCELLED; break; }

        const int32_t *input = loop->sequence;
        size_t input_count = sequence_count;
        if (loop->incremental && index != 0) {
            input = &loop->sequence[sequence_count - 1];
            input_count = 1;
        }

        float *logits = NULL;
        size_t vocabulary = 0;
        const double step_started = k3_generate_now();
        status = loop->step(loop->context, input, input_count,
                            &logits, &vocabulary);
        const double step_seconds = k3_generate_now() - step_started;
        if (status != K3_OK) break;
        if (!logits || vocabulary != loop->vocabulary) {
            status = K3_E_ENGINE;
            break;
        }

        int selected = 0;
        if (k3_sampler_next(&sampler, logits, (int)vocabulary,
                            options->greedy != 0, &selected) != 0) {
            status = K3_E_ENGINE;
            break;
        }
        if (k3_generate_is_stop(options, (int32_t)selected)) break;

        loop->sequence[sequence_count++] = (int32_t)selected;
        size_t piece_bytes = 0;
        const int32_t selected_id = (int32_t)selected;
        status = loop->decode(loop->context, &selected_id, 1,
                              NULL, 0, &piece_bytes);
        if (status != K3_OK) break;
        if (piece_bytes == SIZE_MAX) { status = K3_E_MEMORY; break; }
        char *piece = (char *)malloc(piece_bytes + 1);
        if (!piece) { status = K3_E_MEMORY; break; }
        status = loop->decode(loop->context, &selected_id, 1,
                              piece, piece_bytes, &piece_bytes);
        if (status != K3_OK) { free(piece); break; }
        piece[piece_bytes] = 0;

        K3Token token;
        memset(&token, 0, sizeof token);
        token.index = index;
        token.token_id = selected_id;
        token.piece = piece;
        token.piece_bytes = piece_bytes;
        token.seconds = step_seconds;
        if (loop->bytes_read)
            token.expert_bytes_read = loop->bytes_read(loop->context);
        usage->completion_tokens++;
        const int keep_going = callback(&token, user);
        free(piece);
        if (!keep_going) {
            *loop->cancel = 1;
            status = K3_E_CANCELLED;
            break;
        }
    }

cleanup:
    if (sampler_ready) k3_sampler_free(&sampler);
    usage->seconds_total = k3_generate_now() - started;
    if (loop && loop->finish)
        status = loop->finish(loop->context, status, usage);
    __sync_lock_release(loop->busy);
    return status;
}
