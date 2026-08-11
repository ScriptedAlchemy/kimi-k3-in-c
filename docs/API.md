# C API

For embedding the engine rather than using the `k3` binary. The managed runtime is
declared by `include/k3/k3_runtime.h`; the lower-level kernels and configuration
reader are declared by `include/k3/k3.h` and `include/k3/k3_cfg.h`.

```c
#include <k3/k3_runtime.h>
#include <k3/k3.h>
#include <k3/k3_cfg.h>
```

The managed runtime owns one loaded model, its exact official-weight readers, tokenizer,
expert cache, generation buffers, and recurrent state. The low-level API remains
deliberately small: configuration and weight-binding structs plus the kernels, with no
hidden context object.

## Managed runtime

`make libk3` produces `libk3.so` on Linux and `libk3.dylib` on macOS. The same library
can be built as the CMake `k3_runtime` target. It is suitable for C callers and FFIs
such as Python `ctypes`.

```c
#include <k3/k3_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int print_token(const K3Token *token, void *user)
{
    (void)user;
    fwrite(token->piece, 1, token->piece_bytes, stdout);
    fflush(stdout);
    return 1; /* return 0 to cancel after this callback */
}

int main(void)
{
    K3RuntimeOptions open_options;
    k3_runtime_options_init(&open_options);
    open_options.trunk_dir = "/srv/kimi/trunk";
    open_options.tokenizer_dir = "/srv/kimi/model";
    open_options.context_tokens = 8192;

    K3Runtime *runtime = NULL;
    K3Status status = k3_runtime_open("/srv/kimi/model", &open_options, &runtime);
    if (status != K3_OK) return 1;

    const char *prompt = "Hello";
    size_t prompt_count = 0;
    status = k3_runtime_tokenize(runtime, prompt, strlen(prompt), 0,
                                 NULL, 0, &prompt_count);
    int32_t *prompt_ids = malloc(prompt_count * sizeof *prompt_ids);
    if (status != K3_OK || !prompt_ids) goto fail;
    status = k3_runtime_tokenize(runtime, prompt, strlen(prompt), 0,
                                 prompt_ids, prompt_count, &prompt_count);
    if (status != K3_OK) goto fail;

    K3GenerationOptions generation;
    K3Usage usage;
    k3_generation_options_init(&generation);
    generation.max_tokens = 64;
    generation.greedy = 1;
    status = k3_runtime_generate(runtime, prompt_ids, prompt_count, &generation,
                                 print_token, NULL, &usage);
    free(prompt_ids);
    k3_runtime_close(runtime);
    return status == K3_OK ? 0 : 1;

fail:
    fprintf(stderr, "%s\n", k3_runtime_last_error(runtime));
    free(prompt_ids);
    k3_runtime_close(runtime);
    return 1;
}
```

`k3_runtime_tokenize` and `k3_runtime_decode` support capacity queries: pass a NULL
output with capacity zero, read `needed`, allocate, and call again. Passing a non-NULL
buffer that is too small returns `K3_E_ARGUMENT` without a partial result.

The callback's `piece` is valid only for the duration of that callback. Returning zero
requests cancellation; another thread may also call `k3_runtime_cancel`. In either case
generation returns `K3_E_CANCELLED`, while usage still describes work completed before
the cancellation boundary. Stop-token IDs are not delivered to the callback.

`k3_runtime_info` identifies the loaded representation as `official MXFP4 experts +
BF16 trunk`. The engine consumes the checkpoint's native QAT MXFP4 expert bytes and the
official BF16 always-active weights. It does not apply post-training quantization.

A `K3Runtime` permits one generation at a time. Serialize generation callers outside
the runtime; a concurrent call returns `K3_E_BUSY`. Health and model-discovery endpoints
that do not start generation do not need that generation lock.

## Configuration

```c
K3Cfg cfg;
int   full_attn[128];

if (!k3_cfg_load_file(&cfg, full_attn, 128, "model/config.json")) {
    /* Do not proceed. A partially-populated K3Cfg describes a different model. */
    return 1;
}
```

`k3_cfg_load_file` reads the checkpoint's own configuration and populates every field.

**It never substitutes a default for a missing field.** If a key is absent it collects
every such key, reports them together, and returns 0. This matters more than it appears:
a configuration reader that defaults silently produces an engine that loads, streams,
decodes, and emits fluent text from the wrong architecture, with nothing to indicate it.

Both layouts are accepted, the released nested form (`text_config.*`) and the flat form
used by the test fixtures.

Layer roles:

```c
k3_is_mla(&cfg, layer)     /* Gated MLA layer          */
k3_is_kda(&cfg, layer)     /* Kimi Delta Attention     */
k3_is_dense(&cfg, layer)   /* the single dense layer   */
```

`full_attn` holds ONE-BASED layer indices, matching the checkpoint's own convention.

## Scratch memory

Every kernel takes caller-provided scratch. Sizes come from the engine, not from your
own arithmetic, recomputing them by hand is the easiest way to overrun a buffer
silently.

```c
size_t n = k3_layer_scratch(&cfg, n_tokens);
float *scratch = malloc(n * sizeof(float));
```

| function | covers |
|---|---|
| `k3_layer_scratch(cfg, T)` | a whole decoder layer |
| `k3_kda_scratch(cfg, T)` | a KDA layer |
| `k3_mla_scratch(cfg, T)` | an MLA layer, no KV cache |
| `k3_mla_scratch_cached(cfg, T, cap, mode)` | an MLA layer with a KV cache |
| `k3_moe_scratch(cfg)` | the MoE block |

## Weight structures

`K3KdaW`, `K3MlaW`, `K3MoeW` and `K3LayerW` describe where a layer's tensors live.

> **Zero every weight struct before filling it.** These hold function pointers
> (`K3MoeW::src`) and pointers whose NULL-ness selects a code path, dense versus MoE,
> gated versus ungated MLA. An uninitialised stack struct does not merely read the wrong
> weights; it can jump to an arbitrary address.

```c
K3MoeW moe;
memset(&moe, 0, sizeof moe);   /* required, not defensive */
```

Weight matrices are tagged pointers: `K3_WF32` (0) or `K3_WBF16` (1). Because `K3_WF32`
is zero, a `memset` struct defaults to fp32. Dispatch through `k3_mmw()` rather than
calling the typed matmuls directly.

## Running a layer

```c
k3_decoder_layer(hidden, block_residual, &n_blocks, &weights, &cfg,
                 layer_idx, n_tokens, kda_state, scratch);
```

`k3_decoder_layer_inc()` is the incremental form: MLA attends over a KV cache of earlier
positions and appends its own. KDA needs nothing carried, it updates its recurrent state
in place, and the Attention-Residual block stack is per token.

Both must produce identical tokens. The test suite asserts this rather than assuming it.

## Streaming experts

Provide a `K3ExpertSrc` and the MoE block will fetch experts on demand instead of reading
a resident bank:

```c
typedef struct K3ExpertSrc {
    int (*get)(struct K3ExpertSrc *, int layer, int expert, K3ExpertQ *out);
    int (*getmany)(struct K3ExpertSrc *, int layer, const int *experts, int n);
    void *ctx;
} K3ExpertSrc;
```

- `get` must keep the returned pointers valid until the caller finishes the token.
- `getmany` is an optional batch hint. It may be NULL, and callers must cope, falling
  back to `get` alone is always correct, only slower. It exists because issuing the whole
  top-k at once lets the reads overlap; serial `get` calls give the device a queue depth
  of one, which most NVMe hardware needs depth to saturate.

Experts stay in packed MXFP4 throughout. `k3_matmul_mxfp4` consumes nibbles directly and
never materialises a dequantised matrix, one expert is 17.5 MB packed against 132 MB
expanded, and a token touches 1,472 of them.

## Error handling

Two failure modes need explicit attention from callers.

**`k3_expert_drops`** is a global counter incremented whenever a streamed expert could
not be loaded. Non-zero means some token was computed with part of its routed
contribution missing, silent numerical corruption. The run completes and prints a
plausible token.

```c
if (k3_expert_drops) {
    fprintf(stderr, "%ld experts failed to load; output is corrupt\n", k3_expert_drops);
    return 1;   /* fail the run; do not report success */
}
```

**Configuration load failure** must abort, as above. There is no safe partial state.

## Thread safety

The kernels are reentrant and parallelise internally with OpenMP. They hold no global
state except `k3_expert_drops`.

The cache, the trunk reader, and the safetensors index are **not** thread-safe. One
inference at a time per managed runtime, as enforced by `k3_runtime_generate`.

## Minimal example

```c
K3Cfg cfg; int fa[128];
if (!k3_cfg_load_file(&cfg, fa, 128, "model/config.json")) return 1;

float *scratch = malloc(k3_layer_scratch(&cfg, T) * sizeof(float));

for (int L = 0; L < cfg.n_layers; L++)
    k3_decoder_layer(h, block_res, &nblocks, &layer_w[L], &cfg, L, T, kstate, scratch);

if (k3_expert_drops) return 1;   /* check before trusting the output */
```

See `src/cli/k3_run.c` for the complete path, including trunk streaming and the KV cache.
