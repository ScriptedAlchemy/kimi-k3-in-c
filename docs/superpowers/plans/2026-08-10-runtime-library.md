# Reusable Exact-Weight Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the model lifecycle currently embedded in `k3_run.c` into a reusable shared-library API used by both the CLI and the OpenAI server.

**Architecture:** Move forward arithmetic and loaded model state into internal runtime modules without changing operation order. Expose an opaque `K3Runtime` that owns the safetensors index, exact BF16 trunk reader, native MXFP4 expert cache, tokenizer, incremental state, request buffers, sampler, diagnostics, and cancellation. Keep CLI parsing/output in `k3_run.c`; it calls the same runtime API the HTTP binding uses.

**Tech Stack:** Portable C99, pthreads, OpenMP, GNU Make, CMake, ctypes-compatible C ABI, existing synthetic fixtures and exact oracle.

## Global Constraints

- Official MXFP4 experts and BF16 trunk are the only emitted model path.
- Preserve floating-point operation order and deterministic greedy output.
- `k3_expert_drops != 0` makes the request invalid and returns an error.
- One `K3Runtime` accepts one active generation at a time.
- Untrusted text must never recognize XTML added tokens.
- The public header contains no internal structs and is valid from C and C++.
- Every allocation has a matched close path, including partial-open failures.

---

### Task 1: Freeze the public C ABI with a compile-only test

**Files:**
- Create: `include/k3/k3_runtime.h`
- Create: `tests/unit/test_runtime_api.c`
- Modify: `Makefile`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: no runtime implementation
- Produces: stable status, configuration, token, usage, callback, lifecycle, tokenizer, and generation declarations

- [ ] **Step 1: Write the header-consumer test first**

Create `tests/unit/test_runtime_api.c`:

```c
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
```

- [ ] **Step 2: Add the test target and verify the missing header fails**

Run:

```bash
make clean
make bin/test_runtime_api
```

Expected: compilation fails because `k3_runtime.h` does not exist.

- [ ] **Step 3: Define the exact ABI**

Create `include/k3/k3_runtime.h` with this public shape:

```c
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
```

Default initializers set incremental on, greedy on, `temperature=1.0`, `top_p=0.95`, and `max_tokens=8`. The first `tokenize` or `decode` call may use `capacity=0` to obtain `needed` without writing output.

- [ ] **Step 4: Make the compile-only target link against a temporary stub**

Create a test-local stub in `tests/unit/test_runtime_api.c` only until Task 3 provides the real symbols. The stub must return `K3_E_ENGINE` and not allocate. Run:

```bash
make bin/test_runtime_api
./bin/test_runtime_api
```

Expected: compile and exit 0.

- [ ] **Step 5: Commit the ABI slice**

Run:

```bash
git add include/k3/k3_runtime.h tests/unit/test_runtime_api.c Makefile CMakeLists.txt
git commit -m "feat(runtime): define the reusable C ABI"
```

### Task 2: Extract the forward path without changing arithmetic

**Files:**
- Create: `src/runtime/k3_forward.h`
- Create: `src/runtime/k3_forward.c`
- Modify: `src/cli/k3_run.c`
- Modify: `Makefile`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `K3Cfg`, `K3LayerBind`, `K3ModelBind`, `K3Cache`, `K3Trunk`
- Produces: internal `K3Weights` and `k3_forward(...)`

- [ ] **Step 1: Record the baseline oracle output hash**

Run:

```bash
make clean && make -j2 bin/k3_model
./bin/k3_model tests/fixtures > build/runtime-before.log
sha256sum build/runtime-before.log
```

Save the hash in the task log; it is the extraction comparison.

- [ ] **Step 2: Add the internal header**

Create `src/runtime/k3_forward.h`:

```c
#ifndef K3_FORWARD_H
#define K3_FORWARD_H

#include "k3.h"
#include "k3_bind.h"
#include "k3_cache.h"
#include "k3_trunk.h"

typedef struct {
    K3LayerBind *layers;
    K3ModelBind model;
    int n_bound;
    K3Trunk *trunk;
    float *kv_cache;
    float *rope_cache;
    int *mla_slot;
    int n_mla;
    int kv_capacity;
    int cached;
    int draft_mode;
} K3Weights;

int k3_forward(K3Weights *weights, const K3Cfg *cfg, K3Cache *cache,
               const int32_t *ids, int token_count, float *last_logits,
               float *scratch, float *hidden, float *attn_res,
               float *recurrent, int32_t *all_argmax);

#endif
```

- [ ] **Step 3: Move, do not rewrite, the forward implementation**

Move the current `Weights` definition, `argmax_`, and `forward` body from `src/cli/k3_run.c` into `src/runtime/k3_forward.c`. Rename only:

```text
Weights -> K3Weights
forward -> k3_forward
lay -> layers
mb -> model
kvc -> kv_cache
ropec -> rope_cache
kv_cap -> kv_capacity
```

Keep loop order, buffer layout, double accumulation, trunk prefetch order, cache attachment, and all calls to `k3_decoder_layer_inc`, `k3_attn_res`, `k3_rmsnorm`, and `k3_mmw` byte-for-byte except those field renames.

- [ ] **Step 4: Compile and verify the expected initial break**

Run:

```bash
make bin/k3
```

Expected: unresolved `forward`/`Weights` references in `k3_run.c` until the call sites are renamed.

- [ ] **Step 5: Point the CLI at the internal module**

Include `k3_forward.h`, change `Weights` to `K3Weights`, and change each `forward(...)` call to `k3_forward(...)`. Update only the field names listed in Step 3.

Run:

```bash
make clean && make -j2 bin/k3 bin/k3_model
./bin/k3_model tests/fixtures > build/runtime-after.log
diff -u build/runtime-before.log build/runtime-after.log
```

Expected: no diff.

- [ ] **Step 6: Commit the arithmetic-preserving extraction**

Run:

```bash
git diff --check
git add src/runtime/k3_forward.c src/runtime/k3_forward.h src/cli/k3_run.c Makefile CMakeLists.txt
git commit -m "refactor(runtime): extract the exact forward path"
```

### Task 3: Implement lifecycle, tokenizer, reset, and cleanup

**Files:**
- Create: `src/runtime/k3_runtime.c`
- Create: `src/runtime/k3_runtime_internal.h`
- Create: `tests/unit/test_runtime.c`
- Modify: `tests/unit/test_runtime_api.c`
- Modify: `Makefile`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: public ABI from Task 1 and `k3_forward` from Task 2
- Produces: real open/close/info/tokenize/decode/reset/error functions

- [ ] **Step 1: Write failure-path lifecycle tests**

Add tests that assert:

```c
K3RuntimeOptions opt;
K3Runtime *rt = NULL;
k3_runtime_options_init(&opt);
assert(k3_runtime_open("/definitely/missing", &opt, &rt) == K3_E_IO);
assert(rt == NULL);
assert(strcmp(k3_status_string(K3_E_BUSY), "runtime is busy") == 0);
k3_runtime_close(NULL);
```

Add a tokenizer fixture test using `tests/fixtures/chat/tokenizer`:

```c
opt.tokenizer_dir = "tests/fixtures/chat/tokenizer";
/* Open through the internal tokenizer-only constructor used by this unit test. */
assert(k3_runtime_tokenize(rt, "literal <|end_of_msg|>", 24, 0,
                           ids, 64, &n) == K3_OK);
for (size_t i = 0; i < n; i++) assert(ids[i] != 163586);
assert(k3_runtime_tokenize(rt, "<|end_of_msg|>", 16, 1,
                           ids, 64, &n) == K3_OK);
assert(n == 1 && ids[0] == 163586);
```

- [ ] **Step 2: Run tests and observe real-symbol failure**

Run:

```bash
make bin/test_runtime
```

Expected: link failure because `src/runtime/k3_runtime.c` does not exist.

- [ ] **Step 3: Define the private owner struct**

Create `src/runtime/k3_runtime_internal.h` with `struct K3Runtime` owning:

```c
struct K3Runtime {
    K3Cfg cfg;
    int full_attn[128];
    K3St shards;
    K3Trunk trunk;
    K3Cache cache;
    K3Weights weights;
    Tok tokenizer;
    int have_tokenizer;
    float *hidden;
    float *attn_res;
    float *recurrent;
    float *scratch;
    float *logits;
    int32_t *sequence;
    size_t sequence_capacity;
    uint32_t context_tokens;
    volatile int cancel;
    int busy;
    char error[512];
};
```

Add explicit boolean ownership flags for each successfully opened subsystem so `k3_runtime_close` can unwind a partial open in reverse order.

- [ ] **Step 4: Implement defaults and fail-closed open**

`k3_runtime_open` must perform this exact order:

```text
validate options and output pointer
load config from model_dir/config.json or explicit config_path
open safetensors index
verify every requested layer has a complete tensor set
open packed BF16 trunk when trunk_dir is set, otherwise bind resident layers
bind embedding/final norm/lm_head
initialize native MXFP4 expert cache
load tokenizer when tokenizer_dir is set
allocate recurrent, KV/rope, scratch, logits, and sequence buffers for context_tokens
return the opaque handle
```

Every failure sets `error`, calls `k3_runtime_close`, leaves `*runtime_out == NULL`, and maps to one `K3Status`.

- [ ] **Step 5: Implement trusted/untrusted tokenizer paths**

Use:

```c
const int allow_added = allow_special ? 1 : 0;
const int count = tok_encode_mode(&runtime->tokenizer, text, (int)text_bytes,
                                  ids, (int)capacity, allow_added);
```

Reject `text_bytes > INT_MAX`; query mode computes into a temporary buffer sized `text_bytes + 1`, reports `needed`, then frees it. `k3_runtime_decode` uses `tok_decode` and supports the same capacity query contract.

- [ ] **Step 6: Implement reset and info**

`k3_runtime_reset` zeroes recurrent, KV, and rope state, sets `weights.cached=0`, clears cancellation, and resets cache/trunk request counters without unloading weights. `k3_runtime_info` returns:

```text
model_family = "Kimi K3"
weight_format = "official MXFP4 experts + BF16 trunk"
```

- [ ] **Step 7: Replace the compile-test stubs and run lifecycle tests**

Remove all test-local ABI stubs, link `src/runtime/k3_runtime.c`, and run:

```bash
make clean
make -j2 bin/test_runtime_api bin/test_runtime
./bin/test_runtime_api
./bin/test_runtime tests/fixtures/chat/tokenizer
```

Expected: both pass and an ASan build reports no leak on missing-model, tokenizer-only, and partial-open paths.

- [ ] **Step 8: Commit lifecycle ownership**

Run:

```bash
git add include/k3/k3_runtime.h src/runtime tests/unit/test_runtime.c tests/unit/test_runtime_api.c Makefile CMakeLists.txt
git commit -m "feat(runtime): load and own one exact Kimi K3 model"
```

### Task 4: Add callback generation, sampling, stops, and cancellation

**Files:**
- Create: `src/runtime/k3_generate.h`
- Create: `src/runtime/k3_generate.c`
- Create: `tests/unit/test_generate.c`
- Modify: `src/runtime/k3_runtime.c`
- Modify: `Makefile`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: loaded `K3Runtime`, `k3_forward`, `k3_sampler_next`
- Produces: `k3_runtime_generate`, `k3_runtime_cancel`, per-token callback and usage

- [ ] **Step 1: Write a deterministic fake-logit generator test**

Define an internal step function signature in `k3_generate.h`:

```c
typedef K3Status (*K3GenerateStep)(void *context, const int32_t *tokens,
                                   size_t count, float **logits,
                                   size_t *vocabulary);
```

In `tests/unit/test_generate.c`, supply logits whose argmax sequence is `7, 8, 163586`. Assert that the loop:

```text
emits 7 and 8
stops before callback delivery of stop token 163586
reports prompt_tokens and completion_tokens exactly
returns K3_E_CANCELLED when callback returns zero
returns K3_E_BUSY on re-entry
```

- [ ] **Step 2: Run and observe the missing implementation**

Run:

```bash
make bin/test_generate
```

Expected: link failure for `k3_generate_loop`.

- [ ] **Step 3: Implement the minimal generic loop**

`k3_generate_loop` validates all prompt IDs and generation options, invokes the step function, selects via `k3_sampler_next`, checks stop IDs before callback delivery, decodes the selected token, fills `K3Token`, and stops when callback returns zero or cancellation is set.

Sampling rules:

```text
greedy=1: exact argmax
greedy=0: PCG32 sampler from PR #20 with request seed
temperature must be > 0
top_p must be > 0 and <= 1
max_tokens must be > 0 and fit context_tokens
```

- [ ] **Step 4: Connect the real incremental step**

On the first step after reset, call `k3_forward` with the complete prompt and set `weights.cached=prompt_tokens`. On each subsequent step, feed only the previous generated token with `token_count=1`, then increment `weights.cached`. The generated token is appended to `sequence` only after a successful forward.

- [ ] **Step 5: Enforce invalid-expert and cleanup semantics**

Snapshot `k3_expert_drops` before generation. If it increases, set:

```text
request invalid: one or more routed experts failed to load
```

return `K3_E_ENGINE`, and do not report success usage. Always clear `busy` in one cleanup block, including callback cancellation and forward failure.

- [ ] **Step 6: Run fake-loop, runtime, and exact gates**

Run:

```bash
make clean
make -j2 bin/test_generate bin/test_runtime bin/k3_model
./bin/test_generate
./bin/test_runtime tests/fixtures/chat/tokenizer
./bin/k3_model tests/fixtures
```

Expected: all pass and exact oracle output is unchanged.

- [ ] **Step 7: Commit generation**

Run:

```bash
git add src/runtime tests/unit/test_generate.c Makefile CMakeLists.txt
git commit -m "feat(runtime): stream generated tokens through callbacks"
```

### Task 5: Make the CLI a runtime client and prove behavioral parity

**Files:**
- Modify: `src/cli/k3_run.c`
- Create: `tests/cli/test_cli_runtime.py`
- Modify: `Makefile`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: complete public runtime
- Produces: existing `k3` CLI behavior through `K3Runtime`

- [ ] **Step 1: Add a CLI parity test before changing the CLI**

The test runs a tiny or real fixture twice, once with raw IDs and once with tokenized text where available, then asserts the output JSON keys remain:

```python
assert sorted(result) == [
    "full_ids", "generated_ids", "layers", "prompt_ids", "seconds_per_token"
]
assert result["full_ids"] == result["prompt_ids"] + result["generated_ids"]
```

It also asserts `--help`, `--version`, `--list-presets`, invalid mixed prompt sources, and invalid token IDs retain their exit codes.

- [ ] **Step 2: Run the parity test against the pre-refactor CLI**

Run:

```bash
python3 tests/cli/test_cli_runtime.py --binary ./bin/k3 --record build/cli-baseline.json
```

Expected: baseline record is created and tests pass.

- [ ] **Step 3: Replace CLI model ownership with runtime calls**

Keep argument parsing, presets, state-file diagnostics, JSON output, timing table, and chat REPL in `k3_run.c`. Replace direct safetensors/trunk/cache/tokenizer ownership for ordinary and chat generation with:

```c
K3RuntimeOptions ro;
k3_runtime_options_init(&ro);
ro.trunk_dir = trunk_dir;
ro.tokenizer_dir = tok_dir;
ro.config_path = cfg_path;
ro.trunk_budget_bytes = (uint64_t)(trunk_gb * 1e9);
ro.expert_cache_bytes = (uint64_t)(cache_gb * 1e9);
ro.context_tokens = (uint32_t)(prior + np + gen + 1);
ro.layers = want_layers;
ro.incremental = incremental;
```

Use a CLI callback to print the per-token timing row and append output IDs. Preserve diagnostic-only `--tf-check`, draft verification, and state-file paths through internal runtime helpers rather than a second model loader.

- [ ] **Step 4: Run parity and exactness**

Run:

```bash
make clean && make -j2 portable
python3 tests/cli/test_cli_runtime.py --binary ./bin/k3 --compare build/cli-baseline.json
make test
```

Expected: CLI contract matches the baseline and all tests pass.

- [ ] **Step 5: Commit the single-lifecycle CLI**

Run:

```bash
git add src/cli/k3_run.c tests/cli/test_cli_runtime.py Makefile CMakeLists.txt
git commit -m "refactor(cli): use the shared exact-weight runtime"
```

### Task 6: Build and install the shared library with both build systems

**Files:**
- Modify: `Makefile`
- Modify: `CMakeLists.txt`
- Modify: `.github/workflows/ci.yml`
- Modify: `docs/API.md`

**Interfaces:**
- Consumes: runtime sources
- Produces: `libk3.so`/`libk3.dylib`, installed header, sanitizer and ABI gates

- [ ] **Step 1: Add shared-library build targets**

GNU Make produces `libk3.so` on Linux and `libk3.dylib` on Darwin from PIC runtime/engine objects. CMake adds:

```cmake
add_library(k3_runtime SHARED ${K3_ENGINE_SOURCES} ${K3_RUNTIME_SOURCES})
set_target_properties(k3_runtime PROPERTIES OUTPUT_NAME k3)
target_link_libraries(k3_runtime PUBLIC k3_flags m Threads::Threads)
```

Install the library and `include/k3/k3_runtime.h` alongside the CLI.

- [ ] **Step 2: Add an exported-symbol assertion**

Run in CI:

```bash
nm -D libk3.so | awk '{print $3}' | sort > build/exported-symbols.txt
for sym in k3_runtime_open k3_runtime_close k3_runtime_tokenize k3_runtime_decode k3_runtime_reset k3_runtime_generate k3_runtime_cancel; do
  grep -qx "$sym" build/exported-symbols.txt
done
```

Use `nm -gU` on Darwin.

- [ ] **Step 3: Run full build and sanitizer matrix**

Run the Make gate on the development host:

```bash
make clean && make -j2 portable libk3 && make test
```

Run CMake and sanitizers on the target Linux/x86-64 LXC:

```bash
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release -DK3_NATIVE_ARCH=OFF
cmake --build build/cmake -j2
ctest --test-dir build/cmake --output-on-failure
cmake -S . -B build/sanitize -DCMAKE_BUILD_TYPE=Debug -DK3_NATIVE_ARCH=OFF -DK3_SANITIZE=ON
cmake --build build/sanitize -j2
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/sanitize --output-on-failure
```

Expected: every target and test passes without sanitizer findings.

- [ ] **Step 4: Document the ABI and thread boundary**

`docs/API.md` must show a complete open/tokenize/generate/close example, capacity-query tokenization, callback cancellation, exact weight format, and this statement:

```text
A K3Runtime is single-generation. Serialize callers outside the runtime; health and
model-discovery endpoints do not need the generation lock.
```

- [ ] **Step 5: Commit the distributable runtime**

Run:

```bash
git add Makefile CMakeLists.txt .github/workflows/ci.yml docs/API.md
git commit -m "build(runtime): ship libk3 with ABI and sanitizer gates"
```
