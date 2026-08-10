# Kimi K3 Source Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the ScriptedAlchemy fork and combine the already-verified bit-identical performance commits with upstream Kimi K3 XTML text chat without losing either change set.

**Architecture:** Start from FareedKhan-dev v1.0.0 plus PR #25's three verified commits, then replay PR #20 in its original commit order. Resolve overlapping kernel/build files by retaining PR #25 arithmetic and adding only PR #20 chat targets and tokenizer safety changes. Publish topic branches first; fork `main` remains unchanged until the runtime, server, and deployment plans are green.

**Tech Stack:** Git, GitHub CLI, GNU Make, CMake, C99, Python 3, ASan/UBSan.

## Global Constraints

- Preserve official MXFP4 expert bytes and BF16 trunk behavior exactly.
- Do not add any post-training quantization artifact or default.
- Preserve `e4f6108875f2bdf63b14109287df738032d76cbc`, `73d276525ecb1cf06126ddecff30c09f7adbc517`, and `561724dbd222b54f667886ee9f3642707f7b1e2e` semantics.
- Preserve the exact full-model oracle and all weightless tests as release gates.
- Keep `upstream` pointed at `https://github.com/FareedKhan-dev/kimi-k3-in-c.git`.
- Do not push fork `main` until every plan in this program has passed.

---

### Task 1: Create the fork and normalize remotes

**Files:**
- Modify: Git remote configuration only

**Interfaces:**
- Consumes: authenticated GitHub CLI account `ScriptedAlchemy`
- Produces: GitHub repository `ScriptedAlchemy/kimi-k3-in-c`, local remotes `fork` and `upstream`

- [ ] **Step 1: Prove the current branch and worktree are safe**

Run:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
gh auth status
```

Expected: clean `design/openai-tool-bridge` at the committed design SHA, and GitHub reports `ScriptedAlchemy` with `repo` scope.

- [ ] **Step 2: Check for an existing fork before writing external state**

Run:

```bash
if gh repo view ScriptedAlchemy/kimi-k3-in-c --json nameWithOwner >/dev/null 2>&1; then
  gh repo view ScriptedAlchemy/kimi-k3-in-c --json nameWithOwner,parent,url
else
  gh repo fork FareedKhan-dev/kimi-k3-in-c --clone=false --remote=false
fi
```

Expected: exactly one fork whose parent is `FareedKhan-dev/kimi-k3-in-c`.

- [ ] **Step 3: Set unambiguous remotes**

Run:

```bash
git remote rename origin upstream
git remote add fork https://github.com/ScriptedAlchemy/kimi-k3-in-c.git
git remote -v
```

If `upstream` or `fork` already exists, use `git remote set-url` instead of adding a duplicate. Expected URLs:

```text
fork     https://github.com/ScriptedAlchemy/kimi-k3-in-c.git
upstream https://github.com/FareedKhan-dev/kimi-k3-in-c.git
```

- [ ] **Step 4: Fetch authoritative refs without changing files**

Run:

```bash
git fetch upstream main pull/20/head:refs/remotes/upstream/pr-20-chat pull/25/head:refs/remotes/upstream/pr-25-perf
git fetch fork main
git log --oneline --decorate -4 upstream/main upstream/pr-20-chat upstream/pr-25-perf
```

Expected: the PR heads contain `1912acdfe084cfb3639714b0bb7689963bbe71c8` and `561724dbd222b54f667886ee9f3642707f7b1e2e` unless upstream has replaced them, in which case stop and audit the new diff before proceeding.

### Task 2: Integrate the PR #20 chat core while preserving PR #25 arithmetic

**Files:**
- Create: `src/chat/k3_chat.c`
- Create: `src/chat/k3_chat.h`
- Create: `src/chat/k3_sampler.c`
- Create: `src/chat/k3_sampler.h`
- Create: `tests/unit/test_chat.c`
- Create: `tests/fixtures/chat/**`
- Modify: `src/tokenizer/k3_tok.h`
- Modify: `third_party/tok.h`
- Modify: `src/core/k3_ops.c`
- Modify: `Makefile`
- Modify: `CMakeLists.txt`
- Modify: `NOTICE`

**Interfaces:**
- Consumes: PR #20 commit `66541008257c62e4c28244ca64cfbf08c0d84127`
- Produces: `k3_chat_*` XTML API, `k3_sampler_*` API, safe `tok_encode_mode(..., allow_added)`, chat test target

- [ ] **Step 1: Create the integration branch**

Run:

```bash
git switch -c integration/exact-chat-tools
```

Expected: the branch starts from the committed design-and-plans head and contains the
three performance commits, the approved design, and these execution plans.

- [ ] **Step 2: Replay the chat-core commit and observe conflicts**

Run:

```bash
git cherry-pick 66541008257c62e4c28244ca64cfbf08c0d84127
```

Expected: conflicts are limited to files independently changed after v1.0.0. Do not resolve `src/core/k3_ops.c` by taking the chat side wholesale.

- [ ] **Step 3: Resolve the kernel conflict in favor of bit-identical performance**

For `src/core/k3_ops.c`, keep the branch version in full:

```bash
git checkout --ours src/core/k3_ops.c
git add src/core/k3_ops.c
```

Verify both optimized invariants still exist:

```bash
rg -n "NIBBLE DECODE IN REGISTER|WIDEN x ONCE|K3_KDA_STEP_DV" src/core/k3_ops.c
```

Expected: all three descriptions are present and no per-head KDA `calloc` was restored.

- [ ] **Step 4: Merge build-system intent rather than choosing a side**

Resolve `Makefile` and `CMakeLists.txt` so they include:

```make
INCLUDES += -Isrc/chat
ENGINE_SRC += src/chat/k3_chat.c src/chat/k3_sampler.c
UNIT_TESTS += test_chat
```

and equivalent CMake sources plus:

```cmake
k3_add_test(test_chat tests/unit/test_chat.c)
add_test(NAME chat COMMAND test_chat ${FIX}/chat/tokenizer)
```

Retain the PR #25 sanitizer targets and CI coverage. Stage the resolved build files:

```bash
git add Makefile CMakeLists.txt .github/workflows/ci.yml
```

- [ ] **Step 5: Retain PR #20 tokenizer safety and attribution**

Verify the resulting tokenizer API contains exactly the trusted/untrusted split:

```c
static int tok_encode_mode(Tok *T, const char *text, int len,
                           int *out, int max, int allow_added);
static int tok_encode(Tok *T, const char *text, int len, int *out, int max) {
    return tok_encode_mode(T, text, len, out, max, 1);
}
```

Run:

```bash
rg -n "tok_encode_mode|allow_added|Apache|Kimi K3" third_party/tok.h src/tokenizer/k3_tok.h NOTICE tests/fixtures/chat
git add third_party/tok.h src/tokenizer/k3_tok.h NOTICE src/chat tests/unit/test_chat.c tests/fixtures/chat
git cherry-pick --continue
```

Expected: commit completes and preserves original authorship.

- [ ] **Step 6: Run the focused chat and kernel gates**

Run:

```bash
make clean
make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)" bin/test_chat bin/test_ops bin/k3_model
./bin/test_chat tests/fixtures/chat/tokenizer
./bin/test_ops tests/fixtures/ops
./bin/k3_model tests/fixtures
```

Expected: chat passes, every op kernel passes, and the exact oracle reports 32/32 teacher forcing plus 20/20 generated, greedy, and incremental spans.

### Task 3: Integrate the upstream chat REPL and documentation

**Files:**
- Modify: `src/cli/k3_run.c`
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/QUICKSTART.md`
- Modify: `docs/README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/TESTING.md`

**Interfaces:**
- Consumes: chat API from Task 2
- Produces: `k3 --chat` text-only REPL with reasoning history and deterministic sampler

- [ ] **Step 1: Replay the REPL commit**

Run:

```bash
git cherry-pick 3766a95f6f27a3d41a089cf06299ab1ed7b81516
```

If `src/cli/k3_run.c` conflicts, preserve the current performance branch's forward loop and insert PR #20's chat helpers, flags, validation, prompt construction, and `chat_run` call around it. Do not restore an older `forward()` body.

- [ ] **Step 2: Verify CLI behavior and exactness coexist**

Run:

```bash
make -j2 bin/k3
./bin/k3 --help | rg -- '--chat|--temperature|--top-p|--greedy'
make test
```

Expected: the help exposes text chat and every weightless test remains green.

- [ ] **Step 3: Replay documentation without changing scope claims**

Run:

```bash
git cherry-pick 1912acdfe084cfb3639714b0bb7689963bbe71c8
```

Add one explicit note to `README.md` after the upstream chat section:

```markdown
The built-in REPL is text chat. The optional `k3serve` package adds an
OpenAI-compatible HTTP surface, streamed reasoning, and typed function tools while
using the same exact-weight runtime.
```

Do not claim tool support is complete until the OpenAI-service plan is green.

- [ ] **Step 4: Commit the documentation scope clarification**

Run:

```bash
git add README.md
git commit -m "docs: distinguish REPL chat from OpenAI tools"
```

### Task 4: Prove the integrated source and publish the topic branch

**Files:**
- Modify: no production files unless a gate exposes a defect

**Interfaces:**
- Consumes: Tasks 1-3
- Produces: immutable fork topic branch receipt for subsequent plans

- [ ] **Step 1: Run both build systems**

Run:

```bash
make clean
make -j2 portable
make test
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release -DK3_NATIVE_ARCH=OFF
cmake --build build/cmake -j2
ctest --test-dir build/cmake --output-on-failure
```

Expected: both systems build the same chat-capable CLI and all registered tests pass.

- [ ] **Step 2: Run sanitizers serially**

Run on the target Linux/x86-64 LXC or an equivalent AVX2 Linux runner:

```bash
cmake -S . -B build/sanitize -DCMAKE_BUILD_TYPE=Debug -DK3_NATIVE_ARCH=OFF -DK3_SANITIZE=ON
cmake --build build/sanitize -j2
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/sanitize --output-on-failure
```

Expected: zero ASan or UBSan reports. This target-specific gate is not run through
CMake on arm64 because the project's CMake baseline intentionally emits AVX2/FMA.

- [ ] **Step 3: Re-run the exact oracle after sanitizer builds**

Run:

```bash
make clean && make -j2 portable
./bin/k3_model tests/fixtures | tee build/source-integration-oracle.log
rg "32/32|20/20|ENGINE MATCHES THE REFERENCE EXACTLY" build/source-integration-oracle.log
```

Expected: every exact token gate is present.

- [ ] **Step 4: Publish only the topic branch**

Run:

```bash
git diff --check
git status --short
git push -u fork integration/exact-chat-tools
git rev-parse HEAD
```

Expected: clean worktree and a fork branch SHA that is not yet fork `main`.
