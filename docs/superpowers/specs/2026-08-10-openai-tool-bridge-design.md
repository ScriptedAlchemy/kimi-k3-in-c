# Exact-weight OpenAI chat and tool bridge

Date: 2026-08-10

## Outcome

The project will expose the existing exact-weight Kimi K3 C engine through an
OpenAI-compatible HTTP service that OpenCode and similar agent harnesses can use.
The service will support streamed chat, Kimi's thinking channel, typed tool calls,
and tool-result turns while continuing to read Moonshot's released safetensors
directly.

The deployment will live on the `main` branch of the ScriptedAlchemy fork. The
Proxmox LXC installer will clone that fork and branch by default, record the exact
installed commit, and retain a ref override for reproducible rollbacks.

## Non-negotiable constraints

- Keep the released MXFP4 expert weights and BF16 non-expert trunk. Do not create
  Q2/Q3/Q4 GGUF weights or quantize the BF16 trunk.
- Preserve the engine's deterministic greedy path and the existing full-model
  oracle. Performance changes must leave its output hashes and token oracle intact.
- Continue using the already-downloaded official checkpoint and packed trunk. A
  code upgrade must not redownload, convert, move, or delete model data.
- Bind to loopback by default. A non-loopback bind requires an API key.
- Run one generation at a time because the checkpoint index, trunk reader, cache,
  and runtime state are not thread-safe.

## Source integration and provenance

The fork's first integration branch will combine three independently reviewable
layers:

1. Upstream `FareedKhan-dev/kimi-k3-in-c` v1.0.0 as the validated base.
2. PR #25's three bit-preserving kernel/CI commits. These are already build-tested
   and oracle-tested on the target LXC.
3. PR #20's XTML chat renderer, parser, tokenizer fixtures, and sampler as the
   starting text-chat implementation.

The OpenAI protocol layer will adapt the architecture and relevant XTML test cases
from SQLiteAI WARP's Apache-2.0 `serve/` implementation. Adapted files will retain
their license headers and attribution. Moonshot's released `encoding_k3.py` remains
the differential oracle for prompt bytes and token IDs; it is a test/reference
dependency, not part of the inference runtime.

## Architecture

### 1. Reusable C runtime

The model lifecycle currently embedded in `src/cli/k3_run.c` will move behind a
small opaque runtime API. The CLI will become a client of that API, ensuring the
HTTP path and the existing command use the same loader, cache, trunk, tokenizer,
incremental decode, and error checks.

The public boundary will cover:

- open and close one model instance;
- tokenize trusted XTML markup separately from untrusted content;
- decode individual token pieces;
- reset per-request sequence state;
- generate from input token IDs with a per-token callback and cancellation flag;
- return prompt/completion token counts and engine diagnostics;
- fail a generation if `k3_expert_drops` is non-zero.

Model weights remain loaded between requests. Sequence/KV state resets before each
MVP request because the client sends the complete conversation. Prefix reuse is a
later optimization and is not allowed to complicate correctness of the first
endpoint.

### 2. XTML prompt and response codec

The protocol process will render and parse Kimi K3's token-level XTML. It will never
concatenate user text with structural markup before tokenization.

Supported request content:

- system, user, assistant, and tool messages;
- assistant `reasoning_content` from prior turns;
- OpenAI function tools expressed as JSON Schema;
- `tool_choice` values `auto`, `required`, `none`, or one named function;
- tool results matched and ordered by `tool_call_id`;
- `reasoning_effort` values accepted by Kimi K3.

The streaming parser will classify control boundaries by token ID, not by scanning
decoded text. Literal strings such as `<|end_of_msg|>` in user content or model text
therefore cannot forge a message boundary. Parsed output is divided into
`reasoning_content`, ordinary `content`, and typed tool calls with JSON arguments.

### 3. OpenAI-compatible HTTP service

The initial service will use Python's standard library plus a narrow `ctypes`
binding to the C runtime. Python owns JSON validation, HTTP, SSE framing, and XTML;
C owns all tokenizer and inference arithmetic.

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Both blocking and SSE responses are supported. Chat responses include
`reasoning_content`, `content`, `tool_calls`, `finish_reason`, and token usage.
Streaming responses send those fields as deltas and terminate with `data: [DONE]`.

The server accepts harmless OpenAI defaults that do not change generation. Unknown
or unsupported behavior-changing fields return an OpenAI-shaped HTTP 400 instead of
being ignored. A second generation while one is active returns HTTP 429; health and
model discovery remain responsive. Client disconnect propagates cancellation to the
C loop between generated tokens.

### 4. OpenCode integration

The repository will provide a checked example using OpenCode's native
OpenAI-compatible provider package. It will point `baseURL` at
`http://<kimi-host>:8000/v1`, advertise tool-call and reasoning support, and select
the model ID `kimi-k3-local`.

The acceptance exercise will make OpenCode request a deterministic local tool,
receive a structured tool call, submit the tool result, and receive the final Kimi
response. The transcript must show that no tool markup leaked into response text.

## Security and failures

- Default bind: `127.0.0.1:8000`.
- Non-loopback bind: refuse startup without a bearer API key.
- Cap request bodies and message/tool-schema sizes before allocation.
- Keep user and tool content on the untrusted tokenizer path.
- Refuse unsupported image/audio parts in the MVP.
- Return structured 400, 409/429, 500, or 503 errors rather than partial success.
- If tokenization, model reset, inference, output parsing, or expert loading fails,
  terminate the response and mark it as an engine error.
- Never include model paths, prompts, tool results, or API keys in normal logs.

## Test-first delivery

Implementation follows red-green-refactor in these vertical slices:

1. C runtime lifecycle and token callback against the existing synthetic checkpoint.
2. XTML tool declaration and tool-result rendering, differential against Moonshot's
   encoder and frozen golden token IDs.
3. Incremental output parsing for reasoning, content, and typed tool calls across
   every possible SSE chunk boundary.
4. Blocking HTTP chat against a scripted runtime.
5. Streaming HTTP chat, cancellation, busy handling, request limits, and auth.
6. Whole-stack synthetic-model integration using the real C shared library.
7. Real-checkpoint one-token exact-output smoke test and OpenCode tool round trip.

Every production behavior begins with a test that is observed failing for the
missing behavior. Existing engine tests, sanitizers, ShellCheck, and the full oracle
remain release gates.

## Fork and deployment workflow

The fork will be `ScriptedAlchemy/kimi-k3-in-c`, with `upstream` pointing to
`FareedKhan-dev/kimi-k3-in-c`. Integration is developed on topic branches and merged
to fork `main` only when the complete test matrix passes.

The LXC scripts will default to:

```text
K3_REPO_URL=https://github.com/ScriptedAlchemy/kimi-k3-in-c.git
K3_REPO_REF=main
```

Provisioning records `git rev-parse HEAD` in `/etc/kimi-k3-build`. Reruns refuse to
overwrite a dirty checkout. A release tag will identify each deployable main-branch
state even though new installations intentionally follow `main` by default.

## Explicitly out of scope for the MVP

- Anthropic Messages and OpenAI Responses APIs.
- Ollama protocol and model packaging.
- Vision, audio, embeddings, logprobs, batching, and parallel generations.
- Server-side tool execution; OpenCode remains the tool executor.
- Persistent cross-request KV/prefix reuse.
- Any post-training weight quantization, pruning, expert dropping, or approximate
  routing.

## Acceptance criteria

- The fork's `main` builds on the target Debian LXC and passes every weightless test,
  sanitizer, and exact oracle gate.
- The official checkpoint revision and all 96 shards verify successfully; packed
  trunk preparation completes without changing the source weights.
- Blocking and streaming chat return correctly separated reasoning and content.
- A function tool round trip works through OpenCode with valid JSON arguments and a
  matching tool result.
- Model output contains no leaked XTML control elements.
- No post-training quantized model artifact exists in the deployment.
- The reusable Proxmox installer pulls the fork's `main`, records the installed SHA,
  uses the compressed Btrfs model volume, and keeps that volume excluded from backup.
