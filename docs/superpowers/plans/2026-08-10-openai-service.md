# OpenAI Chat and Tool Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dependency-free OpenAI-compatible Kimi K3 service that streams reasoning, content, and typed function calls through the exact-weight `libk3` runtime.

**Architecture:** Adapt the Kimi XTML renderer, incremental region parser, request normalization, and socket-tested HTTP/SSE shape from SQLiteAI WARP commit `2056d44f075c5e73c69c90d986e0dc89e9c9c6dd`, retaining Apache-2.0 headers and attribution. Replace WARP's model binding with the local `libk3` ABI and delete unsupported vision, raw-completion, grammar, and container-format behavior. A threaded stdlib HTTP server keeps health responsive while a non-blocking generation lock returns 429 to concurrent callers.

**Tech Stack:** Python 3 standard library, `ctypes`, `http.server`, `socketserver`, `unittest`, OpenAI Chat Completions JSON/SSE, Kimi K3 XTML, C99 `libk3`.

## Global Constraints

- Runtime dependency set is Python standard library plus `libk3`; no Flask, FastAPI, uvicorn, or OpenAI SDK.
- Supported generating endpoint is only `POST /v1/chat/completions`.
- User and tool text are tokenized with `allow_special=0`; XTML structure uses `allow_special=1` one segment at a time.
- Parse output control structure by token ID, not by scanning decoded model text.
- Bind to `127.0.0.1:8000` by default.
- Refuse non-loopback startup without a bearer API key.
- Reject unsupported behavior-changing request fields with an OpenAI-shaped 400.
- A concurrent generation receives 429 immediately.
- Preserve WARP's Apache-2.0 copyright headers in adapted files and add NOTICE attribution.

---

### Task 1: Establish provenance and transplant test corpora

**Files:**
- Create: `k3serve/__init__.py`
- Create: `tests/k3serve/__init__.py`
- Create: `tests/k3serve/corpus.py`
- Create: `tests/k3serve/fixtures/xtml_golden.json`
- Create: `tests/k3serve/test_corpus.py`
- Modify: `NOTICE`

**Interfaces:**
- Consumes: SQLiteAI WARP commit `2056d44f075c5e73c69c90d986e0dc89e9c9c6dd`
- Produces: pinned source provenance and K3-only XTML corpus

- [ ] **Step 1: Record the source commit and license before copying code**

Add this NOTICE block:

```text
k3serve XTML, response-region, request-normalization, and HTTP test architecture
is adapted from SQLiteAI WARP, commit
2056d44f075c5e73c69c90d986e0dc89e9c9c6dd.
Copyright 2026 SQLite Cloud, Inc.; licensed under Apache-2.0.
The adapted files retain SPDX and modification notices.
```

- [ ] **Step 2: Copy only the text K3 corpus and goldens**

Take WARP's `tests/serve/corpus.py` and `tests/serve/fixtures/xtml_golden.json`, then remove cases containing images, response schemas, grammar, non-K3 chat formats, or WASTE container metadata. Retain cases for:

```text
system/user/assistant turns
reasoning_content history
tool declarations with JSON Schema
typed arguments: string, number, boolean, null, object, array
parallel tool calls in history
out-of-order tool results matched by tool_call_id
tool_choice auto/required/none/named
thinking effort low/high/max
literal XTML marker text in untrusted content
```

Each copied Python file begins:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# Modified 2026 for kimi-k3-in-c exact-weight serving.
```

- [ ] **Step 3: Add a corpus-shape test**

Create a test that asserts unique case names and required coverage:

```python
names = [name for name, _ in CASES]
self.assertEqual(len(names), len(set(names)))
for required in ("tool_calls_typed", "tool_result_out_of_order",
                 "tool_choice_required", "thinking_effort_max",
                 "literal_control_marker"):
    self.assertIn(required, names)
```

- [ ] **Step 4: Run and commit the provenance slice**

Run:

```bash
python3 -m unittest tests.k3serve.test_corpus -v
git add NOTICE k3serve tests/k3serve
git commit -m "test(server): pin the K3 XTML corpus and provenance"
```

### Task 2: Bind the exact C runtime through ctypes

**Files:**
- Create: `k3serve/engine.py`
- Create: `tests/k3serve/test_engine.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `include/k3/k3_runtime.h`, installed or repository `libk3`
- Produces: `Engine`, `EngineError`, `Cancelled`, `Busy`, and exact tokenizer/generation methods

- [ ] **Step 1: Write structure and signature tests first**

The test imports ctypes structures and asserts their field order matches the C header:

```python
self.assertEqual([n for n, _ in RuntimeOptions._fields_], [
    "trunk_dir", "tokenizer_dir", "config_path",
    "trunk_budget_bytes", "expert_cache_bytes", "context_tokens",
    "layers", "incremental"])
self.assertEqual([n for n, _ in GenerationOptions._fields_], [
    "max_tokens", "temperature", "top_p", "seed", "greedy",
    "stop_token_ids", "stop_token_count"])
```

Also assert every C function has explicit `argtypes` and `restype` after binding.

- [ ] **Step 2: Run and observe the missing module**

Run:

```bash
python3 -m unittest tests.k3serve.test_engine -v
```

Expected: import failure for `k3serve.engine`.

- [ ] **Step 3: Implement library discovery and C declarations**

Search in this exact order:

```python
def library_candidates() -> list[Path]:
    suffix = {"Darwin": "dylib", "Windows": "dll"}.get(platform.system(), "so")
    explicit = os.environ.get("K3_LIB")
    repo = Path(__file__).resolve().parent.parent
    return ([Path(explicit)] if explicit else []) + [
        repo / f"libk3.{suffix}",
        repo / "build" / f"libk3.{suffix}",
        Path(f"libk3.{suffix}"),
    ]
```

Bind every function from `k3_runtime.h`; never rely on ctypes defaults for pointers or strings.

- [ ] **Step 4: Implement the Engine API**

Expose these exact methods:

```python
class Engine:
    def __init__(self, model_dir: str, *, trunk_dir: str,
                 tokenizer_dir: str, trunk_budget_bytes: int,
                 expert_cache_bytes: int, context_tokens: int): ...
    def close(self) -> None: ...
    def info(self) -> dict[str, object]: ...
    def tokenize(self, text: str, *, markup: bool = False) -> list[int]: ...
    def tokenize_segments(self, segments: list[Segment]) -> list[int]: ...
    def decode(self, ids: list[int]) -> bytes: ...
    def marker_ids(self) -> dict[int, str]: ...
    def reset(self) -> None: ...
    def try_generation(self): ...
    def generate(self, prompt: list[int], on_token: TokenCallback, *,
                 max_tokens: int, temperature: float, top_p: float,
                 seed: int, greedy: bool,
                 stop_tokens: list[int]) -> Usage: ...
```

`tokenize_segments` calls `tokenize(segment.text, markup=segment.markup)` for each segment and concatenates IDs; it never joins text first. `decode` returns raw bytes so a streaming UTF-8 incremental decoder can carry a code point across tokens.

- [ ] **Step 5: Implement non-blocking serialization and callback cancellation**

`try_generation` uses:

```python
if not self._generation_lock.acquire(blocking=False):
    raise Busy("another generation is active")
try:
    yield
finally:
    self._generation_lock.release()
```

The ctypes callback catches `BrokenPipeError`, `ConnectionResetError`, and `Cancelled`, calls `k3_runtime_cancel`, and returns zero to C. Store the callback object in a local variable for the whole native call so it cannot be garbage-collected.

- [ ] **Step 6: Test against the real shared library**

Run:

```bash
make -j2 libk3
K3_LIB="$PWD/libk3.$(python3 -c 'import platform; print("dylib" if platform.system()=="Darwin" else "so")')" \
  python3 -m unittest tests.k3serve.test_engine -v
```

Expected: ABI, missing-model mapping, marker IDs, untrusted-marker tokenization, decode sizing, and close idempotency pass.

- [ ] **Step 7: Commit the binding**

Run:

```bash
git add k3serve/engine.py tests/k3serve/test_engine.py Makefile
git commit -m "feat(server): bind the exact libk3 runtime"
```

### Task 3: Implement exact XTML request rendering

**Files:**
- Create: `k3serve/xtml.py`
- Create: `tests/k3serve/test_xtml.py`
- Modify: `tests/k3serve/fixtures/xtml_golden.json`

**Interfaces:**
- Consumes: normalized OpenAI messages/tools and `Segment(text, markup)`
- Produces: `build_chat_segments(...) -> Prompt`

- [ ] **Step 1: Port K3-only renderer tests from WARP**

Tests compare every retained corpus case against the frozen golden and assert:

```python
for segment in prompt.segments:
    if segment.markup:
        self.assertIn(segment.text, {
            "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"
        })
```

Add a differential token test using the checked official tokenizer fixture:

```python
got = engine.tokenize_segments(prompt.segments)
self.assertEqual(got, golden["token_ids"])
```

- [ ] **Step 2: Run and observe the missing renderer**

Run:

```bash
python3 -m unittest tests.k3serve.test_xtml -v
```

Expected: import failure for `k3serve.xtml`.

- [ ] **Step 3: Adapt the renderer with retained license headers**

Define:

```python
@dataclass(frozen=True)
class Segment:
    text: str
    markup: bool

@dataclass(frozen=True)
class Prompt:
    segments: list[Segment]
    in_think: bool
    in_response: bool
```

Use WARP's attribute escaping, typed tool argument formatting, tool declaration format, tool-call history, and `tool_call_id` matching. Delete all image placeholder and response-schema branches. Valid thinking efforts are exactly `{"low", "high", "max"}` because Moonshot's encoder rejects `medium`.

- [ ] **Step 4: Enforce structural isolation**

Only `_marker()` may construct `Segment(..., markup=True)`. All role content, reasoning history, tool names, schemas, tool arguments, and tool results use `markup=False`, including strings that spell `<|end_of_msg|>`.

- [ ] **Step 5: Run corpus and tokenizer differential tests**

Run:

```bash
python3 -m unittest tests.k3serve.test_xtml -v
python3 tests/fixtures/chat/encoding_k3.py --help >/dev/null
```

Expected: all retained WARP goldens and the official fixture token IDs match.

- [ ] **Step 6: Commit the renderer**

Run:

```bash
git add k3serve/xtml.py tests/k3serve/test_xtml.py tests/k3serve/fixtures/xtml_golden.json
git commit -m "feat(server): render Kimi K3 tools as exact XTML"
```

### Task 4: Parse streamed reasoning, content, and typed tool calls by token ID

**Files:**
- Create: `k3serve/regions.py`
- Create: `tests/k3serve/test_regions.py`

**Interfaces:**
- Consumes: `(token_id, decoded_piece)` events and marker-ID map
- Produces: `Delta`, accumulated reasoning/content, OpenAI `tool_calls`

- [ ] **Step 1: Port parser tests before implementation**

Retain WARP cases for plain answers, reasoning, typed arguments, raw JSON argument blocks, multiple calls, truncation, bad element nesting, escaped attributes, and literal marker text. Add an exhaustive chunk test:

```python
for split in range(len(tokens) + 1):
    parser = RegionParser(markers=MARKERS, in_think=True)
    for token_id, piece in tokens[:split]:
        parser.feed_token(token_id, piece)
    for token_id, piece in tokens[split:]:
        parser.feed_token(token_id, piece)
    parser.finish()
    self.assertEqual(parser.message(), expected)
```

- [ ] **Step 2: Run and observe missing parser**

Run:

```bash
python3 -m unittest tests.k3serve.test_regions -v
```

Expected: import failure for `k3serve.regions`.

- [ ] **Step 3: Adapt WARP's incremental parser**

Retain:

```python
@dataclass
class ToolCall:
    name: str = ""
    index: int = 0
    id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    json_block: Optional[str] = None

@dataclass
class Delta:
    reasoning: str = ""
    content: str = ""
    tool_calls: list[int] = field(default_factory=list)
```

`feed_token` treats a piece as structure only when its token ID exists in `markers`. A non-marker token whose decoded bytes spell a marker remains ordinary content.

- [ ] **Step 4: Define malformed-output behavior**

An unfinished think/response element is returned as accumulated text with `finish_reason="length"`. An unfinished tool call is omitted unless it has a non-empty function name. Invalid typed values remain their raw string. No model-produced parse shape raises out of the streaming callback.

- [ ] **Step 5: Run the parser suite**

Run:

```bash
python3 -m unittest tests.k3serve.test_regions -v
```

Expected: all token-boundary, typed-value, malformed-output, and injection tests pass.

- [ ] **Step 6: Commit the parser**

Run:

```bash
git add k3serve/regions.py tests/k3serve/test_regions.py
git commit -m "feat(server): parse streamed XTML by control-token id"
```

### Task 5: Validate OpenAI requests and build response objects

**Files:**
- Create: `k3serve/api.py`
- Create: `tests/k3serve/test_api.py`

**Interfaces:**
- Consumes: decoded JSON request body
- Produces: `ChatRequest`, `APIError`, OpenAI blocking/SSE payload helpers

- [ ] **Step 1: Write validation table tests**

Test accepted top-level keys:

```python
ALLOWED = {
    "model", "messages", "tools", "tool_choice", "stream",
    "stream_options", "max_tokens", "max_completion_tokens",
    "temperature", "top_p", "seed", "stop", "reasoning_effort",
    "thinking_effort", "thinking"
}
```

Test rejection of `n != 1`, logprobs, audio, image content parts, response formats, unknown keys, empty messages, invalid roles, unmatched tool results, invalid tool schemas, and conflicting `max_tokens` fields.

- [ ] **Step 2: Run and observe missing API module**

Run:

```bash
python3 -m unittest tests.k3serve.test_api -v
```

Expected: import failure for `k3serve.api`.

- [ ] **Step 3: Implement OpenAI-shaped errors**

Use:

```python
class APIError(Exception):
    def __init__(self, message: str, *, status: int = 400,
                 type: str = "invalid_request_error",
                 param: Optional[str] = None,
                 code: Optional[str] = None): ...

    def to_json(self) -> dict:
        return {"error": {"message": self.message, "type": self.type,
                          "param": self.param, "code": self.code}}
```

- [ ] **Step 4: Normalize request semantics explicitly**

`validate_chat_request` returns one immutable object with model, messages, tools, tool choice, stream flag, max tokens, temperature, top-p, seed, greedy flag, stop strings, thinking flag, and effort. `reasoning_effort` values `none|minimal|off` disable thinking; low/high/max enable it; `medium` returns 400.

- [ ] **Step 5: Implement response helpers**

Blocking response shape:

```python
{
  "id": request_id,
  "object": "chat.completion",
  "created": created,
  "model": model,
  "choices": [{"index": 0, "message": message,
               "finish_reason": reason, "logprobs": None}],
  "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}
}
```

Streaming chunks use `object="chat.completion.chunk"`; tool-call argument strings arrive as deltas and the final data record is `[DONE]`.

- [ ] **Step 6: Run validation tests and commit**

Run:

```bash
python3 -m unittest tests.k3serve.test_api -v
git add k3serve/api.py tests/k3serve/test_api.py
git commit -m "feat(server): validate OpenAI chat requests fail closed"
```

### Task 6: Serve blocking and SSE chat over real sockets

**Files:**
- Create: `k3serve/server.py`
- Create: `tests/k3serve/fake_engine.py`
- Create: `tests/k3serve/test_server.py`

**Interfaces:**
- Consumes: Engine, XTML renderer, parser, validated request
- Produces: `/health`, `/v1/models`, `/v1/chat/completions`

- [ ] **Step 1: Port socket tests before server code**

Use a real ephemeral port and `urllib`/`http.client`. Cover:

```text
health and model discovery while idle and while generating
blocking reasoning/content response
blocking typed tool call
SSE line and chunked-transfer framing
stream reconstruction equals blocking response
tool-call name and argument deltas
usage when stream_options.include_usage is true
400 malformed JSON and unsupported fields
401 missing/wrong bearer token
404 unknown path
413 body larger than configured limit
429 second concurrent generation
client disconnect cancellation and lock release
request logs do not contain prompt/tool/API-key text
```

- [ ] **Step 2: Run and observe missing server**

Run:

```bash
python3 -m unittest tests.k3serve.test_server -v
```

Expected: import failure for `k3serve.server`.

- [ ] **Step 3: Build a threaded HTTP/1.1 server**

Use:

```python
class K3HTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
```

Set `Content-Length` on JSON responses and manual chunked framing for SSE. Cap request bodies at 8 MiB, total message text at 4 MiB, and serialized tools at 2 MiB before prompt construction.

- [ ] **Step 4: Keep health responsive and reject concurrent generation**

Only the chat generation block enters `engine.try_generation()`. Convert `Busy` to:

```python
APIError("another generation is active", status=429,
         type="rate_limit_error", code="generation_busy")
```

Health and models endpoints never acquire that lock.

- [ ] **Step 5: Stream token deltas and propagate disconnect**

For each native token, use an incremental UTF-8 decoder, call `parser.feed_token(token_id, text)`, and emit only the returned delta. On socket failure, raise `Cancelled`; Engine calls `k3_runtime_cancel`, native generation unwinds, and `try_generation` releases its lock.

- [ ] **Step 6: Run all socket tests**

Run:

```bash
python3 -m unittest tests.k3serve.test_server -v
```

Expected: all endpoint, framing, auth, limit, busy, disconnect, and log-redaction tests pass.

- [ ] **Step 7: Commit HTTP/SSE behavior**

Run:

```bash
git add k3serve/server.py tests/k3serve/fake_engine.py tests/k3serve/test_server.py
git commit -m "feat(server): expose OpenAI chat with streaming tools"
```

### Task 7: Add service CLI, whole-stack tests, and operator documentation

**Files:**
- Create: `k3serve/__main__.py`
- Create: `tests/k3serve/test_integration.py`
- Create: `examples/opencode.jsonc`
- Create: `docs/SERVER.md`
- Modify: `README.md`
- Modify: `Makefile`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: complete server and real shared library
- Produces: `python3 -m k3serve`, whole-stack acceptance commands, OpenCode example

- [ ] **Step 1: Write CLI safety tests**

Assert defaults and refusal:

```python
self.assertEqual(parse_args([MODEL]).host, "127.0.0.1")
self.assertEqual(parse_args([MODEL]).port, 8000)
with self.assertRaises(SystemExit):
    parse_args([MODEL, "--host", "0.0.0.0"])
self.assertEqual(parse_args([MODEL, "--host", "0.0.0.0",
                             "--api-key", "secret"]).host, "0.0.0.0")
```

`--api-key-env K3_API_KEY` reads the key from the environment without placing it in process arguments.

- [ ] **Step 2: Implement service CLI**

Required flags:

```text
model directory
--trunk DIR
--tok DIR (defaults to model directory)
--host 127.0.0.1
--port 8000
--model-id kimi-k3-local
--trunk-gb 110
--cache-gb 13
--ctx 4096
--max-tokens 512
--api-key-env K3_API_KEY
--no-request-log
```

Refuse a non-loopback host when the named key environment variable is unset or empty.

- [ ] **Step 3: Add a whole-stack scripted-model test**

Run real HTTP, real XTML tokenizer, real parser, and fake deterministic generation. Assert one first response produces `get_weather({"city":"Paris"})`; append the assistant tool call and `role="tool"` result to the next request; assert the second response is ordinary content and contains no `<|open|>`, `<|close|>`, `<|sep|>`, or `<|end_of_msg|>`.

- [ ] **Step 4: Add the OpenCode configuration**

Create `examples/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "kimi-k3-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Kimi K3 Local",
      "options": {
        "baseURL": "http://10.10.10.106:8000/v1",
        "apiKey": "{env:K3_API_KEY}"
      },
      "models": {
        "kimi-k3-local": {
          "name": "Kimi K3 Local",
          "reasoning": true,
          "tool_call": true
        }
      }
    }
  },
  "model": "kimi-k3-local/kimi-k3-local"
}
```

- [ ] **Step 5: Document exact weights and request examples**

`docs/SERVER.md` includes startup, curl blocking/streaming/tool examples, OpenCode setup, auth, body caps, 429 behavior, one-generation boundary, model RAM settings, and the exact statement:

```text
The server reads Moonshot's official native MXFP4 expert weights and BF16 trunk.
It does not create or use a post-training quantized model.
```

- [ ] **Step 6: Run the complete Python and C matrix**

Run:

```bash
make clean && make -j2 portable libk3
make test
python3 -m unittest discover -s tests/k3serve -t . -v
python3 -m k3serve --help
```

Expected: C exactness gates and every Python unit/socket/integration test pass.

- [ ] **Step 7: Commit the complete MVP service**

Run:

```bash
git add k3serve tests/k3serve examples/opencode.jsonc docs/SERVER.md README.md Makefile CMakeLists.txt
git commit -m "feat(server): ship the exact-weight OpenAI MVP"
```
