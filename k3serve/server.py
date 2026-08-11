"""Dependency-free OpenAI Chat Completions HTTP/SSE server."""

from __future__ import annotations

import codecs
import hmac
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from . import api
from .engine import Busy, Cancelled, Engine, EngineError
from .regions import Delta, RegionParser

MAX_BODY_BYTES = 2 * 1024 * 1024


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler],
                 *, engine: Engine, model_id: str, api_key: str | None,
                 default_max_tokens: int, default_thinking: bool,
                 log_requests: bool):
        super().__init__(address, handler)
        self.engine = engine
        self.model_id = model_id
        self.api_key = api_key
        self.default_max_tokens = default_max_tokens
        self.default_thinking = default_thinking
        self.log_requests = log_requests
        self.started = api.timestamp()
        self.model_info = engine.info()
        self.markers = engine.marker_ids()
        self.stop_tokens = [token for token, marker in self.markers.items()
                            if marker == "<|end_of_msg|>"]


class Handler(BaseHTTPRequestHandler):
    server_version = "k3serve/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        if self.server.log_requests:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, status: int, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        key = self.server.api_key
        if not key:
            return True
        header = self.headers.get("Authorization", "")
        return header.startswith("Bearer ") and hmac.compare_digest(
            header[7:], key)

    def _body(self) -> dict[str, object]:
        length = self.headers.get("Content-Length")
        if length is None:
            raise api.APIError("Content-Length is required", status=411)
        try:
            count = int(length)
        except ValueError as exc:
            raise api.APIError("Content-Length is invalid") from exc
        if count < 1 or count > MAX_BODY_BYTES:
            raise api.APIError("request body size is invalid", status=413)
        try:
            body = json.loads(self.rfile.read(count))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise api.APIError(f"request body is not valid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise api.APIError("request body must be an object")
        return body

    def do_GET(self) -> None:
        try:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/health":
                self._json(200, {"status": "ok", "model": self.server.model_id,
                                 "weight_format": self.server.model_info[
                                     "weight_format"]})
                return
            if not self._authorized():
                raise api.APIError("invalid API key", status=401,
                                   error_type="authentication_error")
            if path == "/v1/models":
                self._json(200, {"object": "list", "data": [self._model()]})
                return
            if path == f"/v1/models/{self.server.model_id}":
                self._json(200, self._model())
                return
            raise api.APIError(f"no route for GET {path}", status=404,
                               error_type="not_found_error")
        except api.APIError as exc:
            self._json(exc.status, exc.to_json())
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        try:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if not self._authorized():
                raise api.APIError("invalid API key", status=401,
                                   error_type="authentication_error")
            if path != "/v1/chat/completions":
                raise api.APIError(f"no route for POST {path}", status=404,
                                   error_type="not_found_error")
            self._chat(self._body())
        except Busy as exc:
            self._json(429, api.APIError(str(exc), status=429,
                                        error_type="rate_limit_error",
                                        code="generation_busy").to_json())
        except api.APIError as exc:
            self._json(exc.status, exc.to_json())
        except EngineError as exc:
            self._json(500, api.APIError(str(exc), status=500,
                                        error_type="engine_error").to_json())
        except (BrokenPipeError, ConnectionResetError, Cancelled):
            return

    def _model(self) -> dict[str, object]:
        return {"id": self.server.model_id, "object": "model",
                "created": self.server.started, "owned_by": "ScriptedAlchemy",
                "k3": self.server.model_info}

    def _chat(self, body: dict[str, object]) -> None:
        with self.server.engine.try_generation():
            request = api.prepare(
                self.server.engine, body,
                default_max_tokens=self.server.default_max_tokens,
                default_thinking=self.server.default_thinking)
            request_id = api.identifier("chatcmpl")
            created = api.timestamp()
            parser = RegionParser(
                in_think=request.thinking,
                in_response=not request.thinking,
                markers=self.server.markers)
            if request.stream:
                self._stream(request, parser, request_id, created)
            else:
                self._blocking(request, parser, request_id, created)

    def _run(self, request: api.Request, parser: RegionParser,
             emit: Callable[[Delta], None]) -> dict[str, object]:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        state = {"completion": 0, "stopped": False,
                 "content_sent": 0, "reasoning_tokens": 0}

        def stop_position(text: str) -> int | None:
            positions = [text.find(stop) for stop in request.stop_strings
                         if stop and stop in text]
            return min(positions) if positions else None

        def safe_content_end(text: str) -> int:
            end = len(text)
            for stop in request.stop_strings:
                limit = min(len(stop) - 1, len(text))
                for size in range(limit, 0, -1):
                    if text.endswith(stop[:size]):
                        end = min(end, len(text) - size)
                        break
            return end

        def deliver(delta: Delta, *, final: bool = False) -> bool:
            hit = stop_position(parser.content)
            if hit is not None:
                parser.content = parser.content[:hit]
            end = len(parser.content) if final or hit is not None else safe_content_end(
                parser.content)
            sent = state["content_sent"]
            delta.content = parser.content[sent:end] if end >= sent else ""
            state["content_sent"] = end
            if delta.reasoning or delta.content or delta.tool_calls:
                emit(delta)
            return hit is not None

        def on_token(token_id: int, piece: bytes,
                     _info: dict[str, object]) -> bool:
            state["completion"] += 1
            marker = self.server.markers.get(token_id)
            text = marker if marker is not None else decoder.decode(piece, final=False)
            delta = parser.feed_token(token_id, text)
            if delta.reasoning:
                state["reasoning_tokens"] += 1
            if deliver(delta, final=parser.finished):
                state["stopped"] = True
                return False
            if parser.finished:
                state["stopped"] = True
                return False
            return True

        try:
            native_usage = self.server.engine.generate_locked(
                request.tokens, on_token,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
                greedy=request.greedy,
                stop_tokens=self.server.stop_tokens)
        except Cancelled:
            if not state["stopped"]:
                raise
            native_usage = {"prompt_tokens": len(request.tokens),
                            "completion_tokens": state["completion"]}
        trailing = decoder.decode(b"", final=True)
        tail = parser.feed(trailing) if trailing else Delta()
        finished = parser.finish()
        tail.reasoning += finished.reasoning
        tail.content += finished.content
        tail.tool_calls.extend(index for index in finished.tool_calls
                               if index not in tail.tool_calls)
        deliver(tail, final=True)
        native_usage["completion_tokens"] = state["completion"]
        native_usage["reasoning_tokens"] = state["reasoning_tokens"]
        native_usage["hit_limit"] = state["completion"] >= request.max_tokens
        native_usage["stopped"] = state["stopped"]
        return native_usage

    @staticmethod
    def _reason(parser: RegionParser, result: dict[str, object]) -> str:
        if parser.tool_calls:
            return "tool_calls"
        if result["hit_limit"] and not result["stopped"]:
            return "length"
        return "stop"

    def _blocking(self, request: api.Request, parser: RegionParser,
                  request_id: str, created: int) -> None:
        result = self._run(request, parser, lambda _delta: None)
        completion = int(result["completion_tokens"])
        payload = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": self.server.model_id,
            "choices": [{"index": 0, "message": parser.openai_message(),
                         "logprobs": None,
                         "finish_reason": self._reason(parser, result)}],
            "usage": api.usage(len(request.tokens), completion,
                               int(result["reasoning_tokens"])),
        }
        self._json(200, payload)

    def _stream(self, request: api.Request, parser: RegionParser,
                request_id: str, created: int) -> None:
        model = self.server.model_id
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        sent_role = False
        named: set[int] = set()

        def write(payload: object) -> None:
            data = api.sse(payload)
            self.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        def chunk(delta: dict[str, object], reason: str | None = None
                  ) -> dict[str, object]:
            return {"id": request_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta,
                                 "logprobs": None, "finish_reason": reason}]}

        def emit(delta: Delta) -> None:
            nonlocal sent_role
            if not sent_role:
                write(chunk({"role": "assistant", "content": ""}))
                sent_role = True
            if delta.reasoning:
                write(chunk({"reasoning_content": delta.reasoning}))
            if delta.content:
                write(chunk({"content": delta.content}))
            for index in delta.tool_calls:
                if index in named or index >= len(parser.tool_calls):
                    continue
                call = parser.tool_calls[index]
                if call.name:
                    named.add(index)
                    write(chunk({"tool_calls": [{"index": index,
                                "id": call.id or f"call_{index + 1}",
                                "type": "function", "function": {
                                    "name": call.name, "arguments": ""}}]}))

        result = self._run(request, parser, emit)
        if not sent_role:
            write(chunk({"role": "assistant", "content": ""}))
        for index, call in enumerate(parser.tool_calls):
            if index not in named:
                write(chunk({"tool_calls": [{"index": index,
                            "id": call.id or f"call_{index + 1}",
                            "type": "function", "function": {
                                "name": call.name, "arguments": ""}}]}))
            write(chunk({"tool_calls": [{"index": index,
                        "function": {"arguments": call.arguments_json()}}]}))
        write(chunk({}, self._reason(parser, result)))
        if request.include_usage:
            write({"id": request_id, "object": "chat.completion.chunk",
                   "created": created, "model": model, "choices": [],
                   "usage": api.usage(
                       len(request.tokens), int(result["completion_tokens"]),
                       int(result["reasoning_tokens"]))})
        write("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def serve(engine: Engine, *, host: str = "127.0.0.1", port: int = 8000,
          model_id: str = "kimi-k3", api_key: str | None = None,
          default_max_tokens: int = 512, default_thinking: bool = True,
          log_requests: bool = True) -> ChatServer:
    if host not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise ValueError("non-loopback serving requires --api-key")
    if ":" in host:
        ChatServer.address_family = socket.AF_INET6
    return ChatServer((host, port), Handler, engine=engine,
                      model_id=model_id, api_key=api_key,
                      default_max_tokens=default_max_tokens,
                      default_thinking=default_thinking,
                      log_requests=log_requests)
