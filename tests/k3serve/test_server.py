import contextlib
import http.client
import json
import threading
import unittest

from k3serve import xtml
from k3serve.engine import Busy
from k3serve.server import serve


class FakeEngine:
    def __init__(self, message=None, *, thinking=True):
        self._lock = threading.Lock()
        self.thinking = thinking
        self.message = message or {
            "role": "assistant", "reasoning_content": "Check facts. ",
            "content": "It works."}
        self._markers = {1000: "<|open|>", 1001: "<|close|>",
                         1002: "<|sep|>", 1003: "<|end_of_msg|>"}

    def info(self):
        return {"n_layers": 93, "n_experts": 896, "top_k": 16,
                "hidden": 7168, "vocabulary": 163600,
                "context_tokens": 8192, "model_family": "Kimi K3",
                "weight_format": "official MXFP4 experts + BF16 trunk"}

    def marker_ids(self):
        return dict(self._markers)

    def tokenize_segments(self, segments):
        return list(range(1, len(segments) + 1))

    @contextlib.contextmanager
    def try_generation(self):
        if not self._lock.acquire(blocking=False):
            raise Busy("another generation is active")
        try:
            yield
        finally:
            self._lock.release()

    def generate_locked(self, prompt, callback, **_options):
        segments = xtml._render_assistant_segments(
            xtml.normalize_message(self.message),
            xtml._ImagePromptState(None), self.thinking)
        segments = segments[3:]  # generation prompt already opened the channel
        count = 0
        marker_to_id = {value: key for key, value in self._markers.items()}
        for segment in segments:
            token_id = marker_to_id.get(segment.text, 7)
            if callback(token_id, segment.text.encode(), {"index": count}) is False:
                break
            count += 1
        return {"prompt_tokens": len(prompt), "completion_tokens": count,
                "expert_hits": 0, "expert_misses": 0,
                "expert_bytes_read": 0, "seconds_total": 0.01,
                "seconds_io": 0.0}


class RunningServer:
    def __init__(self, engine, **kwargs):
        self.server = serve(engine, port=0, log_requests=False, **kwargs)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=3)
        payload = json.dumps(body) if body is not None else None
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        connection.request(method, path, payload, request_headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, response.getheader("Content-Type"), data


class TestServer(unittest.TestCase):
    def test_health_models_and_auth(self):
        running = RunningServer(FakeEngine(), api_key="secret")
        try:
            status, _, data = running.request("GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(data)["status"], "ok")
            status, _, _ = running.request("GET", "/v1/models")
            self.assertEqual(status, 401)
            status, _, data = running.request(
                "GET", "/v1/models",
                headers={"Authorization": "Bearer secret"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(data)["data"][0]["id"], "kimi-k3")
        finally:
            running.close()

    def test_blocking_reasoning_and_content(self):
        running = RunningServer(FakeEngine())
        try:
            status, _, data = running.request("POST", "/v1/chat/completions", {
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 64,
            })
            self.assertEqual(status, 200, data)
            body = json.loads(data)
            message = body["choices"][0]["message"]
            self.assertEqual(message["reasoning_content"], "Check facts. ")
            self.assertEqual(message["content"], "It works.")
            self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        finally:
            running.close()

    def test_streaming_typed_tool_call(self):
        message = {"role": "assistant", "reasoning_content": "Need tool.",
                   "content": None, "tool_calls": [{"id": "call_weather",
                       "type": "function", "function": {
                           "name": "weather", "arguments": json.dumps({
                               "city": "Paris", "days": 2, "metric": True})}}]}
        running = RunningServer(FakeEngine(message))
        try:
            status, content_type, data = running.request(
                "POST", "/v1/chat/completions", {
                    "model": "kimi-k3", "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "weather?"}],
                    "tools": [{"type": "function", "function": {
                        "name": "weather", "description": "weather",
                        "parameters": {"type": "object"}}}],
                })
            self.assertEqual(status, 200, data)
            self.assertEqual(content_type, "text/event-stream")
            text = data.decode()
            self.assertIn('"reasoning_content": "Need tool."', text)
            self.assertIn('"name": "weather"', text)
            self.assertIn('\\"days\\": 2', text)
            self.assertIn('data: [DONE]', text)
        finally:
            running.close()

    def test_busy_is_429(self):
        engine = FakeEngine()
        running = RunningServer(engine)
        engine._lock.acquire()
        try:
            status, _, data = running.request("POST", "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "test"}]})
            self.assertEqual(status, 429, data)
            self.assertEqual(json.loads(data)["error"]["code"],
                             "generation_busy")
        finally:
            engine._lock.release()
            running.close()

    def test_non_loopback_requires_key(self):
        with self.assertRaisesRegex(ValueError, "requires --api-key"):
            serve(FakeEngine(), host="0.0.0.0", port=0, log_requests=False)


if __name__ == "__main__":
    unittest.main()
