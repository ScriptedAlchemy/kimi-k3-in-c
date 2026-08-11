"""OpenAI request normalization and response helpers."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from . import xtml


class APIError(Exception):
    def __init__(self, message: str, *, status: int = 400,
                 error_type: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.type = error_type
        self.param = param
        self.code = code

    def to_json(self) -> dict[str, object]:
        return {"error": {"message": str(self), "type": self.type,
                          "param": self.param, "code": self.code}}


@dataclass
class Request:
    tokens: list[int]
    thinking: bool
    stream: bool
    include_usage: bool
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    greedy: bool
    stop_strings: list[str]


def _messages(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise APIError("'messages' must be a non-empty array", param="messages")
    result: list[dict[str, Any]] = []
    roles = {"system", "developer", "user", "assistant", "tool"}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise APIError(f"messages[{index}] must be an object",
                           param=f"messages[{index}]")
        role = item.get("role")
        if role not in roles:
            raise APIError(f"messages[{index}].role is unsupported",
                           param=f"messages[{index}].role")
        content = item.get("content")
        if content is not None and not isinstance(content, str):
            raise APIError("only text message content is supported",
                           param=f"messages[{index}].content")
        normalized = dict(item)
        if role == "developer":
            normalized["role"] = "system"
        result.append(normalized)
    return result


def _number(body: dict[str, Any], key: str, default: float,
            low: float, high: float) -> float:
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(f"'{key}' must be a number", param=key)
    if not low <= float(value) <= high:
        raise APIError(f"'{key}' must be between {low} and {high}", param=key)
    return float(value)


def _thinking(body: dict[str, Any], default: bool) -> tuple[bool, str | None]:
    effort = body.get("reasoning_effort", body.get("thinking_effort"))
    explicit = body.get("thinking")
    if explicit is not None and not isinstance(explicit, bool):
        raise APIError("'thinking' must be a boolean", param="thinking")
    if effort is None:
        return (default if explicit is None else explicit), None
    if not isinstance(effort, str):
        raise APIError("'reasoning_effort' must be a string",
                       param="reasoning_effort")
    effort = effort.lower()
    if effort in {"none", "minimal", "off"}:
        return False, None
    if effort not in {"low", "high", "max"}:
        raise APIError("Kimi K3 reasoning_effort must be low, high, max, or none",
                       param="reasoning_effort")
    if explicit is False:
        raise APIError("'thinking' conflicts with 'reasoning_effort'",
                       param="reasoning_effort")
    return True, effort


def _stops(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise APIError("'stop' must be a string or array of strings", param="stop")


def prepare(engine: object, body: dict[str, Any], *,
            default_max_tokens: int, default_thinking: bool) -> Request:
    messages = _messages(body.get("messages"))
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise APIError("'stream' must be a boolean", param="stream")
    if body.get("n") not in (None, 1):
        raise APIError("'n' must be 1", param="n")
    for field in ("presence_penalty", "frequency_penalty"):
        if body.get(field) not in (None, 0, 0.0):
            raise APIError(f"'{field}' is not supported", param=field)
    if body.get("logprobs") not in (None, False):
        raise APIError("'logprobs' is not supported", param="logprobs")
    if body.get("response_format") not in (None, {"type": "text"}):
        raise APIError("only text response_format is supported",
                       param="response_format")

    tools = body.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise APIError("'tools' must be an array", param="tools")
    thinking, effort = _thinking(body, default_thinking)

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function") or {}
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise APIError("named tool_choice requires function.name",
                           param="tool_choice")
        messages.append({"role": "system", "content":
                         f"You must call the tool `{name}` next, and no other tool."})
        tool_choice = "required"
    if tool_choice not in (None, "auto", "none", "required"):
        raise APIError("'tool_choice' must be auto, none, required, or named",
                       param="tool_choice")

    kwargs: dict[str, Any] = {}
    if effort:
        kwargs["thinking_effort"] = effort
    if tool_choice in {"none", "required"}:
        kwargs["tool_choice"] = tool_choice
    try:
        segments = xtml.build_chat_segments(
            messages, tools, thinking=thinking, add_generation_prompt=True,
            **kwargs)
    except (KeyError, TypeError, xtml.XTMLError) as exc:
        raise APIError(f"invalid Kimi K3 conversation: {exc}",
                       param="messages") from exc
    tokens = engine.tokenize_segments(segments)
    if not tokens:
        raise APIError("messages encoded to no tokens", param="messages")

    info = engine.info()
    context = int(info["context_tokens"])
    if len(tokens) >= context:
        raise APIError(f"prompt is {len(tokens)} tokens; context is {context}",
                       code="context_length_exceeded", param="messages")
    limit = body.get("max_completion_tokens", body.get("max_tokens",
                                                        default_max_tokens))
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise APIError("'max_tokens' must be a positive integer",
                       param="max_tokens")
    limit = min(limit, context - len(tokens))
    temperature_in = _number(body, "temperature", 0.0, 0.0, 2.0)
    top_p = _number(body, "top_p", 0.95, 0.000001, 1.0)
    seed = body.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise APIError("'seed' must be an integer", param="seed")
    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise APIError("'stream_options' must be an object",
                       param="stream_options")
    return Request(
        tokens=tokens,
        thinking=thinking,
        stream=stream,
        include_usage=bool(stream_options.get("include_usage")),
        max_tokens=limit,
        temperature=temperature_in if temperature_in > 0 else 1.0,
        top_p=top_p,
        seed=seed & 0xFFFFFFFFFFFFFFFF,
        greedy=temperature_in == 0,
        stop_strings=_stops(body.get("stop")),
    )


def identifier(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def timestamp() -> int:
    return int(time.time())


def usage(prompt: int, completion: int, reasoning: int = 0) -> dict[str, object]:
    value: dict[str, object] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if reasoning:
        value["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return value


def sse(payload: object) -> bytes:
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
