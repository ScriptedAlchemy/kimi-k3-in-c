# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# Modified 2026 for kimi-k3-in-c exact-weight serving.
"""
regions.py — reading K3's reply back into reasoning, answer and tool calls.

k3serve/xtml.py is the port of encoding_k3.py, which only encodes. The model
answers in the same markup, and something has to turn

    <|open|>think<|sep|> …reasoning… <|close|>think<|sep|>
    <|open|>response<|sep|> …answer… <|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="get_weather" index="1"<|sep|>
        <|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>

back into `reasoning_content`, `content` and OpenAI `tool_calls`. That is
this file, and it is the half upstream leaves to the host.

It is a streaming parser, because the server has to emit SSE deltas while
the model is still talking: feed as many chunks as you like, then finish().

There are two ways to feed it, and they are not equally good.

**feed_token(id, piece)** — what the server uses. A marker counts as
structure only when it arrived as its own control token id, which the
engine reports for every token it generates. This is the output-side twin
of the tokenize / tokenize_markup split on the input side, and it exists
for the same reason: a model that writes the *characters* `<|sep|>` inside
a tool argument — because a user asked it to quote one — must not thereby
close the element. At the token level the two are simply different ids and
the question does not arise.

**feed(text)** — for hosts that only have text, and for the goldens. It
finds markers by scanning, so it cannot tell a real `<|sep|>` from a model
that spelled one out; text that looks like markup is treated as markup.
Correct for every reply that does not talk about the markup, ambiguous for
the ones that do. Chunk boundaries fall anywhere, including inside a
marker, so text is held back until it is known not to start one.

Robustness is deliberate on both paths. The parser is fed model output,
which is a prediction and not a guarantee: an unterminated element, a
`<|close|>` for something that was never opened, an `argument` outside a
`call`, a reply that stops mid-marker because it hit the token limit. Every
one of those ends as text or as a dropped element, never as an exception —
a truncated answer is still worth returning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .xtml import CLOSE_TOKEN, END_OF_MSG_TOKEN, OPEN_TOKEN, SEP_TOKEN

# Longest marker we might be holding a prefix of, so feed() knows how much
# tail to keep when a chunk ends mid-marker.
_MARKERS = (OPEN_TOKEN, CLOSE_TOKEN, SEP_TOKEN, END_OF_MSG_TOKEN)
_MAX_MARKER = max(len(m) for m in _MARKERS)


@dataclass
class ToolCall:
    """One `call` element, in the shape OpenAI wants it back."""

    name: str = ""
    index: int = 0
    id: str = ""
    # Typed `argument` children accumulate here.
    arguments: dict[str, Any] = field(default_factory=dict)
    # A raw `json` child instead: kept verbatim, because a model that
    # emitted unparseable JSON should have that surface rather than be
    # silently repaired into something it did not say.
    json_block: Optional[str] = None

    def arguments_json(self) -> str:
        """The `function.arguments` string of an OpenAI tool call."""
        if self.json_block is not None:
            return self.json_block
        return json.dumps(self.arguments, ensure_ascii=False)

    def to_openai(self) -> dict:
        return {
            "id": self.id or f"call_{self.index}",
            "type": "function",
            "function": {"name": self.name,
                         "arguments": self.arguments_json()},
        }


@dataclass
class Delta:
    """What one feed() added, for streaming.

    Every field is the *increment*, not the running total. `tool_calls`
    carries indices of calls that changed; the server reads their current
    state off the parser.
    """

    reasoning: str = ""
    content: str = ""
    tool_calls: list[int] = field(default_factory=list)


def _parse_attrs(text: str) -> tuple[str, dict[str, str]]:
    """Split a tag header into its name and attributes.

    The header is what sits between `<|open|>` and `<|sep|>`, e.g.
        message role="tool" tool="get_weather" index="1"
    Values are always quoted in what the encoder emits, and `&quot;`/`&amp;`
    are unescaped in the reverse order they were escaped — `&quot;` first,
    or the `&` it contains gets turned back into `&amp;`'s output twice.

    Bad input is not an error: a header the model mangled yields whatever
    name and attributes could be read, and the caller carries on.
    """
    text = text.strip()
    if not text:
        return "", {}

    # Tag name runs to the first space.
    sp = text.find(" ")
    if sp < 0:
        return text, {}
    name, rest = text[:sp], text[sp + 1:]

    attrs: dict[str, str] = {}
    i, n = 0, len(rest)
    while i < n:
        while i < n and rest[i].isspace():
            i += 1
        eq = rest.find("=", i)
        if eq < 0:
            break
        key = rest[i:eq].strip()
        j = eq + 1
        if j < n and rest[j] == '"':
            end = rest.find('"', j + 1)
            if end < 0:                      # unterminated: take the rest
                value, i = rest[j + 1:], n
            else:
                value, i = rest[j + 1:end], end + 1
        else:                                # unquoted: to the next space
            end = rest.find(" ", j)
            if end < 0:
                value, i = rest[j:], n
            else:
                value, i = rest[j:end], end
        if key:
            attrs[key] = value.replace("&quot;", '"').replace("&amp;", "&")
    return name, attrs


def _typed_value(raw: str, declared: str) -> Any:
    """Undo _xtml_value: rebuild the Python value an `argument` carried.

    `type` is the model's claim, so it is a hint and not a contract. When
    the text does not parse as the declared type the raw string is kept —
    a tool receiving "3 or 4" for a number is better served by seeing that
    than by a 0 the parser invented.
    """
    if declared == "string":
        return raw
    if declared == "null":
        return None
    if declared == "boolean":
        low = raw.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
        return raw
    if declared in ("number", "object", "array"):
        try:
            return json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            return raw
    # No declared type, or one we do not know: try JSON, fall back to text.
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return raw


class RegionParser:
    """Incremental XTML reader for one assistant turn.

    Usage:
        p = RegionParser()
        for chunk in stream:
            delta = p.feed(chunk)
        p.finish()
        p.reasoning, p.content, p.tool_calls

    The parser tracks an element stack, but only a handful of elements mean
    anything: think, response, tools/call/argument/json. Text outside all of
    them is appended to `content` — a model that answers without opening a
    `response` still gets its answer through, which matters because a
    non-thinking prompt opens `response` in the *prompt*, so the reply
    begins inside an element the parser never saw opened.
    """

    def __init__(self, *, in_response: bool = False, in_think: bool = False,
                 markers: Optional[dict[int, str]] = None):
        # in_response/in_think say which channel the generation prompt left
        # open. build_chat_segments ends with `<|open|>think<|sep|>` or
        # `<|open|>response<|sep|>`, and that opening is in the prompt, not
        # in the completion — so the parser has to be told, or the first
        # channel's text lands in the wrong bucket.
        #
        # markers maps control-token id -> marker string, for feed_token.
        # k3serve.engine.marker_ids builds it from the container.
        self._markers = dict(markers or {})
        self.reasoning = ""
        self.content = ""
        self.tool_calls: list[ToolCall] = []

        self._stack: list[str] = []
        if in_think:
            self._stack.append("think")
        elif in_response:
            self._stack.append("response")

        self._buf = ""            # text not yet known to be marker-free
        self._pending: Optional[str] = None   # tag header being collected
        self._pending_kind = ""   # "open" or "close"
        self._arg_key = ""
        self._arg_type = ""
        self._arg_text = ""
        self._in_json = False
        self._json_text = ""
        self._done = False        # <|end_of_msg|> seen

    # ---- public ---------------------------------------------------------

    @property
    def finished(self) -> bool:
        """True once the model closed the message with <|end_of_msg|>."""
        return self._done

    def feed(self, text: str) -> Delta:
        """Consume a chunk of text; return what it added.

        Markers are found by scanning, so a reply that spells one out in
        prose is read as structure. Prefer feed_token when token ids are
        available — see the module docstring.
        """
        delta = Delta()
        if self._done or not text:
            return delta
        self._buf += text
        self._drain(delta, final=False)
        return delta

    def feed_token(self, token_id: int, piece: str) -> Delta:
        """Consume one generated token; return what it added.

        `piece` is its text, `token_id` its id. Structure is decided by the
        id alone: a piece whose text merely looks like a marker is content,
        which is the whole point of taking the id.
        """
        delta = Delta()
        if self._done:
            return delta
        marker = self._markers.get(token_id)
        if marker is not None:
            self._marker(marker, delta)
        elif piece:
            self._literal(piece, delta)
        return delta

    def finish(self) -> Delta:
        """Flush what is held back and close what the model left open.

        A reply cut off by the token limit ends mid-element; its text is
        still delivered, because a truncated answer beats no answer.
        """
        delta = Delta()
        self._drain(delta, final=True)
        if self._pending is not None:
            # A marker was opened and never closed by <|sep|>: it was not
            # markup after all, so it is text.
            self._emit(delta, (OPEN_TOKEN if self._pending_kind == "open"
                               else CLOSE_TOKEN) + self._pending)
            self._pending = None
        if self._in_json:
            self._close_json()
        if self._arg_key:
            self._close_argument()
        return delta

    def openai_message(self) -> dict:
        """The assistant message an OpenAI client expects back."""
        msg: dict[str, Any] = {"role": "assistant"}
        msg["content"] = self.content if self.content else None
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [c.to_openai() for c in self.tool_calls]
        return msg

    # ---- internals ------------------------------------------------------

    def _drain(self, delta: Delta, *, final: bool) -> None:
        """Split the buffer into markers and the text between them.

        Not final: keep a tail that could still grow into a marker, so a
        chunk ending in "<|se" does not emit those bytes as text and then
        fail to recognise "<|sep|>" once the rest arrives.
        """
        while self._buf:
            pos, marker = _find_first(self._buf, _MARKERS)
            if pos < 0:
                keep = 0 if final else _partial_tail(self._buf, _MARKERS)
                take = len(self._buf) - keep
                if take:
                    self._literal(self._buf[:take], delta)
                self._buf = self._buf[take:]
                return

            if pos:
                self._literal(self._buf[:pos], delta)
            self._buf = self._buf[pos + len(marker):]
            self._marker(marker, delta)
            if self._done:
                self._buf = ""
                return

    def _marker(self, marker: str, delta: Delta) -> None:
        """One structural token, however it was recognised.

        The single place the two feed paths meet: feed() gets here by
        scanning text, feed_token() by matching a token id.
        """
        if marker == END_OF_MSG_TOKEN:
            self._done = True
        elif marker == OPEN_TOKEN:
            self._pending, self._pending_kind = "", "open"
        elif marker == CLOSE_TOKEN:
            self._pending, self._pending_kind = "", "close"
        elif marker == SEP_TOKEN:
            if self._pending is None:
                # A separator with no tag header open. Ignore it: turning it
                # into text would put a control marker in the answer.
                return
            header, kind = self._pending, self._pending_kind
            self._pending = None
            self._pending_kind = ""
            if kind == "open":
                self._open(header, delta)
            else:
                self._close(header, delta)

    def _literal(self, text: str, delta: Delta) -> None:
        """Text that is not structure.

        Inside a tag header — between `<|open|>` and its `<|sep|>` — it is
        the header; everywhere else it belongs to a region.
        """
        if not text:
            return
        if self._pending is not None:
            self._pending += text
            return
        self._emit(delta, text)

    def _emit(self, delta: Delta, text: str) -> None:
        """Route text to whichever region is currently open."""
        if not text:
            return
        if self._in_json:
            self._json_text += text
            if self.tool_calls:
                _mark(delta, len(self.tool_calls) - 1)
            return
        if self._arg_key:
            self._arg_text += text
            if self.tool_calls:
                _mark(delta, len(self.tool_calls) - 1)
            return
        top = self._stack[-1] if self._stack else ""
        if top == "think":
            self.reasoning += text
            delta.reasoning += text
            return
        if top in ("tools", "call"):
            # Whitespace between elements; anything else is the model
            # talking where no channel is open, and is dropped rather than
            # spliced into the answer out of order.
            return
        self.content += text
        delta.content += text

    def _open(self, header: str, delta: Delta) -> None:
        name, attrs = _parse_attrs(header)
        if name == "call":
            call = ToolCall(name=attrs.get("tool", ""))
            try:
                call.index = int(attrs.get("index", len(self.tool_calls) + 1))
            except ValueError:
                call.index = len(self.tool_calls) + 1
            call.id = attrs.get("id", "") or f"call_{call.index}"
            self.tool_calls.append(call)
            _mark(delta, len(self.tool_calls) - 1)
        elif name == "argument":
            self._arg_key = attrs.get("key", "")
            self._arg_type = attrs.get("type", "")
            self._arg_text = ""
            if not self._arg_key:
                # No key: nothing to file the value under. Collect and drop
                # at close, rather than inventing a name for it.
                self._arg_key = ""
        elif name == "json":
            self._in_json = True
            self._json_text = ""
        self._stack.append(name)

    def _close(self, header: str, delta: Delta) -> None:
        name, _ = _parse_attrs(header)
        if name == "argument":
            self._close_argument()
            if self.tool_calls:
                _mark(delta, len(self.tool_calls) - 1)
        elif name == "json":
            self._close_json()
            if self.tool_calls:
                _mark(delta, len(self.tool_calls) - 1)
        # Unwind to the matching open. A model that closed an element it
        # never opened must not pop the stack down to nothing — find the
        # tag first, and if it is not there, ignore the close entirely.
        if name in self._stack:
            while self._stack:
                popped = self._stack.pop()
                if popped == name:
                    break
        elif self._stack and not name:
            self._stack.pop()

    def _close_argument(self) -> None:
        if self._arg_key and self.tool_calls:
            self.tool_calls[-1].arguments[self._arg_key] = _typed_value(
                self._arg_text, self._arg_type)
        self._arg_key = ""
        self._arg_type = ""
        self._arg_text = ""

    def _close_json(self) -> None:
        if self.tool_calls:
            self.tool_calls[-1].json_block = self._json_text
        self._in_json = False
        self._json_text = ""


def _mark(delta: Delta, index: int) -> None:
    if index >= 0 and index not in delta.tool_calls:
        delta.tool_calls.append(index)


def _find_first(text: str, markers: tuple[str, ...]) -> tuple[int, str]:
    """Leftmost marker occurrence, and which one it was."""
    best, which = -1, ""
    for m in markers:
        p = text.find(m)
        if p >= 0 and (best < 0 or p < best):
            best, which = p, m
    return best, which


def _partial_tail(text: str, markers: tuple[str, ...]) -> int:
    """How many trailing bytes could still become one of `markers`.

    Held back rather than emitted, so a marker split across two chunks is
    still recognised when the rest of it arrives.
    """
    limit = min(len(text), _MAX_MARKER - 1)
    for k in range(limit, 0, -1):
        tail = text[-k:]
        if any(m.startswith(tail) for m in markers):
            return k
    return 0


def parse_reply(text: str, *, in_response: bool = False,
                in_think: bool = False) -> RegionParser:
    """Parse a complete reply in one call. The non-streaming path."""
    p = RegionParser(in_response=in_response, in_think=in_think)
    p.feed(text)
    p.finish()
    return p
