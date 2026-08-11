# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# Modified 2026 for kimi-k3-in-c exact-weight serving.
"""
test_regions.py — reading K3's replies back.

The parser has no upstream to be checked against: encoding_k3.py only
encodes. So the check is a **round trip** instead. k3serve/xtml.py renders an
assistant message the way the release does — that half is verified against
upstream in test_xtml.py — and the parser has to recover the same message
from it. Anything the encoder can express, the parser must read back.

The other half is streaming. The server emits SSE deltas while the model is
still talking, so the parser is fed chunks that split anywhere, including
between the `<|` and the `sep|>`. Every test that parses in one shot is run
again split at *every* byte offset, and the result must not change.

    python3 tests/k3serve/test_regions.py
"""

import json
import sys
import unittest
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from k3serve import xtml
from k3serve.regions import RegionParser, parse_reply


def render_assistant(message: dict, *, thinking: bool = True) -> str:
    """The text a model would emit for `message`, minus the turn wrapper.

    build_chat_segments wraps an assistant turn in `<|open|>message …` and
    closes it; what the model actually generates is what sits inside, after
    the channel the generation prompt already opened. So render the turn and
    cut the wrapper off — using the encoder verified against upstream rather
    than hand-writing the expected markup, which would only test the test.

    normalize_message first, exactly as build_chat_segments does: it is what
    turns an OpenAI arguments *string* into the mapping the renderer expects.
    Skipping it renders a call with no arguments at all — silently, since
    a string simply is not a mapping and the renderer moves on.
    """
    segs = xtml._render_assistant_segments(
        xtml.normalize_message(message), xtml._ImagePromptState(None), thinking)
    text = "".join(s.text for s in segs)
    # The generation prompt already emitted the opening of the first
    # channel; the model continues from there.
    opener = "<|open|>think<|sep|>" if thinking else "<|open|>response<|sep|>"
    assert text.startswith(opener), text[:40]
    return text[len(opener):]


def every_split(text: str, fn):
    """Run `fn` over every two-way split of `text`, plus byte-at-a-time."""
    for i in range(len(text) + 1):
        fn([text[:i], text[i:]])
    fn(list(text))


class TestRoundTrip(unittest.TestCase):
    """Everything the encoder can say, the parser must read back."""

    def assert_round_trip(self, message: dict, *, thinking: bool = True):
        text = render_assistant(message, thinking=thinking)
        got = parse_reply(text, in_think=thinking, in_response=not thinking)

        self.assertEqual(got.content, message.get("content") or "")
        expected_reasoning = (message.get("reasoning_content")
                              or message.get("reasoning") or "")
        self.assertEqual(got.reasoning, expected_reasoning if thinking else "")

        expected_calls = message.get("tool_calls") or []
        self.assertEqual(len(got.tool_calls), len(expected_calls),
                         f"tool call count: {got.tool_calls}")
        for want, have in zip(expected_calls, got.tool_calls):
            fn = want.get("function", want)
            self.assertEqual(have.name, fn["name"])
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    # Unparseable arguments survive as the raw block.
                    self.assertEqual(have.json_block, fn["arguments"])
                    continue
            self.assertEqual(have.arguments, args)

    def test_plain_answer(self):
        self.assert_round_trip({"role": "assistant", "content": "Hello there."})

    def test_answer_with_reasoning(self):
        self.assert_round_trip({
            "role": "assistant", "content": "4",
            "reasoning_content": "Two plus two makes four."})

    def test_no_thinking_channel(self):
        self.assert_round_trip({"role": "assistant", "content": "Hi"},
                               thinking=False)

    def test_empty_content(self):
        self.assert_round_trip({"role": "assistant", "content": ""})

    def test_tool_call_typed_arguments(self):
        self.assert_round_trip({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "everything",
                "arguments": {"s": "text", "n": 42, "f": 1.5, "b": True,
                              "b2": False, "nul": None, "obj": {"k": "v"},
                              "arr": [1, "two", None]}}}]})

    def test_parallel_tool_calls(self):
        self.assert_round_trip({
            "role": "assistant", "content": "Checking both.",
            "reasoning_content": "Two cities, two calls.",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {
                    "name": "get_weather", "arguments": '{"city":"Paris"}'}},
                {"id": "b", "type": "function", "function": {
                    "name": "get_weather", "arguments": '{"city":"Rome"}'}},
            ]})

    def test_broken_json_arguments_survive_verbatim(self):
        self.assert_round_trip({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "f", "arguments": '{"city": Paris'}}]})

    def test_multiline_and_unicode(self):
        self.assert_round_trip({
            "role": "assistant",
            "content": "Line one\nLine two\n\n  indented — é 日本語 🎉",
            "reasoning_content": "Multi\nline\nreasoning"})


class TestStreaming(unittest.TestCase):
    """Chunk boundaries must not change the result."""

    def parse_chunked(self, chunks, **kw) -> RegionParser:
        p = RegionParser(**kw)
        for c in chunks:
            p.feed(c)
        p.finish()
        return p

    def test_split_anywhere_matches_one_shot(self):
        message = {
            "role": "assistant", "content": "It is 18C in Paris.",
            "reasoning_content": "Call the weather tool, then answer.",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "get_weather",
                "arguments": {"city": "Paris", "units": "c", "n": 3}}}]}
        text = render_assistant(message)
        want = parse_reply(text, in_think=True)

        def check(chunks):
            got = self.parse_chunked(chunks, in_think=True)
            self.assertEqual(got.reasoning, want.reasoning,
                             f"reasoning差 at split {[len(c) for c in chunks]}")
            self.assertEqual(got.content, want.content)
            self.assertEqual([c.to_openai() for c in got.tool_calls],
                             [c.to_openai() for c in want.tool_calls])

        every_split(text, check)

    def test_deltas_sum_to_the_whole(self):
        """What SSE emitted must equal what the final message says.

        A client assembles the answer from deltas alone, so a parser that
        gets the totals right while dropping a delta ships a truncated
        reply to every streaming client and none to the others.
        """
        message = {"role": "assistant", "content": "Answer here.",
                   "reasoning_content": "Some thinking."}
        text = render_assistant(message)

        for size in (1, 2, 3, 5, 7, 11, 64):
            with self.subTest(chunk=size):
                p = RegionParser(in_think=True)
                reasoning, content = "", ""
                for i in range(0, len(text), size):
                    d = p.feed(text[i:i + size])
                    reasoning += d.reasoning
                    content += d.content
                d = p.finish()
                reasoning += d.reasoning
                content += d.content
                self.assertEqual(reasoning, p.reasoning)
                self.assertEqual(content, p.content)
                self.assertEqual(content, "Answer here.")

    def test_marker_split_across_chunks_is_not_emitted_as_text(self):
        p = RegionParser(in_think=True)
        p.feed("thinking<|clo")
        # The half-marker must be held back, not shipped as reasoning.
        self.assertEqual(p.reasoning, "thinking")
        p.feed("se|>think<|sep|><|open|>response<|sep|>answer")
        p.finish()
        self.assertEqual(p.reasoning, "thinking")
        self.assertEqual(p.content, "answer")

    def test_tool_call_deltas_report_their_index(self):
        message = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {
                "name": "f", "arguments": {"x": 1}}},
            {"id": "b", "type": "function", "function": {
                "name": "g", "arguments": {"y": 2}}}]}
        text = render_assistant(message)
        p = RegionParser(in_think=True)
        seen = set()
        for ch in text:
            for i in p.feed(ch).tool_calls:
                seen.add(i)
        p.finish()
        self.assertEqual(seen, {0, 1})


class TestTokenLevel(unittest.TestCase):
    """feed_token: structure is an id, not a spelling.

    The text path has to guess, because `<|sep|>` written out in prose and
    the control token look identical once they are characters. The token
    path does not: the engine reports which id it sampled, and only the
    reserved ids are structure. This is what the server uses.
    """

    # Ids are arbitrary here — what matters is that they are distinct from
    # whatever ordinary text tokenizes to. On a real container they come
    # from serve.engine.marker_ids.
    MARKERS: ClassVar[dict[int, str]] = {
        1: xtml.OPEN_TOKEN, 2: xtml.CLOSE_TOKEN,
        3: xtml.SEP_TOKEN, 4: xtml.END_OF_MSG_TOKEN,
    }

    def feed_all(self, tokens, **kw) -> RegionParser:
        p = RegionParser(markers=self.MARKERS, **kw)
        for tid, piece in tokens:
            p.feed_token(tid, piece)
        p.finish()
        return p

    def test_literal_marker_text_is_content(self):
        """The case the text path cannot get right.

        A model quoting the markup — because a user asked what `<|sep|>`
        means — emits ordinary text tokens, not the control token. The
        answer must contain those characters and the element must stay open.
        """
        p = self.feed_all([
            (2, ""), (0, "think"), (3, ""),              # </think>
            (1, ""), (0, "response"), (3, ""),           # <response>
            (0, "The marker is "), (0, "<|sep|>"), (0, " in K3."),
            (2, ""), (0, "response"), (3, ""),
        ], in_think=True)
        self.assertEqual(p.content, "The marker is <|sep|> in K3.")

    def test_literal_marker_in_tool_argument(self):
        p = self.feed_all([
            (1, ""), (0, "tools"), (3, ""),
            (1, ""), (0, 'call tool="echo" index="1"'), (3, ""),
            (1, ""), (0, 'argument key="text" type="string"'), (3, ""),
            (0, "not a real <|sep|> marker"),
            (2, ""), (0, "argument"), (3, ""),
            (2, ""), (0, "call"), (3, ""),
            (2, ""), (0, "tools"), (3, ""),
        ], in_response=True)
        self.assertEqual(p.tool_calls[0].arguments,
                         {"text": "not a real <|sep|> marker"})

    def test_matches_the_text_path_on_unambiguous_output(self):
        """Where text is unambiguous, both paths must agree."""
        message = {"role": "assistant", "content": "It is 18C.",
                   "reasoning_content": "Check the tool.",
                   "tool_calls": [{"id": "c1", "type": "function", "function": {
                       "name": "get_weather",
                       "arguments": {"city": "Paris", "n": 2}}}]}
        text = render_assistant(message)
        by_text = parse_reply(text, in_think=True)

        # Re-cut the same reply into tokens: each marker its own id, the
        # text between them one token apiece.
        by_id = {v: k for k, v in self.MARKERS.items()}
        tokens, rest = [], text
        while rest:
            pos, marker = _first_marker(rest)
            if pos < 0:
                tokens.append((0, rest))
                break
            if pos:
                tokens.append((0, rest[:pos]))
            tokens.append((by_id[marker], ""))
            rest = rest[pos + len(marker):]
        by_token = self.feed_all(tokens, in_think=True)

        self.assertEqual(by_token.reasoning, by_text.reasoning)
        self.assertEqual(by_token.content, by_text.content)
        self.assertEqual([c.to_openai() for c in by_token.tool_calls],
                         [c.to_openai() for c in by_text.tool_calls])

    def test_end_of_msg_token_finishes(self):
        p = self.feed_all([(0, "answer"), (4, "")], in_response=True)
        self.assertEqual(p.content, "answer")
        self.assertTrue(p.finished)

    def test_unknown_special_token_is_text(self):
        """A token we have no marker for is content, not structure."""
        p = self.feed_all([(0, "a"), (99, "<|media_pad|>"), (0, "b")],
                          in_response=True)
        self.assertEqual(p.content, "a<|media_pad|>b")


def _first_marker(text: str) -> tuple[int, str]:
    best, which = -1, ""
    for m in (xtml.OPEN_TOKEN, xtml.CLOSE_TOKEN, xtml.SEP_TOKEN,
              xtml.END_OF_MSG_TOKEN):
        p = text.find(m)
        if p >= 0 and (best < 0 or p < best):
            best, which = p, m
    return best, which


class TestMalformed(unittest.TestCase):
    """Model output is a prediction, not a guarantee."""

    def test_truncated_mid_reasoning(self):
        p = parse_reply("half a thought", in_think=True)
        self.assertEqual(p.reasoning, "half a thought")
        self.assertEqual(p.content, "")
        self.assertFalse(p.finished)

    def test_truncated_mid_marker(self):
        p = parse_reply("thinking<|se", in_think=True)
        self.assertEqual(p.reasoning, "thinking<|se")

    def test_truncated_mid_tag_header(self):
        p = parse_reply("done<|close|>think<|sep|><|open|>resp", in_think=True)
        self.assertEqual(p.reasoning, "done")
        # The unfinished tag comes back as text rather than vanishing.
        self.assertIn("resp", p.content)

    def test_truncated_mid_tool_argument(self):
        text = ('<|close|>think<|sep|><|open|>response<|sep|><|close|>response'
                '<|sep|><|open|>tools<|sep|><|open|>call tool="f" index="1"'
                '<|sep|><|open|>argument key="city" type="string"<|sep|>Par')
        p = parse_reply(text, in_think=True)
        self.assertEqual(len(p.tool_calls), 1)
        self.assertEqual(p.tool_calls[0].name, "f")
        # finish() closes the open argument, so the partial value survives.
        self.assertEqual(p.tool_calls[0].arguments, {"city": "Par"})

    def test_close_without_open_is_ignored(self):
        p = parse_reply("<|close|>nonesuch<|sep|>text", in_response=True)
        self.assertEqual(p.content, "text")

    def test_answer_without_any_channel(self):
        """A model that just talks still gets its answer through."""
        p = parse_reply("plain text, no markup at all")
        self.assertEqual(p.content, "plain text, no markup at all")

    def test_end_of_msg_stops_the_parse(self):
        p = parse_reply("answer<|end_of_msg|>trailing garbage",
                        in_response=True)
        self.assertEqual(p.content, "answer")
        self.assertTrue(p.finished)

    def test_feed_after_end_of_msg_is_ignored(self):
        p = RegionParser(in_response=True)
        p.feed("answer<|end_of_msg|>")
        p.feed("more")
        p.finish()
        self.assertEqual(p.content, "answer")

    def test_argument_without_key_is_dropped(self):
        text = ('<|open|>tools<|sep|><|open|>call tool="f" index="1"<|sep|>'
                '<|open|>argument type="string"<|sep|>orphan<|close|>argument'
                '<|sep|><|close|>call<|sep|><|close|>tools<|sep|>')
        p = parse_reply(text, in_response=True)
        self.assertEqual(len(p.tool_calls), 1)
        self.assertEqual(p.tool_calls[0].arguments, {})

    def test_wrong_type_keeps_the_raw_text(self):
        """`type` is the model's claim, so a lie must not become a 0."""
        text = ('<|open|>tools<|sep|><|open|>call tool="f" index="1"<|sep|>'
                '<|open|>argument key="n" type="number"<|sep|>three or four'
                '<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>')
        p = parse_reply(text, in_response=True)
        self.assertEqual(p.tool_calls[0].arguments, {"n": "three or four"})

    def test_stray_sep_is_not_emitted(self):
        p = parse_reply("a<|sep|>b", in_response=True)
        self.assertEqual(p.content, "ab")


class TestOpenAIShape(unittest.TestCase):
    def test_message_shape(self):
        text = render_assistant({
            "role": "assistant", "content": "Sure.",
            "reasoning_content": "thinking",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Paris"}}}]})
        msg = parse_reply(text, in_think=True).openai_message()
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "Sure.")
        self.assertEqual(msg["reasoning_content"], "thinking")
        call = msg["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]),
                         {"city": "Paris"})

    def test_content_is_null_when_only_tool_calls(self):
        """OpenAI clients expect content: null, not "", on a tool-call turn."""
        text = render_assistant({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "f", "arguments": {}}}]})
        msg = parse_reply(text, in_think=True).openai_message()
        self.assertIsNone(msg["content"])
        self.assertEqual(len(msg["tool_calls"]), 1)

    def test_attribute_unescaping(self):
        """&quot; then &amp; — the reverse of the order they were escaped."""
        text = ('<|open|>tools<|sep|><|open|>call tool="say &amp;quot;hi&amp;quot;" '
                'index="1"<|sep|><|close|>call<|sep|><|close|>tools<|sep|>')
        p = parse_reply(text, in_response=True)
        self.assertEqual(p.tool_calls[0].name, 'say &quot;hi&quot;')


if __name__ == "__main__":
    unittest.main(verbosity=2)
