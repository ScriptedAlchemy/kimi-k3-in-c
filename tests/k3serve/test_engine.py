import ctypes as C
import threading
import unittest
from pathlib import Path

from k3serve.engine import (
    Engine, GenerationOptions, RuntimeInfo, RuntimeOptions, Token, Usage,
    _bind, load_library,
)


class TestABI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_library()

    def test_structure_field_order_matches_header(self):
        self.assertEqual([name for name, _ in RuntimeOptions._fields_], [
            "trunk_dir", "tokenizer_dir", "config_path",
            "trunk_budget_bytes", "expert_cache_bytes", "context_tokens",
            "layers", "incremental"])
        self.assertEqual([name for name, _ in GenerationOptions._fields_], [
            "max_tokens", "temperature", "top_p", "seed", "greedy",
            "stop_token_ids", "stop_token_count"])
        self.assertEqual([name for name, _ in Token._fields_], [
            "index", "token_id", "piece", "piece_bytes", "seconds",
            "expert_bytes_read"])
        self.assertEqual([name for name, _ in Usage._fields_][-2:],
                         ["seconds_total", "seconds_io"])
        self.assertEqual([name for name, _ in RuntimeInfo._fields_][-2:],
                         ["model_family", "weight_format"])

    def test_every_runtime_symbol_has_signatures(self):
        _bind(self.lib)
        names = (
            "k3_runtime_options_init", "k3_generation_options_init",
            "k3_runtime_open", "k3_runtime_close", "k3_runtime_info",
            "k3_runtime_tokenize", "k3_runtime_decode", "k3_runtime_reset",
            "k3_runtime_generate", "k3_runtime_cancel",
            "k3_runtime_last_error", "k3_status_string",
        )
        for name in names:
            with self.subTest(symbol=name):
                function = getattr(self.lib, name)
                self.assertIsNotNone(function.argtypes)
                self.assertTrue(hasattr(function, "restype"))

    def test_official_tokenizer_markers_and_untrusted_text(self):
        helper = self.lib.k3_runtime_open_tokenizer_for_test
        helper.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        helper.restype = C.c_int
        runtime = C.c_void_p()
        fixture = Path("tests/fixtures/chat/tokenizer").resolve()
        self.assertEqual(helper(str(fixture).encode(), C.byref(runtime)), 0)
        engine = object.__new__(Engine)
        engine._lib = self.lib
        engine._runtime = runtime
        engine._generation_lock = threading.Lock()
        engine._closed = False
        try:
            markers = engine.marker_ids()
            self.assertEqual(set(markers.values()), {
                "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"})
            structural = engine.tokenize("<|open|>", markup=True)
            untrusted = engine.tokenize("<|open|>", markup=False)
            self.assertEqual(len(structural), 1)
            self.assertNotEqual(structural, untrusted)
            self.assertEqual(engine.decode(untrusted), b"<|open|>")
        finally:
            engine.close()
            engine.close()


if __name__ == "__main__":
    unittest.main()
