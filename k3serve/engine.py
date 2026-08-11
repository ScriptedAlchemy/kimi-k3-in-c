"""ctypes binding for the exact-weight libk3 runtime."""

from __future__ import annotations

import ctypes as C
import os
import platform
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, ClassVar, Iterator

K3_OK = 0
K3_E_BUSY = -5
K3_E_CANCELLED = -6


class EngineError(RuntimeError):
    """The native runtime rejected an operation."""


class Busy(EngineError):
    """Another request owns the single generation lane."""


class Cancelled(EngineError):
    """Generation stopped at a callback or cancellation boundary."""


class RuntimeOptions(C.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("trunk_dir", C.c_char_p),
        ("tokenizer_dir", C.c_char_p),
        ("config_path", C.c_char_p),
        ("trunk_budget_bytes", C.c_uint64),
        ("expert_cache_bytes", C.c_uint64),
        ("context_tokens", C.c_uint32),
        ("layers", C.c_int32),
        ("incremental", C.c_int32),
    ]


class GenerationOptions(C.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("max_tokens", C.c_uint32),
        ("temperature", C.c_double),
        ("top_p", C.c_double),
        ("seed", C.c_uint64),
        ("greedy", C.c_int32),
        ("stop_token_ids", C.POINTER(C.c_int32)),
        ("stop_token_count", C.c_size_t),
    ]


class Token(C.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("index", C.c_uint32),
        ("token_id", C.c_int32),
        ("piece", C.c_void_p),
        ("piece_bytes", C.c_size_t),
        ("seconds", C.c_double),
        ("expert_bytes_read", C.c_uint64),
    ]


class Usage(C.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("prompt_tokens", C.c_uint64),
        ("completion_tokens", C.c_uint64),
        ("expert_hits", C.c_uint64),
        ("expert_misses", C.c_uint64),
        ("expert_bytes_read", C.c_uint64),
        ("seconds_total", C.c_double),
        ("seconds_io", C.c_double),
    ]


class RuntimeInfo(C.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("n_layers", C.c_uint32),
        ("n_experts", C.c_uint32),
        ("top_k", C.c_uint32),
        ("hidden", C.c_uint32),
        ("vocabulary", C.c_uint32),
        ("context_tokens", C.c_uint32),
        ("model_family", C.c_char_p),
        ("weight_format", C.c_char_p),
    ]


TokenCallback = C.CFUNCTYPE(C.c_int, C.POINTER(Token), C.c_void_p)
OnToken = Callable[[int, bytes, dict[str, object]], bool | None]


def library_candidates() -> list[Path]:
    suffix = {"Darwin": "dylib", "Windows": "dll"}.get(
        platform.system(), "so")
    explicit = os.environ.get("K3_LIB")
    repo = Path(__file__).resolve().parent.parent
    candidates = ([Path(explicit)] if explicit else []) + [
        repo / f"libk3.{suffix}",
        repo / "build" / f"libk3.{suffix}",
        Path(f"libk3.{suffix}"),
    ]
    return candidates


def _bind(lib: C.CDLL) -> None:
    lib.k3_runtime_options_init.argtypes = [C.POINTER(RuntimeOptions)]
    lib.k3_runtime_options_init.restype = None
    lib.k3_generation_options_init.argtypes = [C.POINTER(GenerationOptions)]
    lib.k3_generation_options_init.restype = None
    lib.k3_runtime_open.argtypes = [
        C.c_char_p, C.POINTER(RuntimeOptions), C.POINTER(C.c_void_p)]
    lib.k3_runtime_open.restype = C.c_int
    lib.k3_runtime_close.argtypes = [C.c_void_p]
    lib.k3_runtime_close.restype = None
    lib.k3_runtime_info.argtypes = [C.c_void_p, C.POINTER(RuntimeInfo)]
    lib.k3_runtime_info.restype = C.c_int
    lib.k3_runtime_tokenize.argtypes = [
        C.c_void_p, C.c_char_p, C.c_size_t, C.c_int,
        C.POINTER(C.c_int32), C.c_size_t, C.POINTER(C.c_size_t)]
    lib.k3_runtime_tokenize.restype = C.c_int
    lib.k3_runtime_decode.argtypes = [
        C.c_void_p, C.POINTER(C.c_int32), C.c_size_t,
        C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
    lib.k3_runtime_decode.restype = C.c_int
    lib.k3_runtime_reset.argtypes = [C.c_void_p]
    lib.k3_runtime_reset.restype = C.c_int
    lib.k3_runtime_generate.argtypes = [
        C.c_void_p, C.POINTER(C.c_int32), C.c_size_t,
        C.POINTER(GenerationOptions), TokenCallback, C.c_void_p,
        C.POINTER(Usage)]
    lib.k3_runtime_generate.restype = C.c_int
    lib.k3_runtime_cancel.argtypes = [C.c_void_p]
    lib.k3_runtime_cancel.restype = C.c_int
    lib.k3_runtime_last_error.argtypes = [C.c_void_p]
    lib.k3_runtime_last_error.restype = C.c_char_p
    lib.k3_status_string.argtypes = [C.c_int]
    lib.k3_status_string.restype = C.c_char_p


def load_library() -> C.CDLL:
    failures: list[str] = []
    for candidate in library_candidates():
        try:
            lib = C.CDLL(str(candidate))
            _bind(lib)
            return lib
        except OSError as exc:
            failures.append(f"{candidate}: {exc}")
    raise EngineError("unable to load libk3; tried:\n" + "\n".join(failures))


class Engine:
    def __init__(self, model_dir: str, *, trunk_dir: str | None = None,
                 tokenizer_dir: str | None = None,
                 trunk_budget_bytes: int = 110_000_000_000,
                 expert_cache_bytes: int = 13_000_000_000,
                 context_tokens: int = 8192,
                 library: C.CDLL | None = None):
        self._lib = library or load_library()
        _bind(self._lib)
        self._runtime = C.c_void_p()
        self._generation_lock = threading.Lock()
        self._closed = False

        options = RuntimeOptions()
        self._lib.k3_runtime_options_init(C.byref(options))
        model_bytes = os.fsencode(model_dir)
        trunk_bytes = os.fsencode(trunk_dir) if trunk_dir else None
        tokenizer_bytes = os.fsencode(tokenizer_dir or model_dir)
        options.trunk_dir = trunk_bytes
        options.tokenizer_dir = tokenizer_bytes
        options.trunk_budget_bytes = trunk_budget_bytes
        options.expert_cache_bytes = expert_cache_bytes
        options.context_tokens = context_tokens
        status = self._lib.k3_runtime_open(
            model_bytes, C.byref(options), C.byref(self._runtime))
        if status != K3_OK:
            self._closed = True
            raise EngineError(self._status_message(status))

    def _status_message(self, status: int) -> str:
        detail = b""
        if self._runtime:
            detail = self._lib.k3_runtime_last_error(self._runtime) or b""
        base = self._lib.k3_status_string(status) or b"unknown error"
        message = base.decode("utf-8", "replace")
        if detail:
            message += ": " + detail.decode("utf-8", "replace")
        return message

    def _check(self, status: int) -> None:
        if status == K3_OK:
            return
        message = self._status_message(status)
        if status == K3_E_BUSY:
            raise Busy(message)
        if status == K3_E_CANCELLED:
            raise Cancelled(message)
        raise EngineError(message)

    def close(self) -> None:
        if not self._closed:
            self._lib.k3_runtime_close(self._runtime)
            self._runtime = C.c_void_p()
            self._closed = True

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def info(self) -> dict[str, object]:
        info = RuntimeInfo()
        self._check(self._lib.k3_runtime_info(self._runtime, C.byref(info)))
        return {
            "n_layers": info.n_layers,
            "n_experts": info.n_experts,
            "top_k": info.top_k,
            "hidden": info.hidden,
            "vocabulary": info.vocabulary,
            "context_tokens": info.context_tokens,
            "model_family": (info.model_family or b"").decode(),
            "weight_format": (info.weight_format or b"").decode(),
        }

    def tokenize(self, text: str, *, markup: bool = False) -> list[int]:
        raw = text.encode("utf-8")
        needed = C.c_size_t()
        self._check(self._lib.k3_runtime_tokenize(
            self._runtime, raw, len(raw), int(markup), None, 0,
            C.byref(needed)))
        if not needed.value:
            return []
        output = (C.c_int32 * needed.value)()
        self._check(self._lib.k3_runtime_tokenize(
            self._runtime, raw, len(raw), int(markup), output, needed.value,
            C.byref(needed)))
        return list(output[:needed.value])

    def tokenize_segments(self, segments: list[object]) -> list[int]:
        ids: list[int] = []
        for segment in segments:
            ids.extend(self.tokenize(segment.text, markup=segment.markup))
        return ids

    def decode(self, ids: list[int]) -> bytes:
        if not ids:
            return b""
        source = (C.c_int32 * len(ids))(*ids)
        needed = C.c_size_t()
        self._check(self._lib.k3_runtime_decode(
            self._runtime, source, len(ids), None, 0, C.byref(needed)))
        output = C.create_string_buffer(needed.value + 1)
        self._check(self._lib.k3_runtime_decode(
            self._runtime, source, len(ids), output, needed.value,
            C.byref(needed)))
        return output.raw[:needed.value]

    def marker_ids(self) -> dict[int, str]:
        markers = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")
        result: dict[int, str] = {}
        for marker in markers:
            ids = self.tokenize(marker, markup=True)
            if len(ids) != 1:
                raise EngineError(f"control marker {marker!r} encoded as {ids}")
            result[ids[0]] = marker
        return result

    def reset(self) -> None:
        self._check(self._lib.k3_runtime_reset(self._runtime))

    def cancel(self) -> None:
        self._check(self._lib.k3_runtime_cancel(self._runtime))

    @contextmanager
    def try_generation(self) -> Iterator[None]:
        if not self._generation_lock.acquire(blocking=False):
            raise Busy("another generation is active")
        try:
            yield
        finally:
            self._generation_lock.release()

    def generate(self, prompt: list[int], on_token: OnToken, *,
                 max_tokens: int = 256, temperature: float = 1.0,
                 top_p: float = 0.95, seed: int = 0, greedy: bool = True,
                 stop_tokens: list[int] | None = None) -> dict[str, object]:
        with self.try_generation():
            return self.generate_locked(
                prompt, on_token, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, seed=seed,
                greedy=greedy, stop_tokens=stop_tokens)

    def generate_locked(self, prompt: list[int], on_token: OnToken, *,
                        max_tokens: int = 256, temperature: float = 1.0,
                        top_p: float = 0.95, seed: int = 0,
                        greedy: bool = True,
                        stop_tokens: list[int] | None = None
                        ) -> dict[str, object]:
        """Generate while the caller owns ``try_generation()``."""
        if not prompt:
            raise EngineError("prompt is empty")
        prompt_array = (C.c_int32 * len(prompt))(*prompt)
        stop_tokens = stop_tokens or []
        stop_array = ((C.c_int32 * len(stop_tokens))(*stop_tokens)
                      if stop_tokens else None)
        options = GenerationOptions()
        self._lib.k3_generation_options_init(C.byref(options))
        options.max_tokens = max_tokens
        options.temperature = temperature
        options.top_p = top_p
        options.seed = seed
        options.greedy = int(greedy)
        options.stop_token_ids = stop_array
        options.stop_token_count = len(stop_tokens)
        usage = Usage()
        callback_error: list[BaseException] = []

        @TokenCallback
        def callback(token_pointer: C.POINTER(Token), _user: C.c_void_p) -> int:
            try:
                token = token_pointer.contents
                piece = C.string_at(token.piece, token.piece_bytes)
                keep_going = on_token(token.token_id, piece, {
                    "index": token.index,
                    "seconds": token.seconds,
                    "expert_bytes_read": token.expert_bytes_read,
                })
                return 0 if keep_going is False else 1
            except BaseException as exc:
                callback_error.append(exc)
                self._lib.k3_runtime_cancel(self._runtime)
                return 0

        status = self._lib.k3_runtime_generate(
            self._runtime, prompt_array, len(prompt), C.byref(options),
            callback, None, C.byref(usage))
        if callback_error:
            raise callback_error[0]
        self._check(status)
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "expert_hits": usage.expert_hits,
            "expert_misses": usage.expert_misses,
            "expert_bytes_read": usage.expert_bytes_read,
            "seconds_total": usage.seconds_total,
            "seconds_io": usage.seconds_io,
        }
