"""LeanMoE Phase 3A runtime backend.

Phase 3A deliberately keeps policy/orchestration in Python while NativeBridge
owns native execution. HTTP/OpenAI compatibility, KV reuse and dynamic placement are later phases.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .native import (
    BRIDGE_API_VERSION,
    LM_KV_Q8_0,
    ContextConfig,
    ModelConfig,
    NativeBridge,
)
from .sampling import NativeSampler, SamplingConfig


@dataclass(frozen=True)
class BackendConfig:
    n_gpu_layers: int = 41
    n_cpu_moe: int = 30
    n_ctx: int = 81_920
    n_batch: int = 1_024
    n_ubatch: int = 1_024
    flash_attn: int = 1
    offload_kqv: int = 1
    use_mmap: int = 1
    use_mlock: int = 0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str


class LeanMoEBackend:
    """Single-model, single-context Phase 3A backend.

    Lifecycle:
        backend = LeanMoEBackend(runtime_dir, model_path)
        backend.open()
        result = backend.generate("...", max_tokens=128)
        backend.close()

    Phase 3B.1C supports both legacy argmax-greedy and native sampling.
    """

    def __init__(
        self,
        runtime_dir: str | Path,
        model_path: str | Path,
        config: BackendConfig | None = None,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.model_path = Path(model_path)
        self.config = config or BackendConfig()

        self._bridge: Optional[NativeBridge] = None
        self._model = None
        self._context = None
        self._opened = False
        self._position = 0
        self._last_finish_reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def position(self) -> int:
        return self._position

    @property
    def last_finish_reason(self) -> Optional[str]:
        return self._last_finish_reason

    def open(self) -> "LeanMoEBackend":
        if self._opened:
            return self

        if not self.runtime_dir.is_dir():
            raise FileNotFoundError(f"Runtime directory not found: {self.runtime_dir}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        bridge = NativeBridge(self.runtime_dir)
        try:
            bridge.load()
            api = bridge.api_version()
            if api != BRIDGE_API_VERSION:
                raise RuntimeError(
                    f"Bridge API mismatch: runtime={api}, python={BRIDGE_API_VERSION}"
                )
            bridge.init()

            model = bridge.model_load(
                self.model_path,
                ModelConfig(
                    n_gpu_layers=self.config.n_gpu_layers,
                    n_cpu_moe=self.config.n_cpu_moe,
                    use_mmap=self.config.use_mmap,
                    use_mlock=self.config.use_mlock,
                ),
            )

            context = bridge.context_create(
                model,
                ContextConfig(
                    n_ctx=self.config.n_ctx,
                    n_batch=self.config.n_batch,
                    n_ubatch=self.config.n_ubatch,
                    type_k=LM_KV_Q8_0,
                    type_v=LM_KV_Q8_0,
                    flash_attn=self.config.flash_attn,
                    offload_kqv=self.config.offload_kqv,
                ),
            )
        except BaseException:
            # Clean partially-created native state without masking original error.
            try:
                if "context" in locals() and context is not None:
                    bridge.context_free(context)
            except Exception:
                pass
            try:
                if "model" in locals() and model is not None:
                    bridge.model_free(model)
            except Exception:
                pass
            try:
                bridge.shutdown()
            except Exception:
                pass
            try:
                bridge.close()
            except Exception:
                pass
            raise

        self._bridge = bridge
        self._model = model
        self._context = context
        self._position = 0
        self._last_finish_reason = None
        self._opened = True
        return self

    def close(self) -> None:
        bridge = self._bridge
        if bridge is None:
            self._opened = False
            return

        if self._context is not None:
            try:
                bridge.context_free(self._context)
            finally:
                self._context = None

        if self._model is not None:
            try:
                bridge.model_free(self._model)
            finally:
                self._model = None

        try:
            bridge.shutdown()
        finally:
            try:
                bridge.close()
            finally:
                self._bridge = None
                self._opened = False
                self._position = 0
                self._last_finish_reason = None

    def __enter__(self) -> "LeanMoEBackend":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_open(self) -> NativeBridge:
        if not self._opened or self._bridge is None:
            raise RuntimeError("LeanMoEBackend is not open")
        return self._bridge

    def tokenize(self, text: str) -> list[int]:
        bridge = self._require_open()
        return bridge.tokenize(
            self._model,
            text,
            add_special=False,
            parse_special=False,
        )

    def tokenize_chat_prompt(self, text: str) -> list[int]:
        """Tokenize model-rendered chat text while preserving special tokens."""
        bridge = self._require_open()
        return bridge.tokenize(
            self._model, text, add_special=False, parse_special=True,
        )

    def token_to_piece(self, token: int) -> str:
        bridge = self._require_open()
        return bridge.token_to_piece(self._model, token)

    def prefill_tokens(self, tokens: list[int]) -> int:
        """Decode prompt tokens in native batches and return new position."""
        bridge = self._require_open()
        if not tokens:
            return self._position

        if self._position + len(tokens) > self.config.n_ctx:
            raise ValueError(
                f"Context overflow: position={self._position}, "
                f"incoming={len(tokens)}, n_ctx={self.config.n_ctx}"
            )

        offset = 0
        while offset < len(tokens):
            chunk = tokens[offset : offset + self.config.n_batch]
            status = bridge.decode_tokens(
                self._context,
                chunk,
                start_pos=self._position,
            )
            if status != 0:
                raise RuntimeError(
                    f"llama_decode failed during prefill: "
                    f"status={status}, position={self._position}, "
                    f"tokens={len(chunk)}"
                )
            self._position += len(chunk)
            offset += len(chunk)

        return self._position

    def prefill(self, text: str) -> int:
        return self.prefill_tokens(self.tokenize(text))

    def stream_greedy(
        self,
        max_tokens: int = 128,
    ) -> Iterator[tuple[int, str]]:
        """Yield (token_id, piece) using Phase-3A greedy generation."""
        bridge = self._require_open()

        if self._position <= 0:
            raise RuntimeError("Cannot generate before prefill")
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if self._position + max_tokens > self.config.n_ctx:
            raise ValueError(
                f"Generation would exceed context: position={self._position}, "
                f"max_tokens={max_tokens}, n_ctx={self.config.n_ctx}"
            )

        self._last_finish_reason = None
        token = bridge.argmax_token(self._model, self._context)

        for _ in range(max_tokens):
            if bridge.token_is_eog(self._model, token):
                self._last_finish_reason = "stop"
                return

            piece = bridge.token_to_piece(self._model, token)

            status = bridge.decode_tokens(
                self._context,
                [token],
                start_pos=self._position,
            )
            if status != 0:
                raise RuntimeError(
                    f"llama_decode failed during generation: "
                    f"status={status}, position={self._position}"
                )

            self._position += 1
            yield token, piece
            token = bridge.argmax_token(self._model, self._context)

        self._last_finish_reason = "length"

    def stream_sampled(
        self,
        max_tokens: int = 128,
        sampling: SamplingConfig | None = None,
    ) -> Iterator[tuple[int, str]]:
        """Yield (token_id, piece) using the native Bridge API v4 sampler."""
        bridge = self._require_open()

        if self._position <= 0:
            raise RuntimeError("Cannot generate before prefill")
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if self._position + max_tokens > self.config.n_ctx:
            raise ValueError(
                f"Generation would exceed context: position={self._position}, "
                f"max_tokens={max_tokens}, n_ctx={self.config.n_ctx}"
            )

        self._last_finish_reason = None
        cfg = sampling or SamplingConfig()
        with NativeSampler(bridge, self._model, cfg) as sampler:
            for _ in range(max_tokens):
                token = sampler.sample(self._context)
                if bridge.token_is_eog(self._model, token):
                    self._last_finish_reason = "stop"
                    return

                piece = bridge.token_to_piece(self._model, token)

                status = bridge.decode_tokens(
                    self._context,
                    [token],
                    start_pos=self._position,
                )
                if status != 0:
                    raise RuntimeError(
                        f"llama_decode failed during sampled generation: "
                        f"status={status}, position={self._position}"
                    )

                self._position += 1
                yield token, piece

        self._last_finish_reason = "length"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        sampling: SamplingConfig | None = None,
        *,
        chat_prompt: bool = False,
    ) -> GenerationResult:
        """Generate on a fresh context.

        sampling=None preserves the Phase-3A legacy argmax path.
        Passing SamplingConfig uses the Bridge API9 native sampler, including
        SamplingConfig(greedy=True) when native-greedy behavior is desired.
        """
        self._require_open()
        if self._position != 0:
            raise RuntimeError(
                "generate() requires a fresh context. "
                "Create/reopen the backend for another independent request."
            )
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")

        prompt_ids = (
            self.tokenize_chat_prompt(prompt) if chat_prompt else self.tokenize(prompt)
        )
        if not prompt_ids:
            raise ValueError("Prompt tokenized to zero tokens")

        if len(prompt_ids) + max_tokens > self.config.n_ctx:
            raise ValueError(
                f"Prompt + generation exceeds context: "
                f"{len(prompt_ids)} + {max_tokens} > {self.config.n_ctx}"
            )

        self.prefill_tokens(prompt_ids)

        ids: list[int] = []
        pieces: list[str] = []
        stream = (
            self.stream_greedy(max_tokens=max_tokens)
            if sampling is None
            else self.stream_sampled(max_tokens=max_tokens, sampling=sampling)
        )
        for token, piece in stream:
            ids.append(token)
            pieces.append(piece)

        return GenerationResult(
            text="".join(pieces),
            token_ids=tuple(ids),
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(ids),
            finish_reason=self._last_finish_reason or "length",
        )

