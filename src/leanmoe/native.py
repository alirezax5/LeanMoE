from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Sequence


BRIDGE_API_VERSION = 12

LM_OK = 0
LM_ERROR_INVALID_ARGUMENT = -1
LM_ERROR_NOT_INITIALIZED = -2
LM_ERROR_LOAD_MODEL = -3
LM_ERROR_OUT_OF_MEMORY = -7
LM_ERROR_BACKEND = -8
LM_ERROR_INTERNAL = -100
LM_ERROR_BUFFER_TOO_SMALL = -101

LM_KV_F16 = 0
LM_KV_Q8_0 = 1
LM_KV_Q4_0 = 2
LM_KV_Q4_1 = 3

LMToken = ctypes.c_int32

class ChatMessage(ctypes.Structure):
    _fields_ = [
        ("role_utf8", ctypes.c_char_p),
        ("content_utf8", ctypes.c_char_p),
    ]


class LeanMoENativeError(RuntimeError):
    pass


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int32),
        ("n_cpu_moe", ctypes.c_int32),
        ("use_mmap", ctypes.c_uint8),
        ("use_mlock", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 6),
    ]


class ContextConfig(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        # lm_kv_type is a C enum. The current Windows/MSVC bridge ABI uses int.
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("flash_attn", ctypes.c_uint8),
        ("offload_kqv", ctypes.c_uint8),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("reserved", ctypes.c_uint8 * 6),
    ]


class NativeBridge:
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = Path(runtime_dir).resolve()
        self._dll_directory = None
        self._dll: ctypes.CDLL | None = None

    @property
    def dll(self) -> ctypes.CDLL:
        if self._dll is None:
            raise LeanMoENativeError("LeanMoE bridge is not loaded")
        return self._dll

    def load(self) -> None:
        if self._dll is not None:
            return

        dll_path = self.runtime_dir / "leanmoe_bridge.dll"
        if not dll_path.is_file():
            raise LeanMoENativeError(
                f"LeanMoE bridge not found: {dll_path}"
            )

        if os.name == "nt":
            self._dll_directory = os.add_dll_directory(
                str(self.runtime_dir)
            )

        dll = ctypes.CDLL(str(dll_path))
        self._bind_api(dll)

        version = int(dll.lm_api_version())
        if version != BRIDGE_API_VERSION:
            self._close_dll_directory()
            raise LeanMoENativeError(
                f"Bridge API mismatch: expected "
                f"{BRIDGE_API_VERSION}, got {version}"
            )

        self._dll = dll

    def close(self) -> None:
        self._dll = None
        self._close_dll_directory()

    def _close_dll_directory(self) -> None:
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None

    @staticmethod
    def _bind_api(dll: ctypes.CDLL) -> None:
        # Runtime
        dll.lm_api_version.argtypes = []
        dll.lm_api_version.restype = ctypes.c_uint32

        dll.lm_init.argtypes = []
        dll.lm_init.restype = ctypes.c_int

        dll.lm_shutdown.argtypes = []
        dll.lm_shutdown.restype = None

        dll.lm_last_error.argtypes = []
        dll.lm_last_error.restype = ctypes.c_char_p

        # Model
        dll.lm_model_load.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ModelConfig),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        dll.lm_model_load.restype = ctypes.c_int

        dll.lm_model_free.argtypes = [ctypes.c_void_p]
        dll.lm_model_free.restype = None

        # Context
        dll.lm_context_create.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ContextConfig),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        dll.lm_context_create.restype = ctypes.c_int

        dll.lm_context_free.argtypes = [ctypes.c_void_p]
        dll.lm_context_free.restype = None

        dll.lm_context_n_ctx.argtypes = [ctypes.c_void_p]
        dll.lm_context_n_ctx.restype = ctypes.c_uint32

        dll.lm_context_n_batch.argtypes = [ctypes.c_void_p]
        dll.lm_context_n_batch.restype = ctypes.c_uint32

        dll.lm_context_n_ubatch.argtypes = [ctypes.c_void_p]
        dll.lm_context_n_ubatch.restype = ctypes.c_uint32

        # Inference
        dll.lm_vocab_size.argtypes = [ctypes.c_void_p]
        dll.lm_vocab_size.restype = ctypes.c_int32

        dll.lm_token_is_eog.argtypes = [
            ctypes.c_void_p,
            LMToken,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        dll.lm_token_is_eog.restype = ctypes.c_int

        dll.lm_tokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.POINTER(LMToken),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        dll.lm_tokenize.restype = ctypes.c_int

        dll.lm_decode_tokens.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(LMToken),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        dll.lm_decode_tokens.restype = ctypes.c_int

        dll.lm_argmax_token.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(LMToken),
        ]
        dll.lm_argmax_token.restype = ctypes.c_int

        dll.lm_token_to_piece.argtypes = [
            ctypes.c_void_p,
            LMToken,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        dll.lm_token_to_piece.restype = ctypes.c_int

        dll.lm_chat_apply_template.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ChatMessage),
            ctypes.c_int32,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        dll.lm_chat_apply_template.restype = ctypes.c_int
        dll.lm_chat_template_metadata.argtypes = [
            ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_char),
            ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
        ]
        dll.lm_chat_template_metadata.restype = ctypes.c_int
        dll.lm_chat_parse_output.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        dll.lm_chat_parse_output.restype = ctypes.c_int
        dll.lm_chat_apply_structured.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
        ]
        dll.lm_chat_apply_structured.restype = ctypes.c_int
        dll.lm_chat_parse_output_structured.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
        ]
        dll.lm_chat_parse_output_structured.restype = ctypes.c_int

        # Metadata
        dll.lm_model_size.argtypes = [ctypes.c_void_p]
        dll.lm_model_size.restype = ctypes.c_uint64

        dll.lm_model_n_params.argtypes = [ctypes.c_void_p]
        dll.lm_model_n_params.restype = ctypes.c_uint64

        dll.lm_model_description.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        dll.lm_model_description.restype = ctypes.c_int

    @staticmethod
    def _chat_json_bytes(value) -> bytes:
        import json
        return json.dumps(value,ensure_ascii=False,separators=(",",":")).encode("utf-8")

    def chat_apply_structured(self,model,messages,tools=None,*,tool_choice="auto",
                              add_generation_prompt=True,enable_thinking=True,parallel_tool_calls=True):
        import json
        if tool_choice not in {"auto","none","required"}: raise ValueError("unsupported tool_choice")
        mb=self._chat_json_bytes(messages); tb=self._chat_json_bytes(tools); cb=tool_choice.encode()
        need=ctypes.c_int32()
        rc=self.dll.lm_chat_apply_structured(model,mb,tb,cb,int(add_generation_prompt),int(enable_thinking),
                                             int(parallel_tool_calls),None,0,ctypes.byref(need))
        if rc != LM_ERROR_BUFFER_TOO_SMALL: self.check(rc,"lm_chat_apply_structured(size query)")
        size=need.value+1; buf=ctypes.create_string_buffer(size); actual=ctypes.c_int32()
        rc=self.dll.lm_chat_apply_structured(model,mb,tb,cb,int(add_generation_prompt),int(enable_thinking),
                                             int(parallel_tool_calls),buf,size,ctypes.byref(actual))
        self.check(rc,"lm_chat_apply_structured()")
        return json.loads(bytes(buf.raw[:actual.value]).decode("utf-8"))

    def chat_parse_output_structured(self,model,messages,generated,tools=None,*,tool_choice="auto",
                                     enable_thinking=True,parallel_tool_calls=True,is_partial=False):
        import json
        if tool_choice not in {"auto","none","required"}: raise ValueError("unsupported tool_choice")
        mb=self._chat_json_bytes(messages); tb=self._chat_json_bytes(tools); cb=tool_choice.encode(); gb=str(generated).encode()
        need=ctypes.c_int32()
        rc=self.dll.lm_chat_parse_output_structured(model,mb,tb,cb,gb,int(enable_thinking),int(parallel_tool_calls),
                                                     int(is_partial),None,0,ctypes.byref(need))
        if rc != LM_ERROR_BUFFER_TOO_SMALL: self.check(rc,"lm_chat_parse_output_structured(size query)")
        size=need.value+1; buf=ctypes.create_string_buffer(size); actual=ctypes.c_int32()
        rc=self.dll.lm_chat_parse_output_structured(model,mb,tb,cb,gb,int(enable_thinking),int(parallel_tool_calls),
                                                     int(is_partial),buf,size,ctypes.byref(actual))
        self.check(rc,"lm_chat_parse_output_structured()")
        return json.loads(bytes(buf.raw[:actual.value]).decode("utf-8"))

    def api_version(self) -> int:
        return int(self.dll.lm_api_version())

    def last_error(self) -> str:
        raw = self.dll.lm_last_error()
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace")

    def check(self, rc: int, operation: str) -> None:
        if rc == LM_OK:
            return
        detail = self.last_error()
        suffix = f": {detail}" if detail else ""
        raise LeanMoENativeError(
            f"{operation} failed with code {rc}{suffix}"
        )

    # Runtime -------------------------------------------------------------

    def init(self) -> None:
        self.check(self.dll.lm_init(), "lm_init()")

    def shutdown(self) -> None:
        self.dll.lm_shutdown()

    # Model ---------------------------------------------------------------

    def model_load(
        self,
        model_path: Path,
        config: ModelConfig,
    ) -> ctypes.c_void_p:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise LeanMoENativeError(f"Model not found: {path}")

        model = ctypes.c_void_p()
        rc = self.dll.lm_model_load(
            str(path).encode("utf-8"),
            ctypes.byref(config),
            ctypes.byref(model),
        )
        self.check(rc, "lm_model_load()")

        if not model.value:
            raise LeanMoENativeError(
                "lm_model_load() returned a null model handle"
            )
        return model

    def model_free(self, model: ctypes.c_void_p) -> None:
        if model and model.value:
            self.dll.lm_model_free(model)

    def model_size(self, model: ctypes.c_void_p) -> int:
        return int(self.dll.lm_model_size(model))

    def model_n_params(self, model: ctypes.c_void_p) -> int:
        return int(self.dll.lm_model_n_params(model))

    def model_description(
        self,
        model: ctypes.c_void_p,
        buffer_size: int = 1024,
    ) -> str:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        buffer = ctypes.create_string_buffer(buffer_size)
        rc = self.dll.lm_model_description(
            model,
            buffer,
            buffer_size,
        )
        self.check(rc, "lm_model_description()")
        return buffer.value.decode("utf-8", errors="replace")


    def chat_apply_template(
        self,
        model: ctypes.c_void_p,
        messages: Sequence[tuple[str, str]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        if not messages:
            raise ValueError("messages must not be empty")

        encoded = [(str(role).encode("utf-8"), str(content).encode("utf-8"))
                   for role, content in messages]
        arr = (ChatMessage * len(encoded))(
            *(ChatMessage(role, content) for role, content in encoded)
        )
        required = ctypes.c_int32()
        rc = self.dll.lm_chat_apply_template(
            model, arr, len(encoded), int(add_generation_prompt),
            int(enable_thinking), None, 0, ctypes.byref(required)
        )
        if rc != LM_ERROR_BUFFER_TOO_SMALL:
            self.check(rc, "lm_chat_apply_template(size query)")
        if required.value < 0:
            raise LeanMoENativeError("invalid chat template size")

        size = required.value + 1
        buffer = ctypes.create_string_buffer(size)
        actual = ctypes.c_int32()
        rc = self.dll.lm_chat_apply_template(
            model, arr, len(encoded), int(add_generation_prompt),
            int(enable_thinking), buffer, size, ctypes.byref(actual)
        )
        self.check(rc, "lm_chat_apply_template()")
        if actual.value < 0 or actual.value >= size:
            raise LeanMoENativeError("invalid formatted chat size")
        return bytes(buffer.raw[:actual.value]).decode("utf-8", errors="strict")


    def chat_template_metadata(self, model: ctypes.c_void_p, *, enable_thinking: bool = True) -> dict:
        import json
        required=ctypes.c_int32()
        rc=self.dll.lm_chat_template_metadata(model,int(enable_thinking),None,0,ctypes.byref(required))
        if rc != LM_ERROR_BUFFER_TOO_SMALL: self.check(rc,"lm_chat_template_metadata(size query)")
        if required.value < 0: raise LeanMoENativeError("invalid chat metadata size")
        size=required.value+1; buffer=ctypes.create_string_buffer(size); actual=ctypes.c_int32()
        rc=self.dll.lm_chat_template_metadata(model,int(enable_thinking),buffer,size,ctypes.byref(actual))
        self.check(rc,"lm_chat_template_metadata()")
        return json.loads(bytes(buffer.raw[:actual.value]).decode("utf-8"))

    def chat_parse_output(
        self,
        model: ctypes.c_void_p,
        generated: str,
        *,
        enable_thinking: bool = True,
        is_partial: bool = False,
    ) -> dict:
        import json
        encoded = str(generated).encode("utf-8")
        required = ctypes.c_int32()
        rc = self.dll.lm_chat_parse_output(
            model, encoded, int(enable_thinking), int(is_partial),
            None, 0, ctypes.byref(required),
        )
        if rc != LM_ERROR_BUFFER_TOO_SMALL:
            self.check(rc, "lm_chat_parse_output(size query)")
        if required.value < 0:
            raise LeanMoENativeError("invalid parsed chat output size")
        size = required.value + 1
        buffer = ctypes.create_string_buffer(size)
        actual = ctypes.c_int32()
        rc = self.dll.lm_chat_parse_output(
            model, encoded, int(enable_thinking), int(is_partial),
            buffer, size, ctypes.byref(actual),
        )
        self.check(rc, "lm_chat_parse_output()")
        if actual.value < 0 or actual.value >= size:
            raise LeanMoENativeError("invalid parsed chat output size")
        return json.loads(bytes(buffer.raw[:actual.value]).decode("utf-8"))

    # Context -------------------------------------------------------------

    def context_create(
        self,
        model: ctypes.c_void_p,
        config: ContextConfig,
    ) -> ctypes.c_void_p:
        context = ctypes.c_void_p()
        rc = self.dll.lm_context_create(
            model,
            ctypes.byref(config),
            ctypes.byref(context),
        )
        self.check(rc, "lm_context_create()")

        if not context.value:
            raise LeanMoENativeError(
                "lm_context_create() returned a null context handle"
            )
        return context

    def context_free(self, context: ctypes.c_void_p) -> None:
        if context and context.value:
            self.dll.lm_context_free(context)

    def context_n_ctx(self, context: ctypes.c_void_p) -> int:
        return int(self.dll.lm_context_n_ctx(context))

    def context_n_batch(self, context: ctypes.c_void_p) -> int:
        return int(self.dll.lm_context_n_batch(context))

    def context_n_ubatch(self, context: ctypes.c_void_p) -> int:
        return int(self.dll.lm_context_n_ubatch(context))

    # Inference -----------------------------------------------------------

    def vocab_size(self, model: ctypes.c_void_p) -> int:
        size = int(self.dll.lm_vocab_size(model))
        if size <= 0:
            detail = self.last_error()
            suffix = f": {detail}" if detail else ""
            raise LeanMoENativeError(
                f"lm_vocab_size() returned {size}{suffix}"
            )
        return size

    def tokenize(
        self,
        model: ctypes.c_void_p,
        text: str,
        *,
        add_special: bool = False,
        parse_special: bool = False,
    ) -> list[int]:
        encoded = text.encode("utf-8")
        required = ctypes.c_int32()

        rc = self.dll.lm_tokenize(
            model,
            encoded,
            int(add_special),
            int(parse_special),
            None,
            0,
            ctypes.byref(required),
        )

        if rc == LM_OK and required.value == 0:
            return []

        if rc != LM_ERROR_BUFFER_TOO_SMALL:
            self.check(rc, "lm_tokenize(size query)")

        if required.value <= 0:
            raise LeanMoENativeError(
                "lm_tokenize() returned an invalid required token count: "
                f"{required.value}"
            )

        token_array = (LMToken * required.value)()
        actual = ctypes.c_int32()

        rc = self.dll.lm_tokenize(
            model,
            encoded,
            int(add_special),
            int(parse_special),
            token_array,
            required.value,
            ctypes.byref(actual),
        )
        self.check(rc, "lm_tokenize()")

        if actual.value < 0 or actual.value > required.value:
            raise LeanMoENativeError(
                "lm_tokenize() returned an invalid token count: "
                f"{actual.value}"
            )

        return [int(token_array[i]) for i in range(actual.value)]

    def decode_tokens(
        self,
        context: ctypes.c_void_p,
        tokens: Sequence[int],
        *,
        start_pos: int = 0,
    ) -> int:
        if not tokens:
            raise ValueError("tokens must not be empty")

        count = len(tokens)
        if count > 0x7FFFFFFF:
            raise ValueError("too many tokens for the native ABI")

        token_array = (LMToken * count)(*map(int, tokens))
        decode_status = ctypes.c_int32()

        rc = self.dll.lm_decode_tokens(
            context,
            token_array,
            count,
            start_pos,
            ctypes.byref(decode_status),
        )
        self.check(rc, "lm_decode_tokens()")

        # The bridge deliberately returns llama_decode() status separately.
        return int(decode_status.value)

    def argmax_token(
        self,
        model: ctypes.c_void_p,
        context: ctypes.c_void_p,
    ) -> int:
        token = LMToken()
        rc = self.dll.lm_argmax_token(
            model,
            context,
            ctypes.byref(token),
        )
        self.check(rc, "lm_argmax_token()")
        return int(token.value)

    def token_is_eog(self, model: ctypes.c_void_p, token: int) -> bool:
        out = ctypes.c_uint8()
        rc = self.dll.lm_token_is_eog(model, int(token), ctypes.byref(out))
        self.check(rc, "lm_token_is_eog()")
        return bool(out.value)

    def token_to_piece(
        self,
        model: ctypes.c_void_p,
        token: int,
        *,
        render_special: bool = False,
        initial_buffer_size: int = 256,
    ) -> str:
        if initial_buffer_size <= 1:
            raise ValueError("initial_buffer_size must be greater than 1")

        size = initial_buffer_size

        while True:
            buffer = ctypes.create_string_buffer(size)
            out_size = ctypes.c_int32()

            rc = self.dll.lm_token_to_piece(
                model,
                int(token),
                int(render_special),
                buffer,
                size,
                ctypes.byref(out_size),
            )

            if rc == LM_OK:
                if out_size.value < 0 or out_size.value >= size:
                    raise LeanMoENativeError(
                        "lm_token_to_piece() returned an invalid size: "
                        f"{out_size.value}"
                    )
                return bytes(buffer.raw[:out_size.value]).decode(
                    "utf-8",
                    errors="replace",
                )

            if rc != LM_ERROR_BUFFER_TOO_SMALL:
                self.check(rc, "lm_token_to_piece()")

            if out_size.value <= 0:
                raise LeanMoENativeError(
                    "lm_token_to_piece() did not provide a valid "
                    f"required buffer size: {out_size.value}"
                )

            # The bridge reports bytes excluding its optional NUL terminator.
            size = out_size.value + 1

