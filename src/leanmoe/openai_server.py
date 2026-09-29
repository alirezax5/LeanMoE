from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .backend import BackendConfig, LeanMoEBackend
from .sampling import SamplingConfig


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    parts: list[str] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be an object")
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role at messages[{i}]: {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"messages[{i}].content must be a string")
        parts.append(f"<|{role}|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


class LeanMoEOpenAIServer:
    def __init__(
        self,
        runtime_dir: Path,
        model_path: Path,
        host: str = "127.0.0.1",
        port: int = 8080,
        model_id: str = "leanmoe-tiel-coder-35b-a3b",
        backend_config: BackendConfig | None = None,
    ) -> None:
        self.runtime_dir = runtime_dir
        self.model_path = model_path
        self.host = host
        self.port = port
        self.model_id = model_id
        self.backend_config = backend_config or BackendConfig()
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None

    def _request_config(self, body: dict[str, Any]):
        messages = body.get("messages")
        # Keep request validation here; the actual prompt is produced by the
        # model-embedded Jinja template after the backend/model is opened.
        messages_to_prompt(messages)

        max_tokens = body.get("max_tokens", 128)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be an integer >= 0")

        temperature = body.get("temperature", 0.8)
        top_p = body.get("top_p", 0.95)
        seed = body.get("seed", 12345)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("temperature must be numeric")
        if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
            raise ValueError("top_p must be numeric")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")

        sampling = (
            SamplingConfig(greedy=True)
            if float(temperature) == 0.0
            else SamplingConfig(
                greedy=False, temperature=float(temperature), top_k=40,
                top_p=float(top_p), min_p=0.05, seed=seed,
            )
        )
        return messages, max_tokens, sampling

    def _generate(self, body: dict[str, Any]) -> dict[str, Any]:
        messages, max_tokens, sampling = self._request_config(body)
        enable_thinking = bool(body.get("enable_thinking", True))
        with self._lock:
            with LeanMoEBackend(self.runtime_dir, self.model_path, self.backend_config) as backend:
                prompt = backend._bridge.chat_apply_template(
                    backend._model,
                    [(m["role"], m["content"]) for m in messages],
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                result = backend.generate(prompt, max_tokens=max_tokens, sampling=sampling, chat_prompt=True)
                parsed = backend._bridge.chat_parse_output(backend._model, result.text, enable_thinking=enable_thinking, is_partial=False)
        message = {"role": "assistant", "content": parsed.get("content", "")}
        reasoning = parsed.get("reasoning_content", "")
        if reasoning:
            message["reasoning_content"] = reasoning.rstrip()
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": result.finish_reason}],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.generated_tokens,
                "total_tokens": result.prompt_tokens + result.generated_tokens,
            },
        }

    def _stream(self, handler: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        messages, max_tokens, sampling = self._request_config(body)
        enable_thinking = bool(body.get("enable_thinking", True))
        cid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        def send(obj: Any) -> None:
            payload = "data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n\n"
            handler.wfile.write(payload.encode("utf-8"))
            handler.wfile.flush()

        with self._lock:
            with LeanMoEBackend(self.runtime_dir, self.model_path, self.backend_config) as backend:
                prompt = backend._bridge.chat_apply_template(
                    backend._model,
                    [(m["role"], m["content"]) for m in messages],
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                prompt_ids = backend.tokenize_chat_prompt(prompt)
                if not prompt_ids:
                    raise ValueError("Prompt tokenized to zero tokens")
                if len(prompt_ids) + max_tokens > backend.config.n_ctx:
                    raise ValueError(
                        f"Prompt + generation exceeds context: {len(prompt_ids)} + "
                        f"{max_tokens} > {backend.config.n_ctx}"
                    )
                backend.prefill_tokens(prompt_ids)

                send({"id":cid,"object":"chat.completion.chunk","created":created,
                      "model":self.model_id,
                      "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})

                # Correctness-first Phase 3C.3B: buffer model pieces so <think> tags
                # can never leak into content. Incremental reasoning streaming can
                # be optimized later without changing the API contract.
                pieces=[]
                gen=backend.stream_sampled(max_tokens=max_tokens,sampling=sampling)
                try:
                    for _, piece in gen:
                        pieces.append(piece)
                finally:
                    gen.close()

                parsed=backend._bridge.chat_parse_output(backend._model, "".join(pieces), enable_thinking=enable_thinking, is_partial=False)
                split=type("_Parsed", (), {"reasoning": parsed.get("reasoning_content", "").rstrip(), "content": parsed.get("content", "")})()
                if split.reasoning:
                    send({"id":cid,"object":"chat.completion.chunk","created":created,
                          "model":self.model_id,
                          "choices":[{"index":0,"delta":{"reasoning_content":split.reasoning},
                                      "finish_reason":None}]})
                if split.content:
                    send({"id":cid,"object":"chat.completion.chunk","created":created,
                          "model":self.model_id,
                          "choices":[{"index":0,"delta":{"content":split.content},
                                      "finish_reason":None}]})

                finish_reason=backend.last_finish_reason or "length"
                send({"id":cid,"object":"chat.completion.chunk","created":created,
                      "model":self.model_id,
                      "choices":[{"index":0,"delta":{},"finish_reason":finish_reason}]})
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()

    def make_handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "LeanMoE/3C.3B"
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                print("[HTTP] " + (fmt % args))

            def _send(self, status: int, obj: Any) -> None:
                payload = _json_bytes(obj)
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _error(self, status: int, message: str, etype: str = "invalid_request_error") -> None:
                self._send(status, {"error": {"message": message, "type": etype,
                                               "param": None, "code": None}})

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._send(200, {"status": "ok", "phase": "3C.3B"})
                elif self.path == "/v1/models":
                    self._send(200, {"object": "list", "data": [{
                        "id": owner.model_id, "object": "model",
                        "created": 0, "owned_by": "leanmoe",
                    }]})
                else:
                    self._error(404, "Not found")

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self._error(404, "Not found")
                    return
                try:
                    raw_len = self.headers.get("Content-Length")
                    if raw_len is None:
                        raise ValueError("Content-Length is required")
                    length = int(raw_len)
                    if length < 0 or length > 1_048_576:
                        raise ValueError("invalid request body size")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("request body must be a JSON object")
                    requested_model = body.get("model")
                    if requested_model not in (None, owner.model_id):
                        raise ValueError(f"unknown model: {requested_model!r}")

                    if body.get("stream", False):
                        # Validate before headers are committed so malformed
                        # requests can still receive a normal JSON 400.
                        owner._request_config(body)
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        try:
                            owner._stream(self, body)
                        except (BrokenPipeError, ConnectionResetError):
                            print("[INFO] streaming client disconnected; native stream cleanup completed")
                        finally:
                            self.close_connection = True
                    else:
                        self._send(200, owner._generate(body))
                except (ValueError, json.JSONDecodeError) as exc:
                    self._error(400, str(exc))
                except Exception as exc:
                    # If streaming headers were already committed, a JSON 500
                    # cannot be safely emitted. Close the connection instead.
                    if body.get("stream", False) if "body" in locals() else False:
                        print(f"[ERROR] streaming request failed: {type(exc).__name__}: {exc}")
                        self.close_connection = True
                    else:
                        self._error(500, f"{type(exc).__name__}: {exc}", "server_error")

        return Handler

    def serve_forever(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self.make_handler())
        print(f"LeanMoE Phase 3C.3B listening on http://{self.host}:{self.port}")
        print(f"Model: {self.model_id}")
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8080, type=int)
    ap.add_argument("--model-id", default="leanmoe-tiel-coder-35b-a3b")
    a = ap.parse_args()
    cfg = BackendConfig(
        n_gpu_layers=41, n_cpu_moe=30, n_ctx=81920,
        n_batch=1024, n_ubatch=1024, flash_attn=1,
        offload_kqv=1, use_mmap=1, use_mlock=0,
    )
    LeanMoEOpenAIServer(
        a.runtime.resolve(), a.model.resolve(), a.host, a.port,
        a.model_id, cfg
    ).serve_forever()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


