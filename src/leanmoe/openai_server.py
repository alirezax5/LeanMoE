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
        if not isinstance(body,dict): raise ValueError("request body must be a JSON object")
        model=body.get("model")
        if not isinstance(model,str) or not model: raise ValueError("model is required and must be a non-empty string")
        if model!=self.model_id: raise ValueError(f"unknown model: {model!r}")
        messages=body.get("messages"); messages_to_prompt(messages)
        stream=body.get("stream",False)
        if not isinstance(stream,bool): raise ValueError("stream must be a boolean")
        thinking=body.get("enable_thinking",True)
        if not isinstance(thinking,bool): raise ValueError("enable_thinking must be a boolean")
        if "max_tokens" in body and "max_completion_tokens" in body: raise ValueError("use only one of max_tokens or max_completion_tokens")
        max_tokens=body.get("max_completion_tokens",body.get("max_tokens",128))
        if not isinstance(max_tokens,int) or isinstance(max_tokens,bool) or max_tokens<0: raise ValueError("max_tokens/max_completion_tokens must be an integer >= 0")
        temperature=body.get("temperature",0.8); top_p=body.get("top_p",0.95); seed=body.get("seed",12345)
        if not isinstance(temperature,(int,float)) or isinstance(temperature,bool) or not 0<=float(temperature)<=2: raise ValueError("temperature must be numeric between 0 and 2")
        if not isinstance(top_p,(int,float)) or isinstance(top_p,bool) or not 0<float(top_p)<=1: raise ValueError("top_p must be numeric, > 0 and <= 1")
        if not isinstance(seed,int) or isinstance(seed,bool): raise ValueError("seed must be an integer")
        so=body.get("stream_options")
        if so is not None:
            if not isinstance(so,dict): raise ValueError("stream_options must be an object")
            if set(so)-{"include_usage"}: raise ValueError("unsupported stream_options field")
            if "include_usage" in so and not isinstance(so["include_usage"],bool): raise ValueError("stream_options.include_usage must be a boolean")
        sampling=SamplingConfig(greedy=True) if float(temperature)==0 else SamplingConfig(greedy=False,temperature=float(temperature),top_k=40,top_p=float(top_p),min_p=0.05,seed=seed)
        return messages,max_tokens,sampling

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

                # Phase 3C.4: llama.cpp common_chat parses partial output.
                pieces=[]
                sent_reasoning=""
                sent_content=""
                completion_tokens=0
                gen=backend.stream_sampled(max_tokens=max_tokens,sampling=sampling)
                try:
                    for _, piece in gen:
                        completion_tokens+=1
                        pieces.append(piece)
                        parsed_partial=backend._bridge.chat_parse_output(
                            backend._model, "".join(pieces),
                            enable_thinking=enable_thinking, is_partial=True)
                        current_reasoning=parsed_partial.get("reasoning_content", "")
                        current_content=parsed_partial.get("content", "")
                        if current_reasoning.startswith(sent_reasoning):
                            delta=current_reasoning[len(sent_reasoning):]
                            if delta:
                                send({"id":cid,"object":"chat.completion.chunk","created":created,
                                      "model":self.model_id,
                                      "choices":[{"index":0,"delta":{"reasoning_content":delta},"finish_reason":None}]})
                                sent_reasoning=current_reasoning
                        if current_content.startswith(sent_content):
                            delta=current_content[len(sent_content):]
                            if delta:
                                send({"id":cid,"object":"chat.completion.chunk","created":created,
                                      "model":self.model_id,
                                      "choices":[{"index":0,"delta":{"content":delta},"finish_reason":None}]})
                                sent_content=current_content
                finally:
                    gen.close()

                # Final parse flushes any suffix withheld while output was partial.
                parsed=backend._bridge.chat_parse_output(
                    backend._model, "".join(pieces),
                    enable_thinking=enable_thinking, is_partial=False)
                final_reasoning=parsed.get("reasoning_content", "")
                final_content=parsed.get("content", "")
                if final_reasoning.startswith(sent_reasoning):
                    delta=final_reasoning[len(sent_reasoning):]
                    if delta:
                        send({"id":cid,"object":"chat.completion.chunk","created":created,
                              "model":self.model_id,
                              "choices":[{"index":0,"delta":{"reasoning_content":delta},"finish_reason":None}]})
                        sent_reasoning=final_reasoning
                if final_content.startswith(sent_content):
                    delta=final_content[len(sent_content):]
                    if delta:
                        send({"id":cid,"object":"chat.completion.chunk","created":created,
                              "model":self.model_id,
                              "choices":[{"index":0,"delta":{"content":delta},"finish_reason":None}]})
                        sent_content=final_content

                finish_reason=backend.last_finish_reason or "length"
                send({"id":cid,"object":"chat.completion.chunk","created":created,
                      "model":self.model_id,
                      "choices":[{"index":0,"delta":{},"finish_reason":finish_reason}]})
                stream_options=body.get("stream_options") or {}
                if stream_options.get("include_usage",False):
                    send({"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,"choices":[],
                          "usage":{"prompt_tokens":len(prompt_ids),"completion_tokens":completion_tokens,
                                   "total_tokens":len(prompt_ids)+completion_tokens}})
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()

    def make_handler(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            server_version="LeanMoE/3D.1"; protocol_version="HTTP/1.1"
            def log_message(self,fmt: str,*args: Any)->None: print("[HTTP] "+(fmt % args))
            def _send(self,status: int,obj: Any)->None:
                payload=_json_bytes(obj); self.send_response(status)
                self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(payload)))
                self.end_headers(); self.wfile.write(payload)
            def _error(self,status: int,message: str,etype: str="invalid_request_error",param=None,code=None)->None:
                self._send(status,{"error":{"message":message,"type":etype,"param":param,"code":code}})
            def _method_not_allowed(self)->None:
                self.send_response(405); self.send_header("Allow","GET, POST")
                payload=_json_bytes({"error":{"message":"Method not allowed","type":"invalid_request_error","param":None,"code":None}})
                self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(payload)))
                self.end_headers(); self.wfile.write(payload)
            def do_GET(self)->None:
                path=self.path.split("?",1)[0]
                if path=="/health": self._send(200,{"status":"ok","phase":"3D.1","model":owner.model_id})
                elif path=="/v1/models": self._send(200,{"object":"list","data":[{"id":owner.model_id,"object":"model","created":0,"owned_by":"leanmoe"}]})
                elif path==f"/v1/models/{owner.model_id}": self._send(200,{"id":owner.model_id,"object":"model","created":0,"owned_by":"leanmoe"})
                elif path.startswith("/v1/models/"): self._error(404,"Model not found",param="model",code="model_not_found")
                else: self._error(404,"Not found",code="not_found")
            def do_POST(self)->None:
                path=self.path.split("?",1)[0]
                if path!="/v1/chat/completions": self._error(404,"Not found",code="not_found"); return
                body={}; started=False
                try:
                    ctype=self.headers.get("Content-Type","").split(";",1)[0].strip().lower()
                    if ctype!="application/json": raise ValueError("Content-Type must be application/json")
                    raw_len=self.headers.get("Content-Length")
                    if raw_len is None: raise ValueError("Content-Length is required")
                    try: length=int(raw_len)
                    except ValueError: raise ValueError("Content-Length must be an integer")
                    if length<=0: raise ValueError("request body must not be empty")
                    if length>1048576: raise ValueError("request body exceeds 1 MiB limit")
                    try: body=json.loads(self.rfile.read(length))
                    except (json.JSONDecodeError,UnicodeDecodeError) as exc: raise ValueError(f"invalid JSON body: {exc}")
                    owner._request_config(body)
                    if body.get("stream",False):
                        self.send_response(200); self.send_header("Content-Type","text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control","no-cache"); self.send_header("Connection","close"); self.end_headers(); started=True
                        try: owner._stream(self,body)
                        except (BrokenPipeError,ConnectionResetError): print("[INFO] streaming client disconnected")
                        finally: self.close_connection=True
                    else: self._send(200,owner._generate(body))
                except ValueError as exc:
                    if started: print("[ERROR] streaming validation:",exc); self.close_connection=True
                    else: self._error(400,str(exc))
                except Exception as exc:
                    if started: print(f"[ERROR] streaming request failed: {type(exc).__name__}: {exc}"); self.close_connection=True
                    else: self._error(500,f"{type(exc).__name__}: {exc}","server_error")
            do_PUT=lambda self:self._method_not_allowed()
            do_DELETE=lambda self:self._method_not_allowed()
            do_PATCH=lambda self:self._method_not_allowed()
        return Handler

    def serve_forever(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self.make_handler())
        print(f"LeanMoE Phase 3D.1 listening on http://{self.host}:{self.port}")
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


