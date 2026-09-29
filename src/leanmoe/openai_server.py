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


def _stop_list(body: dict[str, Any]) -> list[str]:
    stop=body.get("stop")
    if stop is None:
        return []
    return [stop] if isinstance(stop,str) else list(stop)

def _cut_at_stop(text: str, stops: list[str]) -> tuple[str,bool]:
    hit=None
    for stop in stops:
        pos=text.find(stop)
        if pos >= 0 and (hit is None or pos < hit):
            hit=pos
    return (text,False) if hit is None else (text[:hit],True)

def _safe_prefix(text: str, stops: list[str]) -> str:
    if not stops:
        return text
    keep=max((len(x) for x in stops),default=1)-1
    return text if keep <= 0 else (text[:-keep] if len(text)>keep else "")

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
        if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<=0xFFFFFFFF: raise ValueError("seed must be an integer in [0, 2^32-1]")
        frequency_penalty=body.get("frequency_penalty",0.0); presence_penalty=body.get("presence_penalty",0.0)
        if not isinstance(frequency_penalty,(int,float)) or isinstance(frequency_penalty,bool) or not -2<=float(frequency_penalty)<=2: raise ValueError("frequency_penalty must be numeric between -2 and 2")
        if not isinstance(presence_penalty,(int,float)) or isinstance(presence_penalty,bool) or not -2<=float(presence_penalty)<=2: raise ValueError("presence_penalty must be numeric between -2 and 2")
        stop=body.get("stop")
        if stop is not None:
            if isinstance(stop,str):
                stops=[stop]
            elif isinstance(stop,list) and 1 <= len(stop) <= 4 and all(isinstance(x,str) for x in stop):
                stops=stop
            else:
                raise ValueError("stop must be a string or an array of 1 to 4 strings")
            if any(x=="" for x in stops):
                raise ValueError("stop strings must not be empty")
        so=body.get("stream_options")
        if so is not None:
            if not isinstance(so,dict): raise ValueError("stream_options must be an object")
            if set(so)-{"include_usage"}: raise ValueError("unsupported stream_options field")
            if "include_usage" in so and not isinstance(so["include_usage"],bool): raise ValueError("stream_options.include_usage must be a boolean")
        sampling=SamplingConfig(greedy=float(temperature)==0,temperature=float(temperature) if float(temperature)>0 else 0.8,top_k=40,top_p=float(top_p),min_p=0.05,seed=seed,penalty_last_n=64,repeat_penalty=1.0,frequency_penalty=float(frequency_penalty),presence_penalty=float(presence_penalty))
        return messages,max_tokens,sampling

    def _generate(self, body: dict[str, Any]) -> dict[str, Any]:
        messages,max_tokens,sampling=self._request_config(body)
        enable_thinking=bool(body.get("enable_thinking",True))
        stops=_stop_list(body)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("inference_busy")
        try:
            with LeanMoEBackend(self.runtime_dir,self.model_path,self.backend_config) as backend:
                prompt=backend._bridge.chat_apply_template(
                    backend._model,[(m["role"],m["content"]) for m in messages],
                    add_generation_prompt=True,enable_thinking=enable_thinking)
                prompt_ids=backend.tokenize_chat_prompt(prompt)
                if not prompt_ids:
                    raise ValueError("Prompt tokenized to zero tokens")
                if len(prompt_ids)+max_tokens>backend.config.n_ctx:
                    raise ValueError(f"Prompt + generation exceeds context: {len(prompt_ids)} + {max_tokens} > {backend.config.n_ctx}")
                backend.prefill_tokens(prompt_ids)
                pieces=[]; generated_tokens=0; stop_hit=False
                gen=backend.stream_sampled(max_tokens=max_tokens,sampling=sampling)
                try:
                    for _,piece in gen:
                        generated_tokens+=1
                        pieces.append(piece)
                        _,stop_hit=_cut_at_stop("".join(pieces),stops)
                        if stop_hit:
                            break
                finally:
                    gen.close()
                raw,stop_hit=_cut_at_stop("".join(pieces),stops)
                parsed=backend._bridge.chat_parse_output(
                    backend._model,raw,enable_thinking=enable_thinking,is_partial=False)
                finish_reason="stop" if stop_hit else (backend.last_finish_reason or "length")
        finally:
            self._lock.release()
        message={"role":"assistant","content":parsed.get("content","")}
        reasoning=parsed.get("reasoning_content","")
        if reasoning:
            message["reasoning_content"]=reasoning.rstrip()
        return {"id":"chatcmpl-"+uuid.uuid4().hex,"object":"chat.completion","created":int(time.time()),
                "model":self.model_id,"choices":[{"index":0,"message":message,"finish_reason":finish_reason}],
                "usage":{"prompt_tokens":len(prompt_ids),"completion_tokens":generated_tokens,
                         "total_tokens":len(prompt_ids)+generated_tokens}}

    def _stream(self, handler: BaseHTTPRequestHandler, body: dict[str, Any], *, lock_held: bool = False) -> None:
        messages,max_tokens,sampling=self._request_config(body)
        enable_thinking=bool(body.get("enable_thinking",True))
        stops=_stop_list(body)
        cid="chatcmpl-"+uuid.uuid4().hex
        created=int(time.time())

        def send(obj: Any)->None:
            payload="data: "+json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n\n"
            handler.wfile.write(payload.encode("utf-8"))
            handler.wfile.flush()

        acquired_here=False
        if not lock_held:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError("inference_busy")
            acquired_here=True
        try:
            with LeanMoEBackend(self.runtime_dir,self.model_path,self.backend_config) as backend:
                prompt=backend._bridge.chat_apply_template(
                    backend._model,[(m["role"],m["content"]) for m in messages],
                    add_generation_prompt=True,enable_thinking=enable_thinking)
                prompt_ids=backend.tokenize_chat_prompt(prompt)
                if not prompt_ids:
                    raise ValueError("Prompt tokenized to zero tokens")
                if len(prompt_ids)+max_tokens>backend.config.n_ctx:
                    raise ValueError(f"Prompt + generation exceeds context: {len(prompt_ids)} + {max_tokens} > {backend.config.n_ctx}")
                backend.prefill_tokens(prompt_ids)

                send({"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,
                      "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})

                raw=""; parsed_raw=""; sent_reasoning=""; sent_content=""
                completion_tokens=0; stop_hit=False
                gen=backend.stream_sampled(max_tokens=max_tokens,sampling=sampling)
                try:
                    for _,piece in gen:
                        completion_tokens+=1
                        raw+=piece
                        cut,stop_hit=_cut_at_stop(raw,stops)
                        safe=cut if stop_hit else _safe_prefix(cut,stops)
                        if safe!=parsed_raw or stop_hit:
                            parsed_raw=safe
                            pp=backend._bridge.chat_parse_output(
                                backend._model,safe,enable_thinking=enable_thinking,is_partial=not stop_hit)
                            cr=pp.get("reasoning_content","")
                            cc=pp.get("content","")
                            if cr.startswith(sent_reasoning):
                                delta=cr[len(sent_reasoning):]
                                if delta:
                                    send({"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,
                                          "choices":[{"index":0,"delta":{"reasoning_content":delta},"finish_reason":None}]})
                                    sent_reasoning=cr
                            if cc.startswith(sent_content):
                                delta=cc[len(sent_content):]
                                if delta:
                                    send({"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,
                                          "choices":[{"index":0,"delta":{"content":delta},"finish_reason":None}]})
                                    sent_content=cc
                        if stop_hit:
                            break
                finally:
                    gen.close()

                final_raw,_=_cut_at_stop(raw,stops)
                parsed=backend._bridge.chat_parse_output(
                    backend._model,final_raw,enable_thinking=enable_thinking,is_partial=False)
                fr=parsed.get("reasoning_content","")
                fc=parsed.get("content","")
                if fr.startswith(sent_reasoning):
                    delta=fr[len(sent_reasoning):]
                    if delta:
                        send({"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,
                              "choices":[{"index":0,"delta":{"reasoning_content":delta},"finish_reason":None}]})
                if fc.startswith(sent_content):
                    delta=fc[len(sent_content):]
                    if delta:
                        send({"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,
                              "choices":[{"index":0,"delta":{"content":delta},"finish_reason":None}]})

                finish_reason="stop" if stop_hit else (backend.last_finish_reason or "length")
                send({"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,
                      "choices":[{"index":0,"delta":{},"finish_reason":finish_reason}]})
                if (body.get("stream_options") or {}).get("include_usage",False):
                    send({"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,"choices":[],
                          "usage":{"prompt_tokens":len(prompt_ids),"completion_tokens":completion_tokens,
                                   "total_tokens":len(prompt_ids)+completion_tokens}})
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
        finally:
            if acquired_here:
                self._lock.release()

    def make_handler(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            server_version="LeanMoE/3D.3B"; protocol_version="HTTP/1.1"
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
                if path=="/health": self._send(200,{"status":"ok","phase":"3D.3B","model":owner.model_id})
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
                        if not owner._lock.acquire(blocking=False):
                            raise RuntimeError("inference_busy")
                        try:
                            self.send_response(200); self.send_header("Content-Type","text/event-stream; charset=utf-8")
                            self.send_header("Cache-Control","no-cache"); self.send_header("Connection","close"); self.end_headers(); started=True
                            try: owner._stream(self,body,lock_held=True)
                            except (BrokenPipeError,ConnectionResetError): print("[INFO] streaming client disconnected")
                            finally: self.close_connection=True
                        finally:
                            owner._lock.release()
                    else: self._send(200,owner._generate(body))
                except ValueError as exc:
                    if started: print("[ERROR] streaming validation:",exc); self.close_connection=True
                    else: self._error(400,str(exc))
                except RuntimeError as exc:
                    if str(exc)=="inference_busy" and not started:
                        self._error(429,"LeanMoE is already processing another inference request",
                                    "rate_limit_error",code="inference_busy")
                    elif started:
                        print(f"[ERROR] streaming request failed: {type(exc).__name__}: {exc}")
                        self.close_connection=True
                    else:
                        self._error(500,f"{type(exc).__name__}: {exc}","server_error")
                except Exception as exc:
                    if started:
                        print(f"[ERROR] streaming request failed: {type(exc).__name__}: {exc}")
                        self.close_connection=True
                    else:
                        self._error(500,f"{type(exc).__name__}: {exc}","server_error")
            do_PUT=lambda self:self._method_not_allowed()
            do_DELETE=lambda self:self._method_not_allowed()
            do_PATCH=lambda self:self._method_not_allowed()
        return Handler

    def serve_forever(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self.make_handler())
        print(f"LeanMoE Phase 3D.3B listening on http://{self.host}:{self.port}")
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


