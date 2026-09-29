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


def _validate_tool_calls(calls: Any, where: str) -> None:
    if not isinstance(calls, list) or not calls: raise ValueError(f"{where} must be a non-empty array")
    for i, c in enumerate(calls):
        if not isinstance(c, dict) or c.get("type") != "function": raise ValueError(f"{where}[{i}] must be a function tool call")
        if not isinstance(c.get("id"), str) or not c["id"]: raise ValueError(f"{where}[{i}].id is required")
        f=c.get("function")
        if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]: raise ValueError(f"{where}[{i}].function.name is required")
        if not isinstance(f.get("arguments"), str): raise ValueError(f"{where}[{i}].function.arguments must be a JSON string")

def _validate_tools(tools: Any) -> None:
    if tools is None: return
    if not isinstance(tools, list) or not tools: raise ValueError("tools must be a non-empty array")
    for i,t in enumerate(tools):
        if not isinstance(t,dict) or t.get("type")!="function": raise ValueError(f"tools[{i}] must be a function tool")
        f=t.get("function")
        if not isinstance(f,dict) or not isinstance(f.get("name"),str) or not f["name"]: raise ValueError(f"tools[{i}].function.name is required")
        if "parameters" in f and not isinstance(f["parameters"],dict): raise ValueError(f"tools[{i}].function.parameters must be an object")

def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    if not isinstance(messages,list) or not messages: raise ValueError("messages must be a non-empty array")
    for i,m in enumerate(messages):
        if not isinstance(m,dict): raise ValueError(f"messages[{i}] must be an object")
        role=m.get("role"); content=m.get("content")
        if role not in {"system","developer","user","assistant","tool"}: raise ValueError(f"unsupported role at messages[{i}]: {role!r}")
        if role=="assistant":
            if content is not None and not isinstance(content,str): raise ValueError(f"messages[{i}].content must be a string or null")
            if m.get("tool_calls") is not None: _validate_tool_calls(m["tool_calls"],f"messages[{i}].tool_calls")
            if content is None and not m.get("tool_calls"): raise ValueError(f"messages[{i}] needs content or tool_calls")
        elif not isinstance(content,str): raise ValueError(f"messages[{i}].content must be a string")
        if role=="tool" and (not isinstance(m.get("tool_call_id"),str) or not m["tool_call_id"]): raise ValueError(f"messages[{i}].tool_call_id is required")
    return ""


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
        tools=body.get("tools"); _validate_tools(tools)
        tool_choice=body.get("tool_choice","auto")
        if not isinstance(tool_choice,str) or tool_choice not in {"auto","none","required"}: raise ValueError("tool_choice must be auto, none, or required")
        if tools is None and tool_choice!="auto": raise ValueError("tool_choice requires tools")
        parallel_tool_calls=body.get("parallel_tool_calls",True)
        if not isinstance(parallel_tool_calls,bool): raise ValueError("parallel_tool_calls must be a boolean")
        stream=body.get("stream",False)
        if not isinstance(stream,bool): raise ValueError("stream must be a boolean")
        if "n" in body and (not isinstance(body["n"],int) or isinstance(body["n"],bool) or body["n"]!=1): raise ValueError("n must be 1")
        if body.get("logprobs") not in (None,False) or "top_logprobs" in body: raise ValueError("logprobs are not supported")
        if "response_format" in body and body["response_format"]!={"type":"text"}: raise ValueError("only response_format.type='text' is supported")
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
                tools=body.get("tools")
                structured=tools is not None or any(m.get("role") in {"developer","tool"} or m.get("tool_calls") for m in messages)
                if structured:
                    meta=backend._bridge.chat_apply_structured(backend._model,messages,tools,tool_choice=body.get("tool_choice","auto"),add_generation_prompt=True,enable_thinking=enable_thinking,parallel_tool_calls=body.get("parallel_tool_calls",True))
                    prompt=meta.get("prompt")
                    if not isinstance(prompt,str) or not prompt: raise RuntimeError("structured renderer returned no prompt")
                else:
                    prompt=backend._bridge.chat_apply_template(backend._model,[(m["role"],m["content"]) for m in messages],add_generation_prompt=True,enable_thinking=enable_thinking)
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
                if structured:
                    parsed=backend._bridge.chat_parse_output_structured(backend._model,messages,raw,tools,tool_choice=body.get("tool_choice","auto"),enable_thinking=enable_thinking,parallel_tool_calls=body.get("parallel_tool_calls",True),is_partial=False)
                else:
                    parsed=backend._bridge.chat_parse_output(backend._model,raw,enable_thinking=enable_thinking,is_partial=False)
                finish_reason="stop" if stop_hit else (backend.last_finish_reason or "length")
        finally:
            self._lock.release()
        message={"role":"assistant","content":parsed.get("content","")}
        reasoning=parsed.get("reasoning_content","")
        if reasoning:
            message["reasoning_content"]=reasoning.rstrip()
        if parsed.get("tool_calls"):
            normalized_calls=[]
            for call in parsed["tool_calls"]:
                normalized=dict(call)
                call_id=normalized.get("id")
                if not isinstance(call_id,str) or not call_id:
                    normalized["id"]="call_"+uuid.uuid4().hex
                normalized_calls.append(normalized)
            message["tool_calls"]=normalized_calls
            if not message["content"]: message["content"]=None
            finish_reason="tool_calls"
        return {"id":"chatcmpl-"+uuid.uuid4().hex,"object":"chat.completion","created":int(time.time()),
                "model":self.model_id,"choices":[{"index":0,"message":message,"finish_reason":finish_reason}],
                "usage":{"prompt_tokens":len(prompt_ids),"completion_tokens":generated_tokens,
                         "total_tokens":len(prompt_ids)+generated_tokens}}

    def _stream(self, handler: BaseHTTPRequestHandler, body: dict[str, Any], *, lock_held: bool = False) -> None:
        messages,max_tokens,sampling=self._request_config(body)
        enable_thinking=bool(body.get("enable_thinking",True))
        stops=_stop_list(body)
        tools=body.get("tools")
        tool_choice=body.get("tool_choice","auto")
        parallel_tool_calls=body.get("parallel_tool_calls",True)
        structured=tools is not None or any(m.get("role") in {"developer","tool"} or m.get("tool_calls") for m in messages)
        cid="chatcmpl-"+uuid.uuid4().hex
        created=int(time.time())

        def send(obj: Any)->None:
            payload="data: "+json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n\n"
            handler.wfile.write(payload.encode("utf-8")); handler.wfile.flush()

        def chunk(delta: dict[str,Any], finish_reason=None)->dict[str,Any]:
            return {"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,
                    "choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]}

        acquired_here=False
        if not lock_held:
            if not self._lock.acquire(blocking=False): raise RuntimeError("inference_busy")
            acquired_here=True
        try:
            with LeanMoEBackend(self.runtime_dir,self.model_path,self.backend_config) as backend:
                if structured:
                    meta=backend._bridge.chat_apply_structured(
                        backend._model,messages,tools,tool_choice=tool_choice,add_generation_prompt=True,
                        enable_thinking=enable_thinking,parallel_tool_calls=parallel_tool_calls)
                    prompt=meta.get("prompt")
                    if not isinstance(prompt,str) or not prompt: raise RuntimeError("structured renderer returned no prompt")
                else:
                    prompt=backend._bridge.chat_apply_template(
                        backend._model,[(m["role"],m["content"]) for m in messages],
                        add_generation_prompt=True,enable_thinking=enable_thinking)
                prompt_ids=backend.tokenize_chat_prompt(prompt)
                if not prompt_ids: raise ValueError("Prompt tokenized to zero tokens")
                if len(prompt_ids)+max_tokens>backend.config.n_ctx:
                    raise ValueError(f"Prompt + generation exceeds context: {len(prompt_ids)} + {max_tokens} > {backend.config.n_ctx}")
                backend.prefill_tokens(prompt_ids)
                send(chunk({"role":"assistant"}))

                raw=""; parsed_raw=""; sent_reasoning=""; sent_content=""
                completion_tokens=0; stop_hit=False
                tool_ids=[]; sent_tool_names=[]; sent_tool_args=[]; revision_fault=False

                def emit_parsed(pp: dict[str,Any])->None:
                    nonlocal sent_reasoning,sent_content,revision_fault
                    cr=pp.get("reasoning_content","") or ""; cc=pp.get("content","") or ""
                    if cr.startswith(sent_reasoning):
                        d=cr[len(sent_reasoning):]
                        if d: send(chunk({"reasoning_content":d})); sent_reasoning=cr
                    elif cr!=sent_reasoning: revision_fault=True
                    if cc.startswith(sent_content):
                        d=cc[len(sent_content):]
                        if d: send(chunk({"content":d})); sent_content=cc
                    elif cc!=sent_content: revision_fault=True
                    for i,call in enumerate(pp.get("tool_calls") or []):
                        if not isinstance(call,dict): continue
                        fn=call.get("function") or {}; name=fn.get("name","") or ""; args=fn.get("arguments","") or ""
                        while len(tool_ids)<=i:
                            nid=call.get("id")
                            tool_ids.append(nid if isinstance(nid,str) and nid else "call_"+uuid.uuid4().hex)
                            sent_tool_names.append(""); sent_tool_args.append("")
                        dc={"index":i}; changed=False
                        if not sent_tool_names[i]:
                            dc.update({"id":tool_ids[i],"type":"function",
                                       "function":{"name":name,"arguments":args}})
                            sent_tool_names[i]=name; sent_tool_args[i]=args; changed=True
                        elif name!=sent_tool_names[i]:
                            revision_fault=True; continue
                        elif args.startswith(sent_tool_args[i]):
                            suffix=args[len(sent_tool_args[i]):]
                            if suffix:
                                dc["function"]={"arguments":suffix}; sent_tool_args[i]=args; changed=True
                        elif args!=sent_tool_args[i]:
                            revision_fault=True; continue
                        if changed: send(chunk({"tool_calls":[dc]}))

                gen=backend.stream_sampled(max_tokens=max_tokens,sampling=sampling)
                try:
                    for _,piece in gen:
                        completion_tokens+=1; raw+=piece
                        cut,stop_hit=_cut_at_stop(raw,stops); safe=cut if stop_hit else _safe_prefix(cut,stops)
                        if safe!=parsed_raw or stop_hit:
                            parsed_raw=safe
                            if structured:
                                pp=backend._bridge.chat_parse_output_structured(
                                    backend._model,messages,safe,tools,tool_choice=tool_choice,
                                    enable_thinking=enable_thinking,parallel_tool_calls=parallel_tool_calls,
                                    is_partial=not stop_hit)
                            else:
                                pp=backend._bridge.chat_parse_output(
                                    backend._model,safe,enable_thinking=enable_thinking,is_partial=not stop_hit)
                            emit_parsed(pp)
                        if stop_hit: break
                finally:
                    gen.close()

                final_raw,_=_cut_at_stop(raw,stops)
                if structured:
                    parsed=backend._bridge.chat_parse_output_structured(
                        backend._model,messages,final_raw,tools,tool_choice=tool_choice,
                        enable_thinking=enable_thinking,parallel_tool_calls=parallel_tool_calls,is_partial=False)
                else:
                    parsed=backend._bridge.chat_parse_output(
                        backend._model,final_raw,enable_thinking=enable_thinking,is_partial=False)
                emit_parsed(parsed)
                if revision_fault: raise RuntimeError("structured partial parser revised already-streamed output")
                final_calls=parsed.get("tool_calls") or []
                finish_reason="tool_calls" if final_calls else ("stop" if stop_hit else (backend.last_finish_reason or "length"))
                send(chunk({},finish_reason))
                if (body.get("stream_options") or {}).get("include_usage",False):
                    send({"id":cid,"object":"chat.completion.chunk","created":created,"model":self.model_id,"choices":[],
                          "usage":{"prompt_tokens":len(prompt_ids),"completion_tokens":completion_tokens,
                                   "total_tokens":len(prompt_ids)+completion_tokens}})
        finally:
            if acquired_here and self._lock.locked():
                self._lock.release()
        if lock_held and self._lock.locked():
            self._lock.release()
        handler.wfile.write(b"data: [DONE]\n\n"); handler.wfile.flush()

    def make_handler(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            server_version="LeanMoE/3D.4C3A-C2"; protocol_version="HTTP/1.1"
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
                if path=="/health": self._send(200,{"status":"ok","phase":"3D.4C3A-C2","model":owner.model_id})
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
                            if owner._lock.locked():
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
        print(f"LeanMoE Phase 3D.4C3A-C2 listening on http://{self.host}:{self.port}")
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


