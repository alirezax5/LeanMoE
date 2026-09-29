from __future__ import annotations
import argparse,json
from pathlib import Path
from leanmoe.native import NativeBridge,ModelConfig

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("runtime",type=Path)
    ap.add_argument("model",type=Path)
    a=ap.parse_args()
    b=NativeBridge(a.runtime); b.load()
    assert b.api_version()==8
    b.init(); m=None
    try:
        cfg=ModelConfig()
        cfg.n_gpu_layers=41; cfg.n_cpu_moe=30
        cfg.use_mmap=1; cfg.use_mlock=0
        m=b.model_load(a.model,cfg)

        on_raw="The user requested exact output.\n</think>\n\nV8_ON_OK"
        on=b.chat_parse_output(m,on_raw,enable_thinking=True,is_partial=False)
        assert on.get("content")=="V8_ON_OK",on
        assert on.get("reasoning_content", "").strip()=="The user requested exact output.",on
        assert "<think>" not in on.get("content",""),on
        assert "</think>" not in on.get("content",""),on
        print("[PASS] thinking ON parsed by llama.cpp common_chat")

        off_raw="V8_OFF_OK"
        off=b.chat_parse_output(m,off_raw,enable_thinking=False,is_partial=False)
        assert off.get("content")=="V8_OFF_OK",off
        assert "<think>" not in off.get("content",""),off
        print("[PASS] thinking OFF parsed by llama.cpp common_chat")

        partial=b.chat_parse_output(m,"partial reasoning",enable_thinking=True,is_partial=True)
        assert isinstance(partial,dict),partial
        print("[PASS] partial parser callable")
        print(json.dumps({"thinking_on":on,"thinking_off":off,"partial":partial},
                         ensure_ascii=False,indent=2))
    finally:
        if m: b.model_free(m)
        b.shutdown(); b.close()

if __name__=="__main__":
    main()

