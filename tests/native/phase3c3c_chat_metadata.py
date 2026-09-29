from __future__ import annotations
import argparse,json
from pathlib import Path
from leanmoe.native import NativeBridge,ModelConfig
def main():
    a=argparse.ArgumentParser();a.add_argument("runtime",type=Path);a.add_argument("model",type=Path);x=a.parse_args()
    b=NativeBridge(x.runtime);b.load();assert b.api_version()==7;b.init();m=None
    try:
        cfg=ModelConfig();cfg.n_gpu_layers=41;cfg.n_cpu_moe=30;cfg.use_mmap=1;cfg.use_mlock=0
        m=b.model_load(x.model,cfg)
        on=b.chat_template_metadata(m,enable_thinking=True);off=b.chat_template_metadata(m,enable_thinking=False)
        keys={"supports_thinking","thinking_start_tag","thinking_end_tags","additional_stops","preserved_tokens","parser","generation_prompt"}
        assert keys<=set(on),on
        print("[PASS] Bridge API v7")
        print("[PASS] model-derived chat metadata")
        print(json.dumps({"thinking_on":on,"thinking_off":off},ensure_ascii=False,indent=2))
    finally:
        if m:b.model_free(m)
        b.shutdown();b.close()
if __name__=="__main__":main()
