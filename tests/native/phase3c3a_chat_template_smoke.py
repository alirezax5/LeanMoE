from __future__ import annotations
import argparse, json
from pathlib import Path
from leanmoe.native import NativeBridge, ModelConfig

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("runtime",type=Path); ap.add_argument("model",type=Path)
    ap.add_argument("--report",type=Path,default=Path("reports/phase3c3a-chat-template.json"))
    a=ap.parse_args()
    report={"phase":"3C.3A","status":"RUNNING"}
    b=NativeBridge(a.runtime)
    model=None
    try:
        b.load()
        if b.api_version()!=5: raise RuntimeError("Bridge API is not v5")
        b.init()
        model=b.model_load(a.model,ModelConfig(n_gpu_layers=41,n_cpu_moe=30,use_mmap=1,use_mlock=0))
        msgs=[("system","You are a concise coding assistant."),
              ("user","Reply with exactly: TEMPLATE_OK")]
        on=b.chat_apply_template(model,msgs,add_generation_prompt=True,enable_thinking=True)
        off=b.chat_apply_template(model,msgs,add_generation_prompt=True,enable_thinking=False)
        if "TEMPLATE_OK" not in on or "TEMPLATE_OK" not in off:
            raise RuntimeError("formatted prompt lost user content")
        if on == "<|system|>\nYou are a concise coding assistant.\n<|user|>\nReply with exactly: TEMPLATE_OK\n<|assistant|>\n":
            raise RuntimeError("output still matches LeanMoE's old manual serializer")
        report.update({"status":"PASS","thinking_on":on,"thinking_off":off,
                       "different_by_thinking_flag":on!=off})
        print("[PASS] Bridge API v5")
        print("[PASS] model-embedded Jinja chat template applied")
        print("[INFO] enable_thinking changes prompt:", on!=off)
        print("\n=== THINKING ON ===\n"+on)
        print("\n=== THINKING OFF ===\n"+off)
        return 0
    except Exception as e:
        report.update({"status":"FAIL","error":f"{type(e).__name__}: {e}"})
        raise
    finally:
        if model is not None:
            b.model_free(model)
        try:b.shutdown()
        except Exception:pass
        b.close()
        p=a.report.resolve();p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
        print(f"\n[INFO] report: {p}")
if __name__=="__main__": raise SystemExit(main())
