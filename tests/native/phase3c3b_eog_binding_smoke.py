from __future__ import annotations
import argparse, json
from pathlib import Path
from leanmoe.native import NativeBridge, ModelConfig

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('runtime',type=Path); ap.add_argument('model',type=Path); ap.add_argument('--report',type=Path,default=Path('reports/phase3c3b-eog-binding.json')); a=ap.parse_args()
    report={'phase':'3C.3B','test':'eog-binding','status':'RUNNING'}; b=NativeBridge(a.runtime); model=None
    try:
        b.load(); assert b.api_version()==6, b.api_version(); b.init()
        model=b.model_load(a.model,ModelConfig(n_gpu_layers=41,n_cpu_moe=30,use_mmap=1,use_mlock=0))
        ids=b.tokenize(model,'Hello',add_special=False,parse_special=False)
        assert ids
        ordinary=bool(b.token_is_eog(model,ids[-1]))
        report.update(status='PASS',ordinary_token=ids[-1],ordinary_is_eog=ordinary)
        print('[PASS] Bridge API v6')
        print('[PASS] lm_token_is_eog binding callable')
        print('[INFO] ordinary token is EOG:',ordinary)
        return 0
    except Exception as e:
        report.update(status='FAIL',error=f'{type(e).__name__}: {e}'); raise
    finally:
        if model is not None: b.model_free(model)
        try:b.shutdown()
        except Exception:pass
        b.close(); p=a.report.resolve(); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2),encoding='utf-8'); print('[INFO] report:',p)
if __name__=='__main__': raise SystemExit(main())
