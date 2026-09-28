# LeanMoE Phase 3C.3A — Bridge API v5 / model Jinja chat template

This patch is based on the uploaded current API-v4 `bridge.h`, `bridge.cpp`, and
`native.py`.

## ABI addition

`lm_chat_apply_template()` accepts simple UTF-8 role/content messages and applies
the model's embedded template through llama.cpp `common_chat_templates_*` with
`use_jinja=true`.

This phase deliberately does NOT add tool parsing or reasoning-output parsing.

## Required CMake change

The bridge now includes `chat.h`, so its existing CMake target must include
`${LLAMA_CPP_DIR}/common` and link `llama-common` in addition to `llama`.

Do this against the pinned `cea74625f` checkout only. Do not copy chat sources
from current llama.cpp master into the pinned tree.

## After CI artifact is staged locally

Update the runtime folder with the v5 DLL/dependencies, extract the Python patch,
then:

```powershell
cd C:\LeanMoE
$env:PYTHONPATH = "C:\LeanMoE\src"

python tests\native\phase3c3a_chat_template_smoke.py `
  "C:\LeanMoE\runtime\phase-3c3a-chat-template-v5" `
  "C:\Users\Touresina\.lmstudio\models\peculiar-ragdoll\Tiel-Coder-35B-A3B-GGUF\Tiel-Coder-35B-A3B-UD-IQ4_XS.gguf"
```

Expected:
- Bridge API v5 PASS
- embedded Jinja template PASS
- report at `reports\phase3c3a-chat-template.json`

The test records both `enable_thinking=true` and `false`; they are not required
to differ because that depends on the model template.
