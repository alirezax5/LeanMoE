فایل زیر را باز کن:

notepad C:\\LeanMoE\\docs\\BRIDGE\_API\_V1.md



و این محتوا را داخلش بگذار:

\# LeanMoE Bridge API v1



Status: Draft / Phase 1A

Platform: Windows x86\_64

Python: 3.12

Initial backend: llama.cpp / libllama

Initial tested build: llama.cpp 0.5.0-dev build 11213, commit cea74625f



\## Goals



The native bridge isolates Python from the unstable/internal ABI of llama.cpp.



Python must never construct or depend on:



\- llama\_model\_params

\- llama\_context\_params

\- llama\_batch

\- llama\_sampler

\- GGML internal structs



All llama.cpp structures remain inside the native bridge.



\## ABI



The public LeanMoE bridge ABI must be:



\- C ABI

\- Windows x86\_64

\- opaque handles

\- fixed-width integer types

\- explicit return codes

\- no C++ objects across the ABI

\- no exceptions across the ABI

\- UTF-8 strings



\## Handles



```c

typedef void \* lm\_model\_t;

typedef void \* lm\_context\_t;



NULL represents an invalid handle.

Result Codes

typedef enum lm\_result {

&#x20;   LM\_OK = 0,



&#x20;   LM\_ERROR\_INVALID\_ARGUMENT = -1,

&#x20;   LM\_ERROR\_NOT\_INITIALIZED  = -2,

&#x20;   LM\_ERROR\_LOAD\_MODEL       = -3,

&#x20;   LM\_ERROR\_CREATE\_CONTEXT   = -4,

&#x20;   LM\_ERROR\_TOKENIZE         = -5,

&#x20;   LM\_ERROR\_DECODE           = -6,

&#x20;   LM\_ERROR\_OUT\_OF\_MEMORY    = -7,

&#x20;   LM\_ERROR\_BACKEND          = -8,

&#x20;   LM\_ERROR\_INTERNAL         = -100

} lm\_result;



Runtime API

uint32\_t lm\_api\_version(void);



lm\_result lm\_init(void);



void lm\_shutdown(void);



const char \* lm\_last\_error(void);



lm\_api\_version() returns 1 for Bridge API v1.

lm\_last\_error() returns a thread-local UTF-8 diagnostic string owned by the bridge.

Python must copy the returned string immediately.

Model Configuration

typedef struct lm\_model\_config {

&#x20;   int32\_t n\_gpu\_layers;



&#x20;   // -1 = backend/default behavior

&#x20;   int32\_t n\_cpu\_moe;



&#x20;   uint8\_t use\_mmap;

&#x20;   uint8\_t use\_mlock;



&#x20;   uint8\_t reserved\[6];

} lm\_model\_config;



The public structure belongs to LeanMoE, not llama.cpp.

The bridge translates this structure to the appropriate llama.cpp parameters.

Model API

lm\_result lm\_model\_load(

&#x20;   const char \* path\_utf8,

&#x20;   const lm\_model\_config \* config,

&#x20;   lm\_model\_t \* out\_model

);



void lm\_model\_free(

&#x20;   lm\_model\_t model

);



Model Metadata

uint64\_t lm\_model\_size(

&#x20;   lm\_model\_t model

);



uint64\_t lm\_model\_n\_params(

&#x20;   lm\_model\_t model

);



lm\_result lm\_model\_description(

&#x20;   lm\_model\_t model,

&#x20;   char \* buffer,

&#x20;   size\_t buffer\_size

);



lm\_result lm\_model\_architecture(

&#x20;   lm\_model\_t model,

&#x20;   char \* buffer,

&#x20;   size\_t buffer\_size

);



Metadata functions must not expose pointers to internal llama.cpp objects.

Context Configuration

typedef struct lm\_context\_config {

&#x20;   uint32\_t n\_ctx;

&#x20;   uint32\_t n\_batch;

&#x20;   uint32\_t n\_ubatch;



&#x20;   int32\_t n\_threads;

&#x20;   int32\_t n\_threads\_batch;



&#x20;   int32\_t cache\_type\_k;

&#x20;   int32\_t cache\_type\_v;



&#x20;   uint8\_t flash\_attention;



&#x20;   uint8\_t reserved\[7];

} lm\_context\_config;



Exact LeanMoE cache type constants will be defined separately and translated by the bridge.

Python must not use GGML enum values directly.

Context API

lm\_result lm\_context\_create(

&#x20;   lm\_model\_t model,

&#x20;   const lm\_context\_config \* config,

&#x20;   lm\_context\_t \* out\_context

);



void lm\_context\_free(

&#x20;   lm\_context\_t context

);



uint32\_t lm\_context\_size(

&#x20;   lm\_context\_t context

);



Tokenization

Token IDs exposed by LeanMoE are signed 32-bit integers.

lm\_result lm\_tokenize(

&#x20;   lm\_model\_t model,

&#x20;   const char \* text\_utf8,

&#x20;   uint8\_t add\_special,

&#x20;   uint8\_t parse\_special,

&#x20;   int32\_t \* tokens,

&#x20;   int32\_t capacity,

&#x20;   int32\_t \* out\_count

);



If capacity is insufficient, the bridge returns a defined error and reports the required token count.

Decode

Phase 1 uses a deliberately minimal decode API.

lm\_result lm\_decode\_tokens(

&#x20;   lm\_context\_t context,

&#x20;   const int32\_t \* tokens,

&#x20;   int32\_t token\_count,

&#x20;   int64\_t start\_position

);



The bridge owns creation/destruction of llama\_batch.

Python never creates llama\_batch.

KV Cache

lm\_result lm\_kv\_clear(

&#x20;   lm\_context\_t context

);



lm\_result lm\_kv\_remove(

&#x20;   lm\_context\_t context,

&#x20;   int64\_t position\_start,

&#x20;   int64\_t position\_end

);



Context reuse functionality will be expanded only after basic inference is stable.

Explicitly Out of Scope for Initial Phase 1

\- HTTP server

\- OpenAI API

\- Hermes integration

\- multi-user scheduling

\- parallel slots

\- vision/mmproj

\- embeddings

\- model routing

\- subprocess workers

\- context checkpoints

\- prompt-cache persistence

\- speculative decoding

\- MTP

\- custom CUDA kernels

\- ONNX Runtime

\- hot expert cache

Safety Rules

1\. Python must not directly access llama.cpp structs.

2\. Every native handle must have one owner.

3\. Free operations must tolerate NULL.

4\. Native exceptions must never cross the C ABI.

5\. Every failure must set lm\_last\_error().

6\. Model/context destruction must be deterministic.

7\. The bridge must never spawn llama-server.

8\. Phase 1 must use exactly one process.

9\. No background worker process is allowed.

10\. No optimization is accepted without benchmark evidence.

Phase 1B Acceptance Gate

The first bridge implementation only needs to:

1\. Load the bridge DLL.

2\. Report API version 1.

3\. Initialize libllama.

4\. Load Tiel-Coder-35B-A3B UD-IQ4\_XS.

5\. Return model metadata.

6\. Free the model.

7\. Shut down cleanly.

PASS requires:

\- no crash

\- no child llama-server process

\- no unexplained child process

\- no CUDA error

\- deterministic model unload

\- deterministic runtime shutdown

Context creation and inference are NOT part of Phase 1B.

