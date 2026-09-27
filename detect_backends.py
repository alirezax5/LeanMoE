import ctypes
import os
import sys

LLAMA_DIR = r"C:\llama"

print("=== LeanMoE Backend Detection ===")
print(f"Python : {sys.version.split()[0]}")
print(f"Runtime: {LLAMA_DIR}")
print()

dll_dir = os.add_dll_directory(LLAMA_DIR)

try:
    # Load core libraries
    ggml = ctypes.CDLL(os.path.join(LLAMA_DIR, "ggml.dll"))
    llama = ctypes.CDLL(os.path.join(LLAMA_DIR, "llama.dll"))

    print("[PASS] ggml.dll loaded")
    print("[PASS] llama.dll loaded")

    # llama backend
    llama.llama_backend_init.argtypes = []
    llama.llama_backend_init.restype = None

    llama.llama_backend_free.argtypes = []
    llama.llama_backend_free.restype = None

    # GGML backend registry
    ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    ggml.ggml_backend_load_all_from_path.restype = None

    ggml.ggml_backend_dev_count.argtypes = []
    ggml.ggml_backend_dev_count.restype = ctypes.c_size_t

    ggml.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
    ggml.ggml_backend_dev_get.restype = ctypes.c_void_p

    ggml.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
    ggml.ggml_backend_dev_name.restype = ctypes.c_char_p

    ggml.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
    ggml.ggml_backend_dev_description.restype = ctypes.c_char_p

    ggml.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
    ggml.ggml_backend_dev_type.restype = ctypes.c_int

    ggml.ggml_backend_dev_memory.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    ggml.ggml_backend_dev_memory.restype = None

    print()
    print("[INFO] Initializing libllama...")
    llama.llama_backend_init()
    print("[PASS] libllama initialized")

    print("[INFO] Loading GGML backends...")
    ggml.ggml_backend_load_all_from_path(LLAMA_DIR.encode("utf-8"))

    count = ggml.ggml_backend_dev_count()

    print()
    print(f"[INFO] Devices discovered: {count}")
    print()

    type_names = {
        0: "CPU",
        1: "GPU",
        2: "iGPU",
        3: "ACCEL",
        4: "META",
    }

    gpu_found = False

    for i in range(count):
        dev = ggml.ggml_backend_dev_get(i)

        name = ggml.ggml_backend_dev_name(dev)
        desc = ggml.ggml_backend_dev_description(dev)
        dev_type = ggml.ggml_backend_dev_type(dev)

        free_mem = ctypes.c_size_t()
        total_mem = ctypes.c_size_t()

        ggml.ggml_backend_dev_memory(
            dev,
            ctypes.byref(free_mem),
            ctypes.byref(total_mem),
        )

        name = name.decode("utf-8", errors="replace") if name else "?"
        desc = desc.decode("utf-8", errors="replace") if desc else "?"

        free_mb = free_mem.value / (1024 ** 2)
        total_mb = total_mem.value / (1024 ** 2)

        print(f"Device #{i}")
        print(f"  Name        : {name}")
        print(f"  Description : {desc}")
        print(f"  Type        : {type_names.get(dev_type, dev_type)}")
        print(f"  Memory Free : {free_mb:.0f} MB")
        print(f"  Memory Total: {total_mb:.0f} MB")
        print()

        if dev_type == 1:
            gpu_found = True

    if gpu_found:
        print("[PASS] GPU backend detected")
    else:
        print("[FAIL] No GPU backend detected")

    print()
    print("[INFO] Shutting down libllama...")
    llama.llama_backend_free()

    print("[PASS] clean shutdown")
    print()
    print("=== BACKEND DETECTION COMPLETE ===")

finally:
    dll_dir.close()