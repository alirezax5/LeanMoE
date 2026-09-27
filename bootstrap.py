import ctypes
import os
import sys

LLAMA_DIR = r"C:\llama"
LLAMA_DLL = os.path.join(LLAMA_DIR, "llama.dll")

print("=== LeanMoE Bootstrap Test ===")
print(f"Python: {sys.version.split()[0]}")
print(f"Runtime: {LLAMA_DIR}")

# Keep the directory handle alive for the lifetime of the process.
dll_dir = os.add_dll_directory(LLAMA_DIR)

try:
    llama = ctypes.CDLL(LLAMA_DLL)
    print("[PASS] llama.dll loaded")

    # void llama_backend_init(void);
    llama.llama_backend_init.argtypes = []
    llama.llama_backend_init.restype = None

    # void llama_backend_free(void);
    llama.llama_backend_free.argtypes = []
    llama.llama_backend_free.restype = None

    print("[INFO] Initializing llama backend...")
    llama.llama_backend_init()

    print("[PASS] llama_backend_init()")

    print("[INFO] Shutting down llama backend...")
    llama.llama_backend_free()

    print("[PASS] llama_backend_free()")
    print()
    print("=== BOOTSTRAP PASS ===")

except Exception as exc:
    print()
    print("=== BOOTSTRAP FAIL ===")
    print(f"{type(exc).__name__}: {exc}")
    raise
finally:
    dll_dir.close()