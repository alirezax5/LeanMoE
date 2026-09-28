# Phase 3B CI v4 — dependency isolation

The native bridge, API compile gate, exported sampler symbols and staging already pass.
The remaining failure is Windows runtime dependency loading.

This revision does not change native code. It:
- dumps dependencies for every staged DLL;
- loads staged DLLs individually from leaves to bridge;
- prints the exact DLL that fails;
- reports resolution paths for MSVC and CUDA runtime DLLs;
- verifies Bridge API v4 only after all native DLLs load.

This deliberately diagnoses the remaining loader problem instead of adding
unverified DLL copies to the artifact.
