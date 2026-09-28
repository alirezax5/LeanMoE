# Phase 3B CI v2

Drop this ZIP into the repository root.

This revision fixes the Windows toolchain problem seen in the first Phase 3B CI:
CMake selected MinGW/GNU instead of MSVC.

v2 explicitly discovers Visual Studio with vswhere and enters VsDevCmd for:
- CMake configure
- bridge build
- API compile gate
- dumpbin export verification

It keeps the existing pinned llama.cpp commit and CUDA 13.4 setup.
