// tools/v41/cmake/ggml_prebuilt_stubs.cpp - see ggml_prebuilt.cmake.
//
// A prebuilt libggml.a carries a backend registry compiled with every backend that llama.cpp build was configured
// for, and that registry references each backend's registration function whether or not this engine links it.  This
// machine's build has GGML_VULKAN=ON next to GGML_HIP=ON, so the linker asks for ggml_backend_vk_reg even though the
// engine links only the CPU and HIP backends.  Returning NULL is exactly what ggml's registry does with a backend it
// was not built with (ggml-backend-reg.cpp's register_backend() ignores a NULL registration), so the symbol is
// satisfied and Vulkan stays out of both the link and the runtime device list.
//
// If another prebuilt build references a backend this one does not, the linker names the symbol: add it here.
#include "ggml-backend.h"

extern "C" ggml_backend_reg_t ggml_backend_vk_reg(void) { return nullptr; }
