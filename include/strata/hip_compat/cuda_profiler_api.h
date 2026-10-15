// include/strata/hip_compat/cuda_profiler_api.h - the CUDA one header the DS4.1 expert tier includes that ROCm has
// no counterpart for.  The two calls it makes (cudaProfilerStart / cudaProfilerStop, around a decode profile) map
// straight onto HIP's, which hip_runtime_api.h declares.  Found through the same compat include directory as
// cuda_runtime.h (cmake/hip_backend.cmake puts it BEFORE on the include path).
#pragma once
#include <hip/hip_runtime.h>

#ifndef cudaProfilerStart
#define cudaProfilerStart hipProfilerStart
#endif
#ifndef cudaProfilerStop
#define cudaProfilerStop hipProfilerStop
#endif
