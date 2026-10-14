// include/strata/kernels/f16_dot8.hpp - the correction branch's f16 row product: 8 f16 pairs, each converted
// with one hardware instruction (__half22float2, exact for the widening direction) and the eight products
// accumulated left to right in element order (f32).
//
// lora_corr.cu's A and B passes and the mixed f16/g128 correction kernels share it; the host reference of
// lora_corr_parity writes the same eight `+=` statements, so the shape must stay in step with it.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace strata::kernels {

__device__ __forceinline__ float f16_dot8(const uint4 av, const uint4 xv) {
    const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&av.x));
    const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&av.y));
    const float2 a2 = __half22float2(*reinterpret_cast<const __half2*>(&av.z));
    const float2 a3 = __half22float2(*reinterpret_cast<const __half2*>(&av.w));
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(&xv.x));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&xv.y));
    const float2 x2 = __half22float2(*reinterpret_cast<const __half2*>(&xv.z));
    const float2 x3 = __half22float2(*reinterpret_cast<const __half2*>(&xv.w));
    float s = a0.x * x0.x;
    s += a0.y * x0.y;
    s += a1.x * x1.x;
    s += a1.y * x1.y;
    s += a2.x * x2.x;
    s += a2.y * x2.y;
    s += a3.x * x3.x;
    s += a3.y * x3.y;
    return s;
}

}  // namespace strata::kernels
