// src/kernels/f16_hw_check.cu - do the hardware f16 conversions match the repo's bit-exact software ones?
//
// `f16_bits.hpp` documents that a hardware conversion returned wrong bits once (round 193) and the cause
// was never isolated, so the repo uses software converters everywhere.  The correction kernels spend
// ~75-80% of their time in them, so this check asks the question again, precisely: `__float2half_rn`
// (ROUND-TO-NEAREST, which `__float2half` is NOT - it truncates) and `__half2float` against
// `f16_from_f32`/`f32_from_f16`, bit for bit:
//
//   * widening: every one of the 65,536 fp16 codes, compared as f32 bits (inf/NaN included);
//   * narrowing: a deterministic sweep of ~16M f32 bit patterns over the magnitude range 2^-44..2^44
//     (which covers fp16 subnormals, normals and overflow), plus the boundary values and the specials.
//
// A mismatch prints the input, the two outputs, and exits nonzero.
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {

__device__ __forceinline__ uint32_t f32_bits(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}

__device__ __forceinline__ uint16_t f16_bits_of(__half h) {
    uint16_t b;
    __builtin_memcpy(&b, &h, 2);
    return b;
}

__global__ void widen_kernel(unsigned long long* mism, unsigned long long* nanpay, uint32_t* first_bad, uint32_t* first_hw) {
    const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= 65536u) return;
    const uint16_t bits = (uint16_t) h;
    const float sw = strata::kernels::f32_from_f16(bits);
    __half hh;
    memcpy(&hh, &bits, 2);
    const float hw = __half2float(hh);
    const uint32_t s = f32_bits(sw), w = f32_bits(hw);
    if (s == w) return;
    if (isnan(sw) && isnan(hw)) {   // both NaN: the payload differs, the value does not
        atomicAdd(nanpay, 1ull);
        return;
    }
    atomicAdd(mism, 1ull);
    if (atomicCAS(first_bad, 0xFFFFFFFFu, h) == 0xFFFFFFFFu) *first_hw = w;
}

__device__ __forceinline__ float sweep_value(uint64_t i) {
    // A deterministic spread: sign, mantissa, and an exponent over [-44, 44] - fp16 subnormals,
    // normals, and the overflow boundary are all inside.
    uint32_t u = (uint32_t) (i * 2654435761ull + 0x9E3779B9ull);
    const int e = (int) ((u >> 20) % 89u) - 44;
    const float m = 1.0f + (float) (u & 0x7FFFFFu) / 8388608.0f;
    const float f = ldexpf(m, e);
    return (u & 0x80000000u) ? -f : f;
}

__global__ void narrow_kernel(unsigned long long* mism, uint32_t* first_bad, uint16_t* first_hw, uint64_t n) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float f = sweep_value(i);
    const uint16_t sw = strata::kernels::f16_from_f32(f);
    const uint16_t hw = f16_bits_of(__float2half_rn(f));
    if (sw != hw) {
        atomicAdd(mism, 1ull);
        if (atomicCAS(first_bad, 0xFFFFFFFFu, f32_bits(f)) == 0xFFFFFFFFu) *first_hw = hw;
    }
}

__device__ __forceinline__ uint16_t hw_narrow(float f) { return f16_bits_of(__float2half_rn(f)); }

// The boundary and special values, one thread each.
__global__ void targeted_kernel(const float* in, int n, unsigned long long* mism, unsigned long long* nanpay, int* which, uint16_t* first_hw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float f = in[i];
    const uint16_t sw = strata::kernels::f16_from_f32(f);
    const uint16_t hw = hw_narrow(f);
    if (sw == hw) return;
    if (isnan(f)) {   // NaN -> fp16 NaN: the payload differs, the value does not
        atomicAdd(nanpay, 1ull);
        return;
    }
    atomicAdd(mism, 1ull);
    if (atomicCAS(which, -1, i) == -1) *first_hw = hw;
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

uint32_t hbits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    const uint64_t n = argc > 1 ? strtoull(argv[1], nullptr, 0) : (1ull << 24);
    unsigned long long* dm = nullptr;
    unsigned long long* dnp = nullptr;
    uint32_t* dbad = nullptr;
    uint32_t* dbadhw = nullptr;
    uint16_t* dbadhw16 = nullptr;
    int* dwhich = nullptr;
    cudaMalloc(&dm, 8);
    cudaMalloc(&dnp, 8);
    cudaMalloc(&dbad, 4);
    cudaMalloc(&dbadhw, 4);
    cudaMalloc(&dbadhw16, 2);
    cudaMemset(dm, 0, 8);
    cudaMemset(dnp, 0, 8);
    cudaMemset(dbad, 0xFF, 4);
    check("alloc");

    widen_kernel<<<256, 256>>>(dm, dnp, dbad, dbadhw);
    cudaDeviceSynchronize();
    check("widen");
    unsigned long long m = 0;
    uint32_t bad = 0, badhw = 0;
    cudaMemcpy(&m, dm, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&bad, dbad, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&badhw, dbadhw, 4, cudaMemcpyDeviceToHost);
    if (m) {
        std::printf("widen: %llu of 65536 differ, first 0x%04X: sw %08X hw %08X\n", m, (uint16_t) bad,
                    hbits(strata::kernels::f32_from_f16((uint16_t) bad)), badhw);
        return 1;
    }
    unsigned long long np = 0;
    cudaMemcpy(&np, dnp, 8, cudaMemcpyDeviceToHost);
    std::printf("widen:  65536 of 65536 fp16 codes -> f32 agree (%llu NaN-payload-only differences)\n", np);

    cudaMemset(dm, 0, 8);
    cudaMemset(dbad, 0xFF, 4);
    narrow_kernel<<<(unsigned) ((n + 255) / 256), 256>>>(dm, dbad, dbadhw16, n);
    cudaDeviceSynchronize();
    check("narrow");
    cudaMemcpy(&m, dm, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&bad, dbad, 4, cudaMemcpyDeviceToHost);
    uint16_t bad16 = 0;
    cudaMemcpy(&bad16, dbadhw16, 2, cudaMemcpyDeviceToHost);
    if (m) {
        std::printf("narrow: %llu of %llu differ, first %08X (%.9g): sw %04X hw %04X\n", m, n, bad,
                    (double) from_bits(bad), strata::kernels::f16_from_f32(from_bits(bad)), bad16);
        return 1;
    }
    std::printf("narrow: %llu of %llu sweep values -> fp16 agree\n", n, n);

    const float edge[] = {0.0f, -0.0f, 1.0f, -1.0f,
                          6.103515625e-05f, 6.097555160522461e-05f,   /* 2^-14, just below it */
                          5.960464477539063e-08f, 2.9802322387695312e-08f,
                          1.401298464324817e-45f,                    /* the smallest f32 subnormal */
                          65504.0f, 65519.0f, 65520.0f, 65536.0f, 1e30f, -1e30f, INFINITY, -INFINITY, NAN};
    const int ne = (int) (sizeof edge / sizeof edge[0]);
    float* de = nullptr;
    cudaMalloc(&de, sizeof edge);
    cudaMemcpy(de, edge, sizeof edge, cudaMemcpyHostToDevice);
    cudaMalloc(&dwhich, 4);
    cudaMemset(dwhich, 0xFF, 4);
    cudaMemset(dm, 0, 8);
    targeted_kernel<<<1, 32>>>(de, ne, dm, dnp, dwhich, dbadhw16);
    cudaDeviceSynchronize();
    check("targeted");
    cudaMemcpy(&m, dm, 8, cudaMemcpyDeviceToHost);
    int which = 0;
    cudaMemcpy(&which, dwhich, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&bad16, dbadhw16, 2, cudaMemcpyDeviceToHost);
    if (m) {
        std::printf("targeted: %llu of %d differ, first #%d %08X: sw %04X hw %04X\n", m, ne, which,
                    hbits(edge[which]), strata::kernels::f16_from_f32(edge[which]), bad16);
        return 1;
    }
    std::printf("targeted: %d boundary/special values agree\n", ne);
    std::printf("f16_hw_check: PASS - the hardware conversions are bit-identical to f16_bits.hpp\n");
    return 0;
}
