// PQ2_0 GEMV bandwidth micro-benchmark: one big matrix, decode-shaped (ncols=1), timed on the GPU.
// Compares the in-tree kernel against two alternative thread mappings.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

struct __attribute__((aligned(2))) WB {   // PQ2_0 block
    __half d;
    unsigned char qs[32];
};
struct __attribute__((aligned(4))) XB {   // q8_1 block
    __half2 ds;
    signed char qs[32];
};
static_assert(sizeof(WB) == 34 && sizeof(XB) == 36);

__device__ __forceinline__ int chunk_sum(const int* q8, const WB* wb, int kc) {
    const unsigned short* qs = (const unsigned short*)(wb->qs) + kc * 4;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2], v = q8[j * 2 + 1];
        const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
        const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
        sumi = STRATA_DP4A(u, __byte_perm(qe, qo, 0x5140), sumi);
        sumi = STRATA_DP4A(v, __byte_perm(qe, qo, 0x7362), sumi);
    }
    return sumi;
}

__device__ __forceinline__ float chunk_dot(const int* q8, const WB* wb, int kc) {
    return (float) chunk_sum(q8, wb, kc);
}

// A: one warp per row, one lane per 32-value chunk, warp shuffle reduce (llama.cpp-style)
__global__ void __launch_bounds__(256, 1) gemv_warp_row(const WB* __restrict__ w, const XB* __restrict__ x,
                                                        float* __restrict__ y, int n_in, int n_out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= n_out) return;
    const int bpr = n_in / 128;
    float acc = 0.f;
    for (int b = lane; b < bpr * 4; b += 32) {
        const WB* wb = w + (size_t) warp * bpr + (b >> 2);
        const XB* xb = x + (size_t) (b >> 2) * 4 + (b & 3);
        const float d2 = __half2float(wb->d);
        const float d8 = __low2float(xb->ds);
        acc += d2 * d8 * chunk_dot((const int*) xb->qs, wb, b & 3);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[warp] = acc;
}

// B: one lane per 128-value block, 16-byte loads, warp shuffle reduce
__global__ void __launch_bounds__(256, 1) gemv_lane_block(const WB* __restrict__ w, const XB* __restrict__ x,
                                                          float* __restrict__ y, int n_in, int n_out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= n_out) return;
    const int bpr = n_in / 128;
    float acc = 0.f;
    for (int blk = lane; blk < bpr; blk += 32) {
        const WB* wb = w + (size_t) warp * bpr + blk;
        const XB* xb = x + (size_t) blk * 4;
        const float d2 = __half2float(wb->d);
        const int* q8 = (const int*) xb->qs;                    // 4 chunks of 8 activation words
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
            const unsigned short* qs = (const unsigned short*) wb->qs + kc * 4;
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int qq = qs[j];
                const int u = q8[kc * 8 + j * 2], v = q8[kc * 8 + j * 2 + 1];
                const int qe = __byte_perm(0x020100ff, 0x020100ff, qq >> 0);
                const int qo = __byte_perm(0x020100ff, 0x020100ff, qq >> 2);
                sumi = STRATA_DP4A(u, __byte_perm(qe, qo, 0x5140), sumi);
                sumi = STRATA_DP4A(v, __byte_perm(qe, qo, 0x7362), sumi);
            }
            acc += d2 * __low2float(xb[kc].ds) * (float) sumi;
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[warp] = acc;
}

// C/D: warp per N rows (N = 2 or 4), one lane per 32-value chunk; the activation chunk is read once
// for all N rows, and each row's weights form an independent load stream.
template <int NROWS>
__global__ void __launch_bounds__(256, 1) gemv_warp_rows(const WB* __restrict__ w, const XB* __restrict__ x,
                                                         float* __restrict__ y, int n_in, int n_out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * NROWS;
    if (row0 >= n_out) return;
    const int bpr = n_in / 128;
    float acc[NROWS];
#pragma unroll
    for (int i = 0; i < NROWS; ++i) acc[i] = 0.f;
    for (int b = lane; b < bpr * 4; b += 32) {
        const XB* xb = x + (size_t) (b >> 2) * 4 + (b & 3);
        const int* q8 = (const int*) xb->qs;
        const float d8 = __low2float(xb->ds);
#pragma unroll
        for (int i = 0; i < NROWS; ++i) {
            if (row0 + i < n_out) {
                const WB* wb = w + (size_t) (row0 + i) * bpr + (b >> 2);
                acc[i] += __half2float(wb->d) * d8 * (float) chunk_sum(q8, wb, b & 3);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < NROWS; ++i) {
        float a = acc[i];
#pragma unroll
        for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && row0 + i < n_out) y[row0 + i] = a;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int type = argc > 1 ? std::atoi(argv[1]) : 142;
    const int variant = argc > 2 ? std::atoi(argv[2]) : 0;
    const int n_in = 17408, n_out = 110000;                 // ~512 MiB of PQ2_0 weights
    const size_t wb_bytes = strata::kernels::native_mmvq_weight_bytes(type, n_in, n_out);
    void* w = nullptr;
    void* x = nullptr;
    float* y = nullptr;
    if (cudaMalloc(&w, wb_bytes) != cudaSuccess || cudaMalloc(&x, strata::kernels::native_q8_1_bytes(n_in, 1)) != cudaSuccess ||
        cudaMalloc(&y, (size_t) n_out * 4) != cudaSuccess) {
        std::printf("allocation failed\n");
        return 1;
    }
    cudaMemset(w, 0x55, wb_bytes);
    cudaMemset(x, 0x11, strata::kernels::native_q8_1_bytes(n_in, 1));
    cudaStream_t st = nullptr;
    cudaStreamCreate(&st);
    const int R = 30;

    auto run = [&](int v) {
        auto launch = [&](int vv) {
            if (vv == 0) strata::kernels::native_mmvq(type, w, x, y, n_in, n_out, 1, st);
            else if (vv == 1) gemv_warp_row<<<(n_out + 7) / 8, 256, 0, st>>>((const WB*) w, (const XB*) x, y, n_in, n_out);
            else if (vv == 2) gemv_lane_block<<<(n_out + 7) / 8, 256, 0, st>>>((const WB*) w, (const XB*) x, y, n_in, n_out);
            else if (vv == 3) gemv_warp_rows<2><<<(n_out / 2 + 7) / 8, 256, 0, st>>>((const WB*) w, (const XB*) x, y, n_in, n_out);
            else gemv_warp_rows<4><<<(n_out / 4 + 7) / 8, 256, 0, st>>>((const WB*) w, (const XB*) x, y, n_in, n_out);
        };
        for (int i = 0; i < 3; ++i) launch(v);
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < R; ++i) launch(v);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        std::printf("type %d variant %d: %.2f GiB x%d in %.3f s = %.1f GB/s (%.3f ms/gemv)\n", type, v,
                    (double) wb_bytes / 1e9, R, s, (double) wb_bytes * R / 1e9 / s, 1000.0 * s / R);
    };

    if (variant == 0 || variant == 9) run(0);
    if (variant == 1 || variant == 9) run(1);
    if (variant == 2 || variant == 9) run(2);
    if (variant == 3 || variant == 9) run(3);
    if (variant == 4 || variant == 9) run(4);
    return 0;
}
