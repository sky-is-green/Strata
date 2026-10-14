// src/kernels/cuda/lora_corr.cu - the correction branch: round x to f16, t16 = f16(A x), out += B t16.
//
// The three passes, in order, are the reference `lora_corr_parity` mirrors and the one decode, prefill and
// verify all share:
//
//   round  x -> x16 with the repo's RNE `f16_from_f32` (the A pass's operand rule; done once per x element,
//          which is why it is a separate pass and not a per-multiply conversion in the hot loop).
//   A      one BLOCK - four warps - per (row, output) pair; the reduction axis splits into four contiguous
//          quarters of `n_in / 8` uint4 groups, each warp's lane l takes groups q0+l, q0+l+32, ... with the
//          four accumulator chains of a 4x-unrolled loop (and acc0 for the tail), the partials combine as
//          ((acc0+acc1)+(acc2+acc3)), then the 16/8/4/2/1 lane tree, then the four quarters LEFT TO RIGHT.
//          `dot8` converts the f16 pairs with one hardware conversion each (`__half22float2`, exact for the
//          f16 -> f32 direction) and accumulates in f32 in element order; the rounded f16 result is the
//          kernel's only output.  **The previous form converted every element with the bit-exact scalar
//          converter and was ALU-bound, not memory-bound: measured 69.6 -> 15.6 us on the ssm shape at
//          rows=3, warm.**  The f32 -> f16 rounding that must stay bit-exact is in `f16_from_f32`.
//   B      one WARP per (row, output) pair, lanes striding the reduction axis in uint4 groups of 8 f16,
//          both operands f16 (`t16` from the A pass, B's own rows from the file).  The scalar f32-t form
//          measured 28-43 us per branch at rows 3-5; this form 7-10 us.
//
// A is [rank, n_in] and B is [n_out, rank], both raw f16 rows.  n_in and rank must be multiples of 8 (uint4
// rows); the validator refuses other geometries rather than mis-reading.
#include "strata/kernels/lora_corr.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/f16_dot8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

// Phase 1: one block per output, four warps splitting the reduction axis.  The order is fixed by
// `lora_corr_parity`'s reference.
constexpr int A_WARPS = 4;

__global__ void lora_round_kernel(const float* __restrict__ x, uint16_t* __restrict__ x16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x16[i] = f16_from_f32(x[i]);
}

__global__ void lora_a_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ x16,
                              uint16_t* __restrict__ t16, int64_t n_rows, int64_t n_in, int64_t rank) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t i = blockIdx.x;                      // the output index (tok, r)
    if (i >= n_rows * rank) return;
    const int64_t tok = i / rank, r = i % rank;
    const int64_t n8 = n_in / 8;                       // uint4 groups in the reduction axis
    const uint16_t* row = a + (size_t) r * n_in;
    const uint16_t* xr = x16 + (size_t) tok * n_in;
    const uint4* rp = reinterpret_cast<const uint4*>(row);
    const uint4* xp = reinterpret_cast<const uint4*>(xr);
    const int64_t q0 = (n8 * warp) / A_WARPS;
    const int64_t q1 = (n8 * (warp + 1)) / A_WARPS;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int64_t k = q0 + lane;
    for (; k + 96 < q1; k += 128) {
        acc0 += f16_dot8(rp[k], xp[k]);
        acc1 += f16_dot8(rp[k + 32], xp[k + 32]);
        acc2 += f16_dot8(rp[k + 64], xp[k + 64]);
        acc3 += f16_dot8(rp[k + 96], xp[k + 96]);
    }
    for (; k < q1; k += 32) acc0 += f16_dot8(rp[k], xp[k]);
    float acc = (acc0 + acc1) + (acc2 + acc3);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float part[A_WARPS];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) t16[(size_t) i] = f16_from_f32(((part[0] + part[1]) + part[2]) + part[3]);
}

// Phase 2: one WARP per (row, output) pair, lanes striding the reduction axis in uint4 groups.  Both
// operands are already f16 (B's rows are the file's f16, `t16` is the A pass's rounded output), so the
// pairs convert with one hardware instruction each and the loop is memory-shaped.  Measured against the
// previous scalar form on this box: 28-43 us -> 7-10 us at rows 3-5.
__global__ void lora_b_kernel(const uint16_t* __restrict__ b, const uint16_t* __restrict__ t16,
                              float* __restrict__ out, int64_t n_rows, int64_t rank, int64_t n_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t i = (int64_t) blockIdx.x * WARPS + warp;
    if (i >= n_rows * n_out) return;
    const int64_t tok = i / n_out, o = i % n_out;
    const int64_t n8 = rank / 8;
    const uint4* rp = reinterpret_cast<const uint4*>(b + (size_t) o * rank);
    const uint4* tp = reinterpret_cast<const uint4*>(t16 + (size_t) tok * rank);
    float acc = 0.0f;
    for (int64_t k = lane; k < n8; k += 32) acc += f16_dot8(rp[k], tp[k]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[(size_t) tok * n_out + o] += acc;
}

// The A pass of the fused single-row router branch: the block rounds x into shared once (f16_from_f32, the same
// element-for-element rule as lora_round_kernel) and then the four warps run lora_a_kernel's quarter-split
// reduction over the shared row.  Values and addition order are lora_round_kernel + lora_a_kernel's, bit for bit.
__global__ void lora_a_round_kernel(const uint16_t* __restrict__ a, const float* __restrict__ x,
                                    uint16_t* __restrict__ t16, int64_t n_in, int64_t rank) {
    __shared__ __align__(16) uint16_t xs[4096];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t i = blockIdx.x;                      // the output index r (one row)
    if (i >= rank) return;
    for (int64_t j = threadIdx.x; j < n_in; j += blockDim.x) xs[j] = f16_from_f32(x[j]);
    __syncthreads();
    const int64_t n8 = n_in / 8;                       // uint4 groups in the reduction axis
    const uint16_t* row = a + (size_t) i * n_in;
    const uint4* rp = reinterpret_cast<const uint4*>(row);
    const uint4* xp = reinterpret_cast<const uint4*>(xs);
    const int64_t q0 = (n8 * warp) / A_WARPS;
    const int64_t q1 = (n8 * (warp + 1)) / A_WARPS;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int64_t k = q0 + lane;
    for (; k + 96 < q1; k += 128) {
        acc0 += f16_dot8(rp[k], xp[k]);
        acc1 += f16_dot8(rp[k + 32], xp[k + 32]);
        acc2 += f16_dot8(rp[k + 64], xp[k + 64]);
        acc3 += f16_dot8(rp[k + 96], xp[k + 96]);
    }
    for (; k < q1; k += 32) acc0 += f16_dot8(rp[k], xp[k]);
    float acc = (acc0 + acc1) + (acc2 + acc3);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float part[A_WARPS];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) t16[i] = f16_from_f32(((part[0] + part[1]) + part[2]) + part[3]);
}

}  // namespace

void lora_corr_round_x(const float* x, uint16_t* x16, int64_t n, void* stream) {
    if (x == nullptr || x16 == nullptr || n <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    lora_round_kernel<<<(unsigned) ((n + THREADS - 1) / THREADS), THREADS, 0, s>>>(x, x16, n);
    check("lora_corr_round_x");
}

/// The router branch (f16 pairs) at one row, fused: one launch for the rounding + A pass (lora_a_round_kernel)
/// and one for B (lora_b_kernel unchanged, `out[tok * n_out + o] += acc`, one row).  Bitwise lora_corr_apply at
/// n_rows=1, without the separate round launch and the memset / add around the logits.
void lora_corr_router(const uint16_t* a, const uint16_t* b, const float* x, uint16_t* t16, float* out, int64_t n_in,
                      int64_t rank, int64_t n_out, void* stream) {
    if (a == nullptr || b == nullptr || n_in <= 0 || n_in > 4096) return;
    cudaStream_t s = (cudaStream_t) stream;
    lora_a_round_kernel<<<(unsigned) rank, 32 * A_WARPS, 0, s>>>(a, x, t16, n_in, rank);
    check("lora_corr_router/a");
    lora_b_kernel<<<(unsigned) ((n_out + WARPS - 1) / WARPS), THREADS, 0, s>>>(b, t16, out, 1, rank, n_out);
    check("lora_corr_router/b");
}

void lora_corr_apply(const uint16_t* a, const uint16_t* b, const float* x, uint16_t* x16, uint16_t* t16, float* out,
                     int64_t n_rows, int64_t n_in, int64_t rank, int64_t n_out, void* stream) {
    if (a == nullptr || b == nullptr || n_rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    // Chunked so a caller's x16 scratch is bounded by kLoraCorrRoundRows rows; stream order makes the
    // reuse of the scratch between chunks safe.
    for (int64_t r0 = 0; r0 < n_rows; r0 += kLoraCorrRoundRows) {
        const int64_t rc = std::min(kLoraCorrRoundRows, n_rows - r0);
        lora_corr_round_x(x + (size_t) r0 * n_in, x16, rc * n_in, stream);
        const int64_t pa = rc * rank;
        lora_a_kernel<<<(unsigned) pa, 32 * A_WARPS, 0, s>>>(a, x16, t16 + (size_t) r0 * rank, rc, n_in, rank);
        check("lora_corr_apply/a");
    }
    const int64_t pb = n_rows * n_out;
    lora_b_kernel<<<(unsigned) ((pb + WARPS - 1) / WARPS), THREADS, 0, s>>>(b, t16, out, n_rows, rank, n_out);
    check("lora_corr_apply/b");
}

}  // namespace strata::kernels
