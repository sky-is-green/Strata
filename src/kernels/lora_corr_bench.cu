// src/kernels/lora_corr_bench.cu - scratch bench for the runtime correction branches.
//
// The served path runs the correction branches at VERIFY-WINDOW shapes (rows = the accepted draft
// count + 1, up to --spec + 1 = 5), not at rows = 1.  This bench walks rows = 1..8 for the two real
// shapes and reports, for each: the single-call GPU interval (an event pair around one call, synced -
// this exposes the dispatch gap a graph node does not pay), the back-to-back interval (one event pair
// around `reps` calls - the kernel time proper), and the A/B passes apart.  The A/B copies below are
// the PRE-SWAP software-converter kernels, kept as the before arm: `lora_corr_apply` is the shipping
// path (hardware round trips since 2026-10-13).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/lora_corr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int A_WARPS = 4;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

__global__ void empty_kernel() {}

// ---- the current A pass (split-4), copied for the breakdown ----
__global__ void bench_a_kernel(const uint16_t* __restrict__ a, const float* __restrict__ x, float* __restrict__ t,
                               int64_t n_rows, int64_t n_in, int64_t rank) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t i = blockIdx.x;
    if (i >= n_rows * rank) return;
    const int64_t tok = i / rank, r = i % rank;
    const uint16_t* row = a + (size_t) r * n_in;
    const float* xr = x + (size_t) tok * n_in;
    const int64_t q0 = (n_in * warp) / A_WARPS;
    const int64_t q1 = (n_in * (warp + 1)) / A_WARPS;
    float acc = 0.0f;
    int64_t k = q0 + lane;
    for (; k + 96 < q1; k += 128) {
        const float ea = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(xr[k])) * strata::kernels::f32_from_f16(row[k]);
        const float eb = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(xr[k + 32])) * strata::kernels::f32_from_f16(row[k + 32]);
        const float ec = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(xr[k + 64])) * strata::kernels::f32_from_f16(row[k + 64]);
        const float ed = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(xr[k + 96])) * strata::kernels::f32_from_f16(row[k + 96]);
        acc += ea; acc += eb; acc += ec; acc += ed;
    }
    for (; k < q1; k += 32)
        acc += strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(xr[k])) * strata::kernels::f32_from_f16(row[k]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float part[A_WARPS];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) t[(size_t) i] = ((part[0] + part[1]) + part[2]) + part[3];
}

// ---- diagnostic only: the A pass without the x round-trip (NOT the shipping arithmetic) ----
__global__ void bench_a_nox_kernel(const uint16_t* __restrict__ a, const float* __restrict__ x, float* __restrict__ t,
                                   int64_t n_rows, int64_t n_in, int64_t rank) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t i = blockIdx.x;
    if (i >= n_rows * rank) return;
    const int64_t tok = i / rank, r = i % rank;
    const uint16_t* row = a + (size_t) r * n_in;
    const float* xr = x + (size_t) tok * n_in;
    const int64_t q0 = (n_in * warp) / A_WARPS;
    const int64_t q1 = (n_in * (warp + 1)) / A_WARPS;
    float acc = 0.0f;
    int64_t k = q0 + lane;
    for (; k + 96 < q1; k += 128) {
        acc += xr[k] * strata::kernels::f32_from_f16(row[k]);
        acc += xr[k + 32] * strata::kernels::f32_from_f16(row[k + 32]);
        acc += xr[k + 64] * strata::kernels::f32_from_f16(row[k + 64]);
        acc += xr[k + 96] * strata::kernels::f32_from_f16(row[k + 96]);
    }
    for (; k < q1; k += 32) acc += xr[k] * strata::kernels::f32_from_f16(row[k]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float part[A_WARPS];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) t[(size_t) i] = ((part[0] + part[1]) + part[2]) + part[3];
}

// ---- diagnostic only: the A pass with both operands already f32 (no conversions at all) ----
__global__ void bench_a_f32_kernel(const float* __restrict__ af, const float* __restrict__ x, float* __restrict__ t,
                                   int64_t n_rows, int64_t n_in, int64_t rank) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t i = blockIdx.x;
    if (i >= n_rows * rank) return;
    const int64_t tok = i / rank, r = i % rank;
    const float* row = af + (size_t) r * n_in;
    const float* xr = x + (size_t) tok * n_in;
    const int64_t q0 = (n_in * warp) / A_WARPS;
    const int64_t q1 = (n_in * (warp + 1)) / A_WARPS;
    float acc = 0.0f;
    int64_t k = q0 + lane;
    for (; k + 96 < q1; k += 128) {
        acc += xr[k] * row[k];
        acc += xr[k + 32] * row[k + 32];
        acc += xr[k + 64] * row[k + 64];
        acc += xr[k + 96] * row[k + 96];
    }
    for (; k < q1; k += 32) acc += xr[k] * row[k];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float part[A_WARPS];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) t[(size_t) i] = ((part[0] + part[1]) + part[2]) + part[3];
}

// ---- the current B pass, copied for the breakdown ----
__global__ void bench_b_kernel(const uint16_t* __restrict__ b, const float* __restrict__ t, float* __restrict__ out,
                               int64_t n_rows, int64_t rank, int64_t n_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t i = (int64_t) blockIdx.x * WARPS + warp;
    if (i >= n_rows * n_out) return;
    const int64_t tok = i / n_out, o = i % n_out;
    const uint16_t* row = b + (size_t) o * rank;
    const float* tr = t + (size_t) tok * rank;
    float acc = 0.0f;
    int64_t k = lane;
    for (; k + 96 < rank; k += 128) {
        const float ea = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(tr[k])) * strata::kernels::f32_from_f16(row[k]);
        const float eb = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(tr[k + 32])) * strata::kernels::f32_from_f16(row[k + 32]);
        const float ec = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(tr[k + 64])) * strata::kernels::f32_from_f16(row[k + 64]);
        const float ed = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(tr[k + 96])) * strata::kernels::f32_from_f16(row[k + 96]);
        acc += ea; acc += eb; acc += ec; acc += ed;
    }
    for (; k < rank; k += 32)
        acc += strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(tr[k])) * strata::kernels::f32_from_f16(row[k]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[(size_t) tok * n_out + o] += acc;
}

uint32_t rng_state = 0x12345678u;
float rnd() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float) (rng_state >> 8) / (float) (1u << 24) - 0.5f) * 0.1f;
}

struct Times {
    double single = 0, thru = 0;
};

template <typename F>
Times time_it(F launch, cudaStream_t s, int reps) {
    Times t;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 3; ++i) launch();
    cudaStreamSynchronize(s);
    double tot = 0;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0, s);
        launch();
        cudaEventRecord(e1, s);
        cudaStreamSynchronize(s);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        tot += ms;
    }
    t.single = tot / reps;
    cudaEventRecord(e0, s);
    for (int i = 0; i < reps; ++i) launch();
    cudaEventRecord(e1, s);
    cudaStreamSynchronize(s);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    t.thru = ms / reps;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return t;
}

struct Shape {
    const char* name;
    int64_t n_in, n_out;
};

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 30;
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int64_t rank = 512;
    const int64_t max_rows = 8;
    const Times e = time_it([&] { empty_kernel<<<1, 32, 0, s>>>(); }, s, reps);
    std::printf("empty kernel: single %.1f us, back-to-back %.1f us\n", e.single * 1000.0, e.thru * 1000.0);
    const Shape shapes[] = {{"ssm/attn", 6144, 2560}, {"ffn", 2560, 2560}};
    for (const Shape& sh : shapes) {
        std::vector<uint16_t> a((size_t) (rank * sh.n_in)), b((size_t) (sh.n_out * rank));
        std::vector<float> x((size_t) (max_rows * sh.n_in)), t((size_t) (max_rows * rank)),
            out((size_t) (max_rows * sh.n_out), 0.5f);
        for (auto& v : a) v = strata::kernels::f16_from_f32(rnd());
        for (auto& v : b) v = strata::kernels::f16_from_f32(rnd());
        for (auto& v : x) v = rnd();
        uint16_t *da = nullptr, *db = nullptr, *dx16 = nullptr, *dt16 = nullptr;
        float *dx = nullptr, *dt = nullptr, *dout = nullptr;
        cudaMalloc(&da, a.size() * 2);
        cudaMalloc(&db, b.size() * 2);
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dt, t.size() * 4);
        cudaMalloc(&dt16, t.size() * 2);
        cudaMalloc(&dout, out.size() * 4);
        const int64_t round_rows = std::min<int64_t>(max_rows, strata::kernels::kLoraCorrRoundRows);
        cudaMalloc(&dx16, (size_t) round_rows * sh.n_in * 2);
        cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b.data(), b.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dout, out.data(), out.size() * 4, cudaMemcpyHostToDevice);
        std::vector<float> af((size_t) (rank * sh.n_in));
        for (size_t i = 0; i < af.size(); ++i) af[i] = strata::kernels::f32_from_f16(a[i]);
        float* daf = nullptr;
        cudaMalloc(&daf, af.size() * 4);
        cudaMemcpy(daf, af.data(), af.size() * 4, cudaMemcpyHostToDevice);
        ck("setup");
        std::printf("\n%s  (n_in %lld, rank %lld, n_out %lld)\n", sh.name, (long long) sh.n_in, (long long) rank,
                    (long long) sh.n_out);
        std::printf("  rows |   apply s    apply t   |    A s     A t    |    B s     B t    |  nox t   f32A t\n");
        for (int64_t rows = 1; rows <= max_rows; ++rows) {
            const Times ap = time_it(
                [&] { strata::kernels::lora_corr_apply(da, db, dx, dx16, dt16, dout, rows, sh.n_in, rank, sh.n_out, s); }, s, reps);
            const unsigned pa = (unsigned) (rows * rank);
            const Times ta = time_it([&] { bench_a_kernel<<<pa, 32 * A_WARPS, 0, s>>>(da, dx, dt, rows, sh.n_in, rank); }, s, reps);
            const unsigned pb = (unsigned) ((rows * sh.n_out + WARPS - 1) / WARPS);
            const Times tb = time_it([&] { bench_b_kernel<<<pb, THREADS, 0, s>>>(db, dt, dout, rows, rank, sh.n_out); }, s, reps);
            const Times tnox = time_it([&] { bench_a_nox_kernel<<<pa, 32 * A_WARPS, 0, s>>>(da, dx, dt, rows, sh.n_in, rank); }, s, reps);
            const Times tf32 = time_it([&] { bench_a_f32_kernel<<<pa, 32 * A_WARPS, 0, s>>>(daf, dx, dt, rows, sh.n_in, rank); }, s, reps);
            std::printf("  %4lld | %8.1f %10.1f   | %8.1f %8.1f   | %8.1f %8.1f   | %7.1f %8.1f\n", (long long) rows,
                        ap.single * 1000.0, ap.thru * 1000.0, ta.single * 1000.0, ta.thru * 1000.0, tb.single * 1000.0,
                        tb.thru * 1000.0, tnox.thru * 1000.0, tf32.thru * 1000.0);
            ck("loop");
        }
        cudaFree(daf);
        cudaFree(da);
        cudaFree(db);
        cudaFree(dx16);
        cudaFree(dx);
        cudaFree(dt);
        cudaFree(dt16);
        cudaFree(dout);
    }
    cudaStreamDestroy(s);
    return 0;
}
