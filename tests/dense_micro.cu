// Per-call GPU time of the dense decode pass's small kernels at n=1 (the shapes Bonsai 2 uses).  The numbers
// include the per-launch dispatch, which a HIP-graph replay removes; they are the instrument that found the
// latency-bound loops the unrolling commit fixed.  Run with no arguments.
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_rope.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <functional>
#include <vector>

static cudaStream_t st = nullptr;

static float us_per(int iters, const std::function<void()>& f) {
    for (int i = 0; i < 5; ++i) f();
    cudaStreamSynchronize(st);
    cudaEvent_t a = nullptr, b = nullptr;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, st);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(b, st);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms * 1000.0f / (float) iters;
}

int main() {
    cudaStreamCreate(&st);
    const int E = 5120, FF = 17408, V = 6144, H = 24, D = 256;
    const size_t maxw = FF;
    float *x = nullptr, *xn = nullptr, *g = nullptr, *u = nullptr, *w = nullptr, *sg = nullptr;
    void* q8 = nullptr;
    cudaMalloc(&x, maxw * 4);
    cudaMalloc(&xn, maxw * 4);
    cudaMalloc(&g, maxw * 4);
    cudaMalloc(&u, maxw * 4);
    cudaMalloc(&w, (size_t) E * 4);
    cudaMalloc(&q8, strata::kernels::native_q8_1_bytes((int) maxw, 8));
    std::vector<float> h(maxw);
    for (size_t i = 0; i < maxw; ++i) h[i] = (float) ((int) (i % 37) - 18) * 0.25f;
    cudaMemcpy(x, h.data(), maxw * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(u, h.data(), maxw * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g, h.data(), maxw * 4, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < (size_t) E; ++i) h[i] = 1.0f + 0.001f * (float) (i % 7);
    cudaMemcpy(w, h.data(), (size_t) E * 4, cudaMemcpyHostToDevice);
    std::vector<float> s((size_t) FF, 1.0f);
    cudaMalloc(&sg, (size_t) FF * 4);
    cudaMemcpy(sg, s.data(), (size_t) FF * 4, cudaMemcpyHostToDevice);

    std::printf("fwht_quantize    E  (5120x1):  %7.2f us\n", us_per(2000, [&] {
        strata::kernels::dense_fwht_quantize_rows(x, q8, nullptr, sg, E, 1, 0, 0, 0, st); }));
    std::printf("fwht_quantize    V  (6144x1):  %7.2f us\n", us_per(2000, [&] {
        strata::kernels::dense_fwht_quantize_rows(x, q8, nullptr, sg, V, 1, 0, 0, 0, st); }));
    std::printf("fwht_quantize    FF (17408x1): %7.2f us\n", us_per(1000, [&] {
        strata::kernels::dense_fwht_quantize_rows(x, q8, nullptr, sg, FF, 1, 0, 0, 0, st); }));
    std::printf("swiglu_quant     FF (17408x1): %7.2f us\n", us_per(1000, [&] {
        strata::kernels::dense_swiglu_quantize_rows(g, u, q8, sg, FF, 1, nullptr, nullptr, st); }));
    std::printf("rms_norm         E (rows=1):   %7.2f us\n", us_per(2000, [&] {
        strata::kernels::dense_rms_norm(x, w, xn, 1, E, 1e-6f, st); }));
    std::printf("rms_norm + add   E (rows=1):   %7.2f us\n", us_per(2000, [&] {
        strata::kernels::dense_rms_norm(x, w, xn, 1, E, 1e-6f, st, u); }));
    std::printf("rms_fwht_quant   E (rows=1):   %7.2f us\n", us_per(2000, [&] {
        strata::kernels::dense_rms_fwht_quant_rows(x, nullptr, w, xn, q8, sg, E, 1, 1e-6f, st); }));
    std::printf("split_q_rms     (24x256):      %7.2f us\n", us_per(2000, [&] {
        strata::kernels::dense_split_q_rms(x, w, xn, H, D, 1e-6f, st); }));
    std::printf("quantize         E (5120x1):   %7.2f us\n", us_per(2000, [&] {
        strata::kernels::native_quantize_q8_1(x, q8, E, 1, st); }));
    std::printf("rope rows       (24x256):      %7.2f us\n", us_per(2000, [&] {
        strata::kernels::native_rope_apply_rows(x, xn, H, D, 64, strata::kernels::rope_scaling(), H, 0, st); }));
    cudaStreamSynchronize(st);
    return 0;
}
