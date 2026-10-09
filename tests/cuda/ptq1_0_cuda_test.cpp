// tests/cuda/ptq1_0_cuda_test.cpp - PTQ1_0 (type 143, the V4.1 ternary tier) on the GPU against the CPU kernels.
//   1. iq_dequant_f16(143) == ptq1_0_dequant rounded to fp16, bit for bit (every code and scale).
//   2. iq_mmvq(143) on quantize_q8_1_rows activations vs an exact double reference built from the same integers
//      (the q8_1 d and q of each 32-block are recomputed on the host the way the kernel does, d stored as fp16 in
//      block_q8_1.ds): only the fp32 summation order may differ.  Shapes: the V4.1 expert's gate/up (5120 -> rows) and down (2304 -> rows),
//      1..4 columns.
// Exit 0 = pass.  Needs a CUDA device.
#include "strata/kernels/cpu/ptq1_0.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace gk = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int failures = 0;

static void test_dequant(std::mt19937& rng) {
    const int64_t n = 256 * 64;   // 128 blocks
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    std::vector<uint8_t> q(cpu::ptq1_0_row_bytes(n));
    cpu::ptq1_0_quantize_ref(x.data(), q.data(), n);
    std::vector<float> ref(n);
    cpu::ptq1_0_dequant(q.data(), ref.data(), n);
    void* dq;
    uint16_t* dy;
    CK(cudaMalloc(&dq, q.size()));
    CK(cudaMalloc(&dy, n * 2));
    CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
    gk::iq_dequant_f16(cpu::kPtq1Type, dq, n, dy, nullptr);
    std::vector<uint16_t> got(n);
    CK(cudaMemcpy(got.data(), dy, n * 2, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const __half want = __float2half(ref[i]);
        uint16_t wb;
        std::memcpy(&wb, &want, 2);
        if (wb != got[i] && bad++ < 5) std::printf("  dequant mismatch at %lld: want %g\n", (long long) i, ref[i]);
    }
    std::printf("dequant f16: %lld values, %d mismatches %s\n", (long long) n, bad, bad ? "FAIL" : "ok");
    failures += bad != 0;
    cudaFree(dq);
    cudaFree(dy);
}

// the q8_1 integers the kernel sees for one 32-value block (quantize_q8_1_kernel: d = amax/127, q = round(x/d))
static void q8_block(const float* x, float& d, int8_t* q) {
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    d = amax / 127.f;
    for (int i = 0; i < 32; ++i) q[i] = amax == 0.f ? 0 : (int8_t) std::lround(x[i] / d);
}

static void test_mmvq(std::mt19937& rng, int n_in, int n_out, int ncols) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> W((size_t) n_out * n_in), X((size_t) ncols * n_in);
    for (auto& v : W) v = nd(rng) * 0.02f;
    for (auto& v : X) v = nd(rng);
    const size_t rb = cpu::ptq1_0_row_bytes(n_in);
    std::vector<uint8_t> Q(rb * n_out);
    for (int r = 0; r < n_out; ++r) cpu::ptq1_0_quantize_ref(W.data() + (size_t) r * n_in, Q.data() + rb * r, n_in);
    // exact reference from the integers: sum_b d_w[b] * sum_k d_a[k] * sum_i t_i * q_i
    std::vector<int8_t> T((size_t) n_out * n_in);
    std::vector<float> D((size_t) n_out * (n_in / 128));
    for (int r = 0; r < n_out; ++r)
        cpu::ptq1_0_unpack(Q.data() + rb * r, T.data() + (size_t) r * n_in, D.data() + (size_t) r * (n_in / 128), n_in);
    std::vector<double> ref((size_t) ncols * n_out);
    for (int c = 0; c < ncols; ++c) {
        std::vector<float> da(n_in / 32);
        std::vector<int8_t> qa(n_in);
        for (int k = 0; k < n_in / 32; ++k) q8_block(X.data() + (size_t) c * n_in + 32 * k, da[k], qa.data() + 32 * k);
        for (int r = 0; r < n_out; ++r) {
            double s = 0;
            for (int b = 0; b < n_in / 128; ++b) {
                double sb = 0;
                for (int k = 0; k < 4; ++k) {
                    long isum = 0;
                    for (int i = 0; i < 32; ++i) {
                        const int e = 128 * b + 32 * k + i;
                        isum += (long) T[(size_t) r * n_in + e] * qa[e];
                    }
                    // the kernel reads d from block_q8_1.ds, a half2: q comes from the fp32 d, the dot uses fp16 d
                    sb += (double) __half2float(__float2half(da[4 * b + k])) * (double) isum;
                }
                s += (double) D[(size_t) r * (n_in / 128) + b] * sb;
            }
            ref[(size_t) c * n_out + r] = s;
        }
    }
    void *dw, *dx, *dxq;
    float* dy;
    CK(cudaMalloc(&dw, Q.size()));
    CK(cudaMalloc(&dx, X.size() * 4));
    CK(cudaMalloc(&dxq, (size_t) ncols * (n_in / 32) * 36));
    CK(cudaMalloc(&dy, (size_t) ncols * n_out * 4));
    CK(cudaMemcpy(dw, Q.data(), Q.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, X.data(), X.size() * 4, cudaMemcpyHostToDevice));
    gk::quantize_q8_1_rows((const float*) dx, ncols, n_in, dxq, nullptr);
    gk::iq_mmvq(cpu::kPtq1Type, dw, dxq, dy, n_in, n_out, ncols, nullptr);
    std::vector<float> got((size_t) ncols * n_out);
    CK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0, scale = 0;
    for (double v : ref) scale = std::max(scale, std::fabs(v));
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - ref[i]) / scale);
    // fp32 accumulation of ~n_in/128 block terms: a few ulp of the largest output
    const bool ok = worst < 2e-5;
    std::printf("mmvq %4d -> %3d x %d cols: worst |gpu - exact| / max|exact| = %.2e %s\n", n_in, n_out, ncols, worst,
                ok ? "ok" : "FAIL");
    failures += !ok;
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dxq);
    cudaFree(dy);
}

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) { std::printf("no CUDA device: skipped\n"); return 0; }
    std::mt19937 rng(143);
    test_dequant(rng);
    for (int ncols : {1, 2, 4}) {
        test_mmvq(rng, 5120, 96, ncols);   // V4.1 gate/up rows: n_in = hidden 5120
        test_mmvq(rng, 2304, 96, ncols);   // V4.1 down rows: n_in = moe_inter 2304 (18 blocks per row)
    }
    std::printf("%s (%d failed)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
