// ptq1_dequant_bench: time and checksum of the PTQ1_0 -> FP16 prompt-path dequantizers (iq_dequant_gu_f16 / iq_dequant_f16) on one
// expert's shape (gate/up 640 x 2560, down 2560 x 640), random weight blocks with finite fp16 scales.  No model needed.
// Run twice - as is (generic kernels) and with STRATA_PTQ1_DEQ_NEW=1 (A4 kernels) - and compare the checksums: they must be equal.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

static uint64_t fnv(const std::vector<uint16_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint16_t x : v) { h ^= (uint8_t) x; h *= 1099511628211ull; h ^= (uint8_t) (x >> 8); h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 300;
    const int64_t n_ff = 640, n_embd = 2560;
    const size_t mat_bytes = (size_t) n_ff * (n_embd / 128) * 28;   // gate, up and down all hold 640 x 2560 values
    std::mt19937 rng(7);
    std::vector<uint8_t> g(mat_bytes), u(mat_bytes), d(mat_bytes);
    for (auto* m : {&g, &u, &d}) {
        for (auto& b : *m) b = (uint8_t) rng();
        for (size_t o = 26; o + 2 <= m->size(); o += 28) { (*m)[o] = (uint8_t) rng(); (*m)[o + 1] = 0x20 + (rng() % 0x10) + ((rng() & 1) ? 0x80 : 0); }
    }
    uint8_t *dg, *du, *dd; uint16_t *ogu, *odn;
    cudaMalloc((void**) &dg, mat_bytes); cudaMalloc((void**) &du, mat_bytes); cudaMalloc((void**) &dd, mat_bytes);
    cudaMalloc((void**) &ogu, (size_t) n_ff * 2 * n_embd * 2); cudaMalloc((void**) &odn, (size_t) n_ff * n_embd * 2);
    cudaMemcpy(dg, g.data(), mat_bytes, cudaMemcpyHostToDevice); cudaMemcpy(du, u.data(), mat_bytes, cudaMemcpyHostToDevice); cudaMemcpy(dd, d.data(), mat_bytes, cudaMemcpyHostToDevice);
    cudaStream_t st; cudaStreamCreate(&st);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto gu = [&]() { strata::kernels::iq_dequant_gu_f16(143, dg, du, n_ff, n_embd, ogu, st, nullptr); };
    auto dn = [&]() { strata::kernels::iq_dequant_f16(143, dd, n_ff * n_embd, odn, st, nullptr); };
    for (int round = 0; round < 3; ++round)   // the first rounds only bring the GPU clocks up; the last one is printed
    for (int pass = 0; pass < 2; ++pass) {
        auto fn = pass == 0 ? std::function<void()>(gu) : std::function<void()>(dn);
        for (int i = 0; i < 20; ++i) fn();
        cudaStreamSynchronize(st);
        cudaEventRecord(e0, st);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(e1, st); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        if (round == 2) std::printf("%s: %.1f us per call (%d calls)\n", pass == 0 ? "gate/up (640 rows x 2560, FP16 out 6.5 MB)" : "down (flat 1.6 M values, FP16 out 3.3 MB)", ms * 1000.f / reps, reps);
    }
    std::vector<uint16_t> h1((size_t) n_ff * 2 * n_embd), h2((size_t) n_ff * n_embd);
    cudaMemcpy(h1.data(), ogu, h1.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(h2.data(), odn, h2.size() * 2, cudaMemcpyDeviceToHost);
    std::printf("checksum gate/up %016llx  down %016llx\n", (unsigned long long) fnv(h1), (unsigned long long) fnv(h2));
    return 0;
}
