// src/kernels/native_expert_bench.cpp - the grouped native experts on real GGUF rows: one AMD layout against
// another, bit for bit, and each one's time.
//
//     build/native_expert_bench <shard1.gguf> <layer[,layer...]> <groups> <tokens per group> <ref mode> <mode> [iters]
//
// Modes are STRATA_EXP_MODE values.  A group is one expert (distinct blobs, ~2 MB each, so G >= 8 does not fit
// in L2); its entries read distinct tokens of an 8-token window.  The output must be bitwise equal to the
// reference mode's: the verify window's text depends on it.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 7) {
        std::fprintf(stderr, "usage: native_expert_bench <shard1.gguf> <layer[,layer]> <groups> <tokens> <ref mode> <mode> [iters]\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    std::vector<int> layers;
    {
        std::stringstream ss(argv[2]);
        std::string t;
        while (std::getline(ss, t, ',')) layers.push_back(std::atoi(t.c_str()));
    }
    const int G = std::atoi(argv[3]), T = std::atoi(argv[4]), mref = std::atoi(argv[5]), mode = std::atoi(argv[6]);
    const int iters = argc > 7 ? std::atoi(argv[7]) : 200;
    const bool single = argc > 8 ? std::atoi(argv[8]) != 0 : false;   // the per-entry kernels (single_entry)
    const int64_t H = argc > 9 ? std::atoll(argv[9]) : 2560;          // the expert geometry (Flash-Next default)
    const int64_t FF = argc > 10 ? std::atoll(argv[10]) : 640;
    const int64_t FF_FULL = FF;
    const int NTOK = 8;
    if (T < 1 || T > NTOK || G < 1) { std::fprintf(stderr, "tokens 1..8, groups >= 1\n"); return 2; }
    int failures = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& ti : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
        const K::NativeExpertLayout L = K::native_expert_layout((int) t[0]->type, (int) t[2]->type, H, FF);
        if (L.bytes == 0 || L.gu_row == 0 || L.d_row == 0) {
            std::printf("layer %d: the expert layout does not take types %d/%d at %lldx%lld\n", l, (int) t[0]->type,
                        (int) t[2]->type, (long long) H, (long long) FF);
            ++failures;
            continue;
        }
        const int64_t n_exp = t[0]->shape.size() > 2 ? t[0]->shape[2] : 1;
        const size_t dsz = (size_t) L.bytes - L.down_off, drow = dsz / H;
        std::vector<uint8_t> blobs((size_t) G * L.bytes);
        for (int g = 0; g < G; ++g) {
            const size_t E = (size_t) ((g * 37 + 5) % n_exp);
            uint8_t* b = blobs.data() + (size_t) g * L.bytes;
            std::memcpy(b, gguf.tensor_data(*t[0]) + E * L.up_off, L.up_off);
            std::memcpy(b + L.up_off, gguf.tensor_data(*t[1]) + E * L.up_off, L.up_off);
            for (int64_t r = 0; r < H; ++r)
                std::memcpy(b + L.down_off + r * drow, gguf.tensor_data(*t[2]) + E * dsz + r * drow, drow);
        }
        std::mt19937 rng(17 + l);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) NTOK * H);
        for (auto& v : x) v = nd(rng);
        const int NE = G * T;
        std::vector<unsigned long long> ptr(G);
        std::vector<int32_t> start(G + 1), dst(NE), tok(NE);
        uint8_t* dblob;
        void *dx, *dxq, *dscr;
        float* dout;
        unsigned long long* dptr;
        int32_t *dstart, *dn, *ddst, *dtok;
        cudaMalloc((void**) &dblob, blobs.size());
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dxq, (size_t) NTOK * H / 32 * 36);
        cudaMalloc(&dscr, K::native_expert_scratch_bytes(NE, FF));
        cudaMalloc((void**) &dout, (size_t) NE * H * 4);
        cudaMalloc((void**) &dptr, (size_t) G * 8);
        cudaMalloc((void**) &dstart, (size_t) (G + 1) * 4);
        cudaMalloc((void**) &dn, 4);
        cudaMalloc((void**) &ddst, (size_t) NE * 4);
        cudaMalloc((void**) &dtok, (size_t) NE * 4);
        for (int g = 0; g < G; ++g) {
            ptr[g] = (unsigned long long) (dblob + (size_t) g * L.bytes);
            start[g] = g * T;
            for (int j = 0; j < T; ++j) {
                dst[g * T + j] = g * T + j;
                tok[g * T + j] = (g * 3 + j) % NTOK;
            }
        }
        start[G] = NE;
        cudaMemcpy(dblob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dptr, ptr.data(), (size_t) G * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dstart, start.data(), (size_t) (G + 1) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dn, &G, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddst, dst.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtok, tok.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
        K::quantize_q8_1_rows((const float*) dx, NTOK, H, dxq, s);
        auto run = [&](int m, int phase) {
            K::native_expert_set_mode(m, phase);
            K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, G, NE, dxq, dscr, dout, s, 0, single);
        };
        std::vector<float> ref((size_t) NE * H), got((size_t) NE * H);
        K::native_expert_set_down_rows(false);   // the reference arm: the one-row down mapping
        K::native_expert_set_gu_rows(false);     // ... and the one-row gate/up mapping
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(mref, 0);
        cudaMemcpyAsync(ref.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost, s);
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(mode, 0);
        cudaMemcpyAsync(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) { std::printf("layer %d: CUDA error\n", l); return 1; }
        size_t ndiff = 0;
        double worst = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (std::memcmp(&ref[i], &got[i], 4) != 0) {
                ++ndiff;
                worst = std::fmax(worst, std::fabs((double) ref[i] - got[i]) / (std::fabs((double) ref[i]) + 1e-6));
            }
        auto time = [&](int m, int phase) {
            for (int i = 0; i < 10; ++i) run(m, phase);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) run(m, phase);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1e3 * ms / iters;
        };
        const double gu_b = (double) G * 2 * L.up_off, d_b = (double) G * dsz;
        double us[2][3] = {{1e30, 1e30, 1e30}, {1e30, 1e30, 1e30}};
        for (int round = 0; round < 3; ++round)   // interleaved, the minimum: the clocks wander between runs
            for (int k = 0; k < 2; ++k)
                for (int p = 0; p < 3; ++p) us[k][p] = std::fmin(us[k][p], time(k ? mode : mref, p));
        std::printf("layer %2d %-7s/%-6s G %d T %d%s | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s)"
                    " | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s) | %s\n",
                    l, ggml_type_name((ggml_type) L.gu_type), ggml_type_name((ggml_type) L.d_type), G, T,
                    single ? " single" : "",
                    mref, us[0][0], us[0][1], gu_b / us[0][1] / 1e3, us[0][2], d_b / us[0][2] / 1e3,
                    mode, us[1][0], us[1][1], gu_b / us[1][1] / 1e3, us[1][2], d_b / us[1][2] / 1e3,
                    ndiff ? "DIFFERS" : "bitwise equal");
        // the rows-per-warp kernels: outputs against the one-row mapping (the ref above) and their times
        K::native_expert_set_gu_rows(true);
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(0, 0);
        cudaMemcpyAsync(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost, s);
        const double rows_g = time(0, 1);
        if (cudaStreamSynchronize(s) != cudaSuccess) { std::printf("layer %d: CUDA error (rows gu)\n", l); return 1; }
        size_t gdiff = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (std::memcmp(&ref[i], &got[i], 4) != 0) ++gdiff;
        std::printf("          gu rows-per-warp: %6.1f -> %6.1f us (%3.0f GB/s), outputs %s\n", us[0][1], rows_g,
                    gu_b / rows_g / 1e3, gdiff ? "DIFFER" : "bitwise equal");
        if (gdiff) ++failures;
        K::native_expert_set_down_rows(true);
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(0, 0);
        cudaMemcpyAsync(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost, s);
        const double rows_d = time(0, 2);
        if (cudaStreamSynchronize(s) != cudaSuccess) { std::printf("layer %d: CUDA error (rows)\n", l); return 1; }
        size_t rdiff = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (std::memcmp(&ref[i], &got[i], 4) != 0) ++rdiff;
        std::printf("          down rows-per-warp: %6.1f -> %6.1f us (%3.0f GB/s), outputs %s\n", us[0][2], rows_d,
                    d_b / rows_d / 1e3, rdiff ? "DIFFER" : "bitwise equal");
        if (rdiff) ++failures;
        if (ndiff) {
            std::printf("          %zu of %zu outputs differ, worst rel %.2e\n", ndiff, ref.size(), worst);
            ++failures;
        }
        K::native_expert_set_mode(-1, 0);
        cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
        cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    }
    std::printf("native_expert_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
