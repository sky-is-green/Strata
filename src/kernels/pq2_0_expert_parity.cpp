// src/kernels/pq2_0_expert_parity.cpp - the Scion 35B's PQ2_0 (ggml type 142) expert path: the GPU kernels
// against a float reference built from the block decode, on real blobs from the release file.
//
//     build/pq2_0_expert_parity <35b.gguf> [layer] [experts]
//
// (a) the float reference dequantizes the blob with the container's own rule (value = d * (code - 1)) and runs
// a float SwiGLU expert; (b) the GPU path is `native_expert_grouped` with q8_1 activations, so it differs from
// (a) by the activation rounding only.  A page fault or a large relative error localizes a kernel bug without
// the engine's two-GPU serve in the way.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

static constexpr int64_t H = 2048, FF = 512;
static constexpr int NT = 3, G = 6;      // tokens, groups; entries = G * T
static constexpr int T = 4;              // entries per group

static void CK(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "pq2_0: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: pq2_0_expert_parity <35b.gguf> [layer] [experts]\n"); return 2; }
    const int64_t layer = argc > 2 ? std::atoi(argv[2]) : 0;
    const int experts = argc > 3 ? std::atoi(argv[3]) : (int) G;

    strata::GgufFile gguf(argv[1]);
    const strata::TensorInfo *tg = nullptr, *tu = nullptr, *td = nullptr;
    const std::string gn = "blk." + std::to_string(layer) + ".ffn_gate_exps.weight";
    const std::string un = "blk." + std::to_string(layer) + ".ffn_up_exps.weight";
    const std::string dn = "blk." + std::to_string(layer) + ".ffn_down_exps.weight";
    for (const auto& t : gguf.tensors()) {
        if (t.name == gn) tg = &t;
        if (t.name == un) tu = &t;
        if (t.name == dn) td = &t;
    }
    if (!tg || !tu || !td) { std::fprintf(stderr, "pq2_0: layer %lld has no expert tensors\n", (long long) layer); return 2; }
    if (tg->type != 142 || tu->type != 142 || td->type != 142) {
        std::fprintf(stderr, "pq2_0: experts are type %d/%d/%d, not PQ2_0/142\n", (int) tg->type, (int) tu->type, (int) td->type);
        return 2;
    }
    const uint8_t* gsrc = gguf.tensor_data(*tg);
    const uint8_t* usrc = gguf.tensor_data(*tu);
    const uint8_t* dsrc = gguf.tensor_data(*td);

    const K::NativeExpertLayout L = K::native_expert_layout(142, 142, H, FF);
    const size_t gu_row = L.gu_row, d_row = L.d_row;
    const size_t gsz = (size_t) FF * gu_row, dsz = (size_t) H * d_row;
    if (L.bytes != 2 * gsz + dsz || gu_row != (size_t) (H / 128) * 34 || d_row != (size_t) (FF / 128) * 34) {
        std::fprintf(stderr, "pq2_0: layout mismatch: bytes %zu gu_row %zu d_row %zu\n", L.bytes, gu_row, d_row);
        return 2;
    }
    const int G_use = std::min(experts, (int) tg->shape[2]);

    // one blob per group: [gate | up | down] as the engine's expert_blob() builds it
    const int NE = G_use * T;
    std::vector<uint8_t> blobs((size_t) G_use * L.bytes);
    for (int g = 0; g < G_use; ++g) {
        const size_t E = (size_t) ((g * 37 + 5) % tg->shape[2]);
        uint8_t* b = blobs.data() + (size_t) g * L.bytes;
        std::memcpy(b, gsrc + E * gsz, gsz);
        std::memcpy(b + L.up_off, usrc + E * gsz, gsz);
        std::memcpy(b + L.down_off, dsrc + E * dsz, dsz);
    }

    std::mt19937 rng(11 + (unsigned) layer);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);

    std::vector<unsigned long long> ptr(G_use);
    std::vector<int32_t> start(G_use + 1), dst(NE), tok(NE);
    for (int g = 0; g < G_use; ++g) {
        start[g] = g * T;
        for (int j = 0; j < T; ++j) {
            dst[g * T + j] = g * T + j;
            tok[g * T + j] = (g + j) % NT;
        }
    }
    start[G_use] = NE;

    uint8_t *dblob, *dxq, *dscr;
    void* dx;
    float* dout;
    unsigned long long* dptr;
    int32_t *dstart, *dng, *ddst, *dtok;
    cudaStream_t s;
    CK(cudaStreamCreate(&s), "stream");
    CK(cudaMalloc((void**) &dblob, blobs.size()), "blob");
    CK(cudaMalloc(&dx, x.size() * 4), "x");
    CK(cudaMalloc(&dxq, (size_t) NT * H / 32 * 36), "xq");
    CK(cudaMalloc(&dscr, K::native_expert_scratch_bytes(NE, FF)), "scratch");
    CK(cudaMalloc((void**) &dout, (size_t) NE * H * 4), "out");
    CK(cudaMalloc((void**) &dptr, (size_t) G_use * 8), "ptr");
    CK(cudaMalloc(&dstart, (size_t) (G_use + 1) * 4), "start");
    CK(cudaMalloc(&dng, 4), "n");
    CK(cudaMalloc(&ddst, (size_t) NE * 4), "dst");
    CK(cudaMalloc(&dtok, (size_t) NE * 4), "tok");
    for (int g = 0; g < G_use; ++g) ptr[g] = (unsigned long long) (dblob + (size_t) g * L.bytes);
    CK(cudaMemcpy(dblob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "blob up");
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x up");
    CK(cudaMemcpy(dptr, ptr.data(), (size_t) G_use * 8, cudaMemcpyHostToDevice), "ptr up");
    CK(cudaMemcpy(dstart, start.data(), (size_t) (G_use + 1) * 4, cudaMemcpyHostToDevice), "start up");
    const int32_t ng = G_use;
    CK(cudaMemcpy(dng, &ng, 4, cudaMemcpyHostToDevice), "n up");
    CK(cudaMemcpy(ddst, dst.data(), (size_t) NE * 4, cudaMemcpyHostToDevice), "dst up");
    CK(cudaMemcpy(dtok, tok.data(), (size_t) NE * 4, cudaMemcpyHostToDevice), "tok up");
    K::quantize_q8_1_rows((const float*) dx, NT, H, dxq, s);

    const char* ph = std::getenv("PQ2_PHASE");
    const int phase = ph ? std::atoi(ph) : 0;
    K::native_expert_set_mode(0, phase);
    std::fprintf(stderr, "pq2_0: layer %lld, %d groups x %d entries, %zu-byte blobs, phase %d; launching\n",
                 (long long) layer, G_use, T, L.bytes, phase);
    K::native_expert_grouped(L, dptr, dstart, dng, ddst, dtok, G_use, NE, dxq, dscr, dout, s, 0);
    std::fprintf(stderr, "pq2_0: launched, syncing\n");
    CK(cudaStreamSynchronize(s), "kernel");
    std::fprintf(stderr, "pq2_0: kernel ok\n");
    std::vector<float> got((size_t) NE * H);
    CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost), "out down");

    // q8_1-rounded activations: what the GPU path actually multiplies against (the reference below uses these,
    // so the only remaining difference is the accumulation order inside the dot)
    auto q8_1_rows = [](const float* src, int64_t rows, int64_t cols, std::vector<float>& dst) {
        dst.resize((size_t) rows * cols);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t b = 0; b < cols / 32; ++b) {
                const float* s = src + r * cols + b * 32;
                float amax = 0;
                for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(s[i]));
                const float d = amax / 127.0f;
                for (int i = 0; i < 32; ++i) {
                    float q = d > 0 ? std::nearbyint(s[i] / d) : 0.0f;
                    q = std::fmax(-127.0f, std::fmin(127.0f, q));
                    dst[(size_t) r * cols + b * 32 + i] = q * d;
                }
            }
    };
    std::vector<float> xq_rounded;
    q8_1_rows(x.data(), NT, H, xq_rounded);

    // the float reference: SwiGLU expert on decoded weights and the q8_1-rounded activations
    auto code = [](const uint8_t* qs, int64_t i) { return (qs[i / 4] >> ((i % 4) * 2)) & 3; };
    auto row_dot_ref = [&](const uint8_t* part, size_t row_bytes, int64_t row, int64_t n_in, const float* xv) {
        const uint8_t* r = part + (size_t) row * row_bytes;
        double acc = 0;
        for (int64_t i = 0; i < n_in; ++i) {
            const int64_t blk = i / 128, off = i % 128;
            uint16_t bits;
            std::memcpy(&bits, r + blk * 34, 2);
            __half hh;
            std::memcpy(&hh, &bits, 2);
            const float d = __half2float(hh);
            const int c = (int) code(r + blk * 34 + 2, off);
            acc += (double) d * (double) (c - 1) * (double) xv[i];
        }
        return (float) acc;
    };
    double worst_out = 0, denom = 0, errsum = 0;
    std::vector<float> h((size_t) FF);
    for (int e = 0; e < NE; ++e) {
        const int g = e / T, t = tok[e];
        const uint8_t* b = blobs.data() + (size_t) g * L.bytes;
        for (int64_t j = 0; j < FF; ++j) {
            const float gv = row_dot_ref(b, gu_row, j, H, xq_rounded.data() + (size_t) t * H);
            const float uv = row_dot_ref(b + L.up_off, gu_row, j, H, xq_rounded.data() + (size_t) t * H);
            h[(size_t) j] = gv / (1.0f + std::exp(-gv)) * uv;
        }
        // the down activation is the q8_1 image of h (the GPU's swiglu_quant writes it)
        std::vector<float> hq_rounded;
        q8_1_rows(h.data(), 1, FF, hq_rounded);
        for (int64_t r = 0; r < H; ++r) {
            const float ref = row_dot_ref(b + L.down_off, d_row, r, FF, hq_rounded.data());
            const float gotv = got[(size_t) e * H + r];
            const double err = std::fabs((double) gotv - ref);
            denom += std::fabs((double) ref);
            errsum += err;
            if (err > worst_out) worst_out = err;
        }
    }
    const double mean_rel = errsum / (denom + 1e-30);
    const double max_rel = worst_out / (denom / (double) (NE * H) + 1e-30);
    std::fprintf(stderr, "pq2_0: mean |gpu - ref| / mean |ref| %.4g, max %.4g over %d entries\n", mean_rel, max_rel, NE);
    // the residual is q8_1 boundary effects: the intermediate h is rounded on the GPU and in the reference from
    // slightly different f32 dots, so a few of its codes land on the other side of a rounding edge
    std::fprintf(stderr, "pq2_0: %s\n", mean_rel < 0.02 ? "PASS (mean within q8_1 boundary noise)" : "FAIL (mean too large)");
    return mean_rel < 0.02 ? 0 : 1;
}
