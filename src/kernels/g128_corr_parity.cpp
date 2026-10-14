// src/kernels/g128_corr_parity.cpp - the Scion releases' embedded correction factors (ggml type 43, the g128
// 34-byte block) on the correction path the q36 driver runs: quantize x to q8_1, `iq_mmvq` the [rank, n_in] A
// factor, quantize t, `iq_mmvq` the [n_out, rank] B factor, against a float reference that decodes the block
// layout with the fork's rule (per 128: fp16 scale + 2-bit codes, value = d * (code - 1)), per-chunk exactly as
// ggml-cuda's vec_dot_pq2_0_q8_1.  Real factors from the release file.
//
//     build/g128_corr_parity <35b.gguf> [layer]
//
// The reference reads the q8_1 image back from the device, so the only difference left is the accumulation
// order (the multi kernel's lane-strided sum vs a sequential one); the f16 router pairs are covered by
// lora_corr_parity's f16 reference, this checks the g128 container's decode and the two-pass chain.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/lora_corr.hpp"
#include "strata/kernels/native_mmvq.hpp"

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

static constexpr int T = 3;   // rows

static void CK(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "g128_corr: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// the device's q8_1 block: half d, half s, 32 int8 (native_mmvq.cu's Q81Block)
struct Q81 {
    uint16_t d, s;
    int8_t q[32];
};
static_assert(sizeof(Q81) == 36, "q8_1 block");

static float h16(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, 2);
    return __half2float(h);
}

/// The reference row dot: decode the row's 128-blocks (fp16 d + 2-bit codes, value d * (code - 1)) against the
/// q8_1 codes' int8 values, sum in the fork's chunk order (4 chunks of 32 per block, terms f32).
static float row_dot_ref(const uint8_t* row, int64_t n_in, const Q81* xq) {
    float acc = 0.0f;
    for (int64_t b = 0; b < n_in / 128; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        uint16_t dbits;
        std::memcpy(&dbits, blk, 2);
        const float d2 = h16(dbits);
        for (int c = 0; c < 4; ++c) {
            const Q81* xb = xq + b * 4 + c;
            const uint8_t* qs = blk + 2;
            int sumi = 0;
            for (int i = 0; i < 32; ++i) {
                const int e = c * 32 + i;
                const int code = (qs[e / 4] >> ((e % 4) * 2)) & 3;
                sumi += (code - 1) * (int) xb->q[i];
            }
            acc += d2 * h16(xb->d) * (float) sumi;
        }
    }
    return acc;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: g128_corr_parity <35b.gguf> [layer] [placement: moe|ssm|attn]\n");
        return 2;
    }
    const int64_t layer = argc > 2 ? std::atoi(argv[2]) : 0;
    const std::string placement_in = argc > 3 ? argv[3] : "moe";
    const std::string placement = placement_in == "moe" ? "ffn_moe_out" : placement_in == "ssm" ? "ssm_out" :
                                  placement_in == "attn" ? "attn_output" : placement_in;

    strata::GgufFile gguf(argv[1]);
    const std::string prefix = "blk." + std::to_string(layer) + "." + placement + ".weight";
    const std::string an = prefix + ".lora_a", bn = prefix + ".lora_b";
    const strata::TensorInfo *ta = nullptr, *tb = nullptr;
    for (const auto& t : gguf.tensors()) {
        if (t.name == an) ta = &t;
        if (t.name == bn) tb = &t;
    }
    if (!ta || !tb) {
        std::fprintf(stderr, "g128_corr: layer %lld has no %s factors\n", (long long) layer, placement.c_str());
        return 2;
    }
    if (ta->type != 43 || tb->type != 43) {
        std::fprintf(stderr, "g128_corr: factors are type %d/%d, not the g128 container (43)\n", (int) ta->type, (int) tb->type);
        return 2;
    }
    const int64_t n_in = ta->shape[0], rank = ta->shape[1], n_out = tb->shape[1];
    if (tb->shape[0] != rank) {
        std::fprintf(stderr, "g128_corr: shapes a [%lld, %lld] / b [%lld, %lld] do not pair\n", (long long) n_in,
                     (long long) rank, (long long) tb->shape[0], (long long) n_out);
        return 2;
    }
    const uint8_t* asrc = gguf.tensor_data(*ta);
    const uint8_t* bsrc = gguf.tensor_data(*tb);

    std::mt19937 rng(7 + (unsigned) layer);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) T * n_in);
    for (auto& v : x) v = nd(rng);

    const size_t a_row = (size_t) (n_in / 128) * 34, b_row = (size_t) (rank / 128) * 34;
    const size_t abytes = (size_t) rank * a_row, bbytes = (size_t) n_out * b_row;
    uint8_t *da = nullptr, *db = nullptr;
    cudaStream_t s;
    CK(cudaStreamCreate(&s), "stream");
    CK(cudaMalloc(&da, abytes), "a");
    CK(cudaMalloc(&db, bbytes), "b");
    CK(cudaMemcpy(da, asrc, abytes, cudaMemcpyHostToDevice), "a up");
    CK(cudaMemcpy(db, bsrc, bbytes, cudaMemcpyHostToDevice), "b up");
    float *dx = nullptr, *dt = nullptr, *dout = nullptr;
    void *dxq = nullptr, *dtq = nullptr;
    CK(cudaMalloc(&dx, x.size() * 4), "x");
    CK(cudaMalloc(&dt, (size_t) T * rank * 4), "t");
    CK(cudaMalloc(&dout, (size_t) T * n_out * 4), "out");
    CK(cudaMalloc(&dxq, K::native_q8_1_bytes((int) n_in, 1) * T), "xq");
    CK(cudaMalloc(&dtq, K::native_q8_1_bytes((int) rank, 1) * T), "tq");
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x up");

    K::native_quantize_q8_1(dx, dxq, (int) (n_in * T), 1, s);
    K::iq_mmvq(142, da, dxq, dt, (int) n_in, (int) rank, T, s);
    K::native_quantize_q8_1(dt, dtq, (int) (rank * T), 1, s);
    K::iq_mmvq(142, db, dtq, dout, (int) rank, (int) n_out, T, s);
    CK(cudaStreamSynchronize(s), "sync");

    std::vector<Q81> xq((size_t) T * n_in / 32), tq((size_t) T * rank / 32);
    std::vector<float> tgot((size_t) T * rank), got((size_t) T * n_out);
    CK(cudaMemcpy(xq.data(), dxq, xq.size() * sizeof(Q81), cudaMemcpyDeviceToHost), "xq down");
    CK(cudaMemcpy(tq.data(), dtq, tq.size() * sizeof(Q81), cudaMemcpyDeviceToHost), "tq down");
    CK(cudaMemcpy(tgot.data(), dt, tgot.size() * 4, cudaMemcpyDeviceToHost), "t down");
    CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost), "out down");

    // the B pass with the add folded in (the q36 driver's T=1 decode path) against the staged B output plus the
    // f32 add, bitwise: `dacc` starts at 1.0 everywhere, so the kernel's += must land on 1.0 + staged B
    float* dacc = nullptr;
    CK(cudaMalloc(&dacc, (size_t) T * n_out * 4), "acc");
    std::vector<float> ones((size_t) T * n_out, 1.0f);
    CK(cudaMemcpy(dacc, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice), "ones up");
    const size_t tcol = K::native_q8_1_bytes((int) rank, 1);
    for (int r = 0; r < T; ++r)
        K::iq_corr_g128_b(db, (const char*) dtq + (size_t) r * tcol, (int) rank, (int) n_out,
                          dacc + (size_t) r * n_out, 1, s);
    CK(cudaStreamSynchronize(s), "sync2");
    std::vector<float> acc((size_t) T * n_out);
    CK(cudaMemcpy(acc.data(), dacc, acc.size() * 4, cudaMemcpyDeviceToHost), "acc down");
    bool fused_out = true;
    for (size_t i = 0; i < acc.size() && fused_out; ++i) {
        const float expect = 1.0f + got[i];
        if (std::memcmp(&acc[i], &expect, 4) != 0) fused_out = false;
    }
    std::fprintf(stderr, "g128_corr: fused single-row B pass (add folded) vs the staged B + add: %s (bitwise)\n",
                 fused_out ? "equal" : "DIFFER");

    // the A pass with its q8_1 image fused in (the q36 T=1 path) against the staged quantize of the A output
    void* daq = nullptr;
    CK(cudaMalloc(&daq, (size_t) T * rank / 32 * sizeof(Q81)), "aq");
    for (int r = 0; r < T; ++r)
        K::iq_corr_g128_aq(da, (const char*) dxq + (size_t) r * K::native_q8_1_bytes((int) n_in, 1), (int) n_in,
                           (int) rank, (char*) daq + (size_t) r * K::native_q8_1_bytes((int) rank, 1), s);
    CK(cudaStreamSynchronize(s), "sync3");
    std::vector<Q81> aq((size_t) T * rank / 32);
    CK(cudaMemcpy(aq.data(), daq, aq.size() * sizeof(Q81), cudaMemcpyDeviceToHost), "aq down");
    const bool fused_aq = std::memcmp(aq.data(), tq.data(), aq.size() * sizeof(Q81)) == 0;
    std::fprintf(stderr, "g128_corr: fused A+q vs the staged A + quantize: %s (bitwise)\n", fused_aq ? "equal" : "DIFFER");

    // the mixed f16/g128 pair (q36's T=1 router + MoE corrections in one launch each) against the standalone
    // kernels, bitwise: the same x rows for both (the layer's h is both the MoE A input and the router input)
    const std::string ran = "blk." + std::to_string(layer) + ".ffn_gate_inp.weight.lora_a";
    const std::string rbn = "blk." + std::to_string(layer) + ".ffn_gate_inp.weight.lora_b";
    const strata::TensorInfo *tra = nullptr, *trb = nullptr;
    for (const auto& t : gguf.tensors()) {
        if (t.name == ran) tra = &t;
        if (t.name == rbn) trb = &t;
    }
    bool mixed_ok = true;
    if (tra && trb) {
        const int64_t rr = tra->shape[1], ne = trb->shape[1];
        const uint64_t nba = strata::tensor_payload_bytes(*tra), nbb = strata::tensor_payload_bytes(*trb);
        void *dra = nullptr, *drb = nullptr, *dtq_r = nullptr, *dtq_m = nullptr;
        uint16_t *dt16_r = nullptr, *dt16_m = nullptr;
        float *dout_r = nullptr, *dout_m = nullptr, *dlg_r = nullptr, *dlg_m = nullptr;
        CK(cudaMalloc(&dra, nba), "router a");
        CK(cudaMalloc(&drb, nbb), "router b");
        CK(cudaMemcpy(dra, gguf.tensor_data(*tra), nba, cudaMemcpyHostToDevice), "router a up");
        CK(cudaMemcpy(drb, gguf.tensor_data(*trb), nbb, cudaMemcpyHostToDevice), "router b up");
        CK(cudaMalloc(&dtq_r, (size_t) T * K::native_q8_1_bytes((int) rank, 1)), "tq ref");
        CK(cudaMalloc(&dtq_m, (size_t) T * K::native_q8_1_bytes((int) rank, 1)), "tq mixed");
        CK(cudaMalloc(&dt16_r, (size_t) T * rr * 2), "t16 ref");
        CK(cudaMalloc(&dt16_m, (size_t) T * rr * 2), "t16 mixed");
        CK(cudaMalloc(&dout_r, (size_t) T * n_out * 4), "moe ref");
        CK(cudaMalloc(&dout_m, (size_t) T * n_out * 4), "moe mixed");
        CK(cudaMalloc(&dlg_r, (size_t) T * ne * 4), "logits ref");
        CK(cudaMalloc(&dlg_m, (size_t) T * ne * 4), "logits mixed");
        CK(cudaMemset(dlg_r, 0, (size_t) T * ne * 4), "logits ref zero");
        CK(cudaMemset(dlg_m, 0, (size_t) T * ne * 4), "logits mixed zero");
        for (int r = 0; r < T; ++r) {
            const char* x = (const char*) dx + (size_t) r * n_in * 4;
            const char* xq_r = (const char*) dxq + (size_t) r * K::native_q8_1_bytes((int) n_in, 1);
            const int n_router = (int) tra->shape[0];   // the router's own input width (N), not the placement's
            K::iq_corr_g128_aq(da, xq_r, (int) n_in, (int) rank, (char*) dtq_r + (size_t) r * K::native_q8_1_bytes((int) rank, 1), s);
            K::iq_corr_g128_b(db, (char*) dtq_r + (size_t) r * K::native_q8_1_bytes((int) rank, 1), (int) rank, (int) n_out,
                              dout_r + (size_t) r * n_out, 0, s);
            strata::kernels::lora_corr_router((const uint16_t*) dra, (const uint16_t*) drb, (const float*) x, dt16_r + (size_t) r * rr,
                                              dlg_r + (size_t) r * ne, n_router, rr, ne, s);
            K::iq_corr_mixed_a(da, xq_r, (int) n_in, (int) rank,
                               (char*) dtq_m + (size_t) r * K::native_q8_1_bytes((int) rank, 1), dra, (const float*) x,
                               n_router, (int) rr, dt16_m + (size_t) r * rr, s);
            K::iq_corr_mixed_b(db, (char*) dtq_m + (size_t) r * K::native_q8_1_bytes((int) rank, 1), (int) rank, (int) n_out,
                               dout_m + (size_t) r * n_out, drb, dt16_m + (size_t) r * rr, (int) rr, (int) ne,
                               dlg_m + (size_t) r * ne, s);
        }
        CK(cudaStreamSynchronize(s), "sync mixed");
        std::vector<float> mo(T * n_out), mm(T * n_out), lg(T * ne), lm(T * ne);
        CK(cudaMemcpy(mo.data(), dout_r, mo.size() * 4, cudaMemcpyDeviceToHost), "moe ref down");
        CK(cudaMemcpy(mm.data(), dout_m, mm.size() * 4, cudaMemcpyDeviceToHost), "moe mixed down");
        CK(cudaMemcpy(lg.data(), dlg_r, lg.size() * 4, cudaMemcpyDeviceToHost), "logits ref down");
        CK(cudaMemcpy(lm.data(), dlg_m, lm.size() * 4, cudaMemcpyDeviceToHost), "logits mixed down");
        mixed_ok = std::memcmp(mo.data(), mm.data(), mo.size() * 4) == 0 &&
                   std::memcmp(lg.data(), lm.data(), lg.size() * 4) == 0;
        std::fprintf(stderr, "g128_corr: mixed pair vs the standalone kernels (MoE + router): %s (bitwise)\n",
                     mixed_ok ? "equal" : "DIFFER");
    } else {
        std::fprintf(stderr, "g128_corr: no f16 router factors on layer %lld; mixed pair not checked\n", (long long) layer);
    }

    double errsum = 0, denom = 0, worst = 0;
    double a_errsum = 0, a_denom = 0, a_worst = 0;
    for (int r = 0; r < T; ++r) {
        for (int64_t o = 0; o < rank; ++o) {   // the A pass itself
            const float ref = row_dot_ref(asrc + (size_t) o * (n_in / 128) * 34, n_in,
                                          xq.data() + (size_t) r * n_in / 32);
            const double err = std::fabs((double) tgot[(size_t) r * rank + o] - ref);
            a_errsum += err;
            a_denom += std::fabs((double) ref);
            if (err > a_worst) a_worst = err;
        }
        for (int64_t o = 0; o < n_out; ++o) {   // the B pass on the A pass's own q8_1 image
            const float ref = row_dot_ref(bsrc + (size_t) o * (rank / 128) * 34, rank,
                                          tq.data() + (size_t) r * rank / 32);
            const float gotv = got[(size_t) r * n_out + o];
            const double err = std::fabs((double) gotv - ref);
            errsum += err;
            denom += std::fabs((double) ref);
            if (err > worst) worst = err;
        }
    }
    const double mean_rel = errsum / (denom + 1e-30);
    const double max_rel = worst / (denom / (double) (T * n_out) + 1e-30);
    const double a_mean_rel = a_errsum / (a_denom + 1e-30);
    const double a_max_rel = a_worst / (a_denom / (double) (T * rank) + 1e-30);
    std::fprintf(stderr,
                 "g128_corr: layer %lld %s a [%lld, %lld] b [%lld, %lld], %d rows: A mean |gpu - ref| / mean |ref| "
                 "%.4g, max %.4g; B %.4g, max %.4g\n",
                 (long long) layer, placement.c_str(), (long long) n_in, (long long) rank, (long long) rank,
                 (long long) n_out, T, a_mean_rel, a_max_rel, mean_rel, max_rel);
    const bool pass = a_mean_rel < 1e-5 && mean_rel < 1e-5 && a_max_rel < 1e-4 && max_rel < 1e-4 && fused_out && fused_aq &&
                      mixed_ok;
    std::fprintf(stderr, "g128_corr: %s (f32 summation-order noise only; fused passes bitwise)\n",
                 pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
