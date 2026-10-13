// Real-artifact PTQ1_0 expert rows, two paths:
//
//  (CPU) native_fmt -> native_gu_rows / native_down_rows against a float reference computed from the same GGUF
//        bytes (trit decode * fp16 d, dotted with the dequantized Q8_K activation).  Validates the geometry
//        (row sizes, blob offsets), the decode, and the scale handling.
//  (GPU) the same blob through native_expert_supported / native_expert_layout, native_expert_grouped (both the
//        decode-once and the per-entry kernels, bitwise equal to each other), iq_mmvq, and the prompt-path
//        dequantizers iq_dequant_f16/_f32/_gu_f16 against the same float reference with the device's own q8_1
//        activations (the pattern of native_expert_parity).
#define NOMINMAX
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/ptq1_0.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace sk = strata::kernels;

// Q8_K block: { float d; int8_t qs[256]; int16_t bsums[16]; } = 292 bytes.
float q8k_value(const uint8_t* act, int64_t idx) {
    const uint8_t* b = act + (idx / 256) * 292;
    float d;
    std::memcpy(&d, b, 4);
    const int8_t q = (int8_t) b[4 + (idx % 256)];
    return d * (float) q;
}

// q8_1 block: { ggml_half2 ds; int8_t qs[32]; } = 36 bytes.
float q81_value(const uint8_t* act, int64_t idx) {
    const uint8_t* b = act + (idx / 32) * 36;
    return strata::fp16_to_fp32(strata::read_u16(b)) * (float) (int8_t) b[4 + (idx % 32)];
}

float row_dot_ref(const uint8_t* row, const float* act, int64_t n) {
    // float reference: per 128-block, trits * d, dotted with the float activation
    const int64_t nb = n / 128;
    float acc = 0.f;
    int8_t v[128];
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t* blk = row + b * 28;
        strata::kernels::cpu::ptq1_0_decode128(blk, v);
        const float d = strata::fp16_to_fp32(strata::read_u16(blk + 26));
        for (int k = 0; k < 128; ++k) acc += (float) v[k] * d * act[b * 128 + k];
    }
    return acc;
}

// The same decode, one row to floats (for the dequantizer and expert references).
std::vector<float> row_dequant_ref(const uint8_t* row, int64_t n) {
    const int64_t nb = n / 128;
    std::vector<float> out((size_t) n);
    int8_t v[128];
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t* blk = row + b * 28;
        strata::kernels::cpu::ptq1_0_decode128(blk, v);
        const float d = strata::fp16_to_fp32(strata::read_u16(blk + 26));
        for (int k = 0; k < 128; ++k) out[(size_t) (b * 128 + k)] = d * (float) v[k];
    }
    return out;
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ptq1_0_parity <gguf-with-ptq1_0-experts>\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    const auto* gate = gguf.find("blk.0.ffn_gate_exps.weight");
    const auto* up = gguf.find("blk.0.ffn_up_exps.weight");
    const auto* down = gguf.find("blk.0.ffn_down_exps.weight");
    if (!gate || !up || !down || gate->type != 143 || up->type != 143 || down->type != 143 ||
        gate->shape.size() != 3 || gate->shape[0] != 2560 || gate->shape[1] != 640) {
        std::fprintf(stderr, "expected PTQ1_0 expert tensors [2560, 640, E]\n");
        return 2;
    }
    const int64_t H = (int64_t) gate->shape[0], F = (int64_t) gate->shape[1];
    const int64_t gu_row = H / 128 * 28, d_row = F / 128 * 28;
    const uint8_t* gb = gguf.tensor_data(*gate);
    const uint8_t* ub = gguf.tensor_data(*up);
    const uint8_t* db = gguf.tensor_data(*down);

    // one expert's blob, the pack's layout: [gate F rows | up F rows | down H rows]
    std::vector<uint8_t> blob((size_t) (2 * F * gu_row + H * d_row));
    std::memcpy(blob.data(), gb, (size_t) F * gu_row);
    std::memcpy(blob.data() + F * gu_row, ub, (size_t) F * gu_row);
    std::memcpy(blob.data() + 2 * F * gu_row, db, (size_t) H * d_row);

    strata::kernels::cpu::NativeFmt f;
    std::string err;
    if (!strata::kernels::cpu::native_fmt(143, 143, H, F, f, err)) {
        std::fprintf(stderr, "native_fmt: %s\n", err.c_str());
        return 1;
    }
    if (f.gu_row != (size_t) gu_row || f.d_row != (size_t) d_row || f.bytes != blob.size()) {
        std::fprintf(stderr, "geometry mismatch: gu_row %zu/%lld d_row %zu/%lld bytes %zu/%zu\n",
                     f.gu_row, (long long) gu_row, f.d_row, (long long) d_row, f.bytes, blob.size());
        return 1;
    }

    // activation for gate/up: x in R^H
    std::vector<float> x((size_t) H);
    for (int64_t i = 0; i < H; ++i) x[i] = 0.05f * std::sin(0.11f * (float) i) + 0.01f * (float) ((i * 37) % 7 - 3);
    const auto* tqc = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (tqc == nullptr || tqc->from_float == nullptr) {
        std::fprintf(stderr, "ggml-cpu has no Q8_K from_float\n");
        return 3;
    }
    std::vector<uint8_t> act(strata::kernels::cpu::kNativeActBytes);
    strata::kernels::cpu::native_quant_act(f, x.data(), act.data());
    std::vector<float> ax((size_t) H);
    for (int64_t i = 0; i < H; ++i) ax[(size_t) i] = q8k_value(act.data(), i);

    const int64_t r0 = 0, r1 = 5;
    std::vector<float> ff((size_t) (r1 - r0));
    float* ffp[1] = {ff.data()};
    const void* actp[1] = {act.data()};
    strata::kernels::cpu::native_gu_rows(f, blob.data(), actp, 1, ffp, (int) r0, (int) r1);

    double max_g = 0.0;
    for (int64_t r = r0; r < r1; ++r) {
        const float g = row_dot_ref(blob.data() + (size_t) r * gu_row, ax.data(), H);
        const float u = row_dot_ref(blob.data() + F * gu_row + (size_t) r * gu_row, ax.data(), H);
        const float want = (g / (1.f + std::exp(-g))) * u;
        max_g = std::max(max_g, (double) std::fabs(ff[(size_t) (r - r0)] - want));
    }

    // down: activation in R^F, rows over H
    std::vector<float> h((size_t) F);
    for (int64_t i = 0; i < F; ++i) h[i] = 0.03f * std::cos(0.07f * (float) i);
    std::vector<uint8_t> hq(strata::kernels::cpu::kNativeHBytes);
    strata::kernels::cpu::native_quant_h(f, h.data(), hq.data());
    std::vector<float> ah((size_t) F);
    for (int64_t i = 0; i < F; ++i) ah[(size_t) i] = q8k_value(hq.data(), i);
    std::vector<float> out((size_t) (r1 - r0));
    float* outp[1] = {out.data()};
    const void* hqp[1] = {hq.data()};
    strata::kernels::cpu::native_down_rows(f, blob.data(), hqp, 1, outp, (int) r0, (int) r1);

    double max_d = 0.0;
    for (int64_t r = r0; r < r1; ++r) {
        const float want = row_dot_ref(blob.data() + 2 * F * gu_row + (size_t) r * d_row, ah.data(), F);
        max_d = std::max(max_d, (double) std::fabs(out[(size_t) (r - r0)] - want));
    }

    std::printf("PTQ1_0 experts: gu max|err| %.3e, down max|err| %.3e %s\n", max_g, max_d,
                (max_g <= 1e-3 && max_d <= 1e-3) ? "PASS" : "FAIL");
    int failures = (max_g <= 1e-3 && max_d <= 1e-3) ? 0 : 1;

    // ---------------------------------------------------------------- (GPU) the same blob on the device
    // NT tokens through one group; the reference follows the device's own q8_1 activations, and the down
    // reference reads the device-quantized h out of the grouped kernels' scratch (its layout is the documented
    // native_expert_scratch_bytes contract), so each stage is compared against exactly what the next kernel saw.
    constexpr int NT = 3;
    if (!sk::native_expert_supported(143, 143, H, F)) {
        std::printf("GPU: native_expert_supported(143, 143, 2560, 640) is false\n");
        return 1;
    }
    const sk::NativeExpertLayout L = sk::native_expert_layout(143, 143, H, F);
    if (L.gu_row != f.gu_row || L.d_row != f.d_row || L.up_off != f.up_off || L.down_off != f.down_off ||
        L.bytes != f.bytes || sk::iq_row_bytes(143, H) != (size_t) gu_row ||
        sk::iq_row_bytes(143, F) != (size_t) d_row) {
        std::printf("GPU: native_expert_layout / iq_row_bytes mismatch\n");
        return 1;
    }

    std::vector<float> xs((size_t) NT * H);
    for (int k = 0; k < NT; ++k)
        for (int64_t i = 0; i < H; ++i)
            xs[(size_t) k * H + i] =
                0.05f * std::sin(0.11f * (float) i + 0.7f * (float) k) + 0.01f * (float) ((i * 37 + k) % 7 - 3);

    cudaStream_t s;
    cudaStreamCreate(&s);
    uint8_t* dblob = nullptr;
    float* dx = nullptr;
    uint8_t* dxq = nullptr;
    void* dscr = nullptr;
    float* dout = nullptr;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    cudaMalloc((void**) &dblob, blob.size());
    cudaMalloc((void**) &dx, xs.size() * 4);
    cudaMalloc((void**) &dxq, (size_t) NT * H / 32 * 36);
    const size_t scr_bytes = sk::native_expert_scratch_bytes(NT, F);
    cudaMalloc(&dscr, scr_bytes);
    cudaMalloc((void**) &dout, (size_t) NT * H * 4);
    cudaMalloc((void**) &dptr, 8);
    cudaMalloc((void**) &dstart, 8);
    cudaMalloc((void**) &dn, 4);
    cudaMalloc((void**) &ddst, NT * 4);
    cudaMalloc((void**) &dtok, NT * 4);
    cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice);
    const unsigned long long p = (unsigned long long) dblob;
    const int32_t st[2] = {0, NT}, one = 1, idx[NT] = {0, 1, 2};
    cudaMemcpy(dptr, &p, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dstart, st, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dn, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ddst, idx, NT * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtok, idx, NT * 4, cudaMemcpyHostToDevice);
    sk::quantize_q8_1_rows((const float*) dx, NT, H, dxq, s);
    std::vector<uint8_t> xq((size_t) NT * H / 32 * 36);
    cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
    sk::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
    cudaStreamSynchronize(s);
    std::vector<float> got((size_t) NT * H);
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);

    {   // the decode-once kernels vs the per-entry ones, bit for bit (the Split<143> contract)
        std::vector<float> old_g(got.size());
        const bool was = sk::iq_old_kernels();
        sk::iq_set_old_kernels(!was);
        sk::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
        sk::iq_set_old_kernels(was);
        cudaStreamSynchronize(s);
        cudaMemcpy(old_g.data(), dout, old_g.size() * 4, cudaMemcpyDeviceToHost);
        const bool same = std::memcmp(old_g.data(), got.data(), got.size() * 4) == 0;
        std::printf("GPU: decode-once vs per-entry expert kernels: %s\n", same ? "bitwise equal" : "MISMATCH");
        if (!same) ++failures;
    }

    {   // the reference: full expert per token, over the device's own quantized activations
        std::vector<float> G((size_t) F * H), U((size_t) F * H), D((size_t) H * F);
        for (int64_t r = 0; r < F; ++r) {
            const auto g = row_dequant_ref(blob.data() + (size_t) r * gu_row, H);
            const auto u = row_dequant_ref(blob.data() + F * gu_row + (size_t) r * gu_row, H);
            std::memcpy(G.data() + (size_t) r * H, g.data(), (size_t) H * 4);
            std::memcpy(U.data() + (size_t) r * H, u.data(), (size_t) H * 4);
        }
        for (int64_t r = 0; r < H; ++r) {
            const auto d = row_dequant_ref(blob.data() + 2 * F * gu_row + (size_t) r * d_row, F);
            std::memcpy(D.data() + (size_t) r * F, d.data(), (size_t) F * 4);
        }
        // the scratch layout: gate (0), up (fa), h (2fa), hq (3fa).  The default path fuses SwiGLU + q8_1
        // (swiglu_q8_1_entries_kernel writes hq only), so the f32 h image only exists on the v1 path: run
        // that once for the h comparison, then the default once for the quantized h the down kernel read.
        const size_t fa = ((size_t) NT * F * 4 + 255) & ~(size_t) 255;
        std::vector<float> hdev((size_t) NT * F);
        std::vector<uint8_t> hdevq((size_t) NT * (F / 32) * 36);
        sk::native_grouped_set_v1(true);
        sk::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(hdev.data(), (const uint8_t*) dscr + 2 * fa, hdev.size() * 4, cudaMemcpyDeviceToHost);
        sk::native_grouped_set_v1(false);
        sk::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(hdevq.data(), (const uint8_t*) dscr + 3 * fa, hdevq.size(), cudaMemcpyDeviceToHost);
        std::vector<float> want_h((size_t) NT * F), want((size_t) NT * H);
        for (int k = 0; k < NT; ++k) {
            std::vector<float> xa((size_t) H);
            for (int64_t i = 0; i < H; ++i) xa[(size_t) i] = q81_value(xq.data(), (int64_t) k * H + i);
            for (int64_t r = 0; r < F; ++r) {
                double g = 0, u = 0;
                for (int64_t i = 0; i < H; ++i) {
                    g += (double) G[(size_t) (r * H + i)] * xa[(size_t) i];
                    u += (double) U[(size_t) (r * H + i)] * xa[(size_t) i];
                }
                want_h[(size_t) (k * F + r)] = (float) (g / (1.0 + std::exp(-g)) * u);
            }
            for (int64_t r = 0; r < H; ++r) {
                double o = 0;
                for (int64_t i = 0; i < F; ++i) o += (double) D[(size_t) (r * F + i)] * q81_value(hdevq.data(), (int64_t) k * F + i);
                want[(size_t) (k * H + r)] = (float) o;
            }
        }
        const double rh = rel(hdev, want_h), rd = rel(got, want);
        std::printf("GPU: swiglu h vs reference rel %.2e; expert out vs reference (device q8_1 h): rel %.2e %s\n", rh, rd,
                    (rh < 1e-3 && rd < 1e-3) ? "" : "MISMATCH");
        if (!(rh < 1e-3) || !(rd < 1e-3)) ++failures;
    }

    {   // iq_mmvq: the gate rows against token 0's q8_1 activation
        float* dy = nullptr;
        cudaMalloc((void**) &dy, (size_t) F * 4);
        sk::iq_mmvq(143, dblob, dxq, dy, (int) H, (int) F, 1, s);
        cudaStreamSynchronize(s);
        std::vector<float> m0((size_t) F);
        cudaMemcpy(m0.data(), dy, m0.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(dy);
        std::vector<float> xa((size_t) H), want((size_t) F);
        for (int64_t i = 0; i < H; ++i) xa[(size_t) i] = q81_value(xq.data(), i);
        for (int64_t r = 0; r < F; ++r) {
            const auto g = row_dequant_ref(blob.data() + (size_t) r * gu_row, H);
            double acc = 0;
            for (int64_t i = 0; i < H; ++i) acc += (double) g[(size_t) i] * xa[(size_t) i];
            want[(size_t) r] = (float) acc;
        }
        const double rq = rel(m0, want);
        std::printf("GPU: iq_mmvq gate rows vs reference: rel %.2e %s\n", rq, rq < 1e-3 ? "" : "MISMATCH");
        if (!(rq < 1e-3)) ++failures;
    }

    {   // the prompt-path dequantizers against the same decode (iq_dequant_* = to_float), bit for bit
        std::vector<float> G((size_t) F * H), U((size_t) F * H), Gw((size_t) F * H);
        for (int64_t r = 0; r < F; ++r) {
            const auto g = row_dequant_ref(blob.data() + (size_t) r * gu_row, H);
            const auto u = row_dequant_ref(blob.data() + F * gu_row + (size_t) r * gu_row, H);
            std::memcpy(G.data() + (size_t) r * H, g.data(), (size_t) H * 4);
            std::memcpy(U.data() + (size_t) r * H, u.data(), (size_t) H * 4);
        }
        float* dqf = nullptr;
        uint16_t* dhf = nullptr;
        uint16_t* dgu = nullptr;
        cudaMalloc((void**) &dqf, (size_t) F * H * 4);
        cudaMalloc((void**) &dhf, (size_t) F * H * 2);
        cudaMalloc((void**) &dgu, (size_t) 2 * F * H * 2);
        sk::iq_dequant_f32(143, dblob, F * H, dqf, s);
        sk::iq_dequant_f16(143, dblob, F * H, dhf, s);
        sk::iq_dequant_gu_f16(143, dblob, (const uint8_t*) dblob + L.up_off, F, H, dgu, s);
        cudaStreamSynchronize(s);
        std::vector<float> gotf((size_t) F * H);
        std::vector<uint16_t> goth((size_t) F * H), gotgu((size_t) 2 * F * H);
        cudaMemcpy(gotf.data(), dqf, gotf.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(goth.data(), dhf, goth.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(gotgu.data(), dgu, gotgu.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(dqf);
        cudaFree(dhf);
        cudaFree(dgu);
        std::memcpy(Gw.data(), gotf.data(), Gw.size() * 4);
        const double rd = rel(Gw, G);
        size_t differ = 0, h_differ = 0, gu_differ = 0;
        for (size_t i = 0; i < gotf.size(); ++i) {
            differ += std::memcmp(&gotf[i], &G[i], 4) != 0;
            h_differ += goth[i] != ggml_fp32_to_fp16(G[i]);
        }
        for (int64_t r = 0; r < F; ++r) {
            for (int64_t i = 0; i < H; ++i) {
                gu_differ += gotgu[(size_t) ((2 * r) * H + i)] != ggml_fp32_to_fp16(G[(size_t) (r * H + i)]);
                gu_differ += gotgu[(size_t) ((2 * r + 1) * H + i)] != ggml_fp32_to_fp16(U[(size_t) (r * H + i)]);
            }
        }
        std::printf("GPU: dequant f32 rel %.2e (%zu of %lld differ), f16 %zu differ, gate/up fp16 rows %zu differ\n",
                    rd, differ, (long long) (F * H), h_differ, gu_differ);
        if (!(rd < 1e-7) || differ || h_differ || gu_differ) ++failures;
    }

    cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
    cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    cudaStreamDestroy(s);

    std::printf("PTQ1_0 parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
