// Bit-parity of the dense decode pass's fused kernels against the launches they replace:
//   dense_split_q + rms_norm_weighted(qcur)  vs  dense_split_q_rms
//   dense_rms_norm (+ add) + dense_fwht_quantize_rows  vs  dense_rms_fwht_quant_rows
// Both write their outputs to separate device buffers and compare all bytes on the host.  Run with no arguments;
// exits nonzero on the first mismatch.
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
int g_fail = 0;
cudaStream_t g_st = nullptr;    // native_quantize_q8_1 demands an explicit stream; every call in a case uses it

void check(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    size_t nd = 0, first = a.size();
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) { if (first == a.size()) first = i; ++nd; }
    if (nd) {
        std::printf("FAIL %s: %zu of %zu differ, first at %zu: %a vs %a\n", what, nd, a.size(), first,
                    (double) a[first], (double) b[first]);
        ++g_fail;
    } else {
        std::printf("ok   %s (%zu values)\n", what, a.size());
    }
}

void check_bytes(const char* what, const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
    size_t nd = 0, first = a.size();
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) { if (first == a.size()) first = i; ++nd; }
    if (nd) {
        std::printf("FAIL %s: %zu of %zu bytes differ, first at %zu\n", what, nd, a.size(), first);
        ++g_fail;
    } else {
        std::printf("ok   %s (%zu bytes)\n", what, a.size());
    }
}

void qrms_case(int rows, int D, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    std::vector<float> h((size_t) rows * 2 * D), w(D);
    for (float& v : h) v = d(rng);
    for (float& v : w) v = d(rng);
    float *qf = nullptr, *qc1 = nullptr, *qc2 = nullptr, *wf = nullptr;
    cudaMalloc(&qf, h.size() * 4);
    cudaMalloc(&qc1, (size_t) rows * D * 4);
    cudaMalloc(&qc2, (size_t) rows * D * 4);
    cudaMalloc(&wf, (size_t) D * 4);
    cudaMemcpy(qf, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(wf, w.data(), (size_t) D * 4, cudaMemcpyHostToDevice);
    strata::kernels::dense_split_q(qf, qc1, rows, D, nullptr);
    strata::kernels::rms_norm_weighted(qc1, wf, rows, D, 1e-6f, nullptr);
    strata::kernels::dense_split_q_rms(qf, wf, qc2, rows, D, 1e-6f, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> a((size_t) rows * D), b((size_t) rows * D);
    cudaMemcpy(a.data(), qc1, a.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), qc2, b.size() * 4, cudaMemcpyDeviceToHost);
    char what[96];
    std::snprintf(what, sizeof what, "split_q_rms rows=%d D=%d", rows, D);
    check(what, a, b);
}

void rmsfq_case(int rows, int width, bool with_add, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    std::vector<float> hx((size_t) rows * width), hw((size_t) width), hs((size_t) width), ha((size_t) rows * width);
    for (float& v : hx) v = d(rng);
    for (float& v : ha) v = d(rng) * 0.5f;
    for (float& v : hw) v = 0.5f + d(rng) * 0.1f;
    for (float& v : hs) v = (rng() & 1) ? 1.0f : -1.0f;
    float *x = nullptr, *add = nullptr, *w = nullptr, *s = nullptr, *xn1 = nullptr, *xn2 = nullptr;
    void *q1 = nullptr, *q2 = nullptr;
    const size_t qb = strata::kernels::native_q8_1_bytes(width, rows);
    cudaMalloc(&x, hx.size() * 4);
    cudaMalloc(&add, ha.size() * 4);
    cudaMalloc(&w, (size_t) width * 4);
    cudaMalloc(&s, (size_t) width * 4);
    cudaMalloc(&xn1, hx.size() * 4);
    cudaMalloc(&xn2, hx.size() * 4);
    cudaMalloc(&q1, qb);
    cudaMalloc(&q2, qb);
    cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(add, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), (size_t) width * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(s, hs.data(), (size_t) width * 4, cudaMemcpyHostToDevice);

    if (with_add) strata::kernels::add_inplace(x, add, (int64_t) rows * width, nullptr);
    strata::kernels::dense_rms_norm(x, w, xn1, rows, width, 1e-6f, nullptr);
    strata::kernels::dense_fwht_quantize_rows(xn1, q1, nullptr, s, width, rows, 0, 0, 0, nullptr);
    if (with_add) cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);   // same pre-add state
    strata::kernels::dense_rms_fwht_quant_rows(x, with_add ? add : nullptr, w, xn2, q2, s, width, rows, 1e-6f, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> a((size_t) rows * width), b((size_t) rows * width);
    cudaMemcpy(a.data(), xn1, a.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), xn2, b.size() * 4, cudaMemcpyDeviceToHost);
    char what[96];
    std::snprintf(what, sizeof what, "rms_fwht_quant rows=%d width=%d%s", rows, width, with_add ? " + add" : "");
    check(what, a, b);
    std::vector<unsigned char> qa(qb), qbb(qb);
    cudaMemcpy(qa.data(), q1, qb, cudaMemcpyDeviceToHost);
    cudaMemcpy(qbb.data(), q2, qb, cudaMemcpyDeviceToHost);
    std::snprintf(what, sizeof what, "rms_fwht_quant q8_1 rows=%d width=%d%s", rows, width, with_add ? " + add" : "");
    check_bytes(what, qa, qbb);
}

// dense_fwht_rows (with the grouped GDN V gather when hd != 0) + native_quantize_q8_1 vs dense_fwht_quantize_rows;
// `rot_out` and the plain transform's output are compared element for element, so the butterfly bits are covered.
void fwhtq_case(int rows, int width, int hd, int nk, int rep, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    std::vector<float> hx((size_t) rows * width), hs((size_t) width);
    for (float& v : hx) v = d(rng);
    for (float& v : hs) v = (rng() & 1) ? 1.0f : -1.0f;
    float *x = nullptr, *y = nullptr, *xn = nullptr, *rot = nullptr, *s = nullptr;
    void *q1 = nullptr, *q2 = nullptr;
    const size_t qb = strata::kernels::native_q8_1_bytes(width, rows);
    cudaMalloc(&x, hx.size() * 4);
    cudaMalloc(&y, hx.size() * 4);
    cudaMalloc(&xn, hx.size() * 4);
    cudaMalloc(&rot, hx.size() * 4);
    cudaMalloc(&s, (size_t) width * 4);
    cudaMalloc(&q1, qb);
    cudaMalloc(&q2, qb);
    cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(s, hs.data(), (size_t) width * 4, cudaMemcpyHostToDevice);

    const float* cur = x;
    if (hd) { strata::kernels::dense_permute_gdn_v(x, y, width, rows, hd, nk, rep, g_st); cur = y; }
    strata::kernels::dense_fwht_rows(cur, xn, s, width, rows, false, g_st);
    strata::kernels::native_quantize_q8_1(xn, q1, width, rows, g_st);
    strata::kernels::dense_fwht_quantize_rows(x, q2, rot, s, width, rows, hd, nk, rep, g_st);
    cudaDeviceSynchronize();
    std::vector<float> ra((size_t) rows * width), rb((size_t) rows * width);
    cudaMemcpy(ra.data(), xn, ra.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(rb.data(), rot, rb.size() * 4, cudaMemcpyDeviceToHost);
    char what[128];
    std::snprintf(what, sizeof what, "fwht_quant rot rows=%d width=%d%s", rows, width, hd ? " + gdn gather" : "");
    check(what, ra, rb);
    std::vector<unsigned char> qa(qb), qbb(qb);
    cudaMemcpy(qa.data(), q1, qb, cudaMemcpyDeviceToHost);
    cudaMemcpy(qbb.data(), q2, qb, cudaMemcpyDeviceToHost);
    std::snprintf(what, sizeof what, "fwht_quant q8_1 rows=%d width=%d%s", rows, width, hd ? " + gdn gather" : "");
    check_bytes(what, qa, qbb);
}

// dense_swiglu + dense_fwht_quantize_rows (or native_quantize_q8_1 without signs) vs dense_swiglu_quantize_rows;
// the fused kernel's rot_out is compared with the pair's transformed row too.
void swigluq_case(int rows, int width, bool with_signs, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-4.0f, 4.0f);
    std::vector<float> hg((size_t) rows * width), hu((size_t) rows * width), hs((size_t) width);
    for (float& v : hg) v = d(rng);
    for (float& v : hu) v = d(rng);
    for (float& v : hs) v = (rng() & 1) ? 1.0f : -1.0f;
    float *g = nullptr, *u = nullptr, *h = nullptr, *rot1 = nullptr, *rot2 = nullptr, *s = nullptr;
    void *q1 = nullptr, *q2 = nullptr;
    const size_t qb = strata::kernels::native_q8_1_bytes(width, rows);
    cudaMalloc(&g, hg.size() * 4);
    cudaMalloc(&u, hu.size() * 4);
    cudaMalloc(&h, hg.size() * 4);
    cudaMalloc(&rot1, with_signs ? hg.size() * 4 : 4);
    cudaMalloc(&rot2, with_signs ? hg.size() * 4 : 4);
    cudaMalloc(&s, (size_t) width * 4);
    cudaMalloc(&q1, qb);
    cudaMalloc(&q2, qb);
    cudaMemcpy(g, hg.data(), hg.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(u, hu.data(), hu.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(s, hs.data(), (size_t) width * 4, cudaMemcpyHostToDevice);

    strata::kernels::dense_swiglu(g, u, h, (int64_t) rows * width, g_st);
    if (with_signs) strata::kernels::dense_fwht_quantize_rows(h, q1, rot1, s, width, rows, 0, 0, 0, g_st);
    else strata::kernels::native_quantize_q8_1(h, q1, width, rows, g_st);
    strata::kernels::dense_swiglu_quantize_rows(g, u, q2, with_signs ? s : nullptr, width, rows, nullptr,
                                               with_signs ? rot2 : nullptr, g_st);
    cudaDeviceSynchronize();
    char what[128];
    if (with_signs) {
        std::vector<float> ra((size_t) rows * width), rb((size_t) rows * width);
        cudaMemcpy(ra.data(), rot1, ra.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(rb.data(), rot2, rb.size() * 4, cudaMemcpyDeviceToHost);
        std::snprintf(what, sizeof what, "swiglu_quant rot rows=%d width=%d", rows, width);
        check(what, ra, rb);
    }
    std::vector<unsigned char> qa(qb), qbb(qb);
    cudaMemcpy(qa.data(), q1, qb, cudaMemcpyDeviceToHost);
    cudaMemcpy(qbb.data(), q2, qb, cudaMemcpyDeviceToHost);
    std::snprintf(what, sizeof what, "swiglu_quant q8_1 rows=%d width=%d%s", rows, width, with_signs ? " + rotation" : "");
    check_bytes(what, qa, qbb);
}
}  // namespace

int main() {
    cudaStreamCreate(&g_st);
    std::mt19937 rng(1234);
    qrms_case(24, 256, rng);
    qrms_case(4, 256, rng);
    qrms_case(8, 128, rng);
    rmsfq_case(1, 5120, false, rng);
    rmsfq_case(2, 5120, false, rng);
    rmsfq_case(1, 5120, true, rng);
    rmsfq_case(1, 6144, false, rng);
    rmsfq_case(3, 6144, true, rng);
    rmsfq_case(1, 17408, false, rng);
    rmsfq_case(2, 17408, true, rng);
    fwhtq_case(1, 5120, 0, 0, 0, rng);
    fwhtq_case(2, 6144, 0, 0, 0, rng);
    fwhtq_case(1, 6144, 128, 16, 3, rng);      // the GDN V gather geometry of this model
    fwhtq_case(1, 1024, 128, 4, 2, rng);
    swigluq_case(1, 17408, true, rng);
    swigluq_case(2, 17408, true, rng);
    swigluq_case(1, 17408, false, rng);
    if (g_fail) { std::printf("%d case(s) FAILED\n", g_fail); return 1; }
    std::printf("all dense fusion parities identical\n");
    return 0;
}
