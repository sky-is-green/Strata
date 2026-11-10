// src/kernels/cpu/ptq1_0.cpp - Scion PTQ1_0 (type 143) expert rows on the CPU.
// The decode is a 1:1 port of the fork's dequantize_row_ptq1_0 (ggml-quants.c),
// which the scion packer's tests prove the inverse of; the dot is scalar and
// ISA-independent on purpose (correctness first; SIMD/GPU kernels follow).
#include "strata/kernels/cpu/ptq1_0.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"                // block_q8_K (the vendored block definitions)
#include "strata/artifact/dequant.hpp"  // fp16_to_fp32 / read_u16
#include "strata/kernels/cpu/expert_layout.hpp"  // cpu_avx2_ok

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

constexpr int kQK = 128;   // values per PTQ1_0 block
constexpr int kBB = 28;    // bytes per PTQ1_0 block
constexpr int kQ8K = 256;  // values per Q8_K activation block

// values 0..79 from qs[0..15]: element n*16 + m, trit n most significant
// first, the uint8 truncation of byte * 3^n exactly as in the C reference.
inline void decode_stage16(const uint8_t* qs, int8_t* v) {
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    for (int n = 0; n < 5; ++n) {
        for (int m = 0; m < 16; ++m) {
            const uint8_t q = (uint8_t) (qs[m] * pow3[n]);
            const int xi = ((int) q * 3) >> 8;
            v[n * 16 + m] = (int8_t) (xi - 1);
        }
    }
}

inline void decode_stage8(const uint8_t* qs, int8_t* v) {
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    for (int n = 0; n < 5; ++n) {
        for (int m = 0; m < 8; ++m) {
            const uint8_t q = (uint8_t) (qs[16 + m] * pow3[n]);
            const int xi = ((int) q * 3) >> 8;
            v[80 + n * 8 + m] = (int8_t) (xi - 1);
        }
    }
}

inline void decode_qh(const uint8_t* qh, int8_t* v) {
    static const uint8_t pow3[4] = {1, 3, 9, 27};
    for (int n = 0; n < 4; ++n) {
        for (int h = 0; h < 2; ++h) {
            const uint8_t q = (uint8_t) (qh[h] * pow3[n]);
            const int xi = ((int) q * 3) >> 8;
            v[120 + n * 2 + h] = (int8_t) (xi - 1);
        }
    }
}

// One weight block (128 trits) against the 128-value half of a Q8_K block.
inline int32_t dot128(const int8_t* v, const int8_t* aq) {
    int32_t acc = 0;
    for (int k = 0; k < kQK; ++k) acc += (int32_t) v[k] * (int32_t) aq[k];
    return acc;
}

template <int NT>
void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    float accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = 0.f;
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* b = row + (size_t) i * kBB;
        int8_t v[kQK];
        ptq1_0_decode128(b, v);
        const float dw = fp16_to_fp32(read_u16(b + 26));
        for (int t = 0; t < NT; ++t) {
            const block_q8_K& a = y[t][i / 2];
            const int8_t* aq = a.qs + (size_t) (i % 2) * kQK;
            accf[t] += dw * a.d * (float) dot128(v, aq);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = accf[t];
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
             float* const* ff, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / kQK;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int NT>
void down_rows(const uint8_t* blob, size_t d_row, int n, const void* const* hq, float* const* out,
               int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) hq[t];
    const int nb = n / kQK;
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * d_row, nb, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int NT>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int NT>
void down_rows_nt(int nt, const uint8_t* blob, size_t d_row, int n, const void* const* hq, float* const* out,
                  int r0, int r1) {
    switch (nt) {
        case 1: down_rows<1>(blob, d_row, n, hq, out, r0, r1); break;
        case 2: down_rows<2>(blob, d_row, n, hq, out, r0, r1); break;
        case 3: down_rows<3>(blob, d_row, n, hq, out, r0, r1); break;
        case 4: down_rows<4>(blob, d_row, n, hq, out, r0, r1); break;
        case 5: down_rows<5>(blob, d_row, n, hq, out, r0, r1); break;
        case 6: down_rows<6>(blob, d_row, n, hq, out, r0, r1); break;
        case 7: down_rows<7>(blob, d_row, n, hq, out, r0, r1); break;
        default: down_rows<8>(blob, d_row, n, hq, out, r0, r1); break;
    }
}

}  // namespace

void ptq1_0_decode128(const uint8_t* block, int8_t* out128) {
    decode_stage16(block, out128);
    decode_stage8(block, out128);
    decode_qh(block + 24, out128);
}

void ptq1_0_gu_rows_scalar(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                           int nt, float* const* ff, int r0, int r1) {
    gu_rows_nt<1>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
}

void ptq1_0_down_rows_scalar(const uint8_t* blob, size_t d_row, int n, const void* const* hq,
                             int nt, float* const* out, int r0, int r1) {
    down_rows_nt<1>(nt, blob, d_row, n, hq, out, r0, r1);
}

bool ptq1_0_avx2_active() {
#if defined(__x86_64__) || defined(_M_X64)
    // opt-in: the scalar rows stay the default; STRATA_PTQ1_AVX2=1 selects the AVX2 rows (bit-identical results) where cpu_avx2_ok()
    static const bool on = cpu_avx2_ok() && []{ const char* e = std::getenv("STRATA_PTQ1_AVX2"); return e != nullptr && *e != 0 && std::strcmp(e, "0") != 0; }();
    return on;
#else
    return false;
#endif
}

#if defined(__x86_64__) || defined(_M_X64)
namespace {
// STRATA_PTQ1_VERIFY=1 (measurement only, needs STRATA_PTQ1_AVX2=1): every entry-point call runs the AVX2 rows AND the scalar rows on the same real inputs
// and compares the outputs bit for bit; counts and the first mismatches go to stderr.  Runs at scalar speed.
const bool g_verify = std::getenv("STRATA_PTQ1_VERIFY") != nullptr;
std::atomic<long> g_calls{0}, g_bad{0}, g_rows{0};

void verify_report(const char* what, int nt, int n, int r0, int r1, int row, int t, float a, float b) {
    const long bad = ++g_bad;
    if (bad <= 20)
        std::fprintf(stderr, "strata ptq1_0 VERIFY MISMATCH %s: nt %d n %d rows [%d,%d) row %d token %d avx2 %.9g scalar %.9g\n",
                     what, nt, n, r0, r1, row, t, (double) a, (double) b);
}
void verify_tick() {
    const long c = ++g_calls;
    if (c % 20000 == 0)
        std::fprintf(stderr, "strata ptq1_0 VERIFY: %ld calls, %ld rows compared, %ld mismatching rows\n", c, g_rows.load(), g_bad.load());
}
}  // namespace
#endif

void ptq1_0_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                    int nt, float* const* ff, int r0, int r1) {
#if defined(__x86_64__) || defined(_M_X64)
    if (ptq1_0_avx2_active()) {
        ptq1_0_gu_rows_avx2(blob, gu_row, up_off, n, act, nt, ff, r0, r1);
        if (g_verify) {
            const int T = nt > 8 ? 8 : nt;   // the rows kernels handle at most 8 tokens
            std::vector<std::vector<float>> tmp((size_t) T, std::vector<float>((size_t) r1));
            float* tp[8];
            for (int t = 0; t < T; ++t) tp[t] = tmp[(size_t) t].data();
            ptq1_0_gu_rows_scalar(blob, gu_row, up_off, n, act, nt, tp, r0, r1);
            for (int r = r0; r < r1; ++r) {
                ++g_rows;
                for (int t = 0; t < T; ++t)
                    if (std::memcmp(&ff[t][r], &tp[t][r], 4) != 0) { verify_report("gu", nt, n, r0, r1, r, t, ff[t][r], tp[t][r]); break; }
            }
            verify_tick();
        }
        return;
    }
#endif
    ptq1_0_gu_rows_scalar(blob, gu_row, up_off, n, act, nt, ff, r0, r1);
}

void ptq1_0_down_rows(const uint8_t* blob, size_t d_row, int n, const void* const* hq,
                      int nt, float* const* out, int r0, int r1) {
#if defined(__x86_64__) || defined(_M_X64)
    if (ptq1_0_avx2_active()) {
        ptq1_0_down_rows_avx2(blob, d_row, n, hq, nt, out, r0, r1);
        if (g_verify) {
            const int T = nt > 8 ? 8 : nt;
            std::vector<std::vector<float>> tmp((size_t) T, std::vector<float>((size_t) r1));
            float* tp[8];
            for (int t = 0; t < T; ++t) tp[t] = tmp[(size_t) t].data();
            ptq1_0_down_rows_scalar(blob, d_row, n, hq, nt, tp, r0, r1);
            for (int r = r0; r < r1; ++r) {
                ++g_rows;
                for (int t = 0; t < T; ++t)
                    if (std::memcmp(&out[t][r], &tp[t][r], 4) != 0) { verify_report("down", nt, n, r0, r1, r, t, out[t][r], tp[t][r]); break; }
            }
            verify_tick();
        }
        return;
    }
#endif
    ptq1_0_down_rows_scalar(blob, d_row, n, hq, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu
