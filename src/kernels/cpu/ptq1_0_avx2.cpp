// src/kernels/cpu/ptq1_0_avx2.cpp - AVX2 rows for Scion PTQ1_0 (type 143) experts on the CPU.
//
// The trit unpack and the integer dot follow the AVX2 path of ggml_vec_dot_ptq1_0_q8_0 in prism-ml-llama.cpp
// (scion-main 4819a262e / kTrain 30bc02218): each 256-bit vector holds one 32-value run of the block (one source
// run times one power of 3, one in-lane shuffle, no cross-lane permute); sum((t-1)*y) = sum(t*y) - sum(y) with
// vpmaddubsw.  Differences to that kernel: the activation is Q8_K (256 values, ONE scale; a weight block of 128
// values is one half of it), and the float arithmetic is the scalar kernel's line for line.  The integer sum of a
// block is exact, so with -ffp-contract=off (set in CMakeLists.txt) the result is BIT-IDENTICAL to ptq1_0.cpp's
// scalar rows; ptq1_0_avx2_parity checks that.  One decoded weight block serves all NT activations of a window.
// v2 (2026-11-10): (a) sum(y) of a block half comes from Q8_K's bsums instead of a second maddubs, (b) the four maddubs results are
// added in int16 before one madd, (c) 8 rows at a time: the eight int32 sums of a block are transposed/reduced with hadd into one
// vector whose lanes are the rows, so the float line (scale * scale * dot, then += per block, in block order) runs once per 8 rows
// with the same IEEE operations lane by lane; vcvtph2ps is exact like fp16_to_fp32.  Rows past the last full group use the 1-row path.
#include "strata/kernels/cpu/ptq1_0.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"                // block_q8_K
#include "strata/artifact/dequant.hpp"  // fp16_to_fp32 / read_u16

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr int kQK = 128;   // values per PTQ1_0 block
constexpr int kBB = 28;    // bytes per PTQ1_0 block

struct Consts {
    __m256i c0, c1, c2, c3, sh2, sh3, ones8, ones16, m00ff, mff00, mf8, x80, th1, th2;
    Consts() {
        // powers of 3 per 16-bit lane (both bytes of a lane share the factor)
        c0 = _mm256_setr_epi16(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3);
        c1 = _mm256_setr_epi16(9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27);
        c2 = _mm256_setr_epi16(81, 81, 81, 81, 81, 81, 81, 81, 1, 1, 1, 1, 3, 3, 3, 3);
        c3 = _mm256_setr_epi16(9, 9, 9, 9, 27, 27, 27, 27, 81, 81, 81, 81, 1, 3, 9, 27);
        // sources of values 64..95 and 96..127 (block bytes 12..27: qs[16..23] at 4..11, qh at 12..13)
        sh2 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                               4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11);
        sh3 = _mm256_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11,
                               4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13);
        ones8 = _mm256_set1_epi8(1);
        ones16 = _mm256_set1_epi16(1);
        m00ff = _mm256_set1_epi16(0x00FF);
        mff00 = _mm256_set1_epi16((short) 0xFF00);
        mf8 = _mm256_set1_epi8((char) 0xF8);
        x80 = _mm256_set1_epi8((char) 0x80);
        th1 = _mm256_set1_epi8((char) (85 ^ 0x80));
        th2 = _mm256_set1_epi8((char) (170 ^ 0x80));
    }
};

// per-byte x*c mod 256 (c equal in both bytes of each 16-bit lane)
inline __m256i mul_u8(const Consts& k, __m256i x, __m256i c) {
    const __m256i lo = _mm256_and_si256(_mm256_mullo_epi16(x, c), k.m00ff);
    const __m256i hi = _mm256_mullo_epi16(_mm256_and_si256(x, k.mff00), c);
    return _mm256_or_si256(lo, hi);
}

// trit (3*v)>>8 in {0,1,2}: unsigned v >= 86 and v >= 171 as signed compares of v ^ 0x80
inline __m256i trit(const Consts& k, __m256i v) {
    const __m256i vs = _mm256_xor_si256(v, k.x80);
    const __m256i m1 = _mm256_cmpgt_epi8(vs, k.th1);
    const __m256i m2 = _mm256_cmpgt_epi8(vs, k.th2);
    return _mm256_sub_epi8(_mm256_setzero_si256(), _mm256_add_epi8(m1, m2));
}

inline int32_t hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    return _mm_cvtsi128_si32(s);
}

// one 28-byte weight block -> four vectors of 32 trit codes {0,1,2} (values 0..31, 32..63, 64..95, 96..127)
inline void decode_codes(const Consts& k, const uint8_t* xb, __m256i t[4]) {
    const __m128i a = _mm_loadu_si128((const __m128i*) xb);
    const __m128i b = _mm_loadu_si128((const __m128i*) (xb + 12));   // block bytes 12..27: in bounds
    const __m256i aa = _mm256_broadcastsi128_si256(a);
    const __m256i ab = _mm256_shuffle_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1), k.sh2);
    const __m256i bb = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(b), k.sh3);
    // v3: values 0..63 are the source times 1, 3 | 9, 27: V0 = [x | 3x] by two byte adds and a blend, V1 = 9*V0 as V0 + (V0 << 3)
    // (the byte-wise shift is the 16-bit shift with the bits that crossed into the neighbour byte masked off)
    const __m256i x3 = _mm256_add_epi8(aa, _mm256_add_epi8(aa, aa));
    const __m256i v0 = _mm256_blend_epi32(aa, x3, 0xF0);
    const __m256i v1 = _mm256_add_epi8(v0, _mm256_and_si256(_mm256_slli_epi16(v0, 3), k.mf8));
    t[0] = trit(k, v0);
    t[1] = trit(k, v1);
    t[2] = trit(k, mul_u8(k, ab, k.c2));
    t[3] = trit(k, mul_u8(k, bb, k.c3));
}

// sum(code*y) over the block's 128 values (int16 pair sums added before one madd), exact in int32
inline __m256i block_prod(const Consts& k, const __m256i t[4], const int8_t* aq) {
    const __m256i m01 = _mm256_add_epi16(_mm256_maddubs_epi16(t[0], _mm256_loadu_si256((const __m256i*) aq)),
                                         _mm256_maddubs_epi16(t[1], _mm256_loadu_si256((const __m256i*) (aq + 32))));
    const __m256i m23 = _mm256_add_epi16(_mm256_maddubs_epi16(t[2], _mm256_loadu_si256((const __m256i*) (aq + 64))),
                                         _mm256_maddubs_epi16(t[3], _mm256_loadu_si256((const __m256i*) (aq + 96))));
    return _mm256_madd_epi16(_mm256_add_epi16(m01, m23), k.ones16);
}

// sum of the 128 activation values of block half h, from Q8_K's bsums (16 sums of 16 values)
inline int32_t half_sum(const block_q8_K& a, int h) {
    __m128i s = _mm_madd_epi16(_mm_loadu_si128((const __m128i*) (a.bsums + 8 * h)), _mm_set1_epi16(1));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    return _mm_cvtsi128_si32(s);
}

template <int NT>
inline void row_dot(const Consts& k, const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    float accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = 0.f;
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* b = row + (size_t) i * kBB;
        __m256i codes[4];
        decode_codes(k, b, codes);
        const float dw = fp16_to_fp32(read_u16(b + 26));
        for (int t = 0; t < NT; ++t) {
            const block_q8_K& a = y[t][i / 2];
            const int h = i % 2;
            const int32_t dot = hsum_i32(block_prod(k, codes, a.qs + (size_t) h * kQK)) - half_sum(a, h);
            accf[t] += dw * a.d * (float) dot;   // the scalar kernel's float line
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = accf[t];
}

// eight int32x8 vectors -> one vector holding the horizontal sum of each (lane r = sum of v[r]); integer sums are exact in any order
inline __m256i reduce8(const __m256i v[8]) {
    const __m256i h01 = _mm256_hadd_epi32(v[0], v[1]), h23 = _mm256_hadd_epi32(v[2], v[3]);
    const __m256i h45 = _mm256_hadd_epi32(v[4], v[5]), h67 = _mm256_hadd_epi32(v[6], v[7]);
    const __m256i q0 = _mm256_hadd_epi32(h01, h23), q1 = _mm256_hadd_epi32(h45, h67);
    return _mm256_add_epi32(_mm256_permute2x128_si256(q0, q1, 0x20), _mm256_permute2x128_si256(q0, q1, 0x31));
}

// eight rows (rp[0..7]) against NT activations: res[t] lane r = the scalar kernel's row_dot of row r, token t
template <int NT>
inline void rows8_dot(const Consts& k, const uint8_t* const* rp, int nblocks, const block_q8_K* const* y, __m256* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        __m256i codes[8][4];
        alignas(16) uint16_t dh[8];
        for (int r = 0; r < 8; ++r) {
            const uint8_t* b = rp[r] + (size_t) i * kBB;
            decode_codes(k, b, codes[r]);
            std::memcpy(&dh[r], b + 26, 2);
        }
        const __m256 dwv = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) dh));
        for (int t = 0; t < NT; ++t) {
            const block_q8_K& a = y[t][i / 2];
            const int h = i % 2;
            const int8_t* aq = a.qs + (size_t) h * kQK;
            __m256i acc[8];
            for (int r = 0; r < 8; ++r) acc[r] = block_prod(k, codes[r], aq);
            const __m256i dots = _mm256_sub_epi32(reduce8(acc), _mm256_set1_epi32(half_sum(a, h)));
            const __m256 ps = _mm256_mul_ps(_mm256_mul_ps(dwv, _mm256_set1_ps(a.d)), _mm256_cvtepi32_ps(dots));
            accf[t] = _mm256_add_ps(accf[t], ps);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = accf[t];
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
             float* const* ff, int r0, int r1) {
    const Consts k;
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / kQK;
    int r = r0;
    for (; r + 8 <= r1; r += 8) {
        const uint8_t* gp[8];
        const uint8_t* up[8];
        for (int j = 0; j < 8; ++j) { gp[j] = blob + (size_t) (r + j) * gu_row; up[j] = blob + up_off + (size_t) (r + j) * gu_row; }
        __m256 gv[NT], uv[NT];
        rows8_dot<NT>(k, gp, nb, y, gv);
        rows8_dot<NT>(k, up, nb, y, uv);
        for (int t = 0; t < NT; ++t) {
            alignas(32) float g[8], u[8];
            _mm256_store_ps(g, gv[t]);
            _mm256_store_ps(u, uv[t]);
            for (int j = 0; j < 8; ++j) ff[t][r + j] = (g[j] / (1.f + std::exp(-g[j]))) * u[j];
        }
    }
    float g[NT], u[NT];
    for (; r < r1; ++r) {
        row_dot<NT>(k, blob + (size_t) r * gu_row, nb, y, g);
        row_dot<NT>(k, blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int NT>
void down_rows(const uint8_t* blob, size_t d_row, int n, const void* const* hq, float* const* out,
               int r0, int r1) {
    const Consts k;
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) hq[t];
    const int nb = n / kQK;
    int r = r0;
    for (; r + 8 <= r1; r += 8) {
        const uint8_t* rp[8];
        for (int j = 0; j < 8; ++j) rp[j] = blob + (size_t) (r + j) * d_row;
        __m256 rv[NT];
        rows8_dot<NT>(k, rp, nb, y, rv);
        for (int t = 0; t < NT; ++t) _mm256_storeu_ps(&out[t][r], rv[t]);
    }
    float res[NT];
    for (; r < r1; ++r) {
        row_dot<NT>(k, blob + (size_t) r * d_row, nb, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

}  // namespace

void ptq1_0_gu_rows_avx2(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                         int nt, float* const* ff, int r0, int r1) {
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

void ptq1_0_down_rows_avx2(const uint8_t* blob, size_t d_row, int n, const void* const* hq,
                           int nt, float* const* out, int r0, int r1) {
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

}  // namespace strata::kernels::cpu
