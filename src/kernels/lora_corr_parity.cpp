// src/kernels/lora_corr_parity.cpp - the runtime correction branches' kernels against a host reference
// that reproduces every pass exactly: the f16 rounding of x, the four-quarter split of the A reduction in
// uint4 groups with its four accumulator chains, the 16/8/4/2/1 lane tree, the left-to-right combine of the
// partials, and the B pass's lane-strided accumulation.  Every blk.N.<site>.weight.lora_{a,b} pair in the
// GGUF is applied to three synthetic activation rows (f32), and `out += B (A x)` must match the reference
// BIT FOR BIT, so a reordered reduction (which would break decode/prefill/verify agreement) fails rather
// than skims past a tolerance.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/lora_corr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

/// The A pass's 8-element product, host side: the same eight statements the kernel's `dot8` writes.
float dot8(const uint16_t* arow, const uint16_t* xrow) {
    float s = f32_from_f16(xrow[0]) * f32_from_f16(arow[0]);
    s += f32_from_f16(xrow[1]) * f32_from_f16(arow[1]);
    s += f32_from_f16(xrow[2]) * f32_from_f16(arow[2]);
    s += f32_from_f16(xrow[3]) * f32_from_f16(arow[3]);
    s += f32_from_f16(xrow[4]) * f32_from_f16(arow[4]);
    s += f32_from_f16(xrow[5]) * f32_from_f16(arow[5]);
    s += f32_from_f16(xrow[6]) * f32_from_f16(arow[6]);
    s += f32_from_f16(xrow[7]) * f32_from_f16(arow[7]);
    return s;
}

/// The kernel's A pass, host side: four quarters of the reduction axis in uint4 groups, each quarter's lane
/// l taking groups q0+l, q0+l+32, ..., the four accumulator chains of the 4x-unrolled loop, the
/// ((acc0+acc1)+(acc2+acc3)) combine, the 16/8/4/2/1 tree and the left-to-right quarter combine, the
/// result rounded with `f16_from_f32` (the kernel's write).  `x16` is the f16 rounding of the activation,
/// as the kernel's round pass produces it.
std::vector<uint16_t> ref_a(const std::vector<uint16_t>& a, const std::vector<uint16_t>& x16, int64_t n_rows,
                            int64_t n_in, int64_t rank) {
    constexpr int A_WARPS = 4;
    std::vector<uint16_t> t((size_t) (n_rows * rank));
    const int64_t n8 = n_in / 8;
    for (int64_t tok = 0; tok < n_rows; ++tok) {
        for (int64_t r = 0; r < rank; ++r) {
            const uint16_t* arow = a.data() + (size_t) r * n_in;
            const uint16_t* xrow = x16.data() + (size_t) tok * n_in;
            float part[A_WARPS];
            for (int q = 0; q < A_WARPS; ++q) {
                const int64_t q0 = (n8 * q) / A_WARPS;
                const int64_t q1 = (n8 * (q + 1)) / A_WARPS;
                float p[32] = {};
                for (int lane = 0; lane < 32; ++lane) {
                    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                    int64_t k = q0 + lane;
                    for (; k + 96 < q1; k += 128) {
                        acc0 += dot8(arow + 8 * k, xrow + 8 * k);
                        acc1 += dot8(arow + 8 * (k + 32), xrow + 8 * (k + 32));
                        acc2 += dot8(arow + 8 * (k + 64), xrow + 8 * (k + 64));
                        acc3 += dot8(arow + 8 * (k + 96), xrow + 8 * (k + 96));
                    }
                    for (; k < q1; k += 32) acc0 += dot8(arow + 8 * k, xrow + 8 * k);
                    p[lane] = (acc0 + acc1) + (acc2 + acc3);
                }
                for (int off = 16; off > 0; off >>= 1)
                    for (int l = 0; l < off; ++l) p[l] += p[l + off];
                part[q] = p[0];
            }
            t[(size_t) (tok * rank + r)] = f16_from_f32(((part[0] + part[1]) + part[2]) + part[3]);
        }
    }
    return t;
}

// The kernel's B pass: out[tok][o] += sum_k b[o][k] * t16[tok][k], one warp per output, lanes striding
// the reduction axis in uint4 groups of 8 f16 with `dot8` per group, then the 16/8/4/2/1 tree.
void ref_b_add(const std::vector<uint16_t>& b, const std::vector<uint16_t>& t16, std::vector<float>& out,
               int64_t n_rows, int64_t rank, int64_t n_out) {
    const int64_t n8 = rank / 8;
    for (int64_t tok = 0; tok < n_rows; ++tok) {
        for (int64_t o = 0; o < n_out; ++o) {
            const uint16_t* brow = b.data() + (size_t) o * rank;
            const uint16_t* trow = t16.data() + (size_t) tok * rank;
            float p[32] = {};
            for (int lane = 0; lane < 32; ++lane) {
                float acc = 0.0f;
                for (int64_t k = lane; k < n8; k += 32) acc += dot8(brow + 8 * k, trow + 8 * k);
                p[lane] = acc;
            }
            for (int off = 16; off > 0; off >>= 1)
                for (int l = 0; l < off; ++l) p[l] += p[l + off];
            out[(size_t) (tok * n_out + o)] += p[0];
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: lora_corr_parity <gguf-with-lora-corrections>\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    constexpr int64_t NT = 3;
    cudaStream_t s;
    cudaStreamCreate(&s);
    int failures = 0, tested = 0;
    for (const auto& t : gguf.tensors()) {
        const std::string name = t.name;
        const std::string suf = ".lora_a";
        if (name.size() < suf.size() || name.compare(name.size() - suf.size(), suf.size(), suf) != 0) continue;
        const std::string bname = name.substr(0, name.size() - suf.size()) + ".lora_b";
        const auto* bt = gguf.find(bname);
        if (bt == nullptr || t.type != 1 /*F16*/ || bt->type != 1 || t.shape.size() != 2 || bt->shape.size() != 2) {
            std::printf("%s: not an F16 a/b pair\n", name.c_str());
            ++failures;
            continue;
        }
        const int64_t n_in = t.shape[0], rank = t.shape[1], n_out = bt->shape[1];
        if (bt->shape[0] != rank) {
            std::printf("%s: b shape [%lld, %lld] does not pair with rank %lld\n", bname.c_str(),
                        (long long) bt->shape[0], (long long) bt->shape[1], (long long) rank);
            ++failures;
            continue;
        }
        if (rank > strata::kernels::kLoraCorrMaxRank) {
            std::printf("%s: rank %lld over the scratch\n", name.c_str(), (long long) rank);
            ++failures;
            continue;
        }
        if (n_in % 8 != 0) {
            std::printf("%s: n_in %lld is not a multiple of 8\n", name.c_str(), (long long) n_in);
            ++failures;
            continue;
        }
        const uint8_t* ab = gguf.tensor_data(t);
        const uint8_t* bb = gguf.tensor_data(*bt);
        std::vector<uint16_t> a((size_t) (n_in * rank)), b((size_t) (rank * n_out));
        std::memcpy(a.data(), ab, a.size() * 2);
        std::memcpy(b.data(), bb, b.size() * 2);
        std::vector<float> x((size_t) (NT * n_in)), want((size_t) (NT * n_out), 0.5f);
        for (int64_t k = 0; k < NT * n_in; ++k)
            x[(size_t) k] = 0.03f * std::sin(0.07f * (float) k) + 0.002f * (float) (k % 5 - 2);
        // The kernel rounds x inside `lora_corr_apply`; the reference mirrors that with the same converter.
        std::vector<uint16_t> x16(x.size());
        for (size_t k = 0; k < x.size(); ++k) x16[k] = f16_from_f32(x[k]);

        const std::vector<uint16_t> tref = ref_a(a, x16, NT, n_in, rank);
        ref_b_add(b, tref, want, NT, rank, n_out);

        void *da = nullptr, *db = nullptr, *dx = nullptr, *dx16 = nullptr, *dt16 = nullptr, *dout = nullptr;
        cudaMalloc(&da, a.size() * 2);
        cudaMalloc(&db, b.size() * 2);
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dx16, x16.size() * 2);
        cudaMalloc(&dt16, (size_t) (NT * rank) * 2);
        cudaMalloc(&dout, (size_t) (NT * n_out) * 4);
        cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b.data(), b.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        std::vector<float> init((size_t) (NT * n_out), 0.5f), got;
        cudaMemcpy(dout, init.data(), init.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::lora_corr_apply((const uint16_t*) da, (const uint16_t*) db, (const float*) dx,
                                         (uint16_t*) dx16, (uint16_t*) dt16, (float*) dout, NT, n_in, rank, n_out, s);
        cudaStreamSynchronize(s);
        got.resize((size_t) (NT * n_out));
        cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
        size_t differ = 0;
        double max_err = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            differ += std::memcmp(&got[i], &want[i], 4) != 0;
            max_err = std::max(max_err, (double) std::fabs(got[i] - want[i]));
        }
        std::printf("lora %-48s in %5lld rank %lld out %lld: %zu of %zu differ, max|err| %.3e\n", name.c_str(),
                    (long long) n_in, (long long) rank, (long long) n_out, differ, got.size(), max_err);
        if (differ) ++failures;
        cudaFree(da); cudaFree(db); cudaFree(dx); cudaFree(dx16); cudaFree(dt16); cudaFree(dout);
        ++tested;
    }
    cudaStreamDestroy(s);
    std::printf("lora_corr_parity: %d pairs tested, %d failures\n", tested, failures);
    return failures ? 1 : 0;
}
