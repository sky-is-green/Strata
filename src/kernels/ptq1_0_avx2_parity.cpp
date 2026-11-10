// ptq1_0_avx2_parity: the AVX2 PTQ1_0 expert rows against the scalar rows, bit for bit, plus a single-thread throughput check.
// "ptq1_0_avx2_parity <sec> membw [experts]" instead measures memory bandwidth: plain streaming reads and the AVX2 rows over distinct experts, 1..6 threads pinned to cpus 0..5.
// No model needed: random weight blocks (every byte value) and random / extreme Q8_K activations, one expert of the 176B's shape
// ([gate F rows | up F rows | down H rows], H=2560, F=640), nt = 1..8.  usage: ptq1_0_avx2_parity [bench_seconds]
#include "strata/kernels/cpu/ptq1_0.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <immintrin.h>
#include <sched.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels::cpu;

static void fill_act(std::vector<block_q8_K>& y, int mode, std::mt19937& rng) {
    for (auto& b : y) {
        b.d = 0.001f + (rng() % 1000) * 1e-5f;
        for (int k = 0; k < 256; ++k) {
            int v;
            if (mode == 0) v = (int) (rng() % 256) - 128;               // full range incl. -128
            else if (mode == 1) v = -128;                                // extreme: all -128
            else if (mode == 2) v = 127;
            else v = (int) (rng() % 41) - 20;                            // realistic small
            b.qs[k] = (int8_t) v;
        }
        for (int j = 0; j < 16; ++j) { int s = 0; for (int k = 0; k < 16; ++k) s += b.qs[j * 16 + k]; b.bsums[j] = (int16_t) s; }
    }
}

#if defined(__x86_64__) || defined(_M_X64)
static void pin(int cpu) { cpu_set_t m; CPU_ZERO(&m); CPU_SET(cpu, &m); sched_setaffinity(0, sizeof(m), &m); }

static int membw_main(double secs, long nexp) {
    const int H = 2560, F = 640;
    const size_t gu_row = (size_t) H / 128 * 28, d_row = (size_t) F / 128 * 28;
    const size_t up_off = (size_t) F * gu_row, down_off = 2 * up_off, esz = down_off + (size_t) H * d_row;
    std::printf("experts %ld x %zu B = %.2f GiB\n", nexp, esz, (double) nexp * esz / (1ull << 30));
    uint8_t* buf = (uint8_t*) std::aligned_alloc(4096, ((size_t) nexp * esz + 4095) / 4096 * 4096);
    if (!buf) { std::printf("alloc failed\n"); return 2; }
    { std::vector<std::thread> th; for (int t = 0; t < 6; ++t) th.emplace_back([&, t]() { pin(t); std::mt19937 r(t + 1); size_t n = (size_t) nexp * esz, a = n / 6 * t, b = t == 5 ? n : n / 6 * (t + 1); for (size_t i = a; i < b; ++i) buf[i] = (uint8_t) r(); }); for (auto& x : th) x.join(); }
    // scales finite: bytes 26,27 of every block
    for (size_t o = 26; o + 2 <= (size_t) nexp * esz; o += 28) { buf[o] = (uint8_t) o; buf[o + 1] = 0x20 + ((o >> 3) & 0xF); }
    for (int mode = 0; mode < 2; ++mode) {
        for (int T = 1; T <= 6; ++T) {
            std::atomic<long> go{0}; std::atomic<bool> stop{false}; std::vector<double> bytes(T, 0.0); std::vector<std::thread> th;
            for (int t = 0; t < T; ++t) th.emplace_back([&, t]() {
                pin(t); std::mt19937 rng(100 + t);
                std::vector<block_q8_K> act(H / 256), hq(F / 256 + 1); fill_act(act, 3, rng); fill_act(hq, 3, rng);
                std::vector<float> ff(F), oo(H); const void* ap = act.data(); const void* hp = hq.data(); float* fp = ff.data(); float* op = oo.data();
                volatile __m256i sink; __m256i acc = _mm256_setzero_si256();
                while (go.load() == 0) {}
                double nb = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    for (int k = 0; k < 16; ++k) {
                        const size_t e = rng() % (size_t) nexp; const uint8_t* b = buf + e * esz;
                        if (mode == 0) { for (size_t o = 0; o < esz; o += 32) acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i*) (b + o) )); }
                        else { ptq1_0_gu_rows_avx2(b, gu_row, up_off, H, &ap, 1, &fp, 0, F); ptq1_0_down_rows_avx2(b + down_off, d_row, F, &hp, 1, &op, 0, H); }
                        nb += (double) esz;
                    }
                }
                sink = acc; bytes[t] = nb;
            });
            const auto t0 = std::chrono::steady_clock::now(); go = 1;
            std::this_thread::sleep_for(std::chrono::duration<double>(secs)); stop = true; for (auto& x : th) x.join();
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); double tot = 0; for (double x : bytes) tot += x;
            std::printf("%-14s threads %d: %6.2f GB/s  (%.2f per thread)\n", mode ? "avx2 PTQ1 rows" : "plain read", T, tot / el / 1e9, tot / el / 1e9 / T);
            std::fflush(stdout);
        }
    }
    std::free(buf);
    return 0;
}
#endif

int main(int argc, char** argv) {
#if defined(__x86_64__) || defined(_M_X64)
    const double bench_s = argc > 1 ? std::atof(argv[1]) : 1.0;
    if (argc > 2 && std::strcmp(argv[2], "membw") == 0) return cpu_avx2_ok() ? membw_main(bench_s, argc > 3 ? std::atol(argv[3]) : 6000) : 0;
    if (!cpu_avx2_ok()) { std::printf("cpu_avx2_ok() false: nothing to test\n"); return 0; }
    const int H = 2560, F = 640;
    const size_t gu_row = (size_t) H / 128 * 28, d_row = (size_t) F / 128 * 28;
    const size_t up_off = (size_t) F * gu_row, down_off = 2 * up_off;
    std::vector<uint8_t> blob(down_off + (size_t) H * d_row);
    std::mt19937 rng(12345);
    for (auto& b : blob) b = (uint8_t) rng();
    // blocks with fp16 scales that are finite: overwrite the d field (bytes 26..27) with a small normal half
    for (size_t o = 26; o + 2 <= blob.size(); o += 28) { blob[o] = (uint8_t) rng(); blob[o + 1] = 0x20 + (rng() % 0x10); }
    // edge blocks: all-0 and all-255 codes in the first rows
    std::memset(blob.data(), 0, 28 * 4); std::memset(blob.data() + 28 * 4, 255, 28 * 4);
    for (int blk = 0; blk < 8; ++blk) { blob[28 * blk + 26] = 0x00; blob[28 * blk + 27] = 0x3c; }

    long bad = 0, total = 0;
    for (int mode = 0; mode < 4; ++mode) {
        for (int nt = 1; nt <= 8; ++nt) {
            std::vector<std::vector<block_q8_K>> act(nt, std::vector<block_q8_K>((size_t) H / 256));
            std::vector<std::vector<block_q8_K>> hq(nt, std::vector<block_q8_K>((size_t) F / 256 + 1));
            std::vector<const void*> ap(nt), hp(nt);
            for (int t = 0; t < nt; ++t) { fill_act(act[t], mode, rng); fill_act(hq[t], mode, rng); ap[t] = act[t].data(); hp[t] = hq[t].data(); }
            std::vector<std::vector<float>> ff0(nt, std::vector<float>(F)), ff1(nt, std::vector<float>(F));
            std::vector<std::vector<float>> o0(nt, std::vector<float>(H)), o1(nt, std::vector<float>(H));
            std::vector<float*> f0(nt), f1(nt), p0(nt), p1(nt);
            for (int t = 0; t < nt; ++t) { f0[t] = ff0[t].data(); f1[t] = ff1[t].data(); p0[t] = o0[t].data(); p1[t] = o1[t].data(); }
            ptq1_0_gu_rows_scalar(blob.data(), gu_row, up_off, H, ap.data(), nt, f0.data(), 0, F);
            ptq1_0_gu_rows_avx2(blob.data(), gu_row, up_off, H, ap.data(), nt, f1.data(), 0, F);
            ptq1_0_down_rows_scalar(blob.data() + down_off, d_row, F, hp.data(), nt, p0.data(), 0, H);
            ptq1_0_down_rows_avx2(blob.data() + down_off, d_row, F, hp.data(), nt, p1.data(), 0, H);
            for (int t = 0; t < nt; ++t) {
                total += F + H;
                bad += std::memcmp(ff0[t].data(), ff1[t].data(), sizeof(float) * F) != 0 ? 1 : 0;
                bad += std::memcmp(o0[t].data(), o1[t].data(), sizeof(float) * H) != 0 ? 1 : 0;
                for (int r = 0; r < F; ++r) if (std::memcmp(&ff0[t][r], &ff1[t][r], 4)) { std::printf("gu mismatch mode %d nt %d t %d row %d: %.9g vs %.9g\n", mode, nt, t, r, ff0[t][r], ff1[t][r]); break; }
                for (int r = 0; r < H; ++r) if (std::memcmp(&o0[t][r], &o1[t][r], 4)) { std::printf("down mismatch mode %d nt %d t %d row %d: %.9g vs %.9g\n", mode, nt, t, r, o0[t][r], o1[t][r]); break; }
            }
        }
    }
    std::printf("parity: %ld output rows compared, %ld (mode,nt,token) groups differ -> %s\n", total, bad, bad == 0 ? "BIT-IDENTICAL" : "MISMATCH");

    // v2: arbitrary row ranges (remainders, offsets) with a sentinel around the range, and special fp16 scales (zero, subnormal, negative, inf, NaN)
    {
        std::vector<uint8_t> sblob = blob;
        const uint16_t specials[] = {0x0000, 0x0001, 0x03ff, 0x8001, 0xbc00, 0x7bff, 0x7c00, 0xfc00, 0x7e00};
        size_t si = 0;
        for (size_t o = 26; o + 2 <= sblob.size(); o += 28 * 37) { const uint16_t v = specials[si++ % 9]; sblob[o] = (uint8_t) v; sblob[o + 1] = (uint8_t) (v >> 8); }
        const int ranges[][2] = {{0, 1}, {0, 7}, {0, 8}, {3, 40}, {5, 13}, {7, 23}, {11, 640}, {600, 640}, {8, 16}, {639, 640}};
        long rbad = 0, rtot = 0;
        for (int variant = 0; variant < 2; ++variant) {
            const std::vector<uint8_t>& bl = variant ? sblob : blob;
            for (int nt = 1; nt <= 8; ++nt) {
                std::vector<std::vector<block_q8_K>> act(nt, std::vector<block_q8_K>((size_t) H / 256)), hq(nt, std::vector<block_q8_K>((size_t) F / 256 + 1));
                std::vector<const void*> ap(nt), hp(nt);
                for (int t = 0; t < nt; ++t) { fill_act(act[t], t % 4, rng); fill_act(hq[t], (t + 1) % 4, rng); ap[t] = act[t].data(); hp[t] = hq[t].data(); }
                for (const auto& rg : ranges) {
                    const int r0 = rg[0], r1 = rg[1];
                    std::vector<std::vector<float>> a0(nt, std::vector<float>(F + 16, 12345.f)), a1 = a0, d0(nt, std::vector<float>(H + 16, 12345.f)), d1 = d0;
                    std::vector<float*> f0(nt), f1(nt), p0(nt), p1(nt);
                    for (int t = 0; t < nt; ++t) { f0[t] = a0[t].data(); f1[t] = a1[t].data(); p0[t] = d0[t].data(); p1[t] = d1[t].data(); }
                    ptq1_0_gu_rows_scalar(bl.data(), gu_row, up_off, H, ap.data(), nt, f0.data(), r0, r1);
                    ptq1_0_gu_rows_avx2(bl.data(), gu_row, up_off, H, ap.data(), nt, f1.data(), r0, r1);
                    ptq1_0_down_rows_scalar(bl.data() + down_off, d_row, F, hp.data(), nt, p0.data(), r0, r1);
                    ptq1_0_down_rows_avx2(bl.data() + down_off, d_row, F, hp.data(), nt, p1.data(), r0, r1);
                    for (int t = 0; t < nt; ++t) {
                        ++rtot;
                        if (std::memcmp(a0[t].data(), a1[t].data(), sizeof(float) * (F + 16)) || std::memcmp(d0[t].data(), d1[t].data(), sizeof(float) * (H + 16))) {
                            if (rbad++ < 5) std::printf("range mismatch variant %d nt %d t %d rows [%d,%d)\n", variant, nt, t, r0, r1);
                        }
                    }
                }
            }
        }
        std::printf("ranges+specials: %ld (variant,nt,range,token) groups compared, %ld differ -> %s\n", rtot, rbad, rbad == 0 ? "BIT-IDENTICAL" : "MISMATCH");
        bad += rbad;
    }

    // throughput: one expert (%zu bytes) from cache, one thread; weight bytes per second
    for (int nt : {1, 4}) {
        std::vector<std::vector<block_q8_K>> act(nt, std::vector<block_q8_K>((size_t) H / 256)), hq(nt, std::vector<block_q8_K>((size_t) F / 256 + 1));
        std::vector<const void*> ap(nt), hp(nt);
        std::vector<std::vector<float>> ff(nt, std::vector<float>(F)), oo(nt, std::vector<float>(H));
        std::vector<float*> fp(nt), op(nt);
        for (int t = 0; t < nt; ++t) { fill_act(act[t], 3, rng); fill_act(hq[t], 3, rng); ap[t] = act[t].data(); hp[t] = hq[t].data(); fp[t] = ff[t].data(); op[t] = oo[t].data(); }
        for (int impl = 0; impl < 2; ++impl) {
            auto run = [&]() {
                if (impl == 0) { ptq1_0_gu_rows_scalar(blob.data(), gu_row, up_off, H, ap.data(), nt, fp.data(), 0, F); ptq1_0_down_rows_scalar(blob.data() + down_off, d_row, F, hp.data(), nt, op.data(), 0, H); }
                else { ptq1_0_gu_rows_avx2(blob.data(), gu_row, up_off, H, ap.data(), nt, fp.data(), 0, F); ptq1_0_down_rows_avx2(blob.data() + down_off, d_row, F, hp.data(), nt, op.data(), 0, H); }
            };
            run();
            const auto t0 = std::chrono::steady_clock::now(); long it = 0; double el = 0;
            do { for (int k = 0; k < 20; ++k) run(); it += 20; el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } while (el < bench_s);
            std::printf("bench nt=%d %-6s %8.1f us/expert  %6.2f GB/s weight bytes (expert %zu B)\n", nt, impl ? "avx2" : "scalar", el / it * 1e6, (double) blob.size() * it / el / 1e9, blob.size());
        }
    }
    // v3: the same expert in calls of rpc rows (the pool hands the kernel ranges of about 35 rows), AVX2 only
    for (int nt : {1, 4}) {
        std::vector<std::vector<block_q8_K>> act(nt, std::vector<block_q8_K>((size_t) H / 256)), hq(nt, std::vector<block_q8_K>((size_t) F / 256 + 1));
        std::vector<const void*> ap(nt), hp(nt);
        std::vector<std::vector<float>> ff(nt, std::vector<float>(F + 16)), oo(nt, std::vector<float>(H + 16));
        std::vector<float*> fp(nt), op(nt);
        for (int t = 0; t < nt; ++t) { fill_act(act[t], 3, rng); fill_act(hq[t], 3, rng); ap[t] = act[t].data(); hp[t] = hq[t].data(); fp[t] = ff[t].data(); op[t] = oo[t].data(); }
        for (int rpc : {640, 64, 35, 17, 9}) {
            auto run = [&]() {
                for (int r = 0; r < F; r += rpc) ptq1_0_gu_rows_avx2(blob.data(), gu_row, up_off, H, ap.data(), nt, fp.data(), r, std::min(F, r + rpc));
                for (int r = 0; r < H; r += rpc * 4) ptq1_0_down_rows_avx2(blob.data() + down_off, d_row, F, hp.data(), nt, op.data(), r, std::min(H, r + rpc * 4));
            };
            run();
            const auto t0 = std::chrono::steady_clock::now(); long it = 0; double el = 0;
            do { for (int k = 0; k < 20; ++k) run(); it += 20; el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } while (el < bench_s);
            std::printf("calls of %3d gate/up rows (down %3d): nt=%d %7.1f us/expert\n", rpc, rpc * 4, nt, el / it * 1e6);
        }
    }
    return bad == 0 ? 0 : 1;
#else
    (void) argc; (void) argv; std::printf("not x86-64\n"); return 0;
#endif
}
