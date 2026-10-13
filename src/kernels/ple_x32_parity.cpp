// Scion P5: bit-exact parity of the Q3X32/Q2X32 PLE table reader.
//
// The fixture is written by the fork's tools/ple-table --dump: real 4l table
// rows decoded through ggml get_rows on the CPU backend. This test decodes
// the same rows through PleTable's direct reader and compares bit for bit,
// so the fork's decoder and the engine's reader cannot drift apart.
//
// usage: ple_x32_parity <gguf-with-q3x32-or-q2x32-table> <fixture.plef>
#define NOMINMAX
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace k = strata::kernels;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <gguf-with-q3x32-or-q2x32-table> <fixture.plef>\n", argv[0]);
        return 2;
    }
    FILE* f = std::fopen(argv[2], "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open fixture %s\n", argv[2]);
        return 1;
    }
    char magic[4] = {};
    uint32_t version = 0, ftype = 0, np = 0, dim = 0;
    const bool hdr = std::fread(magic, 1, 4, f) == 4 &&
                     std::fread(&version, 4, 1, f) == 1 &&
                     std::fread(&ftype, 4, 1, f) == 1 &&
                     std::fread(&np, 4, 1, f) == 1 &&
                     std::fread(&dim, 4, 1, f) == 1;
    if (!hdr || std::memcmp(magic, "PLEF", 4) != 0 || version != 1 || dim != 160 || np == 0) {
        std::fprintf(stderr, "bad fixture header\n");
        std::fclose(f);
        return 1;
    }
    std::vector<int32_t> rows(np);
    std::vector<float> want((size_t) np * 160);
    const bool body = std::fread(rows.data(), 4, np, f) == np &&
                      std::fread(want.data(), 4, want.size(), f) == want.size();
    std::fclose(f);
    if (!body) {
        std::fprintf(stderr, "short fixture\n");
        return 1;
    }

    const char* want_name = ftype == 144 ? "Q3X32" : ftype == 145 ? "Q2X32" : ftype == 146 ? "Q4X160" : nullptr;
    if (want_name == nullptr) {
        std::fprintf(stderr, "fixture type %u is not a 32-block table\n", ftype);
        return 1;
    }

    k::PleIoOptions options;
    options.mode = k::PleIo::Direct;
    k::PleTable table;
    std::string err;
    if (!table.open(argv[1], err, options)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const uint32_t max_row = (uint32_t) *std::max_element(rows.begin(), rows.end());
    if (table.rows() <= max_row) {
        std::fprintf(stderr, "fixture row %u is beyond the %llu table rows\n",
                     max_row, (unsigned long long) table.rows());
        return 1;
    }

    int64_t ndiff = 0;
    double max_abs = 0.0;
    auto check = [&](const float* got, const float* w, size_t n) {
        for (size_t j = 0; j < n; ++j) {
            if (std::memcmp(&got[j], &w[j], sizeof(float)) != 0) {
                ++ndiff;
                max_abs = std::max(max_abs, (double) std::fabs(got[j] - w[j]));
            }
        }
    };

    size_t i = 0;
    while (i < np) {
        const size_t m = std::min<size_t>(16, np - i);
        if (m == 16) {
            uint32_t r16[16];
            for (size_t j = 0; j < 16; ++j) r16[j] = (uint32_t) rows[i + j];
            float got[16 * 160];
            if (!table.issue(r16) || !table.collect(got, err)) {
                std::fprintf(stderr, "collect: %s\n", err.c_str());
                return 1;
            }
            check(got, want.data() + i * 160, 16 * 160);
        } else {
            for (size_t j = 0; j < m; ++j) {
                float got[160];
                table.read_row((uint32_t) rows[i + j], got);
                check(got, want.data() + (i + j) * 160, 160);
            }
        }
        i += m;
    }

    const bool fmt_ok = std::strcmp(table.format(), want_name) == 0;
    std::printf("%s PLE: %u fixture rows, %lld of %zu values differ, max_abs %.3e %s\n",
                table.format(), np, (long long) ndiff, want.size(), max_abs,
                (ndiff == 0 && fmt_ok) ? "PASS" : "FAIL");
    return (ndiff == 0 && fmt_ok) ? 0 : 1;
}
