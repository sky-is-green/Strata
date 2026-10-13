// src/kernels/cpu/ptq1_0.hpp - Scion PTQ1_0 (ggml type 143) expert rows on the CPU.
//
// Block: 128 values in 28 bytes - qs[24] (staged base-3 trits, 5/byte:
// bytes 0..15 hold values n*16+m, bytes 16..23 hold 80+n*8+m), qh[2]
// (values 120+n*2+h), fp16 d.  Activation: ggml Q8_K (256 values), one
// block per two weight blocks; the dot uses the block-level d only, like the
// iq_avx kernels.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

void ptq1_0_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                    int nt, float* const* ff, int r0, int r1);
void ptq1_0_down_rows(const uint8_t* blob, size_t d_row, int n, const void* const* hq,
                      int nt, float* const* out, int r0, int r1);

// Decode one 28-byte block to 128 trits in {-1, 0, +1}; exposed for parity tests.
void ptq1_0_decode128(const uint8_t* block, int8_t* out128);

}  // namespace strata::kernels::cpu
