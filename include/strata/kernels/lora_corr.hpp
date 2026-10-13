// include/strata/kernels/lora_corr.hpp - the runtime correction branches of the Scion releases.
//
// The release GGUF carries an embedded adapter (`adapter.embedded = true`, rank 512, `adapter.lora.alpha = 0`
// -> scale 1.0, applied at scale 1.0 by the fork) as 96 LoRA pairs on three [block] outputs:
//
//   blk.N.ffn_moe_out.weight.lora_{a,b}   the MoE block output, from the normalized FFN input   (all layers)
//   blk.N.ssm_out.weight.lora_{a,b}       the GDN output projection, from its input activation  (36 layers)
//   blk.N.attn_output.weight.lora_{a,b}   the attention output projection, from its input        (12 layers)
//
// Applying one branch is `out += B (A x)` with A [rank, n_in] and B [n_out, rank], both raw f16; the
// activations are f32 and reach the kernels exactly as the reference's f32 tensors did.  The A pass rounds
// x to f16 and writes its own result as f16 (`t16`); the B pass rounds B's factor to f16 - the same rule
// llama.cpp's F16 matmuls apply - and accumulation is f32 in both passes.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The largest rank the engine's correction scratch is sized for (the release ships 512; the validator
/// refuses anything larger rather than reading past the scratch).
constexpr int64_t kLoraCorrMaxRank = 512;

/// The x16 rounding scratch is filled at most this many rows at a time (`lora_corr_apply` loops over row
/// chunks), so a caller's scratch must hold `min(n_rows, kLoraCorrRoundRows) * n_in` f16 values.  The cap
/// bounds the scratch for the batched (prefill) path, where n_rows is thousands and the full x16 image
/// would be tens of megabytes.
constexpr int64_t kLoraCorrRoundRows = 1024;

/// Rounds x to f16 into x16 with the repo's RNE converter (`f16_from_f32`), the A pass's operand rule.
void lora_corr_round_x(const float* x, uint16_t* x16, int64_t n, void* stream);

/// `t16` = [n_rows][rank] f16 scratch (A's output, already rounded); `out` = [n_rows][n_out] receives
/// `+= B t16`; `x` = [n_rows][n_in]; `x16` is a caller-owned scratch of
/// `min(n_rows, kLoraCorrRoundRows) * n_in` f16 that this call fills.  Runs on `stream`.  A null `a` or
/// `b` is a no-op (a pack without corrections).  `n_in` and `rank` must be multiples of 8 (uint4 rows).
void lora_corr_apply(const uint16_t* a, const uint16_t* b, const float* x, uint16_t* x16, uint16_t* t16, float* out,
                     int64_t n_rows, int64_t n_in, int64_t rank, int64_t n_out, void* stream);

}  // namespace strata::kernels
