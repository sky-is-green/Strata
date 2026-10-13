// include/strata/core/corrections.hpp - the runtime correction branches (see kernels/lora_corr.hpp).
//
// The engine applies the embedded adapter of a Scion release as three additive branches:
//
//   blk.N.ffn_moe_out.weight.lora_{a,b}   applied to the MoE block output, from the FFN input `bb.mixed`
//   blk.N.ssm_out.weight.lora_{a,b}       applied to the GDN output,        from `b.y` (the ssm_out input)
//   blk.N.attn_output.weight.lora_{a,b}   applied to the attention output,  from the gated attention
//
// `corr_validate` checks every pair the pack carries at startup - shapes, raw-f16 arena form, rank - so a
// malformed adapter is refused before a token runs; a pack without corrections is valid and `corr_apply`
// is a no-op.
#pragma once

#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/lora_corr.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// The three block-output sites.  The order is the scratch layout's, and nothing else's.
enum class CorrSite : int { FfnMoeOut = 0, SsmOut = 1, AttnOutput = 2 };

/// "ffn_moe_out.weight" / "ssm_out.weight" / "attn_output.weight" - the GGUF suffix before ".lora_a".
const char* corr_site_name(CorrSite site);

/// Every `blk.N.<site>.weight.lora_{a,b}` pair in `tables` must be raw f16 (2 B/elem), have a rank of at
/// most `strata::kernels::kLoraCorrMaxRank`, and attach to the layer kind that has the projection (ssm_out
/// to GDN layers, attn_output to QSA layers).  `branches` counts the pairs.  Returns false with `err` set to
/// the first problem found.
bool corr_validate(const WeightTable& tables, const ModelGeometry& g, int& branches, std::string& err);

/// `out[row] += B (A x[row])` for `n_rows` rows, when layer `layer` carries the branch for `site`; a no-op
/// otherwise.  `t16` must hold `n_rows * rank` f16 (rank <= kLoraCorrMaxRank) and `x16` must hold
/// `min(n_rows, kLoraCorrRoundRows) * n_in` f16, both on the stream's device.
bool corr_apply(const WeightTable& tables, const ModelGeometry& g, int64_t layer, CorrSite site, int64_t n_rows,
                const float* x, uint16_t* x16, uint16_t* t16, float* out, void* stream, std::string& err);

}  // namespace strata::core
