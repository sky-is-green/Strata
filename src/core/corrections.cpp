// src/core/corrections.cpp - validation and dispatch for the runtime correction branches.
#include "strata/core/corrections.hpp"

#include "strata/kernels/lora_corr.hpp"

#include <cstdio>

namespace strata::core {
namespace {

const char* site_suffix(CorrSite site) {
    switch (site) {
        case CorrSite::FfnMoeOut: return "ffn_moe_out.weight";
        case CorrSite::SsmOut: return "ssm_out.weight";
        case CorrSite::AttnOutput: return "attn_output.weight";
    }
    return "";
}

struct SiteDims {
    int64_t in = 0, out = 0;
    bool qsa_only = false, gdn_only = false;
};

SiteDims site_dims(const ModelGeometry& g, CorrSite site) {
    switch (site) {
        case CorrSite::FfnMoeOut: return {g.n_embd, g.n_embd, false, false};
        case CorrSite::SsmOut: return {g.ssm_value_dim, g.n_embd, false, true};
        case CorrSite::AttnOutput: return {g.n_head * g.head_dim, g.n_embd, true, false};
    }
    return {};
}

/// The one arena form the correction kernels read: raw f16 rows, 2 bytes per element.  A 4 B/elem
/// promotion in the arena is refused rather than half-read (the same reasoning as check_layer's byte check).
bool raw_f16(const WeightRef& r) {
    return (r.kind == WeightKind::Verbatim || r.kind == WeightKind::F16InF32) &&
           r.bytes == (uint64_t) r.elements * 2 && r.ne0 > 0 && r.ne1 > 0;
}

}  // namespace

const char* corr_site_name(CorrSite site) { return site_suffix(site); }

bool corr_validate(const WeightTable& tables, const ModelGeometry& g, int& branches, std::string& err) {
    branches = 0;
    for (int64_t layer = 0; layer < g.n_layers; ++layer) {
        const LayerView v(tables, layer);
        // layout.hpp's rule, not a second layer map: full attention is
        // `layer % qsa_interval == qsa_interval - 1` (layers 3, 7, ... 47)
        const bool qsa = is_qsa_layer(g, layer);
        for (int s = 0; s < 3; ++s) {
            const CorrSite site = (CorrSite) s;
            const std::string a_name = std::string(site_suffix(site)) + ".lora_a";
            const std::string b_name = std::string(site_suffix(site)) + ".lora_b";
            const WeightRef* a = v.get(a_name.c_str());
            const WeightRef* b = v.get(b_name.c_str());
            if (a == nullptr && b == nullptr) continue;
            if (a == nullptr || b == nullptr) {
                err = v.name((a == nullptr ? a_name : b_name).c_str()) +
                      " has no " + (a == nullptr ? "lora_a" : "lora_b") + " partner";
                return false;
            }
            const SiteDims d = site_dims(g, site);
            if ((d.qsa_only && !qsa) || (d.gdn_only && qsa)) {
                err = v.name(a_name.c_str()) + ": this projection does not exist on layer " +
                      std::to_string(layer) + (qsa ? " (a QSA layer)" : " (a GDN layer)");
                return false;
            }
            if (!raw_f16(*a) || !raw_f16(*b)) {
                err = v.name(a_name.c_str()) + ": the correction factors must be raw f16 (2 B/element)";
                return false;
            }
            if (a->ne1 > strata::kernels::kLoraCorrMaxRank) {
                err = v.name(a_name.c_str()) + ": rank " + std::to_string(a->ne1) + " is over the engine's " +
                      std::to_string(strata::kernels::kLoraCorrMaxRank) + " scratch";
                return false;
            }
            if (a->ne0 % 8 != 0 || a->ne1 % 8 != 0) {
                err = v.name(a_name.c_str()) + ": n_in " + std::to_string(a->ne0) + " / rank " +
                      std::to_string(a->ne1) + " is not a multiple of 8 (the passes read uint4 rows)";
                return false;
            }
            if (a->ne0 != d.in || a->ne1 != b->ne0 || b->ne1 != d.out) {
                char buf[320];
                std::snprintf(buf, sizeof buf,
                              "%s: expected a [%lld, rank] / b [rank, %lld], got a [%lld, %lld] / b [%lld, %lld]",
                              v.name(a_name.c_str()).c_str(), (long long) d.in, (long long) d.out,
                              (long long) a->ne0, (long long) a->ne1, (long long) b->ne0, (long long) b->ne1);
                err = buf;
                return false;
            }
            ++branches;
        }
    }
    return true;
}

bool corr_apply(const WeightTable& tables, const ModelGeometry& g, int64_t layer, CorrSite site, int64_t n_rows,
                const float* x, uint16_t* x16, uint16_t* t16, float* out, void* stream, std::string& err) {
    const LayerView v(tables, layer);
    const std::string a_name = std::string(site_suffix(site)) + ".lora_a";
    const std::string b_name = std::string(site_suffix(site)) + ".lora_b";
    const WeightRef* a = v.get(a_name.c_str());
    const WeightRef* b = v.get(b_name.c_str());
    if (a == nullptr && b == nullptr) return true;   // a pack without corrections
    if (a == nullptr || b == nullptr || !raw_f16(*a) || !raw_f16(*b) || a->ne1 > strata::kernels::kLoraCorrMaxRank) {
        err = v.name(a_name.c_str()) + ": the correction branch is malformed (corr_validate should have refused it)";
        return false;
    }
    strata::kernels::lora_corr_apply((const uint16_t*) a->data, (const uint16_t*) b->data, x, x16, t16, out, n_rows,
                                     a->ne0, a->ne1, b->ne1, stream);
    return true;
}

}  // namespace strata::core
