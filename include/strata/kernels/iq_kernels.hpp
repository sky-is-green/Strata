// include/strata/kernels/iq_kernels.hpp - the i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
// IQ4_NL) and Q2_0 on the GPU for the IQ2_XS / IQ3_XXS model files, and Q4_K / Q5_K / Q5_1 / Q8_0 for Unsloth's
// UD-Q4_K_XL (gate/up Q4_K or Q5_K, down Q5_1 or Q8_0, a Q8_0 embedding).
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, see third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// The token-embedding types iq_embed_rows and iq_dequant_f32 read: the i-quants above and BF16 (30).
bool embed_type_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// The correction branch's A pass with its q8_1 image fused in: tq = q8_1(A xq), both bitwise the staged
/// quantize + iq_mmvq(142, ncols = 1).  n_in is a multiple of 128, rank of 32.
void iq_corr_g128_aq(const void* a, const void* x_q8_1, int n_in, int rank, void* t_q8_1, void* stream);
/// The out projection's MMVQ rows and the correction's A pass in one launch (T = 1; both read the same staged
/// q8_1 image).  False where the pair is not available for the type - the caller runs the two launches.
bool native_mmvq_corr_aq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, const void* a,
                         int rank, void* tq, void* stream);

/// The correction branch's B pass at one row with the add folded in: out = (or +=) B tq, tq the staged q8_1
/// image of t.  Bitwise iq_mmvq(142, ncols = 1) + the f32 add.  rank is a multiple of 128.
void iq_corr_g128_b(const void* b, const void* t_q8_1, int rank, int n_out, float* out, int accumulate, void* stream);

/// The mixed T=1 correction pair: the MoE's g128 delta and the router's f16 delta in one launch each, bitwise
/// the four standalone calls (iq_corr_g128_aq + iq_corr_g128_b + the f16 router pair).  `h` is the f32 router
/// input (rounded in-kernel), `xq_moe` the staged q8_1 image of the same h for the g128 pass.  See the kernels
/// for the shape requirements.
void iq_corr_mixed_a(const void* a_moe, const void* xq_moe, int n_moe, int rank_moe, void* tq_moe,
                     const void* a_router, const float* h, int n_router, int rank_router, void* t16_router,
                     void* stream);
/// The router's f32 GEMV rows and the mixed A pass in one launch (T = 1).  False where a shape does not fit.
bool native_gemv_corr_mixed_a(const void* W, const float* h, float* y, int n_in, int n_out, const void* a_moe,
                              const void* xq_moe, int n_moe, int rank_moe, void* tq_moe, const void* a_router,
                              int n_router, int rank_router, void* t16_router, void* stream);
void iq_corr_mixed_b(const void* b_moe, const void* tq_moe, int rank_moe, int n_out_moe, float* out_moe,
                     const void* b_router, const void* t16_router, int rank_router, int n_out_router, float* out_router,
                     void* stream);
/// The router tail at one row: the MoE output delta, the router f16 delta, top-k and the expert group tables in
/// one launch (256 x 8 routers; the last two bodies are the moved k_group_quant and route_topk's own).
bool native_router_tail_ok(int rank_moe, int n_out_moe, int rank_router, int n_out_router, int n_expert, int k);
void native_router_tail(const void* b_moe, const void* tq_moe, int rank_moe, int n_out_moe, float* out_moe,
                        const void* b_router, const void* t16_router, int rank_router, int n_out_router,
                        float* logits, int32_t* ids, float* weights, int n_expert, int k, const uint8_t* blobs,
                        unsigned long long blob, int n_ent, unsigned long long* grp_ptr, int32_t* grp_start,
                        int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok, void* stream);
/// The expert group tables and the q8_1 staging of h in one launch (the engine's k_group_quant; `qblocks` = 0
/// when the call is already staged, so only the tables run).
void native_group_quant(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs,
                        unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                        int32_t* ent_dst, int32_t* ent_tok, const float* qin, void* qout, int n, int qblocks,
                        void* stream);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32; fp16 times `*scale` when given (device:
/// an NVFP4 expert's s_down, read by the kernel itself so the source may be released right after it).
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream, const float* scale = nullptr);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
/// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.  `tail` (device, NVFP4 only): the blob's {s_gate, s_up,
/// s_down, 0}; gate rows are scaled by s_gate and up rows by s_up (s_down goes on the down matrix: folded into up,
/// the up weights would sit near 1e-8, deep in FP16's subnormals).
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream, const float* tail = nullptr);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
    size_t tail_off = 0;                // NVFP4: {s_gate, s_up, s_down, 0} at the blob's end; 0 = none
};
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
/// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);

/// A770 port (STRATA_PF_XMX=2): up to kXmxGroupMax experts' products in one XMX launch, the weights decoded from the
/// quantized rows into local memory.  Expert i: Y_i[cnt[i]][N] = X_i . W_i^T (X FP16, Y FP32, rows back to back).
/// gate/up: b != nullptr (a = gate, b = up rows of n_embd = K; the output's virtual rows interleave gate/up);
/// down: b == nullptr, a = the [N][K] rows.  N a multiple of 128, K of 32 (and a whole number of rows per 256 block).
constexpr int kXmxGroupMax = 16;
void iq_xmx_grouped(int ty, const uint8_t* const* a, const uint8_t* const* b, const int32_t* cnt, int n,
                    const uint16_t* X, float* Y, int N, int K, void* stream);

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
/// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.
/// `grid_groups` (1 .. cap_groups; 0 = cap_groups) groups run side by side, a block row each striding over the rest:
/// a call that usually has few groups or none (the verify window's PCIe share) launches less for the ones it does
/// not have.  The results do not depend on it.  `single_entry`: at most one entry per group (a single-token
/// decode), where the multi kernels' decode-once-per-row does not pay; the per-entry kernels are selected (bitwise
/// the same values, iq_multi_parity checks it).
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups = 0, bool single_entry = false);
/// true: `native_expert_grouped`'s launches before the group stride (STRATA_GROUPED_V1=1 at startup) - a block row
/// per possible group, SwiGLU and the q8_1 quantization as two kernels over all cap_entries.  Bitwise the same results
/// (native_grouped_parity checks it); kept for A/B timing.  Set before graph capture; captured graphs keep theirs.
void native_grouped_set_v1(bool v1);

/// The bench only: the AMD kernel layout (STRATA_EXP_MODE values; -1 = the environment's) and the phase
/// (0 all, 1 gate/up + SwiGLU + quantize, 2 down).
void native_expert_set_mode(int mode, int phase);
/// The bench only: the down phase's rows-per-warp kernel (bitwise the one-row mapping; on by default where it
/// applies, STRATA_IQ_DOWN_ROWS=0 disables at startup).  Set before graph capture; captured graphs keep theirs.
void native_expert_set_down_rows(bool v);
/// The bench only: the gate/up phase's rows-per-warp kernel (Q = 32/nb >= 2; STRATA_IQ_GU_ROWS=0 disables).
void native_expert_set_gu_rows(bool v);
/// `iq_mmvq` and `native_expert_grouped` decode each weight part once and apply it to every column / entry;
/// true selects the older kernels that decode it again per column (STRATA_OLD_IQ_MMVQ=1 at startup).  Both give
/// bitwise the same results.  Set before graph capture; captured graphs keep the kernels they captured.
void iq_set_old_kernels(bool old);
bool iq_old_kernels();

}  // namespace strata::kernels
