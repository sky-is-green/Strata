// include/strata/kernels/dense_kernels.hpp - the kernels a DENSE qwen35 model (Qwen3.8-27B) needs and the MoE
// engine does not have.
//
// The GDN recurrence, the quantized GEMVs (native_mmvq), RoPE, the per-head RMSNorm and the attention output gate
// already exist and are reused by src/core/dense_model.cpp.  What was missing, because the MoE model routes its
// attention through the QSA indexer and has no plain FFN:
//
//   * an RMSNorm that writes to a second buffer          (dense_rms_norm)
//   * SwiGLU                                             (dense_swiglu)
//   * an F32 GEMV for the two tiny GDN projections       (dense_gemv_f32)   ssm_alpha / ssm_beta, n_out = 48
//   * a K/V cache and full causal attention over it      (dense_kv_append, dense_attn_decode)
//
// The attention is a split-K flash decode: one block per (head, chunk of 256 cells) keeps an online softmax
// (m, l, acc) per warp, the warps are merged inside the block, and a second kernel merges the chunks.  Every
// query head reads the K/V head `h / (n_head / n_kv)` (GQA: contiguous groups, the same map qsa.hpp pins).
//
// KV layout: [n_kv][max_ctx][head_dim] IEEE half, so one head's cells are contiguous.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w        (w NOT shifted: the GGUF already holds 1 + w)
/// `add` (optional) is added into x first, in the same launch; it must not overlap x.
void dense_rms_norm(const float* x, const float* w, float* out, int rows, int cols, float eps, void* stream,
                    const float* add = nullptr);

/// The GDN output norm of a DENSE qwen35 model: y[h, :] = rms_norm(o[h, :]) * w * SiLU(z[h, :]) for each of `heads`
/// value heads of `cols` (= state_size) values.  llama.cpp's build_norm_gated for qwen35 uses SiLU; the MoE
/// engine's native_gdn_out_norm gates with a sigmoid, which is wrong here (it was the cause of the garbage text).
void dense_gdn_out_norm(const float* o, const float* z, const float* w, float* y, int heads, int cols, float eps,
                        void* stream);

/// Greedy pick over one row of `n` logits: *out_id = the index of the largest (the lowest on a tie; NaN ignored) and
/// *out_p = its softmax probability.  Both are DEVICE pointers.  This is what an MTP draft needs (the token and how
/// sure the head is of it, for the --draft-p-min cut-off).
void dense_argmax_prob(const float* logits, int n, int* out_id, float* out_p, void* stream);

/// out[i] = silu(gate[i]) * up[i]
void dense_swiglu(const float* gate, const float* up, float* out, int64_t n, void* stream);

/// dense_swiglu + (dense_fwht_rows when `signs` is given) + native_quantize_q8_1 in one launch, bit-identical to
/// the triple: one 1024-thread block per (row, 1024-block).  width must be a multiple of 1024; `h_out` and
/// `rot_out` (optional) receive the raw and rotated rows for the FFN trace.
void dense_swiglu_quantize_rows(const float* gate, const float* up, void* q8_out, const float* signs, int width,
                                int rows, float* h_out, float* rot_out, void* stream);

/// dense_rms_norm + dense_fwht_rows + native_quantize_q8_1 in one launch, bit-identical to the triple: one
/// 1024-thread block per row writes xn and the rotated row's q8_1 in the same pass.  width a multiple of 1024.
void dense_rms_fwht_quant_rows(float* x, const float* add, const float* w, float* xn, void* q8_out,
                               const float* signs, int width, int rows, float eps, void* stream);

/// y[r] = sum_i W[r * n_in + i] * x[i]           W is a GGUF matrix stored as F32: ne0 = n_in is the contiguous axis
void dense_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream);

/// dst[0..n) = value   (the RoPE position of every row)
void dense_fill_i32(int32_t* dst, int n, int32_t value, void* stream);

/// Writes this token's K and V (n_kv, head_dim) f32 into cell `pos` of every head's cache as half.
void dense_kv_append(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int pos, int n_kv,
                     int head_dim, int max_ctx, void* stream);

/// How the KV cache stores one head's values (the --kv option; the same names as strata's).
///   F16 : head_dim halves.
///   Q8  : head_dim/32 blocks of {half d; int8 q[32]}   (ggml Q8_0, "int8")
///   Q4  : head_dim/32 blocks of {half d; uint8 q[16]}  (ggml Q4_0, "q4_0"), stored after an orthonormal
///         Walsh-Hadamard rotation of the head's 256 values: the attention rotates q the same way and rotates its
///         output back, so the cache needs head_dim == 256 for this format.
/// K and V have a format each: `--kv k8v4` is Q8 keys (exact scores) with Q4 values.
enum DenseKvFormat : int { DENSE_KV_F16 = 0, DENSE_KV_Q8 = 1, DENSE_KV_Q4 = 2 };

/// Bytes of ONE head's ONE cell in `fmt` (a cache is [n_kv][max_ctx][this]).
uint64_t dense_kv_cell_bytes(int fmt, int head_dim);

/// The folded-basis activation transform (PrismML's prism.hadamard.*; Bonsai 2).  Block 1024, explicit +/-1
/// signs: forward computes y = H (s . x) / 32 per block (R x), inverse y = s . (H x / 32) (R^T x).  x and y are
/// [rows, width] row-major, width a multiple of 1024; x == y is allowed (in place).
void dense_fwht_rows(const float* x, float* y, const float* signs, int width, int rows, bool inverse, void* stream);

/// dense_fwht_rows + native_quantize_q8_1 in one launch, bit-identical to the pair: q8_out = q8_1(R x) with the
/// same 1/32, signs and butterfly.  `hd`/`nk`/`rep` non-zero applies dense_permute_gdn_v's gather at the load
/// (hd * nk * rep must equal width); `rot_out`, when non-null, also receives R x in f32 (the FFN trace path).
void dense_fwht_quantize_rows(const float* x, void* q8_out, float* rot_out, const float* signs, int width, int rows,
                              int hd, int nk, int rep, void* stream);

/// Prism's grouped GDN V layout: gather every row's [hd, nk, rep] tiled view into the [hd, rep, nk] order the
/// ssm_out rotation uses (llama-graph.cpp's reshape/permute/reshape on the activation).
void dense_permute_gdn_v(const float* x, float* y, int width, int rows, int hd, int nk, int rep, void* stream);

/// Two-GPU handoff doorbells.  `bell` is a slot in PINNED HOST memory (pinned because it is the only
/// cross-device channel that works on this stack: see the comment in dense_kernels.cu).  ring() stores `seq`
/// behind a system fence (ordered behind the stream's earlier writes); poll() spins with a cache-bypassing
/// load until the peer's bell has reached `seq`.  Both are one tiny kernel on the calling stream.
void dense_split_ring(uint32_t* bell, uint32_t seq, void* stream);
void dense_split_poll(const uint32_t* bell, uint32_t seq, void* stream);

/// dst[j * dst_stride + dst_off + k] = src[j * w + k] for k < w, j < n.  One kernel, used to place a
/// card's own rotated FFN half into the full activation buffer and to scatter the pulled peer half.
void dense_split_copy(const float* src, float* dst, int w, int n, int dst_stride, int dst_off, void* stream);
// Staging for the two-card handoff: copy src (VRAM) to dst (MAPPED PINNED HOST memory) with SM stores and ring the
// doorbell in the same kernel, so the bell cannot reach the consumer before the payload's stores are visible.
// dst must be cudaHostAllocMapped memory.
void dense_split_stage_out(const float* src, float* dst, int64_t n, void* stream);
void dense_split_stage_in(const float* src, float* dst, int64_t n, void* stream);
void dense_split_stage_ring(const float* src, float* dst, int64_t n, uint32_t* bell, uint32_t seq, void* stream);/// dense_kv_append for any format: k_cache / v_cache are [n_kv][max_ctx][dense_kv_cell_bytes(fmt)] bytes.
void dense_kv_append_fmt(void* k_cache, void* v_cache, int k_fmt, int v_fmt, const float* k, const float* v, int pos,
                         int n_kv, int head_dim, int max_ctx, void* stream);

/// dense_attn_decode for any format.  q is NOT rotated by the caller and out comes back in the original space.
void dense_attn_decode_fmt(const float* q, const void* k_cache, const void* v_cache, int k_fmt, int v_fmt, float* out,
                           float* scratch, int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale,
                           void* stream);

/// dense_attn_decode_fmt + dense_gate_apply in one launch, bit-identical to the pair: q_full is this column's
/// [q | gate] rows (row h of the head), and `out` receives the gated attention directly.
void dense_attn_decode_gate_fmt(const float* q, const void* k_cache, const void* v_cache, int k_fmt, int v_fmt,
                                const float* q_full, float* out, float* scratch, int n_head, int n_kv, int head_dim,
                                int n_ctx, int max_ctx, float scale, void* stream);

/// The graph-replay form: pos_slot is a DEVICE int32 holding this column's pos0 (a memcpy node in the captured
/// pass refills it), and n_splits_grid is the captured grid's split count.  The kernels derive n_ctx from the slot.
void dense_attn_decode_gate_dev_fmt(const float* q, const void* k_cache, const void* v_cache, int k_fmt, int v_fmt,
                                    const float* q_full, float* out, float* scratch, int n_head, int n_kv,
                                    int head_dim, int n_splits_grid, const int32_t* pos_slot, int col, int max_ctx,
                                    float scale, void* stream);

/// Bytes of scratch dense_attn_decode needs for a cache of `max_ctx` cells.
uint64_t dense_attn_scratch_bytes(int n_head, int head_dim, int max_ctx);

/// out (n_head, head_dim) = softmax(q . K^T * scale) V over cells [0, n_ctx), n_ctx >= 1 (the current token's
/// cell is already appended).  q is (n_head, head_dim) f32, already normalised and rotated.
void dense_attn_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, float* scratch,
                       int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale, void* stream);

// ============================ the batched prompt path (T consecutive tokens at a time) ============================
//
// Same arithmetic as the per-token kernels above, with the token as the slow index: every buffer is [T, ...] and the
// order-dependent steps (the GDN conv and delta-rule state, the KV append, the causal mask) walk the tokens inside
// one launch.  Feeding a chunk therefore has to land on the same logits as feeding those tokens one at a time.

/// q[r, :] = q_full[r, 0:head_dim] for r in [0, rows): the query half of every head's [q | gate] block.  q_full is
/// [rows, 2*head_dim] and q is [rows, head_dim] (rows = T * n_head).
void dense_split_q(const float* q_full, float* q, int rows, int head_dim, void* stream);

/// out[i] = attn[i] * sigmoid(gate) with the gate the second half of every head's q_full block (rows = T * n_head).
void dense_gate_apply(const float* attn, const float* q_full, float* out, int rows, int n_head, int head_dim,
                      void* stream);

/// dst[i] = pos0 + i / heads   (the position of every row of a [T, heads, head_dim] tensor)
void dense_positions_i32(int32_t* dst, int rows, int heads, int32_t pos0, void* stream);

/// K and V of T consecutive cells (k/v are [T, n_kv, head_dim] f32) into cells pos0..pos0+T-1 of every cache.
void dense_kv_append_rows(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int T, int pos0,
                          int n_kv, int head_dim, int max_ctx, void* stream);

/// dense_kv_append_rows for any KV format: the caches are [n_kv][max_ctx][dense_kv_cell_bytes(fmt)] bytes.
void dense_kv_append_rows_fmt(void* k_cache, void* v_cache, int k_fmt, int v_fmt, const float* k, const float* v,
                              int T, int pos0, int n_kv, int head_dim, int max_ctx, void* stream);

/// The graph-replay form: the first row's cell position comes from a DEVICE int32 slot.
void dense_kv_append_rows_dev_fmt(void* k_cache, void* v_cache, int k_fmt, int v_fmt, const float* k, const float* v,
                                  int T, const int32_t* pos_slot, int n_kv, int head_dim, int max_ctx, void* stream);

/// Causal attention for T consecutive queries at positions pos0..pos0+T-1: one block per (head, token), the same
/// warp-per-cell online softmax as dense_attn_decode, restricted to cells [0, pos0 + t].  q is [T, n_head, head_dim]
/// f32 (normalised and rotated), out is the same shape.
void dense_attn_chunk(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, int T, int pos0,
                      int n_head, int n_kv, int head_dim, int max_ctx, float scale, void* stream);

/// dense_attn_chunk for any KV format.  q is NOT rotated by the caller and out comes back in the original space.
void dense_attn_chunk_fmt(const float* q, const void* k_cache, const void* v_cache, int k_fmt, int v_fmt, float* out,
                          int T, int pos0, int n_head, int n_kv, int head_dim, int max_ctx, float scale, void* stream);

/// y[r, :] = W x[r, :] for r in [0, rows): the batched form of dense_gemv_f32 (one warp per output row per input row).
void dense_gemv_f32_rows(const float* W, const float* X, float* y, int n_in, int n_out, int rows, void* stream);
/// ssm_alpha and ssm_beta over the same activation, one launch (the pair the GDN layer reads together).
void dense_gemv_f32_rows_pair(const float* Wa, const float* Wb, const float* X, float* ya, float* yb, int n_in,
                              int n_out, int rows, void* stream);

/// The 4-tap causal conv + SiLU over a chunk: history is [channels, 3] (oldest first, updated in place to the
/// chunk's last three inputs), qkv and h are [T, channels].  Only the SiLU output is written (the decode path keeps
/// the raw conv output in its own buffer; the prompt path does not need it).
///
/// A speculative pass feeds several columns at once and has to be able to undo the columns after any of them, which
/// means the recurrent state has to be captured after every column.  The chunk kernels walk the columns inside one
/// launch with the state already in registers, so they can write those snapshots as they go: `snap.slot[j]` receives
/// the state as it was right after column j, for j < snap.n.  Doing it here costs one extra store per column instead
/// of a separate copy per column, and it costs nothing at all when snap.n is 0.
struct DenseGdnSnap {
    static constexpr int kMaxSlots = 8;
    float* slot[kMaxSlots] = {};    ///< slot[j] = the state right after column j (state) / history after column j (conv)
    int n = 0;                      ///< how many slots are live; 0 asks for no snapshots
};

/// `l2_channels > 0` folds dense_gdn_l2_norm's call pair (q rows at base 0, k rows at base 2*128*k_heads) for the
/// leading channels into the same launch, bit-identical: channels and l2_channels must be multiples of 256.
void dense_gdn_conv_chunk(float* history, const float* qkv, const float* conv_w, float* h, int channels, int T,
                          int l2_channels = 0, float eps = 0.0f, const DenseGdnSnap& snap = {}, void* stream = nullptr);

/// In-place L2 norm of the q and k heads of every token of a chunk: row (t, r) of the norm lives at
/// h + t*channels + base + r*cols, cols = 128 (the heads of one token are contiguous inside the token's C channels).
/// The same scaling as native_gdn_l2_norm.
void dense_gdn_l2_norm(float* h, int T, int channels, int base, int rows_per_token, int cols, float eps, void* stream);

/// gate[t, h] = softplus(alpha[t, h] + dt[h]) * ssm_a[h];  beta[t, h] = sigmoid(beta[t, h])   (heads = 48)
void dense_gdn_gates(const float* alpha, const float* dt, const float* ssm_a, float* gate, float* beta, int heads,
                     int rows, void* stream);

/// The delta-rule recurrence over a chunk, state in registers, block per (value head, 32 state columns), walking the
/// T tokens in order, then the output norm y = rms_norm(o) * gamma * SiLU(z) - SiLU, not the MoE path's sigmoid.
/// h is [T, 2*S*k_heads + S*v_heads], gate/beta [T, v_heads], z and y [T, v_heads * S], state [S, v_heads, S].
/// `snap` works as in dense_gdn_conv_chunk: snap.slot[j] receives the delta-rule state right after column j.  The
/// state is read from and written to memory once per CHUNK instead of once per token, which is what makes a
/// multi-column decode pass cheap: the state is 3 MiB per layer.
void dense_gdn_rec_chunk(float* state, const float* h, const float* gate, const float* beta, const float* z,
                         const float* gamma, float eps, float* y, int k_heads, int v_heads, int T,
                         const DenseGdnSnap& snap = {}, void* stream = nullptr);

/// dense_gdn_gates + dense_gdn_rec_chunk in one launch, bit-identical to the pair: the block computes its own
/// head's gate/beta per token with the gates kernel's expressions.  `beta` is the raw ssm_beta projection.
void dense_gdn_rec_gates_chunk(float* state, const float* h, const float* alpha, const float* dt,
                               const float* ssm_a, const float* beta, const float* z, const float* gamma, float eps,
                               float* y, int k_heads, int v_heads, int T, const DenseGdnSnap& snap = {},
                               void* stream = nullptr);

}  // namespace strata::kernels
