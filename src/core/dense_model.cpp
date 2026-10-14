// src/core/dense_model.cpp - see include/strata/core/dense_model.hpp.
#include "strata/core/dense_model.hpp"

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/sampler.hpp"
#ifdef STRATA_DENSE_MMQ
#include "strata/prefill/moe_mmq.hpp"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <unordered_set>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <exception>
#include <map>
#include <regex>
#include <set>
#include <utility>

namespace strata::core {
namespace {

using namespace strata::kernels;

constexpr uint32_t kF32 = 0, kF16 = 1, kBF16 = 30;
constexpr int NC = DenseModel::kMaxCols;

// The GGUF types a quantized weight may have: exactly what native_mmvq takes.
const char* kSupported = "Q4_0, Q5_0, Q8_0, Q2_0, PQ2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS";

struct QMat {
    void* dev = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0;
    int row0 = 0;                   // the first weight row this slice holds (a matrix split across GPUs)
};

/// Which of a layer's matrices are stored in the folded (rotated) basis; see prism.hadamard.weight_names.
enum RotBit : uint16_t {
    ROT_QKV = 1u << 0, ROT_Z = 1u << 1, ROT_SSM_OUT = 1u << 2,
    ROT_Q = 1u << 3, ROT_K = 1u << 4, ROT_V = 1u << 5, ROT_O = 1u << 6,
    ROT_FFN_GATE = 1u << 7, ROT_FFN_UP = 1u << 8, ROT_FFN_DOWN = 1u << 9,
};

struct Layer {
    bool attn = false;
    uint16_t rot = 0;                               // RotBit mask of the folded matrices (0 = plain file)
    float* attn_norm = nullptr;
    float* post_norm = nullptr;
    QMat ffn_gate, ffn_up, ffn_down;
    QMat ffn_gate1, ffn_up1, ffn_down1;                 // the FFN split across GPUs: card 1's gate/up/down row slices
    // GDN mixer
    QMat qkv, z, ssm_out;
    float *conv_w = nullptr, *dt = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
    float *alpha_w = nullptr, *beta_w = nullptr;    // ssm_alpha / ssm_beta when the file stores them as floats ...
    QMat alpha_q, beta_q;                           // ... or here when it stores them quantized (UD-Q4_K_M: Q8_0)
    bool ab_quant = false;
    float *state = nullptr, *conv_state = nullptr;
    std::vector<float*> st_slots, cv_slots;         // snapshots after columns 0..draft_max-1; rollback() swaps one in (MTP only)
    // attention mixer
    QMat q, k, v, o;
    float *q_norm = nullptr, *k_norm = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
};

/// The multi-token-prediction block: a full-attention layer plus the projection of [embedding | hidden].
struct Mtp {
    QMat eh_proj;                                   // [2 * n_embd -> n_embd]
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
    QMat q, k, v, o, ffn_gate, ffn_up, ffn_down;
    float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;          // its own KV cache
    QMat head;                                      // the output head: its own, or the trunk's
};

std::string upper_type(uint32_t t) { return strata::ggml_type_name(t); }

// A tensor of one GGUF file, found by name across the shards.
struct Found {
    const strata::GgufFile* file = nullptr;
    const strata::TensorInfo* tensor = nullptr;
};

// Host-side dequantization of `n` values laid out as whole blocks (a row, or a whole small tensor).
bool dequant_row(uint32_t type, const uint8_t* src, int n, float* out) {
    switch (type) {
    case kF32: strata::dequantize_f32(src, out, n); return true;
    case kF16: strata::dequantize_f16(src, out, n); return true;
    case kBF16: strata::dequantize_bf16(src, out, n); return true;
    default: break;
    }
    int elems = 0, bytes = 0;
    if (!strata::block_geometry(type, elems, bytes) || n % elems) return false;
    for (int b = 0; b < n / elems; ++b) {
        const uint8_t* blk = src + (size_t) b * bytes;
        float* o = out + (size_t) b * elems;
        switch (type) {
        case 2: strata::dequantize_q4_0(blk, o); break;
        case 6: strata::dequantize_q5_0(blk, o); break;
        case 8: strata::dequantize_q8_0(blk, o); break;
        case 11: strata::dequantize_q3_K(blk, o); break;
        case 12: strata::dequantize_q4_K(blk, o); break;
        case 13: strata::dequantize_q5_K(blk, o); break;
        case 14: strata::dequantize_q6_K(blk, o); break;
        case 20: strata::dequantize_iq4_nl(blk, o); break;
        case 23: strata::dequantize_iq4_xs(blk, o); break;
        case 42: strata::dequantize_q2_0(blk, o); break;
        case 142: strata::dequantize_pq2_0(blk, o); break;
        default: return false;
        }
    }
    return true;
}

// llama.cpp names a split file <base>-00001-of-0000N.gguf
std::vector<std::string> shard_paths(const std::string& path) {
    static const std::regex re(R"((.*-)(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (!std::regex_match(path, m, re)) return {path};
    const int n = std::stoi(m[3].str());
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%05d", i);
        out.push_back(m[1].str() + buf + "-of-" + m[3].str() + ".gguf");
    }
    return out;
}

// integer metadata that may be a scalar or (per layer) an array: the first element
bool meta_int(const strata::GgufFile& g, const std::string& key, int64_t& out) {
    const strata::MetaValue* v = g.get(key);
    if (!v) return false;
    if (v->type == strata::MetaType::ARRAY) {
        if (v->items.empty()) return false;
        out = (int64_t) v->items[0].u;
        return true;
    }
    out = (int64_t) v->u;
    return true;
}

// Debug aid (STRATA_DENSE_DEBUG): statistics of a device vector, printed to stderr.
void trace_vec(const char* what, int64_t pos, int layer, const char* kind, const float* dev, int n, void* stream) {
    cudaStreamSynchronize((cudaStream_t) stream);
    std::vector<float> h((size_t) n);
    if (cudaMemcpy(h.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "dbg pos=%lld %s: copy failed\n", (long long) pos, what);
        return;
    }
    double mx = 0.0, sum2 = 0.0;
    int nan = 0, inf = 0;
    for (float v : h) {
        if (std::isnan(v)) { ++nan; continue; }
        if (std::isinf(v)) { ++inf; continue; }
        mx = std::max(mx, (double) std::fabs(v));
        sum2 += (double) v * v;
    }
    std::fprintf(stderr, "dbg pos=%lld layer=%2d %-4s %-7s max|v|=%-12.5g rms=%-12.5g nan=%d inf=%d\n", (long long) pos, layer,
                 kind, what, mx, std::sqrt(sum2 / std::max(1, n)), nan, inf);
}

void trace_head(const char* what, int layer, const char* kind, const float* dev, int n, void* stream) {
    cudaStreamSynchronize((cudaStream_t) stream);
    const int k = std::min(n, 8);
    float h[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (cudaMemcpy(h, dev, (size_t) k * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "dbg head %s: copy failed\n", what);
        return;
    }
    std::fprintf(stderr, "dbg layer=%d %-6s %-11s head:", layer, kind, what);
    for (int i = 0; i < k; ++i) std::fprintf(stderr, " %.9g", h[i]);
    std::fprintf(stderr, " bits:");
    for (int i = 0; i < k; ++i) {
        uint32_t b = 0;
        std::memcpy(&b, &h[i], 4);
        std::fprintf(stderr, " %08x", b);
    }
    std::fprintf(stderr, "\n");
}

void trace_logits(int64_t pos, const float* dev, int n, void* stream) {
    cudaStreamSynchronize((cudaStream_t) stream);
    std::vector<float> h((size_t) n);
    if (cudaMemcpy(h.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    std::vector<int> idx;
    int nan = 0;
    for (int i = 0; i < n; ++i) { if (std::isnan(h[(size_t) i])) ++nan; else idx.push_back(i); }
    const size_t k = std::min<size_t>(5, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + (std::ptrdiff_t) k, idx.end(),
                      [&](int a, int b) { return h[(size_t) a] > h[(size_t) b]; });
    std::fprintf(stderr, "dbg pos=%lld logits nan=%d top:", (long long) pos, nan);
    for (size_t i = 0; i < k; ++i) std::fprintf(stderr, " %d(%.3f)", idx[i], h[(size_t) idx[i]]);
    std::fprintf(stderr, "\n");
}

}  // namespace

// STRATA_DENSE_PROF=1: where the time of every run() pass goes (GPU events around the pieces, host enqueue time),
// printed to stderr every 8 passes.  The pass is synchronized at its end, so this slows decoding slightly: it is a
// measuring mode, not a way to run.
struct Prof {
    enum { GEMV_Q4K, GEMV_Q6K, GEMV_Q80, GEMV_OTHER, QUANT, GDN_LOOP, ATTN_LOOP, RMS, SWIGLU, ADD, F32ROW, EMB, NB };
    bool enabled = false, on = false;
    std::vector<cudaEvent_t> ev;
    size_t used = 0;
    struct Seg { int bucket; size_t a, b; };
    std::vector<Seg> segs;
    size_t t0 = 0;
    std::chrono::steady_clock::time_point host0;
    double ms[NB] = {}, total_ms = 0, host_ms = 0;
    long long passes = 0, cols = 0;
    size_t get() {
        if (used == ev.size()) { cudaEvent_t e = nullptr; cudaEventCreate(&e); ev.push_back(e); }
        return used++;
    }
    void begin(void* s) {
        on = true; used = 0; segs.clear();
        t0 = get();
        cudaEventRecord(ev[t0], (cudaStream_t) s);
        host0 = std::chrono::steady_clock::now();
    }
    void finish(void* s, int n) {
        host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - host0).count();
        const size_t t1 = get();
        cudaEventRecord(ev[t1], (cudaStream_t) s);
        cudaEventSynchronize(ev[t1]);
        on = false;
        float tot = 0;
        cudaEventElapsedTime(&tot, ev[t0], ev[t1]);
        total_ms += tot;
        for (const Seg& g : segs) { float m = 0; cudaEventElapsedTime(&m, ev[g.a], ev[g.b]); ms[g.bucket] += m; }
        ++passes; cols += n;
        if (passes % 8 == 0) {
            double known = 0;
            for (int i = 0; i < NB; ++i) known += ms[i];
            const double P = (double) passes;
            std::fprintf(stderr,
                         "strata-dense prof: %.1f cols/pass, GPU %.1f ms/pass = gemv Q4_K %.1f + Q6_K %.1f + Q8_0 %.1f + other %.1f"
                         " | quantize %.1f | gdn %.1f | attn %.1f | rms %.1f | swiglu %.1f | add %.1f | f32 rows %.1f"
                         " | embd/head %.1f | rest %.1f ; host enqueue %.1f ms/pass\n",
                         (double) cols / P, total_ms / P, ms[GEMV_Q4K] / P, ms[GEMV_Q6K] / P, ms[GEMV_Q80] / P, ms[GEMV_OTHER] / P,
                         ms[QUANT] / P, ms[GDN_LOOP] / P, ms[ATTN_LOOP] / P, ms[RMS] / P, ms[SWIGLU] / P, ms[ADD] / P,
                         ms[F32ROW] / P, ms[EMB] / P, (total_ms - known) / P, host_ms / P);
            for (double& m : ms) m = 0;
            total_ms = host_ms = 0; passes = cols = 0;
        }
    }
};

struct PScope {
    Prof* p; int bucket; size_t a = 0; void* s;
    PScope(Prof* pp, int b, void* st) : p(pp), bucket(b), s(st) {
        if (p->on) { a = p->get(); cudaEventRecord(p->ev[a], (cudaStream_t) s); }
    }
    ~PScope() {
        if (!p->on) return;
        const size_t e = p->get();
        cudaEventRecord(p->ev[e], (cudaStream_t) s);
        p->segs.push_back({bucket, a, e});
    }
};

struct DenseModel::Impl {
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::map<std::string, Found> index;
    std::vector<Layer> layers;
    Mtp mtp;
    std::vector<void*> allocations;                 // everything cudaMalloc'ed here, freed in the destructor
    std::vector<void*> host_allocs;                 // weights that did not fit in VRAM: mapped pinned host memory
    uint64_t budget = UINT64_MAX;                   // bytes of weights that may go to VRAM
    uint64_t dev_used = 0, host_bytes = 0;
    int host_tensors = 0;
    struct Ckpt { uint8_t* host = nullptr; int64_t pos = -1; };
    std::vector<Ckpt> ckpt;
    uint64_t ckpt_bytes = 0;
    long long dbg_from = -1, dbg_to = -1;           // STRATA_DENSE_DEBUG=A:B traces the steps with A <= position < B
    Prof prof;                                      // STRATA_DENSE_PROF=1
    int kfmt = DENSE_KV_F16, vfmt = DENSE_KV_F16;   // how the KV cache stores K and V (--kv)
    cudaStream_t stream = nullptr;
    cudaEvent_t embed_done = nullptr;

    // a token embedding table: rows are dequantized on the host and staged through pinned memory
    struct EmbdSrc {
        Found f;
        uint32_t type = 0;
        size_t row_bytes = 0;
    };
    EmbdSrc embd, embd_mtp;
    float* embd_stage = nullptr;                    // pinned, NC * n_embd floats

    // ---- PrismML's folded basis (Bonsai 2's `prism.hadamard.*` metadata).  When a weight is listed there it
    // is stored as D = W R^T, and its *activation* has to carry R (signs, then the block Walsh-Hadamard) before
    // the GEMV; token_embd carries R^T on its output instead.
    bool hadamard = false;                          // the file declares the folded basis
    int rot_block = 0;                              // its block size (the kernels implement 1024)
    bool rot_inverse_embd = false;                  // token_embd.weight is in inverse_weight_names
    bool rot_gdn_v_grouped = false;                 // prism.hadamard.gdn_v_grouped
    bool head_rot = false;                          // output.weight is folded
    struct RotSigns { int width; float* dev; };
    std::vector<RotSigns> rot_signs;                // one +/-1 table per rotated width, on the device
    std::vector<void*> rot_allocs;                  // their cudaMalloc'ed storage
    float* rot_scratch = nullptr;                   // NC * max width floats
    const float* ham_signs(int width) const {
        if (!hadamard) return nullptr;
        for (const RotSigns& s : rot_signs) if (s.width == width) return s.dev;
        return nullptr;
    }

    // ---- phase 1 of the split across GPUs: the FFN's gate/up/down on a second card.  Card 0 keeps the whole
    // model and computes its share (attention/GDN/embeddings plus the first ffn_r0 FFN rows and the first half of
    // the down output rows); card 1 owns the remaining FFN rows and receives the rotated activation per layer.
    bool ffn_split = false;
    int ffn_r0 = 0;                                 // card 0's FFN row count; card 1 gets [ffn_r0, n_ff)
    bool peer_ok = false;                           // direct GPU-to-GPU DMA is available (large-BAR + p2p)
    int split_trace = -1;                            // STRATA_DENSE_FFN_TRACE=<layer>: trace the FFN split's parts
    bool split_check = false;                        // STRATA_DENSE_FFN_SELFCHECK=1: compare with the plain FFN per layer
    float* dbg_mix = nullptr;                        // mapped host: the self-check's copy of the split's output
    float* dbg_b = nullptr;                          // mapped host: the self-check's copy of the reference output
    float* dbg_x = nullptr;                          // mapped host: the self-check's copy of the pre-add residual
    int cur_layer = -1;                             // the layer run() is on (for the split trace)
    // Every payload is staged in MAPPED PINNED HOST memory: the producer's kernels write it with a system
    // fence plus a block-wide barrier before the doorbell, the consumer reads it back with its own kernel.
    // The kernels always use the HOST pointers -- a device pointer obtained on the other card is not valid
    // there, while the host pointer itself is, under unified addressing.
    float* hx_stage = nullptr;                      // pinned: the activation on its way from card 0 to card 1
    float* hh0_stage = nullptr;                     // pinned: card 0's rotated half on its way to card 1
    float* hh1_stage = nullptr;                     // pinned: card 1's rotated half on its way to card 0
    float* hm1_stage = nullptr;                     // pinned: card 1's down output rows on their way to card 0
    float* mix_lo = nullptr;                        // card 0: its half of the down output rows, compact until scattered
    float* h_full = nullptr;                        // card 0: the assembled rotated FFN activation (n_ff wide)
    float* h_full1 = nullptr;                       // card 1: the same assembled activation
    float* h_peer0 = nullptr;                       // card 1: card 0's pulled rotated half
    float* h_peer = nullptr;                        // card 0: the pulled peer half (r1 wide, compact)
    uint32_t* split_bell = nullptr;                 // pinned host: [0] card 0's doorbell, [64] card 1's
    uint32_t* split_bell_dev = nullptr;             // the same memory through a device pointer
    uint32_t split_seq = 0;                         // monotonic handoff counter (both cards share it)
    cudaEvent_t ev_x = nullptr, ev_h0 = nullptr, ev_h1 = nullptr, ev_o = nullptr;   // ring events (EVWAIT)
    cudaStream_t stream1 = nullptr;
    float* dx_dev = nullptr;                        // card 0: card 1's pulled down output, compact
    float* mix1 = nullptr;                          // card 1: its down output
    void* q8_1_1 = nullptr;                         // card 1: activation scratch
    float* x1 = nullptr;                            // card 1: the received activation
    float* ffn_g1 = nullptr, *ffn_u1 = nullptr, *ffn_h1 = nullptr, *h1_rot = nullptr;
    float* signs1 = nullptr;                        // card 1: the n_ff sign table from ffn_r0 on
    std::vector<float> signs_ffn_host;              // host copy of the n_ff signs, for signs1 at load
    std::vector<void*> allocs1;                     // card 1 allocations, freed on device 1

    float* output_norm = nullptr;
    QMat head;
    void* q8_1 = nullptr;                           // shared activation scratch of the native GEMVs
    void* q8_1_ab = nullptr;                        // the GDN layer's second scratch: q8_1 of the un-rotated xn
    int64_t max_context = 0;

    // ---- HIP graphs (STRATA_DENSE_GRAPH=1): the whole single-card decode pass is captured once per attention
    // split count and replayed.  The token's embedding and position are staged in pinned host memory that the
    // pass's own memcpy nodes read, so no launch argument changes between replays.  See run_graphed().
    bool graph_on = false;
    std::vector<cudaGraphExec_t> graph_exec;        // [bucket], bucket = splits_for(pos0 + 1); lazily captured
    int32_t* pos_slot = nullptr;                    // device: this pass's pos0, read by rope/kv/attention
    int32_t* pos_slot_host = nullptr;               // pinned source of the slot's memcpy node

    // ---- the batched prompt path (prefill_setup() / prefill()); see the block definitions below.
#ifdef STRATA_DENSE_MMQ
    struct Pf {
        int64_t T = 0;
        strata::prefill::mmq::Context ctx;      // MMQ keeps a small scratch pool for its stream-k fixup
        float *x = nullptr, *xn = nullptr, *mix = nullptr;
        float *qkv = nullptr, *h = nullptr, *alpha = nullptr, *beta = nullptr, *gate = nullptr;
        float *z = nullptr, *y = nullptr;
        float *q_full = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr;
        float *attn = nullptr, *attn32 = nullptr;
        float *ffn_g = nullptr, *ffn_u = nullptr, *ffn_h = nullptr;
        float* stage = nullptr;                 // pinned embedding stage, T rows
        int32_t* pos_dev = nullptr;             // the position of every row, for RoPE
        void* xq = nullptr;                     // the q8_1 activations MMQ multiplies with
        int32_t* ids = nullptr;                 // iota: MMQ writes each row back to its own index
        int32_t* bounds = nullptr;              // {0, T}: one group, the whole chunk
        // Pinned twins of bounds, cycled so a chunk never overwrites a copy the stream has not read yet.
        static constexpr int kBoundsSlots = 8;
        int32_t* bounds_host = nullptr;
        int bounds_slot = 0;
        std::vector<void*> allocs;              // this arena's cudaMalloc'ed buffers (freed by ~Pf)

        ~Pf() {
            for (void* p : allocs) cudaFree(p);
            if (stage) cudaFreeHost(stage);
            if (bounds_host) cudaFreeHost(bounds_host);
        }
    };
    std::unique_ptr<Pf> pf;
#endif

    // activations (device, f32), NC columns each
    float *x = nullptr, *xn = nullptr, *mix = nullptr;
    float *qkv = nullptr, *h = nullptr, *alpha = nullptr, *beta = nullptr, *gate = nullptr;
    float *z = nullptr, *y = nullptr;
    float *q_full = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr, *attn32 = nullptr;
    float *ffn_g = nullptr, *ffn_u = nullptr, *ffn_h = nullptr;
    float* logits = nullptr;
    float* attn_scratch = nullptr;
    int32_t* pos_dev = nullptr;                     // one position per row, for NC columns' worth of rows
    // The rope configuration the analytic kernel is launched with (rope_scaling.hpp).  Seeded from the
    // process config and re-based on the file's rope.freq_base at load(): the dense path has no CLI rope
    // knobs, so this stays `none` (the identity) unless something else has set a scaling.
    RopeScaling rope;
    // the hidden state and the MTP side
    float *hid = nullptr;                           // post-norm hidden of the last run(), NC columns
    float *h_last = nullptr;                        // post-norm hidden at the last fed position
    float *m_e = nullptr, *hin = nullptr, *cat = nullptr, *mtp_logits = nullptr;
    float* mtp_h = nullptr;                         // the MTP block's own output hidden (chains the next draft step)
    int* d_tok = nullptr;
    float* d_prob = nullptr;

    // ---- helpers
    template <class T> bool alloc(T** out, uint64_t bytes, std::string& err, const char* what) {
        void* p = nullptr;
        const cudaError_t s = cudaMalloc(&p, bytes ? bytes : 16);
        if (s != cudaSuccess) {
            err = std::string("dense model: cannot allocate ") + what + " (" + std::to_string(bytes >> 20) + " MiB): " +
                  cudaGetErrorString(s);
            return false;
        }
        allocations.push_back(p);
        *out = (T*) p;
        return true;
    }

    const Found* find(const std::string& name) const {
        auto it = index.find(name);
        return it == index.end() ? nullptr : &it->second;
    }
    const Found* find_any(std::initializer_list<const char*> names, const std::string& prefix) const {
        for (const char* n : names)
            if (const Found* f = find(prefix + n)) return f;
        return nullptr;
    }

    // A float vector or matrix (F32, F16, BF16 or any type the host can dequantize) uploaded as F32.
    // A row range of a quantized matrix, allocated and filled on `device` (one side of a split across GPUs).
    // Rows are contiguous in the GGUF, so a slice is one memcpy from the mapped file.
    bool upload_mat_slice(const Found* f, const std::string& name, int n_in, int n_out, int row0, int rows, int device,
                          QMat& out, std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        const strata::TensorInfo& t = *f->tensor;
        if (t.shape.size() != 2 || t.shape[0] != (uint64_t) n_in || t.shape[1] != (uint64_t) n_out) {
            err = "dense model: " + name + " has an unexpected shape";
            return false;
        }
        if (!native_mmvq_supported((int) t.type)) {
            err = "dense model: " + name + " is " + upper_type(t.type) + ", which the native GEMVs do not take";
            return false;
        }
        if (row0 < 0 || rows <= 0 || row0 + rows > n_out) { err = "dense model: " + name + ": bad row range"; return false; }
        const uint64_t row_bytes = native_mmvq_weight_bytes((int) t.type, n_in, 1);
        const uint64_t bytes = row_bytes * (uint64_t) rows;
        const uint64_t off = row_bytes * (uint64_t) row0;
        const uint64_t payload = f->file->file_size() - f->file->data_start();
        if (t.offset > payload || off + bytes > payload - t.offset) {
            err = "dense model: " + name + ": slice outside the file";
            return false;
        }
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(device);
        void* dev = nullptr;
        cudaError_t s = cudaMalloc(&dev, bytes);
        if (s == cudaSuccess) s = cudaMemcpy(dev, f->file->tensor_data(t) + off, bytes, cudaMemcpyHostToDevice);
        cudaSetDevice(prev);
        if (s != cudaSuccess) {
            if (dev) {
                cudaSetDevice(device);
                cudaFree(dev);
                cudaSetDevice(prev);
            }
            err = "dense model: " + name + " slice does not fit on device " + std::to_string(device);
            return false;
        }
        out.dev = dev;
        out.type = (int) t.type;
        out.n_in = n_in;
        out.n_out = rows;
        out.row0 = 0;
        (device == 0 ? allocations : allocs1).push_back(dev);
        return true;
    }

    // Input-column range of a quantized matrix, packed so the slice keeps the standard row layout: one memcpy
    // per output row.  col0 and cols must be whole blocks (128 values for PQ2_0) and, for the folded basis, a
    // multiple of the 1024-wide rotation block so the slice can be rotated on its own.
    bool upload_mat_slice_cols(const Found* f, const std::string& name, int n_in, int n_out, int col0, int cols,
                               int device, QMat& out, std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        const strata::TensorInfo& t = *f->tensor;
        if (t.shape.size() != 2 || t.shape[0] != (uint64_t) n_in || t.shape[1] != (uint64_t) n_out) {
            err = "dense model: " + name + " has an unexpected shape";
            return false;
        }
        if (!native_mmvq_supported((int) t.type)) {
            err = "dense model: " + name + " is " + upper_type(t.type) + ", which the native GEMVs do not take";
            return false;
        }
        if (col0 < 0 || cols <= 0 || col0 + cols > n_in || col0 % 128 || cols % 128 || n_in % 128) {
            err = "dense model: " + name + ": bad column range";
            return false;
        }
        const uint64_t src_row = native_mmvq_weight_bytes((int) t.type, n_in, 1);
        const uint64_t dst_row = native_mmvq_weight_bytes((int) t.type, cols, 1);
        const uint64_t blk = src_row / (uint64_t) (n_in / 128);            // bytes of one 128-value block
        const uint64_t off = (uint64_t) (col0 / 128) * blk;
        const uint64_t payload = f->file->file_size() - f->file->data_start();
        if (t.offset > payload || (uint64_t) (n_out - 1) * src_row + off + dst_row > payload - t.offset) {
            err = "dense model: " + name + ": slice outside the file";
            return false;
        }
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(device);
        void* dev = nullptr;
        cudaError_t s = cudaMalloc(&dev, dst_row * (uint64_t) n_out);
        const uint8_t* base = f->file->tensor_data(t) + off;
        for (int r = 0; s == cudaSuccess && r < n_out; ++r)
            s = cudaMemcpy((uint8_t*) dev + (uint64_t) r * dst_row, base + (uint64_t) r * src_row, dst_row,
                           cudaMemcpyHostToDevice);
        cudaSetDevice(prev);
        if (s != cudaSuccess) {
            if (dev) {
                cudaSetDevice(device);
                cudaFree(dev);
                cudaSetDevice(prev);
            }
            err = "dense model: " + name + " slice does not fit on device " + std::to_string(device);
            return false;
        }
        out.dev = dev;
        out.type = (int) t.type;
        out.n_in = cols;
        out.n_out = n_out;
        out.row0 = 0;
        (device == 0 ? allocations : allocs1).push_back(dev);
        return true;
    }

    bool upload_floats(const Found* f, const std::string& name, uint64_t expect, float** out, std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        uint64_t n = 1;
        for (uint64_t d : f->tensor->shape) n *= d;
        if (expect && n != expect) {
            err = "dense model: " + name + " has " + std::to_string(n) + " values, expected " + std::to_string(expect);
            return false;
        }
        const uint8_t* src = f->file->tensor_data(*f->tensor);
        std::vector<float> host(n);
        if (n > (uint64_t) INT_MAX || !dequant_row(f->tensor->type, src, (int) n, host.data())) {
            err = "dense model: " + name + " is " + upper_type(f->tensor->type) +
                  ", which cannot be read as floats (expected F32, F16, BF16 or a quantized type)";
            return false;
        }
        if (!alloc(out, n * 4, err, name.c_str())) return false;
        const cudaError_t s = cudaMemcpy(*out, host.data(), n * 4, cudaMemcpyHostToDevice);
        if (s != cudaSuccess) { err = "dense model: upload " + name + ": " + cudaGetErrorString(s); return false; }
        return true;
    }

    // A quantized matrix, kept in its GGUF blocks: in VRAM while the budget lasts, in mapped host memory after.
    bool upload_mat(const Found* f, const std::string& name, int n_in, int n_out, QMat& out, uint64_t& total,
                    std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        const strata::TensorInfo& t = *f->tensor;
        if (t.shape.size() != 2 || t.shape[0] != (uint64_t) n_in || t.shape[1] != (uint64_t) n_out) {
            err = "dense model: " + name + " has shape [" + std::to_string(t.shape.empty() ? 0 : t.shape[0]) + ", " +
                  std::to_string(t.shape.size() > 1 ? t.shape[1] : 0) + "], expected [" + std::to_string(n_in) + ", " +
                  std::to_string(n_out) + "]";
            return false;
        }
        if (!native_mmvq_supported((int) t.type)) {
            err = "dense model: " + name + " is " + upper_type(t.type) + ", which the native GEMVs do not take. "
                  "Supported: " + kSupported;
            return false;
        }
        int elems = 0, block_bytes = 0;
        if (!strata::block_geometry(t.type, elems, block_bytes) || (uint64_t) n_in % (uint64_t) elems) {
            err = "dense model: " + name + ": invalid block geometry"; return false;
        }
        const uint64_t bytes = native_mmvq_weight_bytes((int) t.type, n_in, n_out);
        const uint64_t in_file = (uint64_t) n_in * n_out / (uint64_t) elems * (uint64_t) block_bytes;
        const uint64_t payload = f->file->file_size() - f->file->data_start();
        if (bytes != in_file || t.offset > payload || bytes > payload - t.offset) {
            err = "dense model: " + name + ": payload size mismatch or truncated file"; return false;
        }
        void* dev = nullptr;
        cudaError_t s = cudaSuccess;
        if (dev_used + bytes <= budget) {
            s = cudaMalloc(&dev, bytes);
            if (s == cudaSuccess) s = cudaMemcpy(dev, f->file->tensor_data(t), bytes, cudaMemcpyHostToDevice);
            if (s != cudaSuccess) {                 // the card is fuller than cudaMemGetInfo said: spill this one
                if (dev) cudaFree(dev);
                dev = nullptr;
                cudaGetLastError();
            } else {
                allocations.push_back(dev);
                dev_used += bytes;
            }
        }
        if (!dev) {
            // Host-resident: pinned + mapped, so the GEMV kernel reads the blocks over PCIe on every token.
            void* host = nullptr;
            s = cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
            if (s == cudaSuccess) s = cudaHostGetDevicePointer(&dev, host, 0);
            if (s != cudaSuccess) {
                if (host) cudaFreeHost(host);
                err = "dense model: " + name + " fits neither in VRAM nor in pinned host memory (" +
                      std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(s);
                return false;
            }
            std::memcpy(host, f->file->tensor_data(t), bytes);
            host_allocs.push_back(host);
            host_bytes += bytes;
            ++host_tensors;
        }
        out = QMat{dev, (int) t.type, n_in, n_out};
        total += bytes;
        return true;
    }

    bool setup_embd(EmbdSrc& e, const Found* f, int n_embd, std::string& err) {
        e.f = *f;
        e.type = f->tensor->type;
        int elems = 0, bytes = 0;
        if (!strata::block_geometry(e.type, elems, bytes) || n_embd % elems) {
            err = "dense model: " + std::string("embedding table has an unsupported type ") + upper_type(e.type);
            return false;
        }
        e.row_bytes = (size_t) n_embd / (size_t) elems * (size_t) bytes;
        std::vector<float> probe((size_t) n_embd);      // a dry run on row 0 tells now whether the host knows the type
        if (!dequant_row(e.type, f->file->tensor_data(*f->tensor), n_embd, probe.data())) {
            err = "dense model: cannot dequantize the embedding table of type " + upper_type(e.type);
            return false;
        }
        return true;
    }

    // The host half of the embedding lookup: n rows dequantized into the pinned stage (no device work).
    bool embed_stage_rows(const EmbdSrc& src, const int32_t* toks, int n, int E, int n_vocab, float* st, std::string& err) {
        for (int j = 0; j < n; ++j) {
            if (toks[j] < 0 || toks[j] >= n_vocab) { err = "dense model: token id outside the vocabulary"; return false; }
            const uint8_t* row = src.f.file->tensor_data(*src.f.tensor) + (size_t) toks[j] * src.row_bytes;
            if (!dequant_row(src.type, row, E, st + (size_t) j * E)) { err = "dense model: embedding dequantization"; return false; }
        }
        return true;
    }
    // The device half: stage -> dst, then R^T for a folded table.  The caller must not overwrite the stage until
    // embed_done (recorded after this) has fired.
    void enqueue_embed(const float* st, float* dst, int n, int E, void* s) {
        cudaMemcpyAsync(dst, st, (size_t) n * E * 4, cudaMemcpyHostToDevice, (cudaStream_t) s);
        if (rot_inverse_embd) {
            // token_embd is stored folded too (inverse_weight_names): the lookup lands in the rotated basis, so
            // the table's output carries R^T = H/32 then the signs (Prism's fwht(x, inverse)).
            dense_fwht_rows(dst, dst, ham_signs(E), E, n, true, (cudaStream_t) s);
        }
    }
    // n embedding rows -> device, one copy; the pinned stage is reused only after the previous copy finished
    bool upload_embeddings(const EmbdSrc& src, const int32_t* toks, int n, int E, int n_vocab, float* dst,
                           std::string& err, float* stage = nullptr) {
        float* st = stage ? stage : embd_stage;
        if (cudaEventSynchronize(embed_done) != cudaSuccess) { err = "dense model: embedding event"; return false; }
        if (!embed_stage_rows(src, toks, n, E, n_vocab, st, err)) return false;
        enqueue_embed(st, dst, n, E, stream);
        cudaEventRecord(embed_done, stream);
        return true;
    }

    // y = W x for n columns; the activation must already be quantized (quantize()) with the same n.  `act` picks a
    // different q8_1 scratch (the GDN layer's plain-quantized xn lives in one while q8_1 holds the rotated one).
    void gemv(const QMat& w, float* y, int n, void* s, const void* act = nullptr) {
        PScope ps(&prof, w.type == 12 ? Prof::GEMV_Q4K : w.type == 14 ? Prof::GEMV_Q6K : w.type == 8 ? Prof::GEMV_Q80 : Prof::GEMV_OTHER, s);
        const uint8_t* base = (const uint8_t*) w.dev;
        if (w.row0) base += (size_t) w.row0 * native_mmvq_weight_bytes(w.type, w.n_in, 1);
        native_mmvq(w.type, base, act ? act : q8_1, y, w.n_in, w.n_out, n, s);
    }
    void quantize(const float* x, int n_in, int n, void* s) {
        PScope ps(&prof, Prof::QUANT, s);
        native_quantize_q8_1(x, q8_1, n_in, n, s);
    }
    // The folded-basis rotation and its q8_1 quantization as one launch (bit-identical to the pair).  Booked as
    // QUANT; `rot_out` keeps the FFN trace's rotated row alive when it is on.
    void fwht_quantize(const float* x, int width, int n, const float* signs, float* rot_out, int hd, int nk, int rep,
                       void* s) {
        PScope ps(&prof, Prof::QUANT, s);
        dense_fwht_quantize_rows(x, q8_1, rot_out, signs, width, n, hd, nk, rep, s);
    }

    // ---- the blocks.  Input is I.xn (the normalized residual) unless stated; the output goes to I.mix.
    // `pre_rot` for a GDN layer of a quantized-alpha/beta file means the caller's rms already produced BOTH
    // quantizations of its output (plain in q8_1_ab for alpha/beta, folded in q8_1 for qkv/z).
    void gdn_block(const DenseConfig& c, Layer& L, int n, bool snapshot, void* s, bool pre_rot = false);
    void attn_block(const DenseConfig& c, const QMat& wq, const QMat& wk, const QMat& wv, const QMat& wo,
                    const float* qn, const float* kn, uint16_t* kc, uint16_t* vc, int n, int pos0,
                    const int32_t* pos_slot, uint16_t rot, void* s, bool pre_rot = false);
    void ffn_block(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d, const float* post_norm, int n,
                   uint16_t rot, void* s, const float* pre_add = nullptr);
    void ffn_selfcheck(const DenseConfig& c, const Layer& L, int n, uint16_t rot, void* s);
    void ffn_block_split(const DenseConfig& c, const Layer& L, int n, uint16_t rot, void* s,
                         const float* pre_add = nullptr);   // phase 1: FFN across GPUs
    bool mtp_block(const DenseConfig& c, int n, int pos0, bool want_logits, void* s);

    // ---- the HIP-graph path (STRATA_DENSE_GRAPH=1)
    bool enqueue_pass(const DenseConfig& c, int n, int64_t pos0, Logits lg, bool keep_hidden, bool snapshot,
                      const int32_t* pos_slot, void* s);
    bool run_graphed(const DenseConfig& c, int32_t token, int64_t pos0, std::string& err);
    bool capture_graph(const DenseConfig& c, int bucket, int64_t pos0, std::string& err);

#ifdef STRATA_DENSE_MMQ
    // ---- the same blocks over a chunk of T tokens, reading and writing Pf instead of the NC-column buffers.
    void pf_quantize(const float* src, int wtype, int n_in, int64_t T, void* s);
    void pf_gemv(const QMat& w, float* y, int64_t T, void* s);
    void pf_gdn_block(const DenseConfig& c, Layer& L, int64_t T, void* s);
    void pf_attn_block(const DenseConfig& c, const QMat& wq, const QMat& wk, const QMat& wv, const QMat& wo,
                       const float* qn, const float* kn, uint16_t* kc, uint16_t* vc, int64_t T, int pos0, void* s);
    void pf_ffn_block(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d, const float* post_norm,
                      int64_t T, void* s);
#endif
};

// The GDN mixer over n columns.  The recurrence is order-dependent, but "order-dependent" does not mean "one launch
// per column": the chunk kernels walk the columns inside a single launch with the state in registers, so n columns
// cost the same handful of launches as one.  That matters twice over here.  The delta-rule state is 3 MiB per layer,
// and the per-column kernels read and write it once per column; over 48 GDN layers and a 3-column speculative pass
// that is 446 MB of traffic that a chunk does not have.  And the rollback snapshots, which used to be two 3 MiB
// copies per column, are now stores the recurrence makes from registers it already holds.
void DenseModel::Impl::gdn_block(const DenseConfig& c, Layer& L, int n, bool snapshot, void* s, bool pre_rot) {
    const int C = c.conv_channels(), V = c.value_dim(), E = c.n_embd;
    const int S = c.ssm_state, KH = c.ssm_k_heads, VH = c.ssm_v_heads, qk = S * KH;
    if (L.ab_quant) {                               // the scratch holds q8_1(xn) for the two quantized projections
        if (pre_rot) {                              // the caller's rms wrote q8_1_ab for these two
            gemv(L.alpha_q, alpha, n, s, q8_1_ab);
            gemv(L.beta_q, beta, n, s, q8_1_ab);
        } else {
            quantize(xn, E, n, s);
            gemv(L.alpha_q, alpha, n, s);
            gemv(L.beta_q, beta, n, s);
        }
    } else {
        // The file stores alpha/beta as floats: the pair reads `xn` itself, and the rotation below overwrites
        // q8_1 -- quantizing xn first (as this used to) was a dead launch on every GDN layer of such a file.
        PScope ps(&prof, Prof::F32ROW, s);
        dense_gemv_f32_rows_pair(L.alpha_w, L.beta_w, xn, alpha, beta, E, VH, n, s);
    }
    // the folded basis: qkv and z read the hidden through R = (1/sqrt n) H diag(S); alpha/beta do not.  With the
    // quantized alpha/beta, the pre-block rms already produced BOTH quantizations of its own output, so `pre_rot`
    // means "q8_1 holds the folded row and q8_1_ab the plain one" here.
    if (L.rot & (ROT_QKV | ROT_Z)) {
        if (!pre_rot) fwht_quantize(xn, E, n, ham_signs(E), nullptr, 0, 0, 0, s);   // else the caller's rms did it
    } else if (!L.ab_quant) {
        quantize(xn, E, n, s);                      // not folded: the qkv/z GEMVs' activation is q8_1(xn)
    }
    gemv(L.qkv, qkv, n, s);
    gemv(L.z, z, n, s);
    {
    PScope ps_gdn(&prof, Prof::GDN_LOOP, s);
    // slot[j] receives the state right after column j, which is what rollback(j) restores.  run() guarantees
    // n - 1 <= draft_max, so the slots allocated at load() are exactly enough.
    DenseGdnSnap snaps, snapc;
    if (snapshot) {
        for (int j = 0; j < n - 1; ++j) { snaps.slot[j] = L.st_slots[(size_t) j]; snapc.slot[j] = L.cv_slots[(size_t) j]; }
        snaps.n = n - 1;
        snapc.n = n - 1;
    }
    // conv + the q/k L2 norms + gates + the recurrence, in two launches: the conv kernel normalizes its own
    // 128-wide head rows and the recurrence computes its own per-token gate/beta, both with the separate
    // kernels' expressions in the same order.
    dense_gdn_conv_chunk(L.conv_state, qkv, L.conv_w, h, C, n, 2 * qk, c.rms_eps, snapc, s);
    dense_gdn_rec_gates_chunk(L.state, h, alpha, L.dt, L.ssm_a, beta, z, L.ssm_norm, c.rms_eps, y, KH, VH, n, snaps, s);
    }
    if (L.rot & ROT_SSM_OUT) {
        // Prism's grouped V layout: [hd, nk, rep] rows -> [hd, rep, nk] at the load of the same transform.
        if (rot_gdn_v_grouped) fwht_quantize(y, V, n, ham_signs(V), nullptr, V / VH, KH, VH / KH, s);
        else fwht_quantize(y, V, n, ham_signs(V), nullptr, 0, 0, 0, s);
    } else {
        quantize(y, V, n, s);
    }
    gemv(L.ssm_out, mix, n, s);
}

void DenseModel::Impl::attn_block(const DenseConfig& c, const QMat& wq, const QMat& wk, const QMat& wv, const QMat& wo,
                                  const float* qn, const float* kn, uint16_t* kc, uint16_t* vc, int n, int pos0,
                                  const int32_t* pos_slot, uint16_t rot, void* s, bool pre_rot) {
    const int E = c.n_embd, H = c.n_head, HK = c.n_head_kv, D = c.head_dim;
    if (rot & (ROT_Q | ROT_K | ROT_V)) {
        if (!pre_rot) fwht_quantize(xn, E, n, ham_signs(E), nullptr, 0, 0, 0, s);   // else the caller's rms did it
    } else {
        quantize(xn, E, n, s);
    }
    gemv(wq, q_full, n, s);
    gemv(wk, kcur, n, s);
    gemv(wv, vcur, n, s);
    const float scale = 1.0f / std::sqrt((float) D);
    {
    PScope ps_attn(&prof, Prof::ATTN_LOOP, s);
    // Everything up to the attention itself is per-row work with no dependency between rows, so it runs over all n
    // columns at once: the strided copy that splits q from its output gate, the per-head norms, RoPE and the KV
    // append are one launch each instead of one per column.  The attention stays the split-K decode kernel, which
    // parallelises over the cache - the one part that wants the columns kept apart.
    dense_split_q_rms(q_full, qn, qcur, n * H, D, c.rms_eps, s);
    rms_norm_weighted(kcur, kn, n * HK, D, c.rms_eps, s);
    // The position of a row is its column, computed inside the rope launch (rows / n_head for q, rows / n_kv for
    // k): no positions buffer, no two extra launches.  The prompt path keeps the explicit buffer.
    if (!pos_slot) {
        native_rope_apply_rows(qcur, qcur, n * H, D, c.n_rot, rope, H, pos0, s);
        native_rope_apply_rows(kcur, kcur, n * HK, D, c.n_rot, rope, HK, pos0, s);
        dense_kv_append_rows_fmt(kc, vc, kfmt, vfmt, kcur, vcur, n, pos0, HK, D, (int) max_context, s);
        for (int j = 0; j < n; ++j) {
            // The merge writes the gated output straight to attn32: the gate is row h's second half in q_full.
            dense_attn_decode_gate_fmt(qcur + (size_t) j * H * D, kc, vc, kfmt, vfmt, q_full + (size_t) j * H * 2 * D,
                                       attn32 + (size_t) j * H * D, attn_scratch, H, HK, D, pos0 + j + 1,
                                       (int) max_context, scale, s);
        }
    } else {
        // The graph replay: this pass's pos0 lives in the device slot the captured pass refills, and the split
        // grid is the capture bucket (the kernel still reads the real n_ctx from the slot).
        native_rope_apply_rows_dev(qcur, qcur, n * H, D, c.n_rot, rope, H, pos_slot, s);
        native_rope_apply_rows_dev(kcur, kcur, n * HK, D, c.n_rot, rope, HK, pos_slot, s);
        dense_kv_append_rows_dev_fmt(kc, vc, kfmt, vfmt, kcur, vcur, n, pos_slot, HK, D, (int) max_context, s);
        for (int j = 0; j < n; ++j) {
            dense_attn_decode_gate_dev_fmt(qcur + (size_t) j * H * D, kc, vc, kfmt, vfmt, q_full + (size_t) j * H * 2 * D,
                                           attn32 + (size_t) j * H * D, attn_scratch, H, HK, D,
                                           (pos0 + j + 1 + 255) / 256, pos_slot, j, (int) max_context, scale, s);
        }
    }
    }
    if (rot & ROT_O) {
        fwht_quantize(attn32, H * D, n, ham_signs(H * D), nullptr, 0, 0, 0, s);
    } else {
        quantize(attn32, H * D, n, s);
    }
    gemv(wo, mix, n, s);
}

// mix = ffn(rms_norm(x + pre_add, post_norm)) for n columns; the caller folds the mix into x through the next
// block's rms (see enqueue_pass: the residual add is deferred, one launch per layer less).
void DenseModel::Impl::ffn_block(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d, const float* post_norm,
                                 int n, uint16_t rot, void* s, const float* pre_add) {
    const int E = c.n_embd;
    if (rot & (ROT_FFN_GATE | ROT_FFN_UP)) {
        // rms + rotation + q8_1 in one launch, with the pending mixed-block add folded in.
        PScope ps(&prof, Prof::QUANT, s);
        dense_rms_fwht_quant_rows(x, pre_add, post_norm, xn, q8_1, ham_signs(E), E, n, c.rms_eps, s);
    } else {
        {
            PScope ps(&prof, Prof::RMS, s);
            dense_rms_norm(x, post_norm, xn, n, E, c.rms_eps, s, pre_add);
        }
        quantize(xn, E, n, s);
    }
    gemv(g, ffn_g, n, s);
    gemv(u, ffn_u, n, s);
    const bool trb = split_trace == cur_layer && cur_layer >= 0;
    if (trb) trace_vec("ffn_g", -1, cur_layer, "single", ffn_g, c.n_ff, s);
    {
        PScope ps(&prof, Prof::SWIGLU, s);
        // swiglu + (the rotation) + the q8_1 the down GEMV reads, in one launch; the trace's h/rot rows come
        // back through h_out/rot_out.
        if (rot & ROT_FFN_DOWN)
            dense_swiglu_quantize_rows(ffn_g, ffn_u, q8_1, ham_signs(c.n_ff), c.n_ff, n, trb ? ffn_h : nullptr,
                                       trb ? rot_scratch : nullptr, s);
        else if (c.n_ff % 1024 == 0)
            dense_swiglu_quantize_rows(ffn_g, ffn_u, q8_1, nullptr, c.n_ff, n, trb ? ffn_h : nullptr, nullptr, s);
        else {
            dense_swiglu(ffn_g, ffn_u, ffn_h, (int64_t) n * c.n_ff, s);
            quantize(ffn_h, c.n_ff, n, s);
        }
    }
    if (trb) trace_vec("ffn_h", -1, cur_layer, "single", ffn_h, c.n_ff, s);
    if (trb && (rot & ROT_FFN_DOWN)) trace_vec("ffn_hrot", -1, cur_layer, "single", rot_scratch, c.n_ff, s);
    gemv(d, mix, n, s);
    if (trb) trace_vec("ffn_mix", -1, cur_layer, "single", mix, E, s);
    if (trb && n > 1) trace_vec("ffn_mix_c1", -1, cur_layer, "single", mix + E, E, s);
    if (trb) trace_head("ffn_mix", cur_layer, "single", mix, E, s);
    if (trb) trace_head("ffn_mix_r0", cur_layer, "single", mix + E / 2, E - E / 2, s);
    if (trb) trace_head("ffn_hrot_r0", cur_layer, "single", rot_scratch + ffn_r0, c.n_ff - ffn_r0, s);
    if (trb) trace_head("ffn_g_r0", cur_layer, "single", ffn_g + ffn_r0, c.n_ff - ffn_r0, s);
    if (trb) trace_head("ffn_vs", cur_layer, "single", rot_scratch, c.n_ff, s);
    // The residual add is DEFERRED: x += mix happens inside the next block's input rms (or the output norm),
    // whose `add` argument is this mix.  Nothing else reads x between here and there.
}

// Debug (STRATA_DENSE_FFN_SELFCHECK=1): the same layer, computed twice.  `mix` holds the split's own down output;
// this saves it and the pre-add residual, recomputes the layer with the plain single-card block, and reports the
// first element that differs -- with its column and row, which is what separates "the wrong rows" from "the wrong
// activation" from "the wrong stride".  Then it restores x = pre-add residual + the SPLIT's output, so the pass
// continues exactly as it would have.  Two extra FFN passes and three n*E copies per layer: a debug mode only.
void DenseModel::Impl::ffn_selfcheck(const DenseConfig& c, const Layer& L, int n, uint16_t rot, void* s) {
    const int E = c.n_embd;
    const size_t tot = (size_t) n * E;
    const cudaStream_t st = (cudaStream_t) s;
    // Read both sides through the same volatile-staging path the handoff uses: a cudaMemcpy of a just-written
    // buffer is a stale read on this box, which is what made the first self-check lie.
    dense_split_stage_out(x, dbg_x, (int64_t) tot, s);               // the pre-add residual
    dense_split_stage_out(mix, dbg_mix, (int64_t) tot, s);           // the split's own down output
    ffn_block(c, L.ffn_gate, L.ffn_up, L.ffn_down, L.post_norm, n, rot, s);   // the reference; the restore below adds
    dense_split_stage_out(mix, dbg_b, (int64_t) tot, s);             // the reference's down output
    cudaStreamSynchronize(st);
    size_t first = tot, ndiff = 0;
    uint32_t ba = 0, bb = 0;
    for (size_t i = 0; i < tot; ++i) {                               // the buffers are mapped: read them on the CPU
        uint32_t ua = 0, ub = 0;
        std::memcpy(&ua, &dbg_mix[i], 4);
        std::memcpy(&ub, &dbg_b[i], 4);
        if (ua != ub) {
            if (first == tot) { first = i; ba = ua; bb = ub; }
            ++ndiff;
        }
    }
    if (first == tot)
        std::fprintf(stderr, "dbg layer=%d split-check: identical over %zu values (n=%d)\n", cur_layer, tot, n);
    else
        std::fprintf(stderr, "dbg layer=%d split-check: %zu of %zu differ, first at col=%zu row=%zu split=%08x ref=%08x\n",
                     cur_layer, ndiff, tot, first / (size_t) E, first % (size_t) E, ba, bb);
    // put the split's own result back: x = pre-add residual + the split's output
    cudaMemcpyAsync(x, dbg_x, tot * 4, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(mix, dbg_mix, tot * 4, cudaMemcpyHostToDevice, st);
    add_inplace(x, mix, (int64_t) tot, st);
}

// Phase 1 of the multi-card work: card 0 keeps the whole model and computes the first ffn_r0 FFN rows; card 1
// holds the rest (gate/up row slices and a packed input-column slice of down).  Card 0 rotates the activation
// and sends it; card 1 returns its partial down output; card 0 sums and adds the residual.  Both cards work in
// parallel: card 0 issues its share on the main stream and card 1 on stream1.
void DenseModel::Impl::ffn_block_split(const DenseConfig& c, const Layer& L, int n, uint16_t rot, void* s,
                                       const float* pre_add) {
    const int E = c.n_embd, r0 = ffn_r0, r1 = c.n_ff - ffn_r0;
    const cudaStream_t s0 = (cudaStream_t) s;
    const bool dbg = [] { const char* e = std::getenv("STRATA_DENSE_FFN_DEBUG"); return e && e[0] != '0'; }();
    const bool trb = split_trace == cur_layer && cur_layer >= 0;
#define SDBG(...) do { if (dbg) { std::fprintf(stderr, "[ffn-split] " __VA_ARGS__); std::fprintf(stderr, "\n"); std::fflush(stderr); } } while (0)
    {
        PScope ps(&prof, Prof::RMS, s);
        dense_rms_norm(x, L.post_norm, xn, n, E, c.rms_eps, s, pre_add);
    }
    // card 0: the (rotated) activation, quantized for its own gate/up and staged for card 1
    const float* xsend = xn;
    if (rot & (ROT_FFN_GATE | ROT_FFN_UP)) {
        dense_fwht_rows(xn, rot_scratch, ham_signs(E), E, n, false, s);
        xsend = rot_scratch;
    }
    quantize(xsend, E, n, s);
    SDBG("s0 activation quantized (n=%d)", n);

    if (dbg)
        std::fprintf(stderr, "[ffn-split] guard peer=%d bell=%d hf=%d hf1=%d\n", (int) peer_ok, split_bell != nullptr,
                     h_full != nullptr, h_full1 != nullptr);
    if (peer_ok && split_bell && h_full && h_full1) {
        // ---- the bit-exact two-card FFN, row-split end to end.  Each card computes different gate/up rows and
        // rotates its own half; the halves are exchanged (each card assembles the full n_ff-wide activation and
        // quantizes it with the same call the single-card path uses); then each card computes a DIFFERENT HALF OF
        // THE OUTPUT ROWS of the down GEMV from that full input -- same kernel, same input, one card per output
        // element, so every output value is bit-identical to the single card's and no sum is ever reassociated.
        // Card 1's half of the output is exchanged back so card 0 does the one-and-only residual add.
        const uint32_t seq_x = ++split_seq;             // card 0 -> card 1: the activation
        const uint32_t seq_h0 = ++split_seq;            // card 0 -> card 1: its rotated half
        const uint32_t seq_h1 = ++split_seq;            // card 1 -> card 0: its rotated half
        const uint32_t seq_o = ++split_seq;             // card 1 -> card 0: its down output rows
        const int half = E / 2;
        // bisect switches (STRATA_DENSE_SPLIT_MODE): "asm" = card 0 computes the whole down GEMV from the assembled
        // input and card 1 skips its half; "c1half" = same, but card 1's half then overwrites its rows, so an
        // identical reply proves card 1's assembled input, quantize and down slice are bit-identical to card 0's.
        const char* mode = std::getenv("STRATA_DENSE_SPLIT_MODE");
        const int smode = mode && mode[0] == 'a' ? 1 : mode && mode[0] == 'c' ? 2 : 0;
        // STRATA_DENSE_SPLIT_SYNC=1: wait for the producing stream on the host before each peer pull.  The
        // handoff then cannot win or lose a race, which is how a visibility bug in the device-side doorbell
        // shows itself.  Slow by design: a debug switch.
        const bool hsync = [] { const char* e = std::getenv("STRATA_DENSE_SPLIT_SYNC"); return e && e[0] != '0'; }();
        // STRATA_DENSE_SPLIT_EVWAIT: how the host waits for the producer's stream before the consumer's fetch.
        //   1 (the split config's mode) = the verified mode: cudaEventSynchronize on the producer's ring event.
        //   2 = the doorbell spin.  Measured wrong on this box (2026-10-14): the pinned bell becomes visible long
        //       before the DMA's posted writes to the staged payload do, so every layer's fetch reads a torn or
        //       stale payload and the reply degenerates into one repeated token.  Kept to reproduce that.
        const int evwait = [] { const char* e = std::getenv("STRATA_DENSE_SPLIT_EVWAIT"); return e ? std::atoi(e) : 0; }();
        auto wait_bell = [](const uint32_t* bell, uint32_t seq) {         // mode 2
            while (*reinterpret_cast<const volatile uint32_t*>(bell) - seq > 0x80000000u) {
            }
        };
        // The staged handoff: every payload goes to pinned host memory on the producer's stream and comes back on
        // the consumer's, so neither card ever reads the other's VRAM -- a peer read is not coherent with the
        // owner's L2 on this box, so a pull of a just-written buffer can return a stale line.  There is still no
        // host round trip: both copies are only enqueued, and the doorbell orders them.
        //
        // The copy's "completion" does not mean the consumer can see it: the DMA engine's writes to host memory
        // are POSTED, so a read from the other card can pass them and return the previous layer's payload.  A
        // tiny read-back on the producer's stream cannot pass its own posted writes (PCIe ordering), so once it
        // completes the payload is at the host's ordering point; only then is the doorbell rung.  Without this the
        // split runs but every layer hands over the previous layer's data.
        cudaMemcpyAsync(hx_stage, xsend, (size_t) n * E * 4, cudaMemcpyDeviceToHost, s0);
        dense_split_ring(split_bell_dev, seq_x, s);
        if (evwait == 1) cudaEventRecord(ev_x, s0);

        QMat g0 = L.ffn_gate, u0 = L.ffn_up;
        g0.n_out = r0;
        u0.n_out = r0;
        gemv(g0, ffn_g, n, s);
        gemv(u0, ffn_u, n, s);
        dense_swiglu(ffn_g, ffn_u, ffn_h, (int64_t) n * r0, s);
        const float* hs0 = ffn_h;
        if (rot & ROT_FFN_DOWN) {
            dense_fwht_rows(ffn_h, rot_scratch, ham_signs(c.n_ff), r0, n, false, s);
            hs0 = rot_scratch;
        }
        dense_split_copy(hs0, h_full, r0, n, c.n_ff, 0, s);
        cudaMemcpyAsync(hh0_stage, hs0, (size_t) n * r0 * 4, cudaMemcpyDeviceToHost, s0);
        dense_split_ring(split_bell_dev + 32, seq_h0, s);
        if (evwait == 1) cudaEventRecord(ev_h0, s0);

        if (trb) {
            trace_head("s0_g", cur_layer, "split0", ffn_g, r0, s);
            trace_head("s0_hrot", cur_layer, "split0", hs0, r0, s);
        }

        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(1);
        if (evwait == 1) cudaEventSynchronize(ev_x);
        else if (evwait == 2) wait_bell(split_bell, seq_x);
        dense_split_poll(split_bell_dev, seq_x, stream1);
        if (hsync) cudaStreamSynchronize(s0);
        cudaMemcpyAsync(x1, hx_stage, (size_t) n * E * 4, cudaMemcpyHostToDevice, stream1);
        native_quantize_q8_1(x1, q8_1_1, E, n, stream1);
        native_mmvq(L.ffn_gate1.type, L.ffn_gate1.dev, q8_1_1, ffn_g1, E, r1, n, stream1);
        native_mmvq(L.ffn_up1.type, L.ffn_up1.dev, q8_1_1, ffn_u1, E, r1, n, stream1);
        dense_swiglu(ffn_g1, ffn_u1, ffn_h1, (int64_t) n * r1, stream1);
        const float* hs1 = ffn_h1;
        if (rot & ROT_FFN_DOWN) {
            dense_fwht_rows(ffn_h1, h1_rot, signs1, r1, n, false, stream1);
            hs1 = h1_rot;
        }
        dense_split_copy(hs1, h_full1, r1, n, c.n_ff, r0, stream1);
        cudaMemcpyAsync(hh1_stage, hs1, (size_t) n * r1 * 4, cudaMemcpyDeviceToHost, stream1);
        dense_split_ring(split_bell_dev + 16, seq_h1, stream1);
        if (evwait == 1) cudaEventRecord(ev_h1, stream1);

        if (evwait == 1) cudaEventSynchronize(ev_h0);
        else if (evwait == 2) wait_bell(split_bell + 32, seq_h0);
        dense_split_poll(split_bell_dev + 32, seq_h0, stream1);
        if (hsync) cudaStreamSynchronize(s0);
        cudaMemcpyAsync(h_peer0, hh0_stage, (size_t) n * r0 * 4, cudaMemcpyHostToDevice, stream1);
        dense_split_copy(h_peer0, h_full1, r0, n, c.n_ff, 0, stream1);
        if (smode != 1) {
            native_quantize_q8_1(h_full1, q8_1_1, c.n_ff, n, stream1);
            native_mmvq(L.ffn_down1.type, L.ffn_down1.dev, q8_1_1, mix1, c.n_ff, E - half, n, stream1);
            cudaMemcpyAsync(hm1_stage, mix1, (size_t) n * (E - half) * 4, cudaMemcpyDeviceToHost, stream1);
            dense_split_ring(split_bell_dev + 48, seq_o, stream1);
            if (evwait == 1) cudaEventRecord(ev_o, stream1);
        }
        if (trb) {
            trace_head("s1_g", cur_layer, "split1", ffn_g1, r1, stream1);
            trace_head("s1_hrot", cur_layer, "split1", hs1, r1, stream1);
            trace_head("s1_hfull", cur_layer, "split1", h_full1, c.n_ff, stream1);
            trace_head("s1_hfull_r0", cur_layer, "split1", h_full1 + r0, c.n_ff - r0, stream1);
            if (smode != 1) trace_head("s1_mix", cur_layer, "split1", mix1, E - half, stream1);
        }
        cudaSetDevice(prev);
        SDBG("card1 leg enqueued");

        if (evwait == 1) cudaEventSynchronize(ev_h1);
        else if (evwait == 2) wait_bell(split_bell + 16, seq_h1);
        dense_split_poll(split_bell_dev + 16, seq_h1, s);
        if (hsync) cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(h_peer, hh1_stage, (size_t) n * r1 * 4, cudaMemcpyHostToDevice, (cudaStream_t) s);
        dense_split_copy(h_peer, h_full, r1, n, c.n_ff, r0, s);
        quantize(h_full, c.n_ff, n, s);
        if (trb) {
            trace_head("s0_hfull", cur_layer, "split0", h_full, c.n_ff, s);
            trace_head("s0_hfull_r0", cur_layer, "split0", h_full + r0, c.n_ff - r0, s);
        }
        if (smode == 1) {
            QMat d0 = L.ffn_down;
            gemv(d0, mix, n, s);                                     // bisect: card 0 alone, all output rows
        } else if (smode == 2) {
            QMat d0 = L.ffn_down;
            gemv(d0, mix, n, s);                                     // bisect: card 0 alone ...
            if (evwait == 1) cudaEventSynchronize(ev_o);
            else if (evwait == 2) wait_bell(split_bell + 48, seq_o);
            dense_split_poll(split_bell_dev + 48, seq_o, s);
            if (hsync) cudaStreamSynchronize(stream1);
            cudaMemcpyAsync(dx_dev, hm1_stage, (size_t) n * (E - half) * 4, cudaMemcpyHostToDevice, (cudaStream_t) s);
            dense_split_copy(dx_dev, mix, E - half, n, E, half, s);  // ... then card 1's half overwrites its rows
        } else {
            {
                QMat d0 = L.ffn_down;
                d0.n_out = half;
                gemv(d0, mix_lo, n, s);                              // card 0's half of the output rows, compact
            }
            if (evwait == 1) cudaEventSynchronize(ev_o);
            else if (evwait == 2) wait_bell(split_bell + 48, seq_o);
            dense_split_poll(split_bell_dev + 48, seq_o, s);
            if (hsync) cudaStreamSynchronize(stream1);
            cudaMemcpyAsync(dx_dev, hm1_stage, (size_t) n * (E - half) * 4, cudaMemcpyHostToDevice, (cudaStream_t) s);
            // The GEMV writes y[col * n_out + row], so each card's half is compact in its own scratch.  Scatter
            // both into mix at the per-column stride E, which is where the one residual add reads them.
            dense_split_copy(mix_lo, mix, half, n, E, 0, s);
            dense_split_copy(dx_dev, mix, E - half, n, E, half, s);
        }
        if (trb || split_check) {
            PScope ps(&prof, Prof::ADD, s);
            ffn_selfcheck(c, L, n, rot, s);                          // does the residual add with the split's output
        } else {
            PScope ps(&prof, Prof::ADD, s);
            add_inplace(x, mix, (int64_t) n * E, s);
        }
        SDBG("s0 adds done");
        return;
    }

    // No usable peer channel (or no second card): compute the whole FFN on card 0, exactly like the single-card
    // path.  Card 1's weights are row slices, so there is no way to give it a share without the doorbell; the
    // honest fallback is the plain block.
    if (trb) std::fprintf(stderr, "dbg layer=%d split fallback (peer path off)\n", cur_layer);
    ffn_block(c, L.ffn_gate, L.ffn_up, L.ffn_down, L.post_norm, n, rot, s, pre_add);
    add_inplace(x, mix, (int64_t) n * E, s);        // ffn_block defers its residual add past this point
#undef SDBG
}

#ifdef STRATA_DENSE_MMQ
// ============================================================================ the batched prompt blocks
// The same arithmetic as the blocks above over a chunk of T tokens.  The projections go through MMQ: the weights
// stay in their GGUF blocks, the activations are rounded to q8_1 and the products run on int8 tensor cores, so a
// weight is read once per CHUNK instead of once per eight tokens.

/// q8_1 activations for the next pf_gemv().  The layout MMQ wants depends on the WEIGHT type, so every type change
/// needs its own pass over the activations.
void DenseModel::Impl::pf_quantize(const float* src, int wtype, int n_in, int64_t T, void* s) {
    PScope ps(&prof, Prof::QUANT, s);
    strata::prefill::mmq::quantize(src, nullptr, pf->xq, wtype, n_in, n_in, T, s);
}

/// y[t, :] = W x[t, :] for the T rows already quantized into pf->xq (one MMQ group: the whole chunk).
void DenseModel::Impl::pf_gemv(const QMat& w, float* y, int64_t T, void* s) {
    PScope ps(&prof, w.type == 12 ? Prof::GEMV_Q4K : w.type == 14 ? Prof::GEMV_Q6K
                                : w.type == 8 ? Prof::GEMV_Q80 : Prof::GEMV_OTHER, s);
    strata::prefill::mmq::Product p;
    p.w = w.dev;
    p.type = w.type;
    p.w_rows = w.n_out;
    p.w_cols = w.n_in;
    p.expert_bytes = strata::prefill::mmq::matrix_bytes(w.type, w.n_out, w.n_in);   // one group: never stepped over
    p.n = 1;
    p.xq = pf->xq;
    p.bounds = pf->bounds;                    // {0, T}
    p.ids = pf->ids;                          // iota: each row goes back to its own index
    p.total_rows = T;
    p.max_rows = T;
    p.dst = y;
    p.ld_dst = w.n_out;
    pf->ctx.run(p, s);
}

void DenseModel::Impl::pf_gdn_block(const DenseConfig& c, Layer& L, int64_t T, void* s) {
    const int C = c.conv_channels(), V = c.value_dim(), E = c.n_embd;
    const int S = c.ssm_state, KH = c.ssm_k_heads, VH = c.ssm_v_heads;
    pf_quantize(pf->xn, L.qkv.type, E, T, s);
    pf_gemv(L.qkv, pf->qkv, T, s);
    if (L.z.type != L.qkv.type) pf_quantize(pf->xn, L.z.type, E, T, s);
    pf_gemv(L.z, pf->z, T, s);
    if (L.ab_quant) {                         // the two tiny projections, quantized in the file
        if (L.alpha_q.type != L.z.type) pf_quantize(pf->xn, L.alpha_q.type, E, T, s);
        pf_gemv(L.alpha_q, pf->alpha, T, s);
        if (L.beta_q.type != L.alpha_q.type) pf_quantize(pf->xn, L.beta_q.type, E, T, s);
        pf_gemv(L.beta_q, pf->beta, T, s);
    } else {
        dense_gemv_f32_rows(L.alpha_w, pf->xn, pf->alpha, E, VH, (int) T, s);
        dense_gemv_f32_rows(L.beta_w, pf->xn, pf->beta, E, VH, (int) T, s);
    }
    {
        PScope ps_gdn(&prof, Prof::GDN_LOOP, s);
        // conv + SiLU over the chunk (the 3-value history carries over), the L2 norm of the q and k heads of every
        // token, the per-head gates, then the delta-rule recurrence walking the chunk with the output norm folded in.
        // {} = no rollback snapshots: a prompt chunk is never partially undone.
        dense_gdn_conv_chunk(L.conv_state, pf->qkv, L.conv_w, pf->h, C, (int) T, 0, 0.0f, {}, s);
        dense_gdn_l2_norm(pf->h, (int) T, C, 0, KH, S, c.rms_eps, s);
        dense_gdn_l2_norm(pf->h, (int) T, C, S * KH, KH, S, c.rms_eps, s);
        dense_gdn_gates(pf->alpha, L.dt, L.ssm_a, pf->gate, pf->beta, VH, (int) T, s);
        dense_gdn_rec_chunk(L.state, pf->h, pf->gate, pf->beta, pf->z, L.ssm_norm, c.rms_eps, pf->y, KH, VH, (int) T, {}, s);
    }
    pf_quantize(pf->y, L.ssm_out.type, V, T, s);
    pf_gemv(L.ssm_out, pf->mix, T, s);
}

void DenseModel::Impl::pf_attn_block(const DenseConfig& c, const QMat& wq, const QMat& wk, const QMat& wv,
                                     const QMat& wo, const float* qn, const float* kn, uint16_t* kc, uint16_t* vc,
                                     int64_t T, int pos0, void* s) {
    const int E = c.n_embd, H = c.n_head, HK = c.n_head_kv, D = c.head_dim;
    pf_quantize(pf->xn, wq.type, E, T, s);
    pf_gemv(wq, pf->q_full, T, s);
    if (wk.type != wq.type) pf_quantize(pf->xn, wk.type, E, T, s);
    pf_gemv(wk, pf->kcur, T, s);
    if (wv.type != wk.type) pf_quantize(pf->xn, wv.type, E, T, s);
    pf_gemv(wv, pf->vcur, T, s);
    const float scale = 1.0f / std::sqrt((float) D);
    {
        PScope ps_attn(&prof, Prof::ATTN_LOOP, s);
        // q is the first head_dim of every head's [q | gate] block; the norms and RoPE are per row, so they batch.
        dense_split_q(pf->q_full, pf->qcur, (int) (T * H), D, s);
        rms_norm_weighted(pf->qcur, qn, T * H, D, c.rms_eps, s);
        rms_norm_weighted(pf->kcur, kn, T * HK, D, c.rms_eps, s);
        dense_positions_i32(pf->pos_dev, (int) (T * H), H, (int32_t) pos0, s);
        native_rope_apply(pf->qcur, pf->qcur, (int) (T * H), D, c.n_rot, rope, pf->pos_dev, s);
        dense_positions_i32(pf->pos_dev, (int) (T * HK), HK, (int32_t) pos0, s);
        native_rope_apply(pf->kcur, pf->kcur, (int) (T * HK), D, c.n_rot, rope, pf->pos_dev, s);
        dense_kv_append_rows_fmt(kc, vc, kfmt, vfmt, pf->kcur, pf->vcur, (int) T, pos0, HK, D, (int) max_context, s);
        dense_attn_chunk_fmt(pf->qcur, kc, vc, kfmt, vfmt, pf->attn, (int) T, pos0, H, HK, D, (int) max_context, scale, s);
        dense_gate_apply(pf->attn, pf->q_full, pf->attn32, (int) (T * H), H, D, s);
    }
    pf_quantize(pf->attn32, wo.type, H * D, T, s);
    pf_gemv(wo, pf->mix, T, s);
}

void DenseModel::Impl::pf_ffn_block(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d,
                                    const float* post_norm, int64_t T, void* s) {
    const int E = c.n_embd;
    dense_rms_norm(pf->x, post_norm, pf->xn, (int) T, E, c.rms_eps, s);
    pf_quantize(pf->xn, g.type, E, T, s);
    pf_gemv(g, pf->ffn_g, T, s);
    if (u.type != g.type) pf_quantize(pf->xn, u.type, E, T, s);
    pf_gemv(u, pf->ffn_u, T, s);
    dense_swiglu(pf->ffn_g, pf->ffn_u, pf->ffn_h, (int64_t) T * c.n_ff, s);
    pf_quantize(pf->ffn_h, d.type, c.n_ff, T, s);
    pf_gemv(d, pf->mix, T, s);
    add_inplace(pf->x, pf->mix, (int64_t) T * E, s);
}
#endif  // STRATA_DENSE_MMQ

// The MTP block over n columns whose [e_norm | h_norm] inputs are in `cat`.  Leaves the head's logits for the
// last column in mtp_logits when asked.
bool DenseModel::Impl::mtp_block(const DenseConfig& c, int n, int pos0, bool want_logits, void* s) {
    const Mtp& M = mtp;
    const int E = c.n_embd;
    quantize(cat, 2 * E, n, s);
    gemv(M.eh_proj, x, n, s);                       // x = the block's input and its attention residual
    dense_rms_norm(x, M.attn_norm, xn, n, E, c.rms_eps, s);
    attn_block(c, M.q, M.k, M.v, M.o, M.q_norm, M.k_norm, M.kc, M.vc, n, pos0, nullptr, 0, s);   // the MTP block is never folded
    add_inplace(x, mix, (int64_t) n * E, s);
    ffn_block(c, M.ffn_gate, M.ffn_up, M.ffn_down, M.post_norm, n, 0, s);
    add_inplace(x, mix, (int64_t) n * E, s);        // ffn_block defers its residual add; the head below reads x
    if (want_logits) {
        dense_rms_norm(x + (size_t) (n - 1) * E, M.head_norm, xn, 1, E, c.rms_eps, s);
        cudaMemcpyAsync(mtp_h, xn, (size_t) E * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) s);
        quantize(xn, E, 1, s);
        gemv(M.head, mtp_logits, 1, s);
    }
    return true;
}

DenseModel::DenseModel() : impl_(new Impl) {}

DenseModel::~DenseModel() {
    if (!impl_) return;
    if (impl_->stream) cudaStreamSynchronize(impl_->stream);
    for (cudaGraphExec_t& e : impl_->graph_exec) if (e) { cudaGraphExecDestroy(e); e = nullptr; }
    if (impl_->pos_slot_host) { cudaFreeHost(impl_->pos_slot_host); impl_->pos_slot_host = nullptr; }
    if (impl_->stream1) {
        cudaSetDevice(1);
        cudaStreamDestroy(impl_->stream1);
        cudaSetDevice(0);
    }
    if (impl_->ev_x) cudaEventDestroy(impl_->ev_x);
    if (impl_->ev_h0) cudaEventDestroy(impl_->ev_h0);
    if (impl_->ev_h1) { cudaSetDevice(1); cudaEventDestroy(impl_->ev_h1); cudaSetDevice(0); }
    if (impl_->ev_o) { cudaSetDevice(1); cudaEventDestroy(impl_->ev_o); cudaSetDevice(0); }
    if (!impl_->allocs1.empty()) {
        cudaSetDevice(1);
        for (void* p : impl_->allocs1) cudaFree(p);
        cudaSetDevice(0);
    }
    for (void* p : impl_->allocations) cudaFree(p);
    for (void* p : impl_->host_allocs) cudaFreeHost(p);
    for (auto& c : impl_->ckpt) if (c.host) cudaFreeHost(c.host);
#ifdef STRATA_DENSE_MMQ
    if (impl_->pf && impl_->pf->stage) cudaFreeHost(impl_->pf->stage);
#endif
    if (impl_->embd_stage) cudaFreeHost(impl_->embd_stage);
    if (impl_->embed_done) cudaEventDestroy(impl_->embed_done);
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
}

float* DenseModel::logits_col(int j) const { return impl_->logits + (size_t) j * cfg_.n_vocab; }
void* DenseModel::stream() const { return impl_->stream; }

bool DenseModel::load(const std::string& path, int64_t max_context, std::string& err, const DenseOptions& opt) {
    Impl& I = *impl_;
    bool with_mtp = opt.mtp;
    const int draft_max = std::max(1, std::min(opt.draft_max, NC - 1));
    try {
        // ---- 1. the files and the tensor index
        for (const std::string& p : shard_paths(path)) I.files.emplace_back(new strata::GgufFile(p));
        for (const auto& f : I.files)
            for (const auto& t : f->tensors()) I.index[t.name] = Found{f.get(), &t};
        const strata::GgufFile& g0 = *I.files.front();

        // ---- 2. the configuration, from the model's own metadata
        const strata::MetaValue* arch = g0.get("general.architecture");
        if (!arch) { err = "dense model: missing general.architecture"; return false; }
        cfg_.arch = arch->s;
        const std::string a = cfg_.arch + ".";
        if (cfg_.arch.rfind("qwen3", 0) != 0) {
            err = "dense model: architecture '" + cfg_.arch + "' is not a qwen35 dense model";
            return false;
        }
        if (g0.get(a + "expert_count") && g0.get(a + "expert_count")->u > 0) {
            err = "dense model: this GGUF is a MoE model; run it with `strata`, not `strata-dense`";
            return false;
        }
        int64_t v = 0;
        auto need = [&](const char* key, int& dst) {
            if (!meta_int(g0, a + key, v)) { err = std::string("dense model: missing ") + a + key; return false; }
            dst = (int) v;
            return true;
        };
        int block_count = 0;
        if (!need("block_count", block_count) || !need("embedding_length", cfg_.n_embd) ||
            !need("feed_forward_length", cfg_.n_ff) || !need("attention.head_count", cfg_.n_head) ||
            !need("attention.head_count_kv", cfg_.n_head_kv) || !need("attention.key_length", cfg_.head_dim) ||
            !need("rope.dimension_count", cfg_.n_rot) || !need("ssm.conv_kernel", cfg_.ssm_d_conv) ||
            !need("ssm.state_size", cfg_.ssm_state) || !need("ssm.group_count", cfg_.ssm_k_heads) ||
            !need("ssm.time_step_rank", cfg_.ssm_v_heads))
            return false;
        // block_count counts the MTP block(s) stored after the trunk; llama.cpp runs only the trunk
        cfg_.n_nextn = meta_int(g0, a + "nextn_predict_layers", v) ? (int) v : 0;
        cfg_.n_layer = block_count - cfg_.n_nextn;
        if (cfg_.n_layer <= 0) { err = "dense model: no trunk layers"; return false; }
        if (const auto* e = g0.get(a + "attention.layer_norm_rms_epsilon")) cfg_.rms_eps = (float) e->num();
        if (const auto* e = g0.get(a + "rope.freq_base")) cfg_.rope_base = (float) e->num();
        // The rope the analytic kernel applies: the process scaling (rope_scaling.hpp, `none` unless a caller
        // has set one) re-based on what this file declares.  The base is the file's, never the caller's: the
        // angles must match the weights.
        I.rope = rope_scaling();
        I.rope.freq_base = (double) cfg_.rope_base;
        if (const char* why = rope_scaling_invalid(I.rope)) {
            err = std::string("dense model: invalid rope scaling configuration: ") + why;
            return false;
        }
        if (const auto* e = g0.get("tokenizer.ggml.eos_token_id")) cfg_.eos_id = (int32_t) e->u;
        if (meta_int(g0, a + "ssm.inner_size", v) && v != cfg_.value_dim()) {
            err = "dense model: ssm.inner_size " + std::to_string(v) + " != state_size * time_step_rank";
            return false;
        }
        if (cfg_.ssm_state != 128 || cfg_.ssm_d_conv != 4 || cfg_.ssm_v_heads % cfg_.ssm_k_heads != 0) {
            err = "dense model: the GDN kernels are written for state_size 128 and conv_kernel 4";
            return false;
        }
        if (cfg_.head_dim % 32 != 0 || cfg_.head_dim > 256 || cfg_.n_head % cfg_.n_head_kv != 0 ||
            cfg_.n_rot > cfg_.head_dim || cfg_.n_rot % 2 != 0) {
            err = "dense model: unsupported attention geometry";
            return false;
        }
        if (cfg_.n_embd % 256 != 0 || cfg_.n_ff % 256 != 0 || cfg_.value_dim() % 256 != 0 ||
            (cfg_.n_head * cfg_.head_dim) % 256 != 0) {
            err = "dense model: every GEMV input width must be a multiple of 256 (K-quant super-block)";
            return false;
        }
        const Found* tok = I.find("token_embd.weight");
        if (!tok || tok->tensor->shape.size() != 2 || tok->tensor->shape[0] != (uint64_t) cfg_.n_embd) {
            err = "dense model: token_embd.weight missing or not [n_embd, n_vocab]";
            return false;
        }
        cfg_.n_vocab = (int) tok->tensor->shape[1];
        max_context_ = max_context;
        I.max_context = max_context;
        if (max_context < 16) { err = "dense model: context must be at least 16"; return false; }

        // ---- the folded basis (PrismML's prism.hadamard.*): the weights are stored as D = W R^T and their
        // activations carry R at run time.  Read the sign tables now; they are tiny (one vector per width).
        if (const strata::MetaValue* hv = g0.get("prism.hadamard.version")) {
            if (hv->u != 1) { err = "dense model: unsupported prism.hadamard.version"; return false; }
            const strata::MetaValue* block = g0.get("prism.hadamard.block_size");
            const strata::MetaValue* trans = g0.get("prism.hadamard.transform");
            const strata::MetaValue* mode = g0.get("prism.hadamard.sign_mode");
            const strata::MetaValue* widths = g0.get("prism.hadamard.sign_widths");
            const strata::MetaValue* values = g0.get("prism.hadamard.sign_values");
            if (!block || !trans || !mode || !widths || !values) {
                err = "dense model: prism.hadamard metadata is incomplete"; return false;
            }
            I.rot_block = (int) block->u;
            if (I.rot_block != 1024) { err = "dense model: prism.hadamard.block_size must be 1024"; return false; }
            if (trans->s != "normalized-sylvester-walsh-hadamard" || mode->s != "explicit") {
                err = "dense model: unsupported prism.hadamard transform or sign mode"; return false;
            }
            if (const strata::MetaValue* gv = g0.get("prism.hadamard.gdn_v_grouped")) I.rot_gdn_v_grouped = gv->u != 0;
            if (const strata::MetaValue* inv = g0.get("prism.hadamard.inverse_weight_names"))
                for (const strata::MetaValue& v : inv->items) if (v.s == "token_embd.weight") I.rot_inverse_embd = true;
            uint64_t off = 0;
            for (const strata::MetaValue& w : widths->items) {
                const int width = (int) w.u;
                if (width <= 0 || width % I.rot_block != 0 || off + (uint64_t) width > values->count) {
                    err = "dense model: bad prism.hadamard sign table"; return false;
                }
                std::vector<float> host((size_t) width);
                if (values->items.size() < (size_t) (off + (uint64_t) width)) {
                    err = "dense model: prism.hadamard sign table was not read whole"; return false;
                }
                for (int i = 0; i < width; ++i) {
                    const int64_t s = (int64_t) values->items[(size_t) (off + i)].u;
                    if (s != 1 && s != -1) {
                        err = "dense model: prism.hadamard signs must be +/-1 (index " + std::to_string(off + i) +
                              ", value " + std::to_string((long long) s) + ", elem " + std::to_string((int) values->elem) +
                              ", count " + std::to_string(values->count) + ")";
                        return false;
                    }
                    host[(size_t) i] = (float) s;
                }
                if (width == cfg_.n_ff) I.signs_ffn_host = host;
                off += (uint64_t) width;
                void* dev = nullptr;
                if (cudaMalloc(&dev, (size_t) width * sizeof(float)) != cudaSuccess ||
                    cudaMemcpy(dev, host.data(), (size_t) width * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
                    if (dev) cudaFree(dev);
                    err = "dense model: prism.hadamard signs do not fit in VRAM"; return false;
                }
                I.rot_allocs.push_back(dev);
                I.rot_signs.push_back({width, (float*) dev});
            }
            I.hadamard = true;
        }
        std::unordered_set<std::string> rot_names;
        if (I.hadamard) {
            const strata::MetaValue* wn = g0.get("prism.hadamard.weight_names");
            if (!wn || wn->items.empty()) { err = "dense model: prism.hadamard.weight_names is empty"; return false; }
            if (wn->items.size() != wn->count) { err = "dense model: prism.hadamard.weight_names was not read whole"; return false; }
            for (const strata::MetaValue& v : wn->items) rot_names.insert(v.s);
            if (!I.ham_signs(cfg_.n_embd) || !I.ham_signs(cfg_.value_dim()) || !I.ham_signs(cfg_.n_ff)) {
                err = "dense model: prism.hadamard signs do not cover this model's widths"; return false;
            }
        }
        // the optional FFN split across GPUs (phase 1 of the multi-card work): opt-in while it is measured
        if (const char* e = std::getenv("STRATA_DENSE_FFN_SPLIT"); e && e[0] != '0') {
            int nd = 0;
            if (cudaGetDeviceCount(&nd) == cudaSuccess && nd >= 2) {
                double share = 0.25;
                if (const char* sh = std::getenv("STRATA_DENSE_FFN_SHARE0")) share = std::atof(sh);
                share = std::min(0.9, std::max(0.05, share));
                int r0 = (int) (share * cfg_.n_ff / 1024.0 + 0.5) * 1024;
                if (r0 < 1024) r0 = 1024;
                if (r0 > cfg_.n_ff - 1024) r0 = cfg_.n_ff - 1024;
                if (!I.hadamard || I.signs_ffn_host.empty()) {
                    std::fprintf(stderr, "strata-dense: FFN split needs a folded (prism.hadamard) file; staying on one card\n");
                } else {
                    I.ffn_split = true;
                    I.ffn_r0 = r0;
                    std::fprintf(stderr, "strata-dense: FFN split across GPUs: card0 %d of %d rows, card1 %d\n", r0,
                                 cfg_.n_ff, cfg_.n_ff - r0);
                }
            } else {
                std::fprintf(stderr, "strata-dense: STRATA_DENSE_FFN_SPLIT set but fewer than two GPUs are visible\n");
            }
        }
        const int E = cfg_.n_embd;
        const int mtp_block_idx = cfg_.n_layer;           // the MTP block follows the trunk
        const std::string mp = "blk." + std::to_string(mtp_block_idx) + ".";
        if (with_mtp && (cfg_.n_nextn < 1 || !I.find(mp + "nextn.eh_proj.weight"))) {
            err = "dense model: this GGUF has no MTP block (nextn_predict_layers = " + std::to_string(cfg_.n_nextn) +
                  "); start without --mtp";
            return false;
        }

        cudaError_t cs = cudaStreamCreate(&I.stream);
        if (cs == cudaSuccess) cs = cudaEventCreateWithFlags(&I.embed_done, cudaEventDisableTiming);
        if (cs == cudaSuccess) cs = cudaHostAlloc((void**) &I.embd_stage, (size_t) NC * E * 4, cudaHostAllocDefault);
        if (cs != cudaSuccess) { err = std::string("dense model: CUDA setup: ") + cudaGetErrorString(cs); return false; }
        if (const char* e = std::getenv("STRATA_DENSE_PROF")) I.prof.enabled = e[0] && e[0] != '0';
        if (const char* e = std::getenv("STRATA_DENSE_FFN_TRACE")) I.split_trace = std::atoi(e);
        if (const char* e = std::getenv("STRATA_DENSE_FFN_SELFCHECK")) I.split_check = e[0] && e[0] != '0';
        // STRATA_DENSE_GRAPH=1: capture the single-card decode pass into one HIP graph per attention split count
        // (lazily, at the first token of each range) and replay it.  One card only: the split path has host waits
        // in the middle of its pass that no capture can contain.
        if (const char* e = std::getenv("STRATA_DENSE_GRAPH")) I.graph_on = e[0] && e[0] != '0';
        if (I.graph_on && I.ffn_split) {
            std::fprintf(stderr, "strata-dense: STRATA_DENSE_GRAPH=1 is off: the FFN split's pass is not capturable\n");
            I.graph_on = false;
        }
        if (I.graph_on) {
            const int buckets = (int) ((max_context + 255) / 256);
            I.graph_exec.assign((size_t) buckets + 1, nullptr);
            if (I.alloc(&I.pos_slot, 8, err, "graph position slot") &&
                cudaHostAlloc((void**) &I.pos_slot_host, 8, cudaHostAllocDefault) == cudaSuccess) {
                I.pos_slot_host[0] = 0;
                cudaMemcpy(I.pos_slot, I.pos_slot_host, sizeof(int32_t), cudaMemcpyHostToDevice);
                std::fprintf(stderr, "strata-dense: HIP graphs on: the decode pass is one graph per 256-cell context range\n");
            } else {
                std::fprintf(stderr, "strata-dense: STRATA_DENSE_GRAPH: position slot allocation failed, staying on launches\n");
                I.graph_on = false;
            }
        }
        if (const char* e = std::getenv("STRATA_DENSE_MMVQ_FAST"))      // the multi-column layout that is faster but not bitwise equal to 1 column
            if (e[0] && e[0] != '0') { native_mmvq_set_multi_exact(false); std::fprintf(stderr, "strata-dense: multi-column GEMV: upstream layout\n"); }
        if (const char* e = std::getenv("STRATA_DENSE_DEBUG")) {
            long long a0 = 0, b0 = 0;
            if (std::sscanf(e, "%lld:%lld", &a0, &b0) == 2) { I.dbg_from = a0; I.dbg_to = b0; }
        }

        // ---- 3. the embedding tables stay in the (mapped) file; a row is dequantized per token
        if (!I.setup_embd(I.embd, tok, E, err)) return false;
        I.embd_mtp = I.embd;
        const Found* mtp_embd = with_mtp ? I.find(mp + "nextn.embed_tokens.weight") : nullptr;
        if (mtp_embd && !I.setup_embd(I.embd_mtp, mtp_embd, E, err)) return false;

        // ---- 3b. the KV cache format (strata's --kv)
        {
            std::string kv = opt.kv;
            for (char& ch : kv) ch = (char) std::tolower((unsigned char) ch);
            if (kv.empty() || kv == "fp16" || kv == "f16") { I.kfmt = DENSE_KV_F16; I.vfmt = DENSE_KV_F16; kv_name_ = "fp16"; }
            else if (kv == "int8" || kv == "q8_0" || kv == "q8") { I.kfmt = DENSE_KV_Q8; I.vfmt = DENSE_KV_Q8; kv_name_ = "int8"; }
            else if (kv == "q4_0" || kv == "q4") { I.kfmt = DENSE_KV_Q4; I.vfmt = DENSE_KV_Q4; kv_name_ = "q4_0"; }
            else if (kv == "k8v4") { I.kfmt = DENSE_KV_Q8; I.vfmt = DENSE_KV_Q4; kv_name_ = "k8v4"; }
            else { err = "dense model: --kv must be fp16, int8, q4_0 or k8v4 (got '" + opt.kv + "')"; return false; }
            if ((I.kfmt == DENSE_KV_Q4 || I.vfmt == DENSE_KV_Q4) && cfg_.head_dim != 256) {
                err = "dense model: --kv " + kv_name_ + " rotates 256 values per head, but this model's head_dim is " +
                      std::to_string(cfg_.head_dim) + "; use fp16 or int8";
                return false;
            }
        }
        const uint64_t kcell = dense_kv_cell_bytes(I.kfmt, cfg_.head_dim), vcell = dense_kv_cell_bytes(I.vfmt, cfg_.head_dim);

        // ---- 4. the VRAM budget for the weights
        uint64_t total = 0;
        const int C = cfg_.conv_channels(), V = cfg_.value_dim(), H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;
        {   // What is free, minus the KV caches, the rollback snapshots and a margin for the CUDA context, the
            // activations and the kernels.  The rest of the weights stays in pinned host memory.
            int n_attn = 0, n_gdn = 0;
            for (int l = 0; l < cfg_.n_layer; ++l) {
                if (I.find("blk." + std::to_string(l) + ".attn_q.weight")) ++n_attn; else ++n_gdn;
            }
            const uint64_t kv_per_layer = (uint64_t) HK * (uint64_t) max_context * (kcell + vcell);
            const uint64_t snap_one = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4 +
                                      (uint64_t) C * (cfg_.ssm_d_conv - 1) * 4;     // one snapshot of one GDN layer
            auto reserved_for = [&](bool mtp) {
                return (uint64_t) (n_attn + (mtp ? 1 : 0)) * kv_per_layer + (mtp ? (uint64_t) n_gdn * snap_one * (uint64_t) draft_max : 0);
            };
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const uint64_t margin = 1024ull << 20;
            auto budget_for = [&](bool mtp) -> int64_t { return (int64_t) free_b - (int64_t) reserved_for(mtp) - (int64_t) margin; };
            // the bytes of every quantized matrix the trunk (and the MTP block) will upload
            auto bytes_of_prefix = [&](const std::string& pre) {
                uint64_t sum = 0;
                for (auto it = I.index.lower_bound(pre); it != I.index.end() && it->first.compare(0, pre.size(), pre) == 0; ++it) {
                    const strata::TensorInfo& t = *it->second.tensor;
                    if (t.shape.size() != 2 || !native_mmvq_supported((int) t.type)) continue;
                    try { sum += native_mmvq_weight_bytes((int) t.type, (int) t.shape[0], (int) t.shape[1]); } catch (const std::exception&) {}
                }
                return sum;
            };
            uint64_t trunk_bytes = 0, mtp_bytes = 0;
            for (int l = 0; l < cfg_.n_layer; ++l) trunk_bytes += bytes_of_prefix("blk." + std::to_string(l) + ".");
            if (I.find("output.weight")) trunk_bytes += bytes_of_prefix("output.weight");
            else if (tok) trunk_bytes += native_mmvq_supported((int) tok->tensor->type)
                    ? native_mmvq_weight_bytes((int) tok->tensor->type, E, cfg_.n_vocab) : 0;
            mtp_bytes = bytes_of_prefix(mp);
            const bool user_cap = std::getenv("STRATA_DENSE_GPU_MIB") != nullptr;
            if (with_mtp && !user_cap && !opt.mtp_force && budget_for(true) < (int64_t) (trunk_bytes + mtp_bytes)) {
                // the longest context at which everything, MTP included, stays in VRAM
                const int64_t per_token = (int64_t) (n_attn + 1) * HK * (int64_t) (kcell + vcell);   // KV bytes per token, MTP layer too
                const int64_t room = (int64_t) free_b - (int64_t) margin - (int64_t) (trunk_bytes + mtp_bytes) -
                                     (int64_t) ((uint64_t) n_gdn * snap_one * (uint64_t) draft_max);
                const int64_t fit_ctx = room > 0 ? (room / per_token) / 1024 * 1024 : 0;
                std::fprintf(stderr,
                             "strata-dense: MTP is OFF: the weights (%.2f GiB with the MTP block), the KV cache for %lld tokens and "
                             "the %d rollback snapshot(s) do not all fit in VRAM, so some weights would be read over PCIe on every "
                             "token - slower, not faster.%s\n",
                             (trunk_bytes + mtp_bytes) / 1073741824.0, (long long) max_context, draft_max,
                             fit_ctx >= 2048 ? "" : " Even a short context would not fit: use a smaller quant.");
                if (fit_ctx >= 2048)
                    std::fprintf(stderr, "strata-dense:   with MTP on, a --context up to about %lld tokens fits in this GPU "
                                 "(or use --draft-max 1); --mtp-force keeps MTP at the current context anyway.\n",
                                 (long long) fit_ctx);
                with_mtp = false;
            }
            const uint64_t reserved = reserved_for(with_mtp);
            const uint64_t kv_need = (uint64_t) (n_attn + (with_mtp ? 1 : 0)) * kv_per_layer;
            if (user_cap) {                                                  // explicit cap for the weights, in MiB
                I.budget = (uint64_t) std::atoll(std::getenv("STRATA_DENSE_GPU_MIB")) << 20;
            } else if ((int64_t) free_b > (int64_t) (reserved + margin)) {
                I.budget = (uint64_t) budget_for(with_mtp);
            } else {
                err = "dense model: the KV cache for a context of " + std::to_string(max_context) + " needs " +
                      std::to_string(kv_need >> 20) + " MiB, but only " + std::to_string(free_b >> 20) +
                      " MiB of VRAM are free (1 GiB is kept as margin). Use a shorter --context.";
                return false;
            }
            std::fprintf(stderr, "strata-dense: VRAM %zu MiB free of %zu; KV cache %llu MiB (%s, context %lld)%s; "
                         "%llu MiB of weights may go to the GPU (the model needs %.0f MiB), the rest stays in host memory\n",
                         free_b >> 20, total_b >> 20, (unsigned long long) (kv_need >> 20), kv_name_.c_str(), (long long) max_context,
                         with_mtp ? " + MTP snapshots" : "", (unsigned long long) (I.budget >> 20),
                         (double) (trunk_bytes + (with_mtp ? mtp_bytes : 0)) / 1048576.0);
        }

        // ---- 5. the MTP block first: it is small, and a draft that has to cross PCIe is not worth making
        if (with_mtp) {
            Mtp& M = I.mtp;
            auto mat = [&](const char* nm, int n_in, int n_out, QMat& out) {
                return I.upload_mat(I.find(mp + nm), mp + nm, n_in, n_out, out, total, err);
            };
            auto flt = [&](const char* nm, uint64_t n, float** out) {
                return I.upload_floats(I.find(mp + nm), mp + nm, n, out, err);
            };
            if (!mat("nextn.eh_proj.weight", 2 * E, E, M.eh_proj) || !flt("nextn.enorm.weight", E, &M.enorm) ||
                !flt("nextn.hnorm.weight", E, &M.hnorm) || !flt("attn_norm.weight", E, &M.attn_norm) ||
                !flt("post_attention_norm.weight", E, &M.post_norm) || !flt("attn_q_norm.weight", D, &M.q_norm) ||
                !flt("attn_k_norm.weight", D, &M.k_norm) || !mat("attn_q.weight", E, H * 2 * D, M.q) ||
                !mat("attn_k.weight", E, HK * D, M.k) || !mat("attn_v.weight", E, HK * D, M.v) ||
                !mat("attn_output.weight", H * D, E, M.o) || !mat("ffn_gate.weight", E, cfg_.n_ff, M.ffn_gate) ||
                !mat("ffn_up.weight", E, cfg_.n_ff, M.ffn_up) || !mat("ffn_down.weight", cfg_.n_ff, E, M.ffn_down))
                return false;
            if (I.find(mp + "nextn.shared_head_norm.weight")) {
                if (!flt("nextn.shared_head_norm.weight", E, &M.head_norm)) return false;
            } else if (!I.upload_floats(I.find("output_norm.weight"), "output_norm.weight", E, &M.head_norm, err)) {
                return false;
            }
            if (I.find(mp + "nextn.shared_head_head.weight") &&
                !mat("nextn.shared_head_head.weight", E, cfg_.n_vocab, M.head))
                return false;                              // otherwise the trunk's head is shared (set below)
            const uint64_t kb = (uint64_t) HK * (uint64_t) max_context * kcell, vb = (uint64_t) HK * (uint64_t) max_context * vcell;
            if (!I.alloc(&M.kc, kb, err, "MTP K cache") || !I.alloc(&M.vc, vb, err, "MTP V cache")) return false;
            kv_bytes_ += kb + vb;
        }

        // ---- 6. the trunk layers
        I.layers.assign((size_t) cfg_.n_layer, Layer{});
        for (int l = 0; l < cfg_.n_layer; ++l) {
            Layer& L = I.layers[(size_t) l];
            const std::string p = "blk." + std::to_string(l) + ".";
            const std::string at = "layer " + std::to_string(l) + ": ";
            const bool has_gdn = I.find(p + "attn_qkv.weight") != nullptr;
            const bool has_attn = I.find(p + "attn_q.weight") != nullptr;
            if (has_gdn == has_attn) { err = "dense model: " + at + "needs exactly one of attn_qkv.weight / attn_q.weight"; return false; }
            L.attn = has_attn;
            if (!I.upload_floats(I.find(p + "attn_norm.weight"), p + "attn_norm.weight", E, &L.attn_norm, err)) return false;
            if (!I.upload_floats(I.find_any({"post_attention_norm.weight", "ffn_norm.weight"}, p), p + "post_attention_norm.weight",
                                 E, &L.post_norm, err)) return false;
            if (!I.upload_mat(I.find(p + "ffn_gate.weight"), p + "ffn_gate.weight", E, cfg_.n_ff, L.ffn_gate, total, err) ||
                !I.upload_mat(I.find(p + "ffn_up.weight"), p + "ffn_up.weight", E, cfg_.n_ff, L.ffn_up, total, err) ||
                !I.upload_mat(I.find(p + "ffn_down.weight"), p + "ffn_down.weight", cfg_.n_ff, E, L.ffn_down, total, err))
                return false;
            if (rot_names.count(p + "ffn_gate.weight")) L.rot |= ROT_FFN_GATE;
            if (rot_names.count(p + "ffn_up.weight")) L.rot |= ROT_FFN_UP;
            if (rot_names.count(p + "ffn_down.weight")) L.rot |= ROT_FFN_DOWN;
            if (I.ffn_split) {
                const int r0 = I.ffn_r0, r1 = cfg_.n_ff - I.ffn_r0;
                if (!I.upload_mat_slice(I.find(p + "ffn_gate.weight"), p + "ffn_gate.weight", E, cfg_.n_ff, r0, r1, 1,
                                        L.ffn_gate1, err) ||
                    !I.upload_mat_slice(I.find(p + "ffn_up.weight"), p + "ffn_up.weight", E, cfg_.n_ff, r0, r1, 1,
                                        L.ffn_up1, err) ||
                    !I.upload_mat_slice(I.find(p + "ffn_down.weight"), p + "ffn_down.weight", cfg_.n_ff, E, E / 2,
                                        E - E / 2, 1, L.ffn_down1, err))
                    return false;
            }
            if (!L.attn) {
                if (!I.upload_mat(I.find(p + "attn_qkv.weight"), p + "attn_qkv.weight", E, C, L.qkv, total, err) ||
                    !I.upload_mat(I.find(p + "attn_gate.weight"), p + "attn_gate.weight", E, V, L.z, total, err) ||
                    !I.upload_mat(I.find(p + "ssm_out.weight"), p + "ssm_out.weight", V, E, L.ssm_out, total, err))
                    return false;
                if (rot_names.count(p + "attn_qkv.weight")) L.rot |= ROT_QKV;
                if (rot_names.count(p + "attn_gate.weight")) L.rot |= ROT_Z;
                if (rot_names.count(p + "ssm_out.weight")) L.rot |= ROT_SSM_OUT;
                if (!I.upload_floats(I.find(p + "ssm_conv1d.weight"), p + "ssm_conv1d.weight", (uint64_t) C * cfg_.ssm_d_conv, &L.conv_w, err) ||
                    !I.upload_floats(I.find_any({"ssm_dt.bias", "ssm_dt"}, p), p + "ssm_dt.bias", cfg_.ssm_v_heads, &L.dt, err) ||
                    !I.upload_floats(I.find_any({"ssm_a", "ssm_a.weight"}, p), p + "ssm_a", cfg_.ssm_v_heads, &L.ssm_a, err) ||
                    !I.upload_floats(I.find(p + "ssm_norm.weight"), p + "ssm_norm.weight", cfg_.ssm_state, &L.ssm_norm, err))
                    return false;
                {   // ssm_alpha / ssm_beta: floats in some files, Q8_0 (or another quant) in others - both are served
                    const Found* fa = I.find(p + "ssm_alpha.weight");
                    const Found* fb = I.find(p + "ssm_beta.weight");
                    if (!fa || !fb) { err = "dense model: " + at + "missing ssm_alpha.weight / ssm_beta.weight"; return false; }
                    auto is_float = [](uint32_t t) { return t == kF32 || t == kF16 || t == kBF16; };
                    if (is_float(fa->tensor->type) != is_float(fb->tensor->type)) {
                        err = "dense model: " + at + "ssm_alpha and ssm_beta are stored in different kinds of type";
                        return false;
                    }
                    L.ab_quant = !is_float(fa->tensor->type);
                    if (L.ab_quant) {
                        if (!I.upload_mat(fa, p + "ssm_alpha.weight", E, cfg_.ssm_v_heads, L.alpha_q, total, err) ||
                            !I.upload_mat(fb, p + "ssm_beta.weight", E, cfg_.ssm_v_heads, L.beta_q, total, err))
                            return false;
                    } else if (!I.upload_floats(fa, p + "ssm_alpha.weight", (uint64_t) E * cfg_.ssm_v_heads, &L.alpha_w, err) ||
                               !I.upload_floats(fb, p + "ssm_beta.weight", (uint64_t) E * cfg_.ssm_v_heads, &L.beta_w, err))
                        return false;
                }
                const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
                const uint64_t cbytes = (uint64_t) C * (cfg_.ssm_d_conv - 1) * 4;
                if (!I.alloc(&L.state, sbytes, err, "GDN state") || !I.alloc(&L.conv_state, cbytes, err, "conv state")) return false;
                if (with_mtp) {
                    L.st_slots.assign((size_t) draft_max, nullptr);
                    L.cv_slots.assign((size_t) draft_max, nullptr);
                    for (int j = 0; j < draft_max; ++j)
                        if (!I.alloc(&L.st_slots[(size_t) j], sbytes, err, "GDN state snapshot") ||
                            !I.alloc(&L.cv_slots[(size_t) j], cbytes, err, "conv snapshot"))
                            return false;
                }
            } else {
                if (!I.upload_mat(I.find(p + "attn_q.weight"), p + "attn_q.weight", E, H * 2 * D, L.q, total, err) ||
                    !I.upload_mat(I.find(p + "attn_k.weight"), p + "attn_k.weight", E, HK * D, L.k, total, err) ||
                    !I.upload_mat(I.find(p + "attn_v.weight"), p + "attn_v.weight", E, HK * D, L.v, total, err) ||
                    !I.upload_mat(I.find(p + "attn_output.weight"), p + "attn_output.weight", H * D, E, L.o, total, err))
                    return false;
                if (rot_names.count(p + "attn_q.weight")) L.rot |= ROT_Q;
                if (rot_names.count(p + "attn_k.weight")) L.rot |= ROT_K;
                if (rot_names.count(p + "attn_v.weight")) L.rot |= ROT_V;
                if (rot_names.count(p + "attn_output.weight")) L.rot |= ROT_O;
                if (!I.upload_floats(I.find(p + "attn_q_norm.weight"), p + "attn_q_norm.weight", D, &L.q_norm, err) ||
                    !I.upload_floats(I.find(p + "attn_k_norm.weight"), p + "attn_k_norm.weight", D, &L.k_norm, err))
                    return false;
                const uint64_t kb = (uint64_t) HK * (uint64_t) max_context * kcell, vb = (uint64_t) HK * (uint64_t) max_context * vcell;
                if (!I.alloc(&L.kc, kb, err, "K cache") || !I.alloc(&L.vc, vb, err, "V cache")) return false;
                kv_bytes_ += kb + vb;
            }
        }

        // ---- 7. the output side
        if (!I.upload_floats(I.find("output_norm.weight"), "output_norm.weight", E, &I.output_norm, err)) return false;
        const Found* head = I.find("output.weight");
        if (!head) head = tok;                                   // tied embeddings
        if (!I.upload_mat(head, head == tok ? "token_embd.weight (tied head)" : "output.weight", E, cfg_.n_vocab,
                          I.head, total, err))
            return false;
        I.head_rot = rot_names.count("output.weight") != 0;
        if (with_mtp && !I.mtp.head.dev) I.mtp.head = I.head;    // the MTP block shares the trunk's head
        weight_bytes_ = total;
        host_bytes_ = I.host_bytes;
        if (I.host_tensors)
            std::fprintf(stderr, "strata-dense: %d tensors (%.2f GiB of %.2f) live in host memory and are read over PCIe "
                         "on every token - a shorter --context or a smaller quant puts more on the GPU\n",
                         I.host_tensors, I.host_bytes / 1073741824.0, total / 1073741824.0);

        // ---- 8. activations and scratch (NC columns each)
        const int max_in = std::max({2 * E, cfg_.n_ff, V, H * D});
        const uint64_t f = 4, K = NC;
        bool ok = I.alloc(&I.q8_1, native_q8_1_bytes(max_in, NC), err, "q8_1 scratch") &&
                  I.alloc(&I.q8_1_ab, native_q8_1_bytes(max_in, NC), err, "q8_1 alpha/beta scratch") &&
                  I.alloc(&I.rot_scratch, K * (uint64_t) max_in * f, err, "rotation scratch") &&
                  I.alloc(&I.x, K * E * f, err, "x") && I.alloc(&I.xn, K * E * f, err, "xn") &&
                  I.alloc(&I.mix, K * E * f, err, "mix") && I.alloc(&I.qkv, K * C * f, err, "qkv") &&
                  I.alloc(&I.h, K * C * f, err, "h") &&
                  I.alloc(&I.alpha, K * cfg_.ssm_v_heads * f, err, "alpha") && I.alloc(&I.beta, K * cfg_.ssm_v_heads * f, err, "beta") &&
                  I.alloc(&I.gate, K * cfg_.ssm_v_heads * f, err, "gate") &&
                  I.alloc(&I.z, K * V * f, err, "z") && I.alloc(&I.y, K * V * f, err, "y") &&
                  I.alloc(&I.q_full, K * H * 2 * D * f, err, "q_full") && I.alloc(&I.qcur, K * H * D * f, err, "qcur") &&
                  I.alloc(&I.kcur, K * HK * D * f, err, "kcur") && I.alloc(&I.vcur, K * HK * D * f, err, "vcur") &&
                  I.alloc(&I.attn, K * H * D * f, err, "attn") && I.alloc(&I.attn32, K * H * D * f, err, "attn32") &&
                  I.alloc(&I.ffn_g, K * cfg_.n_ff * f, err, "ffn_g") && I.alloc(&I.ffn_u, K * cfg_.n_ff * f, err, "ffn_u") &&
                  I.alloc(&I.ffn_h, K * cfg_.n_ff * f, err, "ffn_h") && I.alloc(&I.logits, K * cfg_.n_vocab * f, err, "logits") &&
                  I.alloc(&I.attn_scratch, dense_attn_scratch_bytes(H, D, (int) max_context), err, "attention scratch") &&
                  // one position per row of a full NC-column pass: the attention's RoPE is batched over the columns
                  I.alloc(&I.pos_dev, K * (uint64_t) std::max(H, HK) * 4, err, "positions") &&
                  I.alloc(&I.hid, K * E * f, err, "hidden") && I.alloc(&I.h_last, (uint64_t) E * f, err, "last hidden") &&
                  I.alloc(&I.d_tok, 16, err, "sampler output") && I.alloc(&I.d_prob, 16, err, "draft probability");
        if (ok && with_mtp)
            ok = I.alloc(&I.m_e, K * E * f, err, "MTP embeddings") && I.alloc(&I.hin, K * E * f, err, "MTP hidden input") &&
                 I.alloc(&I.cat, K * 2 * E * f, err, "MTP concat") && I.alloc(&I.mtp_logits, (uint64_t) cfg_.n_vocab * f, err, "MTP logits") &&
                 I.alloc(&I.mtp_h, (uint64_t) E * f, err, "MTP output hidden");
        if (ok && I.ffn_split) {
            int prev_ev = 0;
            cudaGetDevice(&prev_ev);
            ok = cudaEventCreateWithFlags(&I.ev_x, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&I.ev_h0, cudaEventDisableTiming) == cudaSuccess;
            cudaSetDevice(1);
            ok = ok && cudaEventCreateWithFlags(&I.ev_h1, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&I.ev_o, cudaEventDisableTiming) == cudaSuccess;
            cudaSetDevice(prev_ev);
        }
        if (ok && I.ffn_split) {
            const int r1 = cfg_.n_ff - I.ffn_r0;
            int prev = 0;
            cudaGetDevice(&prev);
            cudaSetDevice(1);
            const bool ok1 = cudaStreamCreate(&I.stream1) == cudaSuccess &&
                             cudaMalloc(&I.q8_1_1, native_q8_1_bytes(cfg_.n_ff, NC)) == cudaSuccess &&
                             cudaMalloc((void**) &I.x1, K * E * f) == cudaSuccess &&
                             cudaMalloc((void**) &I.mix1, K * E * f) == cudaSuccess &&
                             cudaMalloc((void**) &I.ffn_g1, K * (uint64_t) r1 * f) == cudaSuccess &&
                             cudaMalloc((void**) &I.ffn_u1, K * (uint64_t) r1 * f) == cudaSuccess &&
                             cudaMalloc((void**) &I.ffn_h1, K * (uint64_t) r1 * f) == cudaSuccess &&
                             cudaMalloc((void**) &I.h1_rot, K * (uint64_t) r1 * f) == cudaSuccess &&
                             cudaMalloc((void**) &I.h_full1, (size_t) K * cfg_.n_ff * 4) == cudaSuccess &&
                             cudaMalloc((void**) &I.h_peer0, (size_t) K * (uint64_t) I.ffn_r0 * 4) == cudaSuccess &&
                             cudaMalloc((void**) &I.signs1, (uint64_t) r1 * f) == cudaSuccess;
            if (ok1) cudaMemcpy(I.signs1, I.signs_ffn_host.data() + I.ffn_r0, (size_t) r1 * 4, cudaMemcpyHostToDevice);
            I.allocs1 = {I.q8_1_1, I.x1, I.mix1, I.ffn_g1, I.ffn_u1, I.ffn_h1, I.h1_rot, I.h_full1, I.h_peer0, I.signs1};
            cudaSetDevice(prev);
            ok = ok1 && cudaMalloc((void**) &I.dx_dev, (size_t) K * E * 4) == cudaSuccess &&
                 cudaMalloc((void**) &I.mix_lo, (size_t) K * (uint64_t) (E / 2) * 4) == cudaSuccess &&
                 cudaMalloc((void**) &I.h_full, (size_t) K * cfg_.n_ff * 4) == cudaSuccess &&
                 cudaMalloc((void**) &I.h_peer, (size_t) K * (uint64_t) r1 * 4) == cudaSuccess &&
                 cudaHostAlloc(&I.hx_stage, (size_t) K * E * 4, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                 cudaHostAlloc(&I.hh0_stage, (size_t) K * (uint64_t) I.ffn_r0 * 4, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                 cudaHostAlloc(&I.hh1_stage, (size_t) K * (uint64_t) r1 * 4, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                 cudaHostAlloc(&I.hm1_stage, (size_t) K * E * 4, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess;
            if (ok && (I.split_trace >= 0 || I.split_check))
                ok = cudaHostAlloc(&I.dbg_mix, (size_t) K * E * 4, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                     cudaHostAlloc(&I.dbg_b, (size_t) K * E * 4, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                     cudaHostAlloc(&I.dbg_x, (size_t) K * E * 4, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess;
            if (!ok) { err = "dense model: the FFN split could not allocate its scratch"; return false; }
            if (I.dbg_mix) I.host_allocs.push_back(I.dbg_mix);
            if (I.dbg_b) I.host_allocs.push_back(I.dbg_b);
            if (I.dbg_x) I.host_allocs.push_back(I.dbg_x);
            I.allocations.push_back(I.dx_dev);
            I.allocations.push_back(I.mix_lo);
            I.allocations.push_back(I.h_full);
            I.allocations.push_back(I.h_peer);
            if (I.hx_stage) I.host_allocs.push_back(I.hx_stage);
            if (I.hh0_stage) I.host_allocs.push_back(I.hh0_stage);
            if (I.hh1_stage) I.host_allocs.push_back(I.hh1_stage);
            if (I.hm1_stage) I.host_allocs.push_back(I.hm1_stage);
            {
                int can = 0;
                I.peer_ok = cudaDeviceCanAccessPeer(&can, 1, 0) == cudaSuccess && can != 0;
                if (I.peer_ok) {
                    cudaSetDevice(1);
                    I.peer_ok = cudaDeviceEnablePeerAccess(0, 0) == cudaSuccess;
                    cudaSetDevice(prev);
                }
            }
            if (I.peer_ok) {
                // the doorbells the two cards ring for each other: pinned host memory (the one channel that works)
                void* bell = nullptr;
                uint32_t* bell_dev = nullptr;
                ok = cudaHostAlloc(&bell, 256, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                     cudaHostGetDevicePointer((void**) &bell_dev, bell, 0) == cudaSuccess;
                if (ok) {
                    std::memset(bell, 0, 256);
                    I.split_bell = (uint32_t*) bell;
                    I.split_bell_dev = bell_dev;
                    I.host_allocs.push_back(bell);
                } else {
                    std::fprintf(stderr, "strata-dense: the FFN split doorbells could not be allocated: %s\n",
                                 cudaGetErrorString(cudaGetLastError()));
                }
            }
            std::fprintf(stderr, "strata-dense: FFN split ready (card0 %d rows, card1 %d rows, handoff: %s)\n", I.ffn_r0,
                         cfg_.n_ff - I.ffn_r0, I.peer_ok ? "doorbell + peer DMA" : "host-staged");
        }
        if (!ok) return false;

        native_gdn_set_enabled(true);
        has_mtp_ = with_mtp;
        draft_max_ = with_mtp ? draft_max : 0;
        reset();
        std::string serr;
        if (!sync(serr)) { err = serr; return false; }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

void DenseModel::reset() {
    Impl& I = *impl_;
    const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cbytes = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    for (Layer& L : I.layers) {
        if (L.state) cudaMemsetAsync(L.state, 0, sbytes, I.stream);
        if (L.conv_state) cudaMemsetAsync(L.conv_state, 0, cbytes, I.stream);
    }
    if (I.h_last) cudaMemsetAsync(I.h_last, 0, (size_t) cfg_.n_embd * 4, I.stream);
    pos_ = 0;
    snap_valid_ = false;
}

bool DenseModel::sync(std::string& err) {
    const cudaError_t s = cudaStreamSynchronize(impl_->stream);
    if (s != cudaSuccess) { err = std::string("dense model: ") + cudaGetErrorString(s); return false; }
    return true;
}
int64_t DenseModel::ckpt_pos(int slot) const { return slot >= 0 && slot < ckpt_n_ ? impl_->ckpt[(size_t) slot].pos : -1; }
void DenseModel::ckpt_drop(int slot) { if (slot >= 0 && slot < ckpt_n_) impl_->ckpt[(size_t) slot].pos = -1; }

bool DenseModel::ckpt_init(int slots, std::string& err) {
    Impl& I = *impl_;
    for (auto& c : I.ckpt) if (c.host) cudaFreeHost(c.host);
    I.ckpt.clear();
    ckpt_n_ = 0;
    if (slots <= 0) return true;
    const uint64_t sb = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cb = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    uint64_t n_gdn = 0;
    for (const Layer& L : I.layers) if (!L.attn) ++n_gdn;
    I.ckpt_bytes = n_gdn * (sb + cb) + (uint64_t) cfg_.n_embd * 4;
    I.ckpt.resize((size_t) slots);
    for (int k = 0; k < slots; ++k) {
        void* p = nullptr;
        const cudaError_t s = cudaHostAlloc(&p, I.ckpt_bytes, cudaHostAllocDefault);
        if (s != cudaSuccess) {
            cudaGetLastError();
            err = "dense model: cannot pin " + std::to_string(I.ckpt_bytes >> 20) + " MiB for checkpoint " +
                  std::to_string(k + 1) + ": " + cudaGetErrorString(s);
            for (auto& c : I.ckpt) if (c.host) cudaFreeHost(c.host);
            I.ckpt.clear();
            return false;
        }
        I.ckpt[(size_t) k].host = (uint8_t*) p;
    }
    ckpt_n_ = slots;
    return true;
}

bool DenseModel::ckpt_save(int slot, std::string& err) {
    Impl& I = *impl_;
    if (slot < 0 || slot >= ckpt_n_) { err = "dense model: no such checkpoint slot"; return false; }
    const uint64_t sb = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cb = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    uint8_t* at = I.ckpt[(size_t) slot].host;
    for (const Layer& L : I.layers) {
        if (L.attn) continue;
        cudaMemcpyAsync(at, L.state, sb, cudaMemcpyDeviceToHost, I.stream); at += sb;
        cudaMemcpyAsync(at, L.conv_state, cb, cudaMemcpyDeviceToHost, I.stream); at += cb;
    }
    cudaMemcpyAsync(at, I.h_last, (size_t) cfg_.n_embd * 4, cudaMemcpyDeviceToHost, I.stream);
    const cudaError_t s = cudaGetLastError();
    if (s != cudaSuccess) { err = std::string("dense model: checkpoint save: ") + cudaGetErrorString(s); return false; }
    I.ckpt[(size_t) slot].pos = pos_;
    return true;
}

bool DenseModel::ckpt_restore(int slot, std::string& err) {
    Impl& I = *impl_;
    if (slot < 0 || slot >= ckpt_n_ || I.ckpt[(size_t) slot].pos < 0) { err = "dense model: that checkpoint is empty"; return false; }
    const uint64_t sb = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cb = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    const uint8_t* at = I.ckpt[(size_t) slot].host;
    for (Layer& L : I.layers) {
        if (L.attn) continue;
        cudaMemcpyAsync(L.state, at, sb, cudaMemcpyHostToDevice, I.stream); at += sb;
        cudaMemcpyAsync(L.conv_state, at, cb, cudaMemcpyHostToDevice, I.stream); at += cb;
    }
    cudaMemcpyAsync(I.h_last, at, (size_t) cfg_.n_embd * 4, cudaMemcpyHostToDevice, I.stream);
    const cudaError_t s = cudaGetLastError();
    if (s != cudaSuccess) { err = std::string("dense model: checkpoint restore: ") + cudaGetErrorString(s); return false; }
    pos_ = I.ckpt[(size_t) slot].pos;
    snap_valid_ = false;
    return true;
}
void DenseModel::set_last_hidden(int col) {
    cudaMemcpyAsync(impl_->h_last, impl_->hid + (size_t) col * cfg_.n_embd, (size_t) cfg_.n_embd * 4,
                    cudaMemcpyDeviceToDevice, impl_->stream);
}

bool DenseModel::rollback(int keep_col, std::string& err) {
    if (!snap_valid_) { err = "dense model: rollback without a snapshot"; return false; }
    if (keep_col < 0 || keep_col >= snap_cols_ - 1) { err = "dense model: rollback column outside the snapshots"; return false; }
    // A captured graph holds the state POINTERS the layers were launched with; the swap below changes them, so
    // the graphs are dropped and recaptured against the new ones (the spec path is the only caller).
    bool had_graph = false;
    for (cudaGraphExec_t& e : impl_->graph_exec) if (e) had_graph = true;
    if (had_graph) {
        cudaStreamSynchronize(impl_->stream);
        for (cudaGraphExec_t& e : impl_->graph_exec) { if (e) { cudaGraphExecDestroy(e); e = nullptr; } }
    }
    for (Layer& L : impl_->layers) {
        if (L.attn) continue;
        std::swap(L.state, L.st_slots[(size_t) keep_col]);
        std::swap(L.conv_state, L.cv_slots[(size_t) keep_col]);
    }
    pos_ = snap_pos0_ + keep_col + 1;
    snap_valid_ = false;
    return true;
}

// The device half of one pass, from the embedding copy to the head, with no host-side work: the caller has already
// dequantized the tokens into embd_stage.  pos_slot == nullptr runs the plain path (the layer blocks take pos0 as
// an argument); a non-null slot is the graph path, where the pass's own memcpy fills the device position and the
// kernels read it back.  Callable inside a stream capture.
bool DenseModel::Impl::enqueue_pass(const DenseConfig& c, int n, int64_t pos0, Logits lg, bool keep_hidden,
                                    bool snapshot, const int32_t* pos_slot, void* s) {
    const int E = c.n_embd;
    enqueue_embed(embd_stage, x, n, E, s);
    if (pos_slot) cudaMemcpyAsync((void*) pos_slot, pos_slot_host, sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) s);
    const bool dbg = dbg_from >= 0 && pos0 < dbg_to && pos0 + n > dbg_from;
    if (dbg) trace_vec("embed", pos0, -1, "", x, n * E, s);
    // The FFN's residual add is deferred: its down output (mix) rides in the NEXT block's input rms, which is the
    // only reader of x before that block's own output overwrites mix.  One launch less per layer (and the output
    // norm folds the last one).
    const float* pending = nullptr;
    for (int l = 0; l < c.n_layer; ++l) {
        Layer& L = layers[(size_t) l];
        // The pre-block rms and the folded-basis rotation of its output are one launch when the block needs the
        // rotation.  A GDN layer of a quantized-alpha/beta file needs BOTH quantizations of that output -- plain
        // for alpha/beta and folded for qkv/z -- and gets them from the same pass (q8_1 and q8_1_ab).
        const bool rot_pre = L.attn ? (L.rot & (ROT_Q | ROT_K | ROT_V)) != 0
                                    : (L.rot & (ROT_QKV | ROT_Z)) != 0;
        const bool pre_rot = rot_pre;
        const bool pre_ab = !L.attn && L.ab_quant && rot_pre;
        if (pre_rot) {
            PScope ps(&prof, Prof::QUANT, s);
            dense_rms_fwht_quant_rows(x, pending, L.attn_norm, xn, q8_1, ham_signs(E), E, n, c.rms_eps, s,
                                      pre_ab ? q8_1_ab : nullptr);
        } else {
            PScope ps(&prof, Prof::RMS, s);
            dense_rms_norm(x, L.attn_norm, xn, n, E, c.rms_eps, s, pending);
        }
        pending = nullptr;
        if (!L.attn) gdn_block(c, L, n, snapshot, s, pre_rot);
        else attn_block(c, L.q, L.k, L.v, L.o, L.q_norm, L.k_norm, L.kc, L.vc, n, (int) pos0, pos_slot, L.rot, s,
                        pre_rot);
        cur_layer = l;
        // The mixed block's residual add rides in the FFN's input rms (the only reader of mix before the FFN's
        // own down GEMV overwrites it), one launch instead of two.  The FFN's own add is deferred the same way.
        // STRATA_DENSE_SPLIT_OFF=1: keep the split load (so ffn_r0 and the traces are the split's) but run the
        // plain single-card FFN.  The reference side of the part-by-part comparison.
        static const bool split_off = std::getenv("STRATA_DENSE_SPLIT_OFF") && std::getenv("STRATA_DENSE_SPLIT_OFF")[0] != '0';
        if (ffn_split && !split_off) {
            ffn_block_split(c, L, n, L.rot, s, mix);            // adds its own output to x
        } else {
            ffn_block(c, L.ffn_gate, L.ffn_up, L.ffn_down, L.post_norm, n, L.rot, s, mix);
            // The x trace reads the post-add residual; with the debug trace on, add like before and keep the
            // next rms reading its own x.
            if (dbg) add_inplace(x, mix, (int64_t) n * E, s);
            else pending = mix;
        }
        if (dbg) trace_vec("x", pos0, l, L.attn ? "attn" : "gdn", x, n * E, s);
    }
    if (lg != Logits::None || keep_hidden) {
        {
            PScope ps(&prof, Prof::RMS, s);
            dense_rms_norm(x, output_norm, hid, n, E, c.rms_eps, s, pending);   // h_nextn: the post-norm hidden
        }
        if (lg == Logits::Last) {
            const float* src = hid + (size_t) (n - 1) * E;
            if (head_rot) fwht_quantize(src, E, 1, ham_signs(E), nullptr, 0, 0, 0, s);  // rotation + q8_1, one launch
            else quantize(src, E, 1, s);
            gemv(head, logits, 1, s);
        } else if (lg == Logits::All) {
            const float* src = hid;
            if (head_rot) fwht_quantize(src, E, n, ham_signs(E), nullptr, 0, 0, 0, s);
            else quantize(src, E, n, s);
            gemv(head, logits, n, s);
        }
        if (dbg && lg != Logits::None) trace_logits(pos0 + n - 1, logits, c.n_vocab, s);
    }
    return true;
}

// Captures one pass into a graph for `bucket` (the split count of the attention decode at this pos0).  The pass
// always computes the head, so the graph serves Logits::None calls too (they just ignore the logits).
bool DenseModel::Impl::capture_graph(const DenseConfig& c, int bucket, int64_t pos0, std::string& err) {
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = std::string("dense model: graph begin capture: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    bool ok = true;
    try {
        enqueue_pass(c, 1, pos0, Logits::Last, true, false, pos_slot, stream);
    } catch (const std::exception& e) {
        ok = false;
        err = std::string("dense model: graph capture: ") + e.what();
    }
    cudaGraph_t built = nullptr;
    const cudaError_t end = cudaStreamEndCapture(stream, &built);
    if (end != cudaSuccess) {
        if (built) cudaGraphDestroy(built);
        if (ok) err = std::string("dense model: graph end capture: ") + cudaGetErrorString(end);
        return false;
    }
    if (!ok) { if (built) cudaGraphDestroy(built); return false; }
    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiate(&exec, built, nullptr, nullptr, 0) != cudaSuccess) {
        err = std::string("dense model: graph instantiate: ") + cudaGetErrorString(cudaGetLastError());
        cudaGraphDestroy(built);
        return false;
    }
    cudaGraphDestroy(built);
    graph_exec[(size_t) bucket] = exec;
    return true;
}

// The graph entry: stages the token's embedding and position in pinned host memory, captures the bucket's graph on
// first use, and replays it.  False with an empty err means the capture failed and the caller should run the plain
// path (graph_on is off for good); false with an err is a real failure.
bool DenseModel::Impl::run_graphed(const DenseConfig& c, int32_t token, int64_t pos0, std::string& err) {
    if (cudaEventSynchronize(embed_done) != cudaSuccess) { err = "dense model: embedding event"; return false; }
    if (!embed_stage_rows(embd, &token, 1, c.n_embd, c.n_vocab, embd_stage, err)) return false;
    pos_slot_host[0] = (int32_t) pos0;
    const int64_t want = (pos0 + 1 + 255) / 256;
    const int bucket = (int) std::min<int64_t>(want, (int64_t) graph_exec.size() - 1);
    if (!graph_exec[(size_t) bucket] && !capture_graph(c, bucket, pos0, err)) {
        graph_on = false;                          // stay on the plain path, whatever went wrong
        for (cudaGraphExec_t& e : graph_exec) { if (e) { cudaGraphExecDestroy(e); e = nullptr; } }
        err.clear();
        return false;
    }
    if (cudaGraphLaunch(graph_exec[(size_t) bucket], stream) != cudaSuccess) {
        err = std::string("dense model: graph launch: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    cudaEventRecord(embed_done, stream);
    return true;
}

bool DenseModel::run(const int32_t* tokens, int n, Logits lg, bool keep_hidden, bool snapshot, std::string& err) {
    Impl& I = *impl_;
    if (n < 1 || n > NC) { err = "dense model: run() takes 1 to " + std::to_string(NC) + " tokens"; return false; }
    if (pos_ + n > max_context_) { err = "dense model: the context is full"; return false; }
    if (snapshot && (!has_mtp_ || n - 1 > draft_max_)) { err = "dense model: bad snapshot request (more columns than --draft-max + 1, or no MTP)"; return false; }
    void* s = I.stream;
    const int E = cfg_.n_embd;
    const int64_t pos0 = pos_;
    const bool dbg = I.dbg_from >= 0 && pos0 < I.dbg_to && pos0 + n > I.dbg_from;
    // The graph path serves the plain single-token pass.  A failed capture returns false with an empty err; the
    // plain path below then runs the token.
    if (I.graph_on && n == 1 && !snapshot && !dbg && !I.prof.enabled && !I.ffn_split) {
        if (I.run_graphed(cfg_, tokens[0], pos0, err)) {
            pos_ += 1;
            snap_valid_ = false;
            snap_pos0_ = pos0;
            snap_cols_ = 1;
            return true;
        }
        if (!err.empty()) return false;
    }
    try {
        if (I.prof.enabled) I.prof.begin(s);
        {
            PScope ps(&I.prof, Prof::EMB, s);
            if (cudaEventSynchronize(I.embed_done) != cudaSuccess) {
                err = "dense model: embedding event";
                I.prof.on = false;
                return false;
            }
            if (!I.embed_stage_rows(I.embd, tokens, n, E, cfg_.n_vocab, I.embd_stage, err)) { I.prof.on = false; return false; }
        }
        I.enqueue_pass(cfg_, n, pos0, lg, keep_hidden, snapshot, nullptr, s);
        cudaEventRecord(I.embed_done, (cudaStream_t) s);
        pos_ += n;
        snap_valid_ = snapshot && n > 1;
        snap_pos0_ = pos0;
        snap_cols_ = n;
        if (I.prof.on) I.prof.finish(s, n);
        return true;
    } catch (const std::exception& e) {
        I.prof.on = false;
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

// ============================================================================ batched prompt processing
//
// run() serves at most kMaxCols columns, so a prompt reads every weight once per eight tokens.  prefill() walks the
// prompt in chunks of prefill_chunk() tokens instead: the projections go through MMQ (the weights stay in their GGUF
// blocks, the activations are rounded to q8_1, the products run on int8 tensor cores) and the order-dependent steps
// walk the chunk's tokens inside one launch.  The recurrent state, the KV cache and the position advance exactly as
// they do through run(), so the logits match.
bool DenseModel::prefill_ready() const { return pf_ready_; }
int64_t DenseModel::prefill_chunk() const { return pf_chunk_; }

bool DenseModel::prefill_setup(int64_t chunk, std::string& err) {
    pf_ready_ = false;
    pf_chunk_ = 0;
#ifdef STRATA_DENSE_MMQ
    Impl& I = *impl_;
    if (chunk <= 0) chunk = kPrefillDefaultCols;
    chunk = std::max<int64_t>(NC, std::min<int64_t>(chunk, kPrefillMaxCols));
    const int E = cfg_.n_embd, C = cfg_.conv_channels(), V = cfg_.value_dim();
    const int H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;

    // Every projection the prompt path runs has to be a type MMQ covers; if one is not, the caller feeds the prompt
    // through run() and loses nothing.
    auto mat_ok = [&](const QMat& w, const char* what) {
        if (w.dev && !strata::prefill::mmq::supported(w.type)) {
            err = std::string("dense model: prefill cannot multiply ") + what + " of GGUF type " +
                  std::to_string(w.type) + "; feed the prompt with run() instead";
            return false;
        }
        return true;
    };
    for (size_t l = 0; l < I.layers.size(); ++l) {
        const Layer& L = I.layers[l];
        const std::string p = "layer " + std::to_string(l);
        if (!L.attn) {
            if (!mat_ok(L.qkv, (p + " attn_qkv").c_str()) || !mat_ok(L.z, (p + " attn_gate").c_str()) ||
                !mat_ok(L.ssm_out, (p + " ssm_out").c_str()))
                return false;
            if (L.ab_quant && (!mat_ok(L.alpha_q, (p + " ssm_alpha").c_str()) ||
                               !mat_ok(L.beta_q, (p + " ssm_beta").c_str())))
                return false;
        } else if (!mat_ok(L.q, (p + " attn_q").c_str()) || !mat_ok(L.k, (p + " attn_k").c_str()) ||
                   !mat_ok(L.v, (p + " attn_v").c_str()) || !mat_ok(L.o, (p + " attn_output").c_str()))
            return false;
        if (!mat_ok(L.ffn_gate, (p + " ffn_gate").c_str()) || !mat_ok(L.ffn_up, (p + " ffn_up").c_str()) ||
            !mat_ok(L.ffn_down, (p + " ffn_down").c_str()))
            return false;
    }

    // Size the scratch for one chunk.  A bigger chunk reads every weight fewer times, so the caller halves the
    // chunk when the scratch does not fit rather than switching the path off: even 64 tokens at a time is many
    // times faster than run().
    auto size_scratch = [&](int64_t t) -> bool {
    I.pf.reset(new Impl::Pf);
    Impl::Pf& P = *I.pf;
    P.T = t;
    const uint64_t f = 4, T = (uint64_t) t;
    const int max_in = std::max({E, V, cfg_.n_ff, H * D});
    auto pf_alloc = [&](void** out, uint64_t bytes, const char* what) {
        void* p = nullptr;
        const cudaError_t s = cudaMalloc(&p, bytes ? bytes : 16);
        if (s != cudaSuccess) {
            err = std::string("dense model: cannot allocate the prefill scratch ") + what + " (" +
                  std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(s);
            return false;
        }
        P.allocs.push_back(p);
        *out = p;
        return true;
    };
    bool ok = pf_alloc((void**) &P.x, T * E * f, "x") && pf_alloc((void**) &P.xn, T * E * f, "xn") &&
              pf_alloc((void**) &P.mix, T * E * f, "mix") && pf_alloc((void**) &P.qkv, T * C * f, "qkv") &&
              pf_alloc((void**) &P.h, T * C * f, "h") && pf_alloc((void**) &P.alpha, T * cfg_.ssm_v_heads * f, "alpha") &&
              pf_alloc((void**) &P.beta, T * cfg_.ssm_v_heads * f, "beta") &&
              pf_alloc((void**) &P.gate, T * cfg_.ssm_v_heads * f, "gate") && pf_alloc((void**) &P.z, T * V * f, "z") &&
              pf_alloc((void**) &P.y, T * V * f, "y") && pf_alloc((void**) &P.q_full, T * H * 2 * D * f, "q_full") &&
              pf_alloc((void**) &P.qcur, T * H * D * f, "qcur") && pf_alloc((void**) &P.kcur, T * HK * D * f, "kcur") &&
              pf_alloc((void**) &P.vcur, T * HK * D * f, "vcur") && pf_alloc((void**) &P.attn, T * H * D * f, "attn") &&
              pf_alloc((void**) &P.attn32, T * H * D * f, "attn32") &&
              pf_alloc((void**) &P.ffn_g, T * cfg_.n_ff * f, "ffn_g") &&
              pf_alloc((void**) &P.ffn_u, T * cfg_.n_ff * f, "ffn_u") &&
              pf_alloc((void**) &P.ffn_h, T * cfg_.n_ff * f, "ffn_h") &&
              pf_alloc((void**) &P.xq, strata::prefill::mmq::q8_bytes(t, max_in), "q8_1") &&
              pf_alloc((void**) &P.ids, T * 4, "row ids") && pf_alloc((void**) &P.bounds, 2 * 4, "group bounds") &&
              pf_alloc((void**) &P.pos_dev, T * (uint64_t) std::max(H, HK) * 4, "positions");
    if (ok) {
        cudaError_t s = cudaHostAlloc((void**) &P.stage, T * E * 4, cudaHostAllocDefault);
        if (s != cudaSuccess)
            err = std::string("dense model: cannot pin the prefill embedding stage: ") + cudaGetErrorString(s);
        else s = cudaHostAlloc((void**) &P.bounds_host, Impl::Pf::kBoundsSlots * 2 * sizeof(int32_t),
                               cudaHostAllocDefault);
        if (s != cudaSuccess)
            err = std::string("dense model: cannot pin the prefill group bounds: ") + cudaGetErrorString(s);
        ok = s == cudaSuccess;
    }
    if (ok) {
        strata::prefill::mmq::iota(P.ids, t, I.stream);
        for (int k = 0; k < Impl::Pf::kBoundsSlots; ++k) {
            P.bounds_host[2 * k] = 0;
            P.bounds_host[2 * k + 1] = (int32_t) t;
        }
        const cudaError_t s = cudaMemcpyAsync(P.bounds, P.bounds_host, 2 * sizeof(int32_t), cudaMemcpyHostToDevice,
                                              I.stream);
        if (s != cudaSuccess) err = std::string("dense model: prefill tables: ") + cudaGetErrorString(s);
        ok = s == cudaSuccess;
    }
    if (!ok) I.pf.reset();   // pf_alloc already named the buffer that did not fit
    return ok;
    };
    for (;;) {
        if (size_scratch(chunk)) break;
        if (chunk <= NC) return false;
        std::fprintf(stderr, "strata-dense: the prefill scratch for %lld tokens does not fit; trying %lld\n",
                     (long long) chunk, (long long) (chunk / 2));
        chunk /= 2;
    }
    pf_chunk_ = chunk;
    pf_ready_ = true;
    std::fprintf(stderr, "strata-dense: prompt prefill through MMQ, %lld tokens at a time\n", (long long) chunk);
    return true;
#else
    (void) chunk;
    err = "dense model: this build has no MMQ, so prefill() is unavailable; feed the prompt with run()";
    return false;
#endif
}

bool DenseModel::prefill(const int32_t* tokens, int64_t n, Logits lg, std::string& err) {
#ifdef STRATA_DENSE_MMQ
    Impl& I = *impl_;
    if (!pf_ready_) { err = "dense model: prefill() without prefill_setup()"; return false; }
    if (n < 1) { err = "dense model: prefill() takes at least one token"; return false; }
    if (pos_ + n > max_context_) { err = "dense model: the context is full"; return false; }
    Impl::Pf& P = *I.pf;
    void* s = I.stream;
    const int E = cfg_.n_embd;
    int64_t last_rows = 0;
    try {
        if (I.prof.enabled) I.prof.begin(s);
        for (int64_t i = 0; i < n;) {
            const int64_t T = std::min(pf_chunk_, n - i);
            const int64_t pos0 = pos_;
            if (!I.upload_embeddings(I.embd, tokens + i, (int) T, E, cfg_.n_vocab, P.x, err, P.stage)) {
                I.prof.on = false;
                return false;
            }
            // One MMQ group: the whole chunk.  The bounds go into a fresh pinned slot every time, so the write can
            // never race a copy the stream has not made yet, and a short final chunk cannot leave stale bounds
            // behind for the next call.
            int32_t* slot = P.bounds_host + 2 * (P.bounds_slot++ % Impl::Pf::kBoundsSlots);
            slot[0] = 0;
            slot[1] = (int32_t) T;
            cudaMemcpyAsync(P.bounds, slot, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) s);
            for (int l = 0; l < cfg_.n_layer; ++l) {
                Layer& L = I.layers[(size_t) l];
                dense_rms_norm(P.x, L.attn_norm, P.xn, (int) T, E, cfg_.rms_eps, s);
                if (!L.attn) I.pf_gdn_block(cfg_, L, T, s);
                else I.pf_attn_block(cfg_, L.q, L.k, L.v, L.o, L.q_norm, L.k_norm, L.kc, L.vc, T, (int) pos0, s);
                add_inplace(P.x, P.mix, T * E, s);
                I.pf_ffn_block(cfg_, L.ffn_gate, L.ffn_up, L.ffn_down, L.post_norm, T, s);
            }
            // The MTP block is fed in kMaxCols groups, exactly as the prompt loop does after a run(): each group's
            // column 0 pairs with the previous group's last hidden (h_last, which mtp_ingest leaves behind).
            if (has_mtp_) {
                for (int64_t off = 0; off < T; off += NC) {
                    const int m = (int) std::min<int64_t>(NC, T - off);
                    dense_rms_norm(P.x + (size_t) off * E, I.output_norm, I.hid, m, E, cfg_.rms_eps, s);
                    if (!mtp_ingest(tokens + i + off, m, (int) (pos0 + off), 0, err)) {
                        I.prof.on = false;
                        return false;
                    }
                }
            }
            last_rows = T;
            pos_ += T;
            i += T;
        }
        snap_valid_ = false;
        snap_cols_ = 0;
        if (lg == Logits::Last) {                       // the head runs once, over the last chunk's last row
            dense_rms_norm(P.x + (size_t) (last_rows - 1) * E, I.output_norm, I.xn, 1, E, cfg_.rms_eps, s);
            I.quantize(I.xn, E, 1, s);
            I.gemv(I.head, I.logits, 1, s);
        }
        if (I.prof.on) I.prof.finish(s, (int) std::min<int64_t>(n, INT_MAX));
        return true;
    } catch (const std::exception& e) {
        I.prof.on = false;
        err = std::string("dense model: prefill: ") + e.what();
        return false;
    }
#else
    (void) tokens; (void) n; (void) lg;
    err = "dense model: prefill() is not available in this build";
    return false;
#endif
}

bool DenseModel::mtp_ingest(const int32_t* tokens, int n, int pos0, int start, std::string& err) {
    Impl& I = *impl_;
    if (!has_mtp_) { err = "dense model: no MTP block loaded"; return false; }
    const int m = n - start;
    if (start < 0 || m < 1 || n > NC) { err = "dense model: bad MTP ingest range"; return false; }
    if ((int64_t) pos0 + n > max_context_) { err = "dense model: the context is full"; return false; }
    void* s = I.stream;
    const int E = cfg_.n_embd;
    try {
        if (!I.upload_embeddings(I.embd_mtp, tokens + start, m, E, cfg_.n_vocab, I.m_e, err)) return false;
        for (int j = start; j < n; ++j) {            // hidden[j-1]; column 0 pairs with the hidden before this run
            const float* src = (j == 0) ? I.h_last : I.hid + (size_t) (j - 1) * E;
            cudaMemcpyAsync(I.hin + (size_t) (j - start) * E, src, (size_t) E * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) s);
        }
        for (int jj = 0; jj < m; ++jj) {
            dense_rms_norm(I.m_e + (size_t) jj * E, I.mtp.enorm, I.cat + (size_t) jj * 2 * E, 1, E, cfg_.rms_eps, s);       // e first,
            dense_rms_norm(I.hin + (size_t) jj * E, I.mtp.hnorm, I.cat + (size_t) jj * 2 * E + E, 1, E, cfg_.rms_eps, s);   // then h
        }
        I.mtp_block(cfg_, m, pos0 + start, false, s);
        cudaMemcpyAsync(I.h_last, I.hid + (size_t) (n - 1) * E, (size_t) E * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) s);
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: MTP: ") + e.what();
        return false;
    }
}

bool DenseModel::mtp_draft(int32_t token, int pos, int max_n, float p_min, int32_t* out, int* n_out, std::string& err) {
    Impl& I = *impl_;
    *n_out = 0;
    if (!has_mtp_) { err = "dense model: no MTP block loaded"; return false; }
    void* s = I.stream;
    const int E = cfg_.n_embd;
    try {
        int32_t tok = token;
        for (int i = 0; i < max_n; ++i) {
            if ((int64_t) pos + i >= max_context_) break;
            // step 0 pairs the token with the trunk's last hidden; later steps chain on the block's own output
            if (!I.upload_embeddings(I.embd_mtp, &tok, 1, E, cfg_.n_vocab, I.m_e, err)) return false;
            dense_rms_norm(I.m_e, I.mtp.enorm, I.cat, 1, E, cfg_.rms_eps, s);
            dense_rms_norm(i == 0 ? I.h_last : I.mtp_h, I.mtp.hnorm, I.cat + E, 1, E, cfg_.rms_eps, s);
            I.mtp_block(cfg_, 1, pos + i, true, s);
            dense_argmax_prob(I.mtp_logits, cfg_.n_vocab, I.d_tok, I.d_prob, s);
            int id = -1;
            float prob = 0.0f;
            cudaMemcpyAsync(&id, I.d_tok, sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t) s);
            cudaMemcpyAsync(&prob, I.d_prob, sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t) s);
            if (!sync(err)) return false;
            if (prob < p_min) break;                    // not sure enough: this token is not offered
            out[(*n_out)++] = id;
            tok = id;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: MTP: ") + e.what();
        return false;
    }
}

}  // namespace strata::core