// tools/v41/engine/ds41_generate.cpp - the Strata DeepSeek-V4.1-Flash engine end to end: Ds41Dense (attention,
// Engram, router, shared expert; CUDA or CPU ggml backend) + Ds41MoeTier (routed experts with per-expert formats:
// VRAM cache, pinned RAM arena, file tier, CPU pool, PCIe share) + sampling.  Token ids in, token ids out (text is
// the server's job: tools/v41/serve/serve_ds41.py).
//
//   ds41_generate -m V41-GSQ.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu] [--experts gpu|cpu]
//                 [--engram DIR:DIR:DIR] [--slots auto|N] [--vram-margin GiB] [--arena-gib 60] [--rank r.bin]
//                 [--vram-lru] [--arena-adapt] [--pcie 0.25] [--prefill-chunk N] [--prompt-batch B] [--chunk-mmq] [--ctx N]
//                 [--temp 0] [--seed 1] [--stop id,id] [--ppl] [--no-actquant] [--serve] [--quiet]
//                 [--enc-skip|--no-enc-skip] [--enc-replay N] [--dump-last F]
// --enc-skip (layer-major batches only; DEFAULT for one-shot runs, --no-enc-skip = full path): prompt tokens before the last --enc-replay (default 2560 = 128 * 20) stop
// after layer 20's persistent-KV writes; only the last 2560 tokens run layers 21-39.  Same last-token logits / decode.
//
// --serve speaks Strata's engine line protocol (~/AI/Strata/serve/server.py StrataEngine, as glm_generate):
//   in:  GEN <max_new> [temperature=.. top_p=.. top_k=.. min_p=.. seed=..] <ids csv> | STOP | QUIT
//   out: READY <ctx> stop, RESUME <reused>, PP <read> <total> <ms> <tok/s>, T <id>, DONE ..., ERR <text>
// Conversation reuse: a request that extends what the state has seen continues; one that shares the last prompt
// restores the snapshot taken at its end (Ds41Dense::snapshot: every other cache is position-indexed).
//
// Load the real model only through tools/ds4/memguard.sh (RAM cap + the shared GPU lock).
#include "ds41_dense.hpp"
#include "ds41_moe.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

struct Args {
    std::string model, ids_csv, ids_file, rank, dump_logits, dump_last;
    std::string prune;            // --prune F: "layer expert" lines (coldest first - v41_prune_list.py), soft or hard pruned
    float prune_penalty = 0.0f;   // --prune-penalty X: selection bias -X on the listed experts (0 = masked out of routing)
    float route_bias = 0.0f;      // --route-bias X: selection bonus +X on the experts resident in VRAM at the seed
    std::vector<std::string> engram;
    int n_predict = 64;
    std::string backend = "cuda", experts = "gpu";
    int64_t slots = 0;            // 0 = auto (free VRAM after the dense half, minus --vram-margin)
    double vram_margin_gib = 1.0, pcie = 0.25, arena_gib = 60.0, pf_b = 0.0;
    int threads = 0, prefill_chunk = 0, prompt_batch = 8192;   // prompt_batch: layer-major batch B (0 = the chunk path)
    int64_t ctx = 0;
    float temp = 0.0f;
    uint64_t seed = 1;
    std::vector<int> stop;
    bool ppl = false, serve = false, quiet = false, vram_lru = false, arena_adapt = false, chunk_mmq = false;
    bool act_quant = true, chunk_prefetch = true;
    int  enc_skip = -1;           // -1 = default (on for one-shot layer-major runs), 1 = --enc-skip, 0 = --no-enc-skip; only run the tokens before the replay window through layers 21-39
    int  enc_replay = 2560;       // --enc-replay N: the replay window (window_size * (n_layer - cand_layer) = 128*20)
    float skip_miss = 0.0f, skip_file = 0.0f;
    double chunk_prefetch_gib = 0.0;
};

std::vector<int> parse_csv(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}

bool parse(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "--ids") a.ids_csv = next();
        else if (k == "--ids-file") a.ids_file = next();
        else if (k == "-n" || k == "--n-predict") a.n_predict = std::atoi(next().c_str());
        else if (k == "--backend") a.backend = next();
        else if (k == "--experts") a.experts = next();
        else if (k == "--engram") { std::stringstream ss(next()); std::string d; while (std::getline(ss, d, ':')) a.engram.push_back(d); }
        else if (k == "--slots") { const std::string v = next(); a.slots = v == "auto" ? 0 : std::atoll(v.c_str()); }
        else if (k == "--vram-margin") a.vram_margin_gib = std::atof(next().c_str());
        else if (k == "--arena-gib") a.arena_gib = std::atof(next().c_str());
        else if (k == "--rank") a.rank = next();
        else if (k == "--vram-lru") a.vram_lru = true;
        else if (k == "--arena-adapt") a.arena_adapt = true;
        else if (k == "--pcie") a.pcie = std::atof(next().c_str());
        else if (k == "--pf-b") a.pf_b = std::atof(next().c_str());
        else if (k == "--prefill-chunk") a.prefill_chunk = std::atoi(next().c_str());
        else if (k == "--prompt-batch") a.prompt_batch = std::atoi(next().c_str());
        else if (k == "--chunk-mmq") a.chunk_mmq = true;
        else if (k == "--no-chunk-prefetch") a.chunk_prefetch = false;
        else if (k == "--chunk-prefetch") a.chunk_prefetch_gib = std::atof(next().c_str());
        else if (k == "--skip-miss") a.skip_miss = (float) std::atof(next().c_str());
        else if (k == "--skip-file") a.skip_file = (float) std::atof(next().c_str());
        else if (k == "--threads") a.threads = std::atoi(next().c_str());
        else if (k == "--ctx") a.ctx = std::atoll(next().c_str());
        else if (k == "--temp") a.temp = (float) std::atof(next().c_str());
        else if (k == "--seed") a.seed = (uint64_t) std::atoll(next().c_str());
        else if (k == "--stop") a.stop = parse_csv(next());
        else if (k == "--ppl") a.ppl = true;
        else if (k == "--no-actquant") a.act_quant = false;
        else if (k == "--serve") a.serve = true;
        else if (k == "--prune") a.prune = next();
        else if (k == "--prune-penalty") a.prune_penalty = (float) std::atof(next().c_str());
        else if (k == "--route-bias") a.route_bias = (float) std::atof(next().c_str());
        else if (k == "--quiet") a.quiet = true;
        else if (k == "--dump-logits") a.dump_logits = next();
        else if (k == "--dump-last") a.dump_last = next();
        else if (k == "--enc-skip") a.enc_skip = 1;
        else if (k == "--no-enc-skip") a.enc_skip = 0;
        else if (k == "--enc-replay") a.enc_replay = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && (a.serve || !a.ids_csv.empty() || !a.ids_file.empty());
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// temperature + top-k / min-p / top-p over the top candidates (top_k 0 = the 256 best); temp 0 = argmax
int sample_p(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    for (int i = 0; i < K; ++i) p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp);
    int n = K;
    if (min_p > 0.0f) while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0, c = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    for (int i = 0; i < n; ++i) if ((r -= p[(size_t) i]) <= 0) return idx[(size_t) i];
    return idx[(size_t) n - 1];
}

double log_softmax_at(const float * lg, int nv, int tok) {
    const float mx = *std::max_element(lg, lg + nv);
    double z = 0;
    for (int i = 0; i < nv; ++i) z += std::exp((double) lg[i] - mx);
    return (double) lg[tok] - mx - std::log(z);
}

}  // namespace

int main(int argc, char ** argv) {
    if (!std::getenv("DS41_GGML_DEBUG"))
        ggml_log_set([](ggml_log_level lv, const char * text, void *) { if (lv != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr); }, nullptr);
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: see tools/v41/engine/ds41_generate.cpp\n");
        return 2;
    }
    std::vector<int> prompt = a.ids_csv.empty() ? std::vector<int>() : parse_csv(a.ids_csv);
    if (!a.ids_file.empty()) {
        std::ifstream f(a.ids_file, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "ds41_generate: cannot open %s\n", a.ids_file.c_str()); return 2; }
        prompt.resize((size_t) f.tellg() / 4);
        f.seekg(0);
        f.read((char *) prompt.data(), (std::streamsize) (prompt.size() * 4));
    }
    std::mt19937_64 rng(a.seed);

    // ---- dense half
    ggml_backend_t be = a.backend == "cuda" ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
                                            : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) { std::fprintf(stderr, "ds41_generate: no %s backend\n", a.backend.c_str()); return 1; }
    if (ggml_backend_is_cpu(be)) ggml_backend_cpu_set_n_threads(be, a.threads > 0 ? a.threads : 8);
    int64_t ctx = a.ctx > 0 ? a.ctx : (int64_t) prompt.size() + a.n_predict + 64;
    ctx = std::max<int64_t>(4096, (ctx + 255) / 256 * 256);
    Ds41DenseConfig dc;
    dc.backend = be;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.ctx_max = ctx;
    // the layer-major batch B: --prompt-batch, capped to the prompt length for a one-shot run (the state tensors hold
    // B columns, so a short prompt must not pay for the flag's full size; --serve sizes for the largest request)
    const int PB = a.prompt_batch > 0 ? (a.serve ? a.prompt_batch : (int) std::min<int64_t>(a.prompt_batch, (int64_t) prompt.size())) : 0;
    // the dense sub-batch size S: --prefill-chunk, or 512 with a layer-major prompt batch and no chunk size given
    const int S = a.prefill_chunk > 0 ? a.prefill_chunk : (a.prompt_batch > 0 ? 512 : 1);
    dc.max_pass = S;
    dc.prompt_batch = PB;
    // encoder-only prompt prefill (enc-skip): tokens before the last `enc_replay` need their layers past layer 20 only
    // through the per-layer 128-token window, so for a long prompt they can stop after the KV-writing part of layer
    // 20 (the layer-major batch path only; the chunk path has no per-layer column split, --ppl/--dump-logits need
    // every row's logits, which the skipped rows do not have).
    // default ON (byte-identical last-row logits vs the full path, 8192 tokens, 2026-10-11) for one-shot layer-major runs;
    // --serve keeps it opt-in until the reuse path is exercised
    bool enc_skip = a.enc_skip == 1 || (a.enc_skip < 0 && !a.serve && PB > 1);
    if (enc_skip && a.enc_replay < 1) { std::fprintf(stderr, "ds41_generate: --enc-replay must be >= 1\n"); return 2; }
    if (enc_skip && (a.ppl || !a.dump_logits.empty())) {
        if (a.enc_skip == 1) std::fprintf(stderr, "note: --enc-skip disabled (--ppl/--dump-logits need every row's logits, the old full path)\n");
        enc_skip = false;
    }
    if (enc_skip && PB <= 1) { std::fprintf(stderr, "ds41_generate: --enc-skip needs the layer-major prompt path (--prompt-batch >= 2)\n"); return 2; }
    int64_t ER = enc_skip ? (int64_t) a.enc_replay : 0;
    dc.engram_dirs = a.engram;
    dc.act_quant = a.act_quant;
    Ds41Dense dense;
    std::string err;
    const double t_load = now_ms();
    if (!dense.init(a.model, dc, err)) { std::fprintf(stderr, "ds41_generate: dense init: %s\n", err.c_str()); return 1; }
    const Ds41Geometry & G = dense.geom();
    const int n_layer = (int) dense.n_layer(), n_embd = (int) G.n_embd, top_k = (int) G.n_used, nv = (int) G.vocab;
    // enc-skip's layer 20: the LAST kv source (every compressed-KV / index-key write is owned by a kv source, and the
    // layers above it only read that owner's state, so the cut belongs there).  Also the window reach R = swa *
    // (n_layer - cut_layer) is what --enc-replay defaults to.
    int CL = 0;
    for (int64_t s : G.kv_sources) CL = std::max<int>(CL, (int) s);
    // exactness needs every later layer's window reach to stay inside the replay: reach = swa * (n_layer - CL)
    const int64_t reach = (int64_t) G.n_swa * (int64_t) (n_layer - CL);
    if (enc_skip && ER < reach) {
        std::fprintf(stderr, "note: --enc-replay %lld < the window reach %lld (swa %lld x %d layers past %d); raised\n",
                     (long long) ER, (long long) reach, (long long) G.n_swa, n_layer - CL, CL);
        ER = reach;
    }
    if (enc_skip) std::fprintf(stderr, "enc-skip: cut at layer %d (last kv source), replay window %lld (reach %lld)\n", CL, (long long) ER, (long long) reach);
    if (a.engram.empty()) std::fprintf(stderr, "WARNING: no --engram dirs: the Engram add is skipped (not the real model)\n");

    // ---- routed experts
    Ds41FileBlobs src(a.model);
    if (!src.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom & TG = src.geom();
    double slot_gib = 0;
    if (a.experts != "cpu" && !ggml_backend_is_cpu(be)) {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        slot_gib = std::max(0.0, ((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0);
        std::fprintf(stderr, "VRAM: %.2f GiB free after the dense half -> %.2f GiB of expert slots (margin %.2f)\n",
                     fr / 1073741824.0, slot_gib, a.vram_margin_gib);
    }
    Ds41MoeTier tier;
    Ds41MoeConfig mc;
    mc.cpu_only = a.experts == "cpu" || ggml_backend_is_cpu(be);
    mc.slots = a.slots > 0 ? a.slots : TG.n_layers * TG.n_experts;
    mc.slot_gib = a.slots > 0 ? 0.0 : slot_gib;
    mc.pcie_frac = a.pcie;
    mc.pf_b = a.pf_b;   // predicted prefetch (Ds41Dense::predict -> tier.prefetch), experts per layer
    mc.threads = a.threads;
    mc.arena_gib = a.arena_gib;
    mc.max_arena_gib = a.arena_gib;
    mc.vram_lru = a.vram_lru;
    mc.arena_adapt = a.arena_adapt;
    mc.chunk_mmq = a.chunk_mmq;
    mc.chunk_prefetch = a.chunk_prefetch;
    mc.chunk_prefetch_gib = a.chunk_prefetch_gib;
    mc.skip_miss = a.skip_miss;
    mc.skip_file = a.skip_file;
    // soft / hard pruning (the GLM recipe, as tools/ds4/ds4_generate): listed experts take no VRAM slot and no arena space;
    // with --prune-penalty X they keep a selection bias -X (still chosen when the router prefers them by more than X, then
    // read from the file tier), without it they are never routed
    std::vector<char> pruned;
    int64_t n_pruned = 0;
    if (!a.prune.empty()) {
        pruned.assign((size_t) (TG.n_layers * TG.n_experts), 0);
        std::FILE * pf = std::fopen(a.prune.c_str(), "r");
        if (!pf) { std::fprintf(stderr, "ds41_generate: cannot open %s\n", a.prune.c_str()); return 2; }
        long l, e;
        while (std::fscanf(pf, "%ld %ld", &l, &e) == 2)
            if (l >= 0 && l < TG.n_layers && e >= 0 && e < TG.n_experts && !pruned[(size_t) (l * TG.n_experts + e)]) {
                pruned[(size_t) (l * TG.n_experts + e)] = 1;
                ++n_pruned;
            }
        std::fclose(pf);
        std::fprintf(stderr, "prune: %lld (layer, expert) pairs %s (%s)\n", (long long) n_pruned,
                     a.prune_penalty > 0 ? ("penalised -" + std::to_string(a.prune_penalty)).c_str() : "masked out of routing", a.prune.c_str());
        mc.exclude = pruned;
    }
    if (!tier.init(TG, &src, mc, err)) { std::fprintf(stderr, "ds41_generate: tier init: %s\n", err.c_str()); return 1; }
    // one-shot runs defer the VRAM cache: the prompt's chunk + graph buffers (~4 GiB) use that VRAM first, then the
    // cache opens into it (measured) - the old fixed --vram-margin had to cover prefill and starved decode of slots
    const bool defer_vram = !mc.cpu_only && !a.serve && (a.prefill_chunk > 1 || a.prompt_batch > 1);
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (!mc.cpu_only) {
        if (!a.rank.empty()) {
            ranked = ds41_read_ranked(a.rank, err);
            if (ranked.empty()) { std::fprintf(stderr, "ds41_generate: --rank: %s\n", err.c_str()); return 1; }
        } else {
            for (int64_t l = 0; l < TG.n_layers; ++l)
                for (int64_t e = 0; e < TG.n_experts; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
        }
        const double t0 = now_ms();
        if (defer_vram ? !tier.build_arena_ranked(ranked, err) : !tier.seed_from_ranked(ranked, err)) {
            std::fprintf(stderr, "ds41_generate: seed: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "tier: %lld experts in VRAM, %lld in the RAM arena (%.1f GiB), %lld file tier; seeded in %.1f s\n",
                     (long long) tier.resident(), (long long) tier.arena_experts(), tier.arena_gib(), (long long) tier.file_tier(),
                     (now_ms() - t0) / 1000.0);
    }
    // the selection-only bias per layer: the prune penalty / mask, + route_bias on the experts resident in VRAM now
    auto apply_bias = [&]() {
        if (n_pruned == 0 && a.route_bias == 0.0f) return;
        std::vector<float> rb((size_t) TG.n_experts);
        for (int64_t l = 0; l < TG.n_layers; ++l) {
            for (int64_t e = 0; e < TG.n_experts; ++e)
                rb[(size_t) e] = (!pruned.empty() && pruned[(size_t) (l * TG.n_experts + e)]) ? (a.prune_penalty > 0 ? -a.prune_penalty : -1e30f)
                               : (a.route_bias != 0.0f && !mc.cpu_only && tier.resident(l, e)) ? a.route_bias : 0.0f;
            dense.set_route_bias((int) l, rb.data());
        }
    };
    apply_bias();   // the prune part now (the prompt routes with it); route_bias again once the VRAM cache is seeded
    std::fprintf(stderr, "loaded in %.1f s: %d layers, ctx %lld, prefill chunk %d, experts %s, %zu classes\n",
                 (now_ms() - t_load) / 1000.0, n_layer, (long long) ctx, a.prefill_chunk, tier.mode(), TG.cls_gu.size());

    // ---- the forward pieces
    std::vector<int> ids1((size_t) top_k);
    std::vector<float> w1((size_t) top_k), routed1((size_t) n_embd), logits1((size_t) nv);
    std::string ferr;
    auto step = [&](int tok, int pos) -> bool {   // one token through every layer; logits via dense.logits
        if (!dense.begin(&tok, 1, pos)) { ferr = dense.last_error(); return false; }
        int32_t top16[16];
        for (int l = 0; l < n_layer; ++l) {
            if (mc.pf_b > 0 && !mc.cpu_only) {   // the DMAs run under this layer's attention
                int t16[16];
                if (dense.predict(l, t16, 16)) {
                    for (int i = 0; i < 16; ++i) top16[i] = t16[i];
                    tier.prefetch(l, top16, 16);
                }
            }
            const float * fn = nullptr;
            if (!dense.attn_router(l, ids1.data(), w1.data(), &fn)) { ferr = dense.last_error(); return false; }
            if (!tier.run(l, ids1.data(), w1.data(), fn, routed1.data())) { ferr = "tier.run failed"; return false; }
            if (!dense.finish(l, routed1.data())) { ferr = dense.last_error(); return false; }
        }
        return true;
    };
    const int NP = std::max(1, a.prefill_chunk);
    const int SB = std::max(1, S);                 // the dense sub-batch size
    std::vector<int> cids((size_t) NP * top_k);
    std::vector<float> cw((size_t) NP * top_k), crouted((size_t) NP * n_embd), clog;
    const bool routed_host = std::getenv("STRATA_ROUTED_HOST") != nullptr;
    std::vector<float> brouted;                    // the layer-major path's routed sums (n_embd * B)
    if (PB > 0) brouted.resize((size_t) n_embd * (size_t) PB);
    // a single run_chunk call streams layer l's experts once for its n tokens; the card's chunk buffers (dominated by
    // c_parts = n * top_k * n_embd * 4) must fit ~1 GiB, so a larger batch is run in pieces (each streams once again)
    // budget 3 GB by default (STRATA_CHUNK_BUDGET_GB): at B=8192 the old 1 GB cap split every layer into 2 pieces and each
    // piece re-uploaded the whole layer over PCIe (1.97M vs 0.98M expert copies); the prompt has ~12 GiB of VRAM free
    // (the VRAM expert cache is only seeded after the prompt, from the memory measured then)
    const double chunk_budget = std::getenv("STRATA_CHUNK_BUDGET_GB") ? std::atof(std::getenv("STRATA_CHUNK_BUDGET_GB")) * 1e9 : 3.0e9;
    const int piece = (int) std::max<int64_t>(1, (int64_t) chunk_budget / std::max<int64_t>(1, (int64_t) n_embd * 4 * (top_k + 3)));
    // the prompt ids[0, n) at positions pos0..: last_logits = the last row; all_logits (if given) every row
    std::vector<float> last_logits((size_t) nv);
    double fd_dense = 0, fd_tier = 0, fd_logits = 0;   // prompt wall split (printed with the prompt line)
    bool enc_reported = false;
    // ---- env-gated per-layer dumps (determinism bisect): STRATA_LAYER_DUMP=<dir> writes, for every layer of the
    // sequential chunk path, the routed experts' input (fn, the dense half's output) and their weighted sum, plus an
    // index source layer's top-k ids.  Two runs' files diff at the first layer where a run-to-run difference appears,
    // and fn-vs-routed tells whether it is the MoE tier or the attention / indexer half.
    const char * dump_dir = std::getenv("STRATA_LAYER_DUMP");
    auto dumpf = [&](const char * name, const void * p, size_t bytes) {
        std::ofstream f(std::string(dump_dir) + "/" + name, std::ios::binary);
        f.write((const char *) p, (std::streamsize) bytes);
    };
    auto feed = [&](const int * ids, int n, int pos0, const std::function<void(int)> & progress,
                    std::vector<float> * all_logits) -> bool {
        for (int c0 = 0; c0 < n; ) {
            if (PB > 1 && n - c0 >= 2) {
                // ---- layer-major: up to B tokens of every layer, the experts run once per layer over the whole batch
                const int B = std::min(PB, n - c0);
                // enc-skip: columns whose global position is before `pos0 + n - ER` are phase A (layers past layer 20
                // are never read for them), rounded DOWN to a sub-batch boundary so a straddling sub-batch still runs
                // the full path (more work, never wrong); the last sub-batches holding the final token are always full.
                int cut = 0;
                if (enc_skip) {
                    const int64_t gpos = (int64_t) pos0 + c0;
                    const int64_t ta = (int64_t) pos0 + n - ER - gpos;
                    cut = ta <= 0 ? 0 : ta >= B ? B : (int) ta;
                    cut = (cut / SB) * SB;
                    if (!enc_reported) { enc_reported = true; std::fprintf(stderr, "enc-skip: first batch B=%d, cut=%d (phase A %d cols, phase B %d)\n", B, cut, cut, B - cut); }
                }
                double t0 = now_ms();
                if (!dense.begin_batch(ids + c0, B, pos0 + c0)) { ferr = dense.last_error(); return false; }
                for (int l = 0; l < n_layer; ++l) {
                    // layer 20 is the last kv source: for phase A only its persistent KV writes run; layers 21-39 run
                    // nothing at all for phase A (their output columns are never read, and no later layer reads state
                    // a phase-A position would leave past layer 20)
                    const int d0 = (cut > 0 && l >= CL) ? cut : 0;
                    if (cut > 0 && l == CL) {
                        for (int s = 0; s < cut; s += SB) {
                            const int sn = std::min(SB, cut - s);
                            if (!dense.attn_router_span_kv(l, s, sn)) { ferr = dense.last_error(); return false; }
                        }
                    }
                    for (int s = d0; s < B; s += SB) {   // sub-batches in order: the layer's KV state advances as today
                        const int sn = std::min(SB, B - s);
                        if (!dense.attn_router_span(l, s, sn)) { ferr = dense.last_error(); return false; }
                    }
                    const double t1 = now_ms();
                    fd_dense += t1 - t0;
                    // GPU: the tier writes the routed sums straight into the dense half's routed tensor (no 2 x n*D*4
                    // host round trip per layer); CPU / STRATA_ROUTED_HOST=1: through the host as before
                    float * rdev = routed_host ? nullptr : dense.routed_dev();
                    for (int s = d0; s < B; s += piece) {   // the experts for the dense rows (in pieces if the buffers cap)
                        const int pn = std::min(piece, B - s);
                        const bool ok = rdev ? tier.run_chunk_dev(l, pn, dense.batch_ids() + (size_t) s * top_k,
                                                                  dense.batch_wts() + (size_t) s * top_k,
                                                                  dense.batch_fn() + (size_t) s * n_embd,
                                                                  rdev + (size_t) s * n_embd)
                                             : tier.run_chunk(l, pn, dense.batch_ids() + (size_t) s * top_k,
                                                              dense.batch_wts() + (size_t) s * top_k,
                                                              dense.batch_fn() + (size_t) s * n_embd,
                                                              brouted.data() + (size_t) s * n_embd);
                        if (!ok) { ferr = "tier run refused"; return false; }
                    }
                    t0 = now_ms();
                    fd_tier += t0 - t1;
                    if (!rdev && !dense.upload_routed(brouted.data(), B)) { ferr = dense.last_error(); return false; }
                    for (int s = d0; s < B; s += SB) {
                        const int sn = std::min(SB, B - s);
                        if (!dense.finish_span(l, s, sn)) { ferr = dense.last_error(); return false; }
                    }
                }
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {   // every row, one sub-batch at a time (a B-wide head would need a ~2 GiB buffer)
                    all_logits->resize((size_t) (c0 + B) * (size_t) nv);
                    for (int s = 0; s < B; s += SB) {
                        const int sn = std::min(SB, B - s);
                        if (!dense.logits(s, sn, all_logits->data() + (size_t) (c0 + s) * nv)) { ferr = dense.last_error(); return false; }
                    }
                    if (c0 + B == n) std::copy(all_logits->end() - nv, all_logits->end(), last_logits.begin());
                } else if (c0 + B == n && !dense.logits(B - 1, 1, last_logits.data())) { ferr = dense.last_error(); return false; }
                fd_logits += now_ms() - tl;
                c0 += B;
            } else {
                // ---- the sequential chunk path (unchanged): one pass of up to NP tokens through every layer
                const int m = std::min(NP, n - c0);
                double t0 = now_ms();
                if (!dense.begin(ids + c0, m, pos0 + c0)) { ferr = dense.last_error(); return false; }
                for (int l = 0; l < n_layer; ++l) {
                    const float * fn = nullptr;
                    if (!dense.attn_router(l, cids.data(), cw.data(), &fn)) { ferr = dense.last_error(); return false; }
                    if (dump_dir) {
                        char nb[64];
                        std::snprintf(nb, sizeof(nb), "L%02d_fn.f32", l);
                        dumpf(nb, fn, (size_t) m * (size_t) n_embd * 4);
                        std::vector<int> sel; std::vector<float> smask; int K = 0;
                        if (dense.read_sel(l, sel, smask, &K) && K > 0 && !sel.empty()) {
                            std::snprintf(nb, sizeof(nb), "L%02d_sel.i32", l);
                            dumpf(nb, sel.data(), sel.size() * 4);
                        }
                    }
                    const double t1 = now_ms();
                    fd_dense += t1 - t0;
                    const bool ok = m == 1 ? tier.run(l, cids.data(), cw.data(), fn, crouted.data())
                                           : tier.run_chunk(l, m, cids.data(), cw.data(), fn, crouted.data());
                    if (!ok) { ferr = "tier run refused"; return false; }
                    if (dump_dir) {
                        char nb[64];
                        std::snprintf(nb, sizeof(nb), "L%02d_routed.f32", l);
                        dumpf(nb, crouted.data(), (size_t) m * (size_t) n_embd * 4);
                    }
                    t0 = now_ms();
                    fd_tier += t0 - t1;
                    if (!dense.finish(l, crouted.data())) { ferr = dense.last_error(); return false; }
                }
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {
                    all_logits->resize((size_t) (c0 + m) * (size_t) nv);
                    if (!dense.logits(0, m, all_logits->data() + (size_t) c0 * nv)) { ferr = dense.last_error(); return false; }
                    if (c0 + m == n) std::copy(all_logits->end() - nv, all_logits->end(), last_logits.begin());
                } else if (c0 + m == n && !dense.logits(m - 1, 1, last_logits.data())) { ferr = dense.last_error(); return false; }
                fd_logits += now_ms() - tl;
                c0 += m;
            }
            progress(c0);
        }
        tier.release_chunk();
        return true;
    };

    // ---- serve
    if (a.serve) {
        std::mutex qm;
        std::condition_variable qc;
        std::deque<std::string> q;
        bool eof = false;
        std::thread([&]() {
            std::string ln;
            while (std::getline(std::cin, ln)) {
                std::lock_guard<std::mutex> lk(qm);
                q.push_back(ln);
                qc.notify_all();
            }
            std::lock_guard<std::mutex> lk(qm);
            eof = true;
            qc.notify_all();
        }).detach();
        auto next_line = [&](std::string & out) -> bool {
            std::unique_lock<std::mutex> lk(qm);
            qc.wait(lk, [&] { return !q.empty() || eof; });
            if (q.empty()) return false;
            out = q.front();
            q.pop_front();
            return true;
        };
        auto stop_pending = [&]() -> bool {
            std::lock_guard<std::mutex> lk(qm);
            for (auto it = q.begin(); it != q.end(); ++it)
                if (*it == "STOP") { q.erase(it); return true; }
            return false;
        };
        auto is_stop = [&](int t) { return std::find(a.stop.begin(), a.stop.end(), t) != a.stop.end() || t == (int) G.eos; };
        const int ctxv = (int) ctx;
        std::vector<int> hist;
        int snap_len = -1;
        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld\n", ctxv, (long long) tier.resident(),
                    (long long) tier.arena_experts());
        std::printf("READY %d stop\n", ctxv);
        std::fflush(stdout);
        std::fprintf(stderr, "serve: ready (ctx %d, prefill chunk %d)\n", ctxv, NP);
        std::string line;
        while (next_line(line)) {
            if (line == "QUIT") break;
            if (line == "STOP" || line.empty()) continue;
            if (line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
            std::istringstream ss(line);
            std::string w, csv;
            int max_new = 0, tk = 0;
            float temp = a.temp, top_p = 1.0f, min_p = 0.0f;
            ss >> w >> max_new;
            while (ss >> w) {
                const size_t eq = w.find('=');
                if (eq == std::string::npos) { csv = w; continue; }
                const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                if (k == "temperature") temp = (float) std::atof(v.c_str());
                else if (k == "top_p") top_p = (float) std::atof(v.c_str());
                else if (k == "top_k") tk = std::atoi(v.c_str());
                else if (k == "min_p") min_p = (float) std::atof(v.c_str());
                else if (k == "seed") rng.seed((uint64_t) std::atoll(v.c_str()));
            }
            std::vector<int> ids = csv.empty() ? std::vector<int>() : parse_csv(csv);
            if (ids.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); continue; }
            if ((int64_t) ids.size() + std::max(0, max_new) > ctxv) {
                std::printf("ERR prompt (%zu) + max tokens (%d) exceed the context (%d)\n", ids.size(), max_new, ctxv);
                std::fflush(stdout);
                continue;
            }
            size_t lcp = 0;
            while (lcp < hist.size() && lcp < ids.size() && hist[lcp] == ids[lcp]) ++lcp;
            int reused = 0;
            if (!hist.empty() && lcp == hist.size() && lcp < ids.size()) reused = (int) lcp;
            else if (snap_len > 0 && lcp >= (size_t) snap_len && (size_t) snap_len < ids.size() && dense.restore()) {
                hist.resize((size_t) snap_len);
                reused = snap_len;
            } else {
                hist.clear();
                snap_len = -1;
            }
            std::fprintf(stderr, "serve: reuse lcp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d\n", lcp, hist.size(), snap_len,
                         ids.size(), reused);
            if (reused > 0) { std::printf("RESUME %d\n", reused); std::fflush(stdout); }
            tier.reset_stats();
            const double tp0 = now_ms();
            const int total = (int) ids.size();
            const bool ok = feed(ids.data() + reused, total - reused, reused, [&](int done) {
                const double ms = now_ms() - tp0;
                std::printf("PP %d %d %.0f %.1f\n", reused + done, total, ms, 1000.0 * done / std::max(ms, 1e-9));
                std::fflush(stdout);
            }, nullptr);
            if (!ok) {
                std::printf("ERR prefill failed: %s\n", ferr.c_str());
                std::fflush(stdout);
                hist.clear();
                snap_len = -1;
                continue;
            }
            hist = ids;
            if (dense.snapshot(total)) snap_len = total;
            const double prompt_ms = now_ms() - tp0;
            std::fprintf(stderr, "serve: prompt %d tokens (%d reused) in %.2f s = %.1f tok/s\n", total, reused,
                         prompt_ms / 1000.0, 1000.0 * (total - reused) / std::max(prompt_ms, 1e-9));
            tier.reset_stats();
            const double td0 = now_ms();
            int gen = 0, pos = total;
            const char * finish = "length";
            std::vector<float> lg = last_logits;
            while (gen < max_new || max_new <= 0) {
                if (stop_pending()) { finish = "cancel"; break; }
                const int tok = sample_p(lg.data(), nv, temp, tk, top_p, min_p, rng);
                std::printf("T %d\n", tok);
                std::fflush(stdout);
                ++gen;
                if (is_stop(tok)) { finish = "stop"; break; }
                if ((max_new > 0 && gen >= max_new) || pos + 1 >= ctxv) break;
                if (!step(tok, pos) || !dense.logits(0, 1, lg.data())) { finish = "error"; break; }
                hist.push_back(tok);
                ++pos;
            }
            const double dec_ms = now_ms() - td0;
            const Ds41MoeStats st = tier.stats();
            if (std::strcmp(finish, "error") == 0) {
                std::fprintf(stderr, "serve: decode failed: %s\n", ferr.c_str());
                hist.clear();
                snap_len = -1;
                finish = "length";
            }
            std::fprintf(stderr, "serve: %d tokens in %.2f s = %.2f tok/s (%s); experts: %lld hit, %lld cpu, %lld pcie, %lld file\n",
                         gen, dec_ms / 1000.0, 1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), finish,
                         (long long) st.hits, (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier);
            std::printf("DONE %d %d %.1f %.1f %s 0 0 %d %lld %lld 0 %lld 0 %d\n", gen, total, prompt_ms, dec_ms, finish, reused,
                        (long long) st.hits, (long long) st.lookups(), (long long) st.file_tier, total - reused);
            std::fflush(stdout);
        }
        tier.close();
        return 0;
    }

    // ---- one-shot: prompt (+ perplexity), then generate
    if (prompt.empty()) { std::fprintf(stderr, "ds41_generate: empty prompt\n"); return 2; }
    tier.reset_stats();
    const double tp0 = now_ms();
    std::vector<float> all;
    const bool ok = feed(prompt.data(), (int) prompt.size(), 0, [&](int done) {
        if (!a.quiet) std::fprintf(stderr, "\rprompt %d/%zu", done, prompt.size());
    }, a.ppl || !a.dump_logits.empty() ? &all : nullptr);
    if (!ok) { std::fprintf(stderr, "\nds41_generate: prefill: %s\n", ferr.c_str()); return 1; }
    const double prompt_ms = now_ms() - tp0;
    if (defer_vram) {   // the prompt is done: its buffers go back, the VRAM expert cache opens into the free space
        dense.release_pass();
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        const double g = std::max(0.0, ((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0);
        tier.set_slot_gib(a.slots > 0 ? 0.0 : g);
        const double ts = now_ms();
        if (!tier.seed_from_ranked(ranked, err)) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
        apply_bias();
        std::fprintf(stderr, "\nVRAM after the prompt: %.2f GiB free -> %.2f GiB of expert slots: %lld experts in VRAM (%.1f s)",
                     fr / 1073741824.0, g, (long long) tier.resident(), (now_ms() - ts) / 1000.0);
    }
    const Ds41MoeStats pst = tier.stats();
    std::fprintf(stderr, "\nprompt: %zu tokens in %.2f s = %.1f tok/s (experts %lld hit / %lld cpu / %lld pcie / %lld file)\n",
                 prompt.size(), prompt_ms / 1000.0, 1000.0 * prompt.size() / prompt_ms, (long long) pst.hits,
                 (long long) pst.cpu, (long long) pst.pcie, (long long) pst.file_tier);
    std::fprintf(stderr, "prompt split: dense %.1f s, experts %.1f s (file wait %.1f s), logits %.1f s\n", fd_dense / 1000.0,
                 fd_tier / 1000.0, pst.file_ms / 1000.0, fd_logits / 1000.0);
    if (a.ppl) {
        double nll = 0;
        int n = 0, top1 = 0;
        for (size_t t = 0; t + 1 < prompt.size(); ++t) {
            const float * lg = all.data() + t * (size_t) nv;
            nll -= log_softmax_at(lg, nv, prompt[t + 1]);
            top1 += (int) (std::max_element(lg, lg + nv) - lg) == prompt[t + 1];
            ++n;
        }
        std::printf("PPL %.4f  mean NLL %.4f over %d tokens, top-1 %.3f\n", std::exp(nll / std::max(1, n)), nll / std::max(1, n), n,
                    (double) top1 / std::max(1, n));
    }
    if (!a.dump_logits.empty()) {
        std::ofstream f(a.dump_logits, std::ios::binary);
        f.write((const char *) all.data(), (std::streamsize) (all.size() * 4));
    }
    if (!a.dump_last.empty()) {   // the last row's logits - the one row enc-skip still computes exactly
        std::ofstream f(a.dump_last, std::ios::binary);
        f.write((const char *) last_logits.data(), (std::streamsize) (last_logits.size() * 4));
    }
    tier.reset_stats();
    const double td0 = now_ms();
    std::vector<float> lg = last_logits;
    int pos = (int) prompt.size(), gen = 0;
    for (; gen < a.n_predict; ++gen) {
        const int tok = sample_p(lg.data(), nv, a.temp, 0, 1.0f, 0.0f, rng);
        std::printf("%d\n", tok);
        std::fflush(stdout);
        if (std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end() || tok == (int) G.eos) { ++gen; break; }
        if (gen + 1 == a.n_predict) { ++gen; break; }
        if (!step(tok, pos) || !dense.logits(0, 1, lg.data())) { std::fprintf(stderr, "ds41_generate: decode: %s\n", ferr.c_str()); return 1; }
        ++pos;
    }
    const double dec_ms = now_ms() - td0;
    const Ds41MoeStats st = tier.stats();
    std::fprintf(stderr, "decode: %d tokens in %.2f s = %.2f tok/s; experts %lld hit / %lld cpu / %lld pcie / %lld file "
                 "(%.1f%% VRAM hits), cpu %.0f ms, file %.0f ms\n",
                 gen, dec_ms / 1000.0, 1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), (long long) st.hits,
                 (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier,
                 100.0 * st.hits / std::max<int64_t>(1, st.lookups()), st.cpu_ms, st.file_ms);
    tier.close();
    return 0;
}
