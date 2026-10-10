// tools/v41/engine/ds41_engram.hpp - DeepSeek-V4.1 Engram: the n-gram hash (model.py / engram.py NgramHashState)
// and the row reader.  The tables are NOT copied anywhere: rows are read straight from the HF safetensors shards
// (layers.{1,14}.engram.embed.weight F8_E4M3 [rows, 256] + .scale F8_E8M0 [rows, 8]), two small preads per row,
// in parallel.  A row dequantizes to 256 floats = fp8 * 2^(e8m0 - 127) per 32 (model.py ParallelEngramEmbedding;
// exactly representable in bf16, so model.py's bf16 cast is a no-op on them).
#pragma once

#include "ds41_geometry.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::v41 {

class Ds41Engram {
public:
    ~Ds41Engram();
    /// dirs: where the shards are (the index json is looked up in each, the shard files too).
    bool open(const Ds41Geometry & g, const std::vector<std::string> & dirs, int threads, std::string & err);
    bool enabled() const { return !tables_.empty(); }
    int64_t n_cols() const { return cols_; }                    ///< hash columns per token per layer (24)
    int64_t row_dim() const { return dim_; }                    ///< 256
    /// Record token `tid` at position `pos` (history for the hashes; positions may be rewritten).
    void set_token(int64_t pos, int tid);
    /// Hash ids of the n-grams ending at `pos` for Engram slot `e` (0 = layer 1, 1 = layer 14): out[cols].
    void hash(int64_t pos, int e, int64_t * out) const;
    /// Rows for tokens at positions pos0..pos0+n-1, every Engram slot: out[e] = n * cols * dim floats (token-major).
    bool rows(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err);
    /// bytes read so far (stats)
    uint64_t bytes_read() const { return bytes_; }

private:
    struct Table { int fd_w = -1, fd_s = -1; uint64_t off_w = 0, off_s = 0; int64_t rows = 0; };
    const Ds41Geometry * g_ = nullptr;
    std::vector<Table> tables_;
    std::vector<int64_t> hist_;      // compressed ids by position
    std::vector<int64_t> offsets_;   // [slot * cols + col] bucket range start
    int64_t cols_ = 0, dim_ = 0;
    uint64_t bytes_ = 0;
    // a small fork-join pool for the reads
    void parallel(int64_t n, const std::function<void(int64_t, int64_t)> & fn);
    void run_parts();
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int64_t, int64_t)> * job_ = nullptr;
    int64_t job_n_ = 0, gen_ = 0, parts_ = 1;
    std::atomic<int64_t> next_ { 0 };
    int active_ = 0;
    bool stop_ = false;
};

}  // namespace strata::v41
