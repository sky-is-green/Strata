// tools/v41/engine/ds41_engram.cpp - see ds41_engram.hpp.
#include "ds41_engram.hpp"

#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::v41 {

namespace {

std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool exists(const std::string & p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }

// "key": "value" in a JSON text (the HF index's weight_map; names are unique)
std::string json_str(const std::string & js, const std::string & key) {
    const std::string pat = "\"" + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return {};
    p = js.find('"', js.find(':', p + pat.size()) + 1);
    const size_t q = js.find('"', p + 1);
    return p == std::string::npos || q == std::string::npos ? std::string() : js.substr(p + 1, q - p - 1);
}

// a safetensors tensor's absolute byte offset (8 + header length + data_offsets[0]) and its byte size
bool st_tensor(const std::string & path, const std::string & name, uint64_t & off, uint64_t & bytes, std::string & dtype) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    uint64_t hl = 0;
    bool ok = ::pread(fd, &hl, 8, 0) == 8 && hl < (1ull << 30);
    std::string h;
    if (ok) {
        h.resize(hl);
        ok = ::pread(fd, h.data(), hl, 8) == (ssize_t) hl;
    }
    ::close(fd);
    if (!ok) return false;
    const size_t p = h.find("\"" + name + "\"");
    if (p == std::string::npos) return false;
    const size_t e = h.find('}', p);
    const std::string ent = h.substr(p, e - p);
    dtype = json_str(ent, "dtype");
    const size_t d = ent.find("\"data_offsets\"");
    if (d == std::string::npos) return false;
    const size_t b = ent.find('[', d);
    char * end = nullptr;
    const uint64_t a0 = std::strtoull(ent.c_str() + b + 1, &end, 10);
    const uint64_t a1 = std::strtoull(std::strchr(end, ',') + 1, nullptr, 10);
    off = 8 + hl + a0;
    bytes = a1 - a0;
    return true;
}

// FP8 E4M3 (fn: no inf, S.1111.111 = NaN) -> float
float fp8_e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = std::ldexp((float) m, -9);                 // subnormal: m/8 * 2^-6
    else if (e == 15 && m == 7) v = NAN;
    else v = std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}

}  // namespace

Ds41Engram::~Ds41Engram() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto & t : workers_) t.join();
    for (auto & t : tables_) {
        if (t.fd_w >= 0) ::close(t.fd_w);
        if (t.fd_s >= 0) ::close(t.fd_s);
    }
}

bool Ds41Engram::open(const Ds41Geometry & g, const std::vector<std::string> & dirs, int threads, std::string & err) {
    g_ = &g;
    cols_ = (g.engram_ngram - 1) * g.engram_heads;
    dim_ = g.engram_dim;
    const int64_t nel = (int64_t) g.engram_layers.size();
    offsets_.assign((size_t) (nel * cols_), 0);
    for (int64_t e = 0; e < nel; ++e) {
        uint64_t acc = 0;
        for (int64_t c = 0; c < cols_; ++c) {
            offsets_[(size_t) (e * cols_ + c)] = (int64_t) acc;
            acc += g.engram_primes[(size_t) (e * cols_ + c)];
        }
    }
    std::string index;
    for (const auto & d : dirs) {
        index = slurp(d + "/model.safetensors.index.json");
        if (!index.empty()) break;
    }
    if (index.empty()) { err = "engram: no model.safetensors.index.json in the given dirs"; return false; }
    for (int64_t e = 0; e < nel; ++e) {
        const std::string base = "layers." + std::to_string(g.engram_layers[(size_t) e]) + ".engram.embed.";
        Table t;
        for (int k = 0; k < 2; ++k) {
            const std::string name = base + (k ? "scale" : "weight");
            const std::string file = json_str(index, name);
            if (file.empty()) { err = "engram: " + name + " not in the index"; return false; }
            std::string path;
            for (const auto & d : dirs) if (exists(d + "/" + file)) { path = d + "/" + file; break; }
            if (path.empty()) { err = "engram: " + file + " not found in the given dirs"; return false; }
            uint64_t off = 0, bytes = 0;
            std::string dt;
            if (!st_tensor(path, name, off, bytes, dt)) { err = "engram: cannot read " + name + " from " + path; return false; }
            const int64_t rows = g.engram_rows[(size_t) e];
            if (bytes != (uint64_t) rows * (k ? (uint64_t) dim_ / 32 : (uint64_t) dim_) ||
                dt != (k ? "F8_E8M0" : "F8_E4M3")) {
                err = "engram: " + name + " has an unexpected dtype/size";
                return false;
            }
            const int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) { err = "engram: cannot open " + path; return false; }
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
            (k ? t.fd_s : t.fd_w) = fd;
            (k ? t.off_s : t.off_w) = off;
            t.rows = rows;
        }
        tables_.push_back(t);
    }
    for (int i = 0; i < std::max(0, threads - 1); ++i)   // the caller is the last worker
        workers_.emplace_back([this] {
            int64_t seen = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                    if (stop_) return;
                    seen = gen_;
                }
                run_parts();
                std::lock_guard<std::mutex> lk(mu_);
                if (--active_ == 0) done_cv_.notify_all();
            }
        });
    return true;
}

void Ds41Engram::set_token(int64_t pos, int tid) {
    if (pos >= (int64_t) hist_.size()) hist_.resize((size_t) pos + 1024, 0);
    hist_[(size_t) pos] = g_->engram_token_map[(size_t) tid];
}

void Ds41Engram::hash(int64_t pos, int e, int64_t * out) const {
    const int64_t N = g_->engram_ngram, H = g_->engram_heads;
    int64_t tok[8];
    for (int64_t k = 0; k < N; ++k) tok[k] = pos - k >= 0 ? hist_[(size_t) (pos - k)] : g_->engram_pad;
    const int64_t * mult = &g_->engram_mult[(size_t) (e * N)];
    int64_t rolling = tok[0] * mult[0];
    for (int64_t i = 1; i < N; ++i) {
        rolling ^= tok[i] * mult[i];
        for (int64_t h = 0; h < H; ++h) {
            const int64_t c = (i - 1) * H + h;
            const int64_t p = (int64_t) g_->engram_primes[(size_t) (e * cols_ + c)];
            out[c] = rolling % p + offsets_[(size_t) (e * cols_ + c)];
        }
    }
}

bool Ds41Engram::rows(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    out.resize((size_t) nel);
    std::vector<int64_t> ids((size_t) (nel * n * cols_));
    for (int64_t e = 0; e < nel; ++e) {
        out[(size_t) e].assign((size_t) (n * cols_ * dim_), 0.0f);
        for (int t = 0; t < n; ++t) hash(pos0 + t, (int) e, &ids[(size_t) ((e * n + t) * cols_)]);
    }
    const int64_t total = nel * n * cols_;
    std::atomic<int> bad { 0 };
    std::atomic<uint64_t> bytes { 0 };
    const std::function<void(int64_t, int64_t)> fn = [&](int64_t a, int64_t b) {
        uint8_t w[512], s[16];
        for (int64_t j = a; j < b; ++j) {
            const int64_t e = j / (n * cols_);
            const Table & T = tables_[(size_t) e];
            const int64_t r = ids[(size_t) j];
            if (r < 0 || r >= T.rows ||
                ::pread(T.fd_w, w, (size_t) dim_, (off_t) (T.off_w + (uint64_t) r * (uint64_t) dim_)) != (ssize_t) dim_ ||
                ::pread(T.fd_s, s, (size_t) (dim_ / 32), (off_t) (T.off_s + (uint64_t) r * (uint64_t) (dim_ / 32))) != (ssize_t) (dim_ / 32)) {
                bad.fetch_add(1);
                continue;
            }
            bytes.fetch_add((uint64_t) (dim_ + dim_ / 32));
            float * o = &out[(size_t) e][(size_t) ((j % (n * cols_)) * dim_)];
            for (int64_t q = 0; q < dim_ / 32; ++q) {
                const float sc = std::ldexp(1.0f, (int) s[q] - 127);
                for (int64_t i = 0; i < 32; ++i) o[q * 32 + i] = fp8_e4m3(w[q * 32 + i]) * sc;
            }
        }
    };
    parallel(total, fn);
    bytes_ += bytes.load();
    if (bad.load()) { err = "engram: " + std::to_string(bad.load()) + " row reads failed"; return false; }
    return true;
}

void Ds41Engram::run_parts() {
    for (;;) {
        const int64_t k = next_.fetch_add(1);
        if (k >= parts_) return;
        const int64_t a = job_n_ * k / parts_, b = job_n_ * (k + 1) / parts_;
        if (a < b) (*job_)(a, b);
    }
}

void Ds41Engram::parallel(int64_t n, const std::function<void(int64_t, int64_t)> & fn) {
    if (workers_.empty() || n < 4) { fn(0, n); return; }
    {
        std::lock_guard<std::mutex> lk(mu_);
        job_ = &fn;
        job_n_ = n;
        parts_ = std::min<int64_t>(n, 4 * (int64_t) (workers_.size() + 1));
        next_.store(0);
        active_ = (int) workers_.size();
        ++gen_;
    }
    cv_.notify_all();
    run_parts();
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return active_ == 0; });
}

}  // namespace strata::v41
