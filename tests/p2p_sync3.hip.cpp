// Cross-device ordering primitives, clean take (no clock64, bounded polls with readouts).
//   E1              cross-device hipStreamWaitEvent (control: C1 without the wait)
//   E2              doorbell by DMA (dest stream) + hipStreamWaitValue32 on the local flag
//   E3              doorbell by DMA + kernel poll with __threadfence_system() each iteration
//   E4              doorbell by DMA + kernel poll with a non-temporal (cache-bypassing) load
//   E5              doorbell by a peer KERNEL store + non-temporal poll
//   E6              doorbell into pinned host memory + non-temporal poll
// Build: /opt/rocm/bin/hipcc -O2 --offload-arch=gfx1100 tests/p2p_sync3.hip.cpp -o build-hip/p2p_sync3
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

__global__ void delay_k(volatile unsigned long long* sink, unsigned long long iters) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long a = 1;
    for (unsigned long long i = 0; i < iters; ++i) a = a * 1664525ull + 1013904223ull;
    *sink = a;
}
__global__ void stamp_k(unsigned long long* res, unsigned int v) {
    if (blockIdx.x || threadIdx.x) return;
    *res = v;
}
__global__ void poll_fence_k(uint32_t* f, uint32_t want, unsigned long long budget, unsigned long long* res) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long polls = 0;
    while (true) {
        __threadfence_system();
        if (*(volatile uint32_t*) f == want) break;
        if (++polls >= budget) { res[0] = 0; res[1] = polls; return; }
    }
    res[0] = 1; res[1] = polls;
}
__device__ __forceinline__ uint32_t nt_load(const uint32_t* p) {
    uint32_t v;
    asm volatile("flat_load_dword %0, %1 glc slc" : "=v"(v) : "v"(reinterpret_cast<const unsigned long long>(p)) : "memory");
    return v;
}
__global__ void poll_nt_k(const uint32_t* f, uint32_t want, unsigned long long budget, unsigned long long* res) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long polls = 0;
    while (nt_load(f) != want) {
        if (++polls >= budget) { res[0] = 0; res[1] = polls; return; }
    }
    res[0] = 1; res[1] = polls;
}
__global__ void store_peer_k(uint32_t* peerflag, uint32_t v) { __threadfence_system(); *(volatile uint32_t*) peerflag = v; }
__device__ __forceinline__ float nt_loadf(const float* p) {
    float v;
    asm volatile("flat_load_dword %0, %1 glc slc" : "=v"(v) : "v"(reinterpret_cast<const unsigned long long>(p)) : "memory");
    return v;
}
__device__ __forceinline__ void backoff(int n) {
    float x = 1.0f;
    for (int k = 0; k < n; ++k) x = fmaf(x, 1.0000001f, 1e-7f);
    if (x == 12345.6789f) asm volatile("" ::: "memory");
}
__global__ void e7_producer_k(volatile uint32_t* flag, int R) {
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 1; i <= R; ++i) { *flag = (uint32_t) i; }
}
__global__ void e7_poller_k(const uint32_t* flag, int R, unsigned long long* polls) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long p = 0;
    for (int i = 1; i <= R; ++i) {
        while (nt_load((const uint32_t*) flag) - (uint32_t) i > 0x80000000u) ++p;
    }
    *polls = p;
}
__global__ void e8_a_k(volatile uint32_t* flagA, const uint32_t* flagB, int R) {
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 1; i <= R; ++i) {
        *flagA = (uint32_t) i;
        while (nt_load((const uint32_t*) flagB) - (uint32_t) i > 0x80000000u) backoff(400);
    }
}
__global__ void e8_b_k(volatile uint32_t* flagB, const uint32_t* flagA, int R) {
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 1; i <= R; ++i) {
        while (nt_load((const uint32_t*) flagA) - (uint32_t) i > 0x80000000u) backoff(400);
        *flagB = (uint32_t) i;
    }
}
__global__ void e9_prod_k(float* data, volatile uint32_t* flag, const uint32_t* ack, int n, int R) {
    const int side = gridDim.x * blockDim.x;
    for (int i = 1; i <= R; ++i) {
        for (int j = threadIdx.x; j < n; j += side) data[j] = (float) (i + j);
        __syncthreads();
        __threadfence_system();
        if (threadIdx.x == 0) *flag = (uint32_t) i;
        __syncthreads();
        if (threadIdx.x == 0) { while (nt_load(ack) - (uint32_t) i > 0x80000000u) {} }
        __syncthreads();
    }
}
__global__ void e9_cons_k(const float* data, const uint32_t* flag, uint32_t* ack, int n, int R, float* sink, unsigned long long* polls) {
    const int side = gridDim.x * blockDim.x;
    float acc = 0;
    unsigned long long p = 0;
    for (int i = 1; i <= R; ++i) {
        while (nt_load((const uint32_t*) flag) - (uint32_t) i > 0x80000000u) ++p;
        for (int j = threadIdx.x; j < n; j += side) acc += nt_loadf(data + j);
        __syncthreads();
        if (threadIdx.x == 0) *ack = (uint32_t) i;
        __syncthreads();
    }
    sink[threadIdx.x] = acc;
    if (threadIdx.x == 0) *polls = p;
}
static double ms_now() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static const char* err(hipError_t e) { return hipGetErrorName(e); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* which = argc > 1 ? argv[1] : "help";
    int ndev = 0; hipGetDeviceCount(&ndev);
    if (ndev < 2) { std::printf("devices %d\n", ndev); return 0; }
    hipSetDevice(1); hipError_t pe1 = hipDeviceEnablePeerAccess(0, 0);
    hipSetDevice(0); hipError_t pe0 = hipDeviceEnablePeerAccess(1, 0);
    hipStream_t s0 = nullptr, s1 = nullptr;
    hipSetDevice(0); hipStreamCreateWithFlags(&s0, hipStreamNonBlocking);
    hipSetDevice(1); hipStreamCreateWithFlags(&s1, hipStreamNonBlocking);
    uint32_t *f0 = nullptr, *f1 = nullptr, *fh = nullptr;
    hipSetDevice(0); hipMalloc(&f0, 16); hipMemset(f0, 0, 16);
    hipSetDevice(1); hipMalloc(&f1, 16); hipMemset(f1, 0, 16);
    hipHostMalloc((void**) &fh, 16, hipHostMallocPortable | hipHostMallocMapped);
    *fh = 0;
    hipStreamSynchronize(s0); hipStreamSynchronize(s1);
    std::printf("peer enable %s / %s\n", err(pe1), err(pe0));

    if (!std::strcmp(which, "D0")) {
        unsigned long long* sink = nullptr;
        hipSetDevice(0); hipMalloc(&sink, 8); hipMemset(sink, 0, 8);
        hipStreamSynchronize(s0);
        const double t0 = ms_now();
        hipLaunchKernelGGL(delay_k, dim3(1), dim3(1), 0, s0, sink, 50000000ull);
        hipError_t y = hipStreamSynchronize(s0);
        unsigned long long v = 0; hipMemcpy(&v, sink, 8, hipMemcpyDeviceToHost);
        std::printf("D0 delay_k 5e7 iters: sync=%s sink=%llu -> %.1f ms\n", err(y), v, ms_now() - t0);
        return 0;
    }
    if (!std::strcmp(which, "E1") || !std::strcmp(which, "C1")) {
        const bool wait = !std::strcmp(which, "E1");
        unsigned long long *sink = nullptr, *res1 = nullptr;
        hipSetDevice(0); hipMalloc(&sink, 8); hipMemset(sink, 0, 8);
        hipSetDevice(1); hipMalloc(&res1, 8); hipMemset(res1, 0, 8);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        hipEvent_t ev = nullptr; hipEventCreateWithFlags(&ev, hipEventDisableTiming);
        const double t0 = ms_now();
        hipSetDevice(0);
        hipLaunchKernelGGL(delay_k, dim3(1), dim3(1), 0, s0, sink, 50000000ull);   // ~0.1-0.4 s of dependent math
        hipEventRecord(ev, s0);
        hipSetDevice(1);
        if (wait) hipStreamWaitEvent(s1, ev, 0);
        hipLaunchKernelGGL(stamp_k, dim3(1), dim3(1), 0, s1, res1, 42u);
        hipError_t y1 = hipStreamSynchronize(s1);
        const double dt = ms_now() - t0;
        unsigned long long r = 0; hipMemcpy(&r, res1, 8, hipMemcpyDeviceToHost);
        hipStreamSynchronize(s0);
        std::printf("%s waitEvent=%d: sync=%s stamp=%llu -> wall %.1f ms (wait gated => >= ~100 ms; ignored => ~0 ms)\n",
                    which, (int) wait, err(y1), r, dt);
        return 0;
    }
    if (!std::strcmp(which, "E2") || !std::strcmp(which, "E3") || !std::strcmp(which, "E4")) {
        // doorbell: the DMA engine writes the value into the poller's local flag (copy on the dest stream)
        const uint32_t val = 0x2323;
        uint32_t* src = nullptr; unsigned long long* res = nullptr;
        hipSetDevice(0); hipMalloc(&src, 4); hipMemcpy(src, &val, 4, hipMemcpyHostToDevice);
        hipSetDevice(1); hipMemset(f1, 0, 16); hipMalloc(&res, 16); hipMemset(res, 0, 16);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        const double t0 = ms_now();
        if (!std::strcmp(which, "E2")) {
            hipSetDevice(1);
            hipError_t w = hipStreamWaitValue32(s1, f1, val, hipStreamWaitValueEq, 0xffffffffu);
            hipLaunchKernelGGL(stamp_k, dim3(1), dim3(1), 0, s1, res, 7u);
            hipError_t c = hipMemcpyPeerAsync(f1, 1, src, 0, 4, s1);
            std::printf("E2: wait=%s copy=%s; syncing\n", err(w), err(c));
            hipError_t y1 = hipStreamSynchronize(s1);
            std::printf("E2 waitValue32 + DMA doorbell: sync=%s after %.2f ms\n", err(y1), ms_now() - t0);
        } else {
            hipSetDevice(1);
            if (!std::strcmp(which, "E3")) hipLaunchKernelGGL(poll_fence_k, dim3(1), dim3(1), 0, s1, f1, val, 200000000ull, res);
            else                           hipLaunchKernelGGL(poll_nt_k, dim3(1), dim3(1), 0, s1, f1, val, 200000000ull, res);
            std::printf("%s: poller launched (%s)\n", which, err(hipGetLastError()));
            usleep(30000);
            hipError_t c = hipMemcpyPeerAsync(f1, 1, src, 0, 4, s1);
            std::printf("%s: DMA doorbell enqueued (%s)\n", which, err(c));
            hipError_t y1 = hipStreamSynchronize(s1);
            unsigned long long h[2] = {0, 0}; uint32_t hv = 0;
            hipMemcpy(h, res, 16, hipMemcpyDeviceToHost);
            hipMemcpy(&hv, f1, 4, hipMemcpyDeviceToHost);
            std::printf("%s: sync=%s seen=%llu/%llu polls, f1=0x%x -> wall %.2f ms\n", which, err(y1), h[0], h[1], hv, ms_now() - t0);
        }
        return 0;
    }
    if (!std::strcmp(which, "E5") || !std::strcmp(which, "E6")) {
        const uint32_t val = 0x4545;
        unsigned long long* res = nullptr;
        uint32_t* target = !std::strcmp(which, "E5") ? f1 : fh;
        hipSetDevice(1); hipMalloc(&res, 16); hipMemset(res, 0, 16); hipMemset(f1, 0, 16);
        *fh = 0;
        hipStreamSynchronize(s1);
        const double t0 = ms_now();
        hipSetDevice(1);
        hipLaunchKernelGGL(poll_nt_k, dim3(1), dim3(1), 0, s1, target, val, 200000000ull, res);
        std::printf("%s: poller on dev1 launched (%s)\n", which, err(hipGetLastError()));
        usleep(30000);
        hipSetDevice(0);
        if (!std::strcmp(which, "E5")) hipLaunchKernelGGL(store_peer_k, dim3(1), dim3(1), 0, s0, f1, val);
        else                           hipLaunchKernelGGL(store_peer_k, dim3(1), dim3(1), 0, s0, fh, val);
        hipError_t ls = hipGetLastError();
        hipError_t y0 = hipStreamSynchronize(s0);
        std::printf("%s: peer store launched (%s/%s); syncing poller\n", which, err(ls), err(y0));
        hipError_t y1 = hipStreamSynchronize(s1);
        unsigned long long h[2] = {0, 0}; uint32_t hv = 0;
        hipMemcpy(h, res, 16, hipMemcpyDeviceToHost);
        if (!std::strcmp(which, "E6")) hv = *fh; else hipMemcpy(&hv, target, 4, hipMemcpyDeviceToHost);
        std::printf("%s: sync=%s seen=%llu/%llu polls, value=0x%x -> wall %.2f ms\n", which, err(y1), h[0], h[1], hv, ms_now() - t0);
        return 0;
    }
    if (!std::strcmp(which, "E7")) {
        const int R = argc > 2 ? std::atoi(argv[2]) : 20000;
        unsigned long long* polls = nullptr;
        *fh = 0;
        hipSetDevice(1); hipMalloc(&polls, 8); hipMemset(polls, 0, 8);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        const double t0 = ms_now();
        hipSetDevice(0); hipLaunchKernelGGL(e7_producer_k, dim3(1), dim3(1), 0, s0, (volatile uint32_t*) fh, R);
        hipSetDevice(1); hipLaunchKernelGGL(e7_poller_k, dim3(1), dim3(1), 0, s1, fh, R, polls);
        hipError_t y1 = hipStreamSynchronize(s1);
        const double dt = ms_now() - t0;
        hipError_t y0 = hipStreamSynchronize(s0);
        unsigned long long p = 0; hipMemcpy(&p, polls, 8, hipMemcpyDeviceToHost);
        std::printf("E7 one-way ring->detect R=%d: %s/%s -> %.2f us/hop (%llu polls total)\n", R, err(y0), err(y1), dt * 1000.0 / R, p);
        return 0;
    }
    if (!std::strcmp(which, "E8")) {
        const int R = argc > 2 ? std::atoi(argv[2]) : 5000;
        uint32_t* fh2 = nullptr;
        hipHostMalloc((void**) &fh2, 16, hipHostMallocPortable | hipHostMallocMapped);
        *fh = 0; *fh2 = 0;
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        const double t0 = ms_now();
        hipSetDevice(0); hipLaunchKernelGGL(e8_a_k, dim3(1), dim3(1), 0, s0, (volatile uint32_t*) fh, fh2, R);
        hipSetDevice(1); hipLaunchKernelGGL(e8_b_k, dim3(1), dim3(1), 0, s1, (volatile uint32_t*) fh2, fh, R);
        hipError_t y0 = hipStreamSynchronize(s0);
        hipError_t y1 = hipStreamSynchronize(s1);
        std::printf("E8 round trip R=%d: %s/%s -> %.2f us/round\n", R, err(y0), err(y1), (ms_now() - t0) * 1000.0 / R);
        return 0;
    }
    if (!std::strcmp(which, "E9")) {
        const int n = 5120, R = argc > 2 ? std::atoi(argv[2]) : 500;
        float* data = nullptr; uint32_t* flag = nullptr; uint32_t* ack = nullptr; float* sink = nullptr; unsigned long long* polls = nullptr;
        hipHostMalloc((void**) &data, (size_t) n * 4, hipHostMallocPortable | hipHostMallocMapped);
        hipHostMalloc((void**) &flag, 16, hipHostMallocPortable | hipHostMallocMapped);
        hipHostMalloc((void**) &ack, 16, hipHostMallocPortable | hipHostMallocMapped);
        *flag = 0; *ack = 0;
        float* dataD = nullptr; uint32_t* flagD = nullptr; uint32_t* ackD = nullptr;
        hipHostGetDevicePointer((void**) &dataD, data, 0);
        hipHostGetDevicePointer((void**) &flagD, flag, 0);
        hipHostGetDevicePointer((void**) &ackD, ack, 0);
        std::printf("E9 ptrs host=%p dev=%p\n", (void*) data, (void*) dataD);
        data = dataD; flag = flagD; ack = ackD;
        hipSetDevice(1); hipMalloc(&sink, 4096); hipMalloc(&polls, 8); hipMemset(polls, 0, 8);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        const double t0 = ms_now();
        hipSetDevice(0); hipLaunchKernelGGL(e9_prod_k, dim3(4), dim3(256), 0, s0, data, (volatile uint32_t*) flag, ack, n, R);
        hipSetDevice(1); hipLaunchKernelGGL(e9_cons_k, dim3(4), dim3(256), 0, s1, data, flag, ack, n, R, sink, polls);
        hipError_t y1 = hipStreamSynchronize(s1);
        const double dt = ms_now() - t0;
        hipError_t y0 = hipStreamSynchronize(s0);
        unsigned long long p = 0; hipMemcpy(&p, polls, 8, hipMemcpyDeviceToHost);
        std::printf("E9 20 KB through host R=%d: %s/%s -> %.2f us/round (%llu polls)\n", R, err(y0), err(y1), dt * 1000.0 / R, p);
        return 0;
    }
    if (!std::strcmp(which, "V2")) {
        const uint32_t val = 0x9999;
        unsigned long long* res = nullptr;
        *fh = 0;
        hipSetDevice(0); hipMalloc(&res, 16); hipMemset(res, 0, 16);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        const double t0 = ms_now();
        hipSetDevice(0); hipLaunchKernelGGL(poll_nt_k, dim3(1), dim3(1), 0, s0, fh, val, 200000000ull, res);
        usleep(30000);
        hipSetDevice(1); hipLaunchKernelGGL(store_peer_k, dim3(1), dim3(1), 0, s1, fh, val);
        hipStreamSynchronize(s1);
        hipError_t y0 = hipStreamSynchronize(s0);
        unsigned long long h[2] = {0, 0}; hipMemcpy(h, res, 16, hipMemcpyDeviceToHost);
        std::printf("V2 dev1->host->dev0: sync=%s seen=%llu/%llu -> wall %.2f ms\n", err(y0), h[0], h[1], ms_now() - t0);
        return 0;
    }
    std::printf("usage: p2p_sync3 E1|C1|E2|E3|E4|E5|E6|E7 R|E8 R|E9 R|V2\n");
    return 0;
}
