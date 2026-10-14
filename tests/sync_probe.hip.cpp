// Device-side sync probe: hipStreamWaitValue32/WriteValue32 + peer memory — the handshake the two-card
// split needs.  One test per process (pass T2 / T3 / ...; no arg runs all) so a hang costs one test.
// Build:  /opt/rocm/bin/hipcc -O2 --offload-arch=gfx1100 tests/sync_probe.hip.cpp -o build-hip/sync_probe
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

__global__ void spin_cycles(long long c) { long long t0 = clock64(); while (clock64() - t0 < c) {} }
__global__ void set_flag_peer(uint32_t* p, uint32_t v) { __threadfence_system(); *(volatile uint32_t*) p = v; }
__global__ void sum_f3(const float3* p, int n3, float* out) {
    float a = 0, b = 0, c = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n3; i += gridDim.x * blockDim.x) {
        float3 v = p[i]; a += v.x; b += v.y; c += v.z;
    }
    if (a + b + c == 12345.0f) *out = a;                 // never true: keeps the reads live
}

// T8: flag ping-pong with system-scope atomics (one thread per side, R hops)
__global__ void hop_master(uint32_t* myflag, uint32_t* peerflag, int R, unsigned long long* status,
                           unsigned long long* cycles) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long t0 = clock64(), spins = 0;
    for (int i = 1; i <= R; ++i) {
        __hip_atomic_store(myflag, (uint32_t) i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        spins = 0;
        while (__hip_atomic_load(peerflag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != (uint32_t) i) {
            if (++spins > 400000000ull) { *status = 0x8000ull + (unsigned) i; return; }
        }
    }
    *status = 0;
    *cycles = clock64() - t0;
}
__global__ void hop_slave(uint32_t* myflag, uint32_t* peerflag, int R, unsigned long long* status) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long spins;
    for (int i = 1; i <= R; ++i) {
        spins = 0;
        while (__hip_atomic_load(peerflag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != (uint32_t) i) {
            if (++spins > 400000000ull) { *status = 0x8000ull + (unsigned) i; return; }
        }
        __hip_atomic_store(myflag, (uint32_t) i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    *status = 0;
}

// T10: a realistic per-layer reduction.  256 threads, nfloats partial; every iteration: write own partial,
// signal, wait for the peer, read the peer's partial (over PCIe) and add it.
__global__ void reduce_worker(uint32_t* myflag, uint32_t* peerflag, float* mybuf, const float* peerbuf,
                              int nfloats, int R, float* sink, unsigned long long* status,
                              unsigned long long* cycles) {
    const int nt = blockDim.x;
    float acc = 0.f;
    unsigned long long t0 = clock64();
    for (int i = 1; i <= R; ++i) {
        float* mb = mybuf + (size_t) i * nfloats;               // one slot per round: no overwrite while the peer reads
        const float* pb = peerbuf + (size_t) i * nfloats;
        for (int j = threadIdx.x; j < nfloats; j += nt) mb[j] = (float) (i * 2 + j);
        __syncthreads();
        if (threadIdx.x == 0) {
            __hip_atomic_store(myflag, (uint32_t) i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            unsigned long long spins = 0;
            while (__hip_atomic_load(peerflag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != (uint32_t) i) {
                if (++spins > 400000000ull) { *status = 0x8000ull + (unsigned) i; return; }
            }
        }
        __syncthreads();
        for (int j = threadIdx.x; j < nfloats; j += nt) acc += pb[j];
    }
    if (threadIdx.x == 0) { *status = 0; *cycles = clock64() - t0; }
    sink[threadIdx.x] = acc;
}

// V-test: does a store on one card become visible to a poller on the other, and which flavor does it?
__global__ void store_volatile_k(uint32_t* f, uint32_t v, long long delay) {
    { long long t0 = clock64(); while (clock64() - t0 < delay) {} }
    __threadfence_system();
    *(volatile uint32_t*) f = v;
}
__global__ void store_atomic_k(uint32_t* f, uint32_t v, long long delay) {
    { long long t0 = clock64(); while (clock64() - t0 < delay) {} }
    __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void poll_volatile_k(volatile uint32_t* f, uint32_t want, unsigned long long budget,
                                unsigned long long* res) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long t0 = clock64();
    while (*f != want) {
        if (clock64() - t0 > budget) { *res = 0; return; }
    }
    *res = clock64() - t0;
}
__global__ void poll_atomic_k(uint32_t* f, uint32_t want, unsigned long long budget, unsigned long long* res) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long t0 = clock64();
    while (__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != want) {
        if (clock64() - t0 > budget) { *res = 0; return; }
    }
    *res = clock64() - t0;
}

static double ms_now() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static const char* err(hipError_t e) { return hipGetErrorName(e); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* which = argc > 1 ? argv[1] : "all";
    auto want = [&](const char* t) { return !std::strcmp(which, "all") || !std::strcmp(which, t); };

    int n = 0;
    hipGetDeviceCount(&n);
    if (n < 2) { std::printf("devices: %d\n", n); return 0; }
    for (int i = 0; i < n; ++i) {
        int a = -1;
        hipDeviceGetAttribute(&a, hipDeviceAttributeCanUseStreamWaitValue, i);
        std::printf("dev%d CanUseStreamWaitValue: %d\n", i, a);
    }
    hipSetDevice(1); hipError_t e1 = hipDeviceEnablePeerAccess(0, 0);
    hipSetDevice(0); hipError_t e0 = hipDeviceEnablePeerAccess(1, 0);
    int c10 = 0, c01 = 0;
    hipDeviceCanAccessPeer(&c10, 1, 0);
    hipDeviceCanAccessPeer(&c01, 0, 1);
    std::printf("peer 1->0 %d 0->1 %d, enable %s / %s\n", c10, c01, err(e1), err(e0));

    hipStream_t s0 = nullptr, s1 = nullptr;
    hipSetDevice(0); hipStreamCreateWithFlags(&s0, hipStreamNonBlocking);
    hipSetDevice(1); hipStreamCreateWithFlags(&s1, hipStreamNonBlocking);
    uint32_t *f0 = nullptr, *f1 = nullptr, *fh = nullptr;
    hipSetDevice(0); hipMalloc(&f0, 16); hipMemset(f0, 0, 16);
    hipSetDevice(1); hipMalloc(&f1, 16); hipMemset(f1, 0, 16);
    hipHostMalloc((void**) &fh, 16, hipHostMallocPortable | hipHostMallocMapped);
    *fh = 0;
    hipStreamSynchronize(s0); hipStreamSynchronize(s1);
    hipError_t ce = hipGetLastError();
    std::printf("setup: %s\n", err(ce));

    const long long delay_cycles = 100000000;            // ~70 ms on this clock

    // T2: same-device sanity — write on s1, wait on another dev1 stream
    if (want("T2")) {
        std::printf("RUN T2 same-device wait/write\n");
        hipStream_t s1b; hipSetDevice(1); hipStreamCreateWithFlags(&s1b, hipStreamNonBlocking);
        hipLaunchKernelGGL(spin_cycles, dim3(1), dim3(1), 0, s1, delay_cycles);
        hipError_t w = hipStreamWriteValue32(s1, f1, 77, 0);
        hipError_t x = hipStreamWaitValue32(s1b, f1, 77, hipStreamWaitValueEq, 0xffffffffu);
        double t = ms_now(); hipError_t sy = hipStreamSynchronize(s1b);
        std::printf("T2 write=%s wait=%s sync=%s -> %.0f ms (expect ~70)\n", err(w), err(x), err(sy), ms_now() - t);
    }
    // T9: prime design — s0 writes into dev1's flag, s1 waits on its LOCAL flag
    if (want("T9")) {
        std::printf("RUN T9 peer-write32 -> local wait\n");
        hipSetDevice(0);
        hipError_t w = hipStreamWriteValue32(s0, f1, 91, 0);
        hipError_t x = hipStreamWaitValue32(s1, f1, 91, hipStreamWaitValueEq, 0xffffffffu);
        double t = ms_now(); hipError_t sy = hipStreamSynchronize(s1);
        std::printf("T9 write=%s wait=%s sync=%s -> %.1f ms (expect ~0)\n", err(w), err(x), err(sy), ms_now() - t);
    }
    // T6: ping-pong, 2000 round trips with the T9 mechanism (peer write32 + local wait)
    if (want("T6")) {
        std::printf("RUN T6 ping-pong (peer write32 + local wait)\n");
        const int R = 2000;
        double t = ms_now();
        for (int i = 1; i <= R; ++i) {
            hipSetDevice(0); hipStreamWriteValue32(s0, f1, 1000 + i, 0);
            hipStreamWaitValue32(s1, f1, 1000 + i, hipStreamWaitValueEq, 0xffffffffu);
            hipStreamWriteValue32(s1, f0, 2000 + i, 0);
            hipStreamWaitValue32(s0, f0, 2000 + i, hipStreamWaitValueEq, 0xffffffffu);
        }
        hipError_t y0 = hipStreamSynchronize(s0), y1 = hipStreamSynchronize(s1);
        std::printf("T6 %s/%s -> %.1f us/round trip\n", err(y0), err(y1), (ms_now() - t) * 1000.0 / R);
    }
    // T3: wait on a flag in the OTHER device's memory (peer wait), written by write32
    if (want("T3")) {
        std::printf("RUN T3 cross wait (s1 waits on dev0 flag), write32 writer\n");
        hipLaunchKernelGGL(spin_cycles, dim3(1), dim3(1), 0, s0, delay_cycles);
        hipError_t w = hipStreamWriteValue32(s0, f0, 55, 0);
        hipError_t x = hipStreamWaitValue32(s1, f0, 55, hipStreamWaitValueEq, 0xffffffffu);
        double t = ms_now(); hipError_t sy = hipStreamSynchronize(s1);
        std::printf("T3 write=%s wait=%s sync=%s -> %.0f ms (expect ~70)\n", err(w), err(x), err(sy), ms_now() - t);
    }
    // T4: same as T3 but the flag is written by a kernel store on dev0
    if (want("T4")) {
        std::printf("RUN T4 cross wait (s1 waits on dev0 flag), kernel-store writer\n");
        hipLaunchKernelGGL(spin_cycles, dim3(1), dim3(1), 0, s0, delay_cycles);
        hipLaunchKernelGGL(set_flag_peer, dim3(1), dim3(1), 0, s0, f0, 55u);
        hipError_t w = hipGetLastError();
        hipError_t x = hipStreamWaitValue32(s1, f0, 55, hipStreamWaitValueEq, 0xffffffffu);
        double t = ms_now(); hipError_t sy = hipStreamSynchronize(s1);
        std::printf("T4 kernel=%s wait=%s sync=%s -> %.0f ms (expect ~70)\n", err(w), err(x), err(sy), ms_now() - t);
    }
    // T5: flag in pinned host memory, written by a kernel on dev0, waited by s1
    if (want("T5")) {
        std::printf("RUN T5 cross wait on host-pinned flag\n");
        hipLaunchKernelGGL(spin_cycles, dim3(1), dim3(1), 0, s0, delay_cycles);
        hipLaunchKernelGGL(set_flag_peer, dim3(1), dim3(1), 0, s0, fh, 55u);
        hipError_t w = hipGetLastError();
        hipError_t x = hipStreamWaitValue32(s1, fh, 55, hipStreamWaitValueEq, 0xffffffffu);
        double t = ms_now(); hipError_t sy = hipStreamSynchronize(s1);
        std::printf("T5 kernel=%s wait=%s sync=%s -> %.0f ms (expect ~70)\n", err(w), err(x), err(sy), ms_now() - t);
    }
    // T7: peer reads from a kernel (card1 reads card0's buffer); local baseline; 20 KB and 2 MB
    if (want("T7")) {
        std::printf("RUN T7 peer kernel reads\n");
        const int n20 = 5120 / 3 + 1;                        // ~20 KB
        const int n2m = (2 << 20) / 12;
        float3 *b0 = nullptr, *b1 = nullptr, *b2m = nullptr;
        float* out = nullptr;
        hipSetDevice(0); hipMalloc(&b0, (size_t) n20 * 12); hipMemset(b0, 0x3f, (size_t) n20 * 12);
        hipMalloc(&b2m, (size_t) n2m * 12); hipMemset(b2m, 0x3f, (size_t) n2m * 12);
        hipSetDevice(1); hipMalloc(&b1, (size_t) n20 * 12); hipMemset(b1, 0x3f, (size_t) n20 * 12); hipMalloc(&out, 4);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        auto bench = [&](const char* name, const float3* p, int cnt, int r) {
            double t = ms_now();
            for (int i = 0; i < r; ++i) hipLaunchKernelGGL(sum_f3, dim3(8), dim3(256), 0, s1, p, cnt, out);
            hipError_t sy = hipStreamSynchronize(s1);
            double dt = ms_now() - t;
            std::printf("T7 %-26s x%-4d: %s -> %.2f us/launch (%.2f GB/s)\n", name, r, err(sy),
                        dt * 1000.0 / r, r * (double) cnt * 12 / dt / 1e6);
        };
        bench("local 20 KB", b1, n20, 500);
        bench("peer 20 KB (card1<-card0)", b0, n20, 500);
        bench("peer 2 MB (card1<-card0)", b2m, n2m, 50);
    }
    // T8: flag ping-pong, system-scope atomics
    if (want("T8")) {
        std::printf("RUN T8 flag ping-pong (system atomics)\n");
        const int R = 20000;
        unsigned long long *st0 = nullptr, *st1 = nullptr, *cyc = nullptr;
        hipSetDevice(0); hipMalloc(&st0, 8); hipMalloc(&cyc, 8);
        hipSetDevice(1); hipMalloc(&st1, 8);
        hipMemset(f0, 0, 16); hipMemset(f1, 0, 16);
        hipMemset(st0, 0, 8); hipMemset(st1, 0, 8); hipMemset(cyc, 0, 8);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        hipSetDevice(0); hipLaunchKernelGGL(hop_master, dim3(1), dim3(1), 0, s0, f0, f1, R, st0, cyc);
        hipSetDevice(1); hipLaunchKernelGGL(hop_slave, dim3(1), dim3(1), 0, s1, f1, f0, R, st1);
        double t = ms_now();
        hipError_t y0 = hipStreamSynchronize(s0), y1 = hipStreamSynchronize(s1);
        unsigned long long h0 = 0, h1 = 0, hc = 0;
        hipMemcpy(&h0, st0, 8, hipMemcpyDeviceToHost);
        hipMemcpy(&h1, st1, 8, hipMemcpyDeviceToHost);
        hipMemcpy(&hc, cyc, 8, hipMemcpyDeviceToHost);
        std::printf("T8 %s/%s status %llu/%llu -> %.2f us/hop, %.0f cycles/hop\n", err(y0), err(y1), h0, h1,
                    (ms_now() - t) * 1000.0 / R, (double) hc / R);
    }
    // T10: the per-layer reduction shape (5120 floats), one slot per round
    if (want("T10")) {
        const int F = 5120, R = 1000;
        std::printf("RUN T10 reduction, %d floats, %d rounds\n", F, R);
        float *mA = nullptr, *mB = nullptr, *sinkA = nullptr, *sinkB = nullptr;
        unsigned long long *st0 = nullptr, *st1 = nullptr, *cy0 = nullptr, *cy1 = nullptr;
        hipSetDevice(0);
        hipMalloc(&mA, (size_t) R * F * 4); hipMalloc(&sinkA, 1024); hipMalloc(&st0, 8); hipMalloc(&cy0, 8);
        hipSetDevice(1);
        hipMalloc(&mB, (size_t) R * F * 4); hipMalloc(&sinkB, 1024); hipMalloc(&st1, 8); hipMalloc(&cy1, 8);
        hipMemset(f0, 0, 16); hipMemset(f1, 0, 16);
        hipMemset(st0, 0, 8); hipMemset(st1, 0, 8); hipMemset(cy0, 0, 8); hipMemset(cy1, 0, 8);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        hipSetDevice(0); hipLaunchKernelGGL(reduce_worker, dim3(1), dim3(256), 0, s0, f0, f1, mA, mB, F, R, sinkA, st0, cy0);
        hipSetDevice(1); hipLaunchKernelGGL(reduce_worker, dim3(1), dim3(256), 0, s1, f1, f0, mB, mA, F, R, sinkB, st1, cy1);
        double t = ms_now();
        hipError_t y0 = hipStreamSynchronize(s0), y1 = hipStreamSynchronize(s1);
        unsigned long long h0 = 0, h1 = 0, c0 = 0, c1 = 0;
        hipMemcpy(&h0, st0, 8, hipMemcpyDeviceToHost);
        hipMemcpy(&h1, st1, 8, hipMemcpyDeviceToHost);
        hipMemcpy(&c0, cy0, 8, hipMemcpyDeviceToHost);
        hipMemcpy(&c1, cy1, 8, hipMemcpyDeviceToHost);
        std::printf("T10 %s/%s status %llu/%llu -> %.2f us/round wall, cycles/round %llu / %llu\n", err(y0), err(y1),
                    h0, h1, (ms_now() - t) * 1000.0 / R, c0 / R, c1 / R);
    }
    // V: visibility matrix.  argv: V <store_mode 0=volatile 1=atomic 2=streamwrite32> <load_mode 0=volatile 1=atomic> <writer_dev>
    if (want("V")) {
        const int store_mode = argc > 2 ? std::atoi(argv[2]) : 0;
        const int load_mode = argc > 3 ? std::atoi(argv[3]) : 0;
        const int wdev = argc > 4 ? std::atoi(argv[4]) : 0;
        const int pdev = wdev ^ 1;
        uint32_t* wflag = wdev == 0 ? f0 : f1;
        uint32_t* pflag = pdev == 0 ? f0 : f1;
        hipStream_t ws = nullptr, ps = nullptr;
        hipSetDevice(wdev); hipStreamCreateWithFlags(&ws, hipStreamNonBlocking);
        hipSetDevice(pdev); hipStreamCreateWithFlags(&ps, hipStreamNonBlocking);
        unsigned long long* res = nullptr;
        hipSetDevice(pdev); hipMalloc(&res, 8); hipMemset(res, 0, 8);
        hipSetDevice(wdev); hipMemset(wflag, 0, 16); hipSetDevice(pdev); hipMemset(pflag, 0, 16);
        hipStreamSynchronize(ws); hipStreamSynchronize(ps);
        const long long delay = 100000000;                       // ~70 ms after launch
        const uint32_t val = 0x5151;
        hipSetDevice(wdev);
        if (store_mode == 0) hipLaunchKernelGGL(store_volatile_k, dim3(1), dim3(1), 0, ws, wflag, val, delay);
        else if (store_mode == 1) hipLaunchKernelGGL(store_atomic_k, dim3(1), dim3(1), 0, ws, wflag, val, delay);
        else { hipLaunchKernelGGL(spin_cycles, dim3(1), dim3(1), 0, ws, delay);
               hipStreamWriteValue32(ws, wflag, val, 0); }
        hipSetDevice(pdev);
        if (load_mode == 0) hipLaunchKernelGGL(poll_volatile_k, dim3(1), dim3(1), 0, ps, pflag, val, 2000000000ull, res);
        else                hipLaunchKernelGGL(poll_atomic_k, dim3(1), dim3(1), 0, ps, pflag, val, 2000000000ull, res);
        double t = ms_now();
        hipError_t yp = hipStreamSynchronize(ps);
        hipError_t yw = hipStreamSynchronize(ws);
        unsigned long long h = 0;
        hipMemcpy(&h, res, 8, hipMemcpyDeviceToHost);
        hipError_t he = hipGetLastError();
        std::printf("V store=%d load=%d writer=dev%d: poller=%s writer=%s last=%s -> seen after %llu cycles%s\n",
                    store_mode, load_mode, wdev, err(yp), err(yw), err(he), h, h ? "" : " (NEVER within 1 s)");
    }
    std::printf("done\n");
    return 0;
}
