// Cross-device signal probe, take 2: host-timed (no clock64() — it is dead on this gfx11 stack and every
// earlier "hang" was a delay loop that never ended).  Tests are one process each:
//   C0                      clock sanity
//   Q <store> <load> <wdev> store: 0 volatile 1 atomic 2 write32 3 peer-store;  load: 0 volatile 1 atomic
//   P <store> <load> <R>    ping-pong round trips (host-timed)
// Build: /opt/rocm/bin/hipcc -O2 --offload-arch=gfx1100 tests/sync_probe2.hip.cpp -o build-hip/sync_probe2
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

__global__ void clock_check(unsigned long long* out) {
    if (blockIdx.x || threadIdx.x) return;
    long long c0 = clock64();
    unsigned long long w0 = wall_clock64();
    long long s = 0;
    for (int i = 0; i < 10000000; ++i) s += i;
    out[0] = (unsigned long long) (clock64() - c0);
    out[1] = wall_clock64() - w0;
    out[2] = (unsigned long long) s;
}
__global__ void store_vol_k(uint32_t* f, uint32_t v) { __threadfence_system(); *(volatile uint32_t*) f = v; }
__global__ void store_atom_k(uint32_t* f, uint32_t v) { __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
__global__ void poll_vol_k(volatile uint32_t* f, uint32_t want) { if (blockIdx.x || threadIdx.x) return; while (*f != want) {} }
__global__ void poll_atom_k(uint32_t* f, uint32_t want) {
    if (blockIdx.x || threadIdx.x) return;
    while (__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != want) {}
}
__global__ void poll_bounded_k(uint32_t* f, uint32_t want, unsigned long long budget, unsigned long long* res) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long polls = 0;
    while (__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != want) {
        if (++polls >= budget) { res[0] = 0; res[1] = polls; return; }
    }
    res[0] = 1; res[1] = polls;
}
__global__ void hop_master(uint32_t* myflag, uint32_t* peerflag, int R, int lmode) {
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 1; i <= R; ++i) {
        __hip_atomic_store(myflag, (uint32_t) i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (lmode == 1) { while (__hip_atomic_load(peerflag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != (uint32_t) i) {} }
        else            { while (*(volatile uint32_t*) peerflag != (uint32_t) i) {} }
    }
}
__global__ void hop_slave(uint32_t* myflag, uint32_t* peerflag, int R, int lmode) {
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 1; i <= R; ++i) {
        if (lmode == 1) { while (__hip_atomic_load(peerflag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != (uint32_t) i) {} }
        else            { while (*(volatile uint32_t*) peerflag != (uint32_t) i) {} }
        __hip_atomic_store(myflag, (uint32_t) i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

static double ms_now() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static const char* err(hipError_t e) { return hipGetErrorName(e); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* which = argc > 1 ? argv[1] : "help";
    int ndev = 0;
    hipGetDeviceCount(&ndev);
    if (ndev < 2) { std::printf("devices: %d\n", ndev); return 0; }
    hipSetDevice(1); hipError_t e1 = hipDeviceEnablePeerAccess(0, 0);
    hipSetDevice(0); hipError_t e0 = hipDeviceEnablePeerAccess(1, 0);
    uint32_t *f0 = nullptr, *f1 = nullptr;
    hipSetDevice(0); hipMalloc(&f0, 16); hipMemset(f0, 0, 16);
    hipSetDevice(1); hipMalloc(&f1, 16); hipMemset(f1, 0, 16);
    hipStream_t s0 = nullptr, s1 = nullptr;
    hipSetDevice(0); hipStreamCreateWithFlags(&s0, hipStreamNonBlocking);
    hipSetDevice(1); hipStreamCreateWithFlags(&s1, hipStreamNonBlocking);
    std::printf("setup peer %s/%s\n", err(e1), err(e0));

    if (!std::strcmp(which, "C0")) {
        unsigned long long* out = nullptr;
        hipSetDevice(0); hipMalloc(&out, 24); hipMemset(out, 0, 24);
        hipLaunchKernelGGL(clock_check, dim3(1), dim3(1), 0, s0, out);
        hipStreamSynchronize(s0);
        unsigned long long h[3] = {0, 0, 0};
        hipMemcpy(h, out, 24, hipMemcpyDeviceToHost);
        std::printf("C0 clock64 delta %llu, wall_clock64 delta %llu (1e7-iteration loop)\n", h[0], h[1]);
        return 0;
    }
    if (!std::strcmp(which, "Q") || !std::strcmp(which, "L0")) {
        const bool local = !std::strcmp(which, "L0");
        const int smode = argc > 2 ? std::atoi(argv[2]) : 0;
        const int lmode = argc > 3 ? std::atoi(argv[3]) : 0;
        const int wdev = local ? 0 : (argc > 4 ? std::atoi(argv[4]) : 0);
        const int pdev = local ? 0 : (wdev ^ 1);
        hipStream_t ws = wdev == 0 ? s0 : s1, ps = local ? nullptr : (pdev == 0 ? s0 : s1);
        uint32_t* fw = wdev == 0 ? f0 : f1;                 // writer's memory
        uint32_t* fp = pdev == 0 ? f0 : f1;                 // poller's memory
        uint32_t* target = smode == 3 ? fp : fw;            // where the store lands (3 = straight into the peer)
        uint32_t* pflag = smode == 3 ? fp : fw;             // what the poller polls (remote unless smode 3)
        const uint32_t val = 0x4c4c;
        hipSetDevice(pdev); hipMemset(fp, 0, 16);
        hipSetDevice(wdev); hipMemset(fw, 0, 16);
        hipStreamSynchronize(ws); hipStreamSynchronize(ps);
        hipSetDevice(pdev);
        std::printf("Q: launching poller on dev%d (flag %s)\n", pdev, pflag == f0 ? "f0/dev0" : "f1/dev1");
        if (lmode == 0) hipLaunchKernelGGL(poll_vol_k, dim3(1), dim3(1), 0, ps, pflag, val);
        else            hipLaunchKernelGGL(poll_atom_k, dim3(1), dim3(1), 0, ps, pflag, val);
        hipError_t lp = hipGetLastError();
        std::printf("Q: poller launch err=%s\n", err(lp));
        usleep(50000);
        const double t0 = ms_now();
        hipSetDevice(wdev);
        std::printf("Q: launching store on dev%d\n", wdev);
        hipError_t ls = hipSuccess;
        if (smode == 0)      { hipLaunchKernelGGL(store_vol_k, dim3(1), dim3(1), 0, ws, target, val); ls = hipGetLastError(); }
        else if (smode == 1) { hipLaunchKernelGGL(store_atom_k, dim3(1), dim3(1), 0, ws, target, val); ls = hipGetLastError(); }
        else if (smode == 2) { ls = hipStreamWriteValue32(ws, target, val, 0); }
        else                 { hipLaunchKernelGGL(store_vol_k, dim3(1), dim3(1), 0, ws, target, val); ls = hipGetLastError(); }
        std::printf("Q: store launch err=%s; syncing writer stream\n", err(ls));
        hipError_t yw = hipStreamSynchronize(ws);            // the store itself is quick; does it even complete?
        std::printf("Q: writer sync err=%s after %.2f ms\n", err(yw), ms_now() - t0);
        std::printf("Q: syncing poller stream\n");
        hipError_t yp = hipStreamSynchronize(ps);            // blocks until the poller sees the flag
        const double dt = ms_now() - t0;
        std::printf("Q store=%d load=%d w=dev%d %s: sync=%s/%s -> seen %.2f ms after the store\n",
                    smode, lmode, wdev, local ? "(local)" : "", err(yp), err(yw), dt);
        return 0;
    }
    if (!std::strcmp(which, "P")) {
        const int lmode = argc > 2 ? std::atoi(argv[2]) : 1;
        const int R = argc > 3 ? std::atoi(argv[3]) : 2000;
        hipSetDevice(0); hipMemset(f0, 0, 16);
        hipSetDevice(1); hipMemset(f1, 0, 16);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        const double t0 = ms_now();
        hipSetDevice(0); hipLaunchKernelGGL(hop_master, dim3(1), dim3(1), 0, s0, f0, f1, R, lmode);
        hipSetDevice(1); hipLaunchKernelGGL(hop_slave, dim3(1), dim3(1), 0, s1, f1, f0, R, lmode);
        hipError_t y0 = hipStreamSynchronize(s0);
        hipError_t y1 = hipStreamSynchronize(s1);
        const double dt = ms_now() - t0;
        std::printf("P load=%d R=%d: %s/%s -> %.2f us/round trip\n", lmode, R, err(y0), err(y1), dt * 1000.0 / R);
        return 0;
    }
    if (!std::strcmp(which, "D1")) {
        // doorbell: the DMA engine copies a 4-byte value into the POLLER's local memory; the poller polls
        // local memory only (the remote-spin route dies on stale caches).
        const int mode = argc > 2 ? std::atoi(argv[2]) : 0;      // 0: copy on the producer's stream, 1: on the consumer's
        uint32_t* src = nullptr;
        hipSetDevice(0); hipMalloc(&src, 4); hipMemset(src, 0, 4);
        hipSetDevice(1); hipMemset(f1, 0, 16);
        uint32_t val = 0x7777;
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        hipSetDevice(1); hipLaunchKernelGGL(poll_atom_k, dim3(1), dim3(1), 0, s1, f1, val);
        std::printf("D1: poller on dev1 launched err=%s\n", err(hipGetLastError()));
        usleep(50000);
        const double t0 = ms_now();
        hipSetDevice(0); hipMemcpyAsync(src, &val, 4, hipMemcpyHostToDevice, s0);   // the value to ring with
        hipError_t e = mode == 0 ? hipMemcpyPeerAsync(f1, 1, src, 0, 4, s0)
                                 : hipMemcpyPeerAsync(f1, 1, src, 0, 4, s1);
        std::printf("D1: peer copy enqueued err=%s; syncing consumer\n", err(e));
        hipError_t y1 = hipStreamSynchronize(s1);
        const double dt = ms_now() - t0;
        hipStreamSynchronize(s0);
        std::printf("D1 mode %d: sync=%s -> poller saw the doorbell %.2f ms after the copy was enqueued\n", mode, err(y1), dt);
        return 0;
    }
    if (!std::strcmp(which, "D3")) {
        // the full double-doorbell exchange, both directions at once, R rounds.
        //  per round: each card writes 20 KB of its own data locally, rings the peer's doorbell by DMA, the
        //  peer waits on its local doorbell and reads its LOCAL copy (which the DMA wrote).
        const int R = argc > 2 ? std::atoi(argv[2]) : 200;
        const int F = 5120;
        float *dA = nullptr, *dB = nullptr;
        uint32_t *sA = nullptr, *sB = nullptr;
        hipSetDevice(0); hipMalloc(&dA, (size_t) R * F * 4); hipMalloc(&sA, (size_t) R * 4);
        hipMemset(dA, 0, (size_t) R * F * 4); hipMemset(sA, 0, (size_t) R * 4);
        hipSetDevice(1); hipMalloc(&dB, (size_t) R * F * 4); hipMalloc(&sB, (size_t) R * 4);
        hipMemset(dB, 0, (size_t) R * F * 4); hipMemset(sB, 0, (size_t) R * 4);
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        const double t0 = ms_now();
        // producer work is faked with a tiny kernel; the DMA rings follow it on the same stream
        for (int i = 1; i <= R; ++i) {
            hipSetDevice(0);
            hipMemcpyAsync(dA + (size_t) (i - 1) * F, sA + i - 1, F * 4, hipMemcpyHostToDevice, s0);  // fake partial
            hipSetDevice(0); hipMemcpyPeerAsync(sB + i - 1, 1, dA + (size_t) (i - 1) * F, 0, 4, s0);
            hipSetDevice(1);
            hipMemcpyAsync(dB + (size_t) (i - 1) * F, sB + i - 1, F * 4, hipMemcpyHostToDevice, s1);
            hipSetDevice(1); hipMemcpyPeerAsync(sA + i - 1, 0, dB + (size_t) (i - 1) * F, 1, 4, s1);
        }
        hipError_t y0 = hipStreamSynchronize(s0), y1 = hipStreamSynchronize(s1);
        const double dt = ms_now() - t0;
        std::printf("D3: %s/%s -> %.1f us/round of queued doorbell pairs (R=%d)\n", err(y0), err(y1), dt * 1000.0 / R, R);
        return 0;
    }
    if (!std::strcmp(which, "D4")) {
        // bounded poller with a result readout: did the DMA doorbell land, and did the local poller see it?
        const int mode = argc > 2 ? std::atoi(argv[2]) : 0;
        uint32_t* src = nullptr; unsigned long long* res = nullptr;
        hipSetDevice(0); hipMalloc(&src, 4); hipMemset(src, 0, 4);
        hipSetDevice(1); hipMemset(f1, 0, 16); hipMalloc(&res, 16); hipMemset(res, 0, 16);
        const uint32_t val = 0x7777;
        hipStreamSynchronize(s0); hipStreamSynchronize(s1);
        hipSetDevice(1);
        hipLaunchKernelGGL(poll_bounded_k, dim3(1), dim3(1), 0, s1, f1, val, 50000000ull, res);
        std::printf("D4: bounded poller launched err=%s\n", err(hipGetLastError()));
        usleep(30000);
        hipSetDevice(0); hipMemcpyAsync(src, &val, 4, hipMemcpyHostToDevice, s0);
        hipError_t e = mode == 0 ? hipMemcpyPeerAsync(f1, 1, src, 0, 4, s0)
                                 : hipMemcpyPeerAsync(f1, 1, src, 0, 4, s1);
        std::printf("D4: peer doorbell enqueued err=%s\n", err(e));
        hipStreamSynchronize(s0);
        usleep(200000);
        hipError_t y1 = hipStreamSynchronize(s1);
        unsigned long long h[2] = {0, 0}; uint32_t hv = 0;
        hipMemcpy(h, res, 16, hipMemcpyDeviceToHost);
        hipMemcpy(&hv, f1, 4, hipMemcpyDeviceToHost);
        std::printf("D4 mode %d: sync=%s seen=%llu after %llu polls; value now in f1 = 0x%x (want 0x%x)\n",
                    mode, err(y1), h[0], h[1], hv, val);
        return 0;
    }
    if (!std::strcmp(which, "D5")) {
        // does hipMemcpyPeerAsync actually move the bytes?
        const int F = 1024;
        float *a = nullptr, *b = nullptr;
        hipSetDevice(0); hipMalloc(&a, F * 4);
        hipSetDevice(1); hipMalloc(&b, F * 4); hipMemset(b, 0, F * 4);
        float* host = (float*) std::malloc(F * 4);
        for (int i = 0; i < F; ++i) host[i] = (float) (i * 3 + 1);
        hipSetDevice(0); hipMemcpy(a, host, F * 4, hipMemcpyHostToDevice);
        hipError_t e = hipMemcpyPeerAsync(b, 1, a, 0, F * 4, s0);
        hipError_t es = hipStreamSynchronize(s0);
        std::memset(host, 0, F * 4);
        hipError_t ec = hipMemcpy(host, b, F * 4, hipMemcpyDeviceToHost);
        int bad = 0; for (int i = 0; i < F; ++i) if (host[i] != (float) (i * 3 + 1)) ++bad;
        std::printf("D5 peer copy (enqueued on s0): enqueue=%s sync=%s readback=%s bad=%d/%d\n", err(e), err(es), err(ec), bad, F);
        hipSetDevice(1); hipMemset(b, 0, F * 4);
        e = hipMemcpyPeerAsync(b, 1, a, 0, F * 4, s1);
        es = hipStreamSynchronize(s1);
        std::memset(host, 0, F * 4);
        ec = hipMemcpy(host, b, F * 4, hipMemcpyDeviceToHost);
        bad = 0; for (int i = 0; i < F; ++i) if (host[i] != (float) (i * 3 + 1)) ++bad;
        std::printf("D5 peer copy (enqueued on s1): enqueue=%s sync=%s readback=%s bad=%d/%d\n", err(e), err(es), err(ec), bad, F);
        return 0;
    }
    std::printf("usage: sync_probe2 C0 | Q s l w | P lmode R | D1 mode | D3 R | D4 mode | D5\n");
    return 0;
}
