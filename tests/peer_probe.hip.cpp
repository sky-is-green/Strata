// Peer-to-peer probe: access capability, copy latency (direct peer vs host-staged), cross-device event wait.
// Build:  /opt/rocm/bin/hipcc -O2 --offload-arch=gfx1100 tests/peer_probe.hip.cpp -o build-hip/peer_probe
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>

__global__ void spin_ms(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}
__global__ void touch(int* p) { *p = 1; }

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int n = 0;
    hipGetDeviceCount(&n);
    std::printf("devices: %d\n", n);
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t p;
        hipGetDeviceProperties(&p, i);
        std::printf("  %d: %s  (vram=%.1f GiB)\n", i, p.name, p.totalGlobalMem / 1073741824.0);
    }
    if (n < 2) return 0;
    int c10 = 0, c01 = 0;
    hipDeviceCanAccessPeer(&c10, 1, 0);
    hipDeviceCanAccessPeer(&c01, 0, 1);
    std::printf("peer 1->0: %d   peer 0->1: %d\n", c10, c01);
    // enable peer access (no-op if not available)
    hipSetDevice(1); hipError_t e1 = hipDeviceEnablePeerAccess(0, 0);
    hipSetDevice(0); hipError_t e0 = hipDeviceEnablePeerAccess(1, 0);
    std::printf("enable: 1->0 %s, 0->1 %s\n", hipGetErrorName(e1), hipGetErrorName(e0));

    hipStream_t s0, s1;
    hipSetDevice(0); hipStreamCreate(&s0);
    hipSetDevice(1); hipStreamCreate(&s1);
    const size_t kb = 20 * 1024;                       // one activation handoff
    void* a = nullptr; void* b = nullptr; void* stage = nullptr;
    hipSetDevice(0); hipMalloc(&a, kb); hipMemset(a, 0x11, kb);
    hipSetDevice(1); hipMalloc(&b, kb);
    hipHostMalloc(&stage, kb, hipHostMallocPortable | hipHostMallocMapped);
    int* flag1 = nullptr;
    hipSetDevice(1); hipMalloc(&flag1, 4); hipSetDevice(0);
    const int R = 2000;

    // (a) staged: D2H on s0, H2D on s1
    hipSetDevice(0);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < R; ++i) {
        hipMemcpyAsync(stage, a, kb, hipMemcpyDeviceToHost, s0);
        hipStreamSynchronize(s0);
        hipSetDevice(1);
        hipMemcpyAsync(b, stage, kb, hipMemcpyHostToDevice, s1);
        hipStreamSynchronize(s1);
        hipSetDevice(0);
    }
    std::printf("staged 20 KB x%d: %.1f us/copy\n", R, ms_since(t0) * 1000.0 / R);

    // (b) direct peer copy on s1
    hipSetDevice(1);
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < R; ++i) {
        hipMemcpyPeerAsync(b, 1, a, 0, kb, s1);
        hipStreamSynchronize(s1);
    }
    std::printf("peer   20 KB x%d: %.1f us/copy (%s)\n", R, ms_since(t0) * 1000.0 / R,
                hipGetErrorName(hipGetLastError()));
    hipSetDevice(0);

    // (c) cross-device event wait with an uncompleted event (hung before P2P)
    hipEvent_t ev;
    hipEventCreateWithFlags(&ev, hipEventDisableTiming);
    hipLaunchKernelGGL(spin_ms, dim3(1), dim3(1), 0, s0, (long long) 200000000);   // ~0.15 s
    hipEventRecord(ev, s0);
    t0 = std::chrono::steady_clock::now();
    hipStreamWaitEvent(s1, ev, 0);
    hipLaunchKernelGGL(touch, dim3(1), dim3(1), 0, s1, flag1);
    hipError_t w = hipStreamSynchronize(s1);
    std::printf("cross-device event wait: %s, waited %.0f ms (expect >= ~100)\n", hipGetErrorName(w), ms_since(t0));
    hipStreamSynchronize(s0);
    return 0;
}
