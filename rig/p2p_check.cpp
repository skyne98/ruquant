// rig/p2p_check.cpp: hipcc -O2 --offload-arch=gfx906 rig/p2p_check.cpp -o p2p_check && HSA_FORCE_FINE_GRAIN_PCIE=1 ./p2p_check
// Direct GPU-to-GPU check, the way the TP all-reduce moves data: peer access per pair (hipDeviceCanAccessPeer and
// hipDeviceEnablePeerAccess), then a kernel on GPU i writing 1 MB straight into a fine-grained buffer on GPU j (no host
// copy, no hipMemcpy), verified on GPU j, timed. A kernel can only dereference a peer pointer through a P2P mapping.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>

__global__ void fill(float * dst, int n, float v) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = v + i;
}
__global__ void check(const float * src, int n, float v, int * bad) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) if (src[i] != v + i) atomicAdd(bad, 1);
}

int main() {
    int nd = 0; hipGetDeviceCount(&nd);
    const int n = 1 << 18; // 1 MB of floats
    std::vector<float *> buf(nd);
    for (int d = 0; d < nd; ++d) {
        hipSetDevice(d);
        hipExtMallocWithFlags((void **) &buf[d], n * sizeof(float), hipDeviceMallocFinegrained);
        for (int p = 0; p < nd; ++p) if (p != d) hipDeviceEnablePeerAccess(p, 0);
        (void) hipGetLastError();
    }
    printf("canAccessPeer:\n");
    for (int i = 0; i < nd; ++i) { for (int j = 0; j < nd; ++j) { int a = 0; if (i != j) hipDeviceCanAccessPeer(&a, i, j); printf(" %d", i == j ? 1 : a); } printf("\n"); }
    printf("kernel on GPU i writes 1 MB into GPU j's memory (GB/s, verified on j):\n");
    for (int i = 0; i < nd; ++i) {
        for (int j = 0; j < nd; ++j) {
            if (i == j) { printf("     -   "); continue; }
            hipSetDevice(i);
            const float v = 1000.0f * i + j;
            fill<<<64, 256>>>(buf[j], n, v); hipDeviceSynchronize();
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 20; ++r) fill<<<64, 256>>>(buf[j], n, v);
            hipDeviceSynchronize();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
            hipSetDevice(j);
            int * bad; hipMalloc(&bad, 4); hipMemset(bad, 0, 4);
            check<<<64, 256>>>(buf[j], n, v, bad);
            int h = -1; hipMemcpy(&h, bad, 4, hipMemcpyDeviceToHost); hipFree(bad);
            printf(" %5.2f%s  ", n * 4 / s / 1e9, h == 0 ? "ok" : "BAD");
        }
        printf("\n");
    }
}
