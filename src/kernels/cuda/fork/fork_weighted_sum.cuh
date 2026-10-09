// src/kernels/cuda/fork/fork_weighted_sum.cuh - weighted_rows_sum: the prompt chunk's top-k mix on the card (MiMo prefill).
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

// out[t][i] = sum_k w[t*K + k] * parts[(t*K + k)][i] - the prompt chunk's top-k mix on the card (MiMo prefill), so
// only n rows come back instead of n*K.  Summed in k order, like the host loop it replaces.
__global__ void weighted_rows_sum_kernel(const float* __restrict__ parts, const float* __restrict__ w, int K,
                                         int64_t H, int64_t n, float* __restrict__ out) {
    const int64_t total = n * H;
    for (int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; q < total; q += (int64_t) gridDim.x * blockDim.x) {
        const int64_t t = q / H, i = q - t * H;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += w[t * K + k] * parts[(t * K + k) * H + i];
        out[q] = acc;
    }
}

void weighted_rows_sum(const float* parts, const float* w, int K, int64_t H, int64_t n, float* out, void* stream) {
    const int64_t total = n * H;
    const unsigned blocks = (unsigned) std::min<int64_t>((total + 255) / 256, 65535);
    weighted_rows_sum_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(parts, w, K, H, n, out);
    check("weighted_rows_sum");
}
