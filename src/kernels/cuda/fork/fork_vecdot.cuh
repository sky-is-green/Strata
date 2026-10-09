// src/kernels/cuda/fork/fork_vecdot.cuh - fork formats' q8_1 dot products: Q2_K (DeepSeek-V4 down), Q3_K and MXFP4 (MiMo GSQ-RCO).
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

// Q2_K (DeepSeek-V4's down projections): llama.cpp's vec_dot_q2_K_q8_1 (vecdotq.cuh), VDR_Q2_K_Q8_1_MMVQ = 1.
constexpr int VDR_Q2_K = 1;

__device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(int v, const int* __restrict__ u,
                                                             const uint8_t* __restrict__ scales,
                                                             const half2& dm2, const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2 * i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);   // the min times the sum of the QUANTIZED activations
    }
    const float2 dm2f = __half22float2(dm2);
    return dm2f.x * sumf_d - dm2f.y * sumf_m;
}
__device__ __forceinline__ float vec_dot_q2_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_K* bq2_K = (const block_q2_K*) vbq + kbx;
    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    const uint8_t* scales = bq2_K->scales + scale_offset;
    const int v = get_int_b4(bq2_K->qs, iqs);
    int u[QR2_K];
    float d8[QR2_K];
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

// Q3_K (MiMo-V2.6-Flash GSQ-RCO's gate/up/down on some layers): llama.cpp's vec_dot_q3_K_q8_1 (vecdotq.cuh),
// VDR_Q3_K_Q8_1_MMVQ = 1.  The high-bit mask is inverted so a clear bit subtracts 4.
constexpr int VDR_Q3_K = 1;
__device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(const int& vl, const int& vh, const int* __restrict__ u,
                                                             const uint8_t* __restrict__ scales, const int& scale_offset,
                                                             const float& d3, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % (QK_K / 32);
        const int sc_shift_low = 4 * (isc / (QK_K / 32));
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;
        const int isc_high = isc % (QK_K / 64);
        const int sc_shift_high = 2 * (isc / (QK_K / 64));
        const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}
__device__ __forceinline__ float vec_dot_q3_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q3_K* bq3_K = (const block_q3_K*) vbq + kbx;
    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    const float d = __half2float(bq3_K->d);
    const int vl = get_int_b2(bq3_K->qs, iqs);
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;
    int u[QR3_K];
    float d8[QR3_K];
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

// MXFP4 (MiMo's source-format down projections on 3 layers): llama.cpp's vec_dot_mxfp4_q8_1 (vecdotq.cuh),
// VDR_MXFP4_Q8_1_MMVQ = 2.  kvalues_mxfp4 holds the e2m1 values doubled, hence the 0.5.
constexpr int VDR_MXFP4 = 2;
__device__ __forceinline__ float vec_dot_mxfp4_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_mxfp4* bq4 = (const block_mxfp4*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4; ++l) {
        const int2 v = get_int_from_table_16(get_int_b1(bq4->qs, iqs + l), kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    return e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds) * sumi;
}
