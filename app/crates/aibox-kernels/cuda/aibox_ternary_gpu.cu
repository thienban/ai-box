#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mutex>
#include <stdint.h>
#include <stdlib.h>

#define GROUP_SIZE 128
#define PACKED_BYTES_PER_GROUP 32
#define BINARY_PACKED_BYTES_PER_GROUP 16
#define GEMM_BM 16
#define WARP_SIZE 32
#define MAX_ATTN_SEQ 4096
#define MAX_FUSED_RMS_DIM 8192
#define ARGMAX_BLOCKS 128

namespace {

struct ScratchBuffer {
    void* ptr = nullptr;
    size_t capacity = 0;
};

std::mutex g_scratch_mutex;
ScratchBuffer g_dev_x;
ScratchBuffer g_dev_out;
ScratchBuffer g_dev_argmax_vals;
ScratchBuffer g_dev_argmax_idxs;
ScratchBuffer g_dev_argmax_out;
ScratchBuffer g_dev_cand_idxs;
ScratchBuffer g_dev_cand_vals;
ScratchBuffer g_dev_cand_count;

cudaStream_t g_active_stream = nullptr;
cudaStream_t g_graph_stream = nullptr;
int* g_dev_pos = nullptr;
cudaGraph_t g_decode_graph = nullptr;
cudaGraphExec_t g_decode_graph_exec = nullptr;

bool ensure_scratch_capacity(ScratchBuffer& buf, size_t required_bytes) {
    if (required_bytes <= buf.capacity) {
        return true;
    }
    if (buf.ptr != nullptr) {
        cudaFree(buf.ptr);
        buf.ptr = nullptr;
        buf.capacity = 0;
    }
    if (cudaMalloc(&buf.ptr, required_bytes) != cudaSuccess) {
        return false;
    }
    buf.capacity = required_bytes;
    return true;
}

}  // namespace

static __device__ __forceinline__ float decode_2bit_weight(uint8_t code) {
    return (code == 1) ? 1.0f : ((code == 3) ? -1.0f : 0.0f);
}

static __device__ __forceinline__ float dot_packed_byte_f4(uint8_t packed, float4 xv) {
    return decode_2bit_weight(packed & 0x03) * xv.x
         + decode_2bit_weight((packed >> 2) & 0x03) * xv.y
         + decode_2bit_weight((packed >> 4) & 0x03) * xv.z
         + decode_2bit_weight((packed >> 6) & 0x03) * xv.w;
}

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

static __device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

static __device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float s_warp[8];
    __shared__ float s_total;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    val = warp_reduce_sum(val);
    if (lane == 0) {
        s_warp[warp_id] = val;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int w = 0; w < num_warps; ++w) {
            acc += s_warp[w];
        }
        s_total = acc;
    }
    __syncthreads();
    return s_total;
}

static __device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float s_warp[8];
    __shared__ float s_max;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    val = warp_reduce_max(val);
    if (lane == 0) {
        s_warp[warp_id] = val;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int w = 0; w < num_warps; ++w) {
            m = fmaxf(m, s_warp[w]);
        }
        s_max = m;
    }
    __syncthreads();
    return s_max;
}

static __device__ __forceinline__ void fwht128_warp_inplace(float* warp_buf, int lane) {
    for (int step = 1; step < GROUP_SIZE; step <<= 1) {
        int mask = step - 1;
        for (int pair = lane; pair < GROUP_SIZE / 2; pair += WARP_SIZE) {
            int block_idx = (pair & ~mask) << 1;
            int i0 = block_idx + (pair & mask);
            int i1 = i0 + step;
            float a = warp_buf[i0];
            float b = warp_buf[i1];
            warp_buf[i0] = a + b;
            warp_buf[i1] = a - b;
        }
        __syncwarp();
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    int base = lane * 4;
    warp_buf[base + 0] *= inv_sqrt_128;
    warp_buf[base + 1] *= inv_sqrt_128;
    warp_buf[base + 2] *= inv_sqrt_128;
    warp_buf[base + 3] *= inv_sqrt_128;
}

__global__ void fwht128_rows_kernel(float* __restrict__ x, int total_groups) {
    int group_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (group_idx >= total_groups) {
        return;
    }

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    float* x_group = x + (size_t)group_idx * GROUP_SIZE;

    __shared__ float s_fwht[4][GROUP_SIZE];
    float* warp_buf = s_fwht[warp_in_block];

    int base = lane * 4;
    float4 xv = *reinterpret_cast<const float4*>(&x_group[base]);
    warp_buf[base + 0] = xv.x;
    warp_buf[base + 1] = xv.y;
    warp_buf[base + 2] = xv.z;
    warp_buf[base + 3] = xv.w;
    __syncwarp();

    fwht128_warp_inplace(warp_buf, lane);

    *reinterpret_cast<float4*>(&x_group[base]) = make_float4(
        warp_buf[base + 0],
        warp_buf[base + 1],
        warp_buf[base + 2],
        warp_buf[base + 3]
    );
}

template <bool AddResidual>
__global__ void ternary_gemv_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const uint16_t* __restrict__ scales,
    float* __restrict__ out,
    int m,
    int n,
    int k
) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int row_n = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int row_m = blockIdx.y;
    if (row_n >= n || row_m >= m) {
        return;
    }

    int groups_per_row = k / GROUP_SIZE;
    const float* x_row = x + (size_t)row_m * k;
    const uint8_t* w_row = w_packed + (size_t)row_n * groups_per_row * PACKED_BYTES_PER_GROUP;
    const uint16_t* s_row = scales + (size_t)row_n * groups_per_row;
    const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_row);

    float total = 0.0f;
    int sub_g = lane >> 3;
    int sub_lane = lane & 7;

    int g = 0;
    for (; g + 4 <= groups_per_row; g += 4) {
        int cur_g = g + sub_g;
        uint32_t packed4 = w_u32[cur_g * 8 + sub_lane];
        const float4* x4 = reinterpret_cast<const float4*>(&x_row[cur_g * GROUP_SIZE + sub_lane * 16]);

        float partial = dot_packed_byte_f4(packed4 & 0xFF, x4[0])
                      + dot_packed_byte_f4((packed4 >> 8) & 0xFF, x4[1])
                      + dot_packed_byte_f4((packed4 >> 16) & 0xFF, x4[2])
                      + dot_packed_byte_f4((packed4 >> 24) & 0xFF, x4[3]);

        partial += __shfl_down_sync(0xFFFFFFFF, partial, 4);
        partial += __shfl_down_sync(0xFFFFFFFF, partial, 2);
        partial += __shfl_down_sync(0xFFFFFFFF, partial, 1);

        if (sub_lane == 0) {
            __half h_scale = *reinterpret_cast<const __half*>(&s_row[cur_g]);
            total += partial * __half2float(h_scale);
        }
    }

    for (; g < groups_per_row; ++g) {
        float4 xv = *reinterpret_cast<const float4*>(&x_row[g * GROUP_SIZE + lane * 4]);
        uint8_t packed = w_row[g * PACKED_BYTES_PER_GROUP + lane];
        float partial = warp_reduce_sum(dot_packed_byte_f4(packed, xv));
        if (lane == 0) {
            __half h_scale = *reinterpret_cast<const __half*>(&s_row[g]);
            total += partial * __half2float(h_scale);
        }
    }

    total += __shfl_down_sync(0xFFFFFFFF, total, 16);
    total += __shfl_down_sync(0xFFFFFFFF, total, 8);
    if (lane == 0) {
        size_t out_idx = (size_t)row_m * n + row_n;
        if (AddResidual) {
            out[out_idx] += total;
        } else {
            out[out_idx] = total;
        }
    }
}

__global__ void argmax_stage1_kernel(
    const float* __restrict__ logits,
    int len,
    float* __restrict__ block_vals,
    uint32_t* __restrict__ block_idxs
) {
    __shared__ float s_val[8];
    __shared__ uint32_t s_idx[8];

    float best_v = -3.402823466e+38f;
    uint32_t best_i = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) {
        float v = logits[i];
        if (v > best_v || (v == best_v && (uint32_t)i < best_i)) {
            best_v = v;
            best_i = (uint32_t)i;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xFFFFFFFF, best_v, offset);
        uint32_t oi = __shfl_down_sync(0xFFFFFFFF, best_i, offset);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (lane == 0) {
        s_val[warp_id] = best_v;
        s_idx[warp_id] = best_i;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        float fv = s_val[0];
        uint32_t fi = s_idx[0];
        for (int w = 1; w < num_warps; ++w) {
            if (s_val[w] > fv || (s_val[w] == fv && s_idx[w] < fi)) {
                fv = s_val[w];
                fi = s_idx[w];
            }
        }
        block_vals[blockIdx.x] = fv;
        block_idxs[blockIdx.x] = fi;
    }
}

__global__ void argmax_stage2_kernel(
    float* __restrict__ block_vals,
    const uint32_t* __restrict__ block_idxs,
    int num_blocks,
    uint32_t* __restrict__ out_idx
) {
    __shared__ float s_val[ARGMAX_BLOCKS];
    __shared__ uint32_t s_idx[ARGMAX_BLOCKS];

    int tid = threadIdx.x;
    if (tid < num_blocks) {
        s_val[tid] = block_vals[tid];
        s_idx[tid] = block_idxs[tid];
    } else {
        s_val[tid] = -3.402823466e+38f;
        s_idx[tid] = 0;
    }
    __syncthreads();

    for (int stride = num_blocks / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float ov = s_val[tid + stride];
            uint32_t oi = s_idx[tid + stride];
            if (ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid])) {
                s_val[tid] = ov;
                s_idx[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[0] = s_val[0];
        *out_idx = s_idx[0];
    }
}

__global__ void filter_candidates_kernel(
    const float* __restrict__ logits,
    int len,
    const float* __restrict__ max_val_ptr,
    float margin,
    uint32_t* __restrict__ out_idxs,
    float* __restrict__ out_vals,
    int max_out,
    int* __restrict__ out_count
) {
    float threshold = *max_val_ptr - margin;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; idx < len; idx += stride) {
        float val = logits[idx];
        if (val >= threshold) {
            int pos = atomicAdd(out_count, 1);
            if (pos < max_out) {
                out_idxs[pos] = static_cast<uint32_t>(idx);
                out_vals[pos] = val;
            }
        }
    }
}

static __device__ __forceinline__ float dot_packed_byte_h4(uint8_t packed, __half2 h0, __half2 h1) {
    float2 f0 = __half22float2(h0);
    float2 f1 = __half22float2(h1);
    return decode_2bit_weight(packed & 0x03) * f0.x
         + decode_2bit_weight((packed >> 2) & 0x03) * f0.y
         + decode_2bit_weight((packed >> 4) & 0x03) * f1.x
         + decode_2bit_weight((packed >> 6) & 0x03) * f1.y;
}

template <bool AddResidual>
static __device__ __forceinline__ void store_gemv_out(__half* out, size_t idx, float total) {
    if (AddResidual) {
        out[idx] = __float2half(__half2float(out[idx]) + total);
    } else {
        out[idx] = __float2half(total);
    }
}

template <bool AddResidual>
static __device__ __forceinline__ void store_gemv_out(float* out, size_t idx, float total) {
    if (AddResidual) {
        out[idx] += total;
    } else {
        out[idx] = total;
    }
}

__global__ void fwht128_rows_f16_kernel(__half* __restrict__ x, int total_groups) {
    int group_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (group_idx >= total_groups) {
        return;
    }

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    __half* x_group = x + (size_t)group_idx * GROUP_SIZE;

    __shared__ float s_fwht[4][GROUP_SIZE];
    float* warp_buf = s_fwht[warp_in_block];

    int base = lane * 4;
    uint2 raw = *reinterpret_cast<const uint2*>(&x_group[base]);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
    float2 f0 = __half22float2(h2[0]);
    float2 f1 = __half22float2(h2[1]);
    warp_buf[base + 0] = f0.x;
    warp_buf[base + 1] = f0.y;
    warp_buf[base + 2] = f1.x;
    warp_buf[base + 3] = f1.y;
    __syncwarp();

    fwht128_warp_inplace(warp_buf, lane);

    __half2 out_h2[2];
    out_h2[0] = __float22half2_rn(make_float2(warp_buf[base + 0], warp_buf[base + 1]));
    out_h2[1] = __float22half2_rn(make_float2(warp_buf[base + 2], warp_buf[base + 3]));
    *reinterpret_cast<uint2*>(&x_group[base]) = *reinterpret_cast<const uint2*>(out_h2);
}

static __device__ __forceinline__ float dot_binary_nibble_h4(uint8_t nibble, __half2 h0, __half2 h1) {
    float2 f0 = __half22float2(h0);
    float2 f1 = __half22float2(h1);
    float s0 = (nibble & 0x01) ? 1.0f : -1.0f;
    float s1 = (nibble & 0x02) ? 1.0f : -1.0f;
    float s2 = (nibble & 0x04) ? 1.0f : -1.0f;
    float s3 = (nibble & 0x08) ? 1.0f : -1.0f;
    return s0 * f0.x + s1 * f0.y + s2 * f1.x + s3 * f1.y;
}

template <int Bits>
static __device__ __forceinline__ float dot_packed_chunk(
    const uint16_t* __restrict__ w_u16,
    const uint32_t* __restrict__ w_u32,
    int idx,
    const __half2* h0,
    const __half2* h1
) {
    if (Bits == 1) {
        uint16_t packed2 = w_u16[idx];
        return dot_binary_nibble_h4(packed2 & 0x0F, h0[0], h0[1])
             + dot_binary_nibble_h4((packed2 >> 4) & 0x0F, h0[2], h0[3])
             + dot_binary_nibble_h4((packed2 >> 8) & 0x0F, h1[0], h1[1])
             + dot_binary_nibble_h4((packed2 >> 12) & 0x0F, h1[2], h1[3]);
    }
    uint32_t packed4 = w_u32[idx];
    return dot_packed_byte_h4(packed4 & 0xFF, h0[0], h0[1])
         + dot_packed_byte_h4((packed4 >> 8) & 0xFF, h0[2], h0[3])
         + dot_packed_byte_h4((packed4 >> 16) & 0xFF, h1[0], h1[1])
         + dot_packed_byte_h4((packed4 >> 24) & 0xFF, h1[2], h1[3]);
}

template <bool AddResidual, typename OutT, int Bits = 2>
__global__ void ternary_gemv_f16_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const uint16_t* __restrict__ scales,
    OutT* __restrict__ out,
    int m,
    int n,
    int k
) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int row_n = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int row_m = blockIdx.y;
    if (row_n >= n || row_m >= m) {
        return;
    }

    int groups_per_row = k / GROUP_SIZE;
    const int bytes_per_group = (Bits == 1) ? BINARY_PACKED_BYTES_PER_GROUP : PACKED_BYTES_PER_GROUP;
    const __half* x_row = x + (size_t)row_m * k;
    const uint8_t* w_row = w_packed + (size_t)row_n * groups_per_row * bytes_per_group;
    const uint16_t* s_row = scales + (size_t)row_n * groups_per_row;
    const uint32_t* w_u32 = reinterpret_cast<const uint32_t*>(w_row);
    const uint16_t* w_u16 = reinterpret_cast<const uint16_t*>(w_row);

    float total = 0.0f;
    int sub_g = lane >> 3;
    int sub_lane = lane & 7;

    int g = 0;
    for (; g + 4 <= groups_per_row; g += 4) {
        int cur_g = g + sub_g;
        const uint4* x_u4 = reinterpret_cast<const uint4*>(&x_row[cur_g * GROUP_SIZE + sub_lane * 16]);
        uint4 u0 = x_u4[0];
        uint4 u1 = x_u4[1];
        const __half2* h0 = reinterpret_cast<const __half2*>(&u0);
        const __half2* h1 = reinterpret_cast<const __half2*>(&u1);

        float partial = dot_packed_chunk<Bits>(w_u16, w_u32, cur_g * 8 + sub_lane, h0, h1);

        partial += __shfl_down_sync(0xFFFFFFFF, partial, 4);
        partial += __shfl_down_sync(0xFFFFFFFF, partial, 2);
        partial += __shfl_down_sync(0xFFFFFFFF, partial, 1);

        if (sub_lane == 0) {
            __half h_scale = *reinterpret_cast<const __half*>(&s_row[cur_g]);
            total += partial * __half2float(h_scale);
        }
    }

    for (; g < groups_per_row; ++g) {
        uint2 x_u2 = *reinterpret_cast<const uint2*>(&x_row[g * GROUP_SIZE + lane * 4]);
        const __half2* h2 = reinterpret_cast<const __half2*>(&x_u2);
        float elem_dot;
        if (Bits == 1) {
            uint8_t nibble = (w_row[g * BINARY_PACKED_BYTES_PER_GROUP + (lane >> 1)] >> ((lane & 1) * 4)) & 0x0F;
            elem_dot = dot_binary_nibble_h4(nibble, h2[0], h2[1]);
        } else {
            uint8_t packed = w_row[g * PACKED_BYTES_PER_GROUP + lane];
            elem_dot = dot_packed_byte_h4(packed, h2[0], h2[1]);
        }
        float partial = warp_reduce_sum(elem_dot);
        if (lane == 0) {
            __half h_scale = *reinterpret_cast<const __half*>(&s_row[g]);
            total += partial * __half2float(h_scale);
        }
    }

    total += __shfl_down_sync(0xFFFFFFFF, total, 16);
    total += __shfl_down_sync(0xFFFFFFFF, total, 8);
    if (lane == 0) {
        size_t out_idx = (size_t)row_m * n + row_n;
        store_gemv_out<AddResidual>(out, out_idx, total);
    }
}

template <int Bits = 2>
__global__ void ternary_gemv_swiglu_f16_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_gate_packed,
    const uint16_t* __restrict__ scales_gate,
    const uint8_t* __restrict__ w_up_packed,
    const uint16_t* __restrict__ scales_up,
    __half* __restrict__ out,
    int n,
    int k
) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int row_n = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (row_n >= n) {
        return;
    }

    int groups_per_row = k / GROUP_SIZE;
    const int bytes_per_group = (Bits == 1) ? BINARY_PACKED_BYTES_PER_GROUP : PACKED_BYTES_PER_GROUP;
    size_t row_w_offset = (size_t)row_n * groups_per_row * bytes_per_group;
    size_t row_s_offset = (size_t)row_n * groups_per_row;

    const uint32_t* wg_u32 = reinterpret_cast<const uint32_t*>(w_gate_packed + row_w_offset);
    const uint16_t* wg_u16 = reinterpret_cast<const uint16_t*>(w_gate_packed + row_w_offset);
    const uint16_t* sg_row = scales_gate + row_s_offset;

    const uint32_t* wu_u32 = reinterpret_cast<const uint32_t*>(w_up_packed + row_w_offset);
    const uint16_t* wu_u16 = reinterpret_cast<const uint16_t*>(w_up_packed + row_w_offset);
    const uint16_t* su_row = scales_up + row_s_offset;

    float total_g = 0.0f;
    float total_u = 0.0f;
    int sub_g = lane >> 3;
    int sub_lane = lane & 7;

    for (int g = 0; g + 4 <= groups_per_row; g += 4) {
        int cur_g = g + sub_g;
        const uint4* x_u4 = reinterpret_cast<const uint4*>(&x[cur_g * GROUP_SIZE + sub_lane * 16]);
        uint4 u0 = x_u4[0];
        uint4 u1 = x_u4[1];
        const __half2* h0 = reinterpret_cast<const __half2*>(&u0);
        const __half2* h1 = reinterpret_cast<const __half2*>(&u1);

        int chunk_idx = cur_g * 8 + sub_lane;
        float pg = dot_packed_chunk<Bits>(wg_u16, wg_u32, chunk_idx, h0, h1);
        float pu = dot_packed_chunk<Bits>(wu_u16, wu_u32, chunk_idx, h0, h1);

        pg += __shfl_down_sync(0xFFFFFFFF, pg, 4);
        pg += __shfl_down_sync(0xFFFFFFFF, pg, 2);
        pg += __shfl_down_sync(0xFFFFFFFF, pg, 1);

        pu += __shfl_down_sync(0xFFFFFFFF, pu, 4);
        pu += __shfl_down_sync(0xFFFFFFFF, pu, 2);
        pu += __shfl_down_sync(0xFFFFFFFF, pu, 1);

        if (sub_lane == 0) {
            total_g += pg * __half2float(*reinterpret_cast<const __half*>(&sg_row[cur_g]));
            total_u += pu * __half2float(*reinterpret_cast<const __half*>(&su_row[cur_g]));
        }
    }

    total_g += __shfl_down_sync(0xFFFFFFFF, total_g, 16);
    total_g += __shfl_down_sync(0xFFFFFFFF, total_g, 8);
    total_u += __shfl_down_sync(0xFFFFFFFF, total_u, 16);
    total_u += __shfl_down_sync(0xFFFFFFFF, total_u, 8);
    if (lane == 0) {
        out[row_n] = __float2half((total_g / (1.0f + expf(-total_g))) * total_u);
    }
}

template <bool AddResidual, typename OutT, int Bits = 2>
__global__ void ternary_gemm_tiled_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const uint16_t* __restrict__ scales,
    OutT* __restrict__ out,
    int m,
    int n,
    int k
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row_n = blockIdx.x * 4 + warp_id;
    int m0 = blockIdx.y * GEMM_BM;
    int m_count = min(GEMM_BM, m - m0);
    if (m_count <= 0) {
        return;
    }

    __shared__ __half s_x[GEMM_BM][GROUP_SIZE];
    int groups_per_row = k / GROUP_SIZE;
    const int bytes_per_group = (Bits == 1) ? BINARY_PACKED_BYTES_PER_GROUP : PACKED_BYTES_PER_GROUP;
    const uint8_t* w_row = (row_n < n) ? (w_packed + (size_t)row_n * groups_per_row * bytes_per_group) : nullptr;
    const uint16_t* s_row = (row_n < n) ? (scales + (size_t)row_n * groups_per_row) : nullptr;

    float acc[GEMM_BM];
    #pragma unroll
    for (int tm = 0; tm < GEMM_BM; ++tm) {
        acc[tm] = 0.0f;
    }

    // 128 threads load m_count * 128 __halfs = m_count * 16 uint4s into shared memory
    int total_u4 = m_count * (GROUP_SIZE >> 3);
    for (int g = 0; g < groups_per_row; ++g) {
        for (int idx = threadIdx.x; idx < total_u4; idx += blockDim.x) {
            int tm = idx >> 4;
            int col_u4 = idx & 15;
            uint4 v = *reinterpret_cast<const uint4*>(&x[(size_t)(m0 + tm) * k + g * GROUP_SIZE + (col_u4 << 3)]);
            *reinterpret_cast<uint4*>(&s_x[tm][col_u4 << 3]) = v;
        }
        __syncthreads();

        if (row_n < n) {
            float scale = __half2float(*reinterpret_cast<const __half*>(&s_row[g]));
            uint8_t w_code = (Bits == 1)
                ? ((w_row[g * BINARY_PACKED_BYTES_PER_GROUP + (lane >> 1)] >> ((lane & 1) * 4)) & 0x0F)
                : w_row[g * PACKED_BYTES_PER_GROUP + lane];

            for (int tm = 0; tm < m_count; ++tm) {
                uint2 x_u2 = *reinterpret_cast<const uint2*>(&s_x[tm][lane * 4]);
                const __half2* h2 = reinterpret_cast<const __half2*>(&x_u2);
                float elem_dot = (Bits == 1)
                    ? dot_binary_nibble_h4(w_code, h2[0], h2[1])
                    : dot_packed_byte_h4(w_code, h2[0], h2[1]);
                acc[tm] += elem_dot * scale;
            }
        }
        __syncthreads();
    }

    if (row_n < n) {
        for (int tm = 0; tm < m_count; ++tm) {
            float sum = warp_reduce_sum(acc[tm]);
            if (lane == 0) {
                store_gemv_out<AddResidual>(out, (size_t)(m0 + tm) * n + row_n, sum);
            }
        }
    }
}

__global__ void rms_norm_f16_kernel(
    const __half* __restrict__ x,
    const float* __restrict__ weight,
    __half* __restrict__ out,
    int dim,
    float eps
) {
    int row = blockIdx.x;
    const __half* x_row = x + (size_t)row * dim;
    __half* out_row = out + (size_t)row * dim;

    float local_sum = 0.0f;
    if ((dim & 7) == 0) {
        int dim8 = dim >> 3;
        const uint4* x8 = reinterpret_cast<const uint4*>(x_row);
        for (int i = threadIdx.x; i < dim8; i += blockDim.x) {
            uint4 raw = x8[i];
            const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
            for (int j = 0; j < 4; ++j) {
                float2 f = __half22float2(h2[j]);
                local_sum += f.x * f.x + f.y * f.y;
            }
        }
    } else {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            float v = __half2float(x_row[i]);
            local_sum += v * v;
        }
    }

    float inv_rms = rsqrtf(block_reduce_sum(local_sum) / (float)dim + eps);
    if ((dim & 7) == 0) {
        int dim8 = dim >> 3;
        const uint4* x8 = reinterpret_cast<const uint4*>(x_row);
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        uint4* o8 = reinterpret_cast<uint4*>(out_row);
        for (int i = threadIdx.x; i < dim8; i += blockDim.x) {
            uint4 raw = x8[i];
            const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
            float4 w0 = w4[i * 2 + 0];
            float4 w1 = w4[i * 2 + 1];
            float2 f0 = __half22float2(h2[0]);
            float2 f1 = __half22float2(h2[1]);
            float2 f2 = __half22float2(h2[2]);
            float2 f3 = __half22float2(h2[3]);
            __half2 out_h2[4];
            out_h2[0] = __float22half2_rn(make_float2(f0.x * inv_rms * w0.x, f0.y * inv_rms * w0.y));
            out_h2[1] = __float22half2_rn(make_float2(f1.x * inv_rms * w0.z, f1.y * inv_rms * w0.w));
            out_h2[2] = __float22half2_rn(make_float2(f2.x * inv_rms * w1.x, f2.y * inv_rms * w1.y));
            out_h2[3] = __float22half2_rn(make_float2(f3.x * inv_rms * w1.z, f3.y * inv_rms * w1.w));
            o8[i] = *reinterpret_cast<const uint4*>(out_h2);
        }
    } else {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            out_row[i] = __float2half(__half2float(x_row[i]) * inv_rms * weight[i]);
        }
    }
}

__global__ void rms_norm_fwht128_f16_kernel(
    const __half* __restrict__ x,
    const float* __restrict__ weight,
    __half* __restrict__ out,
    int dim,
    float eps
) {
    int row = blockIdx.x;
    const __half* x_row = x + (size_t)row * dim;
    __half* out_row = out + (size_t)row * dim;

    __shared__ float s_row[MAX_FUSED_RMS_DIM];
    int dim8 = dim >> 3;
    const uint4* x8 = reinterpret_cast<const uint4*>(x_row);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    float4* s4 = reinterpret_cast<float4*>(s_row);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim8; i += blockDim.x) {
        uint4 raw = x8[i];
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
        float2 f0 = __half22float2(h2[0]);
        float2 f1 = __half22float2(h2[1]);
        float2 f2 = __half22float2(h2[2]);
        float2 f3 = __half22float2(h2[3]);
        s4[i * 2 + 0] = make_float4(f0.x, f0.y, f1.x, f1.y);
        s4[i * 2 + 1] = make_float4(f2.x, f2.y, f3.x, f3.y);
        local_sum += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y
                   + f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
    }

    float inv_rms = rsqrtf(block_reduce_sum(local_sum) / (float)dim + eps);
    int dim4 = dim >> 2;
    for (int i = threadIdx.x; i < dim4; i += blockDim.x) {
        float4 v = s4[i];
        float4 wv = w4[i];
        s4[i] = make_float4(
            v.x * inv_rms * wv.x,
            v.y * inv_rms * wv.y,
            v.z * inv_rms * wv.z,
            v.w * inv_rms * wv.w
        );
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int num_warps = blockDim.x / WARP_SIZE;
    int num_groups = dim / GROUP_SIZE;
    uint2* o2 = reinterpret_cast<uint2*>(out_row);

    for (int g = warp_id; g < num_groups; g += num_warps) {
        float* group_buf = &s_row[g * GROUP_SIZE];
        fwht128_warp_inplace(group_buf, lane);
        int base = lane * 4;
        __half2 out_h2[2];
        out_h2[0] = __float22half2_rn(make_float2(group_buf[base + 0], group_buf[base + 1]));
        out_h2[1] = __float22half2_rn(make_float2(group_buf[base + 2], group_buf[base + 3]));
        o2[(g * GROUP_SIZE) / 4 + lane] = *reinterpret_cast<const uint2*>(out_h2);
    }
}

__global__ void rms_norm_heads_rope_f16_kernel(
    __half* __restrict__ vec,
    const float* __restrict__ norm_w,
    int head_dim,
    int rope_dim,
    int start_pos,
    const int* __restrict__ dev_pos,
    float theta,
    float eps
) {
    int head = blockIdx.x;
    int tok = blockIdx.y;
    int effective_pos = (dev_pos != nullptr) ? *dev_pos : start_pos;
    int pos = effective_pos + tok;
    __half* h_vec = vec + ((size_t)tok * gridDim.x + head) * head_dim;

    if (norm_w != nullptr) {
        float local_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = __half2float(h_vec[i]);
            local_sq += v * v;
        }
        float inv_rms = rsqrtf(block_reduce_sum(local_sq) / (float)head_dim + eps);
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            h_vec[i] = __float2half(__half2float(h_vec[i]) * inv_rms * norm_w[i]);
        }
        __syncthreads();
    }

    int rot_dim = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    int half = rot_dim / 2;
    for (int pair_idx = threadIdx.x; pair_idx < half; pair_idx += blockDim.x) {
        float freq = 1.0f / powf(theta, (float)(2 * pair_idx) / (float)rot_dim);
        float sin_a, cos_a;
        sincosf((float)pos * freq, &sin_a, &cos_a);
        float first = __half2float(h_vec[pair_idx]);
        float second = __half2float(h_vec[pair_idx + half]);
        h_vec[pair_idx] = __float2half(first * cos_a - second * sin_a);
        h_vec[pair_idx + half] = __float2half(first * sin_a + second * cos_a);
    }
}

__global__ void kv_append_f16_kernel(
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    int start_pos,
    const int* __restrict__ dev_pos,
    int kv_dim,
    int total_kv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_kv) {
        int effective_pos = (dev_pos != nullptr) ? *dev_pos : start_pos;
        size_t dst = (size_t)effective_pos * kv_dim + idx;
        k_cache[dst] = k[idx];
        v_cache[dst] = v[idx];
    }
}

__global__ void gqa_attention_f16_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half* __restrict__ out,
    int start_pos,
    const int* __restrict__ dev_pos,
    int kv_dim,
    int heads_per_kv,
    int head_dim
) {
    int q_head = blockIdx.x;
    int tok = blockIdx.y;
    int effective_pos = (dev_pos != nullptr) ? *dev_pos : start_pos;
    int seq_len = effective_pos + tok + 1;
    int kv_offset = (q_head / heads_per_kv) * head_dim;
    size_t q_off = ((size_t)tok * gridDim.x + q_head) * head_dim;
    const __half* q_slice = q + q_off;
    __half* out_slice = out + q_off;
    float scale = rsqrtf((float)head_dim);

    __shared__ float s_scores[MAX_ATTN_SEQ];
    float local_max = -1e30f;

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        const __half* k_slice = k_cache + (size_t)t * kv_dim + kv_offset;
        float dot = 0.0f;
        if ((head_dim & 1) == 0) {
            const __half2* q2 = reinterpret_cast<const __half2*>(q_slice);
            const __half2* k2 = reinterpret_cast<const __half2*>(k_slice);
            int half_dim = head_dim >> 1;
            for (int d = 0; d < half_dim; ++d) {
                float2 qf = __half22float2(q2[d]);
                float2 kf = __half22float2(k2[d]);
                dot += qf.x * kf.x + qf.y * kf.y;
            }
        } else {
            for (int d = 0; d < head_dim; ++d) {
                dot += __half2float(q_slice[d]) * __half2float(k_slice[d]);
            }
        }
        float score = dot * scale;
        s_scores[t] = score;
        local_max = fmaxf(local_max, score);
    }

    float max_score = block_reduce_max(local_max);
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float w = expf(s_scores[t] - max_score);
        s_scores[t] = w;
        local_sum += w;
    }

    float total = block_reduce_sum(local_sum);
    float inv_sum = (total > 0.0f) ? (1.0f / total) : 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            acc += (s_scores[t] * inv_sum) * __half2float(v_cache[(size_t)t * kv_dim + kv_offset + d]);
        }
        out_slice[d] = __float2half(acc);
    }
}

__global__ void swiglu_inplace_f16_kernel(__half* __restrict__ gate, const __half* __restrict__ up, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        gate[idx] = __float2half((g / (1.0f + expf(-g))) * u);
    }
}

__global__ void swiglu_fwht128_inplace_f16_kernel(__half* __restrict__ gate, const __half* __restrict__ up, int total_groups) {
    int group_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (group_idx >= total_groups) {
        return;
    }

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    __half* g_group = gate + (size_t)group_idx * GROUP_SIZE;
    const __half* u_group = up + (size_t)group_idx * GROUP_SIZE;

    __shared__ float s_fwht[4][GROUP_SIZE];
    float* warp_buf = s_fwht[warp_in_block];

    int base = lane * 4;
    uint2 g_raw = *reinterpret_cast<const uint2*>(&g_group[base]);
    uint2 u_raw = *reinterpret_cast<const uint2*>(&u_group[base]);
    const __half2* gh2 = reinterpret_cast<const __half2*>(&g_raw);
    const __half2* uh2 = reinterpret_cast<const __half2*>(&u_raw);
    float2 g0 = __half22float2(gh2[0]);
    float2 g1 = __half22float2(gh2[1]);
    float2 u0 = __half22float2(uh2[0]);
    float2 u1 = __half22float2(uh2[1]);

    warp_buf[base + 0] = (g0.x / (1.0f + expf(-g0.x))) * u0.x;
    warp_buf[base + 1] = (g0.y / (1.0f + expf(-g0.y))) * u0.y;
    warp_buf[base + 2] = (g1.x / (1.0f + expf(-g1.x))) * u1.x;
    warp_buf[base + 3] = (g1.y / (1.0f + expf(-g1.y))) * u1.y;
    __syncwarp();

    fwht128_warp_inplace(warp_buf, lane);

    __half2 out_h2[2];
    out_h2[0] = __float22half2_rn(make_float2(warp_buf[base + 0], warp_buf[base + 1]));
    out_h2[1] = __float22half2_rn(make_float2(warp_buf[base + 2], warp_buf[base + 3]));
    *reinterpret_cast<uint2*>(&g_group[base]) = *reinterpret_cast<const uint2*>(out_h2);
}

__global__ void lm_head_f16_act_f16_kernel(
    const __half* __restrict__ hidden,
    const uint16_t* __restrict__ emb_f16,
    float* __restrict__ logits,
    int vocab_size,
    int hidden_size
) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int v = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (v >= vocab_size) {
        return;
    }

    const uint16_t* row = emb_f16 + (size_t)v * hidden_size;
    float partial = 0.0f;

    if ((hidden_size & 7) == 0 && ((reinterpret_cast<uintptr_t>(row) & 15) == 0)) {
        int chunks = hidden_size >> 3;
        const uint4* row_u4 = reinterpret_cast<const uint4*>(row);
        const uint4* hid_u4 = reinterpret_cast<const uint4*>(hidden);
        for (int i = lane; i < chunks; i += WARP_SIZE) {
            uint4 r8 = row_u4[i];
            uint4 x8 = hid_u4[i];
            const __half2* rh2 = reinterpret_cast<const __half2*>(&r8);
            const __half2* xh2 = reinterpret_cast<const __half2*>(&x8);
            for (int j = 0; j < 4; ++j) {
                float2 rf = __half22float2(rh2[j]);
                float2 xf = __half22float2(xh2[j]);
                partial += rf.x * xf.x + rf.y * xf.y;
            }
        }
    } else {
        for (int i = lane; i < hidden_size; i += WARP_SIZE) {
            __half h = *reinterpret_cast<const __half*>(&row[i]);
            partial += __half2float(h) * __half2float(hidden[i]);
        }
    }

    partial = warp_reduce_sum(partial);
    if (lane == 0) {
        logits[v] = partial;
    }
}

__global__ void lm_head_f32_act_f16_kernel(
    const __half* __restrict__ hidden,
    const float* __restrict__ emb_f32,
    float* __restrict__ logits,
    int vocab_size,
    int hidden_size
) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int v = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (v >= vocab_size) {
        return;
    }

    const float* row = emb_f32 + (size_t)v * hidden_size;
    float partial = 0.0f;

    if ((hidden_size & 7) == 0 && ((reinterpret_cast<uintptr_t>(row) & 15) == 0)) {
        int chunks = hidden_size >> 3;
        const float4* row4 = reinterpret_cast<const float4*>(row);
        const uint4* hid_u4 = reinterpret_cast<const uint4*>(hidden);
        for (int i = lane; i < chunks; i += WARP_SIZE) {
            float4 r0 = row4[i * 2 + 0];
            float4 r1 = row4[i * 2 + 1];
            uint4 x8 = hid_u4[i];
            const __half2* xh2 = reinterpret_cast<const __half2*>(&x8);
            float2 x0 = __half22float2(xh2[0]);
            float2 x1 = __half22float2(xh2[1]);
            float2 x2 = __half22float2(xh2[2]);
            float2 x3 = __half22float2(xh2[3]);
            partial += r0.x * x0.x + r0.y * x0.y + r0.z * x1.x + r0.w * x1.y;
            partial += r1.x * x2.x + r1.y * x2.y + r1.z * x3.x + r1.w * x3.y;
        }
    } else {
        for (int i = lane; i < hidden_size; i += WARP_SIZE) {
            partial += row[i] * __half2float(hidden[i]);
        }
    }

    partial = warp_reduce_sum(partial);
    if (lane == 0) {
        logits[v] = partial;
    }
}

template <int Bits>
__global__ void lm_head_quant_act_f16_kernel(
    const __half* __restrict__ hidden,
    const uint8_t* __restrict__ w_packed,
    const uint16_t* __restrict__ scales,
    float* __restrict__ logits,
    int vocab_size,
    int hidden_size
) {
    extern __shared__ __half s_hidden[];
    const uint4* hid_u4 = reinterpret_cast<const uint4*>(hidden);
    uint4* sh_u4 = reinterpret_cast<uint4*>(s_hidden);
    int chunks_u4 = hidden_size >> 3;
    for (int i = threadIdx.x; i < chunks_u4; i += blockDim.x) {
        sh_u4[i] = hid_u4[i];
    }
    __syncthreads();

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int v = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (v >= vocab_size) {
        return;
    }

    int groups_per_row = hidden_size / GROUP_SIZE;
    int bytes_per_group = (Bits == 4) ? 64 : 128;
    const uint8_t* w_row = w_packed + (size_t)v * groups_per_row * bytes_per_group;
    const uint16_t* s_row = scales + (size_t)v * groups_per_row;
    float total = 0.0f;

    if (Bits == 4) {
        int sub = lane & 3;
        int g_off = lane >> 2;
        for (int g_base = 0; g_base < groups_per_row; g_base += 8) {
            int g = g_base + g_off;
            if (g < groups_per_row) {
                uint4 w16 = *reinterpret_cast<const uint4*>(&w_row[g * 64 + sub * 16]);
                const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&w16);
                const uint4* sh_sub = reinterpret_cast<const uint4*>(&s_hidden[g * GROUP_SIZE + sub * 32]);
                float scale = __half2float(*reinterpret_cast<const __half*>(&s_row[g]));
                float dot = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t w = w32[j];
                    uint4 x_u4 = sh_sub[j];
                    const __half2* xh2 = reinterpret_cast<const __half2*>(&x_u4);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        float2 xf = __half22float2(xh2[k]);
                        uint32_t byte_val = (w >> (k * 8)) & 0xFF;
                        float q0 = static_cast<float>(byte_val & 0x0F) - 8.0f;
                        float q1 = static_cast<float>((byte_val >> 4) & 0x0F) - 8.0f;
                        dot += q0 * xf.x + q1 * xf.y;
                    }
                }
                total += dot * scale;
            }
        }
    } else {
        int sub = lane & 7;
        int g_off = lane >> 3;
        for (int g_base = 0; g_base < groups_per_row; g_base += 4) {
            int g = g_base + g_off;
            if (g < groups_per_row) {
                uint4 w16 = *reinterpret_cast<const uint4*>(&w_row[g * 128 + sub * 16]);
                const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&w16);
                const uint4* sh_sub = reinterpret_cast<const uint4*>(&s_hidden[g * GROUP_SIZE + sub * 16]);
                float scale = __half2float(*reinterpret_cast<const __half*>(&s_row[g]));
                float dot = 0.0f;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    uint4 x_u4 = sh_sub[j];
                    const __half2* xh2 = reinterpret_cast<const __half2*>(&x_u4);
                    #pragma unroll
                    for (int half_idx = 0; half_idx < 2; ++half_idx) {
                        uint32_t w = w32[j * 2 + half_idx];
                        float2 xf0 = __half22float2(xh2[half_idx * 2 + 0]);
                        float2 xf1 = __half22float2(xh2[half_idx * 2 + 1]);
                        int8_t q0 = static_cast<int8_t>(w & 0xFF);
                        int8_t q1 = static_cast<int8_t>((w >> 8) & 0xFF);
                        int8_t q2 = static_cast<int8_t>((w >> 16) & 0xFF);
                        int8_t q3 = static_cast<int8_t>((w >> 24) & 0xFF);
                        dot += (float)q0 * xf0.x + (float)q1 * xf0.y + (float)q2 * xf1.x + (float)q3 * xf1.y;
                    }
                }
                total += dot * scale;
            }
        }
    }

    total = warp_reduce_sum(total);
    if (lane == 0) {
        logits[v] = total;
    }
}

template <bool AddResidual, typename OutT>
static int launch_gemv_f16(
    const uint16_t* dev_x,
    const uint8_t* dev_w,
    const uint16_t* dev_scales,
    OutT* dev_out,
    int m,
    int n,
    int k,
    int bits
) {
    dim3 block(128);
    const __half* hx = reinterpret_cast<const __half*>(dev_x);
    if (m == 1) {
        dim3 grid((n + 3) / 4, 1);
        if (bits == 1) {
            ternary_gemv_f16_kernel<AddResidual, OutT, 1><<<grid, block, 0, g_active_stream>>>(hx, dev_w, dev_scales, dev_out, m, n, k);
        } else {
            ternary_gemv_f16_kernel<AddResidual, OutT, 2><<<grid, block, 0, g_active_stream>>>(hx, dev_w, dev_scales, dev_out, m, n, k);
        }
    } else {
        dim3 grid((n + 3) / 4, (m + GEMM_BM - 1) / GEMM_BM);
        if (bits == 1) {
            ternary_gemm_tiled_kernel<AddResidual, OutT, 1><<<grid, block, 0, g_active_stream>>>(hx, dev_w, dev_scales, dev_out, m, n, k);
        } else {
            ternary_gemm_tiled_kernel<AddResidual, OutT, 2><<<grid, block, 0, g_active_stream>>>(hx, dev_w, dev_scales, dev_out, m, n, k);
        }
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

extern "C" {

int aibox_gpu_is_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

void* aibox_gpu_alloc(size_t bytes) {
    void* dev_ptr = nullptr;
    if (cudaMalloc(&dev_ptr, bytes) != cudaSuccess) {
        return nullptr;
    }
    return dev_ptr;
}

void* aibox_gpu_alloc_copy(const void* host_ptr, size_t bytes) {
    void* dev_ptr = aibox_gpu_alloc(bytes);
    if (dev_ptr == nullptr) {
        return nullptr;
    }
    if (cudaMemcpy(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev_ptr);
        return nullptr;
    }
    return dev_ptr;
}

void aibox_gpu_free(void* dev_ptr) {
    if (dev_ptr != nullptr) {
        cudaFree(dev_ptr);
    }
}

int aibox_gpu_memcpy_h2d(void* dev_dst, const void* host_src, size_t bytes) {
    return (cudaMemcpy(dev_dst, host_src, bytes, cudaMemcpyHostToDevice) == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_memcpy_d2h(void* host_dst, const void* dev_src, size_t bytes) {
    return (cudaMemcpy(host_dst, dev_src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_argmax_f32(const float* dev_logits, int len, uint32_t* host_out_idx) {
    if (len <= 0 || host_out_idx == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_scratch_mutex);
    if (!ensure_scratch_capacity(g_dev_argmax_vals, ARGMAX_BLOCKS * sizeof(float))
        || !ensure_scratch_capacity(g_dev_argmax_idxs, ARGMAX_BLOCKS * sizeof(uint32_t))
        || !ensure_scratch_capacity(g_dev_argmax_out, sizeof(uint32_t))) {
        return -1;
    }
    float* dev_vals = static_cast<float*>(g_dev_argmax_vals.ptr);
    uint32_t* dev_idxs = static_cast<uint32_t*>(g_dev_argmax_idxs.ptr);
    uint32_t* dev_out = static_cast<uint32_t*>(g_dev_argmax_out.ptr);

    argmax_stage1_kernel<<<ARGMAX_BLOCKS, 256>>>(dev_logits, len, dev_vals, dev_idxs);
    argmax_stage2_kernel<<<1, ARGMAX_BLOCKS>>>(dev_vals, dev_idxs, ARGMAX_BLOCKS, dev_out);
    if (cudaGetLastError() != cudaSuccess) {
        return -2;
    }
    if (cudaMemcpy(host_out_idx, dev_out, sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -3;
    }
    return 0;
}

int aibox_gpu_sample_candidates_f32(
    const float* dev_logits,
    int len,
    float margin,
    uint32_t* host_idxs,
    float* host_vals,
    int max_out,
    int* host_count
) {
    if (len <= 0 || max_out <= 0 || host_idxs == nullptr || host_vals == nullptr || host_count == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_scratch_mutex);
    if (!ensure_scratch_capacity(g_dev_argmax_vals, ARGMAX_BLOCKS * sizeof(float))
        || !ensure_scratch_capacity(g_dev_argmax_idxs, ARGMAX_BLOCKS * sizeof(uint32_t))
        || !ensure_scratch_capacity(g_dev_argmax_out, sizeof(uint32_t))
        || !ensure_scratch_capacity(g_dev_cand_idxs, (size_t)max_out * sizeof(uint32_t))
        || !ensure_scratch_capacity(g_dev_cand_vals, (size_t)max_out * sizeof(float))
        || !ensure_scratch_capacity(g_dev_cand_count, sizeof(int))) {
        return -1;
    }
    float* dev_vals = static_cast<float*>(g_dev_argmax_vals.ptr);
    uint32_t* dev_idxs = static_cast<uint32_t*>(g_dev_argmax_idxs.ptr);
    uint32_t* dev_out = static_cast<uint32_t*>(g_dev_argmax_out.ptr);
    uint32_t* dev_cand_idxs = static_cast<uint32_t*>(g_dev_cand_idxs.ptr);
    float* dev_cand_vals = static_cast<float*>(g_dev_cand_vals.ptr);
    int* dev_cand_count = static_cast<int*>(g_dev_cand_count.ptr);

    argmax_stage1_kernel<<<ARGMAX_BLOCKS, 256>>>(dev_logits, len, dev_vals, dev_idxs);
    argmax_stage2_kernel<<<1, ARGMAX_BLOCKS>>>(dev_vals, dev_idxs, ARGMAX_BLOCKS, dev_out);
    if (cudaMemsetAsync(dev_cand_count, 0, sizeof(int)) != cudaSuccess) {
        return -2;
    }
    filter_candidates_kernel<<<ARGMAX_BLOCKS, 256>>>(
        dev_logits, len, dev_vals, margin, dev_cand_idxs, dev_cand_vals, max_out, dev_cand_count
    );
    if (cudaGetLastError() != cudaSuccess) {
        return -2;
    }
    int count = 0;
    if (cudaMemcpy(&count, dev_cand_count, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -3;
    }
    *host_count = count;
    int n_copy = (count < max_out) ? count : max_out;
    if (n_copy > 0) {
        if (cudaMemcpy(host_idxs, dev_cand_idxs, (size_t)n_copy * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(host_vals, dev_cand_vals, (size_t)n_copy * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return -3;
        }
    }
    return 0;
}

int aibox_gpu_matmul_ternary(
    const float* host_x,
    const uint8_t* dev_w,
    const uint16_t* dev_scales,
    float* host_out,
    int m,
    int n,
    int k,
    int use_hadamard
) {
    std::lock_guard<std::mutex> lock(g_scratch_mutex);
    size_t x_bytes = (size_t)m * (size_t)k * sizeof(float);
    size_t out_bytes = (size_t)m * (size_t)n * sizeof(float);
    if (!ensure_scratch_capacity(g_dev_x, x_bytes) || !ensure_scratch_capacity(g_dev_out, out_bytes)) {
        return -1;
    }

    float* dev_x = static_cast<float*>(g_dev_x.ptr);
    float* dev_out = static_cast<float*>(g_dev_out.ptr);
    if (cudaMemcpy(dev_x, host_x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -2;
    }

    if (use_hadamard) {
        int total_groups = m * (k / GROUP_SIZE);
        int blocks = (total_groups + 3) / 4;
        fwht128_rows_kernel<<<blocks, 128>>>(dev_x, total_groups);
        if (cudaGetLastError() != cudaSuccess) {
            return -3;
        }
    }
    dim3 block(128);
    dim3 grid((n + 3) / 4, m);
    ternary_gemv_kernel<false><<<grid, block>>>(dev_x, dev_w, dev_scales, dev_out, m, n, k);
    if (cudaGetLastError() != cudaSuccess) {
        return -3;
    }
    if (cudaMemcpy(host_out, dev_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -4;
    }
    return 0;
}

int aibox_gpu_fwht128_f16(uint16_t* dev_x, int m, int k) {
    int total_groups = m * (k / GROUP_SIZE);
    int blocks = (total_groups + 3) / 4;
    fwht128_rows_f16_kernel<<<blocks, 128, 0, g_active_stream>>>(reinterpret_cast<__half*>(dev_x), total_groups);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_matmul_ternary_f16_dev(
    const uint16_t* dev_x,
    const uint8_t* dev_w,
    const uint16_t* dev_scales,
    uint16_t* dev_out,
    int m,
    int n,
    int k,
    int bits
) {
    return launch_gemv_f16<false, __half>(
        dev_x, dev_w, dev_scales, reinterpret_cast<__half*>(dev_out), m, n, k, bits
    );
}

int aibox_gpu_matmul_ternary_f16_add_dev(
    const uint16_t* dev_x,
    const uint8_t* dev_w,
    const uint16_t* dev_scales,
    uint16_t* dev_out,
    int m,
    int n,
    int k,
    int bits
) {
    return launch_gemv_f16<true, __half>(
        dev_x, dev_w, dev_scales, reinterpret_cast<__half*>(dev_out), m, n, k, bits
    );
}

int aibox_gpu_matmul_ternary_f16_to_f32_dev(
    const uint16_t* dev_x,
    const uint8_t* dev_w,
    const uint16_t* dev_scales,
    float* dev_out,
    int m,
    int n,
    int k,
    int bits
) {
    return launch_gemv_f16<false, float>(dev_x, dev_w, dev_scales, dev_out, m, n, k, bits);
}

int aibox_gpu_rms_norm_f16(
    const uint16_t* dev_x,
    const float* dev_w,
    uint16_t* dev_out,
    int m,
    int dim,
    float eps
) {
    rms_norm_f16_kernel<<<m, 256, 0, g_active_stream>>>(
        reinterpret_cast<const __half*>(dev_x),
        dev_w,
        reinterpret_cast<__half*>(dev_out),
        dim,
        eps
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_rms_norm_fwht128_f16(
    const uint16_t* dev_x,
    const float* dev_w,
    uint16_t* dev_out,
    int m,
    int dim,
    float eps
) {
    if (dim <= MAX_FUSED_RMS_DIM && (dim % GROUP_SIZE) == 0 && (dim & 7) == 0) {
        rms_norm_fwht128_f16_kernel<<<m, 256, 0, g_active_stream>>>(
            reinterpret_cast<const __half*>(dev_x),
            dev_w,
            reinterpret_cast<__half*>(dev_out),
            dim,
            eps
        );
        return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
    }
    if (aibox_gpu_rms_norm_f16(dev_x, dev_w, dev_out, m, dim, eps) != 0) {
        return -1;
    }
    return aibox_gpu_fwht128_f16(dev_out, m, dim);
}

int aibox_gpu_rms_norm_heads_rope_f16(
    uint16_t* dev_vec,
    const float* dev_norm_w,
    int m,
    int num_heads,
    int head_dim,
    int rope_dim,
    int start_pos,
    float theta,
    float eps
) {
    dim3 grid(num_heads, m);
    const int* dev_pos = (g_active_stream != nullptr) ? g_dev_pos : nullptr;
    rms_norm_heads_rope_f16_kernel<<<grid, 128, 0, g_active_stream>>>(
        reinterpret_cast<__half*>(dev_vec),
        dev_norm_w,
        head_dim,
        rope_dim,
        start_pos,
        dev_pos,
        theta,
        eps
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_kv_append_gqa_f16(
    const uint16_t* dev_q,
    const uint16_t* dev_k,
    const uint16_t* dev_v,
    uint16_t* dev_k_cache,
    uint16_t* dev_v_cache,
    uint16_t* dev_out,
    int start_pos,
    int m,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int max_seq = start_pos + m;
    if (max_seq > MAX_ATTN_SEQ) {
        return -2;
    }
    const int* dev_pos = (g_active_stream != nullptr) ? g_dev_pos : nullptr;
    int kv_dim = num_kv_heads * head_dim;
    int total_kv = m * kv_dim;
    int kv_blocks = (total_kv + 255) / 256;
    kv_append_f16_kernel<<<kv_blocks, 256, 0, g_active_stream>>>(
        reinterpret_cast<const __half*>(dev_k),
        reinterpret_cast<const __half*>(dev_v),
        reinterpret_cast<__half*>(dev_k_cache),
        reinterpret_cast<__half*>(dev_v_cache),
        start_pos,
        dev_pos,
        kv_dim,
        total_kv
    );

    int heads_per_kv = num_q_heads / num_kv_heads;
    dim3 attn_grid(num_q_heads, m);
    gqa_attention_f16_kernel<<<attn_grid, 128, 0, g_active_stream>>>(
        reinterpret_cast<const __half*>(dev_q),
        reinterpret_cast<const __half*>(dev_k_cache),
        reinterpret_cast<const __half*>(dev_v_cache),
        reinterpret_cast<__half*>(dev_out),
        start_pos,
        dev_pos,
        kv_dim,
        heads_per_kv,
        head_dim
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_swiglu_inplace_f16(uint16_t* dev_gate, const uint16_t* dev_up, int len) {
    int blocks = (len + 255) / 256;
    swiglu_inplace_f16_kernel<<<blocks, 256, 0, g_active_stream>>>(
        reinterpret_cast<__half*>(dev_gate),
        reinterpret_cast<const __half*>(dev_up),
        len
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_swiglu_fwht128_inplace_f16(uint16_t* dev_gate, const uint16_t* dev_up, int len) {
    if ((len % GROUP_SIZE) == 0) {
        int total_groups = len / GROUP_SIZE;
        int blocks = (total_groups + 3) / 4;
        swiglu_fwht128_inplace_f16_kernel<<<blocks, 128, 0, g_active_stream>>>(
            reinterpret_cast<__half*>(dev_gate),
            reinterpret_cast<const __half*>(dev_up),
            total_groups
        );
        return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
    }
    if (aibox_gpu_swiglu_inplace_f16(dev_gate, dev_up, len) != 0) {
        return -1;
    }
    return aibox_gpu_fwht128_f16(dev_gate, 1, len);
}

int aibox_gpu_lm_head_f16_act_f16(
    const uint16_t* dev_hidden,
    const uint16_t* dev_emb_f16,
    float* dev_logits,
    int vocab_size,
    int hidden_size
) {
    int blocks = (vocab_size + 3) / 4;
    lm_head_f16_act_f16_kernel<<<blocks, 128, 0, g_active_stream>>>(
        reinterpret_cast<const __half*>(dev_hidden),
        dev_emb_f16,
        dev_logits,
        vocab_size,
        hidden_size
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_matmul_ternary_swiglu_f16_dev(
    const uint16_t* dev_x,
    const uint8_t* dev_w_gate,
    const uint16_t* dev_scales_gate,
    const uint8_t* dev_w_up,
    const uint16_t* dev_scales_up,
    uint16_t* dev_out,
    int n,
    int k,
    int bits
) {
    if ((bits != 1 && bits != 2) || ((k / GROUP_SIZE) % 4) != 0) {
        return -1;
    }
    dim3 block(128);
    dim3 grid((n + 3) / 4, 1);
    const __half* hx = reinterpret_cast<const __half*>(dev_x);
    __half* hout = reinterpret_cast<__half*>(dev_out);
    if (bits == 1) {
        ternary_gemv_swiglu_f16_kernel<1><<<grid, block, 0, g_active_stream>>>(
            hx, dev_w_gate, dev_scales_gate, dev_w_up, dev_scales_up, hout, n, k
        );
    } else {
        ternary_gemv_swiglu_f16_kernel<2><<<grid, block, 0, g_active_stream>>>(
            hx, dev_w_gate, dev_scales_gate, dev_w_up, dev_scales_up, hout, n, k
        );
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_lm_head_f32_act_f16(
    const uint16_t* dev_hidden,
    const float* dev_emb_f32,
    float* dev_logits,
    int vocab_size,
    int hidden_size
) {
    int blocks = (vocab_size + 3) / 4;
    lm_head_f32_act_f16_kernel<<<blocks, 128, 0, g_active_stream>>>(
        reinterpret_cast<const __half*>(dev_hidden),
        dev_emb_f32,
        dev_logits,
        vocab_size,
        hidden_size
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_lm_head_quant_act_f16(
    const uint16_t* dev_hidden,
    const uint8_t* dev_w,
    const uint16_t* dev_scales,
    float* dev_logits,
    int vocab_size,
    int hidden_size,
    int bits
) {
    if (bits != 4 && bits != 8) {
        return -1;
    }
    int blocks = (vocab_size + 3) / 4;
    size_t smem_bytes = (size_t)hidden_size * sizeof(__half);
    const __half* hx = reinterpret_cast<const __half*>(dev_hidden);
    if (bits == 4) {
        lm_head_quant_act_f16_kernel<4><<<blocks, 128, smem_bytes, g_active_stream>>>(
            hx, dev_w, dev_scales, dev_logits, vocab_size, hidden_size
        );
    } else {
        lm_head_quant_act_f16_kernel<8><<<blocks, 128, smem_bytes, g_active_stream>>>(
            hx, dev_w, dev_scales, dev_logits, vocab_size, hidden_size
        );
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_graph_begin_capture(void) {
    if (g_graph_stream == nullptr) {
        if (cudaStreamCreateWithFlags(&g_graph_stream, cudaStreamNonBlocking) != cudaSuccess) {
            return -1;
        }
    }
    if (g_dev_pos == nullptr) {
        if (cudaMalloc(&g_dev_pos, sizeof(int)) != cudaSuccess) {
            return -2;
        }
    }
    if (g_decode_graph_exec != nullptr) {
        cudaGraphExecDestroy(g_decode_graph_exec);
        g_decode_graph_exec = nullptr;
    }
    if (g_decode_graph != nullptr) {
        cudaGraphDestroy(g_decode_graph);
        g_decode_graph = nullptr;
    }
    if (cudaStreamBeginCapture(g_graph_stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        return -3;
    }
    g_active_stream = g_graph_stream;
    return 0;
}

int aibox_gpu_graph_end_capture(void) {
    g_active_stream = nullptr;
    if (cudaStreamEndCapture(g_graph_stream, &g_decode_graph) != cudaSuccess) {
        return -1;
    }
    if (cudaGraphInstantiate(&g_decode_graph_exec, g_decode_graph, nullptr, nullptr, 0) != cudaSuccess) {
        cudaGraphDestroy(g_decode_graph);
        g_decode_graph = nullptr;
        return -2;
    }
    return 0;
}

int aibox_gpu_graph_launch(
    uint16_t* dev_hidden,
    const uint16_t* host_hidden_f16,
    int hidden_len,
    int pos
) {
    if (g_decode_graph_exec == nullptr || g_graph_stream == nullptr || g_dev_pos == nullptr) {
        return -1;
    }
    if (pos + 1 > MAX_ATTN_SEQ) {
        return -2;
    }
    if (cudaMemcpyAsync(
            dev_hidden,
            host_hidden_f16,
            (size_t)hidden_len * sizeof(uint16_t),
            cudaMemcpyHostToDevice,
            g_graph_stream
        ) != cudaSuccess) {
        return -3;
    }
    if (cudaMemcpyAsync(g_dev_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, g_graph_stream) != cudaSuccess) {
        return -4;
    }
    if (cudaGraphLaunch(g_decode_graph_exec, g_graph_stream) != cudaSuccess) {
        return -5;
    }
    return (cudaStreamSynchronize(g_graph_stream) == cudaSuccess) ? 0 : -6;
}

__global__ void deinterleave_q_gate_f16_kernel(
    const __half* __restrict__ q_full,
    __half* __restrict__ q_out,
    __half* __restrict__ gate_out,
    int total_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_heads * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int d = idx % head_dim;
    q_out[idx] = __float2half(__half2float(q_full[h * 2 * head_dim + d]));
    gate_out[idx] = __float2half(__half2float(q_full[h * 2 * head_dim + head_dim + d]));
}

__global__ void mul_sigmoid_inplace_f16_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ gate,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    float o = __half2float(out[idx]);
    float g = __half2float(gate[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    out[idx] = __float2half(o * sig);
}

__global__ void gdn_conv_silu_f16_kernel(
    const __half* __restrict__ qkv_in,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_state,
    __half* __restrict__ qkv_out,
    int conv_channels,
    int kernel
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_channels) return;
    int hist = kernel - 1;
    const float* w = conv_w + ch * kernel;
    float* st = conv_state + ch * hist;
    float acc = 0.0f;
    for (int k = 0; k < hist; ++k) {
        acc += st[k] * w[k];
    }
    float x = __half2float(qkv_in[ch]);
    acc += x * w[hist];
    for (int k = 0; k + 1 < hist; ++k) {
        st[k] = st[k + 1];
    }
    if (hist > 0) {
        st[hist - 1] = x;
    }
    float silu_acc = acc / (1.0f + expf(-acc));
    qkv_out[ch] = __float2half(silu_acc);
}

__global__ void gdn_step_f16_kernel(
    const __half* __restrict__ qkv,
    const __half* __restrict__ alpha,
    const __half* __restrict__ beta_raw,
    const float* __restrict__ ssm_a,
    const float* __restrict__ ssm_dt_bias,
    const float* __restrict__ ssm_norm,
    const __half* __restrict__ z,
    __half* ssm_state,
    __half* __restrict__ out,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    float eps
) {
    int hv = blockIdx.x;
    int j = threadIdx.x;
    if (j >= head_dim) return;

    int hk = hv / (num_v_heads / num_k_heads);
    const __half* q_ptr = qkv + hk * head_dim;
    const __half* k_ptr = qkv + num_k_heads * head_dim + hk * head_dim;
    const __half* v_ptr = qkv + 2 * num_k_heads * head_dim + hv * head_dim;

    float q_j = __half2float(q_ptr[j]);
    float k_j = __half2float(k_ptr[j]);
    float q_sq = q_j * q_j;
    float k_sq = k_j * k_j;
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, offset);
        k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, offset);
    }

    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float s_red_q[4];
    __shared__ float s_red_k[4];
    __shared__ float s_inv_q;
    __shared__ float s_inv_k;
    __shared__ float s_decay;
    __shared__ float s_beta;
    __shared__ float s_inv_rms;

    int warp_id = j >> 5;
    int lane = j & 31;
    if (lane == 0) {
        s_red_q[warp_id] = q_sq;
        s_red_k[warp_id] = k_sq;
    }
    __syncthreads();

    if (j == 0) {
        float tot_q = s_red_q[0] + s_red_q[1] + s_red_q[2] + s_red_q[3];
        float tot_k = s_red_k[0] + s_red_k[1] + s_red_k[2] + s_red_k[3];
        s_inv_q = rsqrtf(tot_q + eps) * rsqrtf((float)head_dim);
        s_inv_k = rsqrtf(tot_k + eps);

        float a_val = __half2float(alpha[hv]) + ssm_dt_bias[hv];
        float sp = (a_val > 20.0f) ? a_val : log1pf(expf(a_val));
        s_decay = expf(ssm_a[hv] * sp);
        s_beta = 1.0f / (1.0f + expf(-__half2float(beta_raw[hv])));
    }
    __syncthreads();

    s_q[j] = q_j * s_inv_q;
    s_k[j] = k_j * s_inv_k;
    __syncthreads();

    __half* state_h = ssm_state + (size_t)hv * head_dim * head_dim;
    float decay = s_decay;
    float beta = s_beta;
    float kv_mem = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float s_ij = __half2float(state_h[i * head_dim + j]) * decay;
        kv_mem += s_ij * s_k[i];
    }
    float v_j = __half2float(v_ptr[j]);
    float delta = (v_j - kv_mem) * beta;
    float o_j = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float s_ij = __half2float(state_h[i * head_dim + j]) * decay + s_k[i] * delta;
        state_h[i * head_dim + j] = __float2half(s_ij);
        o_j += s_ij * s_q[i];
    }

    float o_sq = o_j * o_j;
    for (int offset = 16; offset > 0; offset >>= 1) {
        o_sq += __shfl_down_sync(0xFFFFFFFF, o_sq, offset);
    }
    if (lane == 0) {
        s_red_q[warp_id] = o_sq;
    }
    __syncthreads();
    if (j == 0) {
        float tot_o = s_red_q[0] + s_red_q[1] + s_red_q[2] + s_red_q[3];
        s_inv_rms = rsqrtf(tot_o / (float)head_dim + eps);
    }
    __syncthreads();

    float z_j = __half2float(z[hv * head_dim + j]);
    float silu_z = z_j / (1.0f + expf(-z_j));
    out[hv * head_dim + j] = __float2half(o_j * s_inv_rms * ssm_norm[j] * silu_z);
}

int aibox_gpu_deinterleave_q_gate_f16(
    const uint16_t* dev_q_full,
    uint16_t* dev_q_out,
    uint16_t* dev_gate_out,
    int m,
    int num_heads,
    int head_dim
) {
    int total = m * num_heads * head_dim;
    int blocks = (total + 255) / 256;
    deinterleave_q_gate_f16_kernel<<<blocks, 256, 0, g_active_stream>>>(
        reinterpret_cast<const __half*>(dev_q_full),
        reinterpret_cast<__half*>(dev_q_out),
        reinterpret_cast<__half*>(dev_gate_out),
        m * num_heads,
        head_dim
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_mul_sigmoid_inplace_f16(
    uint16_t* dev_out,
    const uint16_t* dev_gate,
    int len
) {
    int blocks = (len + 255) / 256;
    mul_sigmoid_inplace_f16_kernel<<<blocks, 256, 0, g_active_stream>>>(
        reinterpret_cast<__half*>(dev_out),
        reinterpret_cast<const __half*>(dev_gate),
        len
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_gdn_conv_silu_f16(
    const uint16_t* dev_qkv_in,
    const float* dev_conv_w,
    float* dev_conv_state,
    uint16_t* dev_qkv_out,
    int conv_channels,
    int kernel
) {
    int blocks = (conv_channels + 255) / 256;
    gdn_conv_silu_f16_kernel<<<blocks, 256, 0, g_active_stream>>>(
        reinterpret_cast<const __half*>(dev_qkv_in),
        dev_conv_w,
        dev_conv_state,
        reinterpret_cast<__half*>(dev_qkv_out),
        conv_channels,
        kernel
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_gdn_step_f16(
    const uint16_t* dev_qkv,
    const uint16_t* dev_alpha,
    const uint16_t* dev_beta,
    const float* dev_ssm_a,
    const float* dev_ssm_dt_bias,
    const float* dev_ssm_norm,
    const uint16_t* dev_z,
    uint16_t* dev_ssm_state,
    uint16_t* dev_out,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    float eps
) {
    if (head_dim > 128) {
        return -1;
    }
    gdn_step_f16_kernel<<<num_v_heads, 128, 0, g_active_stream>>>(
        reinterpret_cast<const __half*>(dev_qkv),
        reinterpret_cast<const __half*>(dev_alpha),
        reinterpret_cast<const __half*>(dev_beta),
        dev_ssm_a,
        dev_ssm_dt_bias,
        dev_ssm_norm,
        reinterpret_cast<const __half*>(dev_z),
        reinterpret_cast<__half*>(dev_ssm_state),
        reinterpret_cast<__half*>(dev_out),
        num_k_heads,
        num_v_heads,
        head_dim,
        eps
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__global__ void matmul_dense_f16_dev_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    __half* __restrict__ out,
    int n,
    int k
) {
    int warp_global = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int token = blockIdx.y;
    if (warp_global >= n) return;
    int lane = threadIdx.x & 31;
    const __half* x_row = x + token * k;
    const __half* w_row = w + warp_global * k;
    float sum = 0.0f;
    for (int col = lane; col < k; col += 32) {
        sum += __half2float(x_row[col]) * __half2float(w_row[col]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane == 0) {
        out[token * n + warp_global] = __float2half(sum);
    }
}

int aibox_gpu_matmul_dense_f16_dev(
    const uint16_t* dev_x,
    const uint16_t* dev_w,
    uint16_t* dev_out,
    int m,
    int n,
    int k
) {
    dim3 grid((n + 3) / 4, m);
    matmul_dense_f16_dev_kernel<<<grid, 128, 0, g_active_stream>>>(
        reinterpret_cast<const __half*>(dev_x),
        reinterpret_cast<const __half*>(dev_w),
        reinterpret_cast<__half*>(dev_out),
        n,
        k
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}
