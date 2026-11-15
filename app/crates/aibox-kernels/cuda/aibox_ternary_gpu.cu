#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mutex>
#include <stdint.h>
#include <stdlib.h>

#define GROUP_SIZE 128
#define PACKED_BYTES_PER_GROUP 32
#define WARP_SIZE 32
#define MAX_ATTN_SEQ 4096

namespace {

struct ScratchBuffer {
    void* ptr = nullptr;
    size_t capacity = 0;
};

std::mutex g_scratch_mutex;
ScratchBuffer g_dev_x;
ScratchBuffer g_dev_out;

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
    warp_buf[base + 0] = x_group[base + 0];
    warp_buf[base + 1] = x_group[base + 1];
    warp_buf[base + 2] = x_group[base + 2];
    warp_buf[base + 3] = x_group[base + 3];
    __syncwarp();

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
    x_group[base + 0] = warp_buf[base + 0] * inv_sqrt_128;
    x_group[base + 1] = warp_buf[base + 1] * inv_sqrt_128;
    x_group[base + 2] = warp_buf[base + 2] * inv_sqrt_128;
    x_group[base + 3] = warp_buf[base + 3] * inv_sqrt_128;
}

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

    float total = 0.0f;
    for (int g = 0; g < groups_per_row; ++g) {
        int base = g * GROUP_SIZE + lane * 4;
        uint8_t packed = w_row[g * PACKED_BYTES_PER_GROUP + lane];
        float partial = decode_2bit_weight(packed & 0x03) * x_row[base + 0]
                      + decode_2bit_weight((packed >> 2) & 0x03) * x_row[base + 1]
                      + decode_2bit_weight((packed >> 4) & 0x03) * x_row[base + 2]
                      + decode_2bit_weight((packed >> 6) & 0x03) * x_row[base + 3];

        partial = warp_reduce_sum(partial);
        if (lane == 0) {
            __half h_scale = *reinterpret_cast<const __half*>(&s_row[g]);
            total += partial * __half2float(h_scale);
        }
    }

    if (lane == 0) {
        out[(size_t)row_m * n + row_n] = total;
    }
}

__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int dim,
    float eps
) {
    int row = blockIdx.x;
    const float* x_row = x + (size_t)row * dim;
    float* out_row = out + (size_t)row * dim;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v * v;
    }

    local_sum = warp_reduce_sum(local_sum);
    __shared__ float s_warp_sums[8];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (lane == 0) {
        s_warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        float total_sq = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int w = 0; w < num_warps; ++w) {
            total_sq += s_warp_sums[w];
        }
        s_inv_rms = rsqrtf(total_sq / (float)dim + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

__global__ void rms_norm_heads_rope_kernel(
    float* __restrict__ vec,
    const float* __restrict__ norm_w,
    int head_dim,
    int pos,
    float theta,
    float eps
) {
    int head = blockIdx.x;
    float* h_vec = vec + (size_t)head * head_dim;

    if (norm_w != nullptr) {
        float local_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = h_vec[i];
            local_sq += v * v;
        }
        local_sq = warp_reduce_sum(local_sq);
        __shared__ float s_warp[8];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane = threadIdx.x & (WARP_SIZE - 1);
        if (lane == 0) {
            s_warp[warp_id] = local_sq;
        }
        __syncthreads();

        __shared__ float s_inv_rms;
        if (threadIdx.x == 0) {
            float total_sq = 0.0f;
            int num_warps = blockDim.x / WARP_SIZE;
            for (int w = 0; w < num_warps; ++w) {
                total_sq += s_warp[w];
            }
            s_inv_rms = rsqrtf(total_sq / (float)head_dim + eps);
        }
        __syncthreads();

        float inv_rms = s_inv_rms;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            h_vec[i] = h_vec[i] * inv_rms * norm_w[i];
        }
        __syncthreads();
    }

    int half = head_dim / 2;
    for (int pair_idx = threadIdx.x; pair_idx < half; pair_idx += blockDim.x) {
        float freq = 1.0f / powf(theta, (float)(2 * pair_idx) / (float)head_dim);
        float sin_a, cos_a;
        sincosf((float)pos * freq, &sin_a, &cos_a);
        float first = h_vec[pair_idx];
        float second = h_vec[pair_idx + half];
        h_vec[pair_idx] = first * cos_a - second * sin_a;
        h_vec[pair_idx + half] = first * sin_a + second * cos_a;
    }
}

__global__ void kv_append_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    int pos,
    int kv_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < kv_dim) {
        size_t dst = (size_t)pos * kv_dim + idx;
        k_cache[dst] = k[idx];
        v_cache[dst] = v[idx];
    }
}

__global__ void gqa_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int seq_len,
    int kv_dim,
    int heads_per_kv,
    int head_dim
) {
    int q_head = blockIdx.x;
    int kv_offset = (q_head / heads_per_kv) * head_dim;
    const float* q_slice = q + (size_t)q_head * head_dim;
    float* out_slice = out + (size_t)q_head * head_dim;
    float scale = rsqrtf((float)head_dim);

    __shared__ float s_scores[MAX_ATTN_SEQ];
    float local_max = -1e30f;

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        const float* k_slice = k_cache + (size_t)t * kv_dim + kv_offset;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_slice[d] * k_slice[d];
        }
        float score = dot * scale;
        s_scores[t] = score;
        local_max = fmaxf(local_max, score);
    }

    local_max = warp_reduce_max(local_max);
    __shared__ float s_reduce[8];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (lane == 0) {
        s_reduce[warp_id] = local_max;
    }
    __syncthreads();

    __shared__ float s_max_score;
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int w = 0; w < num_warps; ++w) {
            m = fmaxf(m, s_reduce[w]);
        }
        s_max_score = m;
    }
    __syncthreads();

    float max_score = s_max_score;
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float w = expf(s_scores[t] - max_score);
        s_scores[t] = w;
        local_sum += w;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        s_reduce[warp_id] = local_sum;
    }
    __syncthreads();

    __shared__ float s_inv_sum;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int w = 0; w < num_warps; ++w) {
            total += s_reduce[w];
        }
        s_inv_sum = (total > 0.0f) ? (1.0f / total) : 0.0f;
    }
    __syncthreads();

    float inv_sum = s_inv_sum;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            acc += (s_scores[t] * inv_sum) * v_cache[(size_t)t * kv_dim + kv_offset + d];
        }
        out_slice[d] = acc;
    }
}

__global__ void swiglu_inplace_kernel(float* __restrict__ gate, const float* __restrict__ up, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float g = gate[idx];
        gate[idx] = (g / (1.0f + expf(-g))) * up[idx];
    }
}

__global__ void add_inplace_kernel(float* __restrict__ target, const float* __restrict__ delta, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        target[idx] += delta[idx];
    }
}

__global__ void lm_head_f16_kernel(
    const float* __restrict__ hidden,
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
    for (int i = lane; i < hidden_size; i += WARP_SIZE) {
        __half h = *reinterpret_cast<const __half*>(&row[i]);
        partial += __half2float(h) * hidden[i];
    }

    partial = warp_reduce_sum(partial);
    if (lane == 0) {
        logits[v] = partial;
    }
}

__global__ void lm_head_f32_kernel(
    const float* __restrict__ hidden,
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
    for (int i = lane; i < hidden_size; i += WARP_SIZE) {
        partial += row[i] * hidden[i];
    }

    partial = warp_reduce_sum(partial);
    if (lane == 0) {
        logits[v] = partial;
    }
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

int aibox_gpu_fwht128(float* dev_x, int m, int k) {
    int total_groups = m * (k / GROUP_SIZE);
    int blocks = (total_groups + 3) / 4;
    fwht128_rows_kernel<<<blocks, 128>>>(dev_x, total_groups);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_matmul_ternary_dev(
    const float* dev_x,
    const uint8_t* dev_w,
    const uint16_t* dev_scales,
    float* dev_out,
    int m,
    int n,
    int k
) {
    dim3 block(128);
    dim3 grid((n + 3) / 4, m);
    ternary_gemv_kernel<<<grid, block>>>(dev_x, dev_w, dev_scales, dev_out, m, n, k);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_rms_norm(
    const float* dev_x,
    const float* dev_w,
    float* dev_out,
    int m,
    int dim,
    float eps
) {
    rms_norm_kernel<<<m, 256>>>(dev_x, dev_w, dev_out, dim, eps);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_rms_norm_heads_rope(
    float* dev_vec,
    const float* dev_norm_w,
    int num_heads,
    int head_dim,
    int pos,
    float theta,
    float eps
) {
    rms_norm_heads_rope_kernel<<<num_heads, 128>>>(dev_vec, dev_norm_w, head_dim, pos, theta, eps);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_kv_append_gqa(
    const float* dev_q,
    const float* dev_k,
    const float* dev_v,
    float* dev_k_cache,
    float* dev_v_cache,
    float* dev_out,
    int pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int seq_len = pos + 1;
    if (seq_len > MAX_ATTN_SEQ) {
        return -2;
    }
    int kv_dim = num_kv_heads * head_dim;
    int kv_blocks = (kv_dim + 255) / 256;
    kv_append_kernel<<<kv_blocks, 256>>>(dev_k, dev_v, dev_k_cache, dev_v_cache, pos, kv_dim);

    int heads_per_kv = num_q_heads / num_kv_heads;
    gqa_attention_kernel<<<num_q_heads, 128>>>(
        dev_q, dev_k_cache, dev_v_cache, dev_out, seq_len, kv_dim, heads_per_kv, head_dim
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_swiglu_inplace(float* dev_gate, const float* dev_up, int len) {
    int blocks = (len + 255) / 256;
    swiglu_inplace_kernel<<<blocks, 256>>>(dev_gate, dev_up, len);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_add_inplace(float* dev_target, const float* dev_delta, int len) {
    int blocks = (len + 255) / 256;
    add_inplace_kernel<<<blocks, 256>>>(dev_target, dev_delta, len);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_lm_head_f16(
    const float* dev_hidden,
    const uint16_t* dev_emb_f16,
    float* dev_logits,
    int vocab_size,
    int hidden_size
) {
    int blocks = (vocab_size + 3) / 4;
    lm_head_f16_kernel<<<blocks, 128>>>(dev_hidden, dev_emb_f16, dev_logits, vocab_size, hidden_size);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int aibox_gpu_lm_head_f32(
    const float* dev_hidden,
    const float* dev_emb_f32,
    float* dev_logits,
    int vocab_size,
    int hidden_size
) {
    int blocks = (vocab_size + 3) / 4;
    lm_head_f32_kernel<<<blocks, 128>>>(dev_hidden, dev_emb_f32, dev_logits, vocab_size, hidden_size);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
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

    if (use_hadamard && aibox_gpu_fwht128(dev_x, m, k) != 0) {
        return -3;
    }
    if (aibox_gpu_matmul_ternary_dev(dev_x, dev_w, dev_scales, dev_out, m, n, k) != 0) {
        return -3;
    }
    if (cudaMemcpy(host_out, dev_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -4;
    }
    return 0;
}

}
