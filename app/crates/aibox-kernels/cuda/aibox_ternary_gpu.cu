#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>
#include <stdint.h>
#include <stdlib.h>

#define GROUP_SIZE 128
#define PACKED_BYTES_PER_GROUP 32
#define WARP_SIZE 32

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

        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }
        if (lane == 0) {
            __half h_scale = *reinterpret_cast<const __half*>(&s_row[g]);
            total += partial * __half2float(h_scale);
        }
    }

    if (lane == 0) {
        out[(size_t)row_m * n + row_n] = total;
    }
}

extern "C" {

int aibox_gpu_is_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

void* aibox_gpu_alloc_copy(const void* host_ptr, size_t bytes) {
    void* dev_ptr = nullptr;
    if (cudaMalloc(&dev_ptr, bytes) != cudaSuccess) {
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
        int fwht_blocks = (total_groups + 3) / 4;
        fwht128_rows_kernel<<<fwht_blocks, 128>>>(dev_x, total_groups);
    }

    dim3 block(128);
    dim3 grid((n + 3) / 4, m);
    ternary_gemv_kernel<<<grid, block>>>(dev_x, dev_w, dev_scales, dev_out, m, n, k);

    if (cudaGetLastError() != cudaSuccess) {
        return -3;
    }
    if (cudaMemcpy(host_out, dev_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -4;
    }
    return 0;
}

}
