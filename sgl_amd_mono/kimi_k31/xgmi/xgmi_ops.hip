// SPDX-License-Identifier: Apache-2.0
// AMD Iris-style XGMI fused collective & compute kernels for Kimi-K3.1 on gfx950 (8x MI355X).
//
// Operates inside one <= 0.4 GiB uncached HIP IPC symmetric peer heap (214 MiB/GPU):
//   1. xgmi_lamport_allgather       (Decode 1..192 rows: 1/8 latent_input_gather, zero-copy dual split)
//   2. xgmi_deferred_topk_lamport_ar (Decode 1..192 rows: deferred top-k + routed AR + shared RS/AR)
//   3. xgmi_moetail_column_gather   (Decode 1..192 rows: 1/8 up_slice + shared + prefix -> full stream)
//   4. xgmi_fused_rmsnorm_bf16      (Decode 1..192 rows: 1-launch wave64 RMSNorm for latent [M, 3584])
//   5. xgmi_rs_chunk_32cu / push / reduce (Prefill 200..8192 rows: 32-CU RS overlapped with 224-CU o_proj)
//   6. xgmi_fp4_pack_allgather      (Prefill 200..8192 rows: fused E2M1 FP4 / FP8 + scale + topk AllGather)
//   7. xgmi_row_allgather           (Prefill 200..8192 rows: 2112-dim qkv_a AG & chunked 7168 Attn AG)

#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#ifndef C10_HIP_KERNEL_LAUNCH_CHECK
#define C10_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

namespace {

using bf16_t = __bf16;
using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));

constexpr int TP = 8;

constexpr int DECODE_BLOCKS = 128;
constexpr int DECODE_THREADS = 256;

constexpr int RS_BLOCKS = 64;
constexpr int RS_THREADS = 512;
constexpr int RS_RING_SLOTS = 4;

constexpr int AG_BLOCKS = 64;
constexpr int AG_THREADS = 512;

struct PeerTable {
    uint8_t *base[TP];
};

static inline PeerTable make_peers(const at::Tensor &peers_cpu) {
    TORCH_CHECK(peers_cpu.device().is_cpu() && peers_cpu.dtype() == at::kLong &&
                    peers_cpu.numel() == TP && peers_cpu.is_contiguous(),
                "peers_cpu must be contiguous CPU int64[8]");
    PeerTable pt;
    const int64_t *raw = peers_cpu.data_ptr<int64_t>();
    for (int r = 0; r < TP; ++r) {
        pt.base[r] = reinterpret_cast<uint8_t *>(raw[r]);
    }
    return pt;
}

__device__ inline void store_nt16(uint32x4 *ptr, uint32x4 val) {
    __builtin_nontemporal_store(val, ptr);
}

// Pipelined bulk nontemporal load (compiler schedules full MLP across waves/peers!)
__device__ inline uint32x4 load_nt16_bulk(const uint32x4 *ptr) {
    return __builtin_nontemporal_load(ptr);
}

// Synchronized system-scope poll loads (for Lamport tags and arrival flags)
__device__ inline uint32x4 load_nt16_sync(const uint32x4 *ptr) {
    uint32x4 val;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_waitcnt vmcnt(0)"
                 : "=&v"(val) : "v"(ptr) : "memory");
    return val;
}

__device__ inline void store_nt32(uint32_t *ptr, uint32_t val) {
    asm volatile("global_store_dword %0, %1, off sc0 sc1 nt\n\ts_nop 1"
                 ::"v"(ptr), "v"(val) : "memory");
}

__device__ inline uint32_t load_nt32_sync(const uint32_t *ptr) {
    uint32_t val;
    asm volatile("global_load_dword %0, %1, off sc0 sc1 nt\n\ts_waitcnt vmcnt(0)"
                 : "=&v"(val) : "v"(ptr) : "memory");
    return val;
}

__device__ inline void drain_stores() {
    __syncthreads();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

__device__ inline uint32x4 poll_lamport(const uint32x4 *src, uint32_t tag, uint64_t deadline) {
    uint64_t t0 = 0;
    for (int n = 1;; ++n) {
        const uint32x4 pk = load_nt16_sync(src);
        if (pk.y == tag && pk.w == tag) {
            return pk;
        }
        __builtin_amdgcn_s_sleep(1);
        if ((n & 1023) == 0) {
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > deadline) {
                __builtin_trap();
            }
        }
    }
}

__device__ inline void poll_flag(const uint32_t *src, uint32_t tag, uint64_t deadline) {
    uint64_t t0 = 0;
    for (int n = 1;; ++n) {
        if (load_nt32_sync(src) == tag) {
            return;
        }
        __builtin_amdgcn_s_sleep(1);
        if ((n & 1023) == 0) {
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > deadline) {
                __builtin_trap();
            }
        }
    }
}

__device__ inline uint32_t advance_block_epoch(int32_t *block_epoch, int offset) {
    __shared__ uint32_t s_tag;
    if (threadIdx.x == 0) {
        uint32_t cur = static_cast<uint32_t>(block_epoch[offset + blockIdx.x]) + 1u;
        if (cur == 0u) {
            cur = 4u;
        }
        block_epoch[offset + blockIdx.x] = static_cast<int32_t>(cur);
        s_tag = cur;
    }
    __syncthreads();
    return s_tag;
}

__device__ inline uint2 pack_bf16x4(float a0, float a1, float a2, float a3) {
    bf16_t b[4] = {
        static_cast<bf16_t>(a0),
        static_cast<bf16_t>(a1),
        static_cast<bf16_t>(a2),
        static_cast<bf16_t>(a3),
    };
    uint2 u;
    __builtin_memcpy(&u, b, 8);
    return u;
}

__device__ inline void unpack_bf16x4(uint32_t w0, uint32_t w1, float (&out)[4]) {
    bf16_t b[4];
    uint2 u = make_uint2(w0, w1);
    __builtin_memcpy(b, &u, 8);
    out[0] = static_cast<float>(b[0]);
    out[1] = static_cast<float>(b[1]);
    out[2] = static_cast<float>(b[2]);
    out[3] = static_cast<float>(b[3]);
}

__device__ inline void unpack_bf16x8(uint32x4 v, float (&out)[8]) {
    bf16_t b[8];
    __builtin_memcpy(b, &v, 16);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<float>(b[i]);
    }
}

__device__ inline uint32x4 pack_bf16x8(const float (&in)[8]) {
    bf16_t b[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        b[i] = static_cast<bf16_t>(in[i]);
    }
    uint32x4 v;
    __builtin_memcpy(&v, b, 16);
    return v;
}

// ============================================================================
// Kernel 0: Fast 1-launch RMSNorm for Decode latent [M, D] (eliminates 6 eager ops)
// ============================================================================
__global__ __launch_bounds__(256) void xgmi_fused_rmsnorm_bf16_kernel(
    const bf16_t *__restrict__ x,
    const bf16_t *__restrict__ w,
    bf16_t *__restrict__ out,
    int D,
    float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int vecs = D >> 3;
    const uint32x4 *xr = reinterpret_cast<const uint32x4 *>(x + static_cast<int64_t>(row) * D);
    const uint32x4 *wr = reinterpret_cast<const uint32x4 *>(w);
    uint32x4 *orow = reinterpret_cast<uint32x4 *>(out + static_cast<int64_t>(row) * D);

    float sum_sq = 0.0f;
    for (int v = tid; v < vecs; v += 256) {
        float fv[8];
        unpack_bf16x8(xr[v], fv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sum_sq += fv[j] * fv[j];
        }
    }
    // Wave64 reduction + shared reduction across 4 waves
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset, 64);
    }
    __shared__ float s_wave[4];
    if ((tid & 63) == 0) {
        s_wave[tid >> 6] = sum_sq;
    }
    __syncthreads();
    __shared__ float s_inv;
    if (tid == 0) {
        float tot = s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3];
        s_inv = rsqrtf(tot / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (int v = tid; v < vecs; v += 256) {
        float fv[8], wv[8];
        unpack_bf16x8(xr[v], fv);
        unpack_bf16x8(wr[v], wv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            fv[j] = fv[j] * inv * wv[j];
        }
        orow[v] = pack_bf16x8(fv);
    }
}

// ============================================================================
// Kernel 1: Decode 1/8 column AllGather (with optional zero-copy dual split)
// If out_b != nullptr: first split_cols of each rank's slice go to out_a
// [M, 8*split_cols] and the rest go to out_b [M, 8*(slice_cols-split_cols)].
// ============================================================================
__global__ __launch_bounds__(DECODE_THREADS) void xgmi_lamport_allgather_kernel(
    const bf16_t *__restrict__ x_slice,
    bf16_t *__restrict__ out_a,
    bf16_t *__restrict__ out_b, // nullable
    int M,
    int slice_cols,
    int split_cols,
    int rank,
    PeerTable peers,
    int64_t decode_off,
    int64_t half_bytes,
    int64_t ag_sub_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int64_t reg_off = decode_off + static_cast<int64_t>(tag & 1u) * half_bytes + ag_sub_off;
    const int lines_per_row = slice_cols >> 2;
    const int split_lines = split_cols >> 2;
    const int b_slice_cols = slice_cols - split_cols;
    const int total_lines = M * lines_per_row;
    const int a_total_cols = (out_b == nullptr ? slice_cols : split_cols) * TP;
    const int b_total_cols = b_slice_cols * TP;
    const int tid = blockIdx.x * DECODE_THREADS + threadIdx.x;
    const int stride = DECODE_BLOCKS * DECODE_THREADS;

    auto write_val = [&](int row, int src, int l, uint2 v) {
        if (out_b == nullptr) {
            reinterpret_cast<uint2 *>(out_a + static_cast<int64_t>(row) * a_total_cols + src * slice_cols)[l] = v;
        } else if (l < split_lines) {
            reinterpret_cast<uint2 *>(out_a + static_cast<int64_t>(row) * a_total_cols + src * split_cols)[l] = v;
        } else {
            reinterpret_cast<uint2 *>(out_b + static_cast<int64_t>(row) * b_total_cols + src * b_slice_cols)[l - split_lines] = v;
        }
    };

    for (int idx = tid; idx < total_lines; idx += stride) {
        const int row = idx / lines_per_row;
        const int l = idx - row * lines_per_row;
        const uint2 v = reinterpret_cast<const uint2 *>(x_slice + static_cast<int64_t>(row) * slice_cols)[l];
        write_val(row, rank, l, v);
        const uint32x4 pkt = {v.x, tag, v.y, tag};
        const int64_t slot_idx = static_cast<int64_t>(rank) * total_lines + idx;
        #pragma unroll
        for (int dst = 0; dst < TP; ++dst) {
            if (dst != rank) {
                uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + reg_off) + slot_idx;
                store_nt16(dst_ptr, pkt);
            }
        }
    }

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + reg_off);
    for (int idx = tid; idx < total_lines; idx += stride) {
        const int row = idx / lines_per_row;
        const int l = idx - row * lines_per_row;
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            if (src != rank) {
                const int64_t slot_idx = static_cast<int64_t>(src) * total_lines + idx;
                const uint32x4 pkt = poll_lamport(own_ag + slot_idx, tag, deadline);
                write_val(row, src, l, make_uint2(pkt.x, pkt.z));
            }
        }
    }
}

// ============================================================================
// Kernel 2: Decode Deferred Top-K + Routed Latent AR + Shared RS/AR (Lamport)
// ============================================================================
__global__ __launch_bounds__(DECODE_THREADS) void xgmi_deferred_topk_lamport_ar_kernel(
    const bf16_t *__restrict__ expert_out,
    const int32_t *__restrict__ sorted_idx,
    const float *__restrict__ topk_weights,
    const bf16_t *__restrict__ shared_partial,
    bf16_t *__restrict__ latent_out,
    bf16_t *__restrict__ shared_out,
    int M,
    int top_k,
    int D_lat,
    int D_sh,
    int gather_full_shared,
    int rank,
    PeerTable peers,
    int64_t decode_off,
    int64_t half_bytes,
    int64_t ag_sub_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int64_t half_off = decode_off + static_cast<int64_t>(tag & 1u) * half_bytes;
    const int64_t rs_off = half_off;
    const int64_t ag_off = half_off + ag_sub_off;

    const int lat_slice = D_lat / TP;
    const int sh_slice = D_sh / TP;
    const int lat_lines = lat_slice >> 2;
    const int sh_lines = sh_slice >> 2;
    const int lat_slice_total = M * lat_lines;
    const int sh_slice_total = M * sh_lines;
    const int64_t rs_sh_off_vecs = static_cast<int64_t>(TP) * lat_slice_total;
    const int64_t ag_sh_off_vecs = static_cast<int64_t>(TP) * lat_slice_total;

    const int tid = blockIdx.x * DECODE_THREADS + threadIdx.x;
    const int stride = DECODE_BLOCKS * DECODE_THREADS;

    const int total_lat_all = M * (D_lat >> 2);
    for (int idx = tid; idx < total_lat_all; idx += stride) {
        const int row = idx / (D_lat >> 2);
        const int gl = idx - row * (D_lat >> 2);
        const int owner = gl / lat_lines;
        const int ll = gl - owner * lat_lines;
        const int col0 = gl << 2;

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const float *w_row = topk_weights + static_cast<int64_t>(row) * top_k;
        for (int k = 0; k < top_k; ++k) {
            int64_t exp_row;
            if (sorted_idx == nullptr) {
                exp_row = static_cast<int64_t>(row) * top_k + k;
            } else {
                const int sidx = sorted_idx[static_cast<int64_t>(row) * top_k + k];
                if (sidx < 0) {
                    continue;
                }
                exp_row = static_cast<int64_t>(sidx);
            }
            const float w = w_row[k];
            const uint2 raw = *reinterpret_cast<const uint2 *>(expert_out + exp_row * D_lat + col0);
            float vals[4];
            unpack_bf16x4(raw.x, raw.y, vals);
            acc[0] += w * vals[0];
            acc[1] += w * vals[1];
            acc[2] += w * vals[2];
            acc[3] += w * vals[3];
        }
        const uint2 packed = pack_bf16x4(acc[0], acc[1], acc[2], acc[3]);
        const uint32x4 pkt = {packed.x, tag, packed.y, tag};
        const int64_t dst_idx = static_cast<int64_t>(rank) * lat_slice_total + static_cast<int64_t>(row) * lat_lines + ll;
        uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[owner] + rs_off) + dst_idx;
        store_nt16(dst_ptr, pkt);
    }

    const int total_sh_all = M * (D_sh >> 2);
    for (int idx = tid; idx < total_sh_all; idx += stride) {
        const int row = idx / (D_sh >> 2);
        const int gl = idx - row * (D_sh >> 2);
        const int owner = gl / sh_lines;
        const int sl = gl - owner * sh_lines;
        const uint2 v = reinterpret_cast<const uint2 *>(shared_partial + static_cast<int64_t>(row) * D_sh)[gl];
        const uint32x4 pkt = {v.x, tag, v.y, tag};
        const int64_t dst_idx = rs_sh_off_vecs + static_cast<int64_t>(rank) * sh_slice_total +
                                static_cast<int64_t>(row) * sh_lines + sl;
        uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[owner] + rs_off) + dst_idx;
        store_nt16(dst_ptr, pkt);
    }

    const uint32x4 *own_rs = reinterpret_cast<const uint32x4 *>(peers.base[rank] + rs_off);
    for (int idx = tid; idx < lat_slice_total; idx += stride) {
        const int row = idx / lat_lines;
        const int ll = idx - row * lat_lines;
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            const int64_t s_idx = static_cast<int64_t>(src) * lat_slice_total + idx;
            const uint32x4 pkt = poll_lamport(own_rs + s_idx, tag, deadline);
            float vals[4];
            unpack_bf16x4(pkt.x, pkt.z, vals);
            sum[0] += vals[0];
            sum[1] += vals[1];
            sum[2] += vals[2];
            sum[3] += vals[3];
        }
        const uint2 red = pack_bf16x4(sum[0], sum[1], sum[2], sum[3]);
        reinterpret_cast<uint2 *>(latent_out + static_cast<int64_t>(row) * D_lat + rank * lat_slice)[ll] = red;
        const uint32x4 ag_pkt = {red.x, tag, red.y, tag};
        const int64_t ag_idx = static_cast<int64_t>(rank) * lat_slice_total + idx;
        #pragma unroll
        for (int dst = 0; dst < TP; ++dst) {
            if (dst != rank) {
                uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + ag_off) + ag_idx;
                store_nt16(dst_ptr, ag_pkt);
            }
        }
    }

    for (int idx = tid; idx < sh_slice_total; idx += stride) {
        const int row = idx / sh_lines;
        const int sl = idx - row * sh_lines;
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            const int64_t s_idx = rs_sh_off_vecs + static_cast<int64_t>(src) * sh_slice_total + idx;
            const uint32x4 pkt = poll_lamport(own_rs + s_idx, tag, deadline);
            float vals[4];
            unpack_bf16x4(pkt.x, pkt.z, vals);
            sum[0] += vals[0];
            sum[1] += vals[1];
            sum[2] += vals[2];
            sum[3] += vals[3];
        }
        const uint2 red = pack_bf16x4(sum[0], sum[1], sum[2], sum[3]);
        if (gather_full_shared) {
            reinterpret_cast<uint2 *>(shared_out + static_cast<int64_t>(row) * D_sh + rank * sh_slice)[sl] = red;
            const uint32x4 ag_pkt = {red.x, tag, red.y, tag};
            const int64_t ag_idx = ag_sh_off_vecs + static_cast<int64_t>(rank) * sh_slice_total + idx;
            #pragma unroll
            for (int dst = 0; dst < TP; ++dst) {
                if (dst != rank) {
                    uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + ag_off) + ag_idx;
                    store_nt16(dst_ptr, ag_pkt);
                }
            }
        } else {
            reinterpret_cast<uint2 *>(shared_out + static_cast<int64_t>(row) * sh_slice)[sl] = red;
        }
    }

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + ag_off);
    for (int idx = tid; idx < lat_slice_total; idx += stride) {
        const int row = idx / lat_lines;
        const int ll = idx - row * lat_lines;
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            if (src != rank) {
                const int64_t s_idx = static_cast<int64_t>(src) * lat_slice_total + idx;
                const uint32x4 pkt = poll_lamport(own_ag + s_idx, tag, deadline);
                reinterpret_cast<uint2 *>(latent_out + static_cast<int64_t>(row) * D_lat + src * lat_slice)[ll] = make_uint2(pkt.x, pkt.z);
            }
        }
    }

    if (gather_full_shared) {
        for (int idx = tid; idx < sh_slice_total; idx += stride) {
            const int row = idx / sh_lines;
            const int sl = idx - row * sh_lines;
            #pragma unroll
            for (int src = 0; src < TP; ++src) {
                if (src != rank) {
                    const int64_t s_idx = ag_sh_off_vecs + static_cast<int64_t>(src) * sh_slice_total + idx;
                    const uint32x4 pkt = poll_lamport(own_ag + s_idx, tag, deadline);
                    reinterpret_cast<uint2 *>(shared_out + static_cast<int64_t>(row) * D_sh + src * sh_slice)[sl] = make_uint2(pkt.x, pkt.z);
                }
            }
        }
    }
}

// ============================================================================
// Kernel 3: Decode 1/8 MoE Tail Column Gather (up_slice + shared + prefix -> stream)
// ============================================================================
__global__ __launch_bounds__(DECODE_THREADS) void xgmi_moetail_column_gather_kernel(
    const bf16_t *__restrict__ up_slice,
    const bf16_t *__restrict__ shared_slice,
    const bf16_t *__restrict__ prefix_slice,
    bf16_t *__restrict__ stream_out,
    int M,
    int slice_cols,
    int rank,
    PeerTable peers,
    int64_t decode_off,
    int64_t half_bytes,
    int64_t ag_sub_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int64_t reg_off = decode_off + static_cast<int64_t>(tag & 1u) * half_bytes + ag_sub_off;
    const int lines_per_row = slice_cols >> 2;
    const int total_lines = M * lines_per_row;
    const int total_cols = slice_cols * TP;
    const int tid = blockIdx.x * DECODE_THREADS + threadIdx.x;
    const int stride = DECODE_BLOCKS * DECODE_THREADS;

    for (int idx = tid; idx < total_lines; idx += stride) {
        const int row = idx / lines_per_row;
        const int l = idx - row * lines_per_row;
        const int64_t off = static_cast<int64_t>(row) * slice_cols;
        const uint2 u_up = reinterpret_cast<const uint2 *>(up_slice + off)[l];
        const uint2 u_sh = reinterpret_cast<const uint2 *>(shared_slice + off)[l];
        float a[4], b[4];
        unpack_bf16x4(u_up.x, u_up.y, a);
        unpack_bf16x4(u_sh.x, u_sh.y, b);
        if (prefix_slice != nullptr) {
            const uint2 u_pr = reinterpret_cast<const uint2 *>(prefix_slice + off)[l];
            float c[4];
            unpack_bf16x4(u_pr.x, u_pr.y, c);
            a[0] += b[0] + c[0];
            a[1] += b[1] + c[1];
            a[2] += b[2] + c[2];
            a[3] += b[3] + c[3];
        } else {
            a[0] += b[0];
            a[1] += b[1];
            a[2] += b[2];
            a[3] += b[3];
        }
        const uint2 packed = pack_bf16x4(a[0], a[1], a[2], a[3]);
        reinterpret_cast<uint2 *>(stream_out + static_cast<int64_t>(row) * total_cols + rank * slice_cols)[l] = packed;
        const uint32x4 pkt = {packed.x, tag, packed.y, tag};
        const int64_t slot_idx = static_cast<int64_t>(rank) * total_lines + idx;
        #pragma unroll
        for (int dst = 0; dst < TP; ++dst) {
            if (dst != rank) {
                uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + reg_off) + slot_idx;
                store_nt16(dst_ptr, pkt);
            }
        }
    }

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + reg_off);
    for (int idx = tid; idx < total_lines; idx += stride) {
        const int row = idx / lines_per_row;
        const int l = idx - row * lines_per_row;
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            if (src != rank) {
                const int64_t slot_idx = static_cast<int64_t>(src) * total_lines + idx;
                const uint32x4 pkt = poll_lamport(own_ag + slot_idx, tag, deadline);
                reinterpret_cast<uint2 *>(stream_out + static_cast<int64_t>(row) * total_cols + src * slice_cols)[l] = make_uint2(pkt.x, pkt.z);
            }
        }
    }
}

// ============================================================================
// Kernel 4a: Prefill 32/64-block 7-Link-Parallel Chunked Reduce-Scatter
// Supports BOTH row sub-chunking [row_off : row_off + sub_rows) AND zero-copy
// column-chunked output placement out_shard[:, out_col_off : out_col_off + cols]
// so N-chunked o_proj GEMMs overlap 100% with 7-link-parallel XGMI Reduce-Scatter!
// ============================================================================
__global__ __launch_bounds__(RS_THREADS) void xgmi_rs_chunk_32cu_kernel(
    const bf16_t *__restrict__ partial,
    bf16_t *__restrict__ out_shard,
    int rows_per_rank,
    int row_off,
    int sub_rows,
    int cols,
    int out_col_off,
    int out_stride_cols,
    int rank,
    PeerTable peers,
    int64_t rs_off,
    int64_t rs_slot_bytes,
    int64_t rs_flags_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int slot = static_cast<int>(tag & (RS_RING_SLOTS - 1));
    const int64_t slot_off = rs_off + static_cast<int64_t>(slot) * rs_slot_bytes;
    const int64_t flag_base = rs_flags_off + static_cast<int64_t>(slot * RS_BLOCKS * TP + blockIdx.x * TP) * 4;

    const int vecs_per_row = cols >> 3;
    const int out_vecs_stride = out_stride_cols >> 3;
    const int out_vec_col0 = out_col_off >> 3;
    const int64_t total_vecs = static_cast<int64_t>(sub_rows) * vecs_per_row;
    const int64_t per_blk = (total_vecs + RS_BLOCKS - 1) / RS_BLOCKS;
    const int64_t v_begin = static_cast<int64_t>(blockIdx.x) * per_blk;
    const int64_t v_end = v_begin + per_blk < total_vecs ? v_begin + per_blk : total_vecs;

    // All 7 XGMI links fire concurrently via block-rotated peer order!
    for (int pi = 0; pi < TP - 1; ++pi) {
        const int p_off = 1 + ((pi + static_cast<int>(blockIdx.x)) % (TP - 1));
        const int dst = (rank + p_off) & (TP - 1);
        const int64_t src_row0 = static_cast<int64_t>(dst) * rows_per_rank + row_off;
        const uint32x4 *src_base = reinterpret_cast<const uint32x4 *>(partial + src_row0 * cols);
        uint32x4 *dst_base = reinterpret_cast<uint32x4 *>(peers.base[dst] + slot_off) +
                             static_cast<int64_t>(rank) * total_vecs;
        for (int64_t v = v_begin + threadIdx.x; v < v_end; v += RS_THREADS) {
            store_nt16(dst_base + v, src_base[v]);
        }
    }

    drain_stores();
    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int dst = threadIdx.x;
        uint32_t *flag_ptr = reinterpret_cast<uint32_t *>(peers.base[dst] + flag_base) + rank;
        store_nt32(flag_ptr, tag);
    }

    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int src = threadIdx.x;
        const uint32_t *flag_ptr = reinterpret_cast<const uint32_t *>(peers.base[rank] + flag_base) + src;
        poll_flag(flag_ptr, tag, deadline);
    }
    __syncthreads();

    const uint32x4 *own_slot = reinterpret_cast<const uint32x4 *>(peers.base[rank] + slot_off);
    const int64_t own_row0 = static_cast<int64_t>(rank) * rows_per_rank + row_off;
    const uint32x4 *local_src = reinterpret_cast<const uint32x4 *>(partial + own_row0 * cols);
    uint32x4 *out_base = reinterpret_cast<uint32x4 *>(out_shard + static_cast<int64_t>(row_off) * out_stride_cols);
    const bool direct_out = (out_stride_cols == cols && out_col_off == 0);

    for (int64_t v = v_begin + threadIdx.x; v < v_end; v += RS_THREADS) {
        uint32x4 raws[TP];
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            raws[src] = (src == rank)
                            ? local_src[v]
                            : load_nt16_bulk(own_slot + static_cast<int64_t>(src) * total_vecs + v);
        }
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            float vals[8];
            unpack_bf16x8(raws[src], vals);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j] += vals[j];
            }
        }
        const uint32x4 packed = pack_bf16x8(acc);
        if (direct_out) {
            out_base[v] = packed;
        } else {
            const int r = static_cast<int>(v) / vecs_per_row;
            const int cv = static_cast<int>(v) - r * vecs_per_row;
            out_base[r * out_vecs_stride + out_vec_col0 + cv] = packed;
        }
    }
}

// ============================================================================
// Kernel 4b/4c: 2-Half Overlapped 32-CU RS (4-link parallel push + 3-link push & reduce)
// Half 0 (peers base..base+3, NOT containing rank): 1 huge GEMM [4*sub, 7168] -> push_half
// Half 1 (peers other..other+3, containing rank): 1 huge GEMM [4*sub, 7168] -> finish_half
// ============================================================================
__global__ __launch_bounds__(RS_THREADS) void xgmi_rs_push_half_32cu_kernel(
    const bf16_t *__restrict__ partial_4peers, // [4, rows_per_rank, cols] or [4*sub_rows, cols]
    int peer_base, // 0 or 4 (must NOT contain rank)
    int rows_per_rank,
    int row_off,
    int sub_rows,
    int cols,
    int rank,
    PeerTable peers,
    int64_t rs_off,
    int64_t rs_slot_bytes,
    int64_t rs_flags_off,
    const int32_t *__restrict__ block_epoch,
    int epoch_off) {
    uint32_t tag = static_cast<uint32_t>(block_epoch[epoch_off + blockIdx.x]) + 1u;
    if (tag == 0u) {
        tag = 4u;
    }
    const int slot = static_cast<int>(tag & (RS_RING_SLOTS - 1));
    const int64_t slot_off = rs_off + static_cast<int64_t>(slot) * rs_slot_bytes;
    const int64_t flag_base = rs_flags_off + static_cast<int64_t>(slot * RS_BLOCKS * TP + blockIdx.x * TP) * 4;
    const int vecs_per_row = cols >> 3;
    const int64_t total_vecs = static_cast<int64_t>(sub_rows) * vecs_per_row;
    const int64_t per_blk = (total_vecs + RS_BLOCKS - 1) / RS_BLOCKS;
    const int64_t v_begin = static_cast<int64_t>(blockIdx.x) * per_blk;
    const int64_t v_end = v_begin + per_blk < total_vecs ? v_begin + per_blk : total_vecs;

    // Rotate 4 peers across blocks so all 4 XGMI links fire concurrently!
    for (int ki = 0; ki < 4; ++ki) {
        const int local_p = (ki + static_cast<int>(blockIdx.x)) & 3;
        const int dst = peer_base + local_p;
        const int64_t src_row0 = static_cast<int64_t>(local_p) * rows_per_rank + row_off;
        const uint32x4 *src_base = reinterpret_cast<const uint32x4 *>(partial_4peers + src_row0 * cols);
        uint32x4 *dst_base = reinterpret_cast<uint32x4 *>(peers.base[dst] + slot_off) +
                             static_cast<int64_t>(rank) * total_vecs;
        for (int64_t v = v_begin + threadIdx.x; v < v_end; v += RS_THREADS) {
            store_nt16(dst_base + v, src_base[v]);
        }
    }
    drain_stores();
    if (threadIdx.x < 4) {
        const int dst = peer_base + threadIdx.x;
        store_nt32(reinterpret_cast<uint32_t *>(peers.base[dst] + flag_base) + rank, tag);
    }
}

__global__ __launch_bounds__(RS_THREADS) void xgmi_rs_finish_half_32cu_kernel(
    const bf16_t *__restrict__ partial_4peers, // [4, rows_per_rank, cols] containing rank
    bf16_t *__restrict__ out_shard,            // [rows_per_rank, cols]
    int peer_base, // 0 or 4 (contains rank)
    int rows_per_rank,
    int row_off,
    int sub_rows,
    int cols,
    int rank,
    PeerTable peers,
    int64_t rs_off,
    int64_t rs_slot_bytes,
    int64_t rs_flags_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int slot = static_cast<int>(tag & (RS_RING_SLOTS - 1));
    const int64_t slot_off = rs_off + static_cast<int64_t>(slot) * rs_slot_bytes;
    const int64_t flag_base = rs_flags_off + static_cast<int64_t>(slot * RS_BLOCKS * TP + blockIdx.x * TP) * 4;
    const int vecs_per_row = cols >> 3;
    const int64_t total_vecs = static_cast<int64_t>(sub_rows) * vecs_per_row;
    const int64_t per_blk = (total_vecs + RS_BLOCKS - 1) / RS_BLOCKS;
    const int64_t v_begin = static_cast<int64_t>(blockIdx.x) * per_blk;
    const int64_t v_end = v_begin + per_blk < total_vecs ? v_begin + per_blk : total_vecs;

    // Push the 3 remaining remote peers in this half (rotated across blocks)
    for (int ki = 0; ki < 4; ++ki) {
        const int local_p = (ki + static_cast<int>(blockIdx.x)) & 3;
        const int dst = peer_base + local_p;
        if (dst == rank) {
            continue;
        }
        const int64_t src_row0 = static_cast<int64_t>(local_p) * rows_per_rank + row_off;
        const uint32x4 *src_base = reinterpret_cast<const uint32x4 *>(partial_4peers + src_row0 * cols);
        uint32x4 *dst_base = reinterpret_cast<uint32x4 *>(peers.base[dst] + slot_off) +
                             static_cast<int64_t>(rank) * total_vecs;
        for (int64_t v = v_begin + threadIdx.x; v < v_end; v += RS_THREADS) {
            store_nt16(dst_base + v, src_base[v]);
        }
    }
    drain_stores();
    if (threadIdx.x < 4) {
        const int dst = peer_base + threadIdx.x;
        if (dst != rank) {
            store_nt32(reinterpret_cast<uint32_t *>(peers.base[dst] + flag_base) + rank, tag);
        }
    }

    // Wait for all 7 remote peers arrival flags
    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int src = threadIdx.x;
        poll_flag(reinterpret_cast<const uint32_t *>(peers.base[rank] + flag_base) + src, tag, deadline);
    }
    __syncthreads();

    // Deterministic fp32 rank-order reduction into out_shard[row_off : row_off + sub_rows]
    const uint32x4 *own_slot = reinterpret_cast<const uint32x4 *>(peers.base[rank] + slot_off);
    const int own_local_p = rank - peer_base;
    const int64_t own_row0 = static_cast<int64_t>(own_local_p) * rows_per_rank + row_off;
    const uint32x4 *local_src = reinterpret_cast<const uint32x4 *>(partial_4peers + own_row0 * cols);
    uint32x4 *out_vec = reinterpret_cast<uint32x4 *>(out_shard + static_cast<int64_t>(row_off) * cols);

    for (int64_t v = v_begin + threadIdx.x; v < v_end; v += RS_THREADS) {
        uint32x4 raws[TP];
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            raws[src] = (src == rank)
                            ? local_src[v]
                            : load_nt16_bulk(own_slot + static_cast<int64_t>(src) * total_vecs + v);
        }
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int src = 0; src < TP; ++src) {
            float vals[8];
            unpack_bf16x8(raws[src], vals);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j] += vals[j];
            }
        }
        out_vec[v] = pack_bf16x8(acc);
    }
}

// ============================================================================
// Kernel 5: Prefill Fused E2M1 FP4 (or FP8) + Scale + TopK Pack & AllGather
// ============================================================================
__global__ __launch_bounds__(AG_THREADS) void xgmi_fp4_pack_allgather_kernel(
    const uint8_t *__restrict__ q_data,
    const uint8_t *__restrict__ q_scale,
    const int32_t *__restrict__ topk_ids,
    const float *__restrict__ topk_weights,
    uint8_t *__restrict__ q_data_all,
    uint8_t *__restrict__ q_scale_all,
    int32_t *__restrict__ topk_ids_all,
    float *__restrict__ topk_weights_all,
    int rows,
    int q_vecs,
    int s_vecs,
    int tk_vecs,
    int rank,
    PeerTable peers,
    int64_t ag_off,
    int64_t ag_half_bytes,
    int64_t ag_flags_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int par = static_cast<int>(tag & 1u);
    const int64_t half_off = ag_off + static_cast<int64_t>(par) * ag_half_bytes;
    const int64_t flag_base = ag_flags_off + static_cast<int64_t>(par * AG_BLOCKS * TP + blockIdx.x * TP) * 4;

    const int row_vecs = q_vecs + s_vecs + 2 * tk_vecs;
    const int64_t total_vecs = static_cast<int64_t>(rows) * row_vecs;
    const int64_t per_blk = (total_vecs + AG_BLOCKS - 1) / AG_BLOCKS;
    const int64_t v_begin = static_cast<int64_t>(blockIdx.x) * per_blk;
    const int64_t v_end = v_begin + per_blk < total_vecs ? v_begin + per_blk : total_vecs;

    const uint32x4 *qd_in = reinterpret_cast<const uint32x4 *>(q_data);
    const uint32x4 *qs_in = reinterpret_cast<const uint32x4 *>(q_scale);
    const uint32x4 *ti_in = reinterpret_cast<const uint32x4 *>(topk_ids);
    const uint32x4 *tw_in = reinterpret_cast<const uint32x4 *>(topk_weights);

    uint32x4 *qd_out = reinterpret_cast<uint32x4 *>(q_data_all);
    uint32x4 *qs_out = reinterpret_cast<uint32x4 *>(q_scale_all);
    uint32x4 *ti_out = reinterpret_cast<uint32x4 *>(topk_ids_all);
    uint32x4 *tw_out = reinterpret_cast<uint32x4 *>(topk_weights_all);

    for (int64_t idx = v_begin + threadIdx.x; idx < v_end; idx += AG_THREADS) {
        const int r = static_cast<int>(idx / row_vecs);
        const int lv = static_cast<int>(idx - static_cast<int64_t>(r) * row_vecs);
        const int dst_r = rank * rows + r;
        uint32x4 val;
        if (lv < q_vecs) {
            val = qd_in[static_cast<int64_t>(r) * q_vecs + lv];
            qd_out[static_cast<int64_t>(dst_r) * q_vecs + lv] = val;
        } else if (lv < q_vecs + s_vecs) {
            const int sv = lv - q_vecs;
            val = qs_in[static_cast<int64_t>(r) * s_vecs + sv];
            qs_out[static_cast<int64_t>(dst_r) * s_vecs + sv] = val;
        } else if (lv < q_vecs + s_vecs + tk_vecs) {
            const int tv = lv - q_vecs - s_vecs;
            val = ti_in[static_cast<int64_t>(r) * tk_vecs + tv];
            ti_out[static_cast<int64_t>(dst_r) * tk_vecs + tv] = val;
        } else {
            const int tv = lv - q_vecs - s_vecs - tk_vecs;
            val = tw_in[static_cast<int64_t>(r) * tk_vecs + tv];
            tw_out[static_cast<int64_t>(dst_r) * tk_vecs + tv] = val;
        }
        const int64_t peer_idx = static_cast<int64_t>(rank) * total_vecs + idx;
        #pragma unroll
        for (int pi = 0; pi < TP - 1; ++pi) {
            const int p_off = 1 + ((pi + static_cast<int>(blockIdx.x)) % (TP - 1));
            const int dst = (rank + p_off) & (TP - 1);
            uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + half_off) + peer_idx;
            store_nt16(dst_ptr, val);
        }
    }

    drain_stores();
    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int dst = threadIdx.x;
        uint32_t *flag_ptr = reinterpret_cast<uint32_t *>(peers.base[dst] + flag_base) + rank;
        store_nt32(flag_ptr, tag);
    }

    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int src = threadIdx.x;
        const uint32_t *flag_ptr = reinterpret_cast<const uint32_t *>(peers.base[rank] + flag_base) + src;
        poll_flag(flag_ptr, tag, deadline);
    }
    __syncthreads();

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + half_off);
    for (int p_off = 1; p_off < TP; ++p_off) {
        const int src = (rank + p_off) & (TP - 1);
        const uint32x4 *src_base = own_ag + static_cast<int64_t>(src) * total_vecs;
        for (int64_t idx = v_begin + threadIdx.x; idx < v_end; idx += AG_THREADS) {
            const int r = static_cast<int>(idx / row_vecs);
            const int lv = static_cast<int>(idx - static_cast<int64_t>(r) * row_vecs);
            const int dst_r = src * rows + r;
            const uint32x4 val = load_nt16_bulk(src_base + idx);
            if (lv < q_vecs) {
                qd_out[static_cast<int64_t>(dst_r) * q_vecs + lv] = val;
            } else if (lv < q_vecs + s_vecs) {
                qs_out[static_cast<int64_t>(dst_r) * s_vecs + (lv - q_vecs)] = val;
            } else if (lv < q_vecs + s_vecs + tk_vecs) {
                ti_out[static_cast<int64_t>(dst_r) * tk_vecs + (lv - q_vecs - s_vecs)] = val;
            } else {
                tw_out[static_cast<int64_t>(dst_r) * tk_vecs + (lv - q_vecs - s_vecs - tk_vecs)] = val;
            }
        }
    }
}

// ============================================================================
// Kernel 6: Prefill High-BW Row AllGather (full MLP bulk loads!)
// ============================================================================
__global__ __launch_bounds__(AG_THREADS) void xgmi_row_allgather_kernel(
    const bf16_t *__restrict__ x_shard,
    bf16_t *__restrict__ out_all,
    int rows,
    int cols,
    int rank,
    PeerTable peers,
    int64_t ag_off,
    int64_t ag_half_bytes,
    int64_t ag_flags_off,
    int32_t *__restrict__ block_epoch,
    int epoch_off,
    uint64_t deadline) {
    const uint32_t tag = advance_block_epoch(block_epoch, epoch_off);
    const int par = static_cast<int>(tag & 1u);
    const int64_t half_off = ag_off + static_cast<int64_t>(par) * ag_half_bytes;
    const int64_t flag_base = ag_flags_off + static_cast<int64_t>(par * AG_BLOCKS * TP + blockIdx.x * TP) * 4;

    const int vecs_per_row = cols >> 3;
    const int64_t total_vecs = static_cast<int64_t>(rows) * vecs_per_row;
    const int64_t per_blk = (total_vecs + AG_BLOCKS - 1) / AG_BLOCKS;
    const int64_t v_begin = static_cast<int64_t>(blockIdx.x) * per_blk;
    const int64_t v_end = v_begin + per_blk < total_vecs ? v_begin + per_blk : total_vecs;

    const uint32x4 *src_vec = reinterpret_cast<const uint32x4 *>(x_shard);
    uint32x4 *own_out = reinterpret_cast<uint32x4 *>(out_all) + static_cast<int64_t>(rank) * total_vecs;

    for (int64_t v = v_begin + threadIdx.x; v < v_end; v += AG_THREADS) {
        const uint32x4 val = src_vec[v];
        own_out[v] = val;
        const int64_t peer_idx = static_cast<int64_t>(rank) * total_vecs + v;
        #pragma unroll
        for (int pi = 0; pi < TP - 1; ++pi) {
            const int p_off = 1 + ((pi + static_cast<int>(blockIdx.x)) % (TP - 1));
            const int dst = (rank + p_off) & (TP - 1);
            uint32x4 *dst_ptr = reinterpret_cast<uint32x4 *>(peers.base[dst] + half_off) + peer_idx;
            store_nt16(dst_ptr, val);
        }
    }

    drain_stores();
    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int dst = threadIdx.x;
        uint32_t *flag_ptr = reinterpret_cast<uint32_t *>(peers.base[dst] + flag_base) + rank;
        store_nt32(flag_ptr, tag);
    }

    if (threadIdx.x < TP && threadIdx.x != rank) {
        const int src = threadIdx.x;
        const uint32_t *flag_ptr = reinterpret_cast<const uint32_t *>(peers.base[rank] + flag_base) + src;
        poll_flag(flag_ptr, tag, deadline);
    }
    __syncthreads();

    const uint32x4 *own_ag = reinterpret_cast<const uint32x4 *>(peers.base[rank] + half_off);
    uint32x4 *out_base = reinterpret_cast<uint32x4 *>(out_all);
    for (int p_off = 1; p_off < TP; ++p_off) {
        const int src = (rank + p_off) & (TP - 1);
        const uint32x4 *src_buf = own_ag + static_cast<int64_t>(src) * total_vecs;
        uint32x4 *dst_out = out_base + static_cast<int64_t>(src) * total_vecs;
        for (int64_t v = v_begin + threadIdx.x; v < v_end; v += AG_THREADS) {
            dst_out[v] = load_nt16_bulk(src_buf + v);
        }
    }
}

// ============================================================================
// Host C++ Entrypoints
// ============================================================================
void fused_rmsnorm_bf16_op(
    const at::Tensor &x,
    const at::Tensor &w,
    at::Tensor &out,
    double eps) {
    TORCH_CHECK(x.is_contiguous() && x.dtype() == at::kBFloat16);
    TORCH_CHECK(w.is_contiguous() && w.dtype() == at::kBFloat16);
    TORCH_CHECK(out.is_contiguous() && out.dtype() == at::kBFloat16);
    const int M = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));
    TORCH_CHECK((D & 7) == 0 && w.numel() == D);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x.device().index()).stream();
    xgmi_fused_rmsnorm_bf16_kernel<<<M, 256, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(x.data_ptr()),
        reinterpret_cast<const bf16_t *>(w.data_ptr()),
        reinterpret_cast<bf16_t *>(out.data_ptr()),
        D, static_cast<float>(eps));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void lamport_allgather_op(
    const at::Tensor &x_slice,
    at::Tensor &out_a,
    at::Tensor &out_b,
    int64_t split_cols,
    const at::Tensor &peers_cpu,
    int64_t rank,
    int64_t decode_off,
    int64_t half_bytes,
    int64_t ag_sub_off,
    at::Tensor &block_epoch,
    int64_t epoch_off,
    double deadline_s) {
    TORCH_CHECK(x_slice.is_contiguous() && x_slice.dtype() == at::kBFloat16);
    TORCH_CHECK(out_a.is_contiguous() && out_a.dtype() == at::kBFloat16);
    const int M = static_cast<int>(x_slice.size(0));
    const int slice_cols = static_cast<int>(x_slice.size(1));
    TORCH_CHECK((slice_cols & 3) == 0 && (split_cols & 3) == 0);
    bf16_t *b_ptr = nullptr;
    if (out_b.numel() > 0) {
        TORCH_CHECK(out_b.is_contiguous() && out_b.dtype() == at::kBFloat16);
        b_ptr = reinterpret_cast<bf16_t *>(out_b.data_ptr());
    }
    const PeerTable peers = make_peers(peers_cpu);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x_slice.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x_slice.device().index()).stream();
    xgmi_lamport_allgather_kernel<<<DECODE_BLOCKS, DECODE_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(x_slice.data_ptr()),
        reinterpret_cast<bf16_t *>(out_a.data_ptr()),
        b_ptr,
        M, slice_cols, static_cast<int>(split_cols), static_cast<int>(rank), peers,
        decode_off, half_bytes, ag_sub_off,
        block_epoch.data_ptr<int32_t>(), static_cast<int>(epoch_off),
        static_cast<uint64_t>(deadline_s * 1e8));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void deferred_topk_lamport_ar_op(
    const at::Tensor &expert_out,
    const at::Tensor &sorted_idx,
    const at::Tensor &topk_weights,
    const at::Tensor &shared_partial,
    at::Tensor &latent_out,
    at::Tensor &shared_out,
    bool gather_full_shared,
    const at::Tensor &peers_cpu,
    int64_t rank,
    int64_t decode_off,
    int64_t half_bytes,
    int64_t ag_sub_off,
    at::Tensor &block_epoch,
    int64_t epoch_off,
    double deadline_s) {
    TORCH_CHECK(expert_out.is_contiguous() && expert_out.dtype() == at::kBFloat16);
    TORCH_CHECK(topk_weights.is_contiguous() && topk_weights.dtype() == at::kFloat);
    TORCH_CHECK(shared_partial.is_contiguous() && shared_partial.dtype() == at::kBFloat16);
    TORCH_CHECK(latent_out.is_contiguous() && latent_out.dtype() == at::kBFloat16);
    TORCH_CHECK(shared_out.is_contiguous() && shared_out.dtype() == at::kBFloat16);
    const int M = static_cast<int>(topk_weights.size(0));
    const int top_k = static_cast<int>(topk_weights.size(1));
    const int D_lat = static_cast<int>(latent_out.size(1));
    const int D_sh = static_cast<int>(shared_partial.size(1));
    TORCH_CHECK((D_lat % (TP * 4)) == 0 && (D_sh % (TP * 4)) == 0);
    const int32_t *sidx_ptr = nullptr;
    if (sorted_idx.numel() > 0) {
        TORCH_CHECK(sorted_idx.is_contiguous() && sorted_idx.dtype() == at::kInt);
        sidx_ptr = sorted_idx.data_ptr<int32_t>();
    }
    const PeerTable peers = make_peers(peers_cpu);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(shared_partial.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(shared_partial.device().index()).stream();
    xgmi_deferred_topk_lamport_ar_kernel<<<DECODE_BLOCKS, DECODE_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(expert_out.data_ptr()),
        sidx_ptr,
        topk_weights.data_ptr<float>(),
        reinterpret_cast<const bf16_t *>(shared_partial.data_ptr()),
        reinterpret_cast<bf16_t *>(latent_out.data_ptr()),
        reinterpret_cast<bf16_t *>(shared_out.data_ptr()),
        M, top_k, D_lat, D_sh, gather_full_shared ? 1 : 0,
        static_cast<int>(rank), peers,
        decode_off, half_bytes, ag_sub_off,
        block_epoch.data_ptr<int32_t>(), static_cast<int>(epoch_off),
        static_cast<uint64_t>(deadline_s * 1e8));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void moetail_column_gather_op(
    const at::Tensor &up_slice,
    const at::Tensor &shared_slice,
    const at::Tensor &prefix_slice,
    at::Tensor &stream_out,
    const at::Tensor &peers_cpu,
    int64_t rank,
    int64_t decode_off,
    int64_t half_bytes,
    int64_t ag_sub_off,
    at::Tensor &block_epoch,
    int64_t epoch_off,
    double deadline_s) {
    TORCH_CHECK(up_slice.is_contiguous() && up_slice.dtype() == at::kBFloat16);
    TORCH_CHECK(shared_slice.is_contiguous() && shared_slice.dtype() == at::kBFloat16);
    TORCH_CHECK(stream_out.is_contiguous() && stream_out.dtype() == at::kBFloat16);
    const int M = static_cast<int>(up_slice.size(0));
    const int slice_cols = static_cast<int>(up_slice.size(1));
    TORCH_CHECK((slice_cols & 3) == 0 && stream_out.size(0) == M && stream_out.size(1) == slice_cols * TP);
    const bf16_t *pref_ptr = nullptr;
    if (prefix_slice.numel() > 0) {
        TORCH_CHECK(prefix_slice.is_contiguous() && prefix_slice.dtype() == at::kBFloat16);
        pref_ptr = reinterpret_cast<const bf16_t *>(prefix_slice.data_ptr());
    }
    const PeerTable peers = make_peers(peers_cpu);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(up_slice.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(up_slice.device().index()).stream();
    xgmi_moetail_column_gather_kernel<<<DECODE_BLOCKS, DECODE_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(up_slice.data_ptr()),
        reinterpret_cast<const bf16_t *>(shared_slice.data_ptr()),
        pref_ptr,
        reinterpret_cast<bf16_t *>(stream_out.data_ptr()),
        M, slice_cols, static_cast<int>(rank), peers,
        decode_off, half_bytes, ag_sub_off,
        block_epoch.data_ptr<int32_t>(), static_cast<int>(epoch_off),
        static_cast<uint64_t>(deadline_s * 1e8));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void rs_chunk_32cu_op(
    const at::Tensor &partial,
    at::Tensor &out_shard,
    int64_t rows_per_rank,
    int64_t row_off,
    int64_t sub_rows,
    int64_t out_col_off,
    const at::Tensor &peers_cpu,
    int64_t rank,
    int64_t rs_off,
    int64_t rs_slot_bytes,
    int64_t rs_flags_off,
    at::Tensor &block_epoch,
    int64_t epoch_off,
    double deadline_s) {
    TORCH_CHECK(partial.is_contiguous() && partial.dtype() == at::kBFloat16);
    TORCH_CHECK(out_shard.is_contiguous() && out_shard.dtype() == at::kBFloat16);
    const int cols = static_cast<int>(partial.size(1));
    const int out_stride_cols = static_cast<int>(out_shard.size(1));
    TORCH_CHECK((cols & 7) == 0 && (out_col_off & 7) == 0 && (out_stride_cols & 7) == 0);
    TORCH_CHECK(partial.size(0) == rows_per_rank * TP && out_shard.size(0) == rows_per_rank);
    TORCH_CHECK(row_off >= 0 && row_off + sub_rows <= rows_per_rank);
    TORCH_CHECK(out_col_off >= 0 && out_col_off + cols <= out_stride_cols);
    TORCH_CHECK(static_cast<int64_t>(sub_rows) * TP * cols * 2 <= rs_slot_bytes);
    const PeerTable peers = make_peers(peers_cpu);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(partial.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(partial.device().index()).stream();
    xgmi_rs_chunk_32cu_kernel<<<RS_BLOCKS, RS_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(partial.data_ptr()),
        reinterpret_cast<bf16_t *>(out_shard.data_ptr()),
        static_cast<int>(rows_per_rank), static_cast<int>(row_off), static_cast<int>(sub_rows),
        cols, static_cast<int>(out_col_off), out_stride_cols,
        static_cast<int>(rank), peers,
        rs_off, rs_slot_bytes, rs_flags_off,
        block_epoch.data_ptr<int32_t>(), static_cast<int>(epoch_off),
        static_cast<uint64_t>(deadline_s * 1e8));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void rs_push_half_32cu_op(
    const at::Tensor &partial_4peers,
    int64_t peer_base,
    int64_t rows_per_rank,
    int64_t row_off,
    int64_t sub_rows,
    const at::Tensor &peers_cpu,
    int64_t rank,
    int64_t rs_off,
    int64_t rs_slot_bytes,
    int64_t rs_flags_off,
    const at::Tensor &block_epoch,
    int64_t epoch_off) {
    TORCH_CHECK(partial_4peers.is_contiguous() && partial_4peers.dtype() == at::kBFloat16);
    const int cols = static_cast<int>(partial_4peers.size(1));
    TORCH_CHECK((cols & 7) == 0 && partial_4peers.size(0) == 4 * rows_per_rank);
    TORCH_CHECK(static_cast<int64_t>(sub_rows) * TP * cols * 2 <= rs_slot_bytes);
    const PeerTable peers = make_peers(peers_cpu);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(partial_4peers.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(partial_4peers.device().index()).stream();
    xgmi_rs_push_half_32cu_kernel<<<RS_BLOCKS, RS_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(partial_4peers.data_ptr()),
        static_cast<int>(peer_base), static_cast<int>(rows_per_rank),
        static_cast<int>(row_off), static_cast<int>(sub_rows), cols,
        static_cast<int>(rank), peers, rs_off, rs_slot_bytes, rs_flags_off,
        block_epoch.data_ptr<int32_t>(), static_cast<int>(epoch_off));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void rs_finish_half_32cu_op(
    const at::Tensor &partial_4peers,
    at::Tensor &out_shard,
    int64_t peer_base,
    int64_t rows_per_rank,
    int64_t row_off,
    int64_t sub_rows,
    const at::Tensor &peers_cpu,
    int64_t rank,
    int64_t rs_off,
    int64_t rs_slot_bytes,
    int64_t rs_flags_off,
    at::Tensor &block_epoch,
    int64_t epoch_off,
    double deadline_s) {
    TORCH_CHECK(partial_4peers.is_contiguous() && partial_4peers.dtype() == at::kBFloat16);
    TORCH_CHECK(out_shard.is_contiguous() && out_shard.dtype() == at::kBFloat16);
    const int cols = static_cast<int>(partial_4peers.size(1));
    TORCH_CHECK((cols & 7) == 0 && partial_4peers.size(0) == 4 * rows_per_rank && out_shard.size(0) == rows_per_rank);
    const PeerTable peers = make_peers(peers_cpu);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(partial_4peers.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(partial_4peers.device().index()).stream();
    xgmi_rs_finish_half_32cu_kernel<<<RS_BLOCKS, RS_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(partial_4peers.data_ptr()),
        reinterpret_cast<bf16_t *>(out_shard.data_ptr()),
        static_cast<int>(peer_base), static_cast<int>(rows_per_rank),
        static_cast<int>(row_off), static_cast<int>(sub_rows), cols,
        static_cast<int>(rank), peers, rs_off, rs_slot_bytes, rs_flags_off,
        block_epoch.data_ptr<int32_t>(), static_cast<int>(epoch_off),
        static_cast<uint64_t>(deadline_s * 1e8));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void fp4_pack_allgather_op(
    const at::Tensor &q_data,
    const at::Tensor &q_scale,
    const at::Tensor &topk_ids,
    const at::Tensor &topk_weights,
    at::Tensor &q_data_all,
    at::Tensor &q_scale_all,
    at::Tensor &topk_ids_all,
    at::Tensor &topk_weights_all,
    const at::Tensor &peers_cpu,
    int64_t rank,
    int64_t ag_off,
    int64_t ag_half_bytes,
    int64_t ag_flags_off,
    at::Tensor &block_epoch,
    int64_t epoch_off,
    double deadline_s) {
    TORCH_CHECK(q_data.is_contiguous() && q_data.dtype() == at::kByte);
    TORCH_CHECK(q_scale.is_contiguous() && q_scale.dtype() == at::kByte);
    TORCH_CHECK(topk_ids.is_contiguous() && topk_ids.dtype() == at::kInt);
    TORCH_CHECK(topk_weights.is_contiguous() && topk_weights.dtype() == at::kFloat);
    const int rows = static_cast<int>(q_data.size(0));
    const int q_bytes = static_cast<int>(q_data.size(1));
    const int s_bytes = static_cast<int>(q_scale.size(1));
    const int top_k = static_cast<int>(topk_ids.size(1));
    TORCH_CHECK((q_bytes & 15) == 0 && (s_bytes & 15) == 0 && ((top_k * 4) & 15) == 0);
    const int64_t packed_total = static_cast<int64_t>(TP) * rows * (q_bytes + s_bytes + 2 * top_k * 4);
    TORCH_CHECK(packed_total <= ag_half_bytes, "packed FP4 AG exceeds AG half buffer");
    const PeerTable peers = make_peers(peers_cpu);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(q_data.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(q_data.device().index()).stream();
    xgmi_fp4_pack_allgather_kernel<<<AG_BLOCKS, AG_THREADS, 0, stream>>>(
        q_data.data_ptr<uint8_t>(),
        q_scale.data_ptr<uint8_t>(),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        q_data_all.data_ptr<uint8_t>(),
        q_scale_all.data_ptr<uint8_t>(),
        topk_ids_all.data_ptr<int32_t>(),
        topk_weights_all.data_ptr<float>(),
        rows, q_bytes >> 4, s_bytes >> 4, (top_k * 4) >> 4,
        static_cast<int>(rank), peers,
        ag_off, ag_half_bytes, ag_flags_off,
        block_epoch.data_ptr<int32_t>(), static_cast<int>(epoch_off),
        static_cast<uint64_t>(deadline_s * 1e8));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void row_allgather_op(
    const at::Tensor &x_shard,
    at::Tensor &out_all,
    const at::Tensor &peers_cpu,
    int64_t rank,
    int64_t ag_off,
    int64_t ag_half_bytes,
    int64_t ag_flags_off,
    at::Tensor &block_epoch,
    int64_t epoch_off,
    double deadline_s) {
    TORCH_CHECK(x_shard.is_contiguous() && x_shard.dtype() == at::kBFloat16);
    TORCH_CHECK(out_all.is_contiguous() && out_all.dtype() == at::kBFloat16);
    const int rows = static_cast<int>(x_shard.size(0));
    const int cols = static_cast<int>(x_shard.size(1));
    TORCH_CHECK((cols & 7) == 0 && out_all.size(0) == rows * TP && out_all.size(1) == cols);
    TORCH_CHECK(static_cast<int64_t>(rows) * TP * cols * 2 <= ag_half_bytes);
    const PeerTable peers = make_peers(peers_cpu);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x_shard.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x_shard.device().index()).stream();
    xgmi_row_allgather_kernel<<<AG_BLOCKS, AG_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16_t *>(x_shard.data_ptr()),
        reinterpret_cast<bf16_t *>(out_all.data_ptr()),
        rows, cols, static_cast<int>(rank), peers,
        ag_off, ag_half_bytes, ag_flags_off,
        block_epoch.data_ptr<int32_t>(), static_cast<int>(epoch_off),
        static_cast<uint64_t>(deadline_s * 1e8));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm_bf16(Tensor x, Tensor w, Tensor(a!) out, float eps) -> ()");
    m.impl("fused_rmsnorm_bf16", at::kCUDA, &fused_rmsnorm_bf16_op);

    m.def("lamport_allgather(Tensor x_slice, Tensor(a!) out_a, Tensor(b!) out_b, int split_cols, "
          "Tensor peers_cpu, int rank, int decode_off, int half_bytes, int ag_sub_off, "
          "Tensor(c!) block_epoch, int epoch_off, float deadline_s) -> ()");
    m.impl("lamport_allgather", at::kCUDA, &lamport_allgather_op);

    m.def("deferred_topk_lamport_ar(Tensor expert_out, Tensor sorted_idx, Tensor topk_weights, "
          "Tensor shared_partial, Tensor(a!) latent_out, Tensor(b!) shared_out, "
          "bool gather_full_shared, Tensor peers_cpu, int rank, int decode_off, int half_bytes, "
          "int ag_sub_off, Tensor(c!) block_epoch, int epoch_off, float deadline_s) -> ()");
    m.impl("deferred_topk_lamport_ar", at::kCUDA, &deferred_topk_lamport_ar_op);

    m.def("moetail_column_gather(Tensor up_slice, Tensor shared_slice, Tensor prefix_slice, "
          "Tensor(a!) stream_out, Tensor peers_cpu, int rank, int decode_off, int half_bytes, "
          "int ag_sub_off, Tensor(b!) block_epoch, int epoch_off, float deadline_s) -> ()");
    m.impl("moetail_column_gather", at::kCUDA, &moetail_column_gather_op);

    m.def("rs_chunk_32cu(Tensor partial, Tensor(a!) out_shard, int rows_per_rank, int row_off, "
          "int sub_rows, int out_col_off, Tensor peers_cpu, int rank, int rs_off, int rs_slot_bytes, "
          "int rs_flags_off, Tensor(b!) block_epoch, int epoch_off, float deadline_s) -> ()");
    m.impl("rs_chunk_32cu", at::kCUDA, &rs_chunk_32cu_op);

    m.def("rs_push_half_32cu(Tensor partial_4peers, int peer_base, int rows_per_rank, int row_off, "
          "int sub_rows, Tensor peers_cpu, int rank, int rs_off, int rs_slot_bytes, int rs_flags_off, "
          "Tensor block_epoch, int epoch_off) -> ()");
    m.impl("rs_push_half_32cu", at::kCUDA, &rs_push_half_32cu_op);

    m.def("rs_finish_half_32cu(Tensor partial_4peers, Tensor(a!) out_shard, int peer_base, "
          "int rows_per_rank, int row_off, int sub_rows, Tensor peers_cpu, int rank, int rs_off, "
          "int rs_slot_bytes, int rs_flags_off, Tensor(b!) block_epoch, int epoch_off, "
          "float deadline_s) -> ()");
    m.impl("rs_finish_half_32cu", at::kCUDA, &rs_finish_half_32cu_op);

    m.def("fp4_pack_allgather(Tensor q_data, Tensor q_scale, Tensor topk_ids, Tensor topk_weights, "
          "Tensor(a!) q_data_all, Tensor(b!) q_scale_all, Tensor(c!) topk_ids_all, "
          "Tensor(d!) topk_weights_all, Tensor peers_cpu, int rank, int ag_off, "
          "int ag_half_bytes, int ag_flags_off, Tensor(e!) block_epoch, int epoch_off, "
          "float deadline_s) -> ()");
    m.impl("fp4_pack_allgather", at::kCUDA, &fp4_pack_allgather_op);

    m.def("row_allgather(Tensor x_shard, Tensor(a!) out_all, Tensor peers_cpu, int rank, "
          "int ag_off, int ag_half_bytes, int ag_flags_off, Tensor(b!) block_epoch, "
          "int epoch_off, float deadline_s) -> ()");
    m.impl("row_allgather", at::kCUDA, &row_allgather_op);
}
