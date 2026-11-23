# SPDX-License-Identifier: Apache-2.0
"""Prefill (200..32768 rows) Row-Sharded Sequence-Parallel XGMI pipeline."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from sgl_amd_mono.kimi_k31.xgmi.heap import PREFILL_MAX_ROWS, TP, XgmiHeap
from sgl_amd_mono.kimi_k31.xgmi.kernels import (
    fp4_pack_allgather,
    reduce_scatter_rows,
    row_allgather,
)

PREFILL_MIN_ROWS = 200


def _default_gemm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden, weight)


def pad_rows_to_tp(hidden: torch.Tensor, tp: int = TP) -> tuple[torch.Tensor, int]:
    num_tokens = hidden.shape[0]
    padded = ((num_tokens + tp - 1) // tp) * tp
    if padded == num_tokens:
        return hidden.contiguous(), num_tokens
    return F.pad(hidden, (0, 0, 0, padded - num_tokens)).contiguous(), num_tokens


def overlapped_o_proj_rs(
    heap: XgmiHeap,
    hidden: torch.Tensor,
    o_proj_weight: torch.Tensor,
    *,
    gemm_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    n_chunks: Optional[int] = None,
) -> torch.Tensor:
    """TP-sharded o_proj GEMM + 7-link-parallel XGMI Reduce-Scatter (0 extra weight).

    Measured on 8x MI355X: single-stream 1-shot RS beats RCCL up to M=4096 and avoids
    cross-stream CUDACachingAllocator fragmentation above M=8192; 4-chunk N-overlap wins
    in the 4097..8192 window where each chunk fits in one 28 MiB RS ring slot.
    """
    gemm = _default_gemm if gemm_fn is None else gemm_fn
    x_pad, _ = pad_rows_to_tp(hidden, TP)
    padded_rows = x_pad.shape[0]
    rows_per_rank = padded_rows // TP
    out_cols = o_proj_weight.shape[0]

    if n_chunks is None:
        n_chunks = 4 if 4096 < padded_rows <= PREFILL_MAX_ROWS else 1
    while n_chunks > 1 and (out_cols % (n_chunks * 8) != 0):
        n_chunks //= 2

    if n_chunks == 1:
        partial = gemm(x_pad, o_proj_weight)
        return reduce_scatter_rows(heap, partial)

    out_shard = x_pad.new_empty(rows_per_rank, out_cols)
    chunk_cols = out_cols // n_chunks
    main_stream = torch.cuda.current_stream(hidden.device)
    comm_stream = heap.streams.comm
    events = heap.chunk_events
    # Keep all chunk partials alive until wait_stream(comm_stream) so we never
    # call record_stream(comm_stream) or allocate on comm_stream.
    part_chunks: list[torch.Tensor] = []

    try:
        for chunk_idx in range(n_chunks):
            col_off = chunk_idx * chunk_cols
            part_chunk = gemm(
                x_pad, o_proj_weight[col_off : col_off + chunk_cols]
            ).contiguous()
            part_chunks.append(part_chunk)
            events[chunk_idx].record(main_stream)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(events[chunk_idx])
                reduce_scatter_rows(
                    heap, part_chunk, out_shard=out_shard, out_col_off=col_off
                )
    finally:
        main_stream.wait_stream(comm_stream)
        part_chunks.clear()
    return out_shard


def sharded_front_fp4_ag(
    heap: XgmiHeap,
    normed_shard: torch.Tensor,
    num_tokens: int,
    router_down_weight: torch.Tensor,
    router_cols: int,
    topk_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    *,
    use_fp4: bool = True,
    gemm_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """1/8 row-sharded front + MXFP4/MXFP8 quant + fused XGMI AllGather."""
    gemm = _default_gemm if gemm_fn is None else gemm_fn
    fused = gemm(normed_shard, router_down_weight)
    router_logits = fused[:, :router_cols].contiguous()
    routed_input = fused[:, router_cols:].to(normed_shard.dtype).contiguous()
    topk_ids, topk_weights = topk_fn(normed_shard, router_logits)
    q_data, q_scale = _quant_routed_input(routed_input, use_fp4=use_fp4)
    q_all, s_all, ids_all, w_all = fp4_pack_allgather(
        heap, q_data, q_scale, topk_ids, topk_weights
    )
    return (
        q_all[:num_tokens],
        s_all[:num_tokens],
        ids_all[:num_tokens],
        w_all[:num_tokens],
    )


def _quant_routed_input(
    routed_input: torch.Tensor, *, use_fp4: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_fp4:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        q_data, q_scale = dynamic_mxfp4_quant(routed_input)
        return q_data.view(torch.uint8), q_scale.view(torch.uint8)
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    q_data, q_scale = per_1x32_mx_quant_hip(
        routed_input, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    return q_data.view(torch.uint8), q_scale.view(torch.uint8)


def moe_tail_mix_qkv_a(
    heap: XgmiHeap,
    latent_shard: torch.Tensor,
    shared_shard: torch.Tensor,
    prefix_shard: Optional[torch.Tensor],
    num_tokens: int,
    *,
    latent_norm_fn: Callable[[torch.Tensor], torch.Tensor],
    up_proj_weight: torch.Tensor,
    next_attn_norm_fn: Optional[
        Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    ] = None,
    next_qkv_a_weight: Optional[torch.Tensor] = None,
    gemm_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """1/8 row-sharded MoE tail + optional pre-projection of next MLA 2112-dim qkv_a."""
    gemm = _default_gemm if gemm_fn is None else gemm_fn
    stream_shard = gemm(latent_norm_fn(latent_shard), up_proj_weight) + shared_shard
    if prefix_shard is not None:
        stream_shard = stream_shard + prefix_shard
    if next_qkv_a_weight is None:
        return stream_shard, None, None
    if next_attn_norm_fn is not None:
        next_normed_shard, next_written = next_attn_norm_fn(stream_shard)
    else:
        next_normed_shard, next_written = stream_shard, stream_shard
    qkv_a_shard = gemm(next_normed_shard, next_qkv_a_weight)
    qkv_a_all = row_allgather(heap, qkv_a_shard)[:num_tokens]
    return stream_shard, next_written, qkv_a_all


def prefill_reduce_scatter_32cu(
    heap: XgmiHeap,
    partial: torch.Tensor,
    *,
    out_shard: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return reduce_scatter_rows(heap, partial, out_shard=out_shard)


def prefill_row_allgather(
    heap: XgmiHeap,
    x_shard: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    return row_allgather(heap, x_shard)[:num_tokens]
