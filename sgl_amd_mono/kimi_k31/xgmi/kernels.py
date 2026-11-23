# SPDX-License-Identifier: Apache-2.0
"""Python bindings for gfx950 XGMI fused HIP kernels (``xgmi_ops.hip``)."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import torch

from sgl_amd_mono.common import hip_ext
from sgl_amd_mono.kimi_k31.xgmi.heap import (
    AG_FLAGS_OFF,
    AG_REGION_OFF,
    DECODE_HALF_BYTES,
    DECODE_REGION_OFF,
    DECODE_RS_BYTES,
    EPOCH_AG_OFF,
    EPOCH_DECODE_OFF,
    EPOCH_RS_OFF,
    PREFILL_AG_HALF_BYTES,
    RS_FLAGS_OFF,
    RS_REGION_OFF,
    RS_SLOT_BYTES,
    TP,
    XgmiHeap,
)

_EMPTY_INT: dict[torch.device, torch.Tensor] = {}
_EMPTY_BF16: dict[torch.device, torch.Tensor] = {}


def _empty_int(device: torch.device) -> torch.Tensor:
    tensor = _EMPTY_INT.get(device)
    if tensor is None:
        tensor = torch.empty(0, dtype=torch.int32, device=device)
        _EMPTY_INT[device] = tensor
    return tensor


def _empty_bf16(device: torch.device) -> torch.Tensor:
    tensor = _EMPTY_BF16.get(device)
    if tensor is None:
        tensor = torch.empty(0, dtype=torch.bfloat16, device=device)
        _EMPTY_BF16[device] = tensor
    return tensor


@functools.cache
def ops():
    """Compile (once, cached) and load ``xgmi_ops.hip`` for gfx950."""
    src_path = Path(__file__).with_name("xgmi_ops.hip")
    return hip_ext.load(
        "k31_xgmi_ops",
        {"xgmi_ops.hip": src_path.read_text(encoding="utf-8")},
        arch="gfx950",
    )


def fused_rmsnorm_bf16(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    hidden = hidden.contiguous()
    if out is None:
        out = torch.empty_like(hidden)
    ops().fused_rmsnorm_bf16(hidden, weight.contiguous(), out, eps)
    return out


def lamport_allgather(
    heap: XgmiHeap,
    x_slice: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x_slice = x_slice.contiguous()
    num_tokens, slice_cols = x_slice.shape
    if out is None:
        out = x_slice.new_empty(num_tokens, slice_cols * TP)
    ops().lamport_allgather(
        x_slice,
        out,
        _empty_bf16(x_slice.device),
        0,
        heap.peer_ptrs_cpu,
        heap.rank,
        DECODE_REGION_OFF,
        DECODE_HALF_BYTES,
        DECODE_RS_BYTES,
        heap.block_epoch,
        EPOCH_DECODE_OFF,
        heap.deadline_s,
    )
    return out


def lamport_allgather_split(
    heap: XgmiHeap,
    x_slice: torch.Tensor,
    split_cols: int,
    *,
    out_first: Optional[torch.Tensor] = None,
    out_second: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_slice = x_slice.contiguous()
    num_tokens, slice_cols = x_slice.shape
    second_slice = slice_cols - split_cols
    if out_first is None:
        out_first = x_slice.new_empty(num_tokens, split_cols * TP)
    if out_second is None:
        out_second = x_slice.new_empty(num_tokens, second_slice * TP)
    ops().lamport_allgather(
        x_slice,
        out_first,
        out_second,
        split_cols,
        heap.peer_ptrs_cpu,
        heap.rank,
        DECODE_REGION_OFF,
        DECODE_HALF_BYTES,
        DECODE_RS_BYTES,
        heap.block_epoch,
        EPOCH_DECODE_OFF,
        heap.deadline_s,
    )
    return out_first, out_second


def deferred_topk_lamport_ar(
    heap: XgmiHeap,
    expert_out: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_partial: torch.Tensor,
    *,
    sorted_idx: Optional[torch.Tensor] = None,
    gather_full_shared: bool = False,
    latent_out: Optional[torch.Tensor] = None,
    shared_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_out = expert_out.contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    shared_partial = shared_partial.contiguous()
    num_tokens = topk_weights.shape[0]
    latent_dim = expert_out.shape[-1]
    shared_dim = shared_partial.shape[-1]
    if latent_out is None:
        latent_out = shared_partial.new_empty(num_tokens, latent_dim)
    if shared_out is None:
        shared_out = shared_partial.new_empty(
            num_tokens, shared_dim if gather_full_shared else shared_dim // TP
        )
    sidx = (
        _empty_int(shared_partial.device)
        if sorted_idx is None
        else sorted_idx.to(torch.int32).contiguous()
    )
    ops().deferred_topk_lamport_ar(
        expert_out,
        sidx,
        topk_weights,
        shared_partial,
        latent_out,
        shared_out,
        gather_full_shared,
        heap.peer_ptrs_cpu,
        heap.rank,
        DECODE_REGION_OFF,
        DECODE_HALF_BYTES,
        DECODE_RS_BYTES,
        heap.block_epoch,
        EPOCH_DECODE_OFF,
        heap.deadline_s,
    )
    return latent_out, shared_out


def moetail_column_gather(
    heap: XgmiHeap,
    up_slice: torch.Tensor,
    shared_slice: torch.Tensor,
    prefix_slice: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    up_slice = up_slice.contiguous()
    shared_slice = shared_slice.contiguous()
    num_tokens, slice_cols = up_slice.shape
    if out is None:
        out = up_slice.new_empty(num_tokens, slice_cols * TP)
    pref = (
        _empty_bf16(up_slice.device)
        if prefix_slice is None
        else prefix_slice.contiguous()
    )
    ops().moetail_column_gather(
        up_slice,
        shared_slice,
        pref,
        out,
        heap.peer_ptrs_cpu,
        heap.rank,
        DECODE_REGION_OFF,
        DECODE_HALF_BYTES,
        DECODE_RS_BYTES,
        heap.block_epoch,
        EPOCH_DECODE_OFF,
        heap.deadline_s,
    )
    return out


def reduce_scatter_rows(
    heap: XgmiHeap,
    partial: torch.Tensor,
    *,
    out_shard: Optional[torch.Tensor] = None,
    sub_rows_cap: Optional[int] = None,
    out_col_off: int = 0,
) -> torch.Tensor:
    partial = partial.contiguous()
    total_rows, cols = partial.shape
    assert total_rows % TP == 0
    rows_per_rank = total_rows // TP
    if out_shard is None:
        out_shard = partial.new_empty(rows_per_rank, cols)
    max_sub = RS_SLOT_BYTES // (TP * cols * 2)
    if sub_rows_cap is not None:
        max_sub = min(max_sub, sub_rows_cap)
    for row_off in range(0, rows_per_rank, max_sub):
        sub_rows = min(max_sub, rows_per_rank - row_off)
        ops().rs_chunk_32cu(
            partial,
            out_shard,
            rows_per_rank,
            row_off,
            sub_rows,
            out_col_off,
            heap.peer_ptrs_cpu,
            heap.rank,
            RS_REGION_OFF,
            RS_SLOT_BYTES,
            RS_FLAGS_OFF,
            heap.block_epoch,
            EPOCH_RS_OFF,
            heap.deadline_s,
        )
    return out_shard


def fp4_pack_allgather(
    heap: XgmiHeap,
    q_data: torch.Tensor,
    q_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_data = q_data.view(torch.uint8).contiguous()
    q_scale = q_scale.view(torch.uint8).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    rows = q_data.shape[0]
    q_cols, s_cols, top_k = q_data.shape[1], q_scale.shape[1], topk_ids.shape[1]
    row_bytes = q_cols + s_cols + 2 * top_k * 4
    max_rows_per_rank = PREFILL_AG_HALF_BYTES // (TP * row_bytes)
    q_data_all = q_data.new_empty(rows * TP, q_cols)
    q_scale_all = q_scale.new_empty(rows * TP, s_cols)
    topk_ids_all = topk_ids.new_empty(rows * TP, top_k)
    topk_weights_all = topk_weights.new_empty(rows * TP, top_k)
    if rows <= max_rows_per_rank:
        ops().fp4_pack_allgather(
            q_data,
            q_scale,
            topk_ids,
            topk_weights,
            q_data_all,
            q_scale_all,
            topk_ids_all,
            topk_weights_all,
            heap.peer_ptrs_cpu,
            heap.rank,
            AG_REGION_OFF,
            PREFILL_AG_HALF_BYTES,
            AG_FLAGS_OFF,
            heap.block_epoch,
            EPOCH_AG_OFF,
            heap.deadline_s,
        )
        return q_data_all, q_scale_all, topk_ids_all, topk_weights_all
    return _fp4_pack_allgather_chunked(
        heap=heap,
        q_data=q_data,
        q_scale=q_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        q_data_all=q_data_all,
        q_scale_all=q_scale_all,
        topk_ids_all=topk_ids_all,
        topk_weights_all=topk_weights_all,
        max_rows_per_rank=max_rows_per_rank,
    )


def _fp4_pack_allgather_chunked(
    *,
    heap: XgmiHeap,
    q_data: torch.Tensor,
    q_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    q_data_all: torch.Tensor,
    q_scale_all: torch.Tensor,
    topk_ids_all: torch.Tensor,
    topk_weights_all: torch.Tensor,
    max_rows_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = q_data.shape[0]
    q_cols, s_cols, top_k = q_data.shape[1], q_scale.shape[1], topk_ids.shape[1]
    tmp_q = q_data.new_empty(max_rows_per_rank * TP, q_cols)
    tmp_s = q_scale.new_empty(max_rows_per_rank * TP, s_cols)
    tmp_ids = topk_ids.new_empty(max_rows_per_rank * TP, top_k)
    tmp_w = topk_weights.new_empty(max_rows_per_rank * TP, top_k)
    for row_start in range(0, rows, max_rows_per_rank):
        row_end = min(row_start + max_rows_per_rank, rows)
        sub_rows = row_end - row_start
        sub_q = tmp_q[: sub_rows * TP]
        sub_s = tmp_s[: sub_rows * TP]
        sub_ids = tmp_ids[: sub_rows * TP]
        sub_w = tmp_w[: sub_rows * TP]
        ops().fp4_pack_allgather(
            q_data[row_start:row_end].contiguous(),
            q_scale[row_start:row_end].contiguous(),
            topk_ids[row_start:row_end].contiguous(),
            topk_weights[row_start:row_end].contiguous(),
            sub_q,
            sub_s,
            sub_ids,
            sub_w,
            heap.peer_ptrs_cpu,
            heap.rank,
            AG_REGION_OFF,
            PREFILL_AG_HALF_BYTES,
            AG_FLAGS_OFF,
            heap.block_epoch,
            EPOCH_AG_OFF,
            heap.deadline_s,
        )
        q_data_all.view(TP, rows, q_cols)[:, row_start:row_end, :].copy_(
            sub_q.view(TP, sub_rows, q_cols)
        )
        q_scale_all.view(TP, rows, s_cols)[:, row_start:row_end, :].copy_(
            sub_s.view(TP, sub_rows, s_cols)
        )
        topk_ids_all.view(TP, rows, top_k)[:, row_start:row_end, :].copy_(
            sub_ids.view(TP, sub_rows, top_k)
        )
        topk_weights_all.view(TP, rows, top_k)[:, row_start:row_end, :].copy_(
            sub_w.view(TP, sub_rows, top_k)
        )
    return q_data_all, q_scale_all, topk_ids_all, topk_weights_all


def row_allgather(
    heap: XgmiHeap,
    x_shard: torch.Tensor,
    *,
    out_all: Optional[torch.Tensor] = None,
    tmp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x_shard = x_shard.contiguous()
    rows, cols = x_shard.shape
    if out_all is None:
        out_all = x_shard.new_empty(rows * TP, cols)
    max_rows_per_rank = PREFILL_AG_HALF_BYTES // (TP * cols * 2)
    if rows <= max_rows_per_rank:
        ops().row_allgather(
            x_shard,
            out_all,
            heap.peer_ptrs_cpu,
            heap.rank,
            AG_REGION_OFF,
            PREFILL_AG_HALF_BYTES,
            AG_FLAGS_OFF,
            heap.block_epoch,
            EPOCH_AG_OFF,
            heap.deadline_s,
        )
        return out_all
    if tmp is None:
        tmp = x_shard.new_empty(max_rows_per_rank * TP, cols)
    else:
        assert (
            tmp.shape[0] >= max_rows_per_rank * TP
            and tmp.shape[1] == cols
            and tmp.dtype == x_shard.dtype
            and tmp.is_contiguous()
        ), (tuple(tmp.shape), tmp.dtype, max_rows_per_rank * TP, cols, x_shard.dtype)
    for row_start in range(0, rows, max_rows_per_rank):
        row_end = min(row_start + max_rows_per_rank, rows)
        sub_rows = row_end - row_start
        sub_out = tmp[: sub_rows * TP]
        ops().row_allgather(
            x_shard[row_start:row_end],
            sub_out,
            heap.peer_ptrs_cpu,
            heap.rank,
            AG_REGION_OFF,
            PREFILL_AG_HALF_BYTES,
            AG_FLAGS_OFF,
            heap.block_epoch,
            EPOCH_AG_OFF,
            heap.deadline_s,
        )
        out_all.view(TP, rows, cols)[:, row_start:row_end, :].copy_(
            sub_out.view(TP, sub_rows, cols)
        )
    return out_all
