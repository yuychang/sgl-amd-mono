# SPDX-License-Identifier: Apache-2.0
"""Decode (1..192 rows) AMD Iris-style XGMI fused pipeline for Kimi-K3.1."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from sgl_amd_mono.kimi_k31.xgmi.heap import DECODE_MAX_ROWS, TP, XgmiHeap
from sgl_amd_mono.kimi_k31.xgmi.kernels import (
    deferred_topk_lamport_ar,
    fused_rmsnorm_bf16,
    lamport_allgather,
    lamport_allgather_split,
    moetail_column_gather,
)


def _default_gemm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden, weight)


def rms_norm_bf16(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    return fused_rmsnorm_bf16(hidden, weight, eps=eps)


def slice_replicated_weight(
    weight: torch.Tensor, rank: int, tp: int = TP
) -> torch.Tensor:
    """Return this rank's 1/TP output-row view of a replicated [N, K] weight."""
    out_dim = weight.shape[0]
    assert out_dim % tp == 0
    step = out_dim // tp
    sliced = weight[rank * step : (rank + 1) * step]
    return sliced if sliced.is_contiguous() else sliced.contiguous()


def latent_input_gather(
    heap: XgmiHeap,
    normed_hidden: torch.Tensor,
    router_down_weight_slice: torch.Tensor,
    *,
    router_cols: int = 896,
    latent_cols: int = 3584,
    gemm_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1/8 replicated [router; down_proj] GEMM + zero-copy dual-output Lamport AG."""
    assert 1 <= normed_hidden.shape[0] <= DECODE_MAX_ROWS
    gemm = _default_gemm if gemm_fn is None else gemm_fn
    router_slice = router_cols // TP
    local_slice = gemm(normed_hidden, router_down_weight_slice)
    return lamport_allgather_split(heap, local_slice, router_slice)


def latent_down_gather(
    heap: XgmiHeap,
    normed_hidden: torch.Tensor,
    down_weight_slice: torch.Tensor,
    *,
    gemm_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """1/8 replicated latent_down_proj [448, 7168] + Lamport AG -> [M, 3584]."""
    assert 1 <= normed_hidden.shape[0] <= DECODE_MAX_ROWS
    gemm = _default_gemm if gemm_fn is None else gemm_fn
    return lamport_allgather(heap, gemm(normed_hidden, down_weight_slice))


def deferred_topk_ar(
    heap: XgmiHeap,
    expert_out: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_partial: torch.Tensor,
    *,
    sorted_idx: Optional[torch.Tensor] = None,
    gather_full_shared: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused deferred top-k reduction + routed latent AR + shared RS/AR."""
    assert 1 <= topk_weights.shape[0] <= DECODE_MAX_ROWS
    return deferred_topk_lamport_ar(
        heap,
        expert_out,
        topk_weights,
        shared_partial,
        sorted_idx=sorted_idx,
        gather_full_shared=gather_full_shared,
    )


def moetail_gather(
    heap: XgmiHeap,
    latent: torch.Tensor,
    up_weight_slice: torch.Tensor,
    shared_slice: torch.Tensor,
    prefix_slice: Optional[torch.Tensor] = None,
    *,
    latent_norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
    gemm_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """1/8 column-sharded latent_up_proj (8x lower weight BW) + add3 + Lamport AG."""
    assert 1 <= latent.shape[0] <= DECODE_MAX_ROWS
    gemm = _default_gemm if gemm_fn is None else gemm_fn
    if latent_norm_fn is not None:
        normed = latent_norm_fn(latent)
    elif norm_weight is not None:
        normed = rms_norm_bf16(latent, norm_weight, norm_eps)
    else:
        normed = latent
    up_slice = gemm(normed, up_weight_slice)
    return moetail_column_gather(heap, up_slice, shared_slice, prefix_slice)
