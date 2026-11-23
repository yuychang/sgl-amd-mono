# SPDX-License-Identifier: Apache-2.0
"""Opt-in runtime hook wiring AMD Iris-style XGMI fusion into Kimi-K3.1 models."""

from __future__ import annotations

import functools
import os
from typing import Optional

import msgspec
import torch

from sgl_amd_mono.kimi_k31.xgmi.decode import slice_replicated_weight
from sgl_amd_mono.kimi_k31.xgmi.heap import (
    DECODE_MAX_ROWS,
    DEFAULT_DEADLINE_S,
    PREFILL_AG_HALF_BYTES,
    PREFILL_MAX_ROWS,
    TP,
    XgmiConfig,
    XgmiHeap,
    get_global_heap,
)
from sgl_amd_mono.kimi_k31.xgmi.kernels import fp4_pack_allgather, row_allgather
from sgl_amd_mono.kimi_k31.xgmi.prefill import (
    _quant_routed_input,
    moe_tail_mix_qkv_a,
    overlapped_o_proj_rs,
    prefill_reduce_scatter_32cu,
    prefill_row_allgather,
)

SGLANG_READS = {
    "srt/layers/layer_boundary/residual/attn_bank.py": ("AttnBankState", "_FfnReadout"),
    "srt/layers/layer_boundary/residual/attn_bank_hip.py": ("RowShardRead",),
    "srt/models/kimi_k3.py": (
        "KimiK3ForConditionalGeneration",
        "KimiK3LinearForCausalLM",
        "KimiK3LinearModel",
        "KimiK3MLAAttention",
        "KimiK3MoE",
        "_k3_bf16_gemm",
    ),
}

# Stays strictly above SGLANG_ROCM_K3_COLUMN_TAIL_MAX_TOKENS (384) so DSA layers
# during wide verify steps (up to 64x6=384 rows) never enter row-sharded prefill.
DEFAULT_HOOK_PREFILL_MIN = 512
DEFAULT_HOOK_PREFILL_MAX = 32768


class XgmiHookSummary(msgspec.Struct, frozen=True):
    hooked_layers: int
    rebound_banks: int
    qkv_a_links: int
    heap_bytes: int
    heap_mib: float
    prefill_range: tuple[int, int]
    decode_max_rows: int
    use_fp4: bool


def xgmi_enabled() -> bool:
    return os.environ.get("SGLANG_AMD_MONO_K31_XGMI", "0") == "1"


def _resolve_config() -> XgmiConfig:
    fp4_env = os.environ.get("SGLANG_AMD_MONO_K31_XGMI_FP4")
    use_fp4 = (
        fp4_env == "1"
        if fp4_env is not None
        else os.environ.get("AITER_SITUV2_A4W4", "0") == "1"
    )
    return XgmiConfig(
        prefill_min_rows=int(
            os.environ.get(
                "SGLANG_AMD_MONO_K31_XGMI_PREFILL_MIN", str(DEFAULT_HOOK_PREFILL_MIN)
            )
        ),
        prefill_max_rows=int(
            os.environ.get(
                "SGLANG_AMD_MONO_K31_XGMI_PREFILL_MAX", str(DEFAULT_HOOK_PREFILL_MAX)
            )
        ),
        use_fp4=use_fp4,
        enable_decode=os.environ.get("SGLANG_AMD_MONO_K31_XGMI_DECODE", "1") == "1",
        deadline_s=float(
            os.environ.get(
                "SGLANG_AMD_MONO_K31_XGMI_DEADLINE_S", str(DEFAULT_DEADLINE_S)
            )
        ),
    )


def _unwrap_backbone(model):
    from sglang.srt.models.kimi_k3 import (
        KimiK3ForConditionalGeneration,
        KimiK3LinearForCausalLM,
        KimiK3LinearModel,
    )

    if isinstance(model, KimiK3ForConditionalGeneration):
        model = model.language_model
    if isinstance(model, KimiK3LinearForCausalLM):
        model = model.model
    if isinstance(model, KimiK3LinearModel):
        return model
    return model


def _find_attn_bank_state(layer):
    from sglang.srt.layers.layer_boundary.residual.attn_bank import (
        AttnBankState,
        _FfnReadout,
    )

    boundary = layer.ffn_boundary
    if boundary is None:
        return None
    for edges in boundary.plan.edges.values():
        read = edges.incoming.need.read
        if isinstance(read, _FfnReadout) and isinstance(read.state, AttnBankState):
            return read.state
    decl_read = boundary.declaration.read
    if isinstance(decl_read, _FfnReadout) and isinstance(
        decl_read.state, AttnBankState
    ):
        return decl_read.state
    return None


def _attach_o_proj_overlap(layer, heap: XgmiHeap, takes_fn, gemm_fn) -> None:
    from sglang.srt.models.kimi_k3 import KimiK3MLAAttention

    attn = layer.self_attn
    o_proj = attn.o_proj
    if o_proj.weight.dtype != torch.bfloat16:
        return
    has_mla_gate = isinstance(attn, KimiK3MLAAttention) and attn.use_output_gate
    inner_forward = o_proj.forward
    o_proj._xgmi_reduced_rows = None

    def _xgmi_o_proj_forward(hidden, *args, **kwargs):
        if (
            kwargs.get("output_tensor") is None
            and not isinstance(hidden, tuple)
            and hidden.dtype == torch.bfloat16
            and takes_fn(hidden.shape[0])
            and not (has_mla_gate and attn._return_pre_o_proj)
        ):
            proj_in = (
                _apply_mla_gate(attn, inner_forward, hidden, args, kwargs)
                if has_mla_gate
                else hidden
            )
            if proj_in is None:
                return inner_forward(hidden, *args, **kwargs)
            n_chunks = 1 if torch.cuda.is_current_stream_capturing() else None
            o_proj._xgmi_reduced_rows = overlapped_o_proj_rs(
                heap, proj_in, o_proj.weight, gemm_fn=gemm_fn, n_chunks=n_chunks
            )
            return proj_in.new_empty(1, 1).expand(
                proj_in.shape[0], o_proj.weight.shape[0]
            ), None
        return inner_forward(hidden, *args, **kwargs)

    o_proj.forward = _xgmi_o_proj_forward


def _apply_mla_gate(
    attn, inner_forward, hidden, args, kwargs
) -> Optional[torch.Tensor]:
    attn._return_pre_o_proj = True
    try:
        inner_forward(hidden, *args, **kwargs)
        gated = attn._pre_o_proj_out
        attn._pre_o_proj_out = None
    finally:
        attn._return_pre_o_proj = False
    return gated


def _take_xgmi_o_proj_rows(layer, _num_tokens: int) -> Optional[torch.Tensor]:
    o_proj = layer.self_attn.o_proj
    reduced = o_proj._xgmi_reduced_rows
    if reduced is not None:
        o_proj._xgmi_reduced_rows = None
    return reduced


def _make_shard_front(mlp, heap: XgmiHeap, config: XgmiConfig, takes_fn, gemm_fn):
    router_cols = mlp._front_sizes[1]
    rd_weight = mlp._front_w[mlp._front_sizes[0] :]
    orig_shard_front = mlp.shard_front

    def _xgmi_shard_front(normed: torch.Tensor, num_tokens: int) -> bool:
        if not takes_fn(num_tokens):
            return orig_shard_front(normed, num_tokens)
        if (
            not mlp._row_front_on
            or normed.dtype != torch.bfloat16
            or mlp._a8w4_moe_min_tokens is None
            or num_tokens > mlp._a8w4_moe_max_tokens
        ):
            return False

        # Overlap normed_all on comm stream only up to 8192 rows to avoid 450 MB cross-stream blocks.
        need_normed_all = mlp._row_front_shared is None
        overlap_normed = (
            need_normed_all
            and num_tokens <= PREFILL_MAX_ROWS
            and not torch.cuda.is_current_stream_capturing()
        )
        normed_all, comm_keep = _gather_normed_all(
            heap, normed, need_normed_all, overlap_normed
        )

        try:
            fused = gemm_fn(normed, rd_weight, out_dtype=torch.float32)
            router_logits = fused[:, :router_cols].contiguous()
            routed_input = fused[:, router_cols:].to(normed.dtype).contiguous()
            topk_out = mlp.topk(normed, router_logits)
            q_data, q_scale = _quant_routed_input(routed_input, use_fp4=config.use_fp4)
        finally:
            if overlap_normed:
                torch.cuda.current_stream(normed.device).wait_stream(heap.streams.comm)
            comm_keep = ()

        q_all, s_all, ids_all, w_all = fp4_pack_allgather(
            heap,
            q_data,
            q_scale,
            topk_out.topk_ids.to(torch.int32),
            topk_out.topk_weights.to(torch.float32),
        )
        mlp._row_front_out = (
            (q_all[:num_tokens], s_all[:num_tokens]),
            ids_all[:num_tokens],
            w_all[:num_tokens],
            normed,
            normed_all,
        )
        return True

    return _xgmi_shard_front


def _gather_normed_all(
    heap: XgmiHeap,
    normed: torch.Tensor,
    need_normed_all: bool,
    overlap_normed: bool,
) -> tuple[Optional[torch.Tensor], tuple[torch.Tensor, ...]]:
    if not need_normed_all:
        return None, ()
    if not overlap_normed:
        return row_allgather(heap, normed), ()
    # Allocate out_all/tmp on main_stream and keep inputs alive until wait_stream(comm_stream)
    # to avoid cross-stream CUDACachingAllocator pools and record_stream fragmentation.
    normed_c = normed.contiguous()
    rows, cols = normed_c.shape
    out_all = normed_c.new_empty(rows * TP, cols)
    max_rows_per_rank = PREFILL_AG_HALF_BYTES // (TP * cols * 2)
    tmp = (
        normed_c.new_empty(max_rows_per_rank * TP, cols)
        if rows > max_rows_per_rank
        else None
    )
    main_stream = torch.cuda.current_stream(normed_c.device)
    comm_stream = heap.streams.comm
    comm_stream.wait_stream(main_stream)
    try:
        with torch.cuda.stream(comm_stream):
            row_allgather(heap, normed_c, out_all=out_all, tmp=tmp)
    except BaseException:
        main_stream.wait_stream(comm_stream)
        raise
    keep = (normed_c, tmp) if tmp is not None else (normed_c,)
    return out_all, keep


def _attach_next_qkv_a_bypass(next_layer):
    from sglang.srt.models.kimi_k3 import KimiK3MLAAttention

    attn = next_layer.self_attn
    if not isinstance(attn, KimiK3MLAAttention):
        return None
    proj = attn.fused_qkv_a_proj_with_mqa
    if (
        proj is None
        or proj.weight.dtype != torch.bfloat16
        or (proj.weight.shape[0] & 7) != 0
    ):
        return None
    inner = proj.forward
    proj._xgmi_pre_qkv_a = None

    def _xgmi_qkv_a_forward(hidden, *args, **kwargs):
        cached = proj._xgmi_pre_qkv_a
        if cached is not None and cached.shape[0] == hidden.shape[0]:
            proj._xgmi_pre_qkv_a = None
            return cached, None
        return inner(hidden, *args, **kwargs)

    proj.forward = _xgmi_qkv_a_forward
    return proj


def _make_row_sharded_tail(mlp, heap: XgmiHeap, takes_fn):
    orig_rs_tail = mlp._row_sharded_tail

    def _xgmi_row_sharded_tail(
        latent_rows: torch.Tensor,
        shared_rows: torch.Tensor,
        num_tokens: int,
        prefix_sum: torch.Tensor,
    ) -> torch.Tensor:
        if not takes_fn(num_tokens):
            return orig_rs_tail(latent_rows, shared_rows, num_tokens, prefix_sum)
        if shared_rows.shape[0] > num_tokens:
            latent_rows[num_tokens:].zero_()
            shared_rows[num_tokens:].zero_()
        latent = prefill_reduce_scatter_32cu(heap, latent_rows)
        shared = prefill_reduce_scatter_32cu(heap, shared_rows)
        return mlp._row_tail(latent, shared, num_tokens, prefix_sum)

    return _xgmi_row_sharded_tail


def _make_row_tail(mlp, heap: XgmiHeap, takes_fn, next_proj, gemm_fn):
    orig_row_tail = mlp._row_tail

    def _xgmi_row_tail(
        latent: torch.Tensor,
        shared: torch.Tensor,
        num_tokens: int,
        prefix_sum: torch.Tensor,
    ) -> torch.Tensor:
        if not takes_fn(num_tokens) or mlp._row_stream:
            return orig_row_tail(latent, shared, num_tokens, prefix_sum)
        rows = shared.shape[0]
        start = heap.rank * rows
        pref = (
            prefix_sum[start : min(start + rows, num_tokens)]
            if prefix_sum.shape[0] == num_tokens
            else prefix_sum
        )
        if pref.shape[0] < rows:
            pref = torch.nn.functional.pad(pref, (0, 0, 0, rows - pref.shape[0]))
        stream_sh, _, qkv_a_all = moe_tail_mix_qkv_a(
            heap,
            latent,
            shared,
            pref,
            num_tokens,
            latent_norm_fn=mlp._latent_norm,
            up_proj_weight=mlp.routed_expert_up_proj.weight,
            next_qkv_a_weight=next_proj.weight,
            gemm_fn=gemm_fn,
        )
        next_proj._xgmi_pre_qkv_a = qkv_a_all
        return prefill_row_allgather(heap, stream_sh, num_tokens)

    return _xgmi_row_tail


def _bind_decode_slices(mlp, rank: int) -> None:
    if mlp.routed_expert_up_proj is not None:
        up_w = mlp.routed_expert_up_proj.weight
        if (
            isinstance(up_w, torch.Tensor)
            and up_w.dtype == torch.bfloat16
            and up_w.shape[0] % TP == 0
        ):
            mlp._xgmi_up_weight_slice = slice_replicated_weight(up_w, rank, TP)
    if mlp._front_sizes is not None and len(mlp._front_sizes) == 3:
        gate_sz, router_sz, latent_sz = mlp._front_sizes
        router_w = mlp._front_w[gate_sz : gate_sz + router_sz]
        down_w = mlp._front_w[gate_sz + router_sz : gate_sz + router_sz + latent_sz]
        if router_w.shape[0] % TP == 0 and down_w.shape[0] % TP == 0:
            mlp._xgmi_router_weight_slice = slice_replicated_weight(router_w, rank, TP)
            mlp._xgmi_down_weight_slice = slice_replicated_weight(down_w, rank, TP)
            mlp._xgmi_router_cols = router_sz
            mlp._xgmi_latent_cols = latent_sz


def install_xgmi_hooks(
    model,
    *,
    cpu_group=None,
    rank: Optional[int] = None,
    device: Optional[torch.device] = None,
    heap: Optional[XgmiHeap] = None,
    config: Optional[XgmiConfig] = None,
) -> XgmiHookSummary:
    """Install AMD Iris-style XGMI Decode + Prefill hooks on a Kimi-K3.1 model."""
    from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import RowShardRead
    from sglang.srt.models.kimi_k3 import KimiK3MoE, _k3_bf16_gemm

    cfg = _resolve_config() if config is None else config
    backbone = _unwrap_backbone(model)
    layers = backbone.layers
    if heap is None:
        heap = get_global_heap(cpu_group=cpu_group, rank=rank, device=device)

    hooked = 0
    rebound_banks = 0
    qkv_links = 0

    def takes_xgmi(num_tokens: int) -> bool:
        return cfg.prefill_min_rows <= num_tokens <= cfg.prefill_max_rows

    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        if not isinstance(mlp, KimiK3MoE) or not mlp.use_latent_moe:
            continue

        orig_elig = mlp._row_sharded_tail_eligible

        def _tail_eligible(
            num_tokens: int, has_prefix: bool, _mlp=mlp, _orig=orig_elig
        ) -> bool:
            if (
                has_prefix
                and takes_xgmi(num_tokens)
                and _mlp.routed_expert_up_proj is not None
            ):
                return True
            return _orig(num_tokens, has_prefix)

        def _takes_any(num_tokens: int, _orig=orig_elig) -> bool:
            return takes_xgmi(num_tokens) or bool(_orig(num_tokens, True))

        mlp._row_sharded_tail_eligible = _tail_eligible

        if mlp._front_sizes is not None and len(mlp._front_sizes) == 3:
            shard_front_fn = _make_shard_front(
                mlp, heap, cfg, takes_xgmi, _k3_bf16_gemm
            )
            mlp.shard_front = shard_front_fn
            attn_state = _find_attn_bank_state(layer)
            if attn_state is not None and attn_state.ffn_row_shard is not None:
                _attach_o_proj_overlap(layer, heap, takes_xgmi, _k3_bf16_gemm)
                attn_state.ffn_row_shard = RowShardRead(
                    _takes_any,
                    shard_front=shard_front_fn,
                    attn_rows=functools.partial(_take_xgmi_o_proj_rows, layer),
                )
                rebound_banks += 1

        mlp._row_sharded_tail = _make_row_sharded_tail(mlp, heap, takes_xgmi)

        if layer_idx + 1 < len(layers):
            next_proj = _attach_next_qkv_a_bypass(layers[layer_idx + 1])
            if next_proj is not None and mlp.routed_expert_up_proj is not None:
                mlp._row_tail = _make_row_tail(
                    mlp, heap, takes_xgmi, next_proj, _k3_bf16_gemm
                )
                qkv_links += 1

        if cfg.enable_decode:
            _bind_decode_slices(mlp, heap.rank)

        hooked += 1

    model._xgmi_heap = heap
    return XgmiHookSummary(
        hooked_layers=hooked,
        rebound_banks=rebound_banks,
        qkv_a_links=qkv_links,
        heap_bytes=heap.heap_bytes,
        heap_mib=heap.heap_mib,
        prefill_range=(cfg.prefill_min_rows, cfg.prefill_max_rows),
        decode_max_rows=DECODE_MAX_ROWS if cfg.enable_decode else 0,
        use_fp4=cfg.use_fp4,
    )


def maybe_install_xgmi_hooks(
    model, *, cpu_group, rank: int, device: torch.device, log_fn=None
) -> Optional[XgmiHookSummary]:
    if not xgmi_enabled():
        return None
    summary = install_xgmi_hooks(model, cpu_group=cpu_group, rank=rank, device=device)
    if log_fn is not None:
        log_fn(
            f"xgmi: hooked {summary.hooked_layers} MoE layers "
            f"(rebound {summary.rebound_banks} banks, {summary.qkv_a_links} qkv_a links), "
            f"heap {summary.heap_mib:.1f} MiB, prefill {summary.prefill_range}, fp4={summary.use_fp4}"
        )
    return summary
