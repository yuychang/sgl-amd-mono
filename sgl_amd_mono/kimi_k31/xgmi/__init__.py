# SPDX-License-Identifier: Apache-2.0
"""AMD Iris-style XGMI full-pipeline fusion for Kimi-K3.1 on 8x MI355X (gfx950)."""

from sgl_amd_mono.kimi_k31.xgmi.decode import (
    deferred_topk_ar,
    latent_down_gather,
    latent_input_gather,
    moetail_gather,
    rms_norm_bf16,
    slice_replicated_weight,
)
from sgl_amd_mono.kimi_k31.xgmi.heap import (
    DECODE_BLOCKS,
    DECODE_MAX_ROWS,
    DEFAULT_DEADLINE_S,
    HEAP_BYTES,
    PREFILL_AG_BLOCKS,
    PREFILL_MAX_ROWS,
    RS_BLOCKS,
    RS_MAX_CHUNK_ROWS,
    TP,
    CuMaskStreams,
    XgmiConfig,
    XgmiHeap,
    get_global_heap,
)
from sgl_amd_mono.kimi_k31.xgmi.hook import (
    XgmiHookSummary,
    install_xgmi_hooks,
    maybe_install_xgmi_hooks,
    xgmi_enabled,
)
from sgl_amd_mono.kimi_k31.xgmi.prefill import (
    PREFILL_MIN_ROWS,
    moe_tail_mix_qkv_a,
    overlapped_o_proj_rs,
    prefill_reduce_scatter_32cu,
    prefill_row_allgather,
    sharded_front_fp4_ag,
)

__all__ = [
    "TP",
    "HEAP_BYTES",
    "DEFAULT_DEADLINE_S",
    "DECODE_BLOCKS",
    "DECODE_MAX_ROWS",
    "RS_BLOCKS",
    "RS_MAX_CHUNK_ROWS",
    "PREFILL_AG_BLOCKS",
    "PREFILL_MIN_ROWS",
    "PREFILL_MAX_ROWS",
    "CuMaskStreams",
    "XgmiConfig",
    "XgmiHeap",
    "XgmiHookSummary",
    "get_global_heap",
    "latent_input_gather",
    "latent_down_gather",
    "deferred_topk_ar",
    "moetail_gather",
    "rms_norm_bf16",
    "slice_replicated_weight",
    "overlapped_o_proj_rs",
    "sharded_front_fp4_ag",
    "moe_tail_mix_qkv_a",
    "prefill_reduce_scatter_32cu",
    "prefill_row_allgather",
    "install_xgmi_hooks",
    "maybe_install_xgmi_hooks",
    "xgmi_enabled",
]
