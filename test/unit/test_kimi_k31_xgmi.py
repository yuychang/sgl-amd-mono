# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Kimi-K3.1 AMD Iris-style XGMI heap layout, slicing, and hooks."""

from __future__ import annotations

import types

import torch

from sgl_amd_mono.kimi_k31.xgmi import (
    DECODE_BLOCKS,
    DECODE_MAX_ROWS,
    DEFAULT_DEADLINE_S,
    HEAP_BYTES,
    PREFILL_AG_BLOCKS,
    PREFILL_MAX_ROWS,
    RS_BLOCKS,
    TP,
    XgmiHookSummary,
    install_xgmi_hooks,
    slice_replicated_weight,
    xgmi_enabled,
)
from sgl_amd_mono.kimi_k31.xgmi.heap import (
    AG_FLAGS_BYTES,
    AG_FLAGS_OFF,
    AG_REGION_BYTES,
    AG_REGION_OFF,
    DECODE_REGION_BYTES,
    DECODE_REGION_OFF,
    EPOCH_TOTAL_WORDS,
    PREFILL_AG_HALF_BYTES,
    RS_FLAGS_BYTES,
    RS_FLAGS_OFF,
    RS_REGION_BYTES,
    RS_REGION_OFF,
    RS_SLOT_BYTES,
)


def test_heap_budget_under_400mb():
    assert HEAP_BYTES <= int(0.4 * (1 << 30))
    assert HEAP_BYTES % (2 << 20) == 0
    assert DECODE_REGION_OFF + DECODE_REGION_BYTES <= RS_REGION_OFF
    assert RS_REGION_OFF + RS_REGION_BYTES <= AG_REGION_OFF
    assert AG_REGION_OFF + AG_REGION_BYTES <= RS_FLAGS_OFF
    assert RS_FLAGS_OFF + RS_FLAGS_BYTES <= AG_FLAGS_OFF
    assert AG_FLAGS_OFF + AG_FLAGS_BYTES <= HEAP_BYTES
    assert PREFILL_MAX_ROWS * 2032 <= PREFILL_AG_HALF_BYTES
    assert PREFILL_MAX_ROWS * 3824 <= PREFILL_AG_HALF_BYTES
    assert PREFILL_MAX_ROWS * (2112 * 2) <= PREFILL_AG_HALF_BYTES
    assert PREFILL_MAX_ROWS * 1792 * 2 <= RS_SLOT_BYTES
    assert EPOCH_TOTAL_WORDS == DECODE_BLOCKS + RS_BLOCKS + PREFILL_AG_BLOCKS
    assert DEFAULT_DEADLINE_S >= 900.0


def test_zero_copy_weight_slice():
    weight = torch.randn(7168, 3584, dtype=torch.bfloat16)
    sliced = slice_replicated_weight(weight, rank=3, tp=TP)
    assert sliced.shape == (896, 3584)
    assert sliced.is_contiguous()
    assert sliced.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()


def test_xgmi_opt_in_default_off(monkeypatch):
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_XGMI", raising=False)
    assert not xgmi_enabled()
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_XGMI", "1")
    assert xgmi_enabled()


def test_install_xgmi_hooks_rebinds_bank_and_preserves_mla_gate():
    from sglang.srt.layers.layer_boundary.residual.attn_bank import (
        AttnBankState,
        _FfnReadout,
    )
    from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import RowShardRead
    from sglang.srt.models.kimi_k3 import KimiK3MLAAttention, KimiK3MoE

    class DummyHeap:
        rank = 0
        world_size = TP
        heap_bytes = HEAP_BYTES
        heap_mib = HEAP_BYTES / (1 << 20)

    def make_layer(is_mla: bool):
        o_proj = types.SimpleNamespace(
            weight=torch.empty(7168, 1536, dtype=torch.bfloat16),
            forward=lambda hidden, *args, **kwargs: (hidden, None),
        )
        if is_mla:
            attn = object.__new__(KimiK3MLAAttention)
            attn.o_proj = o_proj
            attn.use_output_gate = True
            attn._return_pre_o_proj = False
            attn._pre_o_proj_out = None
            attn.fused_qkv_a_proj_with_mqa = types.SimpleNamespace(
                weight=torch.empty(2112, 7168, dtype=torch.bfloat16),
                forward=lambda hidden, *args, **kwargs: (hidden, None),
            )
        else:
            attn = types.SimpleNamespace(o_proj=o_proj)

        mlp = object.__new__(KimiK3MoE)
        mlp.use_latent_moe = True
        mlp.routed_expert_up_proj = types.SimpleNamespace(
            weight=torch.empty(7168, 3584, dtype=torch.bfloat16)
        )
        mlp._front_w = torch.empty(768 + 896 + 3584, 7168, dtype=torch.bfloat16)
        mlp._front_sizes = (768, 896, 3584)
        mlp._row_front_on = True
        mlp._row_front_shared = None
        mlp._row_stream = False
        mlp._a8w4_moe_min_tokens = 256
        mlp._a8w4_moe_max_tokens = 65536
        mlp._row_sharded_tail_eligible = lambda num_tokens, has_prefix: (
            num_tokens >= 4096 and has_prefix
        )
        mlp.shard_front = lambda normed, num_tokens: False
        mlp._row_sharded_tail = lambda *args: None
        mlp._row_tail = lambda *args: None
        mlp._latent_norm = lambda tensor: tensor

        bank_state = AttnBankState(
            bank=None,
            attn_score_proj=None,
            attn_score_norm=None,
            ffn_score_proj=None,
            ffn_score_norm=None,
            ffn_row_shard=RowShardRead(takes=lambda num_tokens: num_tokens >= 4096),
        )
        readout = _FfnReadout(bank_state)
        edge = types.SimpleNamespace(
            incoming=types.SimpleNamespace(need=types.SimpleNamespace(read=readout))
        )
        boundary = types.SimpleNamespace(
            plan=types.SimpleNamespace(edges={"default": edge}),
            declaration=types.SimpleNamespace(read=readout),
        )
        return types.SimpleNamespace(self_attn=attn, mlp=mlp, ffn_boundary=boundary)

    model = types.SimpleNamespace(layers=[make_layer(False), make_layer(True)])
    summary = install_xgmi_hooks(model, heap=DummyHeap())
    assert isinstance(summary, XgmiHookSummary)
    assert summary.hooked_layers == 2
    assert summary.rebound_banks == 2
    assert summary.qkv_a_links == 1
    assert summary.prefill_range == (512, 32768)
    assert summary.decode_max_rows == DECODE_MAX_ROWS
    assert model.layers[0].mlp._row_sharded_tail_eligible(512, True)
    assert model.layers[0].mlp._row_sharded_tail_eligible(32768, True)
    assert not model.layers[0].mlp._row_sharded_tail_eligible(384, True)
    assert (
        model.layers[0].mlp._xgmi_up_weight_slice.untyped_storage().data_ptr()
        == model.layers[0].mlp.routed_expert_up_proj.weight.untyped_storage().data_ptr()
    )
    proj = model.layers[1].self_attn.fused_qkv_a_proj_with_mqa
    fake_qkv = torch.randn(256, 2112, dtype=torch.bfloat16)
    proj._xgmi_pre_qkv_a = fake_qkv
    out, _ = proj.forward(torch.empty(256, 7168, dtype=torch.bfloat16))
    assert out is fake_qkv
    assert proj._xgmi_pre_qkv_a is None
