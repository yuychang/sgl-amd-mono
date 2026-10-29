"""The Kimi-K3 gfx950 monokernel spec on CPU (RFC #43949): the layers its forward
covers on the Kimi-K3 layer structure, its registry entry, its per-output bounds
and its config refusal reasons. SGLang checks its declared requirements, the same
way for every spec."""

from types import SimpleNamespace

import pytest

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.kimi_k3 import HF_CONFIG, LINEAR_ATTN_CONFIG, MAX_BLOCKS, MAX_ROWS
from sgl_amd_mono.kimi_k3 import spec as k3

# Kimi-K3's layer structure (its config.json): 93 layers, a dense layer 0, then
# the KDA / full-attention cadence. Every layer past the dense one runs a MoE.
FULL_ATTN = list(range(4, 93, 4))
KDA = [i for i in range(1, 93) if i not in FULL_ATTN]
TEXT = SimpleNamespace(
    num_hidden_layers=93,
    attn_res_block_size=12,
    linear_attn_config={**LINEAR_ATTN_CONFIG, "full_attn_layers": FULL_ATTN + [93],
                        "kda_layers": KDA},
    **HF_CONFIG,
)  # fmt: skip
K3 = SimpleNamespace(
    model_type="kimi_k3",
    architectures=["KimiK3ForConditionalGeneration"],
    text_config=TEXT,
)


def test_the_forward_covers_every_moe_layer():
    # the dense layer 0 keeps the model's own forward; every other layer is a
    # MoE layer and takes the launch
    assert k3._covered_layers(K3) == list(range(1, 93))


def test_coverage_is_not_an_attention_kind():
    # The launch starts after the attention, so both kinds are covered. A spec
    # that declared only the KDA layers would still serve and still be correct,
    # just slower -- which is why this is pinned.
    covered = set(k3._covered_layers(K3))
    assert set(KDA) <= covered and set(FULL_ATTN) <= covered
    assert len(covered) == 92


def test_the_dense_prefix_and_the_moe_cadence_are_read_from_the_config():
    dense = SimpleNamespace(**{**vars(TEXT), "first_k_dense_replace": 4})
    assert k3._covered_layers(SimpleNamespace(text_config=dense)) == list(range(4, 93))
    every_other = SimpleNamespace(**{**vars(TEXT), "moe_layer_freq": 2})
    assert k3._covered_layers(SimpleNamespace(text_config=every_other)) == list(
        range(2, 93, 2)
    )


def test_the_plugin_registers_the_spec():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND,
        architectures=["KimiK3ForConditionalGeneration"],
        isa="gfx950",
    )
    assert found is k3.SPEC, why


def test_the_spec_is_tp8_and_at_most_eight_rows():
    limits = k3.SPEC.devices["gfx950"]
    assert limits.tp_sizes == (8,)
    assert limits.widths == tuple(range(1, MAX_ROWS + 1))
    # With DSpark the runtime serves only the widths that hold whole requests,
    # so block size 7 (8 tokens a request) leaves batch size 1.
    assert [w // 8 for w in limits.widths if w % 8 == 0] == [1]


def test_the_shapes_are_checked_on_the_text_config():
    # The architecture names the VLM wrapper, whose top-level config carries
    # none of the shapes, so requires.hf_config cannot reach them.
    assert "hidden_size" not in k3.SPEC.requires.hf_config
    assert k3.SPEC.requires.hf_config == {"model_type": "kimi_k3"}
    wrong = SimpleNamespace(**{**vars(TEXT), "num_experts_per_token": 8})
    reasons = k3._config_disable_reasons(None, SimpleNamespace(text_config=wrong))
    assert any("num_experts_per_token" in r for r in reasons), reasons


def test_a_block_size_past_the_launch_is_refused():
    # The bank rows a layer's seam reads grow with the layer index; at block
    # size 12 the last layer reads exactly MAX_BLOCKS of them.
    tight = SimpleNamespace(**{**vars(TEXT), "attn_res_block_size": 8})
    reasons = k3._config_disable_reasons(None, SimpleNamespace(text_config=tight))
    assert any("attention-residual blocks" in r for r in reasons), reasons
    ok = k3._config_disable_reasons(None, K3)
    assert not any("attention-residual blocks" in r for r in ok), ok


def test_the_bounds_name_the_launches_and_their_outputs():
    for name in ("k2.prefix.cosine", "k2.x.cosine", "k2.out.cosine", "moe.out.cosine"):
        assert 0.0 < k3.SPEC.numerics.bound(name) <= 1.0
    with pytest.raises(KeyError):
        k3.SPEC.numerics.bound("k2.nothing.cosine")


def test_the_build_needs_a_real_model():
    with pytest.raises((AttributeError, TypeError)):
        k3.SPEC.build(None, K3, None)


def test_max_blocks_matches_the_reference_config():
    # layer 92 at block size 12 reads 8 blocks: cdiv(92, 12) = 8, and 92 is not
    # a block-write layer. The launch is built for exactly that many.
    last = max(k3._covered_layers(K3))
    block = TEXT.attn_res_block_size
    assert -(-last // block) + int(last % block == 0) == MAX_BLOCKS


def _fake_layer(*, gate=True, kda=False, o_proj=None, ri=384):
    """A covered layer as model_disable_reasons reads it, everything passing."""
    import torch

    def lin(*shape):
        return SimpleNamespace(
            weight=torch.zeros(*shape, dtype=torch.bfloat16), bias=None
        )

    attn = SimpleNamespace(
        o_proj=o_proj or lin(HF_CONFIG["hidden_size"], 1536),
        forward_pre_o_proj=lambda **kw: None,
    )
    if not kda:  # only the MLA layers carry the gate
        attn.use_output_gate = gate
    e, lat = HF_CONFIG["num_experts"], HF_CONFIG["routed_expert_hidden_size"]
    experts = SimpleNamespace(
        w13_weight=SimpleNamespace(is_shuffled=True, shape=(e, 2 * ri, lat // 2)),
        w2_weight=SimpleNamespace(is_shuffled=True, shape=(e, lat, ri // 2)),
    )
    up = SimpleNamespace(
        weight=torch.zeros(HF_CONFIG["hidden_size"], 8, dtype=torch.bfloat16)
    )
    return SimpleNamespace(
        self_attn=attn,
        mlp=SimpleNamespace(
            use_latent_moe=True,
            routed_expert_norm=object(),
            routed_expert_up_proj=up,
            experts=experts,
        ),
        use_attn_residuals=True,
        _sp_moe=False,
        all_reduce_fusion=False,
    )


def _reasons(monkeypatch, layer):
    import sglang.srt.utils.common as common

    monkeypatch.setattr(common, "require_mlp_sync", lambda: False)
    model = SimpleNamespace(model=SimpleNamespace(layers={i: layer for i in range(93)}))
    return k3._model_disable_reasons(model, K3, {})


def test_a_covered_layer_passes(monkeypatch):
    assert _reasons(monkeypatch, _fake_layer()) == []
    assert _reasons(monkeypatch, _fake_layer(kda=True)) == []


def test_an_mla_layer_without_the_output_gate_is_refused(monkeypatch):
    # hasattr(attn, "forward_pre_o_proj") is true for both attention classes,
    # so it cannot carry this: for MLA the split rides on the gate's wrapper.
    reasons = _reasons(monkeypatch, _fake_layer(gate=False))
    assert any("output gate" in r for r in reasons), reasons


def _fp8_o_proj(*, quark=True):
    """o_proj as SGLang leaves a Quark per-channel FP8 one: the [out, in]
    storage transposed for its GEMM, an fp32 scale a row."""
    import torch
    from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

    hidden = HF_CONFIG["hidden_size"]
    scheme = QuarkW8A8Fp8(
        {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": True}
    )
    return SimpleNamespace(
        weight=torch.zeros(hidden, 1536, dtype=torch.float8_e4m3fn).t(),
        weight_scale=torch.ones(hidden, 1, dtype=torch.float32),
        scheme=scheme if quark else object(),
        bias=None,
    )


def test_a_quark_fp8_o_proj_passes(monkeypatch):
    assert _reasons(monkeypatch, _fake_layer(o_proj=_fp8_o_proj())) == []


def test_an_fp8_o_proj_outside_quark_is_refused(monkeypatch):
    reasons = _reasons(monkeypatch, _fake_layer(o_proj=_fp8_o_proj(quark=False)))
    assert any("Quark per-channel FP8" in r for r in reasons), reasons


@pytest.mark.parametrize("shuffled", [True, False])
def test_the_fp8_o_proj_layout_is_the_one_sglang_chose(monkeypatch, shuffled):
    # SGLang preshuffles the FP8 weight exactly when its own predicate says so;
    # the launch must read the same layout, or it reads the bytes in a wrong order.
    import sglang.srt.layers.quantization.fp8_utils as fp8_utils

    from sgl_amd_mono.kimi_k3.gfx950.stages import oproj

    monkeypatch.setattr(fp8_utils, "use_aiter_bpreshuffle_gemm", lambda n: shuffled)
    fmt, storage, scale = k3._o_proj_operands(_fp8_o_proj())
    assert fmt == (oproj.O_FP8_SHUFFLED if shuffled else oproj.O_FP8)
    assert storage.is_contiguous() and tuple(storage.shape) == (
        HF_CONFIG["hidden_size"],
        1536,
    )
    assert tuple(scale.shape) == (HF_CONFIG["hidden_size"],)


def test_padded_experts_are_refused(monkeypatch):
    # Without SGLANG_OPT_HIP_K3_AITER the intermediate is padded to 512.
    reasons = _reasons(monkeypatch, _fake_layer(ri=512))
    assert any("padded intermediate" in r for r in reasons), reasons


def test_the_separated_a4w4_expert_layout_is_not_refused(monkeypatch):
    # AITER_SITUV2_A4W4's preshuffle is byte for byte the launch's gate_up=False one.
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")
    monkeypatch.setenv("AITER_SITUV2_A8W4", "0")
    monkeypatch.setenv("SGLANG_OPT_HIP_K3_AITER", "1")
    reasons = k3._config_disable_reasons(None, K3)
    assert not any("interleaved" in r or "K3_AITER" in r for r in reasons), reasons


def test_the_k3_aiter_switch_is_read_under_its_new_name(monkeypatch):
    monkeypatch.delenv("SGLANG_AITER_K3_OPT", raising=False)
    monkeypatch.delenv("SGLANG_OPT_HIP_K3_AITER", raising=False)
    assert any("K3_AITER" in r for r in k3._config_disable_reasons(None, K3))
    monkeypatch.setenv("SGLANG_OPT_HIP_K3_AITER", "1")
    assert not any("K3_AITER" in r for r in k3._config_disable_reasons(None, K3))


def test_the_guard_flags_sit_past_every_width(monkeypatch):
    """Every width reads the guard flags at the same offset.

    The runner allocates one peer buffer for all the widths and puts the flags
    after the data. A launch that derived the offset from its own width wrote
    its flags inside the data region of a wider launch. Two widths in one run
    then disagreed on where the flags are, and the narrower one read live data
    as an error word.
    """
    from sgl_amd_mono.kimi_k3.gfx950 import layer
    from sgl_amd_mono.kimi_k3.gfx950.stages import moe

    off = layer.flags_offset()
    for width in range(1, MAX_ROWS + 1):
        assert layer.peer_bytes(width) <= off, width
        assert moe.peer_bytes(width) <= off, width
    # the widest launch ends exactly there: a larger offset wastes a peer page
    assert off == max(layer.peer_bytes(MAX_ROWS), moe.peer_bytes(MAX_ROWS))
