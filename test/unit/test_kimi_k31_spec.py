# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3.1 spec routes one request's verify steps to Kimi-K3's launches and
wider ones to the wide launch."""

from types import SimpleNamespace

import pytest
import torch

import sgl_amd_mono
from sgl_amd_mono.kimi_k3 import MAX_ROWS
from sgl_amd_mono.kimi_k31 import spec as k31
from sgl_amd_mono.kimi_k31 import wide


def test_registered_under_the_k31_architecture():
    assert (
        sgl_amd_mono.SPECS[("KimiK31ForConditionalGeneration", None)]
        == "sgl_amd_mono.kimi_k31.spec:SPEC"
    )


def test_the_narrow_and_wide_widths_meet_without_a_gap():
    # _Built routes rows up to NARROW to Kimi-K3's launches, which are built up to
    # MAX_ROWS; a width between the two would route to a launch never built
    assert wide.NARROW == MAX_ROWS
    widths = k31.SPEC.widths("gfx950")
    assert tuple(range(1, MAX_ROWS + 1)) == widths[:MAX_ROWS]
    assert all(w > wide.NARROW for w in widths[MAX_ROWS:])


def test_every_wide_width_has_a_bucket_at_each_draft_length():
    # the binding serves a width at every L it is a multiple of: an 8-token width
    # that reads as more 6-token requests than the largest bucket holds raised at
    # the launch
    top = max(wide.REQ_BUCKETS)
    for w in wide.wide_widths():
        for L in wide.SUPPORTED_L:
            if w % L == 0:
                assert w // L <= top, (w, L)


def test_the_launches_name_their_server_args():
    server_args = k31.SPEC.requires.server_args
    assert server_args["enable_linear_replayssm_spec"] == frozenset({True})
    assert server_args["speculative_algorithm"] == frozenset({"DSPARK", "DFLASH"})


def _reasons(monkeypatch, *, in_proj=torch.bfloat16, o_proj=torch.bfloat16):
    """kimi_k31's checks on one covered KDA layer, Kimi-K3's own checks taken as
    passing (test_kimi_k3_spec pins those: they accept a Quark FP8 o_proj).
    ``in_proj=None`` is a layer whose KDA in_proj is not merged."""
    monkeypatch.setattr(k31.k3, "_model_disable_reasons", lambda *a: [])
    monkeypatch.setattr(wide, "_is_covered", lambda layer: True)

    def lin(dtype):
        return SimpleNamespace(weight=torch.zeros(8, 8, dtype=dtype))

    attn = SimpleNamespace(
        _qkvgbfa_layer=None if in_proj is None else lin(in_proj), o_proj=lin(o_proj)
    )
    layers = [SimpleNamespace(self_attn=attn)]
    return k31._model_disable_reasons(
        SimpleNamespace(model=SimpleNamespace(layers=layers)), None, {}
    )


def test_a_bf16_kda_layer_passes(monkeypatch):
    assert _reasons(monkeypatch) == []


@pytest.mark.parametrize(
    "dtypes",
    [
        {"o_proj": torch.float8_e4m3fn},
        {"in_proj": torch.float8_e4m3fn},
        {"in_proj": torch.float32},
    ],
    ids=["fp8-o_proj", "fp8-in_proj", "fp32-in_proj"],
)
def test_a_kda_weight_the_wide_launch_cannot_read_is_refused(monkeypatch, dtypes):
    # Kimi-K3's checks accept a Quark FP8 o_proj (K1 and the layer launch read it),
    # but the wide launch reads the in_proj and o_proj weights as bf16: an FP8 one
    # would be read as if it were bf16.
    reasons = _reasons(monkeypatch, **dtypes)
    assert any("not bf16" in r for r in reasons), reasons


def test_an_unmerged_in_proj_is_reported_once(monkeypatch):
    # an unmerged layer has no merged weight to read; it is refused for that alone
    reasons = _reasons(monkeypatch, in_proj=None, o_proj=torch.float8_e4m3fn)
    assert reasons == ["layers [0]: the KDA in_proj is not merged"], reasons
