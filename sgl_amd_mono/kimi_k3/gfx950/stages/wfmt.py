"""A GEMV weight the launches read in place, in the format SGLang stored it.

K1's in_proj (``attention.kda``) and the layer launch's o_proj (``stages.oproj``)
take the same three formats:

* ``BF16``: bf16 [rows, K], row-major.
* ``FP8``: per-channel FP8 E4M3 [rows, K], row-major, with an fp32 scale a row.
* ``FP8_SHUFFLED``: the same in AITER's (16, 16) preshuffle, the B operand
  SGLang stores whenever its FP8 GEMM runs (``use_aiter_bpreshuffle_gemm``).

A lane reads 8 elements of its row at k = 128 ch + 32 q + 8 g (g = lane / 16),
which is what ``stage_gemv`` / ``stage_o`` read from LDS. FP8 is widened to bf16
in registers (E4M3 widens exactly) and its row's scale is applied to the fp32
sum, as the GEMM applies it. The scale load stays with each caller: o_proj
reads two rows' scales in one load.
"""

import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import CM_NT, fp8_to_bf16x8, rsrc
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import ROWS

BF16, FP8, FP8_SHUFFLED = 0, 1, 2


def fp8_byte(fmt, lane, rg, ch, q, k):
    """The byte of a lane's 8 FP8 elements (row 16 rg + lane % 16, k = 128 ch +
    32 q + 8 g) of a [rows, ``k``] weight. AITER's (16, 16) preshuffle of a byte
    weight [N, K] is [N / 16][K / 32][2][16 rows][16 B], so those 8 bytes stay
    contiguous."""
    r, g = lane % ROWS, lane // ROWS
    if fmt == FP8_SHUFFLED:
        tile = rg * (k // 32) + ch * 4 + q
        return tile * 512 + (g // 2) * 256 + r * 16 + (g % 2) * 8
    return (rg * ROWS + r) * k + ch * 128 + q * 32 + g * 8


def fp8_loads(w, fmt, lane, rg, chunks, k, rg_max=None):
    """A lane's FP8 operands of rows 16 rg .. of the [rows, ``k``] weight at
    ``w``: for each chunk index in ``chunks`` and q < 4, two dwords. ``rg`` is
    clamped to ``rg_max`` when given. ``chunks`` may be a generator, so each
    chunk index is computed where the loop reaches it."""
    r_w = rsrc(w)
    if rg_max is not None:
        rg = fx.min(rg, rg_max)
    out = []
    for ch in chunks:
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        fp8_byte(fmt, lane, rg, ch, q, k) // 4,
                        vec_width=2,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
                for q in range(4)
            ]
        )
    return out


def operand(fmt, w):
    """A lane's 8 weight values as the MFMA's bf16 operand."""
    if fmt == BF16:
        return w.bitcast(fx.BFloat16)
    return fp8_to_bf16x8(w[0], w[1])


def check(fmt, w, scale, rows: int, k: int) -> None:
    """Host-side: ``w`` and ``scale`` are what a ``fmt`` launch reads. bf16 is
    exactly [rows, k]; FP8 has at least ``rows`` rows (SGLang pads a merged one),
    a multiple of 16 when preshuffled, and an fp32 scale for each."""
    assert w.is_contiguous()
    if fmt == BF16:
        assert w.dtype == torch.bfloat16 and tuple(w.shape) == (rows, k), (
            w.dtype,
            tuple(w.shape),
        )
        assert scale is None
        return
    assert w.dtype == torch.float8_e4m3fn, w.dtype
    assert w.shape[0] >= rows and w.shape[1] == k, tuple(w.shape)
    assert fmt != FP8_SHUFFLED or w.shape[0] % ROWS == 0, tuple(w.shape)
    assert (
        scale is not None
        and scale.dtype == torch.float32
        and scale.numel() == w.shape[0]
        and scale.is_contiguous()
    )
