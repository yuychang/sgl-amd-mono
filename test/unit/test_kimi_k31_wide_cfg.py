# SPDX-License-Identifier: Apache-2.0
"""Verify Kimi-K3.1 wide launch tile configs stay within one BLOCKS round."""

import math

import pytest
from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.kimi_k31 import wide


def _parse(cfg: str) -> tuple[int, int, int, int, int]:
    bm, bn, wm, wn, ks = (int(x) for x in cfg.split(":"))
    return bm, bn, wm, wn, ks


@pytest.mark.parametrize("L", wide.SUPPORTED_L)
def test_cfg_rows_tile_budgets(L: int) -> None:
    """Every bucket's FRONT (17 col tiles) and INP (50 col tiles) fit in one round."""
    for b in wide.buckets(L):
        up, inp, oproj, front, sdown = wide.cfg_rows(b, L)
        for cfg in (up, inp, oproj, front, sdown):
            bm, bn, wm, wn, ks = _parse(cfg)
            assert bm > 0 and bn > 0 and wm > 0 and wn > 0 and ks > 0
        if b <= max(wide.CFGS):
            bm_f, _, _, _, ks_f = _parse(front)
            assert 17 * math.ceil(b / bm_f) * ks_f <= BLOCKS, (L, b, front)
            bm_i, _, _, _, ks_i = _parse(inp)
            assert 50 * math.ceil(b / bm_i) * ks_i <= BLOCKS, (L, b, inp)


def test_row_ranges_are_sorted_and_disjoint() -> None:
    for table in (
        wide.FRONT_ROW_RANGES,
        wide.INP_ROW_RANGES,
        wide.UP_ROW_RANGES,
        wide.SDOWN_ROW_RANGES,
    ):
        prev_hi = -1
        for lo, hi, _ in table:
            assert lo <= hi
            assert lo > prev_hi, f"overlapping or unsorted range {(lo, hi)}"
            prev_hi = hi
