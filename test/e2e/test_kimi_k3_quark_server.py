"""Kimi-K3's serving checks on the Quark checkpoint (amd/Kimi-K3-Quark-MXFP4-AttnFP8),
with the bf16 KDA state and the ReplaySSM ring: kimi_k3_quark_server.

moonshotai/Kimi-K3 reaches none of these paths: its attention projections are
bf16, and kimi_k3_server keeps the fp32 state and per-token snapshots. Here the
launches read the Quark FP8 in-projection and o_proj, and K1 writes the ring
rows SGLang's commit replays. The checks are test_kimi_k3_server's: the startup
line, GSM8K for gross failures and the covered steps scored again by prefill.
The fault test is not repeated; the guard does not depend on the checkpoint.
"""

import unittest
from pathlib import Path

import test_kimi_k3_server as k3
from kimi_k3_quark_server import ARGS, ENV, MODEL


class TestMonokernelKimiK3Quark(k3.TestMonokernelKimiK3):
    recipe = (MODEL, ENV, ARGS)

    def test_the_tested_batches_run_the_monokernels(self):
        super().test_the_tested_batches_run_the_monokernels()
        log = "".join(Path(f.name).read_text() for f in self.logs)
        self.assertIn("Kimi-K3 K1: engaged on a decode step", log)
        self.assertNotIn("not built for the KDA layers", log)


if __name__ == "__main__":
    unittest.main()
