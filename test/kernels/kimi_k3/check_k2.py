# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 layer launch against SGLang's ops, at TP8 (RFC #43949).

Three outputs, each against the spec's bound:

* ``prefix``: o_proj, the attention-TP all-reduce and the residual add.
* ``x``: the MLP attention-residual seam, then the FFN input norm.
* ``out``: the layer output, the MoE on top of the seam.

The MoE is compared on the seam output the kernel itself produced, read back
out of its scratch. A wrong seam then cannot hide behind a MoE that agrees
with whatever it was given.

``--o-proj`` picks o_proj's format: bf16, or a per-channel FP8 one row-major
(``fp8``) or in AITER's (16, 16) preshuffle (``fp8-shuffled``), the layout
SGLang stores a Quark FP8 o_proj in for its GEMM. The FP8 reference multiplies
the same E4M3 values in fp32 and scales each row, so the bound is the bf16 one.

Both prefix modes run. A plain layer accumulates into the prefix. A block-write
layer restarts it at the attention output. The control at the end of the file
is why that matters. Without it, a kernel that ignored the prefix would match
both references and prove nothing. So would a harness reading a stale
scratch.
"""

import argparse

import torch
from check_common import (
    BF,
    MAX_BLOCKS,
    NVB,
    ROWS,
    TP,
    all_reduce_fp32,
    cos,
    finish,
    moe_reference,
    moe_weights,
    randn,
    rel,
    rms,
    setup,
)

from sgl_amd_mono.kimi_k3.spec import NUMERICS

# A wait's deadline here: long enough that only a real stall trips it.
DEADLINE_S = 30.0


def seam_reference(core, w_o, prefix_in, blocks, nvb, w, *, reset, o_scale=None):
    """o_proj, its all-reduce and the MLP attention-residual seam, in torch.

    This mirrors AttnResidual's eager path. The sources are the ``nvb`` banked
    rows and the updated prefix. Each is scored through the score RMSNorm and
    the one-row projection. A softmax over the sources gives the mix, and the
    output RMSNorm follows. Returns (the updated prefix, the MoE input).
    """
    import torch.nn.functional as F

    if o_scale is None:
        partial = F.linear(core, w_o)
    else:  # an FP8 o_proj: the E4M3 values in fp32, each row's sum scaled
        partial = (F.linear(core.float(), w_o.float()) * o_scale).to(BF)
    attn = all_reduce_fp32(partial).float()
    # A block-write layer restarts the prefix at the attention output.
    prefix = (attn if reset else prefix_in.float() + attn).to(BF)
    rows = torch.cat([blocks[:, :nvb, :], prefix.unsqueeze(1)], dim=1)
    scores = (rms(rows, w["ares_nw"], w["eps"]) * w["ares_qk"].float()).sum(-1)
    probs = torch.softmax(scores, dim=-1)
    mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
    return prefix, rms(mixed, w["in_nw"], w["out_eps"]).to(BF)


def layer_weights(rank, device):
    """The layer launch's own weights, on top of the MoE's.

    o_proj is a TP shard, so it is drawn per rank. The attention-residual
    stream and its two norms are replicated, so they have to match on every
    rank."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN, PROJ

    gc = torch.Generator(device=device).manual_seed(23)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    return {
        "w_o": randn(gr, HIDDEN, PROJ, scale=0.02),
        "core": randn(gr, ROWS, PROJ, scale=0.3),
        # The bank is allocated for every block of the model, so only its
        # first nvb rows are live: that is the production stride.
        "blocks": randn(gc, ROWS, MAX_BLOCKS, HIDDEN, scale=0.3),
        "prefix": randn(gc, ROWS, HIDDEN, scale=0.3),
        "ares_nw": (1 + randn(gc, HIDDEN, scale=0.1)).to(BF),
        "ares_qk": randn(gc, HIDDEN, scale=0.02),
        "in_nw": (1 + randn(gc, HIDDEN, scale=0.1)).to(BF),
        "eps": 1e-5,
        "out_eps": 1e-5,
    }


def fp8_o_proj(w_o, shuffled):
    """o_proj quantized per row to E4M3 as Quark's per-channel FP8 is ->
    (the values, their storage in the launch's layout, the fp32 scale a row)."""
    from aiter.ops.shuffle import shuffle_weight

    scale = w_o.float().abs().amax(dim=1).clamp(min=1e-12) / 448.0
    q = (w_o.float() / scale[:, None]).to(torch.float8_e4m3fn)
    storage = shuffle_weight(q.clone(), (16, 16)) if shuffled else q
    return q, storage.contiguous(), scale.contiguous()


def run(k2, w, scratch, *, reset, peers, epoch, rank, layer, guint):
    """One layer launch -> (out, the updated prefix, the build key the caller
    reads the seam output back with)."""
    key = k2.K2Build(
        tokens=ROWS,
        nblocks=NVB,
        eps=w["eps"],
        out_eps=w["out_eps"],
        reset=reset,
        guint=guint,
        ofmt=w["ofmt"],
    )
    # The kernel updates the prefix in place, so hand it a copy and keep the
    # input for the reference.
    prefix = w["prefix"].clone()
    out = torch.empty_like(prefix)
    k2.k2_launch(
        key,
        core=w["core"],
        w_o=w["w_o_store"],
        w_os=w["w_os"],
        prefix=prefix,
        blocks=w["blocks"],
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_gate=w["w_gate"],
        bias=w["bias"],
        w_ld=w["w_ld"],
        w_sgu=w["w_sgu"],
        w_sd=w["w_sd"],
        w13=w["w13"],
        w13s=w["w13s"],
        w2=w["w2"],
        w2s=w["w2s"],
        ln_w=w["ln_w"],
        w_up=w["w_up_shard"],
        out=out,
        scratch=scratch,
        peers=peers.addresses,
        rank=rank,
        epoch=epoch,
        layer=layer,
    )
    return out, prefix, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--o-proj", choices=("bf16", "fp8", "fp8-shuffled"), default="bf16")
    args = ap.parse_args()

    rank, dev, cpu = setup()
    import torch.distributed as dist

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    # The launch runs under the guard, so the peer buffer carries the flag word
    # a rank and the epoch buffer carries the guard block. Without a deadline
    # every wait reads garbage, decides it is late and faults the GPU at 0x0.
    #
    # Size this from flags_offset(), the same way the runner does. A buffer
    # sized for this check's own width puts the flags where the launch does not
    # read them, and the check then passes on a layout no server uses.
    peers = PeerBuffer(k2.flags_offset() + guard.FLAG_BYTES, cpu, rank, TP, dev)
    peers.bytes.zero_()
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(DEADLINE_S)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    scratch = torch.zeros(
        k2.scratch_bytes(k2.K2Build(tokens=ROWS, nblocks=MAX_BLOCKS)),
        dtype=torch.uint8,
        device=dev,
    )
    from sgl_amd_mono.kimi_k3.gfx950.stages import oproj

    w = moe_weights(rank, dev, guint=False)
    w.update(layer_weights(rank, dev))
    if args.o_proj == "bf16":
        w.update(ofmt=oproj.O_BF16, w_o_store=w["w_o"], w_os=None, o_scale=None)
    else:
        shuffled = args.o_proj == "fp8-shuffled"
        q, store, scale = fp8_o_proj(w["w_o"], shuffled)
        w.update(
            ofmt=oproj.O_FP8_SHUFFLED if shuffled else oproj.O_FP8,
            w_o=q,
            w_o_store=store,
            w_os=scale,
            o_scale=scale,
        )

    fails, seen, first = [], {}, {}
    for tag, reset in (("accumulate", False), ("reset", True)):
        epoch[0].add_(1)
        out, prefix, key = run(
            k2, w, scratch, reset=reset, peers=peers, epoch=epoch,
            rank=rank, layer=1, guint=False,
        )  # fmt: skip
        torch.cuda.synchronize()
        # the seam output the kernel wrote, out of its own scratch
        off, nbytes = k2.scratch_layout(key)["xrow"]
        xrow = scratch[off : off + nbytes].view(BF).view(ROWS, HIDDEN).clone()
        out, prefix = out.clone(), prefix.clone()
        ref_prefix, ref_x = seam_reference(
            w["core"], w["w_o"], w["prefix"], w["blocks"], NVB, w, reset=reset,
            o_scale=w["o_scale"],
        )  # fmt: skip
        ref_moe = moe_reference(xrow, w, rank)
        gathered = [torch.empty_like(out) for _ in range(TP)]
        dist.all_gather(gathered, out.contiguous())
        checks = {
            "finite": bool(torch.isfinite(out).all()),
            "ranks agree": all(torch.equal(gathered[0], g) for g in gathered[1:]),
            "prefix": cos(prefix, ref_prefix) > NUMERICS.bound("k2.prefix.cosine"),
            "x": cos(xrow, ref_x) > NUMERICS.bound("k2.x.cosine"),
            "out": cos(out, ref_moe) > NUMERICS.bound("k2.out.cosine")
            and rel(out, ref_moe) < NUMERICS.bound("k2.out.rel"),
        }
        if rank == 0:
            print(
                f"{args.o_proj} {tag}: prefix cos {cos(prefix, ref_prefix):.6f}  "
                f"x cos {cos(xrow, ref_x):.6f}  "
                f"out cos {cos(out, ref_moe):.6f} rel {rel(out, ref_moe):.4g}",
                flush=True,
            )
        fails += [f"{tag}/{n}" for n, ok in checks.items() if not ok]
        seen[tag] = xrow
        first[tag] = out

    # Control: the two modes read the input prefix differently, so they must
    # disagree. Without this the two checks above would pass for a kernel that
    # ignored the prefix entirely.
    if cos(seen["accumulate"], seen["reset"]) >= NUMERICS.bound("k2.x.cosine"):
        fails.append("the two prefix modes agree, so neither check means anything")

    # The captured graph replays the launch. A replay must be bit for bit what
    # the eager launch produced, which is what makes the decode graph safe.
    epoch[0].add_(1)
    eager_out, eager_prefix, _ = run(
        k2, w, scratch, reset=False, peers=peers, epoch=epoch,
        rank=rank, layer=1, guint=False,
    )  # fmt: skip
    torch.cuda.synchronize()
    eager_out, eager_prefix = eager_out.clone(), eager_prefix.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_prefix, _ = run(
            k2, w, scratch, reset=False, peers=peers, epoch=epoch,
            rank=rank, layer=1, guint=False,
        )  # fmt: skip
    for i in range(args.replays):
        epoch[0].add_(1)
        graph.replay()
        if i % 10 == 0 or i == args.replays - 1:
            torch.cuda.synchronize()
            if not torch.equal(graph_out, eager_out):
                fails.append(f"replay {i}: the output is not the eager one")
                break
    torch.cuda.synchronize()
    if rank == 0:
        print(f"{args.replays} replays bit for bit: {not fails}", flush=True)

    peers.close()
    finish(fails, dev)


if __name__ == "__main__":
    main()
