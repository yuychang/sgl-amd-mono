# SPDX-License-Identifier: Apache-2.0
"""8-GPU correctness, CUDA-graph replay, and latency benchmark for K3.1 XGMI fusion.

Run:
    kimi_k31/.venv/sglang/bin/torchrun --nproc-per-node 8 \
        kimi_k31/sgl-amd-mono/test/kernels/kimi_k31/xgmi_bench.py [--bench]
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sgl_amd_mono.kimi_k31.xgmi.decode import (
    deferred_topk_ar,
    latent_input_gather,
    moetail_gather,
    rms_norm_bf16,
    slice_replicated_weight,
)
from sgl_amd_mono.kimi_k31.xgmi.heap import TP, XgmiHeap
from sgl_amd_mono.kimi_k31.xgmi.prefill import (
    moe_tail_mix_qkv_a,
    overlapped_o_proj_rs,
    pad_rows_to_tp,
    prefill_reduce_scatter_32cu,
    prefill_row_allgather,
    sharded_front_fp4_ag,
)

H = 7168
LAT = 3584
ROUTER = 896
TOP_K = 16
QKV_A = 2112
O_PROJ_K_LOCAL = 1536  # 96 heads / 8 * 128 head_dim


def _bench_us(fn, iters: int = 30, warmup: int = 8) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / iters


def check_decode(heap: XgmiHeap, rank: int, dev: torch.device, do_bench: bool):
    g = torch.Generator(device=dev).manual_seed(42)
    w_router = (torch.randn(ROUTER, H, generator=g, device=dev) * 0.02).bfloat16()
    w_down = (torch.randn(LAT, H, generator=g, device=dev) * 0.02).bfloat16()
    w_up = (torch.randn(H, LAT, generator=g, device=dev) * 0.02).bfloat16()
    w_norm = torch.ones(LAT, dtype=torch.bfloat16, device=dev)

    r_sl = slice_replicated_weight(w_router, rank)
    l_sl = slice_replicated_weight(w_down, rank)
    w_front_sl = torch.cat([r_sl, l_sl], dim=0).contiguous()  # [560, 7168]
    w_up_sl = slice_replicated_weight(w_up, rank)  # [896, 3584]

    for m in (1, 8, 64, 192):
        # 3 rotating inputs to verify double-buffer epoch hygiene
        for it in range(3):
            gi = torch.Generator(device=dev).manual_seed(1000 + m * 10 + it)
            x = (torch.randn(m, H, generator=gi, device=dev) * 0.5).bfloat16()

            # 1) 1/8 latent_input_gather
            r_got, l_got = latent_input_gather(heap, x, w_front_sl)
            r_ref = F.linear(x, w_router)
            l_ref = F.linear(x, w_down)
            # Minor GEMM tiling rounding between [560,H] and [4480,H] in bf16
            r_err = (r_got.float() - r_ref.float()).abs().max().item()
            l_err = (l_got.float() - l_ref.float()).abs().max().item()
            assert r_err < 2e-2 and l_err < 2e-2, (
                f"M={m} it={it}: r_err={r_err}, l_err={l_err}"
            )

            # 2) deferred_topk_lamport_ar (direct + sorted-index)
            # Build all ranks' partials deterministically so reference sums in rank order
            exp_all = (
                torch.randn(TP, m, TOP_K, LAT, generator=gi, device=dev) * 0.25
            ).bfloat16()
            tw = torch.softmax(
                torch.randn(m, TOP_K, generator=gi, device=dev), dim=-1
            ).float()
            sh_all = (torch.randn(TP, m, H, generator=gi, device=dev) * 0.25).bfloat16()

            lat_got, sh_sl_got = deferred_topk_ar(
                heap, exp_all[rank], tw, sh_all[rank], gather_full_shared=False
            )
            # Reference: per-rank fp32 top-k -> bf16, then rank-order fp32 sum across TP8 -> bf16
            per_rank_lat = (
                (exp_all.float() * tw[None, :, :, None]).sum(dim=2).bfloat16()
            )
            lat_ref = torch.zeros(m, LAT, dtype=torch.float32, device=dev)
            sh_ref = torch.zeros(m, H, dtype=torch.float32, device=dev)
            for r in range(TP):
                lat_ref = lat_ref + per_rank_lat[r].float()
                sh_ref = sh_ref + sh_all[r].float()
            lat_ref = lat_ref.bfloat16()
            sh_ref = sh_ref.bfloat16()
            sl = H // TP
            sh_sl_ref = sh_ref[:, rank * sl : (rank + 1) * sl]

            lat_err = (lat_got.float() - lat_ref.float()).abs().max().item()
            sh_err = (sh_sl_got.float() - sh_sl_ref.float()).abs().max().item()
            assert lat_err <= 2e-2 and sh_err == 0.0, (
                f"M={m} it={it}: lat_err={lat_err}, sh_err={sh_err}"
            )

            # Also check sorted_idx path with gather_full_shared=True on it==0
            if it == 0:
                perm = torch.randperm(m * TOP_K, generator=gi, device=dev).to(
                    torch.int32
                )
                sidx = perm.view(m, TOP_K)
                exp_flat = torch.empty(m * TOP_K, LAT, dtype=torch.bfloat16, device=dev)
                exp_flat[perm.long()] = exp_all[rank].reshape(m * TOP_K, LAT)
                lat2, sh_full = deferred_topk_ar(
                    heap,
                    exp_flat,
                    tw,
                    sh_all[rank],
                    sorted_idx=sidx,
                    gather_full_shared=True,
                )
                assert torch.equal(lat2, lat_got), f"sorted_idx mismatch at M={m}"
                assert torch.equal(sh_full, sh_ref), (
                    f"gather_full_shared mismatch at M={m}"
                )

            # 3) 1/8 moetail_gather
            prefix = (torch.randn(m, H, generator=gi, device=dev) * 0.25).bfloat16()
            pref_sl = prefix[:, rank * sl : (rank + 1) * sl].contiguous()
            stream_got = moetail_gather(
                heap, lat_got, w_up_sl, sh_sl_got, pref_sl, norm_weight=w_norm
            )
            stream_ref = (
                F.linear(rms_norm_bf16(lat_got, w_norm), w_up).float()
                + sh_ref.float()
                + prefix.float()
            ).bfloat16()
            st_err = (stream_got.float() - stream_ref.float()).abs().max().item()
            assert st_err < 3e-2, f"moetail M={m} it={it}: st_err={st_err}"

    # 4) CUDA Graph capture & 10 replays with rotating inputs
    m_g = 64
    x_buf = torch.randn(m_g, H, dtype=torch.bfloat16, device=dev) * 0.5
    exp_buf = torch.randn(m_g, TOP_K, LAT, dtype=torch.bfloat16, device=dev) * 0.25
    tw_buf = torch.softmax(torch.randn(m_g, TOP_K, device=dev), dim=-1).float()
    sh_buf = torch.randn(m_g, H, dtype=torch.bfloat16, device=dev) * 0.25
    pref_buf = torch.randn(m_g, H // TP, dtype=torch.bfloat16, device=dev) * 0.25

    # Warmup before capture
    for _ in range(2):
        _r, _l = latent_input_gather(heap, x_buf, w_front_sl)
        _lat, _sh = deferred_topk_ar(heap, exp_buf, tw_buf, sh_buf)
        _st = moetail_gather(heap, _lat, w_up_sl, _sh, pref_buf, norm_weight=w_norm)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_r, g_l = latent_input_gather(heap, x_buf, w_front_sl)
        g_lat, g_sh = deferred_topk_ar(heap, exp_buf, tw_buf, sh_buf)
        g_st = moetail_gather(heap, g_lat, w_up_sl, g_sh, pref_buf, norm_weight=w_norm)
    torch.cuda.synchronize()
    dist.barrier()

    for rep in range(10):
        gr = torch.Generator(device=dev).manual_seed(7000 + rep)
        x_new = (torch.randn(m_g, H, generator=gr, device=dev) * 0.5).bfloat16()
        exp_all = (
            torch.randn(TP, m_g, TOP_K, LAT, generator=gr, device=dev) * 0.25
        ).bfloat16()
        sh_all = (torch.randn(TP, m_g, H, generator=gr, device=dev) * 0.25).bfloat16()
        x_buf.copy_(x_new)
        exp_buf.copy_(exp_all[rank])
        sh_buf.copy_(sh_all[rank])
        graph.replay()
        torch.cuda.synchronize()
        ref_r = F.linear(x_new, w_router)
        assert (g_r.float() - ref_r.float()).abs().max().item() < 2e-2, (
            f"graph rep={rep} failed"
        )

    if rank == 0:
        print(
            "[PASS] Decode (M=1,8,64,192 + 10x CUDA Graph replays): latent_input_gather, deferred_topk_ar, moetail_gather"
        )

    if do_bench:
        if rank == 0:
            print("\n=== Decode Benchmark (us/call, TP8 MI355X) ===")
        for m in (8, 64, 192):
            x = torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.5
            exp = torch.randn(m, TOP_K, LAT, dtype=torch.bfloat16, device=dev) * 0.25
            tw = torch.softmax(torch.randn(m, TOP_K, device=dev), dim=-1).float()
            sh = torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.25
            lat = torch.randn(m, LAT, dtype=torch.bfloat16, device=dev) * 0.5
            sh_sl = torch.randn(m, H // TP, dtype=torch.bfloat16, device=dev) * 0.25
            pref_full = torch.randn(m, H, dtype=torch.bfloat16, device=dev) * 0.25
            pref_sl = pref_full[
                :, rank * (H // TP) : (rank + 1) * (H // TP)
            ].contiguous()
            w_front_full = torch.cat([w_router, w_down], dim=0).contiguous()

            t_front_xgmi = _bench_us(lambda: latent_input_gather(heap, x, w_front_sl))
            t_front_base = _bench_us(lambda: F.linear(x, w_front_full))

            t_ar_xgmi = _bench_us(lambda: deferred_topk_ar(heap, exp, tw, sh))

            def _ar_base():
                lat_p = (exp.float() * tw[:, :, None]).sum(dim=1).bfloat16()
                buf = torch.cat([lat_p, sh], dim=-1)
                dist.all_reduce(buf)
                return buf

            t_ar_base = _bench_us(_ar_base)

            t_tail_xgmi = _bench_us(
                lambda: moetail_gather(
                    heap, lat, w_up_sl, sh_sl, pref_sl, norm_weight=w_norm
                )
            )

            def _tail_base():
                return F.linear(rms_norm_bf16(lat, w_norm), w_up) + sh + pref_full

            t_tail_base = _bench_us(_tail_base)

            if rank == 0:
                print(
                    f"M={m:3d} | 1/8 front+AG: {t_front_xgmi:6.1f} vs full {t_front_base:6.1f} us "
                    f"| def-topk+LamportAR: {t_ar_xgmi:6.1f} vs RCCL {t_ar_base:6.1f} us "
                    f"| 1/8 moetail+AG: {t_tail_xgmi:6.1f} vs full {t_tail_base:6.1f} us"
                )


def check_prefill(heap: XgmiHeap, rank: int, dev: torch.device, do_bench: bool):
    g = torch.Generator(device=dev).manual_seed(99)
    w_oproj_all = (
        torch.randn(TP, H, O_PROJ_K_LOCAL, generator=g, device=dev) * 0.02
    ).bfloat16()
    w_oproj = w_oproj_all[rank].contiguous()
    w_rd = (torch.randn(ROUTER + LAT, H, generator=g, device=dev) * 0.02).bfloat16()
    w_up = (torch.randn(H, LAT, generator=g, device=dev) * 0.02).bfloat16()
    w_qkv_a = (torch.randn(QKV_A, H, generator=g, device=dev) * 0.02).bfloat16()
    w_norm = torch.ones(LAT, dtype=torch.bfloat16, device=dev)

    def _topk_fn(normed_s, logits):
        vals, ids = torch.topk(logits.float(), TOP_K, dim=-1)
        return ids.to(torch.int32), torch.softmax(vals, dim=-1).float()

    for m in (200, 1024, 4096, 8192):
        padded = ((m + TP - 1) // TP) * TP
        r_per = padded // TP
        gi = torch.Generator(device=dev).manual_seed(2000 + m)

        # 1) overlapped_o_proj_rs (224-CU GEMM x 32-CU XGMI RS, 0 extra weights)
        x_all = (
            torch.randn(TP, m, O_PROJ_K_LOCAL, generator=gi, device=dev) * 0.3
        ).bfloat16()
        got_rs = overlapped_o_proj_rs(heap, x_all[rank], w_oproj)

        # Reference: each rank runs full F.linear, pad, reduce_scatter_tensor
        part_local = F.linear(x_all[rank], w_oproj)
        part_pad, _ = pad_rows_to_tp(part_local, TP)
        ref_rs = torch.empty(r_per, H, dtype=torch.bfloat16, device=dev)
        dist.reduce_scatter_tensor(ref_rs, part_pad)
        err_rs = (got_rs.float() - ref_rs.float()).abs().max().item()
        assert err_rs < 5e-2, f"o_proj RS M={m}: err={err_rs}"

        # 2) standalone 32-CU RS check (bit-exact vs rank-order fp32)
        p_all = (
            torch.randn(TP, padded, LAT, generator=gi, device=dev) * 0.25
        ).bfloat16()
        got_lat_rs = prefill_reduce_scatter_32cu(heap, p_all[rank])
        acc = torch.zeros(r_per, LAT, dtype=torch.float32, device=dev)
        for r in range(TP):
            acc = acc + p_all[r, rank * r_per : (rank + 1) * r_per].float()
        assert torch.equal(got_lat_rs, acc.bfloat16()), f"32-CU RS M={m} mismatch!"

        # 3) sharded_front_fp4_ag (E2M1 FP4 AND MXFP8)
        normed_all = (torch.randn(padded, H, generator=gi, device=dev) * 0.4).bfloat16()
        normed_shard = normed_all[rank * r_per : (rank + 1) * r_per].contiguous()
        for use_fp4 in (True, False):
            captured = {}

            def _cap_topk(ns, logits):
                ids_l, w_l = _topk_fn(ns, logits)
                captured["ids"], captured["w"] = ids_l, w_l
                return ids_l, w_l

            q_all, s_all, ids_all, w_all = sharded_front_fp4_ag(
                heap, normed_shard, m, w_rd, ROUTER, _cap_topk, use_fp4=use_fp4
            )
            ids_ref = torch.empty(padded, TOP_K, dtype=torch.int32, device=dev)
            w_ref = torch.empty(padded, TOP_K, dtype=torch.float32, device=dev)
            dist.all_gather_into_tensor(ids_ref, captured["ids"].contiguous())
            dist.all_gather_into_tensor(w_ref, captured["w"].contiguous())
            assert torch.equal(ids_all, ids_ref[:m]), (
                f"fp4={use_fp4} M={m} topk_ids mismatch"
            )
            assert torch.equal(w_all, w_ref[:m]), (
                f"fp4={use_fp4} M={m} topk_weights mismatch"
            )
            expected_q_cols = LAT // 2 if use_fp4 else LAT
            assert q_all.shape == (m, expected_q_cols) and s_all.shape == (m, LAT // 32)

        # 4) moe_tail_mix_qkv_a (1/8 up_proj + 1/8 2112 qkv_a + XGMI AG)
        lat_all = (torch.randn(padded, LAT, generator=gi, device=dev) * 0.3).bfloat16()
        sh_all = (torch.randn(padded, H, generator=gi, device=dev) * 0.3).bfloat16()
        pr_all = (torch.randn(padded, H, generator=gi, device=dev) * 0.3).bfloat16()
        sl = slice(rank * r_per, (rank + 1) * r_per)
        st_sh, _, qkv_a_got = moe_tail_mix_qkv_a(
            heap,
            lat_all[sl].contiguous(),
            sh_all[sl].contiguous(),
            pr_all[sl].contiguous(),
            m,
            latent_norm_fn=lambda z: rms_norm_bf16(z, w_norm),
            up_proj_weight=w_up,
            next_qkv_a_weight=w_qkv_a,
        )
        # Verify against full-row qkv_a projection
        st_full = (
            F.linear(rms_norm_bf16(lat_all[:m], w_norm), w_up) + sh_all[:m] + pr_all[:m]
        )
        qkv_a_ref = F.linear(st_full, w_qkv_a)
        err_qkv = (qkv_a_got.float() - qkv_a_ref.float()).abs().max().item()
        rel_qkv = err_qkv / (qkv_a_ref.float().abs().max().item() + 1e-6)
        assert rel_qkv < 1e-2, f"moe_tail_mix_qkv_a M={m}: err={err_qkv} rel={rel_qkv}"

        # 5) 7168-dim auto-chunked row_allgather check (including M=8192!)
        got_7168 = prefill_row_allgather(heap, normed_shard, m)
        assert torch.equal(got_7168, normed_all[:m]), (
            f"row_allgather 7168 M={m} mismatch!"
        )

    if rank == 0:
        print(
            f"[PASS] Prefill (M=200,1024,4096,8192, cu_masked={heap.streams.cu_masked} "
            f"{heap.streams.compute_cus}+{heap.streams.comm_cus} CUs, heap={heap.heap_mib:.1f} MiB): "
            "overlapped_o_proj_rs, 32-CU RS, FP4/FP8 pack AG, moe_tail_mix_qkv_a, row_allgather"
        )

    if do_bench:
        if rank == 0:
            print("\n=== Prefill Benchmark (us/call, TP8 MI355X, 0 extra weights) ===")
        for m in (512, 2048, 4096, 8192):
            r_per = m // TP
            x_loc = (
                torch.randn(m, O_PROJ_K_LOCAL, dtype=torch.bfloat16, device=dev) * 0.3
            )
            out_rccl = torch.empty(r_per, H, dtype=torch.bfloat16, device=dev)
            normed_sh = torch.randn(r_per, H, dtype=torch.bfloat16, device=dev) * 0.4
            normed_full = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
            lat_sh = torch.randn(r_per, LAT, dtype=torch.bfloat16, device=dev) * 0.3
            sh_sh = torch.randn(r_per, H, dtype=torch.bfloat16, device=dev) * 0.3
            pr_sh = torch.randn(r_per, H, dtype=torch.bfloat16, device=dev) * 0.3

            t_oproj_ov = _bench_us(lambda: overlapped_o_proj_rs(heap, x_loc, w_oproj))

            def _oproj_base():
                p = F.linear(x_loc, w_oproj)
                dist.reduce_scatter_tensor(out_rccl, p)
                return out_rccl

            t_oproj_base = _bench_us(_oproj_base)

            t_fp4_ag = _bench_us(
                lambda: sharded_front_fp4_ag(
                    heap, normed_sh, m, w_rd, ROUTER, _topk_fn, use_fp4=True
                )
            )

            def _bf16_ag_base():
                dist.all_gather_into_tensor(normed_full, normed_sh)
                fused = F.linear(normed_full, w_rd)
                return _topk_fn(normed_full, fused[:, :ROUTER])

            t_bf16_base = _bench_us(_bf16_ag_base)

            t_qkv_mix = _bench_us(
                lambda: moe_tail_mix_qkv_a(
                    heap,
                    lat_sh,
                    sh_sh,
                    pr_sh,
                    m,
                    latent_norm_fn=lambda z: rms_norm_bf16(z, w_norm),
                    up_proj_weight=w_up,
                    next_qkv_a_weight=w_qkv_a,
                )
            )

            def _tail_ag_qkv_base():
                up = F.linear(rms_norm_bf16(lat_sh, w_norm), w_up) + sh_sh + pr_sh
                dist.all_gather_into_tensor(normed_full, up)
                return F.linear(normed_full, w_qkv_a)

            t_qkv_base = _bench_us(_tail_ag_qkv_base)

            if rank == 0:
                print(
                    f"M={m:4d} | o_proj+32CU_RS: {t_oproj_ov:7.1f} vs RCCL {t_oproj_base:7.1f} us ({t_oproj_base / t_oproj_ov:.2f}x) "
                    f"| 1/8 front+FP4_AG: {t_fp4_ag:7.1f} vs BF16_AG+full {t_bf16_base:7.1f} us ({t_bf16_base / t_fp4_ag:.2f}x) "
                    f"| tail+qkv_a_2112_AG: {t_qkv_mix:7.1f} vs 7168_AG+full {t_qkv_base:7.1f} us ({t_qkv_base / t_qkv_mix:.2f}x)"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == TP
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=dev)
    cpu_group = dist.new_group(backend="gloo")

    heap = XgmiHeap(cpu_group, rank, dev)
    if rank == 0:
        print(
            f"Initialized XgmiHeap: {heap.heap_mib:.2f} MiB ({heap.heap_bytes / (1 << 30):.3f} GiB) / GPU, "
            f"cu_masked={heap.streams.cu_masked} ({heap.streams.compute_cus}+{heap.streams.comm_cus} CUs)"
        )

    try:
        check_decode(heap, rank, dev, args.bench)
        check_prefill(heap, rank, dev, args.bench)
    finally:
        dist.barrier()
        heap.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
