"""The Kimi-K3 server the serving tests launch on the Quark checkpoint
(amd/Kimi-K3-Quark-MXFP4-AttnFP8): MI35x, TP8, DSpark block 7.

This is the checkpoint that reaches the launches' Quark FP8 attention
projections, and the recipe adds the bf16 KDA state (--mamba-ssm-dtype
bfloat16) and the ReplaySSM ring (--enable-linear-replayssm-spec). It is the
AgentX c1 recipe (sglang-gpu-perf kimi_k3/agentx/run_conc1.sh) without what
only a benchmark needs: simulated acceptance, the 1M context, the chat parsers
and the host's ROCm runtime tuning. As in kimi_k3_server, one request runs a
step, so every decode step is a DSpark verify step of 8 rows.
"""

import os

MODEL = os.environ.get("KIMI_K3_QUARK_MODEL_PATH", "amd/Kimi-K3-Quark-MXFP4-AttnFP8")
DRAFT = os.environ.get(
    "KIMI_K3_DSPARK_MODEL_PATH", "moonshotai/RadixArk-Kimi-K3-DSpark"
)
ENV = {
    "SGLANG_USE_AITER": "1",
    # experts at their native 384-wide intermediate; the old name for SGLang
    # before the rename
    "SGLANG_OPT_HIP_K3_AITER": "1",
    "SGLANG_AITER_K3_OPT": "1",
    "AITER_FLYDSL_FORCE": "1",
    # the MXFP4 experts in AITER's separated A4W4 layout, as the recipe serves them
    "AITER_SITUV2_A4W4": "1",
    "AITER_SITUV2_A8W4": "0",
    "AITER_FLYDSL_STAGE2_FP8": "1",
    # AITER's fused KDA decode, which takes the bf16 state
    "SGLANG_USE_HIP_K3_FLYDSL_SOURCE": "aiter",
    "SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND": "aiter",
    "SGLANG_OPT_HIP_K3_FUSE_KDA_INPROJ": "1",
    "SGLANG_OPT_HIP_K3_RADIX4_TOPK": "1",
    "SGLANG_OPT_HIP_K3_RADIX4_FUSE_SORT": "1",
    "SGLANG_OPT_HIP_K3_MLA_Q_CACHE_FUSION": "1",
    "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT": "1",
    "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MIN_TOKENS": "1",
    "SGLANG_AITER_FP8_PREFILL_ATTN": "1",
    # MLA decode and the 8-token verify on the folded fp8-Q asm kernel
    "SGLANG_AITER_MLA_DECODE_BACKEND": "asm",
    "SGLANG_AITER_MLA_VERIFY_BACKEND": "asm",
    "SGLANG_OPT_HIP_MLA_FP8_Q_ASM": "1",
    "SGLANG_OPT_HIP_MLA_FP8_Q_ASM_MAX_QLEN": "8",
    "SGLANG_AITER_MLA_PERSIST": "0",
    "SGLANG_AITER_MLA_GLUON": "1",
    "SGLANG_AITER_MLA_AUTO_KV_SPLITS": "1",
    "SGLANG_MLA_DECODE_TUNE": "1",
    "SGLANG_MAX_KV_CHUNK_CAPACITY": "65536",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT": "0",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    # the aiter backend derates mem-fraction past a 8192 context; opt out so
    # --mem-fraction-static means what it says
    "SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION": "1",
}
# Without --enable-amd-monokernel, so a test can launch the normal path too.
ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--prefill-attention-backend",
    "aiter",
    "--decode-attention-backend",
    "aiter",
    "--dtype",
    "bfloat16",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--mem-fraction-static",
    "0.9",
    # one request a step: see kimi_k3_server
    "--max-running-requests",
    "1",
    "--cuda-graph-max-bs-decode",
    "1",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--mamba-track-interval",
    "1024",
    "--max-mamba-cache-size",
    "224",
    "--mamba-max-states-per-path",
    "2",
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "16384",
    "--page-size",
    "128",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DRAFT,
    "--speculative-dspark-block-size",
    "7",
    "--enable-linear-replayssm-spec",
]
