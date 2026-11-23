# SPDX-License-Identifier: Apache-2.0
"""Unified <= 0.4 GiB XGMI symmetric peer heap and comm stream (gfx950 TP8).

Layout inside one ``PeerBuffer`` (uncached HIP IPC memory, 2 MiB granule):
  - Region 0 (decode_lamport): 2 parity halves x 16,515,072 B  = 31.50 MiB
  - Region 1 (prefill_rs_ring): 4 ring slots x 29,360,128 B    = 112.00 MiB
  - Region 2 (prefill_ag): 2 parity halves x 36,700,160 B      = 70.00 MiB
  - Region 3 (flags): RS ring flags + AG flags                 = 0.125 MiB
  Total: 214 MiB (0.209 GiB) per GPU -- zero replicated model weights.
"""

from __future__ import annotations

import ctypes
import functools
import os
from typing import Optional

import msgspec
import torch

from sgl_amd_mono.common.peer import PeerBuffer, hip_runtime

SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
}

TP = 8

DECODE_MAX_ROWS = 192
DECODE_MAX_COLS = 10752  # 3584 latent + 7168 shared
DECODE_BLOCKS = 128

RS_MAX_CHUNK_ROWS = 2048
RS_MAX_COLS = 7168
RS_RING_SLOTS = 4
RS_BLOCKS = 64

PREFILL_MAX_ROWS = 8192
PREFILL_AG_HALF_BYTES = 35 << 20  # >= 8192 * 4224 B (2112-dim bf16 qkv_a)
PREFILL_AG_BLOCKS = 64

# Match wide.py's 900 s watchdog so allocator GC stalls on 32K prefill batches do not trap peers.
DEFAULT_DEADLINE_S = 900.0

_ALIGN = 4096


def _align_up(nbytes: int, alignment: int = _ALIGN) -> int:
    return (nbytes + alignment - 1) // alignment * alignment


DECODE_RS_BYTES = _align_up(DECODE_MAX_ROWS * DECODE_MAX_COLS * 4)
DECODE_AG_BYTES = _align_up(DECODE_MAX_ROWS * DECODE_MAX_COLS * 4)
DECODE_HALF_BYTES = _align_up(DECODE_RS_BYTES + DECODE_AG_BYTES)
DECODE_REGION_OFF = 0
DECODE_REGION_BYTES = 2 * DECODE_HALF_BYTES

RS_SLOT_BYTES = _align_up(RS_MAX_CHUNK_ROWS * RS_MAX_COLS * 2)
RS_REGION_OFF = _align_up(DECODE_REGION_OFF + DECODE_REGION_BYTES)
RS_REGION_BYTES = RS_RING_SLOTS * RS_SLOT_BYTES

AG_REGION_OFF = _align_up(RS_REGION_OFF + RS_REGION_BYTES)
AG_REGION_BYTES = 2 * PREFILL_AG_HALF_BYTES

RS_FLAGS_OFF = _align_up(AG_REGION_OFF + AG_REGION_BYTES)
RS_FLAGS_BYTES = _align_up(RS_RING_SLOTS * RS_BLOCKS * TP * 4)
AG_FLAGS_OFF = _align_up(RS_FLAGS_OFF + RS_FLAGS_BYTES)
AG_FLAGS_BYTES = _align_up(2 * PREFILL_AG_BLOCKS * TP * 4)

HEAP_BYTES = _align_up(AG_FLAGS_OFF + AG_FLAGS_BYTES, 2 << 20)

# Per-block monotonic epoch table offsets (each block updates only its own word -> CUDA-graph safe).
EPOCH_DECODE_OFF = 0
EPOCH_RS_OFF = DECODE_BLOCKS
EPOCH_AG_OFF = EPOCH_RS_OFF + RS_BLOCKS
EPOCH_TOTAL_WORDS = EPOCH_AG_OFF + PREFILL_AG_BLOCKS


class XgmiConfig(msgspec.Struct, frozen=True):
    """Immutable runtime settings resolved once at hook installation."""

    prefill_min_rows: int = 512
    prefill_max_rows: int = 32768
    use_fp4: bool = False
    enable_decode: bool = True
    deadline_s: float = DEFAULT_DEADLINE_S


@functools.cache
def _bind_cu_mask_api():
    lib = hip_runtime()
    try:
        create_fn = lib.hipExtStreamCreateWithCUMask
        destroy_fn = lib.hipStreamDestroy
    except AttributeError:
        return None
    create_fn.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    create_fn.restype = ctypes.c_int
    destroy_fn.argtypes = [ctypes.c_void_p]
    destroy_fn.restype = ctypes.c_int
    return create_fn, destroy_fn


class CuMaskStreams:
    """Compute + high-priority XGMI comm streams on MI355X.

    ``use_cu_mask`` defaults to False because ROCm 7.2 ``hipExtStreamCreateWithCUMask``
    regresses hipBLASLt GEMM latency ~2x (81.5 -> 171.6 us at M=4096).
    """

    def __init__(
        self,
        device: torch.device,
        *,
        comm_cus: int = 32,
        use_cu_mask: bool = False,
    ):
        self.device = device
        self.comm_cus = comm_cus
        self._raw_handles: list[int] = []
        props = torch.cuda.get_device_properties(device)
        total_cus = props.multi_processor_count
        api = _bind_cu_mask_api()
        if (
            use_cu_mask
            and api is not None
            and total_cus >= comm_cus + 32
            and total_cus % 32 == 0
        ):
            if self._init_masked(api, device, total_cus, comm_cus):
                return
        self.cu_masked = False
        self.compute_cus = total_cus
        self.compute = torch.cuda.Stream(device=device)
        self.comm = torch.cuda.Stream(device=device, priority=-1)

    def _init_masked(
        self, api, device: torch.device, total_cus: int, comm_cus: int
    ) -> bool:
        create_fn, destroy_fn = api
        num_words = total_cus // 32
        comm_per_word = max(1, comm_cus // num_words)
        comm_bits = ((1 << comm_per_word) - 1) << (32 - comm_per_word)
        comp_bits = 0xFFFFFFFF ^ comm_bits
        compute_mask = (ctypes.c_uint32 * num_words)(*([comp_bits] * num_words))
        comm_mask = (ctypes.c_uint32 * num_words)(*([comm_bits] * num_words))
        with torch.cuda.device(device):
            h_comp = ctypes.c_void_p()
            h_comm = ctypes.c_void_p()
            err1 = create_fn(ctypes.byref(h_comp), num_words, compute_mask)
            err2 = create_fn(ctypes.byref(h_comm), num_words, comm_mask)
            if err1 == 0 and err2 == 0:
                self._raw_handles = [int(h_comp.value), int(h_comm.value)]
                self.compute = torch.cuda.ExternalStream(
                    int(h_comp.value), device=device
                )
                self.comm = torch.cuda.ExternalStream(int(h_comm.value), device=device)
                self.cu_masked = True
                self.compute_cus = total_cus - comm_cus
                return True
            if err1 == 0:
                destroy_fn(h_comp)
            if err2 == 0:
                destroy_fn(h_comm)
        return False

    def close(self) -> None:
        api = _bind_cu_mask_api()
        if api is not None:
            for handle in self._raw_handles:
                api[1](ctypes.c_void_p(handle))
        self._raw_handles = []


class XgmiHeap:
    """Single <= 0.4 GiB symmetric XGMI heap shared by Decode and Prefill."""

    def __init__(
        self,
        cpu_group,
        rank: int,
        device: torch.device,
        *,
        world_size: int = TP,
        comm_cus: int = 32,
        deadline_s: Optional[float] = None,
    ):
        if world_size != TP:
            raise ValueError(f"XgmiHeap requires TP={TP}, got {world_size}")
        if deadline_s is None:
            deadline_s = float(
                os.environ.get(
                    "SGLANG_AMD_MONO_K31_XGMI_DEADLINE_S", str(DEFAULT_DEADLINE_S)
                )
            )
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.deadline_s = deadline_s
        self.peer = PeerBuffer(HEAP_BYTES, cpu_group, rank, world_size, device)
        self.peer.bytes.zero_()
        self.peer_ptrs_cpu = self.peer.addresses.to(
            "cpu", non_blocking=False
        ).contiguous()
        self.block_epoch = torch.zeros(
            EPOCH_TOTAL_WORDS, dtype=torch.int32, device=device
        )
        self.streams = CuMaskStreams(device, comm_cus=comm_cus)
        self.chunk_events = [torch.cuda.Event() for _ in range(8)]

    @property
    def heap_bytes(self) -> int:
        return HEAP_BYTES

    @property
    def heap_mib(self) -> float:
        return HEAP_BYTES / (1 << 20)

    def close(self) -> None:
        self.streams.close()
        self.peer.close()


_GLOBAL_HEAP: Optional[XgmiHeap] = None


def get_global_heap(
    *,
    cpu_group=None,
    rank: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> XgmiHeap:
    """Return the process-wide singleton XgmiHeap (collective on first call)."""
    global _GLOBAL_HEAP
    if _GLOBAL_HEAP is None:
        if cpu_group is None:
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
            cpu_group = parallel.tp_group.cpu_group
            rank = parallel.tp_rank
            device = torch.device("cuda", torch.cuda.current_device())
        assert rank is not None and device is not None
        _GLOBAL_HEAP = XgmiHeap(cpu_group, rank, device)
    return _GLOBAL_HEAP
