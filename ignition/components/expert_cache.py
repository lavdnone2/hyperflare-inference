# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Device-side SLRU expert cache and in-graph MappedHost -> HBM staging.

Each MoE layer keeps all E canonical experts in page-locked MappedHost memory
and S < E active slots in an HBM b12x PreparedExperts pool. Inside captured
CUDA graphs (zero host sync, zero allocation, deterministic across TP ranks):

  1. _slru_resolve_kernel maps canonical router IDs [tokens, top_k] to HBM slot
     IDs [0, S) (preserving negative padded IDs), updates 25%/75% segmented LRU
     metadata, selects deterministic non-busy victims, and emits a deduplicated
     (miss_src, miss_dst, miss_count) copy list.
  2. _copy_expert_plane_kernel streams missed expert weight/scale planes from
     MappedHost into victim HBM slots using a fixed 8-CTA grid with .cg loads.
  3. _copy_expert_scalars_kernel updates per-expert FP32 global/activation scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import triton
import triton.language as tl

from ignition.preparation.persistent import persistent_empty, persistent_zeros
from ignition.preparation.triton import launch_triton, prepare_triton

DEFAULT_COPY_CTAS = 8
MAX_PLANES = 12


@triton.jit
def _slru_resolve_kernel(
    ids_ptr,
    mapped_ids_ptr,
    ovf_ids_ptr,
    expert_to_slot_ptr,
    slot_to_expert_ptr,
    slot_tier_ptr,
    slot_ts_ptr,
    slot_busy_ptr,
    step_clock_ptr,
    stats_ptr,
    miss_src_ptr,
    miss_dst_ptr,
    miss_count_ptr,
    ovf_src_ptr,
    ovf_dst_ptr,
    ovf_count_ptr,
    M: tl.constexpr,
    E: tl.constexpr,
    S: tl.constexpr,
    S_POW2: tl.constexpr,
    PROT_CAP: tl.constexpr,
    MAX_MISSES: tl.constexpr,
    OVF_CAP: tl.constexpr,
    OVF_POW2: tl.constexpr,
    UPDATE_CACHE: tl.constexpr,
    HANDLE_OVERFLOW: tl.constexpr,
):
    offs_s = tl.arange(0, S_POW2)
    in_s = offs_s < S
    INF: tl.constexpr = 1 << 62
    TIER1_OFF: tl.constexpr = 1 << 50
    TIER2_OFF: tl.constexpr = 2 << 50

    # Clear per-step busy flags and advance monotonic step clock.
    tl.store(slot_busy_ptr + offs_s, 0, mask=in_s)
    step = tl.load(step_clock_ptr) + 1
    if UPDATE_CACHE:
        tl.store(step_clock_ptr, step)
    tl.debug_barrier()

    base_ts = step * (M + 1)
    lookups = tl.zeros((), dtype=tl.int64)
    hits = tl.zeros((), dtype=tl.int64)
    promotions = tl.zeros((), dtype=tl.int64)

    # Pass 1: resolve hits, mark their slots busy, and promote probationary hits -> protected.
    for i in range(M):
        if HANDLE_OVERFLOW:
            tl.store(ovf_ids_ptr + i, -1)
        e = tl.load(ids_ptr + i)
        if (e >= 0) & (e < E):
            lookups += 1
            slot = tl.load(expert_to_slot_ptr + e)
            if slot >= 0:
                tl.store(mapped_ids_ptr + i, slot)
                tl.store(slot_busy_ptr + slot, 1)
                if UPDATE_CACHE:
                    tl.store(slot_ts_ptr + slot, base_ts + i)
                    tier = tl.load(slot_tier_ptr + slot)
                    if tier == 0:
                        tl.debug_barrier()
                        s_tier = tl.load(slot_tier_ptr + offs_s, mask=in_s, other=0)
                        s_exp = tl.load(slot_to_expert_ptr + offs_s, mask=in_s, other=-1)
                        is_prot = in_s & (s_tier == 1) & (s_exp >= 0)
                        n_prot = tl.sum(is_prot.to(tl.int32), axis=0)
                        if n_prot >= PROT_CAP:
                            s_ts = tl.load(slot_ts_ptr + offs_s, mask=in_s, other=0)
                            dkey = tl.where(is_prot, (s_ts << 10) + offs_s.to(tl.int64), INF)
                            demote = tl.argmin(dkey, axis=0)
                            tl.store(slot_tier_ptr + demote, 0)
                        tl.store(slot_tier_ptr + slot, 1)
                        promotions += 1
                hits += 1
            else:
                tl.store(mapped_ids_ptr + i, -1)
        else:
            tl.store(mapped_ids_ptr + i, -1)
        tl.debug_barrier()

    # Pass 2: assign deterministic non-busy SLRU victims for unique misses.
    n_miss = tl.zeros((), dtype=tl.int32)
    n_ovf = tl.zeros((), dtype=tl.int32)
    offs_o = tl.arange(0, OVF_POW2)
    if UPDATE_CACHE:
        for i in range(M):
            e = tl.load(ids_ptr + i)
            if (e >= 0) & (e < E):
                slot = tl.load(expert_to_slot_ptr + e)
                if slot >= 0:
                    tl.store(mapped_ids_ptr + i, slot)
                else:
                    tl.debug_barrier()
                    s_exp = tl.load(slot_to_expert_ptr + offs_s, mask=in_s, other=0)
                    s_busy = tl.load(slot_busy_ptr + offs_s, mask=in_s, other=1)
                    avail = in_s & (s_busy == 0)
                    has_avail = tl.sum(avail.to(tl.int32), axis=0) > 0
                    if has_avail:
                        s_tier = tl.load(slot_tier_ptr + offs_s, mask=in_s, other=2)
                        s_ts = tl.load(slot_ts_ptr + offs_s, mask=in_s, other=0)
                        is_empty = avail & (s_exp < 0)
                        is_prob = avail & (s_exp >= 0) & (s_tier == 0)
                        is_prot = avail & (s_exp >= 0) & (s_tier == 1)
                        ts_key = (s_ts << 10) + offs_s.to(tl.int64)
                        vkey = tl.where(
                            is_empty,
                            offs_s.to(tl.int64),
                            tl.where(
                                is_prob,
                                TIER1_OFF + ts_key,
                                tl.where(is_prot, TIER2_OFF + ts_key, INF),
                            ),
                        )
                        victim = tl.argmin(vkey, axis=0)
                        old_e = tl.load(slot_to_expert_ptr + victim)
                        if old_e >= 0:
                            tl.store(expert_to_slot_ptr + old_e, -1)
                        tl.store(slot_to_expert_ptr + victim, e)
                        tl.store(expert_to_slot_ptr + e, victim)
                        tl.store(slot_tier_ptr + victim, 0)
                        tl.store(slot_ts_ptr + victim, base_ts + i)
                        tl.store(slot_busy_ptr + victim, 1)
                        tl.store(mapped_ids_ptr + i, victim)
                        if n_miss < MAX_MISSES:
                            tl.store(miss_src_ptr + n_miss, e)
                            tl.store(miss_dst_ptr + n_miss, victim)
                            n_miss += 1
                    else:
                        if HANDLE_OVERFLOW:
                            in_o = offs_o < n_ovf
                            o_exp = tl.load(ovf_src_ptr + offs_o, mask=in_o, other=-1)
                            o_match = in_o & (o_exp == e)
                            if tl.sum(o_match.to(tl.int32), axis=0) > 0:
                                ovf_slot = tl.argmin(tl.where(o_match, offs_o, OVF_POW2), axis=0)
                                tl.store(ovf_ids_ptr + i, ovf_slot)
                            else:
                                if n_ovf < OVF_CAP:
                                    tl.store(ovf_src_ptr + n_ovf, e)
                                    tl.store(ovf_dst_ptr + n_ovf, n_ovf)
                                    tl.store(ovf_ids_ptr + i, n_ovf)
                                    n_ovf += 1
                    tl.debug_barrier()

    tl.store(miss_count_ptr, n_miss)
    if HANDLE_OVERFLOW:
        tl.store(ovf_count_ptr, n_ovf)
    tl.store(stats_ptr + 0, tl.load(stats_ptr + 0) + lookups)
    tl.store(stats_ptr + 1, tl.load(stats_ptr + 1) + hits)
    tl.store(stats_ptr + 2, tl.load(stats_ptr + 2) + (lookups - hits))
    tl.store(stats_ptr + 3, tl.load(stats_ptr + 3) + promotions)


@triton.jit
def _copy_expert_plane_kernel(
    dst_ptr,
    src_ptr,
    miss_src_ptr,
    miss_dst_ptr,
    miss_count_ptr,
    ROW_I32: tl.constexpr,
    BLOCK_I32: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    """Stream missed expert rows (viewed as int32) from MappedHost to HBM across a fixed CTA grid."""
    n_miss = tl.load(miss_count_ptr)
    if n_miss <= 0:
        return
    tiles_per_row: tl.constexpr = (ROW_I32 + BLOCK_I32 - 1) // BLOCK_I32
    total_tiles = n_miss * tiles_per_row
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_I32)
    t = pid
    while t < total_tiles:
        m_idx = t // tiles_per_row
        tile = t - m_idx * tiles_per_row
        src_e = tl.load(miss_src_ptr + m_idx).to(tl.int64)
        dst_s = tl.load(miss_dst_ptr + m_idx).to(tl.int64)
        col = tile * BLOCK_I32 + offs
        mask = col < ROW_I32
        vals = tl.load(
            src_ptr + src_e * ROW_I32 + col,
            mask=mask,
            cache_modifier=".cg",
        )
        tl.store(dst_ptr + dst_s * ROW_I32 + col, vals, mask=mask)
        t += NUM_CTAS


@triton.jit
def _copy_expert_scalars_kernel(
    dst0_ptr, src0_ptr,
    dst1_ptr, src1_ptr,
    dst2_ptr, src2_ptr,
    dst3_ptr, src3_ptr,
    miss_src_ptr,
    miss_dst_ptr,
    miss_count_ptr,
    HAS_2: tl.constexpr,
    HAS_3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy up to 4 per-expert scalar (int32/float32) vectors for all misses in one CTA."""
    n_miss = tl.load(miss_count_ptr)
    if n_miss <= 0:
        return
    offs = tl.arange(0, BLOCK)
    mask = offs < n_miss
    src_e = tl.load(miss_src_ptr + offs, mask=mask, other=0).to(tl.int64)
    dst_s = tl.load(miss_dst_ptr + offs, mask=mask, other=0).to(tl.int64)
    tl.store(dst0_ptr + dst_s, tl.load(src0_ptr + src_e, mask=mask), mask=mask)
    tl.store(dst1_ptr + dst_s, tl.load(src1_ptr + src_e, mask=mask), mask=mask)
    if HAS_2:
        tl.store(dst2_ptr + dst_s, tl.load(src2_ptr + src_e, mask=mask), mask=mask)
    if HAS_3:
        tl.store(dst3_ptr + dst_s, tl.load(src3_ptr + src_e, mask=mask), mask=mask)


@triton.jit
def _prefill_resident_ids_kernel(
    ids_ptr,
    mapped_ids_ptr,
    expert_to_slot_ptr,
    M: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Map resident canonical IDs -> HBM slot IDs and mask cold experts to -1 (read-only cache)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    e = tl.load(ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (e >= 0) & (e < E)
    slot = tl.load(expert_to_slot_ptr + tl.where(valid, e, 0), mask=valid, other=-1)
    tl.store(mapped_ids_ptr + offs, tl.where(valid, slot, -1), mask=mask)


@triton.jit
def _prefill_slice_map_kernel(
    ids_ptr,
    slice_ids_ptr,
    expert_to_slot_ptr,
    slice_present_ptr,
    M: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Select non-resident experts in [LO, HI) into local staging slots [0, HI-LO) and mark presence."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    e = tl.load(ids_ptr + offs, mask=mask, other=-1)
    in_slice = mask & (e >= LO) & (e < HI)
    slot = tl.load(expert_to_slot_ptr + tl.where(in_slice, e, 0), mask=in_slice, other=0)
    cold = in_slice & (slot < 0)
    local = e - LO
    tl.store(slice_ids_ptr + offs, tl.where(cold, local, -1), mask=mask)
    tl.store(slice_present_ptr + tl.where(cold, local, 0), 1, mask=cold)


@triton.jit
def _prefill_slice_compact_kernel(
    slice_present_ptr,
    miss_src_ptr,
    miss_dst_ptr,
    miss_count_ptr,
    LO: tl.constexpr,
    K: tl.constexpr,
):
    """Compact marked experts in [LO, LO+K) into (miss_src, miss_dst, miss_count) and reset flags."""
    n = tl.zeros((), dtype=tl.int32)
    for k in range(K):
        p = tl.load(slice_present_ptr + k)
        if p != 0:
            tl.store(miss_src_ptr + n, LO + k)
            tl.store(miss_dst_ptr + n, k)
            tl.store(slice_present_ptr + k, 0)
            n += 1
    tl.store(miss_count_ptr, n)


@triton.jit
def _add_bf16_inplace_kernel(
    acc_ptr,
    delta_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(acc_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(delta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(acc_ptr + offs, (a + b).to(tl.bfloat16), mask=mask)


@dataclass(frozen=True)
class ExpertPlanePair:
    """One expert-indexed tensor plane shared between the HBM slot pool and MappedHost tier."""

    name: str
    dst_hbm: torch.Tensor
    src_host_dev: torch.Tensor
    row_bytes: int


def collect_expert_planes(
    hbm_experts,
    host_planes: dict[str, torch.Tensor],
) -> tuple[tuple[ExpertPlanePair, ...], tuple[ExpertPlanePair, ...]]:
    """Return (bulk_planes, scalar_planes) pairing HBM PreparedExperts tensors with MappedHost views.

    Deduplicates aliases by HBM data_ptr() so shared views (e.g. w13_scale aliasing w1_blockscale)
    are copied exactly once.
    """
    impl = hbm_experts._impl if hasattr(hbm_experts, "_impl") else hbm_experts
    num_slots = int(hbm_experts.num_experts)
    seen_ptrs: set[int] = set()
    bulk: list[ExpertPlanePair] = []
    scalars: list[ExpertPlanePair] = []

    def register(name: str, dst: torch.Tensor | None):
        if dst is None or not isinstance(dst, torch.Tensor):
            return
        if dst.ndim < 1 or dst.shape[0] != num_slots:
            return
        ptr = dst.data_ptr()
        if ptr in seen_ptrs:
            return
        if name not in host_planes:
            raise KeyError(f"MappedHost tier is missing expert plane {name!r} required by HBM pool")
        src = host_planes[name]
        if not dst.is_contiguous() or not src.is_contiguous():
            raise ValueError(f"expert plane {name!r} must be contiguous in both HBM and MappedHost")
        if dst.dtype != src.dtype or dst.shape[1:] != src.shape[1:]:
            raise ValueError(
                f"expert plane {name!r} shape/dtype mismatch: HBM {tuple(dst.shape)} {dst.dtype} "
                f"vs host {tuple(src.shape)} {src.dtype}"
            )
        row_bytes = dst[0].numel() * dst.element_size()
        if row_bytes % 4 != 0:
            raise ValueError(f"expert plane {name!r} row size ({row_bytes} B) must be 4-byte aligned")
        seen_ptrs.add(ptr)
        pair = ExpertPlanePair(name=name, dst_hbm=dst, src_host_dev=src, row_bytes=row_bytes)
        if row_bytes == 4:
            scalars.append(pair)
        else:
            bulk.append(pair)

    for field in (
        "w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
        "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale",
    ):
        register(field, getattr(impl, field, None))

    representation = getattr(impl, "representation", None)
    if representation is not None:
        rep_val = getattr(representation, "value", representation)
        for field in (
            "w13_scale", "w2_scale", "w13_sfb", "w2_sfb",
            "w13_global_scale", "w2_global_scale",
            "micro_w13_scale", "micro_w2_scale",
            "micro_w13_global_scale", "micro_w2_global_scale",
        ):
            register(field, getattr(rep_val, field, None))

    if not bulk:
        raise ValueError("no bulk expert planes found on HBM PreparedExperts")
    return tuple(bulk), tuple(scalars)


class ExpertCacheState:
    """Persistent device-side SLRU tables for one MoE layer."""

    def __init__(
        self,
        num_experts: int,
        num_slots: int,
        device: torch.device,
        *,
        initial_experts: Sequence[int] | None = None,
    ):
        if not 1 <= num_slots <= num_experts <= 1024:
            raise ValueError(f"invalid cache geometry: slots={num_slots}, experts={num_experts}")
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.prot_cap = max(1, (num_slots * 3) // 4)
        self.device = torch.device(device)

        self.expert_to_slot = persistent_empty((num_experts,), torch.int32, self.device)
        self.slot_to_expert = persistent_empty((num_slots,), torch.int32, self.device)
        self.slot_tier = persistent_zeros((num_slots,), torch.int32, self.device)
        self.slot_ts = persistent_zeros((num_slots,), torch.int64, self.device)
        self.slot_busy = persistent_zeros((num_slots,), torch.int32, self.device)
        self.step_clock = persistent_zeros((1,), torch.int64, self.device)
        self.stats = persistent_zeros((4,), torch.int64, self.device)
        self.initial_experts = None if initial_experts is None else tuple(int(e) for e in initial_experts)[:num_slots]

        self.reset(initial_experts=self.initial_experts)

    def reset_to_initial(self) -> None:
        """Restore initial residency, clean SLRU tiers/timestamps, and zeroed stats."""
        self.reset(initial_experts=self.initial_experts)

    def reset(self, *, initial_experts: Sequence[int] | None = None) -> None:
        """Initialize residency from a ranked expert list (top 75% protected, rest probationary)."""
        e2s = torch.full((self.num_experts,), -1, dtype=torch.int32)
        s2e = torch.full((self.num_slots,), -1, dtype=torch.int32)
        tier = torch.zeros((self.num_slots,), dtype=torch.int32)
        ts = torch.zeros((self.num_slots,), dtype=torch.int64)
        if initial_experts is not None:
            init = list(initial_experts)[: self.num_slots]
            if len(set(init)) != len(init):
                raise ValueError("initial_experts must be unique")
            for slot, e in enumerate(init):
                if not 0 <= int(e) < self.num_experts:
                    raise ValueError(f"initial expert id {e} out of range [0, {self.num_experts})")
                e2s[e] = slot
                s2e[slot] = e
                # Most frequent experts start in protected tier with higher timestamps (evicted last).
                tier[slot] = 1 if slot < self.prot_cap else 0
                ts[slot] = len(init) - slot
        self.expert_to_slot.copy_(e2s)
        self.slot_to_expert.copy_(s2e)
        self.slot_tier.copy_(tier)
        self.slot_ts.copy_(ts)
        self.slot_busy.zero_()
        self.step_clock.zero_()
        self.stats.zero_()

    def copy_initial_weights(
        self,
        bulk_planes: Sequence[ExpertPlanePair],
        scalar_planes: Sequence[ExpertPlanePair],
    ) -> None:
        """Populate HBM slots from MappedHost for all initially resident experts."""
        s2e = self.slot_to_expert.cpu().tolist()
        resident_slots = [s for s, e in enumerate(s2e) if e >= 0]
        if not resident_slots:
            return
        src_idx = torch.tensor([s2e[s] for s in resident_slots], dtype=torch.int64, device=self.device)
        dst_idx = torch.tensor(resident_slots, dtype=torch.int64, device=self.device)
        for pair in (*bulk_planes, *scalar_planes):
            dst_u8 = pair.dst_hbm.view(torch.uint8).reshape(self.num_slots, pair.row_bytes)
            src_u8 = pair.src_host_dev.view(torch.uint8).reshape(self.num_experts, pair.row_bytes)
            dst_u8.index_copy_(0, dst_idx, src_u8.index_select(0, src_idx))

    def checksum(self) -> int:
        """Deterministic host-side checksum of residency + SLRU state (for TP2 lockstep checks)."""
        parts = (
            self.expert_to_slot.cpu(),
            self.slot_to_expert.cpu(),
            self.slot_tier.cpu(),
            self.slot_ts.cpu(),
        )
        acc = 0x9E3779B97F4A7C15
        for t in parts:
            for v in t.tolist():
                acc = ((acc ^ (int(v) & 0xFFFFFFFFFFFFFFFF)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        return acc

    def summary_stats(self) -> dict[str, int]:
        vals = self.stats.cpu().tolist()
        return {"lookups": vals[0], "hits": vals[1], "misses": vals[2], "promotions": vals[3]}


class PreparedExpertCache:
    """CUDA-graph-capturable SLRU resolver + MappedHost -> HBM miss stager for one shape."""

    def __init__(
        self,
        state: ExpertCacheState,
        bulk_planes: Sequence[ExpertPlanePair],
        scalar_planes: Sequence[ExpertPlanePair],
        *,
        tokens: int,
        top_k: int,
        copy_ctas: int = DEFAULT_COPY_CTAS,
        update_cache: bool = True,
        staging_bulk_planes: Sequence[ExpertPlanePair] = (),
        staging_scalar_planes: Sequence[ExpertPlanePair] = (),
        staging_slots: int = 0,
    ):
        if tokens <= 0 or top_k <= 0:
            raise ValueError("tokens and top_k must be positive")
        routes = tokens * top_k
        handle_overflow = bool(update_cache and routes > state.num_slots)
        if handle_overflow and (staging_slots <= 0 or routes > state.num_slots + staging_slots):
            raise ValueError(
                f"decode route count ({tokens}*{top_k}={routes}) exceeds layer slot pool "
                f"({state.num_slots}) + overflow staging ({staging_slots}); use streaming prefill path"
            )
        self.state = state
        self.tokens = tokens
        self.top_k = top_k
        self.routes = routes
        self.copy_ctas = copy_ctas
        self.update_cache = update_cache
        self.has_overflow = handle_overflow
        self.staging_slots = staging_slots if handle_overflow else 0
        self.device = state.device
        self.max_misses = min(routes, state.num_slots) if update_cache else 0

        # Persistent per-shape scratch buffers for miss list (tiny: <= 1 KiB).
        self.miss_src = persistent_zeros((max(1, self.max_misses),), torch.int32, self.device)
        self.miss_dst = persistent_zeros((max(1, self.max_misses),), torch.int32, self.device)
        self.miss_count = persistent_zeros((1,), torch.int32, self.device)

        ovf_cap = max(1, self.staging_slots)
        ovf_pow2 = max(16, triton.next_power_of_2(ovf_cap))
        self.ovf_src = persistent_zeros((ovf_cap,), torch.int32, self.device)
        self.ovf_dst = persistent_zeros((ovf_cap,), torch.int32, self.device)
        self.ovf_count = persistent_zeros((1,), torch.int32, self.device)

        s_pow2 = max(16, triton.next_power_of_2(state.num_slots))
        dummy_ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=self.device)
        dummy_mapped = torch.zeros((tokens, top_k), dtype=torch.int32, device=self.device)
        dummy_ovf = torch.zeros((tokens, top_k), dtype=torch.int32, device=self.device)

        self._resolve = prepare_triton(
            _slru_resolve_kernel,
            (
                dummy_ids,
                dummy_mapped,
                dummy_ovf,
                state.expert_to_slot,
                state.slot_to_expert,
                state.slot_tier,
                state.slot_ts,
                state.slot_busy,
                state.step_clock,
                state.stats,
                self.miss_src,
                self.miss_dst,
                self.miss_count,
                self.ovf_src,
                self.ovf_dst,
                self.ovf_count,
                routes,
                state.num_experts,
                state.num_slots,
                s_pow2,
                state.prot_cap,
                max(1, self.max_misses),
                ovf_cap,
                ovf_pow2,
                int(update_cache),
                int(handle_overflow),
            ),
            (1, 1, 1),
            num_warps=1,
        )

        def _build_copies(b_planes, s_planes, dst_slots, m_src, m_dst, m_cnt, max_m):
            b_ops = []
            for pair in b_planes:
                row_i32 = pair.row_bytes // 4
                dst_i32 = pair.dst_hbm.view(torch.uint8).reshape(dst_slots, pair.row_bytes).view(torch.int32)
                src_i32 = pair.src_host_dev.view(torch.uint8).reshape(state.num_experts, pair.row_bytes).view(torch.int32)
                block_i32 = 4096 if row_i32 >= 4096 else max(64, triton.next_power_of_2(row_i32))
                b_ops.append(
                    prepare_triton(
                        _copy_expert_plane_kernel,
                        (dst_i32, src_i32, m_src, m_dst, m_cnt, row_i32, block_i32, copy_ctas),
                        (copy_ctas, 1, 1),
                        num_warps=8,
                    )
                )
            s_ops = []
            s_block = max(16, triton.next_power_of_2(max(1, max_m)))
            s_list = list(s_planes)
            for idx in range(0, len(s_list), 4):
                chunk = s_list[idx : idx + 4]
                while len(chunk) < 4:
                    chunk.append(chunk[-1])
                c_len = min(4, len(s_list) - idx)
                args = []
                for p in chunk:
                    args.append(p.dst_hbm.view(torch.int32).reshape(-1))
                    args.append(p.src_host_dev.view(torch.int32).reshape(-1))
                s_ops.append(
                    prepare_triton(
                        _copy_expert_scalars_kernel,
                        (*args, m_src, m_dst, m_cnt, int(c_len >= 3), int(c_len >= 4), s_block),
                        (1, 1, 1),
                        num_warps=1,
                    )
                )
            return b_ops, s_ops

        self._bulk_copies, self._scalar_copies = (
            _build_copies(
                bulk_planes, scalar_planes, state.num_slots,
                self.miss_src, self.miss_dst, self.miss_count, self.max_misses,
            )
            if update_cache else ([], [])
        )
        self._ovf_bulk_copies, self._ovf_scalar_copies = (
            _build_copies(
                staging_bulk_planes, staging_scalar_planes, staging_slots,
                self.ovf_src, self.ovf_dst, self.ovf_count, staging_slots,
            )
            if handle_overflow else ([], [])
        )

    def __call__(
        self,
        ids: torch.Tensor,
        mapped_ids: torch.Tensor | None = None,
        ovf_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Resolve canonical IDs -> HBM slot IDs and stage any missed experts from MappedHost."""
        if mapped_ids is None:
            mapped_ids = torch.empty_like(ids)
        if self.has_overflow:
            if ovf_ids is None:
                ovf_ids = torch.empty_like(ids)
            ovf_arg = ovf_ids
        else:
            ovf_arg = mapped_ids
        launch_triton(
            self._resolve.kernel,
            self._resolve.compiled,
            (ids, mapped_ids, ovf_arg, *self._resolve.args[3:]),
            self._resolve.grid,
            **self._resolve.options,
        )
        for op in self._bulk_copies:
            op()
        for op in self._scalar_copies:
            op()
        if self.has_overflow:
            for op in self._ovf_bulk_copies:
                op()
            for op in self._ovf_scalar_copies:
                op()
            return mapped_ids, ovf_ids
        return mapped_ids


class PreparedPrefillExpertStreamer:
    """CUDA-graph-capturable multi-pass prefill streaming over a shared K-slot HBM staging pool.

    Leaves the layer's decode SLRU cache state untouched:
      - Pass 0 runs resident hits directly on the layer's HBM slot pool (zero PCIe copy).
      - Passes 1..ceil(E/K) copy only demanded cold experts in each [lo, hi) slice from
        MappedHost into the shared K-slot HBM staging pool, run staging b12x, and accumulate.
    """

    def __init__(
        self,
        state: ExpertCacheState,
        staging_bulk_planes: Sequence[ExpertPlanePair],
        staging_scalar_planes: Sequence[ExpertPlanePair],
        *,
        tokens: int,
        top_k: int,
        hidden_size: int,
        staging_slots: int = 64,
        copy_ctas: int = DEFAULT_COPY_CTAS,
    ):
        if tokens <= 0 or top_k <= 0 or staging_slots <= 0:
            raise ValueError("tokens, top_k, and staging_slots must be positive")
        self.state = state
        self.tokens = tokens
        self.top_k = top_k
        self.routes = tokens * top_k
        self.hidden_size = hidden_size
        self.staging_slots = staging_slots
        self.device = state.device
        self.num_passes = (state.num_experts + staging_slots - 1) // staging_slots

        self.slice_present = persistent_zeros((staging_slots,), torch.int32, self.device)
        self.miss_src = persistent_zeros((staging_slots,), torch.int32, self.device)
        self.miss_dst = persistent_zeros((staging_slots,), torch.int32, self.device)
        self.miss_count = persistent_zeros((1,), torch.int32, self.device)

        dummy_ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=self.device)
        dummy_mapped = torch.zeros((tokens, top_k), dtype=torch.int32, device=self.device)
        dummy_out = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16, device=self.device)

        self._map_resident = prepare_triton(
            _prefill_resident_ids_kernel,
            (dummy_ids, dummy_mapped, state.expert_to_slot, self.routes, state.num_experts, 256),
            (triton.cdiv(self.routes, 256), 1, 1),
            num_warps=4,
        )

        self._slice_maps = []
        self._slice_compacts = []
        for p in range(self.num_passes):
            lo = p * staging_slots
            hi = min(state.num_experts, lo + staging_slots)
            width = hi - lo
            self._slice_maps.append(
                prepare_triton(
                    _prefill_slice_map_kernel,
                    (dummy_ids, dummy_mapped, state.expert_to_slot, self.slice_present, self.routes, lo, hi, 256),
                    (triton.cdiv(self.routes, 256), 1, 1),
                    num_warps=4,
                )
            )
            self._slice_compacts.append(
                prepare_triton(
                    _prefill_slice_compact_kernel,
                    (self.slice_present, self.miss_src, self.miss_dst, self.miss_count, lo, width),
                    (1, 1, 1),
                    num_warps=1,
                )
            )

        self._bulk_copies = []
        for pair in staging_bulk_planes:
            row_i32 = pair.row_bytes // 4
            dst_i32 = pair.dst_hbm.view(torch.uint8).reshape(staging_slots, pair.row_bytes).view(torch.int32)
            src_i32 = pair.src_host_dev.view(torch.uint8).reshape(state.num_experts, pair.row_bytes).view(torch.int32)
            block_i32 = 4096 if row_i32 >= 4096 else max(64, triton.next_power_of_2(row_i32))
            self._bulk_copies.append(
                prepare_triton(
                    _copy_expert_plane_kernel,
                    (dst_i32, src_i32, self.miss_src, self.miss_dst, self.miss_count, row_i32, block_i32, copy_ctas),
                    (copy_ctas, 1, 1),
                    num_warps=8,
                )
            )

        self._scalar_copies = []
        s_block = max(16, triton.next_power_of_2(staging_slots))
        s_list = list(staging_scalar_planes)
        for idx in range(0, len(s_list), 4):
            chunk = s_list[idx : idx + 4]
            while len(chunk) < 4:
                chunk.append(chunk[-1])
            c_len = min(4, len(s_list) - idx)
            args = []
            for p in chunk:
                args.append(p.dst_hbm.view(torch.int32).reshape(-1))
                args.append(p.src_host_dev.view(torch.int32).reshape(-1))
            self._scalar_copies.append(
                prepare_triton(
                    _copy_expert_scalars_kernel,
                    (*args, self.miss_src, self.miss_dst, self.miss_count, int(c_len >= 3), int(c_len >= 4), s_block),
                    (1, 1, 1),
                    num_warps=1,
                )
            )

        total_elems = tokens * hidden_size
        self._add_inplace = prepare_triton(
            _add_bf16_inplace_kernel,
            (dummy_out, dummy_out, total_elems, 1024),
            (triton.cdiv(total_elems, 1024), 1, 1),
            num_warps=4,
        )

    def map_resident(self, ids: torch.Tensor, mapped_ids: torch.Tensor) -> torch.Tensor:
        launch_triton(
            self._map_resident.kernel,
            self._map_resident.compiled,
            (ids, mapped_ids, *self._map_resident.args[2:]),
            self._map_resident.grid,
            **self._map_resident.options,
        )
        return mapped_ids

    def stage_slice(self, pass_idx: int, ids: torch.Tensor, slice_ids: torch.Tensor) -> torch.Tensor:
        sm = self._slice_maps[pass_idx]
        launch_triton(sm.kernel, sm.compiled, (ids, slice_ids, *sm.args[2:]), sm.grid, **sm.options)
        self._slice_compacts[pass_idx]()
        for op in self._bulk_copies:
            op()
        for op in self._scalar_copies:
            op()
        return slice_ids

    def accumulate(self, acc: torch.Tensor, delta: torch.Tensor) -> None:
        launch_triton(
            self._add_inplace.kernel,
            self._add_inplace.compiled,
            (acc, delta, *self._add_inplace.args[2:]),
            self._add_inplace.grid,
            **self._add_inplace.options,
        )
