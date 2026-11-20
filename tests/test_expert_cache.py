# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 GPU tests for SLRU ExpertCache + MappedHost -> HBM staging + b12x parity."""

from __future__ import annotations

import collections
import time

try:
    import pytest
except ImportError:  # pragma: no cover - allow running standalone without pytest
    class _DummyMark:
        def skipif(self, *args, **kwargs):
            return lambda fn: fn

        def parametrize(self, *args, **kwargs):
            return lambda fn: fn

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()

import torch

from ignition.components.expert_cache import (
    ExpertCacheState,
    ExpertPlanePair,
    PreparedExpertCache,
    PreparedPrefillExpertStreamer,
    collect_expert_planes,
)


def _cuda_available() -> bool:
    return torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="requires CUDA GPU")


class _RefSLRU:
    """Exact CPU reference matching _slru_resolve_kernel semantics."""

    def __init__(self, E: int, S: int, initial_experts=None):
        self.E = E
        self.S = S
        self.prot_cap = max(1, (S * 3) // 4)
        self.e2s = [-1] * E
        self.s2e = [-1] * S
        self.tier = [0] * S
        self.ts = [0] * S
        self.step = 0
        self.stats = [0, 0, 0, 0]
        if initial_experts:
            init = list(initial_experts)[:S]
            for slot, e in enumerate(init):
                self.e2s[e] = slot
                self.s2e[slot] = e
                self.tier[slot] = 1 if slot < self.prot_cap else 0
                self.ts[slot] = len(init) - slot

    def resolve(self, ids_flat: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
        M = len(ids_flat)
        self.step += 1
        base_ts = self.step * (M + 1)
        busy = [False] * self.S
        mapped = [-1] * M
        lookups = hits = promos = 0

        for i, e in enumerate(ids_flat):
            if 0 <= e < self.E:
                lookups += 1
                slot = self.e2s[e]
                if slot >= 0:
                    mapped[i] = slot
                    busy[slot] = True
                    self.ts[slot] = base_ts + i
                    if self.tier[slot] == 0:
                        prot_slots = [s for s in range(self.S) if self.tier[s] == 1 and self.s2e[s] >= 0]
                        if len(prot_slots) >= self.prot_cap:
                            demote = min(prot_slots, key=lambda s: (self.ts[s], s))
                            self.tier[demote] = 0
                        self.tier[slot] = 1
                        promos += 1
                    hits += 1

        misses = []
        for i, e in enumerate(ids_flat):
            if 0 <= e < self.E:
                slot = self.e2s[e]
                if slot >= 0:
                    mapped[i] = slot
                else:
                    avail = [s for s in range(self.S) if not busy[s]]
                    empties = [s for s in avail if self.s2e[s] < 0]
                    if empties:
                        victim = min(empties)
                    else:
                        probs = [s for s in avail if self.tier[s] == 0]
                        if probs:
                            victim = min(probs, key=lambda s: (self.ts[s], s))
                        else:
                            prots = [s for s in avail if self.tier[s] == 1]
                            victim = min(prots, key=lambda s: (self.ts[s], s))
                    old_e = self.s2e[victim]
                    if old_e >= 0:
                        self.e2s[old_e] = -1
                    self.s2e[victim] = e
                    self.e2s[e] = victim
                    self.tier[victim] = 0
                    self.ts[victim] = base_ts + i
                    busy[victim] = True
                    mapped[i] = victim
                    misses.append((e, victim))

        self.stats[0] += lookups
        self.stats[1] += hits
        self.stats[2] += lookups - hits
        self.stats[3] += promos
        return mapped, misses


@pytest.mark.parametrize("num_slots", [64, 144])
def test_slru_matches_reference_and_tp2_lockstep(num_slots: int):
    device = torch.device("cuda", 0)
    E, tokens, top_k = 512, 4, 10
    gen = torch.Generator().manual_seed(1234 + num_slots)
    init = torch.randperm(E, generator=gen)[: num_slots // 2].tolist()

    # Build tiny synthetic bulk + scalar planes to test resolver + copy together.
    row_bytes = 256
    src_bulk = torch.randint(0, 256, (E, row_bytes), dtype=torch.uint8, generator=gen).to(device)
    dst_bulk = torch.zeros((num_slots, row_bytes), dtype=torch.uint8, device=device)
    src_sc = [torch.randn((E,), dtype=torch.float32, generator=gen).to(device) for _ in range(4)]
    dst_sc = [torch.zeros((num_slots,), dtype=torch.float32, device=device) for _ in range(4)]

    bulk = (ExpertPlanePair("w1_fp4", dst_bulk, src_bulk, row_bytes),)
    scalars = tuple(ExpertPlanePair(f"sc{i}", dst_sc[i], src_sc[i], 4) for i in range(4))

    state = ExpertCacheState(E, num_slots, device, initial_experts=init)
    state.copy_initial_weights(bulk, scalars)
    cache = PreparedExpertCache(state, bulk, scalars, tokens=tokens, top_k=top_k)

    # Second independent rank simulating TP2 lockstep.
    dst_bulk_r1 = torch.zeros_like(dst_bulk)
    dst_sc_r1 = [torch.zeros_like(t) for t in dst_sc]
    bulk_r1 = (ExpertPlanePair("w1_fp4", dst_bulk_r1, src_bulk, row_bytes),)
    scalars_r1 = tuple(ExpertPlanePair(f"sc{i}", dst_sc_r1[i], src_sc[i], 4) for i in range(4))
    state_r1 = ExpertCacheState(E, num_slots, device, initial_experts=init)
    state_r1.copy_initial_weights(bulk_r1, scalars_r1)
    cache_r1 = PreparedExpertCache(state_r1, bulk_r1, scalars_r1, tokens=tokens, top_k=top_k)

    ref = _RefSLRU(E, num_slots, initial_experts=init)

    for step in range(250):
        mode = step % 4
        if mode == 0:
            # Uniform random (high miss pressure)
            ids_cpu = torch.randint(0, E, (tokens, top_k), dtype=torch.int32, generator=gen)
        elif mode == 1:
            # Skewed hot set (high hit + promotion pressure)
            ids_cpu = torch.randint(0, num_slots + 24, (tokens, top_k), dtype=torch.int32, generator=gen)
        elif mode == 2:
            # Duplicate experts within the same window + padded (-1) tail row
            base = torch.randint(0, E, (1, top_k), dtype=torch.int32, generator=gen)
            ids_cpu = base.repeat(tokens, 1)
            ids_cpu[-1, :] = -1
        else:
            # Adversarial: pick currently non-resident experts to force max evictions
            cold = [e for e in range(E) if ref.e2s[e] < 0][: tokens * top_k]
            ids_cpu = torch.tensor(cold, dtype=torch.int32).reshape(tokens, top_k)

        ids_gpu = ids_cpu.to(device)
        mapped = cache(ids_gpu)
        mapped_r1 = cache_r1(ids_gpu)
        ref_mapped, _ = ref.resolve(ids_cpu.reshape(-1).tolist())

        assert mapped.reshape(-1).cpu().tolist() == ref_mapped, f"mapped_ids mismatch at step {step}"
        assert mapped_r1.reshape(-1).cpu().tolist() == ref_mapped
        assert state.expert_to_slot.cpu().tolist() == ref.e2s, f"e2s mismatch at step {step}"
        assert state.slot_to_expert.cpu().tolist() == ref.s2e, f"s2e mismatch at step {step}"
        assert state.slot_tier.cpu().tolist() == ref.tier, f"tier mismatch at step {step}"
        assert state.slot_ts.cpu().tolist() == ref.ts, f"ts mismatch at step {step}"

        # Verify copied HBM bytes match canonical experts for every resident slot.
        for s, e in enumerate(ref.s2e):
            if e >= 0:
                assert torch.equal(dst_bulk[s], src_bulk[e]), f"bulk byte mismatch slot={s} exp={e} step={step}"
                for k in range(4):
                    assert dst_sc[k][s].item() == src_sc[k][e].item()

    assert state.stats.cpu().tolist() == ref.stats
    assert state.checksum() == state_r1.checksum()


def test_mapped_host_cuda_graph_replay_and_bandwidth():
    """Verify MappedHost -> HBM staging inside a captured CUDA graph over 500 replays with zero allocation."""
    from b12x.sequence._shared.disk_table import MappedHostAllocation

    device = torch.device("cuda", 0)
    E, S, tokens, top_k = 512, 128, 4, 10
    # Exact TP2 half-expert bulk plane sizes for Qwen3.8-Flash-Next (hidden=2560, local_inter=320)
    shapes = (
        ("w1_fp4", (E, 640, 1280), torch.uint8),
        ("w2_fp4", (E, 2560, 160), torch.uint8),
        ("w1_blockscale", (E, 640, 160), torch.uint8),
        ("w2_blockscale", (E, 2560, 20), torch.uint8),
    )
    allocs = [MappedHostAllocation(shape, dtype, device) for _, shape, dtype in shapes]
    sc_allocs = [MappedHostAllocation((E,), torch.float32, device) for _ in range(4)]
    try:
        # Fill each expert with a deterministic byte signature so any corruption is immediately caught.
        for idx, (alloc, (name, shape, _)) in enumerate(zip(allocs, shapes)):
            pat = ((torch.arange(E, dtype=torch.int64) * 37 + idx * 13 + 7) % 251).to(torch.uint8)
            alloc.host_view.copy_(pat.view(E, 1, 1).expand(shape))
        for idx, alloc in enumerate(sc_allocs):
            alloc.host_view.copy_(torch.arange(E, dtype=torch.float32) * 0.01 + idx)

        dst_bulks = [torch.zeros((S, *shape[1:]), dtype=dtype, device=device) for _, shape, dtype in shapes]
        dst_scs = [torch.zeros((S,), dtype=torch.float32, device=device) for _ in range(4)]

        bulk_pairs = tuple(
            ExpertPlanePair(name, dst, alloc.device_view, dst[0].numel() * dst.element_size())
            for (name, _, _), dst, alloc in zip(shapes, dst_bulks, allocs)
        )
        scalar_pairs = tuple(
            ExpertPlanePair(f"sc{i}", dst, alloc.device_view, 4)
            for i, (dst, alloc) in enumerate(zip(dst_scs, sc_allocs))
        )

        state = ExpertCacheState(E, S, device, initial_experts=list(range(S)))
        state.copy_initial_weights(bulk_pairs, scalar_pairs)
        cache = PreparedExpertCache(state, bulk_pairs, scalar_pairs, tokens=tokens, top_k=top_k, copy_ctas=8)

        ids_buf = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
        mapped_buf = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)

        # Warmup outside capture.
        ids_buf.copy_(torch.arange(tokens * top_k, dtype=torch.int32).reshape(tokens, top_k))
        cache(ids_buf, mapped_buf)
        torch.cuda.synchronize(device)

        # Capture into a CUDA graph.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cache(ids_buf, mapped_buf)
        torch.cuda.synchronize(device)

        mem_before = torch.cuda.memory_allocated(device)
        gen = torch.Generator().manual_seed(999)

        # Replay 500 times with mixed hits/misses; verify zero allocation and byte-exact residency.
        t0 = time.perf_counter()
        for step in range(500):
            if step % 3 == 0:
                nxt = torch.randint(0, E, (tokens, top_k), dtype=torch.int32, generator=gen)
            else:
                nxt = torch.randint(0, S + 32, (tokens, top_k), dtype=torch.int32, generator=gen)
            ids_buf.copy_(nxt, non_blocking=True)
            g.replay()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        mem_after = torch.cuda.memory_allocated(device)
        assert mem_after == mem_before, f"CUDA graph replay allocated {mem_after - mem_before} bytes"

        stats = state.summary_stats()
        miss_bytes = stats["misses"] * sum(p.row_bytes for p in bulk_pairs)
        print(
            f"\nMappedHost graph replay (500 steps): hits={stats['hits']} misses={stats['misses']} "
            f"elapsed={elapsed*1e3:.1f} ms, effective PCIe={miss_bytes / elapsed / 1e9:.2f} GB/s"
        )

        # Verify every resident slot in HBM matches its canonical expert in MappedHost.
        s2e = state.slot_to_expert.cpu().tolist()
        for s, e in enumerate(s2e):
            if e >= 0:
                for idx, (dst, alloc) in enumerate(zip(dst_bulks, allocs)):
                    expected_val = (e * 37 + idx * 13 + 7) % 251
                    got = dst[s]
                    assert int(got.min().item()) == expected_val and int(got.max().item()) == expected_val, (
                        f"slot {s} expert {e} plane {idx} corrupted"
                    )
                for idx, (dst, alloc) in enumerate(zip(dst_scs, sc_allocs)):
                    assert abs(dst[s].item() - (e * 0.01 + idx)) < 1e-5
    finally:
        for a in (*allocs, *sc_allocs):
            a.close()


@pytest.mark.parametrize("num_slots", [64, 128])
@pytest.mark.parametrize("tokens", [1, 4, 8])
def test_single_layer_b12x_parity_against_all_resident(num_slots: int, tokens: int):
    """Compare single-layer b12x NVFP4 MoE output: all 512 resident (ref) vs S-slot SLRU tiered."""
    from b12x.moe import fused_moe
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    E, hidden, local_inter, top_k = 512, 2560, 320, 10
    gen = torch.Generator().manual_seed(2026 + num_slots + tokens)

    # Synthesize realistic NVFP4 packed weights + swizzled scales for all 512 experts.
    w13_fp4 = torch.randint(0, 256, (E, 2 * local_inter, hidden // 2), dtype=torch.uint8, generator=gen, device="cpu")
    w2_fp4 = torch.randint(0, 256, (E, hidden, local_inter // 2), dtype=torch.uint8, generator=gen, device="cpu")
    # Small bounded FP8 scales (powers of 2 around 0.25..1.0) to avoid NaN/Inf in synthetic random FP4 weights.
    s13_shape = (E, ((2 * local_inter + 127) // 128) * 128, hidden // 16)
    s2_shape = (E, hidden, ((local_inter // 16 + 3) // 4) * 4)
    w13_scale = torch.full(s13_shape, 0.25, dtype=torch.float8_e4m3fn)
    w2_scale = torch.full(s2_shape, 0.25, dtype=torch.float8_e4m3fn)
    w13_gscale = torch.full((E,), 1.0, dtype=torch.float32)
    w2_gscale = torch.full((E,), 1.0, dtype=torch.float32)
    a1_gscale = torch.full((E,), 1.0, dtype=torch.float32)
    a2_gscale = torch.full((E,), 1.0, dtype=torch.float32)

    def build_b12x_experts(n: int, w13, s13, w2, s2, w1g, w2g, a1g, a2g):
        plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
            geometry=fused_moe.MoEGeometry(num_experts=n, hidden_size=hidden, intermediate_size=local_inter),
        )
        packed = fused_moe.PackedWeights(
            w13=w13.to(device),
            w2=w2.to(device),
            w13_block_scales=s13.to(device),
            w2_block_scales=s2.to(device),
            w13_global_scales=w1g.to(device),
            w2_global_scales=w2g.to(device),
            input_scale=a1g.to(device),
            intermediate_scale=a2g.to(device),
            immutable_input_scales=True,
        )
        return fused_moe.prepare_weights(plan=plan, weights=packed)

    # 1) All-512 resident reference in HBM.
    ref_experts = build_b12x_experts(
        E, w13_fp4, w13_scale, w2_fp4, w2_scale, w13_gscale, w2_gscale, a1_gscale, a2_gscale
    )
    # 2) Tiered S-slot HBM pool (initially slots 0..S-1).
    tier_experts = build_b12x_experts(
        num_slots,
        w13_fp4[:num_slots], w13_scale[:num_slots],
        w2_fp4[:num_slots], w2_scale[:num_slots],
        w13_gscale[:num_slots], w2_gscale[:num_slots],
        a1_gscale[:num_slots], a2_gscale[:num_slots],
    )

    # 3) Copy all 512 prepared planes from ref_experts into MappedHost (exact b12x layout!).
    from b12x.moe.fused_moe._binding_ops import materialize_scale_templates
    materialize_scale_templates(ref_experts)
    materialize_scale_templates(tier_experts)

    ref_impl = ref_experts._impl
    host_allocs = []
    host_planes: dict[str, torch.Tensor] = {}

    def mirror_to_host(name: str, t: torch.Tensor | None):
        if t is None or not isinstance(t, torch.Tensor) or t.ndim < 1 or t.shape[0] != E:
            return
        if name in host_planes:
            return
        alloc = MappedHostAllocation(tuple(t.shape), t.dtype, device)
        alloc.host_view.copy_(t.cpu())
        host_allocs.append(alloc)
        host_planes[name] = alloc.device_view

    for field in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
                  "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale"):
        mirror_to_host(field, getattr(ref_impl, field, None))
    rep = getattr(ref_impl, "representation", None)
    if rep is not None:
        rv = getattr(rep, "value", rep)
        for field in ("w13_scale", "w2_scale", "w13_sfb", "w2_sfb",
                      "w13_global_scale", "w2_global_scale",
                      "micro_w13_scale", "micro_w2_scale",
                      "micro_w13_global_scale", "micro_w2_global_scale"):
            mirror_to_host(field, getattr(rv, field, None))

    try:
        bulk_pairs, scalar_pairs = collect_expert_planes(tier_experts, host_planes)
        state = ExpertCacheState(E, num_slots, device, initial_experts=list(range(num_slots)))
        state.copy_initial_weights(bulk_pairs, scalar_pairs)
        cache = PreparedExpertCache(state, bulk_pairs, scalar_pairs, tokens=tokens, top_k=top_k)

        x = torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=gen).to(device) * 0.25
        ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
        mapped_ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
        weights = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
        out_ref = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)
        out_tier = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)

        with RecordedPreparationSession(
            device=device,
            checkpoint=f"expert-cache-parity-s{num_slots}-t{tokens}",
            workload_schema="qwen38-expert-cache-test",
            numerical_policy="default",
            autotune_effort="off",
        ) as session:
            ref_op = PreparedRoutedExperts(x, ref_experts, ids, weights, out_ref, session=session, name="ref")
            tier_op = PreparedRoutedExperts(x, tier_experts, mapped_ids, weights, out_tier, session=session, name="tier")
            session.tune()

            # Warmup both paths before graph capture.
            init_ids = torch.arange(tokens * top_k, dtype=torch.int32).reshape(tokens, top_k) % E
            ids.copy_(init_ids)
            ref_op()
            cache(ids, mapped_ids)
            tier_op()
            torch.cuda.synchronize(device)

            # Capture both reference and tiered execution into CUDA graphs.
            g_ref = torch.cuda.CUDAGraph()
            g_tier = torch.cuda.CUDAGraph()
            with session.capture():
                with torch.cuda.graph(g_ref):
                    ref_op()
                with torch.cuda.graph(g_tier):
                    cache(ids, mapped_ids)
                    tier_op()
            session.freeze()
            torch.cuda.synchronize(device)

            # Replay across random, skewed, duplicate, and adversarial (all-miss) routings.
            mem_before = torch.cuda.memory_allocated(device)
            max_abs = 0.0
            for step in range(40):
                x.copy_(torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=gen) * 0.25)
                mode = step % 4
                if mode == 0:
                    step_ids = torch.randint(0, E, (tokens, top_k), dtype=torch.int32, generator=gen)
                elif mode == 1:
                    step_ids = torch.randint(0, num_slots + 16, (tokens, top_k), dtype=torch.int32, generator=gen)
                elif mode == 2:
                    # Adversarial: pick experts currently NOT resident to force 100% misses
                    e2s = state.expert_to_slot.cpu().tolist()
                    cold = [e for e in range(E) if e2s[e] < 0][: tokens * top_k]
                    step_ids = torch.tensor(cold, dtype=torch.int32).reshape(tokens, top_k)
                else:
                    # Deduplicated experts across tokens + padded -1 tail row if tokens > 1
                    row0 = torch.randperm(E, generator=gen)[:top_k].to(torch.int32)
                    step_ids = row0.unsqueeze(0).repeat(tokens, 1)
                    if tokens > 1:
                        step_ids[-1, :] = -1
                # Ensure unique experts per token row (b12x contract)
                for r in range(tokens):
                    if step_ids[r, 0].item() >= 0 and len(set(step_ids[r].tolist())) < top_k:
                        step_ids[r] = torch.randperm(E, generator=gen)[:top_k].to(torch.int32)

                ids.copy_(step_ids)
                mem_step = torch.cuda.memory_allocated(device)
                g_ref.replay()
                g_tier.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == mem_step, "graph replay allocated GPU memory"

                diff = (out_tier.float() - out_ref.float()).abs()
                step_max = float(diff.max().item())
                ref_scale = max(float(out_ref.float().abs().max().item()), 1e-3)
                del diff
                max_abs = max(max_abs, step_max)
                assert torch.isfinite(out_tier).all(), f"non-finite output at step {step}"
                # Allow small BF16 reduction-order difference since slot order != canonical ID order.
                assert step_max <= 0.05 * ref_scale + 1e-2, (
                    f"parity mismatch step={step} mode={mode}: max_abs={step_max:.4f}, ref_max={ref_scale:.4f}"
                )
            assert torch.cuda.memory_allocated(device) == mem_before
            print(f"\nb12x parity OK (S={num_slots}, T={tokens}): max_abs={max_abs:.5f}, stats={state.summary_stats()}")
    finally:
        for a in host_allocs:
            a.close()


@pytest.mark.parametrize("tokens", [256, 2048, 4096])
def test_prefill_streaming_b12x_parity_and_speed(tokens: int):
    """Phase 2 gate: multi-pass K=64 prefill streaming parity vs all-512 reference + speed."""
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._binding_ops import materialize_scale_templates
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    E, S, K, hidden, local_inter, top_k = 512, 144, 64, 2560, 320, 10
    gen = torch.Generator().manual_seed(7777 + tokens)

    w13_fp4 = torch.randint(0, 256, (E, 2 * local_inter, hidden // 2), dtype=torch.uint8, generator=gen, device="cpu")
    w2_fp4 = torch.randint(0, 256, (E, hidden, local_inter // 2), dtype=torch.uint8, generator=gen, device="cpu")
    s13_shape = (E, ((2 * local_inter + 127) // 128) * 128, hidden // 16)
    s2_shape = (E, hidden, ((local_inter // 16 + 3) // 4) * 4)
    w13_scale = torch.full(s13_shape, 0.25, dtype=torch.float8_e4m3fn)
    w2_scale = torch.full(s2_shape, 0.25, dtype=torch.float8_e4m3fn)
    ones = torch.full((E,), 1.0, dtype=torch.float32)

    def build_b12x(n: int, w13, s13, w2, s2, sc):
        plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
            geometry=fused_moe.MoEGeometry(num_experts=n, hidden_size=hidden, intermediate_size=local_inter),
        )
        packed = fused_moe.PackedWeights(
            w13=w13.to(device), w2=w2.to(device),
            w13_block_scales=s13.to(device), w2_block_scales=s2.to(device),
            w13_global_scales=sc.to(device), w2_global_scales=sc.to(device),
            input_scale=sc.to(device), intermediate_scale=sc.to(device),
            immutable_input_scales=True,
        )
        ex = fused_moe.prepare_weights(plan=plan, weights=packed)
        materialize_scale_templates(ex)
        return ex

    ref_experts = build_b12x(E, w13_fp4, w13_scale, w2_fp4, w2_scale, ones)
    res_experts = build_b12x(S, w13_fp4[:S], w13_scale[:S], w2_fp4[:S], w2_scale[:S], ones[:S])
    stg_experts = build_b12x(K, w13_fp4[:K], w13_scale[:K], w2_fp4[:K], w2_scale[:K], ones[:K])

    host_allocs = []
    host_planes: dict[str, torch.Tensor] = {}
    ref_impl = ref_experts._impl
    for field in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
                  "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale"):
        t = getattr(ref_impl, field, None)
        if t is not None and t.ndim >= 1 and t.shape[0] == E:
            alloc = MappedHostAllocation(tuple(t.shape), t.dtype, device)
            alloc.host_view.copy_(t.cpu())
            host_allocs.append(alloc)
            host_planes[field] = alloc.device_view

    try:
        res_bulk, res_sc = collect_expert_planes(res_experts, host_planes)
        stg_bulk, stg_sc = collect_expert_planes(stg_experts, host_planes)

        # Put every 3rd expert (144 non-contiguous experts) into resident slots to test arbitrary mapping.
        init_res = [i * 3 for i in range(S)]
        state = ExpertCacheState(E, S, device, initial_experts=init_res)
        state.copy_initial_weights(res_bulk, res_sc)
        cksum_before = state.checksum()

        streamer = PreparedPrefillExpertStreamer(
            state, stg_bulk, stg_sc,
            tokens=tokens, top_k=top_k, hidden_size=hidden, staging_slots=K, copy_ctas=8,
        )

        x = torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=gen).to(device) * 0.25
        ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
        mapped_ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
        slice_ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
        weights = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
        out_ref = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)
        out_stream = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)
        out_slice = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)

        with RecordedPreparationSession(
            device=device,
            checkpoint=f"prefill-stream-t{tokens}",
            workload_schema="qwen38-prefill-stream-test",
            numerical_policy="default",
            autotune_effort="off",
        ) as session:
            ref_op = PreparedRoutedExperts(x, ref_experts, ids, weights, out_ref, session=session, name="ref")
            res_op = PreparedRoutedExperts(x, res_experts, mapped_ids, weights, out_stream, session=session, name="res")
            stg_op = PreparedRoutedExperts(x, stg_experts, slice_ids, weights, out_slice, session=session, name="stg")
            session.tune()

            def run_stream():
                streamer.map_resident(ids, mapped_ids)
                res_op()
                for p in range(streamer.num_passes):
                    streamer.stage_slice(p, ids, slice_ids)
                    stg_op()
                    streamer.accumulate(out_stream, out_slice)

            # Unique top_k experts per row across all 512 experts
            step_ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(tokens)]).to(torch.int32)
            ids.copy_(step_ids)

            # Warmup & capture
            ref_op()
            run_stream()
            torch.cuda.synchronize(device)

            g_ref = torch.cuda.CUDAGraph()
            g_stream = torch.cuda.CUDAGraph()
            with session.capture():
                with torch.cuda.graph(g_ref):
                    ref_op()
                with torch.cuda.graph(g_stream):
                    run_stream()
            session.freeze()
            torch.cuda.synchronize(device)

            g_ref.replay()
            g_stream.replay()
            torch.cuda.synchronize(device)

            # Verify residency was completely untouched
            assert state.checksum() == cksum_before, "prefill mutated decode cache residency!"
            assert torch.isfinite(out_stream).all(), "non-finite prefill streaming output"
            diff = (out_stream.float() - out_ref.float()).abs()
            max_abs = float(diff.max().item())
            ref_scale = max(float(out_ref.float().abs().max().item()), 1e-3)
            rel_l2 = float(diff.norm().item() / max(out_ref.float().norm().item(), 1e-6))
            assert rel_l2 <= 0.02, f"prefill parity mismatch T={tokens}: rel_l2={rel_l2:.4f}, max_abs={max_abs:.4f}"

            # Benchmark steady-state single-layer prefill streaming (5 replays)
            t0 = time.perf_counter()
            for _ in range(5):
                g_stream.replay()
            torch.cuda.synchronize(device)
            ms_layer = (time.perf_counter() - t0) / 5 * 1e3
            est_48_toks = tokens / (48 * ms_layer / 1e3)
            print(
                f"\nprefill streaming OK (T={tokens}, S={S}, K={K}): rel_l2={rel_l2:.5f}, "
                f"max_abs={max_abs:.2f}/{ref_scale:.0f}, {ms_layer:.2f} ms/layer -> ~{est_48_toks:.0f} tok/s (48L PCIe-x16 local)"
            )
    finally:
        for a in host_allocs:
            a.close()


def test_compiled_prepared_qwen_moe_and_mapped_embed():
    """Verify torch.compile (compile_forward) + CUDA graph capture for tiered PreparedQwenMoe (decode & prefill) and MappedHost PreparedEmbedding."""
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._binding_ops import materialize_scale_templates
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from ignition.components.embedding import PreparedEmbedding
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.qwen38.moe import PreparedQwenMoe, QwenMoeWeights, SharedExpertWeights
    from ignition.recipes.qwen38.weights import TieredExpertWeights

    device = torch.device("cuda", 0)
    E, S, K, hidden, local_inter = 512, 128, 64, 2560, 320
    gen = torch.Generator().manual_seed(4242)

    w13_fp4 = torch.randint(0, 256, (E, 2 * local_inter, hidden // 2), dtype=torch.uint8, generator=gen)
    w2_fp4 = torch.randint(0, 256, (E, hidden, local_inter // 2), dtype=torch.uint8, generator=gen)
    s13_shape = (E, ((2 * local_inter + 127) // 128) * 128, hidden // 16)
    s2_shape = (E, hidden, ((local_inter // 16 + 3) // 4) * 4)
    w13_scale = torch.full(s13_shape, 0.25, dtype=torch.float8_e4m3fn)
    w2_scale = torch.full(s2_shape, 0.25, dtype=torch.float8_e4m3fn)
    ones = torch.full((E,), 1.0, dtype=torch.float32)

    def build_b12x(n: int):
        plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
            geometry=fused_moe.MoEGeometry(num_experts=n, hidden_size=hidden, intermediate_size=local_inter),
        )
        packed = fused_moe.PackedWeights(
            w13=w13_fp4[:n].to(device), w2=w2_fp4[:n].to(device),
            w13_block_scales=w13_scale[:n].to(device), w2_block_scales=w2_scale[:n].to(device),
            w13_global_scales=ones[:n].to(device), w2_global_scales=ones[:n].to(device),
            input_scale=ones[:n].to(device), intermediate_scale=ones[:n].to(device),
            immutable_input_scales=True,
        )
        ex = fused_moe.prepare_weights(plan=plan, weights=packed)
        materialize_scale_templates(ex)
        return ex

    hbm_ex = build_b12x(S)
    stg_ex = build_b12x(K)
    host_allocs = []
    host_planes: dict[str, torch.Tensor] = {}
    for field in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
                  "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale"):
        t = getattr(hbm_ex._impl, field)
        alloc = MappedHostAllocation((E, *t.shape[1:]), t.dtype, device)
        alloc.host_view.zero_()
        alloc.host_view[:S].view(torch.uint8).copy_(t.view(torch.uint8).cpu())
        host_allocs.append(alloc)
        host_planes[field] = alloc.device_view

    # Also test MappedHost embed_tokens table with PreparedEmbedding
    emb_alloc = MappedHostAllocation((1024, hidden), torch.bfloat16, device)
    emb_alloc.host_view.copy_(torch.randn((1024, hidden), dtype=torch.bfloat16, generator=gen))
    host_allocs.append(emb_alloc)

    try:
        bulk_p, sc_p = collect_expert_planes(hbm_ex, host_planes)
        state = ExpertCacheState(E, S, device, initial_experts=list(range(S)))
        tiered = TieredExpertWeights(
            hbm_experts=hbm_ex, host_planes=host_planes, host_allocations=tuple(host_allocs),
            cache_state=state, bulk_planes=bulk_p, scalar_planes=sc_p, num_experts=E, num_slots=S,
        )
        shared = SharedExpertWeights(
            gate_up=torch.randn((2 * local_inter, hidden), dtype=torch.bfloat16, device=device) * 0.02,
            down=torch.randn((hidden, local_inter), dtype=torch.bfloat16, device=device) * 0.02,
            gate=torch.randn((1, hidden), dtype=torch.bfloat16, device=device) * 0.02,
        )
        router_w = torch.randn((E, hidden), dtype=torch.bfloat16, device=device) * 0.02
        moe_w = QwenMoeWeights(router=router_w, experts=hbm_ex, shared=shared, tiered=tiered)

        x_dec = torch.randn((4, hidden), dtype=torch.bfloat16, device=device) * 0.25
        x_pre = torch.randn((256, hidden), dtype=torch.bfloat16, device=device) * 0.25
        tok_ids = torch.tensor([1, 5, 42, 99], dtype=torch.int32, device=device)
        emb_out = torch.empty((4, hidden), dtype=torch.bfloat16, device=device)
        num_rows = torch.tensor([4], dtype=torch.int32, device=device)

        with RecordedPreparationSession(
            device=device,
            checkpoint="compiled-tiered-qwen-moe",
            workload_schema="qwen38-compiled-test",
            numerical_policy="default",
            autotune_effort="off",
        ) as session:
            emb_op = PreparedEmbedding(emb_alloc.device_view, tok_ids, emb_out, num_rows=num_rows, session=session, name="emb")
            dec_moe = PreparedQwenMoe(x_dec, moe_w, session=session, name="dec.moe", prefill=False)
            pre_moe = PreparedQwenMoe(x_pre, moe_w, session=session, name="pre.moe", prefill=True, prefill_staging_experts=stg_ex)
            emb_op.functional(session=session, dynamic_inputs=True)
            dec_moe.functional(session=session)
            pre_moe.functional(session=session)
            session.tune()

            out_dec = torch.empty_like(x_dec)
            out_pre = torch.empty_like(x_pre)

            def fn_dec():
                e_out = emb_op(tok_ids, num_rows=num_rows)
                out_dec.copy_(dec_moe(e_out))

            def fn_pre():
                out_pre.copy_(pre_moe(x_pre))

            c_dec = compile_forward(fn_dec)
            c_pre = compile_forward(fn_pre)
            c_dec()
            c_pre()
            torch.cuda.synchronize(device)

            g_dec, g_pre = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with session.capture():
                with torch.cuda.graph(g_dec):
                    c_dec()
                with torch.cuda.graph(g_pre):
                    c_pre()
            session.freeze()

            for _ in range(10):
                g_dec.replay()
            g_pre.replay()
            torch.cuda.synchronize(device)
            assert torch.isfinite(out_dec).all() and torch.isfinite(out_pre).all()
            print(f"\nCOMPILED PreparedQwenMoe (decode+prefill) + MappedHost PreparedEmbedding OK! stats={state.summary_stats()}")
    finally:
        for a in host_allocs:
            a.close()


def _make_random_nvfp4_experts(num_experts: int, hidden: int, local_inter: int, device: torch.device, gen: torch.Generator):
    from b12x.moe import fused_moe

    w13_fp4 = torch.randint(0, 256, (num_experts, 2 * local_inter, hidden // 2), dtype=torch.uint8, generator=gen, device="cpu")
    w2_fp4 = torch.randint(0, 256, (num_experts, hidden, local_inter // 2), dtype=torch.uint8, generator=gen, device="cpu")
    s13_shape = (num_experts, ((2 * local_inter + 127) // 128) * 128, hidden // 16)
    s2_shape = (num_experts, hidden, ((local_inter // 16 + 3) // 4) * 4)
    w13_scale = torch.full(s13_shape, 0.25, dtype=torch.float8_e4m3fn)
    w2_scale = torch.full(s2_shape, 0.25, dtype=torch.float8_e4m3fn)
    ones = torch.full((num_experts,), 1.0, dtype=torch.float32)
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=num_experts, hidden_size=hidden, intermediate_size=local_inter),
    )
    packed = fused_moe.PackedWeights(
        w13=w13_fp4.to(device), w2=w2_fp4.to(device),
        w13_block_scales=w13_scale.to(device), w2_block_scales=w2_scale.to(device),
        w13_global_scales=ones.to(device), w2_global_scales=ones.to(device),
        input_scale=ones.to(device), intermediate_scale=ones.to(device),
        immutable_input_scales=True,
    )
    return fused_moe.prepare_weights(plan=plan, weights=packed)


def test_slru_overflow_staging_parity():
    """Verify C2 decode overflow path (S < tokens*top_k <= S+64) is exact for both 0 and >0 overflows."""
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._binding_ops import materialize_scale_templates
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    E, S, K, tokens, top_k, hidden, local_inter = 512, 64, 64, 10, 10, 2560, 320
    gen = torch.Generator(device="cpu").manual_seed(909)
    ref_experts = _make_random_nvfp4_experts(E, hidden, local_inter, device, gen)
    tier_experts = _make_random_nvfp4_experts(S, hidden, local_inter, device, torch.Generator(device="cpu").manual_seed(910))
    stg_experts = _make_random_nvfp4_experts(K, hidden, local_inter, device, torch.Generator(device="cpu").manual_seed(911))
    materialize_scale_templates(ref_experts)
    materialize_scale_templates(tier_experts)
    materialize_scale_templates(stg_experts)

    ref_impl = ref_experts._impl
    host_allocs, host_planes = [], {}
    for field in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
                  "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale"):
        t = getattr(ref_impl, field, None)
        if t is not None and isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == E:
            a = MappedHostAllocation(tuple(t.shape), t.dtype, device)
            a.host_view.view(torch.uint8).reshape(-1).copy_(t.view(torch.uint8).reshape(-1).cpu())
            host_allocs.append(a)
            host_planes[field] = a.device_view
    try:
        bulk_p, sc_p = collect_expert_planes(tier_experts, host_planes)
        stg_bulk, stg_sc = collect_expert_planes(stg_experts, host_planes)
        state = ExpertCacheState(E, S, device, initial_experts=list(range(S)))
        state.copy_initial_weights(bulk_p, sc_p)
        cache = PreparedExpertCache(
            state, bulk_p, sc_p, tokens=tokens, top_k=top_k,
            staging_bulk_planes=stg_bulk, staging_scalar_planes=stg_sc, staging_slots=K,
        )
        assert cache.has_overflow

        x = torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=gen).to(device) * 0.25
        ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
        mapped_ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
        ovf_ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
        w = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
        out_ref = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)
        out_main = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)
        out_ovf = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)

        with RecordedPreparationSession(
            device=device, checkpoint="overflow-parity", workload_schema="qwen38-ovf-test",
            numerical_policy="default", autotune_effort="off",
        ) as session:
            ref_op = PreparedRoutedExperts(x, ref_experts, ids, w, out_ref, session=session, name="ref")
            main_op = PreparedRoutedExperts(x, tier_experts, mapped_ids, w, out_main, session=session, name="main")
            ovf_op = PreparedRoutedExperts(x, stg_experts, ovf_ids, w, out_ovf, session=session, name="ovf")
            session.tune()

            # Case 1: normal C2 step (30 unique experts <= S=64 -> ovf_count == 0)
            normal_ids = torch.arange(30, dtype=torch.int32).repeat(4)[: tokens * top_k].reshape(tokens, top_k)
            ids.copy_(normal_ids)
            ref_op()
            cache(ids, mapped_ids, ovf_ids)
            main_op()
            ovf_op()
            torch.cuda.synchronize(device)
            assert cache.ovf_count.item() == 0
            combined = out_main + out_ovf
            diff1 = (combined.float() - out_ref.float()).abs().max().item()
            assert diff1 <= 0.05 * max(out_ref.float().abs().max().item(), 1e-3) + 1e-2, f"normal diff={diff1}"

            # Case 2: worst-case overflow step (100 distinct experts > S=64 -> ovf_count == 36)
            ovf_step_ids = torch.arange(100, 200, dtype=torch.int32).reshape(tokens, top_k)
            ids.copy_(ovf_step_ids)
            ref_op()
            cache(ids, mapped_ids, ovf_ids)
            main_op()
            ovf_op()
            torch.cuda.synchronize(device)
            assert cache.ovf_count.item() == 36, f"expected 36 overflow experts, got {cache.ovf_count.item()}"
            combined2 = out_main + out_ovf
            diff2 = (combined2.float() - out_ref.float()).abs().max().item()
            assert diff2 <= 0.05 * max(out_ref.float().abs().max().item(), 1e-3) + 1e-2, f"overflow diff={diff2}"
            print(f"\nSLRU overflow parity OK (S=64, M=100): normal_diff={diff1:.5f}, overflow36_diff={diff2:.5f}")
    finally:
        for a in host_allocs:
            a.close()


def test_declare_weights_and_reset_and_vision_mapped_host():
    """Verify declare_weights excludes runtime SLRU/host_allocs, reset restores initial state, and W4A16/BF16 MappedHost."""
    from dataclasses import fields as dc_fields, replace as dc_replace
    from types import SimpleNamespace
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from ignition.components.linear import QuantizedProjection, UnquantizedProjection
    from ignition.components.vocabulary import nvfp4_draft_head
    import ignition.components.weights  # ensure declare_container_fields registered
    from ignition.preparation.rebuild_manifest import declare_weights, declared_views, weight_payload_declaration
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.qwen38.decoder import DecoderLayerWeights, ModelWeights
    from ignition.recipes.qwen38.moe import QwenMoeWeights, SharedExpertWeights
    from ignition.recipes.qwen38.vision import VisionWeights
    from ignition.recipes.qwen38.weights import TieredExpertWeights

    device = torch.device("cuda:0")
    E, S, hidden, local_inter = 512, 64, 2560, 320
    gen = torch.Generator(device="cpu").manual_seed(777)
    hbm_ex = _make_random_nvfp4_experts(S, hidden, local_inter, device, gen)

    host_allocs, host_planes = [], {}
    impl = hbm_ex._impl
    for field in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
                  "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale"):
        t = getattr(impl, field, None)
        if t is not None and isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == S:
            a = MappedHostAllocation((E, *t.shape[1:]), t.dtype, device)
            a.host_view.view(torch.uint8).reshape(E, -1)[:S].copy_(t.view(torch.uint8).reshape(S, -1).cpu())
            host_allocs.append(a)
            host_planes[field] = a.device_view

    # Also test W4A16 NVFP4 and BF16 UnquantizedProjection on MappedHost (vision_in_ram).
    w_bf16 = torch.randn(256, 256, dtype=torch.bfloat16, device=device) * 0.05
    r_bf16 = MappedHostAllocation((256, 256), torch.bfloat16, device)
    r_bf16.host_view.copy_(w_bf16.cpu())
    host_allocs.append(r_bf16)

    w_fp4 = nvfp4_draft_head(w_bf16)
    updates = {}
    for f in dc_fields(w_fp4):
        v = getattr(w_fp4, f.name)
        if isinstance(v, torch.Tensor):
            if v.is_contiguous():
                r = MappedHostAllocation(tuple(v.shape), v.dtype, device)
                r.host_view.view(torch.uint8).reshape(-1).copy_(v.view(torch.uint8).reshape(-1).cpu())
                host_allocs.append(r)
                updates[f.name] = r.device_view
            else:
                shape, stride = tuple(v.shape), tuple(v.stride())
                span = 1 + sum((sz - 1) * st for sz, st in zip(shape, stride, strict=True) if sz > 0)
                flat_src = torch.empty(0, dtype=v.dtype, device=device).set_(
                    v.untyped_storage(), v.storage_offset(), (span,), (1,)
                )
                r = MappedHostAllocation((span,), v.dtype, device)
                r.host_view.view(torch.uint8).copy_(flat_src.view(torch.uint8).cpu())
                host_allocs.append(r)
                updates[f.name] = r.device_view.as_strided(shape, stride)
    w_fp4_mh = dc_replace(w_fp4, **updates)

    try:
        bulk_p, sc_p = collect_expert_planes(hbm_ex, host_planes)
        init_ids = list(range(10, 10 + S))
        state = ExpertCacheState(E, S, device, initial_experts=init_ids)
        state.copy_initial_weights(bulk_p, sc_p)
        init_cksum = state.checksum()

        tiered = TieredExpertWeights(
            hbm_experts=hbm_ex, host_planes=host_planes, host_allocations=tuple(host_allocs),
            cache_state=state, bulk_planes=bulk_p, scalar_planes=sc_p, num_experts=E, num_slots=S,
        )
        shared = SharedExpertWeights(
            gate_up=torch.zeros((2 * local_inter, hidden), dtype=torch.bfloat16, device=device),
            down=torch.zeros((hidden, local_inter), dtype=torch.bfloat16, device=device),
            gate=torch.zeros((1, hidden), dtype=torch.bfloat16, device=device),
        )
        moe_w = QwenMoeWeights(
            router=torch.zeros((E, hidden), dtype=torch.bfloat16, device=device),
            experts=hbm_ex, shared=shared, tiered=tiered,
        )
        vision_w = VisionWeights(
            parameters={"patch_embed.proj.weight": r_bf16.device_view},
            projections={"blocks.0.attn.qkv": w_fp4_mh},
            host_allocations=(r_bf16,),
        )
        tree = torch.nn.Module()
        tree.model = torch.nn.Module()
        tree.model.language_model = torch.nn.Module()
        tree.model.language_model.embed_tokens = torch.nn.Embedding(4, 4, device=device, dtype=torch.bfloat16)
        tree.model.language_model.hyper_connection_mixer = torch.nn.Module()
        tree.model.language_model.layers = torch.nn.ModuleList([])
        tree.lm_head = torch.nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)

        mw = ModelWeights(
            tree=tree,
            layers=(DecoderLayerWeights(attention={}, moe=moe_w),),
            vision=vision_w,
            allocated_bytes=0,
            prefill_staging_experts=hbm_ex,
        )
        decl = declare_weights(
            {"weights": mw},
            components={
                "tree.model.language_model.embed_tokens": "embedding",
                "tree.model.language_model.hyper_connection_mixer": "hyper-connection",
                "tree.lm_head": "head",
                "vision": "vision",
            },
            layers=("layers", "tree.model.language_model.layers"),
        )
        views = list(declared_views(decl))
        names = [n for n, *_ in views]
        # Verify host_planes ARE in layer-000, and NO CPU tensors, NO cache_state, NO host_allocations!
        assert any("tiered.host_planes['w1_fp4']" in n for n in names), f"missing host_planes in {names}"
        assert not any("cache_state" in n for n in names), f"cache_state leaked into declared_views: {names}"
        assert not any("host_allocations" in n for n in names), f"host_allocations leaked: {names}"
        assert not any("bulk_planes" in n or "scalar_planes" in n for n in names), f"duplicate planes leaked: {names}"
        for n, _owner, t, _grp, _st in views:
            assert t.device.type == "cuda", f"CPU tensor leaked into declared_views: {n} ({t.device})"
        payload = weight_payload_declaration(decl)
        assert len(payload["views"]) == len(views)

        # Verify reset_expert_caches restores exact initial checksum & zeroed stats after dirtying cache.
        cache = PreparedExpertCache(state, bulk_p, sc_p, tokens=2, top_k=10)
        cache(torch.arange(200, 220, dtype=torch.int32, device=device).reshape(2, 10))
        torch.cuda.synchronize(device)
        assert state.checksum() != init_cksum
        assert state.summary_stats()["lookups"] == 20
        mw.reset_expert_caches()
        assert state.checksum() == init_cksum
        assert state.summary_stats() == {"lookups": 0, "hits": 0, "misses": 0, "promotions": 0}

        # Verify MappedHost BF16 UnquantizedProjection and W4A16 NVFP4 QuantizedProjection parity.
        x = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
        y = torch.zeros(4, 256, dtype=torch.bfloat16, device=device)
        with RecordedPreparationSession(
            device=device, checkpoint="vision-mh-test", workload_schema="qwen38-vision-mh",
            numerical_policy="default", autotune_effort="off",
        ) as session:
            p_uq = UnquantizedProjection(x, r_bf16.device_view, y.clone(), session=session, name="uq_mh")
            p_hbm = QuantizedProjection(x, w_fp4, y.clone(), session=session, name="fp4_hbm", activation_mode="a16")
            p_mh = QuantizedProjection(x, w_fp4_mh, y.clone(), session=session, name="fp4_mh", activation_mode="a16")
            p_uq.functional(session=session)
            p_hbm.functional(session=session)
            p_mh.functional(session=session)
            session.tune()
            o_uq = p_uq(x)
            o_hbm = p_hbm(x)
            o_mh = p_mh(x)
            torch.cuda.synchronize(device)
            uq_err = (o_uq - (x @ w_bf16.t())).abs().max().item()
            fp4_err = (o_hbm - o_mh).abs().max().item()
            assert uq_err < 1e-2, f"UnquantizedProjection MappedHost err={uq_err}"
            assert fp4_err == 0.0, f"NVFP4 W4A16 MappedHost err={fp4_err}"
        print(f"\ndeclare_weights ({len(views)} views) + reset_expert_caches + Vision MappedHost (uq={uq_err:.5f}, nvfp4={fp4_err:.5f}) OK!")
    finally:
        for a in host_allocs:
            a.close()
