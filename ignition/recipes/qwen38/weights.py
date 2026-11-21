# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned Qwen checkpoint for b12x without dequantizing weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import CheckpointManifest
from ignition.preparation.qwen38 import TEXT_PREFIX

if TYPE_CHECKING:
    import torch
    from b12x.gemm.blockscaled import Weight
    from b12x.moe.fused_moe import PreparedExperts
    from ignition.components.expert_cache import CsfScaleSpec, ExpertCacheState, ExpertPlanePair


@dataclass(frozen=True, slots=True)
class ExpertSlotsConfig:
    """Per-layer HBM slot budgets and optional frequency-ranked initial experts."""

    slots: tuple[int, ...]
    initial_experts: tuple[tuple[int, ...] | None, ...]

    @property
    def total_slots(self) -> int:
        return sum(self.slots)

    @property
    def avg_slots(self) -> float:
        return self.total_slots / len(self.slots)


def load_expert_slots_config(spec: str | int | ExpertSlotsConfig) -> ExpertSlotsConfig:
    """Parse --expert-slots from a uniform integer or a non-uniform JSON profile path."""
    if isinstance(spec, ExpertSlotsConfig):
        return spec
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdecimal()):
        s = int(spec)
        if not 1 <= s <= 512:
            raise ValueError(f"--expert-slots must be in [1, 512], got {s}")
        return ExpertSlotsConfig(slots=(s,) * 48, initial_experts=(None,) * 48)
    path = Path(spec)
    if not path.is_file():
        raise ValueError(f"--expert-slots expects an integer or JSON profile file; not found: {spec!r}")
    data = json.loads(path.read_text())
    slots = tuple(int(v) for v in data["slots"])
    if len(slots) != 48 or any(not 1 <= s <= 512 for s in slots):
        raise ValueError(f"{path}: 'slots' must contain 48 integers in [1, 512]")
    raw_init = data.get("initial_experts")
    if raw_init is None:
        inits = (None,) * 48
    else:
        if len(raw_init) != 48:
            raise ValueError(f"{path}: 'initial_experts' must have 48 layer lists")
        inits = tuple(tuple(int(e) for e in layer_ids[: slots[i]]) for i, layer_ids in enumerate(raw_init))
    return ExpertSlotsConfig(slots=slots, initial_experts=inits)


@dataclass(frozen=True)
class TieredExpertWeights:
    """HBM S-slot PreparedExperts paired with a 512-expert MappedHost tier and SLRU state."""

    hbm_experts: PreparedExperts
    host_planes: dict[str, torch.Tensor]
    host_allocations: tuple[object, ...]
    cache_state: ExpertCacheState
    bulk_planes: tuple[ExpertPlanePair, ...]
    scalar_planes: tuple[ExpertPlanePair, ...]
    num_experts: int = 512
    num_slots: int = 0
    csf_spec: CsfScaleSpec | None = None
    hbm_csf_planes: dict[str, torch.Tensor] | None = None
    csf_exc_planes: dict[str, torch.Tensor] | None = None


def _csf_dst_offset(row, col, rows: int, cols: int, *, a16_layout: bool):
    """Exact destination byte offset in [0, rows*cols) for a4 (F8_128x4) or a16 packed MMA scales."""
    if a16_layout:
        r64 = row % 64
        q = r64 // 8
        inv = (r64 % 8) * 8 + (q // 4) * 4 + ((q % 2) * 2 + ((q % 4) // 2))
        return col * rows + (row // 64) * 64 + inv
    return (
        (row // 128) * (128 * cols)
        + (col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (col % 4)
    )


def _a16_scale_lut(device, *, scale_factor: float = 1.0):
    """Build 256-entry uint8 value table matching b12x _process_nvfp4_packed_scales for BF16 a16."""
    import torch
    from b12x.moe._shared.kernels.w4a16.prepare import _process_nvfp4_packed_scales
    from ignition.preparation.persistent import persistent_copy

    alphabet = torch.arange(256, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    alphabet = alphabet[:, None].expand(256, 4).contiguous().to(torch.bfloat16)
    table = _process_nvfp4_packed_scales(alphabet, scale_factor=scale_factor).view(torch.uint8)[:, 0].contiguous()
    return persistent_copy(table)


def _rebind_prepared_experts_persistent(prepared, shared_w1_scratch, shared_w2_scratch, *, a16: bool):
    """Clone fp4/scalar planes to persistent HBM and rebind block-scale pointers to shared scratch."""
    from dataclasses import replace
    from ignition.preparation.persistent import persistent_copy

    impl = prepared._impl
    slots = int(prepared.num_experts)
    w1_fp4 = persistent_copy(impl.w1_fp4)
    w2_fp4 = persistent_copy(impl.w2_fp4)
    w1_alphas = persistent_copy(impl.w1_alphas)
    w2_alphas = persistent_copy(impl.w2_alphas)
    a1_gscale = persistent_copy(impl.a1_gscale)
    a2_gscale = persistent_copy(impl.a2_gscale)
    w1_bs = shared_w1_scratch[:slots].view(impl.w1_blockscale.dtype).view(impl.w1_blockscale.shape)
    w2_bs = shared_w2_scratch[:slots].view(impl.w2_blockscale.dtype).view(impl.w2_blockscale.shape)

    rep = getattr(impl, "representation", None)
    new_rep = rep
    if rep is not None:
        rv = getattr(rep, "value", rep)
        updates = {}
        if getattr(rv, "w13", None) is not None:
            updates["w13"] = w1_fp4
        if getattr(rv, "w2", None) is not None:
            updates["w2"] = w2_fp4
        if getattr(rv, "w13_scale", None) is not None:
            updates["w13_scale"] = w1_bs
        if getattr(rv, "w2_scale", None) is not None:
            updates["w2_scale"] = w2_bs
        if getattr(rv, "micro_w13_scale", None) is not None:
            updates["micro_w13_scale"] = w1_bs
        if getattr(rv, "micro_w2_scale", None) is not None:
            updates["micro_w2_scale"] = w2_bs
        if getattr(rv, "w13_global_scale", None) is not None:
            updates["w13_global_scale"] = w1_alphas
        if getattr(rv, "w2_global_scale", None) is not None:
            updates["w2_global_scale"] = w2_alphas
        if getattr(rv, "micro_w13_global_scale", None) is not None:
            updates["micro_w13_global_scale"] = w1_alphas
        if getattr(rv, "micro_w2_global_scale", None) is not None:
            updates["micro_w2_global_scale"] = w2_alphas
        new_rv = replace(rv, **updates)
        new_rep = replace(rep, value=new_rv) if hasattr(rep, "value") else new_rv

    new_impl = replace(
        impl,
        w1_fp4=w1_fp4,
        w2_fp4=w2_fp4,
        w1_blockscale=w1_bs,
        w2_blockscale=w2_bs,
        w1_alphas=w1_alphas,
        w2_alphas=w2_alphas,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        representation=new_rep,
    )
    return replace(prepared, _impl=new_impl)


def pack_legacy_nvfp4_csf_for_shard(
    sources: Mapping[str, object],
    layer_prefix: str,
    *,
    first: int,
    last: int,
    hidden_size: int = 2560,
    intermediate_size: int = 640,
    num_experts: int = 512,
    a16_layout: bool = False,
    w1_lut_cpu=None,
    w2_lut_cpu=None,
):
    """Slice legacy nvfp4_csf_fixed & pre-swizzle nvfp4_csf_exceptions for one TP shard.

    Returns (w1_fixed_cpu, w2_fixed_cpu, w1_exc_cpu, w1_starts_cpu, w2_exc_cpu, w2_starts_cpu).
    For a16_layout=True, w13 uses gate/up (w31) order matching b12x's row_rotation=local_I.
    """
    import torch

    local_i = last - first
    c1 = hidden_size // 16
    c2_full = intermediate_size // 16
    c2 = local_i // 16
    if first % 16 != 0 or last % 16 != 0 or (first // 16) % 2 != 0 or c2 % 2 != 0:
        raise ValueError(f"TP shard [{first}, {last}) must be 32-aligned for byte-exact CSF slicing")
    slab_lo, slab_hi = first // 16, last // 16
    col_byte_lo, col_byte_hi = (first // 16) // 2, (last // 16) // 2
    r1, r2 = 2 * local_i, hidden_size
    w1_row_bytes = (r1 // 16) * (16 * (1 + c1 // 2))
    w2_row_bytes = (r2 // 16) * (16 * (1 + c2 // 2))

    w1_fixed = torch.empty((num_experts, w1_row_bytes), dtype=torch.uint8)
    w2_fixed = torch.empty((num_experts, w2_row_bytes), dtype=torch.uint8)
    w1_exc_parts: list[torch.Tensor] = []
    w2_exc_parts: list[torch.Tensor] = []
    w1_starts = [0]
    w2_starts = [0]
    # a4 uses w13 (up, gate); a16 rotates by local_i to w31 (gate, up).
    w1_projs = ("gate_proj", "up_proj") if a16_layout else ("up_proj", "gate_proj")

    for e in range(num_experts):
        p = f"{layer_prefix}.{e}"
        f0 = sources[f"{p}.{w1_projs[0]}.weight_scale.nvfp4_csf_fixed"][slab_lo:slab_hi]
        f1 = sources[f"{p}.{w1_projs[1]}.weight_scale.nvfp4_csf_fixed"][slab_lo:slab_hi]
        w1_fixed[e].copy_(torch.cat((f0, f1), dim=0).reshape(-1))

        # down_proj fixed: (R2//16, 16 + 16*(c2_full//2)) -> slice column bytes [col_byte_lo:col_byte_hi]
        dn = sources[f"{p}.down_proj.weight_scale.nvfp4_csf_fixed"]
        dn_bases = dn[:, :16]
        dn_codes = dn[:, 16:].reshape(r2 // 16, 16, c2_full // 2)[:, :, col_byte_lo:col_byte_hi].reshape(r2 // 16, 16 * (c2 // 2))
        w2_fixed[e].copy_(torch.cat((dn_bases, dn_codes), dim=1).reshape(-1))

        e1_chunks = []
        for half_idx, proj in enumerate(w1_projs):
            raw = sources[f"{p}.{proj}.weight_scale.nvfp4_csf_exceptions"]
            if raw.numel() == 0:
                continue
            words = raw.to(torch.int64)
            idx = words & 0xFFFFFF
            val = (words >> 24) & 0xFF
            sr = idx // c1
            sc = idx - sr * c1
            keep = (sr >= first) & (sr < last)
            if bool(keep.any()):
                r_loc = (sr[keep] - first) + half_idx * local_i
                c_loc = sc[keep]
                v_loc = val[keep]
                if w1_lut_cpu is not None:
                    v_loc = w1_lut_cpu[v_loc].to(torch.int64)
                off = _csf_dst_offset(r_loc, c_loc, r1, c1, a16_layout=a16_layout)
                e1_chunks.append(((v_loc << 24) | off).to(torch.uint32))
        if e1_chunks:
            cat1 = torch.cat(e1_chunks, dim=0)
            w1_exc_parts.append(cat1)
            w1_starts.append(w1_starts[-1] + int(cat1.numel()))
        else:
            w1_starts.append(w1_starts[-1])

        raw2 = sources[f"{p}.down_proj.weight_scale.nvfp4_csf_exceptions"]
        if raw2.numel() > 0:
            words2 = raw2.to(torch.int64)
            idx2 = words2 & 0xFFFFFF
            val2 = (words2 >> 24) & 0xFF
            sr2 = idx2 // c2_full
            sc2 = idx2 - sr2 * c2_full
            c0 = first // 16
            c1_end = last // 16
            keep2 = (sc2 >= c0) & (sc2 < c1_end)
            if bool(keep2.any()):
                r_loc2 = sr2[keep2]
                c_loc2 = sc2[keep2] - c0
                v_loc2 = val2[keep2]
                if w2_lut_cpu is not None:
                    v_loc2 = w2_lut_cpu[v_loc2].to(torch.int64)
                off2 = _csf_dst_offset(r_loc2, c_loc2, r2, c2, a16_layout=a16_layout)
                cat2 = ((v_loc2 << 24) | off2).to(torch.uint32)
                w2_exc_parts.append(cat2)
                w2_starts.append(w2_starts[-1] + int(cat2.numel()))
            else:
                w2_starts.append(w2_starts[-1])
        else:
            w2_starts.append(w2_starts[-1])

    w1_exc = torch.cat(w1_exc_parts, dim=0) if w1_exc_parts else torch.zeros((1,), dtype=torch.uint32)
    w2_exc = torch.cat(w2_exc_parts, dim=0) if w2_exc_parts else torch.zeros((1,), dtype=torch.uint32)
    return (
        w1_fixed,
        w2_fixed,
        w1_exc,
        torch.tensor(w1_starts, dtype=torch.int64),
        w2_exc,
        torch.tensor(w2_starts, dtype=torch.int64),
    )


@contextmanager
def _suspend_persistent_storage():
    """Temporarily divert persistent_empty() to regular PyTorch allocations for bounded staging."""
    from ignition.preparation.persistent import _ACTIVE_STORAGE

    token = _ACTIVE_STORAGE.set(None)
    try:
        yield
    finally:
        _ACTIVE_STORAGE.reset(token)


def _expert_ids(layer: int, expert_ids: Sequence[int] | None) -> tuple[int, ...]:
    if type(layer) is not int or not 0 <= layer < 48:
        raise ValueError("decoder layer must be an integer in [0, 48)")
    ids = tuple(range(512)) if expert_ids is None else tuple(expert_ids)
    if not ids or any(type(i) is not int or not 0 <= i < 512 for i in ids):
        raise ValueError("expert IDs must be a nonempty sequence of integers in [0, 512)")
    if len(set(ids)) != len(ids):
        raise ValueError("expert IDs must be unique")
    return ids


@dataclass(frozen=True, slots=True)
class ExpertMemoryPlan:
    """Exact source sizes and shared scratch; excludes b12x derived metadata."""

    expert_ids: tuple[int, ...]
    packed_bytes: int
    csf_bytes: int
    scalar_bytes: int
    scale_scratch_bytes: int


def expert_memory_plan(
    manifest: CheckpointManifest, layer: int, expert_ids: Sequence[int] | None = None
) -> ExpertMemoryPlan:
    ids = _expert_ids(layer, expert_ids)
    sizes = {"packed": 0, "csf": 0, "scalar": 0}
    for expert in ids:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts.{expert}.{projection}"
            for suffix, kind in (
                ("weight", "packed"), ("weight_scale_2", "scalar"),
                ("input_scale", "scalar"),
                ("weight_scale.nvfp4_csf_fixed", "csf"), ("weight_scale.nvfp4_csf_exceptions", "csf"),
                *((f"weight_scale.csf32_{field}", "csf")
                  for field in ("bases", "codes", "page_bases", "offsets", "exceptions")),
            ):
                tensor = manifest.tensors.get(f"{prefix}.{suffix}")
                if tensor is not None:
                    sizes[kind] += tensor.nbytes
    return ExpertMemoryPlan(ids, sizes["packed"], sizes["csf"], sizes["scalar"], len(ids) * 307200)


def allocate_expert_scale_scratch(
    device: torch.device | str, expert_count: int = 512, *, shard: TensorParallelShard = TensorParallelShard(0, 1)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate scale expansion storage shared by serial decoder layer calls."""
    import torch

    if type(expert_count) is not int or not 1 <= expert_count <= 512:
        raise ValueError("expert_count must be an integer in [1, 512]")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("expert scale scratch requires a CUDA device")
    return (
        torch.empty((expert_count, ((1280 // shard.world_size + 127) // 128) * 128, 160), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((expert_count, 2560, ((40 // shard.world_size + 3) // 4) * 4), dtype=torch.float8_e4m3fn, device=device),
    )


def _expert_scale_vectors(sources: Mapping[str, torch.Tensor], layer: int, ids: tuple[int, ...], *, prefix=None):
    import torch

    w13, w2, a1, a2 = [], [], [], []
    root = prefix or f"{TEXT_PREFIX}.layers.{layer}.mlp.experts"
    for expert in ids:
        prefix = f"{root}.{expert}"
        values = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_scale_2", "input_scale"):
                name = f"{prefix}.{projection}.{suffix}"
                if name not in sources:
                    raise ValueError(f"{name} is required for calibrated NVFP4 activations")
                value = sources[name]
                if value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1:
                    raise ValueError(f"{name} requires a CPU float32 scalar")
                scalar = value.item()
                if not isfinite(scalar) or scalar <= 0:
                    raise ValueError(f"{name} must be finite and positive")
                values[projection, suffix] = scalar
        for suffix in ("weight_scale_2", "input_scale"):
            if values["gate_proj", suffix] != values["up_proj", suffix]:
                raise ValueError(f"{prefix}: gate/up {suffix} differ; fused NVFP4 requires equal scales")
        w13.append(values["up_proj", "weight_scale_2"])
        w2.append(values["down_proj", "weight_scale_2"])
        a1.append(values["up_proj", "input_scale"])
        a2.append(values["down_proj", "input_scale"])
    vectors = (
        torch.tensor(w13, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32),
        torch.tensor(a1, dtype=torch.float32).reciprocal(),
        torch.tensor(a2, dtype=torch.float32).reciprocal(),
    )
    if any(not bool(torch.isfinite(vector).all()) for vector in vectors):
        raise ValueError("expert activation reciprocal scales must remain finite in float32")
    return vectors


class QwenWeightLoader(DirectWeightLoader):
    """Direct checkpoint transfers and explicit, bounded b12x weight preparation.

    ``prefixes`` bounds component qualification; omit it for a complete recipe.
    MTP and PLE table payloads are excluded. Returned tensors own GPU storage
    beyond this context's lifetime. Compressed expert scales use at most one
    layer of GPU staging for partitioning and native-layout preparation.
    """

    def __init__(self, *args, shard=TensorParallelShard(0, 1), force_moe_decode_a16=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shard = shard
        self.force_moe_decode_a16 = force_moe_decode_a16

    def _selected_names(self):
        return {
            name for name, tensor in self.manifest.tensors.items()
            if tensor.component not in ("mtp", "ple_tables")
            and (self.prefixes is None or name.startswith(self.prefixes))
        }

    def load_gdn_projections(self, prefix: str) -> dict[str, Weight]:
        qkv = self.shard.fused_partitions((2048, 2048, 6144))
        value = self.shard.partition(6144)
        heads = self.shard.partition(48)
        return {
            "in_proj_qkvz": self.load_linear_group(
                (f"{prefix}.in_proj_qkv",) * 3 + (f"{prefix}.in_proj_z",),
                row_ranges=(*qkv, value)),
            "in_proj_ba": self.load_linear_group(
                (f"{prefix}.in_proj_b", f"{prefix}.in_proj_a"), row_ranges=(heads, heads)),
            "out_proj": self.load_linear(f"{prefix}.out_proj", columns=value),
        }

    def load_qsa_projections(self, prefix: str) -> dict[str, Weight]:
        kv = tuple(head * 256 for head in self.shard.replicated_partition(2))
        return {
            "qkv_proj": self.load_linear_group(
                tuple(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj")),
                row_ranges=(self.shard.partition(12288), kv, kv)),
            "o_proj": self.load_linear(f"{prefix}.o_proj", columns=self.shard.partition(6144)),
            "indexer.index_qk_proj": self.load_linear(f"{prefix}.indexer.index_qk_proj"),
        }

    def load_experts(
        self, layer: int, *, expert_ids: Sequence[int] | None = None,
        scale_scratch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> PreparedExperts:
        """Prepare routed experts; subset IDs define local ordering in tests.

        Complete model layers require all 512 experts. Reusing ``scale_scratch``
        across layers requires serialized execution on one CUDA stream.
        """
        self._require_open()
        from ignition.preparation.quantization import expert_format

        ids = _expert_ids(layer, expert_ids)
        prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts"
        fmt, mode = expert_format(self.quantization, prefix)
        activation_options = {}
        if self.force_moe_decode_a16:
            if fmt not in ("nvfp4", "nvfp4_csf"):
                raise ValueError("force_moe_decode_a16 requires calibrated NVFP4 routed experts")
            if (640 // self.shard.world_size) % 64:
                raise ValueError("force_moe_decode_a16 requires rank-local expert intermediate width divisible by 64")
            mode = "a16"
            activation_options["a4_prefill"] = True
        if fmt != "nvfp4_csf":
            return self.load_fp4_experts(
                tuple(f"{prefix}.{expert}" for expert in ids), hidden_size=2560, intermediate_size=640,
                format=fmt, activation_mode=mode, intermediate_range=self.shard.partition(640, alignment=32 if fmt == "mxfp4" else 16),
                scale_vectors=_expert_scale_vectors(self._sources, layer, ids) if mode == "a4" or self.force_moe_decode_a16 else None,
                **activation_options)
        return self.load_nvfp4_experts(
            tuple(f"{TEXT_PREFIX}.layers.{layer}.mlp.experts.{expert}" for expert in ids),
            hidden_size=2560, intermediate_size=640,
            intermediate_range=self.shard.partition(640, alignment=16),
            scale_vectors=_expert_scale_vectors(self._sources, layer, ids),
            scale_scratch=scale_scratch, activation_mode=mode, **activation_options,
        )

    def load_tiered_experts(
        self,
        layer: int,
        *,
        num_slots: int,
        initial_experts: Sequence[int] | None = None,
        host_allocator=None,
        batch_size: int = 64,
        scale_compression: str | None = None,
        shared_scale_scratch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> TieredExpertWeights:
        """Load all 512 experts in bounded batches into MappedHost, plus an S-slot HBM pool."""
        import gc
        import torch
        from b12x.moe.fused_moe._binding_ops import materialize_scale_templates
        from b12x.sequence._shared.disk_table import MappedHostAllocation
        from ignition.components.expert_cache import CsfScaleSpec, ExpertCacheState, collect_expert_planes
        from ignition.preparation.persistent import persistent_copy, persistent_empty

        self._require_open()
        if not 1 <= num_slots <= 512:
            raise ValueError(f"tiered expert num_slots must be in [1, 512], got {num_slots}")
        if scale_compression not in (None, "csf"):
            raise ValueError(f"unsupported tiered expert scale_compression {scale_compression!r}")
        use_csf = scale_compression == "csf"
        if use_csf and shared_scale_scratch is None:
            raise ValueError("tiered expert scale_compression='csf' requires shared_scale_scratch")
        init_ids = tuple(range(num_slots)) if initial_experts is None else tuple(int(e) for e in initial_experts)[:num_slots]
        if len(init_ids) < num_slots:
            seen = set(init_ids)
            init_ids = init_ids + tuple(e for e in range(512) if e not in seen)[: num_slots - len(init_ids)]

        host_allocs: list[object] = []
        host_views: dict[str, torch.Tensor] = {}
        host_dev_views: dict[str, torch.Tensor] = {}

        def _alloc_host(tag: str, shape, dtype):
            if host_allocator is not None:
                reg = host_allocator(tag, shape, dtype)
            else:
                reg = MappedHostAllocation(shape, dtype, self.device)
            host_allocs.append(reg)
            return reg

        def _extract_planes(prepared, count: int, *, skip_raw_scales: bool = False) -> dict[str, torch.Tensor]:
            impl = prepared._impl
            found: dict[str, torch.Tensor] = {}
            seen_ptrs: set[int] = set()

            def add(name: str, t):
                if t is None or not isinstance(t, torch.Tensor) or t.ndim < 1 or t.shape[0] != count:
                    return
                if t.data_ptr() in seen_ptrs:
                    return
                seen_ptrs.add(t.data_ptr())
                found[name] = t

            impl_fields = (
                ("w1_fp4", "w2_fp4", "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale")
                if skip_raw_scales
                else (
                    "w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
                    "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale",
                )
            )
            for f in impl_fields:
                add(f, getattr(impl, f, None))
            rep = getattr(impl, "representation", None)
            if rep is not None:
                rv = getattr(rep, "value", rep)
                rep_fields = (
                    ("w13_global_scale", "w2_global_scale", "micro_w13_global_scale", "micro_w2_global_scale")
                    if skip_raw_scales
                    else (
                        "w13_scale", "w2_scale", "w13_sfb", "w2_sfb",
                        "w13_global_scale", "w2_global_scale",
                        "micro_w13_scale", "micro_w2_scale",
                        "micro_w13_global_scale", "micro_w2_global_scale",
                    )
                )
                for f in rep_fields:
                    add(f, getattr(rv, f, None))
            return found

        # 1) Stream all 512 experts in bounded GPU batches into MappedHost (peak HBM ~90 MiB).
        for start in range(0, 512, batch_size):
            end = min(512, start + batch_size)
            batch_ids = tuple(range(start, end))
            with _suspend_persistent_storage():
                batch_ex = self.load_experts(layer, expert_ids=batch_ids)
                materialize_scale_templates(batch_ex)
                planes = _extract_planes(batch_ex, len(batch_ids), skip_raw_scales=use_csf)
                if start == 0:
                    for name, t in planes.items():
                        reg = _alloc_host(f"layer{layer}.experts.{name}", (512, *t.shape[1:]), t.dtype)
                        host_views[name] = reg.host_view
                        host_dev_views[name] = reg.device_view
                for name, t in planes.items():
                    dst_u8 = host_views[name].view(torch.uint8).reshape(512, -1)
                    src_u8 = t.view(torch.uint8).reshape(len(batch_ids), -1).cpu()
                    dst_u8[start:end].copy_(src_u8)
                del planes, batch_ex
            gc.collect()
            torch.cuda.empty_cache()

        csf_spec = None
        hbm_csf_planes = None
        csf_exc_planes = None

        if use_csf:
            a16 = bool(self.force_moe_decode_a16)
            w1_lut = w2_lut = w1_lut_cpu = w2_lut_cpu = None
            if a16:
                from b12x.moe._shared.kernels.w4a16.prepare import _process_nvfp4_packed_global_scale

                w1_lut = _a16_scale_lut(self.device, scale_factor=1.0)
                w2_lut = _a16_scale_lut(self.device, scale_factor=1.0)
                w1_lut_cpu = w1_lut.cpu()
                w2_lut_cpu = w2_lut.cpu()
                # Normalize all 512 experts' a16 global scales to deterministic scale_factor=1.0
                # so batches cannot drift from the shared 256-entry scale LUT.
                all_sv = _expert_scale_vectors(self._sources, layer, tuple(range(512)))
                g1 = _process_nvfp4_packed_global_scale(all_sv[0].to(self.device), a_dtype=torch.bfloat16).to(torch.float32).cpu()
                g2 = _process_nvfp4_packed_global_scale(all_sv[1].to(self.device), a_dtype=torch.bfloat16).to(torch.float32).cpu()
                host_views["w1_alphas"].copy_(g1)
                host_views["w2_alphas"].copy_(g2)

            first, last = self.shard.partition(640, alignment=16)
            local_i = last - first
            w1_fix_cpu, w2_fix_cpu, w1_exc_cpu, w1_st_cpu, w2_exc_cpu, w2_st_cpu = pack_legacy_nvfp4_csf_for_shard(
                self._sources,
                f"{TEXT_PREFIX}.layers.{layer}.mlp.experts",
                first=first,
                last=last,
                hidden_size=2560,
                intermediate_size=640,
                num_experts=512,
                a16_layout=a16,
                w1_lut_cpu=w1_lut_cpu,
                w2_lut_cpu=w2_lut_cpu,
            )
            for cname, cpu_t in (("w1_csf_fixed", w1_fix_cpu), ("w2_csf_fixed", w2_fix_cpu)):
                reg = _alloc_host(f"layer{layer}.experts.{cname}", tuple(cpu_t.shape), cpu_t.dtype)
                reg.host_view.copy_(cpu_t)
                host_views[cname] = reg.host_view
                host_dev_views[cname] = reg.device_view

            # All 48 layers x 512 experts of pre-swizzled CSF exceptions total only ~46 MiB/rank;
            # keeping them resident in HBM eliminates PCIe scatter latency (<0.2 ms / 48L).
            w1_exc_hbm = persistent_copy(w1_exc_cpu.to(self.device))
            w2_exc_hbm = persistent_copy(w2_exc_cpu.to(self.device))
            csf_exc_planes = {"w1_csf_exc": w1_exc_hbm, "w2_csf_exc": w2_exc_hbm}

            w1_starts_hbm = persistent_copy(w1_st_cpu.to(self.device))
            w2_starts_hbm = persistent_copy(w2_st_cpu.to(self.device))
            w1_fix_hbm = persistent_empty((num_slots, w1_fix_cpu.shape[1]), torch.uint8, self.device)
            w2_fix_hbm = persistent_empty((num_slots, w2_fix_cpu.shape[1]), torch.uint8, self.device)
            init_idx_cpu = torch.tensor(init_ids, dtype=torch.int64)
            w1_fix_hbm.copy_(w1_fix_cpu.index_select(0, init_idx_cpu))
            w2_fix_hbm.copy_(w2_fix_cpu.index_select(0, init_idx_cpu))

            hbm_csf_planes = {
                "w1_csf_fixed": w1_fix_hbm,
                "w2_csf_fixed": w2_fix_hbm,
                "w1_csf_starts": w1_starts_hbm,
                "w2_csf_starts": w2_starts_hbm,
                **({"w1_csf_lut": w1_lut, "w2_csf_lut": w2_lut} if a16 else {}),
            }

            with _suspend_persistent_storage():
                tmp_hbm = self.load_experts(layer, expert_ids=init_ids)
                materialize_scale_templates(tmp_hbm)
            hbm_experts = _rebind_prepared_experts_persistent(
                tmp_hbm, shared_scale_scratch[0], shared_scale_scratch[1], a16=a16
            )
            del tmp_hbm
            if a16:
                hbm_experts._impl.w1_alphas.copy_(host_views["w1_alphas"].index_select(0, init_idx_cpu))
                hbm_experts._impl.w2_alphas.copy_(host_views["w2_alphas"].index_select(0, init_idx_cpu))
            gc.collect()
            torch.cuda.empty_cache()

            csf_spec = CsfScaleSpec(
                w1_fixed_hbm=w1_fix_hbm,
                w2_fixed_hbm=w2_fix_hbm,
                w1_exc_dev=w1_exc_hbm,
                w1_starts=w1_starts_hbm,
                w2_exc_dev=w2_exc_hbm,
                w2_starts=w2_starts_hbm,
                w1_scratch=shared_scale_scratch[0],
                w2_scratch=shared_scale_scratch[1],
                r1=2 * local_i,
                c1=2560 // 16,
                r2=2560,
                c2=local_i // 16,
                a16_layout=a16,
                w1_lut=w1_lut,
                w2_lut=w2_lut,
            )
            bulk_planes, scalar_planes = collect_expert_planes(
                hbm_experts, host_dev_views, csf_hbm_planes=hbm_csf_planes
            )
        else:
            # 2) Allocate persistent HBM S-slot pool populated with initial_experts.
            hbm_experts = self.load_experts(layer, expert_ids=init_ids)
            materialize_scale_templates(hbm_experts)
            torch.cuda.empty_cache()
            bulk_planes, scalar_planes = collect_expert_planes(hbm_experts, host_dev_views)

        cache_state = ExpertCacheState(512, num_slots, self.device, initial_experts=init_ids)
        return TieredExpertWeights(
            hbm_experts=hbm_experts,
            host_planes=host_dev_views,
            host_allocations=tuple(host_allocs),
            cache_state=cache_state,
            bulk_planes=bulk_planes,
            scalar_planes=scalar_planes,
            num_experts=512,
            num_slots=num_slots,
            csf_spec=csf_spec,
            hbm_csf_planes=hbm_csf_planes,
            csf_exc_planes=csf_exc_planes,
        )

    def allocate_shared_scale_scratch(self, *, max_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate one shared HBM block-scale expansion scratch reused across all 48 layers + staging."""
        import torch
        from ignition.preparation.persistent import persistent_empty

        first, last = self.shard.partition(640, alignment=16)
        local_i = last - first
        if self.force_moe_decode_a16:
            s1_shape = (max_slots, 2560 // 16, 2 * local_i)
            s2_shape = (max_slots, local_i // 16, 2560)
        else:
            s1_shape = (max_slots, 2 * local_i, 2560 // 16)
            s2_shape = (max_slots, 2560, local_i // 16)
        return (
            persistent_empty(s1_shape, torch.float8_e4m3fn, self.device),
            persistent_empty(s2_shape, torch.float8_e4m3fn, self.device),
        )

    def allocate_prefill_staging_experts(
        self,
        *,
        staging_slots: int = 64,
        shared_scale_scratch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Allocate one shared K-slot persistent HBM PreparedExperts reused by all layers during prefill.

        When ``shared_scale_scratch`` is supplied (CSF mode), returns ``(staging_experts, staging_csf_planes)``
        whose block scales alias ``shared_scale_scratch[:staging_slots]``.
        """
        import gc
        import torch
        from b12x.moe.fused_moe._binding_ops import materialize_scale_templates
        from ignition.preparation.persistent import persistent_empty

        if shared_scale_scratch is not None:
            first, last = self.shard.partition(640, alignment=16)
            local_i = last - first
            r1, c1 = 2 * local_i, 2560 // 16
            r2, c2 = 2560, local_i // 16
            w1_row_bytes = (r1 // 16) * (16 * (1 + c1 // 2))
            w2_row_bytes = (r2 // 16) * (16 * (1 + c2 // 2))
            with _suspend_persistent_storage():
                tmp_stg = self.load_experts(0, expert_ids=tuple(range(staging_slots)))
                materialize_scale_templates(tmp_stg)
            staging = _rebind_prepared_experts_persistent(
                tmp_stg, shared_scale_scratch[0], shared_scale_scratch[1], a16=bool(self.force_moe_decode_a16)
            )
            del tmp_stg
            gc.collect()
            torch.cuda.empty_cache()
            stg_csf = {
                "w1_csf_fixed": persistent_empty((staging_slots, w1_row_bytes), torch.uint8, self.device),
                "w2_csf_fixed": persistent_empty((staging_slots, w2_row_bytes), torch.uint8, self.device),
            }
            return staging, stg_csf

        staging = self.load_experts(0, expert_ids=tuple(range(staging_slots)))
        materialize_scale_templates(staging)
        torch.cuda.empty_cache()
        return staging, None
