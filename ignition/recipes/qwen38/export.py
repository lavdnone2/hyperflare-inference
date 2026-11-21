# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Disposable Qwen storage cleared before publishing a runtime core."""

from ignition.preparation.disposable import storage_ranges


def disposable_ranges(*, cache, staging_experts=None, staging_csf=None, shared_scale_scratch=None):
    values = []
    for state in cache.values():
        values.extend((state.main_k_cache, state.main_v_cache, state.compressed_k_cache, state.raw_k_ring))
    if staging_experts is not None:
        impl = getattr(staging_experts, "_impl", None)
        if impl is not None:
            for field in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
                          "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale"):
                t = getattr(impl, field, None)
                if t is not None:
                    values.append(t)
    if staging_csf is not None:
        values.extend(staging_csf.values())
    if shared_scale_scratch is not None:
        values.extend(shared_scale_scratch)
    return storage_ranges(values)
