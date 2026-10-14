# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native cache and non-speculative decode contracts for Qwen3.5 XLite."""

from typing import Any

import torch


def cache_support_reason(
    conv: torch.Tensor,
    ssm: torch.Tensor,
    *,
    channels: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> str | None:
    """Validate the actual native views, including page padding and offsets."""
    if width not in (2, 3, 4) or channels <= 0 or heads <= 0 or channels % 64 or key_dim % 16 or value_dim % 64:
        return "unsupported GDN dimensions"
    if not 0 < key_dim <= 128 or not 0 < value_dim <= 128:
        return "unsupported GDN head dimensions"
    if conv.device != device or ssm.device != device:
        return "GDN cache is not on the runner device"
    if conv.ndim != 3 or tuple(conv.shape[1:]) != (width - 1, channels):
        return "unsupported conv layout (requires native SD)"
    if conv.stride()[1:] != (channels, 1) or conv.stride(0) < (width - 1) * channels:
        return "unsupported conv strides"
    if ssm.ndim != 4 or tuple(ssm.shape[1:]) != (heads, value_dim, key_dim):
        return "unsupported native SSM layout (requires VK)"
    if ssm.stride()[1:] != (value_dim * key_dim, key_dim, 1):
        return "unsupported SSM inner strides"
    if ssm.stride(0) < heads * value_dim * key_dim:
        return "overlapping SSM pages"
    if conv.shape[0] != ssm.shape[0] or conv.shape[0] < 2:
        return "invalid GDN page pool"
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return "unsupported activation dtype"
    if conv.dtype != dtype or ssm.dtype not in (dtype, torch.float32):
        return "unsupported native cache dtype combination"
    return None


def decode_metadata(metadata: Any, device: torch.device) -> dict[str, Any] | None:
    """Use existing host mirrors and device sources; never read NPU values on host.

    Only initialized, single-token non-speculative decode is supported. Prefill
    and checkpoint production remain on the native runner.
    """
    count = getattr(metadata, "num_actual_tokens", 0)
    if (
        count <= 0
        or getattr(metadata, "num_prefills", 0) != 0
        or getattr(metadata, "num_decodes", 0) != count
        or getattr(metadata, "max_query_len", None) != 1
    ):
        return None
    host = getattr(metadata, "seq_lens_cpu", None)
    seq_lens = getattr(metadata, "seq_lens_gpu", None)
    starts = getattr(metadata, "query_start_loc_gpu", None)
    if not isinstance(host, torch.Tensor) or host.device.type != "cpu" or host.ndim != 1:
        return None
    if host.numel() < count:
        return None
    totals = host[:count].tolist()
    if any(total <= 1 for total in totals):
        return None
    ends = getattr(metadata, "actual_seq_lengths_q", None)
    if not isinstance(ends, list) or ends[:count] != list(range(1, count + 1)):
        return None
    for value, minimum in ((seq_lens, count), (starts, count + 1)):
        if (
            not isinstance(value, torch.Tensor)
            or value.device != device
            or value.dtype != torch.int32
            or value.ndim != 1
            or value.numel() < minimum
        ):
            return None
    lens = torch.diff(starts[: count + 1])
    if starts.stride(0) != 1:
        return None
    return {
        "lens_cpu": [1] * count,
        "cached_lens_cpu": [total - 1 for total in totals],
        "lens": lens,
        "cached_lens": (seq_lens[:count] - lens).contiguous(),
        "query_start_loc": starts[:count],
    }


def physical_slot_mapping(slots: torch.Tensor, *, block_size: int, page_factor: int) -> torch.Tensor:
    """Map logical attention slots to padded physical pages; preserve padding."""
    if page_factor == 1:
        return slots.to(dtype=torch.int32).contiguous()
    mapped = (slots // block_size) * (page_factor * block_size) + slots % block_size
    return torch.where(slots < 0, slots, mapped).to(dtype=torch.int32).contiguous()


def decode_positions(
    positions: torch.Tensor, count: int, host_positions: torch.Tensor | None = None
) -> torch.Tensor | None:
    """V2 uses 1D positions. Collapse text MRoPE using its existing host mirror."""
    if positions.dtype != torch.int64:
        return None
    if positions.ndim == 2:
        if (
            positions.shape[0] != 3
            or not isinstance(host_positions, torch.Tensor)
            or host_positions.device.type != "cpu"
            or host_positions.ndim != 2
            or host_positions.shape[0] != 3
            or host_positions.shape[1] < count
        ):
            return None
        host = host_positions[:, :count]
        if not torch.equal(host[0], host[1]) or not torch.equal(host[0], host[2]):
            return None
        positions = positions[0]
    if positions.ndim != 1 or positions.numel() < count:
        return None
    return positions[:count].contiguous()
