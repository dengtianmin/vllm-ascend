# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for native GDN pages; this module never allocates request state."""

from collections.abc import Mapping
from typing import Any

import torch


def cache_support_reason(
    conv: torch.Tensor, ssm: torch.Tensor, *, channels: int, heads: int,
    key_dim: int, value_dim: int, width: int, dtype: torch.dtype,
) -> str | None:
    if width not in (2, 3, 4) or channels % 64 or key_dim % 16 or value_dim % 64:
        return "unsupported GDN dimensions"
    if not 0 < key_dim <= 128 or not 0 < value_dim <= 128:
        return "unsupported GDN head dimensions"
    if conv.device != ssm.device or conv.device.type != "npu":
        return "GDN cache is not on the runner NPU"
    if conv.ndim != 3 or tuple(conv.shape[1:]) != (width - 1, channels):
        return "unsupported conv layout (requires native SD)"
    if conv.stride()[1:] != (channels, 1) or conv.stride(0) < (width - 1) * channels:
        return "unsupported conv strides"
    if ssm.ndim != 4 or tuple(ssm.shape[1:]) != (heads, value_dim, key_dim):
        return "unsupported native SSM layout"
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


def decode_indices(metadata: Any, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read final per-step native indices without deriving a page from attention KV."""
    if metadata is None or count <= 0:
        return None
    if getattr(metadata, "num_prefills", 0) or getattr(metadata, "num_spec_decodes", 0):
        return None
    if getattr(metadata, "spec_sequence_masks", None) is not None:
        return None
    if getattr(metadata, "num_decodes", 0) != count or getattr(metadata, "num_actual_tokens", 0) != count:
        return None
    decode = getattr(metadata, "non_spec_decode_metadata", None)
    causal = getattr(decode, "causal_conv1d", None)
    conv = getattr(causal, "cache_indices", None)
    ssm = getattr(metadata, "non_spec_state_indices_tensor", None)
    boundaries = getattr(metadata, "non_spec_query_start_loc", None)
    if not isinstance(boundaries, torch.Tensor) or boundaries.numel() < count + 1:
        return None
    if not isinstance(conv, torch.Tensor) or not isinstance(ssm, torch.Tensor):
        return None
    # Matches the native causal-conv normalization; the GDN builder owns selection.
    if conv.ndim == 2:
        conv = conv[:, 0]
    if conv.ndim != 1 or ssm.ndim != 1 or conv.numel() < count or ssm.numel() < count:
        return None
    if conv.dtype != torch.int32 or ssm.dtype != torch.int32 or conv.device != device or ssm.device != device:
        return None
    return conv[:count].contiguous(), ssm[:count].contiguous()


def layer_metadata(metadata: Any, prefix: str) -> Any:
    if isinstance(metadata, list):
        # Micro-batched/ubatched metadata is not the pure-decode contract.
        return None
    return metadata.get(prefix) if isinstance(metadata, Mapping) else None


def prefill_metadata(
    metadata: Any, batch: int, tokens: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Consume native non-spec metadata for packed prefill, including mixed decode rows."""
    if metadata is None or batch <= 0 or tokens <= 0:
        return None
    if getattr(metadata, "num_spec_decodes", 0) or getattr(metadata, "spec_sequence_masks", None) is not None:
        return None
    if (getattr(metadata, "num_prefills", 0) <= 0
        or getattr(metadata, "num_prefills", 0) + getattr(metadata, "num_decodes", 0) != batch
        or getattr(metadata, "num_actual_tokens", 0) != tokens):
        return None
    causal = getattr(getattr(metadata, "non_spec_prefill_metadata", None), "causal_conv1d", None)
    conv = getattr(causal, "cache_indices", None)
    ssm = getattr(metadata, "non_spec_state_indices_tensor", None)
    starts = getattr(causal, "query_start_loc", None)
    conv_initial = getattr(causal, "initial_state_mode", None)
    ssm_initial = getattr(metadata, "has_initial_state", None)
    tensors = (conv, ssm, starts, conv_initial, ssm_initial)
    if any(not isinstance(t, torch.Tensor) or t.device != device for t in tensors):
        return None
    if conv.ndim == 2:
        conv = conv[:, 0]
    if (any(t.ndim != 1 for t in (conv, ssm, starts, conv_initial, ssm_initial))
        or any(t.dtype != torch.int32 for t in (conv, ssm, starts))
        or any(t.dtype not in (torch.bool, torch.int32) for t in (conv_initial, ssm_initial))
        or starts.numel() != batch + 1
        or any(t.numel() < batch for t in (conv, ssm, conv_initial, ssm_initial))):
        return None
    # Only execution metadata is converted; cache tensors keep their original storage/dtype.
    return (conv[:batch].contiguous(), ssm[:batch].contiguous(), starts.contiguous(),
            conv_initial[:batch].to(torch.int32).contiguous(),
            ssm_initial[:batch].to(torch.int32).contiguous())
