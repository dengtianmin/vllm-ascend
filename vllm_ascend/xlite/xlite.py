#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Xlite integration module for vLLM-Ascend."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import regex as re
import torch
import torch.nn as nn
import torch_npu
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from xlite._C import (
    AttnDSA,
    AttnHybrid,
    AttnMeta,
    AttnMetaV2,
    AttnMHA,
    AttnMLA,
    Runtime,
    ScoringFuncSigmoid,
    ScoringFuncSoftmax,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding
from vllm_ascend.xlite.gdn_cache import (
    cache_support_reason,
    decode_metadata,
    decode_positions,
    physical_slot_mapping,
)
from vllm_ascend.xlite.utils import (
    AttnMetadataRouter,
    WeightGetterConfig,
    XModel,
    XModelConfig,
    get_dotted_attr,
    get_layer_weights,
    set_dummy_tensor,
)

if TYPE_CHECKING:
    from vllm_ascend.xlite.xlite_model_runner import XliteModelRunner

XliteForwardResult: TypeAlias = torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]

_architecture_strategy_map: dict[str, type["XliteModelBase"]] = {}
"""Mapping from model architecture names in `config.json` to their corresponding xlite adapter classes."""


class XliteModelBase(ABC):
    """Base adapter for converting vLLM models into xlite runtime models.

    Subclasses are responsible for mapping architecture-specific configuration and weights into the `xlite._C.Model`
    interface.

    Attributes:
        npu_runner (XliteModelRunner): The NPU model runner instance.
        npu_runnable (nn.Module): The original runnable model used by vLLM-ascend. Used as the source of truth for
            weight extraction for xlite model construction.
        vllm_config (VllmConfig): The configuration object provided by vLLM. Used to build xlite configuration at
            runtime.
        device (torch.device): The device on which the model is loaded. Typically an Ascend NPU device.
        dtype (torch.dtype): The data type used for model weights and computations. Derived from `vllm_config`.
        rottary_embed (AscendRotaryEmbedding | None): The rotary embedding module extracted from the NPU runnable model.
        xlite_config (XModelConfig): Native xlite configuration object populated by subclasses.
        xlite_model (XModel): Native xlite model container populated by subclasses.
        rope_cossin_cache (torch.Tensor): Precomputed RoPE frequency cache tensor for rotary positional embeddings.
    """

    from vllm_ascend.attention.attention_v1 import AscendMetadata

    _attn_metadata_type: type | tuple[type, ...] = AscendMetadata
    """The expected type of attention metadata in the forward context for this architecture. Used for runtime checks
    before forwarding. See :meth:`XliteWrapper.__call__` for usage."""
    _supported_architectures: Sequence[str] | str
    """The list of model architecture names (from HuggingFace `config.json` "architectures" field) supported by this
    adapter. Used for automatic adapter selection and registration."""
    _decoder_layer_mlp_module: str = "mlp"
    """The module name used to identify MLP modules (including MoE) in the decoder layers (nn.Module) of the runnable
    model: e.g., in `Glm4MoeForCausalLM`, the MLP modules need to be accessed via `.model.layers[i].mlp`, thus `mlp`."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register subclasses in the architecture strategy map and metadata type set."""
        ts = getattr(cls, "_attn_metadata_type", None)
        if ts is None or (not isinstance(ts, type) and not all(isinstance(t, type) for t in ts)):
            raise ValueError(
                f"XliteModel subclass {cls.__name__} must define _attn_metadata_type as a type or a tuple of types."
            )

        arcs = getattr(cls, "_supported_architectures", None)
        if arcs is None:
            raise ValueError(f"XliteModel subclass {cls.__name__} must define _supported_architectures attribute.")
        if isinstance(arcs, str):
            arcs = [arcs]
        for arc in arcs:
            if arc in _architecture_strategy_map:
                raise ValueError(f"Duplicate xlite adapter for architecture {arc}: {_architecture_strategy_map[arc]}")
            _architecture_strategy_map[arc] = cls
        super().__init_subclass__(**kwargs)

    def __init__(self, npu_runner: "XliteModelRunner", vllm_config: VllmConfig) -> None:
        """Initialize the xlite model adapter.

        Args:
            npu_runner (XliteModelRunner): The NPU model runner instance.
            vllm_config (VllmConfig): Runtime configuration used for model setup.

        Notes:
            The constructor stores the runnable model and vLLM config, and prepares empty xlite configuration and model
            containers for subclass-specific population.
        """
        self.npu_runner = npu_runner
        self.npu_runnable = npu_runner.fallback_model
        self.vllm_config = vllm_config
        self.device = self.npu_runner.device
        self.dtype = vllm_config.model_config.dtype
        self.rottary_embed: AscendRotaryEmbedding | None = self._get_rope()
        if self.rottary_embed:
            logger.info_once(
                "xlite: found rotary embedding module `%s` in the NPU runnable.",
                re.sub(r"\s+", " ", repr(self.rottary_embed).replace("\n", "").strip()),
            )
        else:
            logger.warning_once("xlite: no rotary embedding module found in `%s`", self.npu_runnable)

        self.xlite_config = XModelConfig()
        self.xlite_model = XModel()
        self._build_model_config()
        self._build_model()
        self.xlite_model.init(self.xlite_config, torch.distributed.get_rank())
        self.rope_cossin_cache = self._precompute_rope_cache()

        if torch.distributed.get_rank() == 0:
            logger.info_once("xlite: built model config: %s", self.xlite_config)
            logger.info_once("xlite: built model weights: %s", self.xlite_model)

    def extract_kv_cache(self, kv_caches: list[tuple[torch.Tensor, ...]], /) -> list[tuple[torch.Tensor, ...]]:
        """Extract xlite-compatible KV cache from the vLLM-ascend KV cache.

        Args:
            kv_caches (list[tuple[torch.Tensor, ...]]): The KV cache from the upstream vLLM/vLLM-ascend.

        Returns:
            list[tuple[torch.Tensor, ...]]: The KV cache in the format expected by xlite model implementations.
        """
        # naively, the `kv_cache` is a list of <n_layers> tuples, each tuple contains two tensors of shape
        # [n_blocks, block_size, n_heads, head_dim] for K and V respectively.
        return kv_caches

    @abstractmethod
    def _build_model_config(self) -> None:
        """Build architecture-specific xlite model configuration.

        This method extracts necessary configuration attributes from the vLLM config (e.g., HuggingFace metadata) and
        populates an xlite :class:`ModelConfig` object.

        Returns:
            None: `self` attribute :attr:`xlite_config` is updated in-place.
        """

    @abstractmethod
    def _build_model(self) -> None:
        """Build architecture-specific xlite model weights.

        This method traverses the runnable model's parameters and maps them into the xlite :class:`Model` interface
        according to the architecture's specific structure.

        Returns:
            None: `self` attribute :attr:`xlite_model` is updated in-place.

        Notes:
            :meth:`_build_model_config` should be called prior to this method to ensure the xlite configuration is
            populated before weight mapping.
        """

    def _get_layers_and_model_prefix(self) -> tuple[Sequence[nn.Module], str]:
        """Extract transformer layers and parameter prefix from runnable.

        Returns:
            tuple[Sequence[nn.Module], str]: A pair of `(layers, model_prefix)` for model traversal.
        """
        if hasattr(self.npu_runnable, "language_model"):
            layers = cast(
                Sequence[nn.Module], get_dotted_attr(self.npu_runnable.language_model, "model.layers", default=[])
            )
            prefix = "language_model."
        else:
            layers = cast(Sequence[nn.Module], get_dotted_attr(self.npu_runnable, "model.layers", default=[]))
            prefix = ""
        return layers, prefix

    def _get_rope(self, layer_idx: int = 0) -> AscendRotaryEmbedding | None:
        """Extract the rotary embedding module from the NPU runnable model :data:`npu_runnable`.

        Args:
            layer_idx (int): The index of the transformer layer to extract the rotary embedding from. Defaults to 0.

        Returns:
            AscendRotaryEmbedding | None: The rotary embedding module found in the specified transformer layer, or None
            if not found.
        """
        layers, _ = self._get_layers_and_model_prefix()
        rottary_embed: AscendRotaryEmbedding | None = get_dotted_attr(layers, f"{layer_idx}.self_attn.rotary_emb")
        if not rottary_embed:
            logger.warning_once("xlite adapter: Rotary embedding not found in the model layers. RoPE will be disabled.")
        return rottary_embed

    def _precompute_rope_cache(self) -> torch.Tensor:
        """Precomputes frequency-based complex exponential values for rotary positional embeddings (RoPE).

        This method generates the RoPE frequency cache (cosine and sine values) required by the xlite attention
        implementation. The cache should be precomputed on the NPU device to avoid unnecessary host-device transfers
        during inference.

        Returns:
            torch.Tensor: The precomputed RoPE frequency cache tensor ready for use in xlite attention computations.

        Notes:
            :meth:`_build_model_config` should be called prior to this method.
        """
        if not self.rottary_embed:
            return torch.empty(2, dtype=torch.float32, device=self.device)
        cossin_cache: torch.Tensor | None = getattr(self.rottary_embed, "cossin_cache", None)
        if not isinstance(cossin_cache, torch.Tensor):
            cossin_cache = self.rottary_embed._compute_cos_sin_cache()
        return cossin_cache.to(dtype=torch.float32, device=self.device)

    @staticmethod
    def is_tensor_nz(t: torch.Tensor) -> bool:
        """Check if a tensor is in NZ format.

        Args:
            t (torch.Tensor): The tensor to check.

        Returns:
            bool: True if the tensor is in NZ format, False otherwise.
        """
        format = torch_npu.get_npu_format(t)
        return format == torch_npu.Format.FRACTAL_NZ

    @staticmethod
    def all_tensors_zero(tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None) -> bool:
        """Check if all tensors in the list/tuple are zero tensors.

        Args:
            tensors (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None): The tensors to check.

        Returns:
            bool: True if all tensors are zero tensors (or empty), False otherwise.
        """
        if tensors is None:
            return True
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        if len(tensors) == 0:
            return True
        return all(torch.allclose(t, t.new_zeros(1)) for t in tensors)

    @staticmethod
    def _transform_deq_scale(deq_scale: torch.Tensor) -> torch.Tensor:
        """Repack a dequantization scale into the fixpipe hardware's expected format.

        Data is stored in ``uint64_t``: the upper 32 bits are 0 and the lower 32 bits hold an FP32 value whose lower 10
        bits are not involved in computation, making the effective data format TF32.

        Args:
            deq_scale (torch.Tensor): The original dequantization scale tensor.

        Returns:
            torch.Tensor: The repacked scale tensor for fixpipe computation.
        """
        deq_scale_fp32 = deq_scale.to(torch.float32)
        scale = deq_scale_fp32.new_zeros(deq_scale.shape[0] * 2)
        scale[0::2] = deq_scale_fp32[0::1]
        return scale

    @property
    def hf_text_config(self) -> PretrainedConfig:
        """Convenience property to access HuggingFace text configuration from vLLM config.

        Returns:
            PretrainedConfig: The HuggingFace text configuration object extracted from vLLM config.
        """
        hf_config = self.vllm_config.model_config.hf_text_config
        return cast(PretrainedConfig, getattr(hf_config, "text_config", hf_config))

    @property
    def hf_vision_config(self) -> PretrainedConfig | None:
        """Convenience property to access HuggingFace vision configuration from vLLM config, if exists.

        Returns:
            PretrainedConfig | None: The HuggingFace vision configuration object extracted from vLLM config, or None if
                not present.
        """
        return getattr(self.vllm_config.model_config.hf_config, "vision_config", None)


class StandardXliteModel(XliteModelBase):
    """xlite adapter base for standard architectures.

    This is the *de facto* base adapter for all xlite-supported architectures. Configurations for various model types,
    dense and MoE, are included.

    Subclasses of :class:`XliteModelBase`, i.e., xlite model adapters, should inherit from :class:`StandardXliteModel`,
    unless there is a major divergence.

    **Developer Guide**: The :class:`StandardXliteModel` class is designed to be a flexible base for xlite adapters.
    When developing a new adapter for a specific architecture to dock with the xlite backend, consider adding weight
    loading lines in :meth:`StandardXliteModel._build_model` directly if feasible, using the highly flexible and
    error-checked `get_layer_weights` utility. For :meth:`_build_model_config`, override it in the subclass only if the
    architecture has unique configuration needs.
    """

    _supported_architectures = [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3VLForConditionalGeneration",
    ]

    def _build_model_config(self) -> None:
        xlite_config, vllm_config, hf_config = self.xlite_config, self.vllm_config, self.hf_text_config

        xlite_config.vocab_size = hf_config.vocab_size
        xlite_config.hidden_size = hf_config.hidden_size
        xlite_config.n_layers = hf_config.num_hidden_layers
        xlite_config.n_heads = hf_config.num_attention_heads
        xlite_config.n_kv_heads = hf_config.num_key_value_heads
        if hasattr(hf_config, "head_dim"):
            xlite_config.head_dim = hf_config.head_dim
        else:
            xlite_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        xlite_config.rope_head_dim = self.rottary_embed.rotary_dim if self.rottary_embed else xlite_config.head_dim
        xlite_config.norm_eps = hf_config.rms_norm_eps
        if hasattr(hf_config, "rope_theta"):
            xlite_config.rope_theta = hf_config.rope_theta
        else:
            xlite_config.rope_theta = getattr(hf_config, "rope_parameters", {}).get("rope_theta", 10000.0)
        xlite_config.softmax_scale = xlite_config.head_dim**-0.5
        xlite_config.n_dense_layers = hf_config.num_hidden_layers
        xlite_config.intermediate_size = hf_config.intermediate_size
        xlite_config.def_tp_size = tp_size = get_tensor_model_parallel_world_size()
        xlite_config.def_dp_size = vllm_config.parallel_config.data_parallel_size
        try:
            ep_word_size = get_ep_group().world_size
            xlite_config.moe_ep_size = ep_word_size if vllm_config.parallel_config.enable_expert_parallel else 1
            xlite_config.moe_tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else ep_word_size
        except AssertionError:
            xlite_config.moe_ep_size, xlite_config.moe_tp_size = 1, 1
        xlite_config.experts_weight_transpose = True

        xlite_config.attn_type = AttnMHA
        xlite_config.scoring_func = ScoringFuncSoftmax
        xlite_config.weight_nz = get_ascend_config().weight_nz_mode == 2
        max_m = vllm_config.scheduler_config.max_num_batched_tokens
        if not get_ascend_config().xlite_graph_config.full_mode:
            # decode-sized token budget: max_num_seqs * (1 + num_speculative_tokens) instead of max_num_batched_tokens
            n_spec_tokens = getattr(vllm_config.speculative_config, "num_speculative_tokens", 0)
            max_m = min(max_m, vllm_config.scheduler_config.max_num_seqs * (n_spec_tokens + 1))
        xlite_config.max_m = (max_m + tp_size - 1) // tp_size * tp_size  # round up to nearest multiple of tp_size
        xlite_config.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        xlite_config.max_seq_len = vllm_config.model_config.max_model_len
        xlite_config.block_size = vllm_config.cache_config.block_size

        rope_parameters = getattr(hf_config, "rope_parameters", {})
        xlite_config.deepstack_num_level = len(getattr(self.hf_vision_config, "deepstack_visual_indexes", []))
        xlite_config.mrope_section = rope_parameters.get("mrope_section", [])
        xlite_config.mrope_interleaved = rope_parameters.get("mrope_interleaved", False)
        self.quantization = vllm_config.quant_config is not None

    def _build_model(self) -> None:
        xlite_model, xlite_config, hf_config = self.xlite_model, self.xlite_config, self.hf_text_config
        layers, model_prefix = self._get_layers_and_model_prefix()

        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        if hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)

        xlite_model.attn_norm = get_layer_weights(layers, "input_layernorm.weight")
        self.init_matmul_weights(layers, "mha_qkv", "self_attn.qkv_proj")
        self.init_matmul_weights(layers, "attn_out", "self_attn.o_proj")

        with xlite_model.condition(lambda tensors: len(tensors) == xlite_config.n_layers):
            xlite_model.mha_q_norm = get_layer_weights(layers, "self_attn.q_norm.weight")
            xlite_model.mha_k_norm = get_layer_weights(layers, "self_attn.k_norm.weight")
            xlite_config.qk_norm = bool(xlite_model.mha_q_norm) and bool(xlite_model.mha_k_norm)

        # Dense MLP weights
        self.init_matmul_weights(layers, "mlp_up_gate", "mlp.gate_up_proj")
        self.init_matmul_weights(layers, "mlp_down", "mlp.down_proj")
        xlite_model.mlp_norm = get_layer_weights(layers, "post_attention_layernorm.weight")

        # MoE-specific weights
        mlp_prefix = self._decoder_layer_mlp_module
        xlite_model.gate = get_layer_weights(layers, f"{mlp_prefix}.gate.weight")
        xlite_config.gate_captured = self.is_tensor_nz(xlite_model.gate[0]) if xlite_model.gate else False
        xlite_model.gate_bias = get_layer_weights(
            layers,
            f"{mlp_prefix}.gate.e_score_correction_bias",
            f"{mlp_prefix}.e_score_correction_bias",  # for MiniMax-2.x compatibility
            post_processor=lambda b: b.to(torch.float32),  # type conversion for numerical stability in xlite backend
        )
        self.init_matmul_weights(layers, "se_up_gate", f"{mlp_prefix}.shared_experts.gate_up_proj")
        self.init_matmul_weights(layers, "se_down", f"{mlp_prefix}.shared_experts.down_proj")

        re_prefix = f"{mlp_prefix}.experts.routed_experts"
        re_kwargs: WeightGetterConfig = {"secondary_flattening": f"{re_prefix}.local_num_experts"}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{re_prefix}.w13_weight", **re_kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{re_prefix}.w2_weight", **re_kwargs)
        xlite_config.experts_weight_nz = bool(xlite_model.re_up_gate) and self.is_tensor_nz(xlite_model.re_up_gate[0])

        # bias terms
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mha_qkv_bias = get_layer_weights(layers, "self_attn.qkv_proj.bias")
            xlite_config.qkv_bias = len(xlite_model.mha_qkv_bias) == xlite_config.n_layers
            xlite_model.norm_bias = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.bias", raises=True)
            xlite_model.attn_norm_bias = get_layer_weights(layers, "input_layernorm.bias")
            xlite_model.mlp_norm_bias = get_layer_weights(layers, "post_attention_layernorm.bias")
            if xlite_config.qk_norm:
                xlite_model.mha_q_norm_bias = get_layer_weights(layers, "self_attn.q_norm.bias")
                xlite_model.mha_k_norm_bias = get_layer_weights(layers, "self_attn.k_norm.bias")

        if not self.quantization:
            return

        xlite_config.quant_attn_weight_nz = bool(xlite_model.mha_qkv) and self.is_tensor_nz(xlite_model.mha_qkv[0])
        xlite_config.quant_attn_weight_transpose = bool(xlite_model.mha_qkv)

        re_kwargs["post_processor"] = self._transform_deq_scale
        xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{re_prefix}.w13_weight_scale", **re_kwargs)
        xlite_model.re_down_scale = get_layer_weights(layers, f"{re_prefix}.w2_weight_scale", **re_kwargs)

    def init_matmul_weights(
        self,
        layers: Sequence[torch.nn.Module],
        xlite_prefix: str,
        model_prefix: str,
        *,
        mask: Sequence[bool] | None = None,
    ) -> None:
        """
        Initialize MatMul-related weights with quantization support.

        Args:
            layers (Sequence[torch.nn.Module]): The transformer layers to extract weights from.
            xlite_prefix (str): The prefix for the xlite model attributes to set.
            model_prefix (str): The prefix for the model attributes to look up in each layer.
            mask (Sequence[bool] | None): Optional mask to select specific layers for weight extraction. See
                :meth:`get_layer_weights` for details.
        """
        xlite_model = self.xlite_model
        setattr(xlite_model, xlite_prefix, get_layer_weights(layers, f"{model_prefix}.weight", mask=mask))
        if not self.quantization:
            return

        def set_xlite_attr(xlite_attr: str, layer_attr: str) -> None:
            setattr(xlite_model, xlite_attr, get_layer_weights(layers, layer_attr, mask=mask))

        wt_kwargs: WeightGetterConfig = {"mask": mask, "post_processor": self._transform_deq_scale}
        deq_scale = get_layer_weights(layers, f"{model_prefix}.deq_scale", **wt_kwargs)
        if len(deq_scale) > 0:  # static quant
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", deq_scale)
            set_xlite_attr(f"{xlite_prefix}_input_scale", f"{model_prefix}.aclnn_input_scale_reciprocal")
            set_xlite_attr(f"{xlite_prefix}_input_offset", f"{model_prefix}.aclnn_input_offset")
            set_xlite_attr(f"{xlite_prefix}_quant_bias", f"{model_prefix}.quant_bias")
        else:
            weight_scale = get_layer_weights(layers, f"{model_prefix}.weight_scale", **wt_kwargs)
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", weight_scale)

    def _precompute_rope_cache(self) -> torch.Tensor:
        # TODO: remove the type conversion once `xlite` accepts and defaults the cache in FP32 format
        return super()._precompute_rope_cache().to(dtype=self.dtype, device=self.device)


class QwenMoeXliteModel(StandardXliteModel):
    """xlite adapter for Qwen MoE architectures."""

    _supported_architectures = ["Qwen3MoeForCausalLM", "Qwen3VLMoeForConditionalGeneration"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob


class Qwen35XliteModel(StandardXliteModel):
    """XLite adapter for dense Qwen3.5's hybrid GDN/MHA architecture."""

    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    _attn_metadata_type = (StandardXliteModel._attn_metadata_type, GDNAttentionMetadata)
    _supported_architectures = ["Qwen3_5ForConditionalGeneration"]

    @staticmethod
    def _is_full_attention(layer: nn.Module) -> bool:
        return getattr(layer, "layer_type", None) == "full_attention"

    def _get_rope(self, layer_idx: int = 0) -> AscendRotaryEmbedding | None:
        # Qwen3.5 commonly starts with GDN layers, which have no rotary embedding.
        layers, _ = self._get_layers_and_model_prefix()
        for layer in layers:
            if self._is_full_attention(layer):
                rope = get_dotted_attr(layer, "self_attn.rotary_emb")
                if rope is not None:
                    return rope
        logger.warning_once("xlite adapter: Rotary embedding not found in Qwen3.5 full-attention layers.")
        return None

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config
        if self.quantization or self.vllm_config.lora_config is not None:
            raise NotImplementedError("Qwen3.5 XLite requires unquantized weights without LoRA.")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError("Qwen3.5 XLite full attention requires FP16 or BF16 activations.")
        mode = self.vllm_config.cache_config.mamba_cache_mode
        if mode not in ("none", "align"):
            raise NotImplementedError("Qwen3.5 XLite supports Mamba cache modes none and align.")
        if not getattr(hf_config, "attn_output_gate", True):
            raise NotImplementedError("Qwen3.5 XLite requires the full-attention output gate.")
        if not hasattr(xlite_config, "linear_block_table_indices"):
            raise RuntimeError("Qwen3.5 AttnMetaV2 requires the matching GVirt XLite extension.")
        # GDN projections consume ND weights directly.
        xlite_config.weight_nz = False
        # V2 carries 1D positions. Equal text axes use ordinary RoPE; retaining
        # MRoPE masks would make the kernel read past this 1D position tensor.
        xlite_config.mrope_section = []
        xlite_config.mrope_interleaved = False
        xlite_config.deepstack_num_level = 0
        tp_size = xlite_config.def_tp_size
        max_decode = min(
            self.vllm_config.scheduler_config.max_num_batched_tokens, self.vllm_config.scheduler_config.max_num_seqs
        )
        xlite_config.max_m = (max_decode + tp_size - 1) // tp_size * tp_size

        layer_types = list(getattr(hf_config, "layer_types", []))
        if not layer_types:
            layers, _ = self._get_layers_and_model_prefix()
            layer_types = [getattr(layer, "layer_type", "") for layer in layers]
        full_attention_indices = [i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention"]
        if not full_attention_indices:
            raise ValueError("Qwen3.5 XLite adapter requires at least one full-attention layer.")

        full_attention_interval = full_attention_indices[0] + 1
        expected_full_attention_indices = list(
            range(full_attention_interval - 1, len(layer_types), full_attention_interval)
        )
        if full_attention_indices != expected_full_attention_indices:
            raise ValueError(
                "XLite Qwen3.5 expects full-attention layers at a fixed interval; "
                f"got indices {full_attention_indices}."
            )

        xlite_config.attn_type = AttnHybrid
        xlite_config.full_attention_interval = full_attention_interval
        xlite_config.linear_num_k_heads = hf_config.linear_num_key_heads
        xlite_config.linear_num_v_heads = hf_config.linear_num_value_heads
        xlite_config.linear_key_head_dim = hf_config.linear_key_head_dim
        xlite_config.linear_value_head_dim = hf_config.linear_value_head_dim
        xlite_config.linear_conv_kernel_dim = hf_config.linear_conv_kernel_dim
        xlite_config.qkv_bias = bool(getattr(hf_config, "qkv_bias", False))
        xlite_config.qk_norm = True
        xlite_config.attn_output_gate = bool(getattr(hf_config, "attn_output_gate", True))
        if len(layer_types) != xlite_config.n_layers or any(
            value not in ("full_attention", "linear_attention") for value in layer_types
        ):
            raise ValueError("Qwen3.5 layer types must describe every decoder layer.")
        if (
            xlite_config.linear_num_k_heads <= 0
            or xlite_config.linear_num_v_heads <= 0
            or xlite_config.linear_num_k_heads % tp_size
            or xlite_config.linear_num_v_heads % tp_size
            or xlite_config.linear_num_v_heads % xlite_config.linear_num_k_heads
        ):
            raise ValueError("Qwen3.5 GDN requires positive TP-divisible heads and integral V/K grouping.")
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        xlite_config.block_size = AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        state_block_size = self.vllm_config.cache_config.mamba_block_size
        if state_block_size is None or state_block_size <= 0:
            raise ValueError("Qwen3.5 requires the resolved Mamba block size.")
        # Stable V2 entries before native Model.init: MHA zero, then one per GDN
        # layer. Registration resolves these entries to actual vLLM cache groups.
        block_sizes = [xlite_config.block_size]
        table_indices = []
        for layer_type in layer_types:
            if layer_type == "full_attention":
                table_indices.append(0)
            else:
                table_indices.append(len(block_sizes))
                block_sizes.append(state_block_size)
        xlite_config.block_sizes = block_sizes
        xlite_config.linear_block_table_indices = table_indices
        xlite_config.linear_cache_aligned = mode == "align"

    def _build_model(self) -> None:
        if self.quantization:
            raise NotImplementedError("Qwen3.5 XLite adapter currently supports unquantized weights only.")

        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        hf_config = self.hf_text_config
        layers, model_prefix = self._get_layers_and_model_prefix()

        def gemma_norm(weight: torch.Tensor) -> torch.Tensor:
            # Qwen3.5 uses Gemma-style RMSNorm, while XLite expects the effective scale.
            return (weight.float() + 1.0).to(weight.dtype).contiguous()

        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = gemma_norm(
            get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        )
        if hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)

        xlite_model.attn_norm = [gemma_norm(w) for w in get_layer_weights(layers, "input_layernorm.weight")]
        xlite_model.mlp_norm = [gemma_norm(w) for w in get_layer_weights(layers, "post_attention_layernorm.weight")]
        xlite_model.mlp_up_gate = get_layer_weights(layers, "mlp.gate_up_proj.weight")
        xlite_model.mlp_down = get_layer_weights(layers, "mlp.down_proj.weight")

        def nd(weight: torch.Tensor) -> torch.Tensor:
            return torch_npu.npu_format_cast(weight, torch_npu.Format.ND).contiguous()

        xlite_model.embed = nd(xlite_model.embed)
        xlite_model.head = xlite_model.embed if hf_config.tie_word_embeddings else nd(xlite_model.head)
        xlite_model.mlp_up_gate = [nd(weight) for weight in xlite_model.mlp_up_gate]
        xlite_model.mlp_down = [nd(weight) for weight in xlite_model.mlp_down]
        empty = torch.empty(0, dtype=self.dtype, device=self.device)
        mha_qkv, attn_out, mha_q_norm, mha_k_norm = [], [], [], []
        mha_qkv_bias = []
        linear_qkv, linear_z, linear_b, linear_a = [], [], [], []
        linear_conv, linear_a_log, linear_dt_bias, linear_norm, linear_out = [], [], [], [], []

        tp_size = get_tensor_model_parallel_world_size()
        for layer in layers:
            if self._is_full_attention(layer):
                attn = layer.self_attn
                proj = attn.qkv_proj
                q_width = attn.num_heads * attn.head_dim
                kv_width = attn.num_kv_heads * attn.head_dim

                # vLLM stores each query head as [Q | output-gate], followed by K and V.
                # XLite's hybrid kernel expects [Q | K | V | output-gate].
                qkv_weight = nd(proj.weight)
                q_gate = qkv_weight[: 2 * q_width].view(attn.num_heads, 2, attn.head_dim, -1)
                q_weight = q_gate[:, 0].reshape(q_width, -1).contiguous()
                gate_weight = q_gate[:, 1].reshape(q_width, -1).contiguous()
                k_start = 2 * q_width
                k_weight = qkv_weight[k_start : k_start + kv_width]
                v_weight = qkv_weight[k_start + kv_width : k_start + 2 * kv_width]
                mha_qkv.append(torch.cat((q_weight, k_weight, v_weight, gate_weight), dim=0).contiguous())
                attn_out.append(nd(attn.o_proj.weight))
                mha_q_norm.append(gemma_norm(attn.q_norm.weight))
                mha_k_norm.append(gemma_norm(attn.k_norm.weight))

                if xlite_config.qkv_bias:
                    qkv_bias = proj.bias
                    if qkv_bias is None:
                        raise ValueError("Qwen3.5 config enables qkv_bias, but a full-attention layer has no bias.")
                    q_gate_bias = qkv_bias[: 2 * q_width].view(attn.num_heads, 2, attn.head_dim)
                    q_bias = q_gate_bias[:, 0].reshape(-1).contiguous()
                    gate_bias = q_gate_bias[:, 1].reshape(-1).contiguous()
                    k_bias = qkv_bias[k_start : k_start + kv_width]
                    v_bias = qkv_bias[k_start + kv_width : k_start + 2 * kv_width]
                    mha_qkv_bias.append(torch.cat((q_bias, k_bias, v_bias, gate_bias), dim=0).contiguous())

                linear_qkv.append(empty)
                linear_z.append(empty)
                linear_b.append(empty)
                linear_a.append(empty)
                linear_conv.append(empty)
                linear_a_log.append(empty)
                linear_dt_bias.append(empty)
                linear_norm.append(empty)
                linear_out.append(empty)
            else:
                linear = layer.linear_attn
                qkvz_weight = nd(linear.in_proj_qkvz.weight)
                local_key_dim = linear.key_dim // tp_size
                local_value_dim = linear.value_dim // tp_size
                qkv_end = 2 * local_key_dim + local_value_dim
                linear_qkv.append(qkvz_weight[:qkv_end].contiguous())
                linear_z.append(qkvz_weight[qkv_end : qkv_end + local_value_dim].contiguous())

                ba_weight = nd(linear.in_proj_ba.weight)
                local_value_heads = hf_config.linear_num_value_heads // tp_size
                linear_b.append(ba_weight[:local_value_heads].contiguous())
                linear_a.append(ba_weight[local_value_heads : 2 * local_value_heads].contiguous())
                linear_conv.append(linear.conv1d.weight)
                linear_a_log.append(linear.A_log)
                linear_dt_bias.append(linear.dt_bias)
                # GDN's gated RMSNorm uses the stored scale directly (unlike Gemma RMSNorm).
                linear_norm.append(linear.norm.weight)
                linear_out.append(nd(linear.out_proj.weight))

                mha_qkv.append(empty)
                attn_out.append(empty)
                mha_q_norm.append(empty)
                mha_k_norm.append(empty)
                if xlite_config.qkv_bias:
                    mha_qkv_bias.append(empty)

        xlite_model.mha_qkv = mha_qkv
        xlite_model.attn_out = attn_out
        xlite_model.mha_q_norm = mha_q_norm
        xlite_model.mha_k_norm = mha_k_norm
        if xlite_config.qkv_bias:
            xlite_model.mha_qkv_bias = mha_qkv_bias
        xlite_model.linear_in_proj_qkv = linear_qkv
        xlite_model.linear_in_proj_z = linear_z
        xlite_model.linear_in_proj_b = linear_b
        xlite_model.linear_in_proj_a = linear_a
        xlite_model.linear_conv1d = linear_conv
        xlite_model.linear_a_log = linear_a_log
        xlite_model.linear_dt_bias = linear_dt_bias
        xlite_model.linear_norm = linear_norm
        xlite_model.linear_out_proj = linear_out

    def extract_kv_cache(self, kv_caches: list[tuple[torch.Tensor, ...]], /) -> list[tuple[torch.Tensor, ...]]:
        """Reference caches bound to attention modules without changing storage."""
        from vllm_ascend.xlite.utils import _DUMMY_TENSOR

        layers, _ = self._get_layers_and_model_prefix()
        if len(layers) != self.xlite_config.n_layers:
            raise ValueError("Qwen3.5 model/cache layer count mismatch.")
        full_caches: list[tuple[torch.Tensor, ...]] = []
        for layer_idx, layer in enumerate(layers):
            if not self._is_full_attention(layer):
                full_caches.append((_DUMMY_TENSOR, _DUMMY_TENSOR))
                continue
            cache = layer.self_attn.attn.kv_cache
            if len(cache) < 2:
                raise ValueError(f"Qwen3.5 full-attention layer {layer_idx} requires K and V cache tensors.")
            full_caches.append((cache[0], cache[1]))
        return full_caches


class Glm4MoeXliteModel(StandardXliteModel):
    """xlite adapter for GLM4 MoE architectures."""

    _supported_architectures = ["Glm4MoeForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.n_dense_layers = getattr(hf_config, "first_k_dense_replace", 0)
        xlite_config.n_routed_experts = hf_config.n_routed_experts
        xlite_config.n_shared_experts = hf_config.n_shared_experts
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob
        xlite_config.scoring_func = ScoringFuncSigmoid
        xlite_config.route_scale = hf_config.routed_scaling_factor


class DeepseekV3XliteModel(Glm4MoeXliteModel):
    """xlite adapter for DeepseekV3 MoE architectures with MLA attention."""

    from vllm_ascend.attention.mla_v1 import AscendMLAMetadata

    _attn_metadata_type = AscendMLAMetadata  # type: ignore[assignment]
    _supported_architectures = ["DeepseekV3ForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        # MLA attention type
        xlite_config.attn_type = AttnMLA
        xlite_config.n_kv_heads = 1  # MLA uses latent cache
        xlite_config.head_dim = 0  # Ignored by MLA

        # MLA dimensions (override Llama's head_dim-based rope_head_dim)
        xlite_config.rope_head_dim = hf_config.qk_rope_head_dim
        xlite_config.nope_head_dim = hf_config.qk_nope_head_dim
        xlite_config.q_lora_rank = hf_config.q_lora_rank
        xlite_config.kv_lora_rank = hf_config.kv_lora_rank
        xlite_config.v_head_dim = hf_config.v_head_dim
        xlite_config.softmax_scale = (hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim) ** -0.5
        # correct softmax_scale for yarn-style RoPE if max_seq_len > original_max_position_embeddings
        rope_params: dict[str, int | float | str] = getattr(hf_config, "rope_parameters", {})
        original_max_len: int = rope_params.get("original_max_position_embeddings", hf_config.max_position_embeddings)  # type: ignore[assignment]
        if xlite_config.max_seq_len > original_max_len and "mscale" in rope_params and "factor" in rope_params:
            mscale: float = 1.0 + 0.1 * rope_params["mscale"] * math.log(rope_params["factor"])  # type: ignore[operator,arg-type]
            xlite_config.softmax_scale *= mscale**2

        # MoE configuration (from Glm4MoeXliteModel, adapted for DeepseekV3)
        xlite_config.n_expert_groups = getattr(hf_config, "n_group", 1)
        xlite_config.n_limited_groups = getattr(hf_config, "topk_group", 1)

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, _ = self._get_layers_and_model_prefix()

        # MLA attention weights
        self.init_matmul_weights(layers, "mla_qkv_a", "self_attn.fused_qkv_a_proj")
        self.init_matmul_weights(layers, "mla_q_b", "self_attn.q_b_proj")
        xlite_model.mla_q_norm = get_layer_weights(layers, "self_attn.q_a_layernorm.weight")
        xlite_model.mla_kv_norm = get_layer_weights(layers, "self_attn.kv_a_layernorm.weight")
        xlite_model.mla_wuv = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UV")
        xlite_model.mla_wuk_t = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UK_T")
        xlite_config.mla_wuv_weight_nz = bool(wuv_lst := xlite_model.mla_wuv) and self.is_tensor_nz(wuv_lst[0])
        xlite_config.mla_wukt_weight_nz = bool(wuk_lst := xlite_model.mla_wuk_t) and self.is_tensor_nz(wuk_lst[0])

        if not self.quantization:
            return

        self.xlite_config.quant_attn_weight_nz = bool(wt_lst := xlite_model.mla_qkv_a) and self.is_tensor_nz(wt_lst[0])
        self.xlite_config.quant_attn_weight_transpose = True
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mla_q_norm_bias = get_layer_weights(layers, "self_attn.q_a_layernorm.bias")
            xlite_model.mla_kv_norm_bias = get_layer_weights(layers, "self_attn.kv_a_layernorm.bias")

    def _precompute_rope_cache(self) -> torch.Tensor:
        # TODO: remove this method once `xlite` no longer requires complex64 format
        cossin_cache = super(StandardXliteModel, self)._precompute_rope_cache()
        if not self.rottary_embed:
            return cossin_cache.to(dtype=torch.complex64, device=self.device)
        # For MLA attention, we need to return the cache as a complex instead of separate cos/sin tensors
        if not (isinstance(cossin_cache, torch.Tensor) and cossin_cache.ndim == 2 and cossin_cache.shape[1] % 2 == 0):
            raise ValueError("RoPE cache must be a 2D tensor with even second dimension for MLA attention.")
        sep = cossin_cache.shape[-1] // 2
        return torch.complex(cossin_cache[:, :sep], cossin_cache[:, sep:]).to(dtype=torch.complex64, device=self.device)


class DeepseekV32XliteModel(DeepseekV3XliteModel):
    """xlite adapter for Deepseek-V3.2/GLM-5.x architectures with Deepseek sparse attention (DSA)."""

    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

    _attn_metadata_type = AscendSFAMetadata  # type: ignore[assignment]
    _supported_architectures = ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.attn_type = AttnDSA
        xlite_config.index_head_dim = hf_config.index_head_dim
        xlite_config.index_n_heads = hf_config.index_n_heads
        xlite_config.index_topk = hf_config.index_topk
        xlite_config.index_rope_interleaved = getattr(hf_config, "indexer_rope_interleave", False)
        index_types: list[str] = getattr(hf_config, "indexer_types", []) or []
        xlite_config.index_full_mask = list(map(lambda x: str(x).lower().startswith("full"), index_types))

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model = self.xlite_model
        layers, _ = self._get_layers_and_model_prefix()

        mask: list[bool] | None = getattr(self.xlite_config, "index_full_mask", None) or None
        self.init_matmul_weights(layers, "index_q_b", f"{(prefix := 'self_attn.indexer')}.wq_b", mask=mask)
        xlite_model.index_k_weights_proj = get_layer_weights(layers, f"{prefix}.wk_weights_proj.weight", mask=mask)
        # xlite backend expects index_k_norm and index_k_norm_bias to have dtype=torch.float32 or the model dtype
        xlite_model.index_k_norm = get_layer_weights(layers, f"{prefix}.k_norm.weight", mask=mask)
        xlite_model.index_k_norm_bias = get_layer_weights(layers, f"{prefix}.k_norm.bias", mask=mask)

    def extract_kv_cache(self, kv_caches: list[tuple[torch.Tensor, ...]], /) -> list[tuple[torch.Tensor, ...]]:
        from vllm_ascend.xlite.utils import _DUMMY_TENSOR

        # For DSA, the kv_caches are passed as [(indexer_k_cache,), (k_nope_cache, k_pe_cache), ..., <mtp_layer>]
        # TODO: consider the compatibility with `enable_sparse_sfa_c8` and `enable_sparse_li_c8`
        indexer_caches: list[tuple[torch.Tensor, ...]] = []
        mla_caches: list[tuple[torch.Tensor, ...]] = []

        idx = 0
        index_mask = getattr(self.xlite_config, "index_full_mask", None) or [True] * self.xlite_config.n_layers
        dummy_indexer_cache = (_DUMMY_TENSOR,)
        for mask in index_mask:
            indexer_caches.append(kv_caches[idx] if mask else dummy_indexer_cache)
            mla_caches.append(kv_caches[idx + int(mask)])
            idx += 1 + int(mask)
        return [mla_c[:2] + indexer_c[:1] for mla_c, indexer_c in zip(mla_caches, indexer_caches)]


class MiniMaxM2XliteModel(StandardXliteModel):
    """xlite adapter for MiniMax M2 architectures."""

    _supported_architectures = ["MiniMaxM2ForCausalLM"]
    _decoder_layer_mlp_module = "block_sparse_moe"

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.rope_head_dim = hf_config.rotary_dim
        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_local_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.intermediate_size
        xlite_config.norm_topk_prob = True
        xlite_config.qk_norm_full = True
        xlite_config.scoring_func = ScoringFuncSigmoid


def get_adapter_xlite_model(npu_runner: "XliteModelRunner", vllm_config: VllmConfig) -> XliteModelBase:
    """Look up and initialize the appropriate xlite model adapter based on the architecture specified in vLLM config and
    the runnable model.

    Args:
        npu_runner (XliteModelRunner): The NPU model runner instance.
        vllm_config (VllmConfig): Runtime configuration for model execution.

    Raises:
        ValueError: If the model architecture is not supported by xlite.

    Returns:
        XliteModelBase: An initialized xlite model adapter ready for inference.
    """
    architecture = vllm_config.model_config.architectures[0]
    if not (strategy_class := _architecture_strategy_map.get(architecture)):
        raise ValueError(f"{architecture} not supported!")
    return strategy_class(npu_runner, vllm_config)


class XliteWrapper:
    """A graph-based wrapper that dispatches between xlite and runnable paths.

    Since v0.28.0, batches are routed by token count (max across DP ranks) instead of the batch attention state:
    batches within the preallocated budget (:attr:`max_tokens`, sized by ``full_mode``) run on xlite, others fall
    back to the native runnable.
    """

    def __init__(self, npu_runner: "XliteModelRunner", vllm_config: VllmConfig, device: torch.device) -> None:
        """Initialize xlite runtime, model tensors, and hidden-state workspace.

        Args:
            npu_runner (XliteModelRunner): The native NPU model runner.
            vllm_config (VllmConfig): Runtime configuration for execution.
            device (torch.device): The device to initialize the xlite model on.

        Raises:
            ValueError: If xlite runtime tensor-pool initialization fails.
        """
        self.npu_runner = npu_runner
        self.npu_runnable = npu_runner.fallback_model
        self.vllm_config = vllm_config
        self.device = device
        # `full_mode` only sizes the token budget (see the comment in `StandardXliteModel._build_config`).
        self.full_mode: bool = get_ascend_config().xlite_graph_config.full_mode
        self.data_parallel_size = vllm_config.parallel_config.data_parallel_size

        set_dummy_tensor(torch.empty(1, device=device, dtype=vllm_config.model_config.dtype))
        self.adapter_xlite_model = get_adapter_xlite_model(npu_runner, vllm_config)  # python adapter instance
        self._qwen35_hybrid = isinstance(self.adapter_xlite_model, Qwen35XliteModel)
        self.gdn_cache_reason: str | None = "cache not registered"
        self.xlite_model = self.adapter_xlite_model.xlite_model  # xlite C++ model instance
        self.freq_cis = self.adapter_xlite_model.rope_cossin_cache
        xlite_config = self.adapter_xlite_model.xlite_config
        self.xlite_rt = Runtime(
            devid=device.index,
            size=0,
            rank=torch.distributed.get_rank(),
            tp_size=xlite_config.def_tp_size,
            dp_size=xlite_config.def_dp_size,
            moe_tp_size=xlite_config.moe_tp_size,
            moe_ep_size=xlite_config.moe_ep_size,
        )
        # The xlite token budget: batches with more tokens (max across DP ranks) are routed to the native runnable.
        self.max_tokens = xlite_config.max_m
        self.hidden_states = torch.empty(
            self.max_tokens, xlite_config.hidden_size, device=self.device, dtype=self.adapter_xlite_model.dtype
        )

        rt_pool_size = self.xlite_model.get_tensor_pool_size()
        if torch.distributed.get_rank() == 0:
            logger.info("xlite runtime pool size: %s MB", rt_pool_size)
        if self.xlite_rt.init_tensor_pool(rt_pool_size) != 0:
            raise ValueError(f"xlite wrapper init failed! runtime pool size: {rt_pool_size} MB")

    def __getattr__(self, key: str) -> Any:
        """Proxy unknown attributes to the wrapped runnable model.

        Args:
            key (str): The attribute name requested by the caller.

        Raises:
            AttributeError: If neither wrapper nor runnable has the attribute.

        Returns:
            Any: Attribute value resolved from the runnable.
        """
        try:
            return getattr(self.npu_runnable, key)
        except Exception as e:  # runnable may raise various exceptions
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {key}") from e

    def register_kv_caches(self, kv_caches: list[tuple[torch.Tensor, ...]]) -> None:
        """Register KV cache references used by xlite runtime.

        Args:
            kv_caches (list[tuple[torch.Tensor, ...]]): Runtime KV cache handles or tensors.
        """
        self.kv_caches = self.adapter_xlite_model.extract_kv_cache(kv_caches)
        if self._qwen35_hybrid:
            self._register_qwen35_caches()

    def _register_qwen35_caches(self) -> None:
        """Bind native views and resolve static V2 entries to framework groups."""
        adapter = cast(Qwen35XliteModel, self.adapter_xlite_model)
        layers, _ = adapter._get_layers_and_model_prefix()
        cfg = adapter.xlite_config
        groups = self.npu_runner.kv_cache_config.kv_cache_groups
        group_by_layer = {name: gid for gid, group in enumerate(groups) for name in group.layer_names}
        self.cache_table_groups = [-1] * len(cfg.block_sizes)
        self.full_attention_prefixes = []
        self.full_attention_page_factor = None
        self.gdn_cache_reason = None
        local_k = cfg.linear_num_k_heads // cfg.def_tp_size
        local_v = cfg.linear_num_v_heads // cfg.def_tp_size
        channels = 2 * local_k * cfg.linear_key_head_dim + local_v * cfg.linear_value_head_dim
        for index, layer in enumerate(layers):
            full = adapter._is_full_attention(layer)
            prefix = layer.self_attn.attn.layer_name if full else layer.linear_attn.prefix
            gid = group_by_layer.get(prefix)
            if gid is None:
                self.gdn_cache_reason = f"missing framework cache group for {prefix}"
                break
            entry = cfg.linear_block_table_indices[index]
            previous = self.cache_table_groups[entry]
            if previous != -1 and previous != gid:
                self.gdn_cache_reason = "full-attention layers require a shared block table"
                break
            self.cache_table_groups[entry] = gid
            table = self.npu_runner.input_batch.block_table[gid]
            if table.block_size != cfg.block_sizes[entry]:
                self.gdn_cache_reason = "framework and XLite cache block sizes differ"
                break
            if full:
                self.full_attention_prefixes.append(prefix)
                k, v = self.kv_caches[index]
                heads = max(cfg.n_kv_heads // cfg.def_tp_size, 1)
                logical_stride = cfg.block_sizes[0] * heads * cfg.head_dim
                inner = (heads * cfg.head_dim, cfg.head_dim, 1)
                expected = (cfg.block_sizes[0], heads, cfg.head_dim)
                if (
                    k.ndim != 4
                    or tuple(k.shape[1:]) != expected
                    or k.shape != v.shape
                    or k.stride()[1:] != inner
                    or v.stride() != k.stride()
                    or k.stride(0) % logical_stride
                    or k.stride(0) < logical_stride
                    or k.dtype != adapter.dtype
                    or v.dtype != k.dtype
                    or k.device != self.device
                    or v.device != self.device
                ):
                    self.gdn_cache_reason = "unsupported full-attention cache descriptor"
                    break
                factor = k.stride(0) // logical_stride
                if self.full_attention_page_factor not in (None, factor):
                    self.gdn_cache_reason = "full-attention page strides differ across layers"
                    break
                self.full_attention_page_factor = factor
            else:
                state = getattr(layer.linear_attn, "kv_cache", None)
                if not isinstance(state, (tuple, list)) or len(state) != 2:
                    self.gdn_cache_reason = "missing initialized native GDN cache"
                    break
                conv, ssm = state
                reason = cache_support_reason(
                    conv,
                    ssm,
                    channels=channels,
                    heads=local_v,
                    key_dim=cfg.linear_key_head_dim,
                    value_dim=cfg.linear_value_head_dim,
                    width=cfg.linear_conv_kernel_dim,
                    dtype=adapter.dtype,
                    device=self.device,
                )
                if reason is not None:
                    self.gdn_cache_reason = reason
                    break
                self.kv_caches[index] = (conv, ssm)
        if self.gdn_cache_reason is None and any(gid < 0 for gid in self.cache_table_groups):
            self.gdn_cache_reason = "incomplete V2 cache table mapping"
        if self.gdn_cache_reason:
            logger.warning("xlite Qwen3.5 native fallback: %s", self.gdn_cache_reason)
        else:
            logger.info("xlite Qwen3.5 registered native caches with raw V2 state tables")

    def _qwen35_forward(
        self,
        context: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
    ) -> torch.Tensor | None:
        config = self.vllm_config
        parallel = config.parallel_config
        metadata = context.attn_metadata
        if (
            self.gdn_cache_reason is not None
            or (input_ids is None and inputs_embeds is None)
            or intermediate_tensors is not None
            or self.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or config.lora_config is not None
            or config.cache_config.mamba_cache_mode not in ("none", "align")
            or not isinstance(metadata, dict)
            or not self.full_attention_prefixes
            or getattr(self.npu_runnable, "deepstack_input_embeds", [])
        ):
            return None
        mha = metadata.get(self.full_attention_prefixes[0])
        decoded = decode_metadata(mha, self.device)
        if decoded is None:
            return None
        count = len(decoded["lens_cpu"])
        if count > self.max_tokens or count > config.scheduler_config.max_num_seqs:
            return None
        for prefix in self.full_attention_prefixes[1:]:
            current = decode_metadata(metadata.get(prefix), self.device)
            if current is None or current["cached_lens_cpu"] != decoded["cached_lens_cpu"]:
                return None
        source = inputs_embeds if inputs_embeds is not None else input_ids
        if (
            source is None
            or source.ndim == 0
            or source.shape[0] < count
            or source.device != self.device
            or not source.is_contiguous()
        ):
            return None
        if inputs_embeds is not None and (
            inputs_embeds.ndim != 2
            or inputs_embeds.shape[1] != self.adapter_xlite_model.xlite_config.hidden_size
            or inputs_embeds.dtype != self.adapter_xlite_model.dtype
        ):
            return None
        if input_ids is not None and (
            input_ids.ndim != 1
            or input_ids.dtype != torch.int64
            or input_ids.device != self.device
            or input_ids.numel() < count
            or not input_ids.is_contiguous()
        ):
            return None
        host_positions = getattr(getattr(self.npu_runner, "mrope_positions", None), "cpu", None)
        position = decode_positions(positions, count, host_positions)
        if position is None or position.device != self.device:
            return None
        # Read device sources directly. The router must not move block tables to CPU.
        router = AttnMetadataRouter(attn_metadata=mha, device=self.device)
        full_table = router.block_tables
        slots = getattr(mha, "slot_mapping", None)
        if (
            not isinstance(slots, torch.Tensor)
            or slots.device != self.device
            or slots.ndim != 1
            or slots.numel() < count
            or slots.dtype not in (torch.int32, torch.int64)
            or full_table.ndim != 2
            or full_table.shape[0] < count
            or full_table.dtype != torch.int32
            or full_table.device != self.device
        ):
            return None
        cfg = self.adapter_xlite_model.xlite_config
        factor = self.full_attention_page_factor
        if full_table.shape[1] < (max(decoded["cached_lens_cpu"]) + cfg.block_sizes[0]) // cfg.block_sizes[0]:
            return None
        tables = [(full_table[:count] * factor).contiguous() if factor != 1 else full_table[:count].contiguous()]
        mappings = [physical_slot_mapping(slots[:count], block_size=cfg.block_sizes[0], page_factor=factor)]
        empty = slots.new_empty(0, dtype=torch.int32)
        for entry, gid in enumerate(self.cache_table_groups[1:], 1):
            # This is the raw framework table, never the builder's gathered state indices.
            table = self.npu_runner.input_batch.block_table[gid].get_device_tensor()
            minimum = (
                1
                if config.cache_config.mamba_cache_mode == "none"
                else (max(decoded["cached_lens_cpu"]) + cfg.block_sizes[entry]) // cfg.block_sizes[entry]
            )
            if (
                table.ndim != 2
                or table.shape[0] < count
                or table.shape[1] < minimum
                or table.dtype != torch.int32
                or table.device != self.device
                or table.stride(1) != 1
                or table.stride(0) < table.shape[1]
            ):
                return None
            tables.append(table[:count])
            mappings.append(empty)
        meta = AttnMetaV2()
        for key, value in decoded.items():
            setattr(meta, key, value)
        meta.positions = position
        meta.block_tables = tables
        meta.slot_mapping = mappings
        output = self.hidden_states[:count]
        args = dict(
            rt=self.xlite_rt,
            attn_meta=meta,
            kv_cache=self.kv_caches,
            output=output,
            curr_stream=torch.npu.current_stream().npu_stream,
        )
        if inputs_embeds is None:
            self.xlite_model.forward_v2(input=input_ids[:count], freqs_cis=[self.freq_cis], **args)
        else:
            self.xlite_model.forward_with_inputs_embeds_v2(
                input=inputs_embeds[:count],
                freqs_cis=self.freq_cis,
                input_ids=input_ids[:count] if input_ids is not None else None,
                **args,
            )
        logger.info_once("xlite Qwen3.5 route: initialized decode with native caches and AttnMetaV2")
        return output

    def __call__(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> XliteForwardResult:
        """Run one forward step through xlite graph or fallback runnable path.

        Args:
            input_ids (torch.Tensor | None): Token IDs for current step.

            positions (torch.Tensor): Position IDs used by attention.
            intermediate_tensors (IntermediateTensors | None): Optional intermediate tensors from pipeline stages.
            inputs_embeds (torch.Tensor | None): Optional external input embeddings (e.g. multimodal/deepstack
                scenarios).
            **model_kwargs (Any): Additional keyword arguments for the runnable.

        Returns:
            XliteForwardResult: Forward outputs from xlite graph or the original runnable implementation.
        """
        forward_context = get_forward_context()
        if self._qwen35_hybrid:
            result = (
                None
                if model_kwargs
                else self._qwen35_forward(forward_context, input_ids, positions, inputs_embeds, intermediate_tensors)
            )
            if result is not None:
                return result
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
        attn_metadata: Any = forward_context.attn_metadata
        attn_metadata = attn_metadata[0] if isinstance(attn_metadata, list) else attn_metadata
        attn_metadata = attn_metadata.get("model.layers.0.self_attn.attn", next(iter(attn_metadata.values()), None))
        # Since v0.28.0, batches are routed by token count (max across DP ranks) instead of the batch attention
        # state: batches within the preallocated budget run on xlite, others fall back to the native acl runnable.
        if (num_tokens := forward_context.max_tokens_across_dp) > self.max_tokens:
            if self.full_mode:
                logger.warning_once(
                    "xlite: token number exceeded expected limit (%d >%d), consider opening an issue at %s",
                    num_tokens,
                    self.max_tokens,
                    "https://atomgit.com/openeuler/GVirt",
                )
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        attn_metadata_router = AttnMetadataRouter(attn_metadata=attn_metadata, device="cpu")
        num_actual_tokens = attn_metadata_router.num_actual_tokens
        seq_lens = attn_metadata_router.seq_lens
        cum_query_lens = attn_metadata_router.cu_query_lens[-seq_lens.size(0) :]
        query_lens = torch.diff(cum_query_lens, prepend=seq_lens.new_zeros(1))
        cached_lens = torch.clamp(seq_lens - query_lens, min=0)
        block_tables = attn_metadata_router.block_tables
        num_reqs = max(1, (block_tables[:, 0] != 0).sum().item())  # number of non-empty requests

        xlite_attn_metadata = AttnMeta()
        xlite_attn_metadata.lens = query_lens[:num_reqs].tolist()
        xlite_attn_metadata.cached_lens = cached_lens[:num_reqs].tolist()
        xlite_attn_metadata.block_tables_cpu = block_tables[:num_reqs].tolist()
        xlite_attn_metadata.positions = torch.atleast_2d(positions)[:, :num_actual_tokens].contiguous()

        # under DP, `num_tokens` is the max number of tokens across all DP ranks for data alignment
        h = self.hidden_states[:num_tokens]
        stream = torch.npu.current_stream().npu_stream
        input_ids = cast(torch.Tensor, input_ids)
        if inputs_embeds is None or inputs_embeds.size(0) == 0:
            self.xlite_model.forward(
                rt=self.xlite_rt,
                input=input_ids[:num_actual_tokens],
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=self.freq_cis,
                output=h,
                curr_stream=stream,
            )
        else:
            inputs_embeds = inputs_embeds[:num_actual_tokens]
            deepstack_input_embeds = getattr(self.npu_runnable, "deepstack_input_embeds", [])
            xlite_deepstack_input_embeds = [
                deepstack_input[: inputs_embeds.size(0)] for deepstack_input in deepstack_input_embeds
            ]
            self.xlite_model.forward_with_inputs_embeds(
                rt=self.xlite_rt,
                input=inputs_embeds,
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=self.freq_cis,
                output=h,
                curr_stream=stream,
                deepstack_input=xlite_deepstack_input_embeds,
            )
            if deepstack_input_embeds and callable(
                _clear_stack := getattr(self.npu_runnable, "_clear_deepstack_input_embeds", None)
            ):
                _clear_stack(inputs_embeds.size(0))
        return h[:num_actual_tokens] if self.data_parallel_size == 1 else h[:num_tokens]
