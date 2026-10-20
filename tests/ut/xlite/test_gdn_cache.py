# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native page descriptors and metadata selection, without loading the runner."""
import ast
import importlib.util
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

_path = Path(__file__).resolve().parents[3] / "vllm_ascend/xlite/gdn_cache.py"
_spec = importlib.util.spec_from_file_location("gdn_cache_contract", _path)
cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache)


class FakeCache:
    def __init__(self, shape, strides, dtype=torch.bfloat16):
        self.shape = shape
        self.ndim = len(shape)
        self._strides = strides
        self.dtype = dtype
        self.device = NS(type="npu", index=0)

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]


class NativeCacheContractTest(unittest.TestCase):
    def setUp(self):
        self.conv = FakeCache((8, 3, 64), (256, 64, 1))
        self.ssm = FakeCache((8, 2, 64, 16), (2560, 1024, 16, 1), torch.float32)
        self.args = dict(channels=64, heads=2, key_dim=16, value_dim=64,
                         width=4, dtype=torch.bfloat16)

    def test_padded_pages_and_mixed_dtype_supported(self):
        self.assertIsNone(cache.cache_support_reason(self.conv, self.ssm, **self.args))

    def test_partial_vector_rows_rejected(self):
        self.args["value_dim"] = 96
        self.assertIsNotNone(cache.cache_support_reason(self.conv, self.ssm, **self.args))

    def test_ds_layout_rejected(self):
        self.conv.shape = (8, 64, 3)
        self.assertIsNotNone(cache.cache_support_reason(self.conv, self.ssm, **self.args))

    def test_overlapping_pages_rejected(self):
        self.ssm._strides = (2047, 1024, 16, 1)
        self.assertIsNotNone(cache.cache_support_reason(self.conv, self.ssm, **self.args))

    def test_mismatched_page_pools_rejected(self):
        self.ssm.shape = (9, 2, 64, 16)
        self.assertIsNotNone(cache.cache_support_reason(self.conv, self.ssm, **self.args))

    def test_adapter_references_layer_bound_views(self):
        # Load the actual method in isolation from the runner's device dependencies.
        source = _path.with_name("xlite.py").read_text()
        adapter_class = next(node for node in ast.parse(source).body
                             if isinstance(node, ast.ClassDef) and node.name == "Qwen35XliteModel")
        method = next(node for node in adapter_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == "extract_kv_cache")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(_path), "exec"), namespace)
        raw = torch.zeros(8 * 512)
        k = raw.as_strided((8, 2, 2, 64), (512, 128, 64, 1))
        v = raw.as_strided(k.shape, k.stride(), 256)
        full = NS(full=True, self_attn=NS(attn=NS(kv_cache=(k, v))))
        gdn = NS(full=False)
        adapter = NS(xlite_config=NS(n_layers=2),
                     _get_layers_and_model_prefix=lambda: ([gdn, full], "model"),
                     _is_full_attention=lambda layer: layer.full)
        dummy = torch.empty(0)
        with patch.dict(sys.modules, {"vllm_ascend.xlite.utils": NS(_DUMMY_TENSOR=dummy)}):
            actual = MethodType(namespace["extract_kv_cache"], adapter)([(dummy, dummy), (k, v)])
        self.assertIs(actual[0][0], dummy)
        self.assertIs(actual[1][0], k)
        self.assertIs(actual[1][1], v)
        self.assertEqual(actual[1][1].storage_offset(), 256)
        self.assertEqual(actual[1][0].stride(0), 512)

    def metadata(self):
        conv = torch.tensor([[7, 99], [2, 98], [5, 97]], dtype=torch.int32)
        ssm = torch.tensor([4, 6, 1], dtype=torch.int32)
        return NS(num_prefills=0, num_spec_decodes=0, num_decodes=3,
                  num_actual_tokens=3, spec_sequence_masks=None,
                  non_spec_query_start_loc=torch.arange(4, dtype=torch.int32),
                  non_spec_decode_metadata=NS(causal_conv1d=NS(cache_indices=conv)),
                  non_spec_state_indices_tensor=ssm)

    def test_conv_and_ssm_use_separate_native_indices(self):
        indices = cache.decode_indices(self.metadata(), 3, torch.device("cpu"))
        self.assertEqual(indices[0].tolist(), [7, 2, 5])
        self.assertEqual(indices[1].tolist(), [4, 6, 1])
        self.assertTrue(indices[0].is_contiguous())
        self.assertTrue(indices[1].is_contiguous())

    def test_reorder_uses_current_metadata(self):
        meta = self.metadata()
        meta.non_spec_state_indices_tensor = torch.tensor([1, 4, 6], dtype=torch.int32)
        self.assertEqual(cache.decode_indices(meta, 3, torch.device("cpu"))[1].tolist(), [1, 4, 6])

    def test_prefill_speculation_and_wrong_dtype_rejected(self):
        for name, value in (("num_prefills", 1), ("num_spec_decodes", 1),
                            ("spec_sequence_masks", torch.tensor([True])),
                            ("non_spec_state_indices_tensor", torch.ones(3, dtype=torch.int64))):
            meta = self.metadata()
            setattr(meta, name, value)
            with self.subTest(name=name):
                self.assertIsNone(cache.decode_indices(meta, 3, torch.device("cpu")))

    def test_short_metadata_and_ubatch_rejected(self):
        self.assertIsNone(cache.decode_indices(self.metadata(), 4, torch.device("cpu")))
        self.assertIsNone(cache.layer_metadata([{"layer": self.metadata()}], "layer"))


class PrefillContractTest(unittest.TestCase):
    def metadata(self):
        return NS(num_prefills=2, num_decodes=1, num_spec_decodes=0,
                  num_actual_tokens=8, spec_sequence_masks=None,
                  non_spec_state_indices_tensor=torch.tensor([6, 3, 1], dtype=torch.int32),
                  has_initial_state=torch.tensor([True, False, True]),
                  non_spec_prefill_metadata=NS(causal_conv1d=NS(
                      cache_indices=torch.tensor([[5], [2], [7]], dtype=torch.int32),
                      query_start_loc=torch.tensor([0, 1, 4, 8], dtype=torch.int32),
                      initial_state_mode=torch.tensor([True, False, True]))))

    def test_mixed_prefill_uses_native_indices_and_initial_flags(self):
        result = cache.prefill_metadata(self.metadata(), 3, 8, torch.device("cpu"))
        self.assertIsNotNone(result)
        self.assertEqual([t.tolist() for t in result],
                         [[5, 2, 7], [6, 3, 1], [0, 1, 4, 8], [1, 0, 1], [1, 0, 1]])
        self.assertTrue(all(t.dtype == torch.int32 and t.is_contiguous() for t in result))

    def test_incomplete_or_spec_metadata_rejected_before_execution(self):
        for name, value in (("has_initial_state", None), ("num_actual_tokens", 9),
                            ("num_spec_decodes", 1), ("num_prefills", 0),
                            ("non_spec_state_indices_tensor", torch.zeros(3, dtype=torch.int64))):
            meta = self.metadata()
            setattr(meta, name, value)
            self.assertIsNone(cache.prefill_metadata(meta, 3, 8, torch.device("cpu")))
        meta = self.metadata()
        meta.non_spec_prefill_metadata.causal_conv1d.query_start_loc = torch.zeros(3, dtype=torch.int32)
        self.assertIsNone(cache.prefill_metadata(meta, 3, 8, torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
