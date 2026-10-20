# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""For cache mode none, Ascend must use upstream allocation counts exactly."""
import unittest

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import MambaSpec
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


class MambaManagerDelegationTest(unittest.TestCase):
    def make_manager(self, cls):
        spec = MambaSpec(block_size=128, shapes=((3, 64), (2, 128, 128)),
                         dtypes=(torch.bfloat16, torch.float32), mamba_cache_mode="none")
        return cls(spec, BlockPool(64, False, 128), enable_caching=False,
                   kv_cache_group_id=0, scheduler_block_size=128)

    def test_none_matches_upstream_at_prefill_decode_boundaries(self):
        upstream = AscendMambaManager.__bases__[0]
        for computed, scheduled, end in ((0, 1, 1), (0, 127, 127), (0, 128, 128),
                                          (0, 129, 129), (32, 33, 32),
                                          (127, 128, 127), (128, 129, 128)):
            with self.subTest(computed=computed, scheduled=scheduled):
                args = dict(request_id="request", num_tokens=scheduled,
                            new_computed_blocks=[], total_computed_tokens=computed,
                            num_local_computed_tokens=computed, num_tokens_main_model=scheduled,
                            apply_admission_cap=False, prefill_end=end)
                actual = self.make_manager(AscendMambaManager).get_num_blocks_to_allocate(**args)
                expected = self.make_manager(upstream).get_num_blocks_to_allocate(**args)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
