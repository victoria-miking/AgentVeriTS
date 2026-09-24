import unittest

import numpy as np
import torch

from agentverits.visual.encoder import _make_mask, _pool_tokens
from agentverits.visual.screening import _harmonic_aggregation, _aligned_scores, detection_intervals
from agentverits.types import Interval


class NumericsTest(unittest.TestCase):
    def test_zero_based_patch_mapping_preserves_first_and_last_patch(self):
        # A 1x1 mask should be the identity, including patch index zero.
        mask = _make_mask(32, 16, 16)
        result = _harmonic_aggregation((1, 2, 2), torch.tensor([[1., 2., 3., 4.]]), mask)
        torch.testing.assert_close(result, torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.float64))

    def test_harmonic_aggregation_matches_membership_definition(self):
        mask = _make_mask(48, 16, 32)
        scores = torch.tensor([[1., 2., 4., 8.], [3., 4., 5., 6.]])
        result = _harmonic_aggregation((2, 3, 3), scores, mask).reshape(2, -1)
        for patch in range(9):
            selected = (mask == patch).any(dim=0)
            expected = selected.sum() / (1.0 / scores[:, selected].double()).sum(dim=1)
            torch.testing.assert_close(result[:, patch], expected)

    def test_vectorized_pooling_preserves_token_means(self):
        tokens = torch.arange(2 * 9 * 4).reshape(2, 9, 4).float()
        mask = _make_mask(48, 16, 32)
        expected = torch.stack([tokens[:, ids].mean(dim=1) for ids in mask.T], dim=1)
        torch.testing.assert_close(_pool_tokens(tokens, mask), expected)

    def test_overlap_is_averaged_before_top_fraction(self):
        # At the overlap, complementary visual rows average to 0.5; reducing each
        # window first would incorrectly return 1.0.
        maps = torch.tensor([[[1., 1.], [0., 0.]], [[0., 0.], [1., 1.]]])
        result = _aligned_scores(maps, np.array([0, 1]), 3, 2, 2, 0.5, 1)
        np.testing.assert_allclose(result, [1., 0.5, 1.])

    def test_irregular_tail_is_mapped_to_actual_positions(self):
        maps = torch.tensor([[[1., 1.], [1., 1.]], [[3., 3.], [3., 3.]]])
        result = _aligned_scores(maps, np.array([0, 3]), 7, 4, 2, 0.5, 2)
        np.testing.assert_allclose(result, [1, 1, 1, 2, 3, 3, 3])

    def test_threshold_and_inclusive_endpoints(self):
        intervals, threshold, _ = detection_intervals(np.array([0., 0., 10., 10.]), 0.1, False)
        self.assertAlmostEqual(threshold, 11.407757827723003)
        self.assertEqual(intervals, [])
        intervals, _, _ = detection_intervals(np.array([0., 0., 10., 10.]), 0.5, False)
        self.assertEqual(intervals, [Interval(2, 3)])


if __name__ == "__main__":
    unittest.main()
