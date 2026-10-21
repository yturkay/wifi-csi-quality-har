import unittest

import numpy as np

from wifi_csi_quality.corruptions import (
    ImpairmentConfig,
    forward_fill,
    impair_regular_tensor,
    impair_regular_tensor_with_quality,
    recover_regular_grid,
    simulate_impairments,
)
from wifi_csi_quality.readers import CSISample


class CorruptionTests(unittest.TestCase):
    def setUp(self) -> None:
        values = np.arange(60, dtype=np.float32).reshape(10, 3, 2)
        timestamps = np.arange(10, dtype=np.float64) * 0.01
        self.sample = CSISample(values, timestamps, {"sample": "synthetic"})

    def test_seed_is_reproducible(self) -> None:
        config = ImpairmentConfig(
            random_loss_probability=0.2,
            burst_start_probability=0.1,
            burst_length=2,
            jitter_std_fraction_of_median_interval=0.1,
            stale_start_probability=0.1,
            stale_length=2,
            seed=42,
        )
        first = simulate_impairments(self.sample, config)
        second = simulate_impairments(self.sample, config)
        np.testing.assert_equal(first.values, second.values)
        np.testing.assert_equal(first.quality_features, second.quality_features)

    def test_full_random_loss_and_forward_fill(self) -> None:
        view = simulate_impairments(
            self.sample, ImpairmentConfig(random_loss_probability=1.0, seed=1)
        )
        self.assertFalse(view.observed.any())
        self.assertTrue(np.isnan(view.values).all())
        self.assertTrue(np.equal(forward_fill(view), 0.0).all())

    def test_stale_samples_have_positive_age(self) -> None:
        view = simulate_impairments(
            self.sample,
            ImpairmentConfig(stale_start_probability=1.0, stale_length=2, seed=7),
        )
        self.assertTrue((view.age_seconds[view.stale] > 0).all())
        self.assertEqual(view.quality_features.shape, (10, 4))

    def test_clean_regular_tensor_is_an_exact_copy(self) -> None:
        values = np.arange(40, dtype=np.float32).reshape(10, 4)
        recovered, diagnostics = impair_regular_tensor(values, ImpairmentConfig())
        np.testing.assert_array_equal(recovered, values)
        self.assertIsNot(recovered, values)
        self.assertEqual(diagnostics["observed_fraction"], 1.0)

    def test_full_loss_regular_tensor_uses_fill_value(self) -> None:
        values = np.arange(40, dtype=np.float32).reshape(10, 4)
        recovered, diagnostics = impair_regular_tensor(
            values,
            ImpairmentConfig(random_loss_probability=1.0, seed=9),
            fill_value=-2.0,
        )
        np.testing.assert_array_equal(recovered, np.full_like(values, -2.0))
        self.assertEqual(diagnostics["observed_fraction"], 0.0)

    def test_irregular_view_returns_to_target_grid(self) -> None:
        view = simulate_impairments(
            self.sample,
            ImpairmentConfig(jitter_std_fraction_of_median_interval=0.2, seed=5),
        )
        recovered = recover_regular_grid(view, self.sample.timestamps)
        self.assertEqual(recovered.shape, self.sample.values.shape)
        self.assertTrue(np.isfinite(recovered).all())

    def test_quality_channels_are_aligned_to_regular_tensor(self) -> None:
        values = np.arange(40, dtype=np.float32).reshape(10, 4)
        recovered, quality, diagnostics = impair_regular_tensor_with_quality(
            values,
            ImpairmentConfig(
                random_loss_probability=0.2,
                jitter_std_fraction_of_median_interval=0.1,
                stale_start_probability=0.1,
                stale_length=2,
                seed=42,
            ),
        )
        self.assertEqual(recovered.shape, values.shape)
        self.assertEqual(quality.shape, (10, 4))
        self.assertTrue(np.isfinite(quality).all())
        self.assertTrue(((quality[:, :2] >= 0) & (quality[:, :2] <= 1)).all())
        self.assertGreaterEqual(diagnostics["observed_fraction"], 0.0)

    def test_clean_quality_channels_have_nominal_values(self) -> None:
        values = np.arange(40, dtype=np.float32).reshape(10, 4)
        recovered, quality, _ = impair_regular_tensor_with_quality(
            values, ImpairmentConfig()
        )
        np.testing.assert_array_equal(recovered, values)
        np.testing.assert_array_equal(quality[:, 0], 1.0)
        np.testing.assert_array_equal(quality[:, 1], 0.0)
        np.testing.assert_array_equal(quality[:, 2], 1.0)
        np.testing.assert_array_equal(quality[:, 3], 0.0)

