from types import SimpleNamespace
import unittest

import numpy as np

from burstsurveyor.spectral_review import spectral_review


class SpectralReviewTests(unittest.TestCase):
    def data(self, raw, variance=None):
        nf = len(raw)
        var = np.ones(nf) if variance is None else variance
        return SimpleNamespace(e=SimpleNamespace(raw=raw), indices=np.arange(nf),
                               var=var, rms_var=var.copy())

    def setUp(self):
        self.rng = np.random.default_rng(401)
        self.t = np.arange(160)
        self.pulse = 6 * np.exp(-0.5 * ((self.t - 45) / 5)**2)
        self.model = np.tile(self.pulse, (64, 1))

    def test_repeatable_spectral_ripples_are_not_new_pulses(self):
        scales = 1 + 0.65 * np.sin(np.arange(64) * 0.8)
        raw = scales[:, None] * self.model + self.rng.normal(size=self.model.shape)
        saved = self.model.copy()
        result = spectral_review(self.data(raw), self.model)
        self.assertIn("spectral_model_limited", result["flags"])
        self.assertEqual(result["time_split_status"], "replicated")
        np.testing.assert_array_equal(self.model, saved)

    def test_uniform_amplitude_error_is_not_spectral_structure(self):
        raw = 1.7 * self.model + self.rng.normal(size=self.model.shape)
        result = spectral_review(self.data(raw), self.model)
        self.assertNotIn("spectral_model_limited", result["flags"])

    def test_distinct_later_pulse_is_not_spectral_structure(self):
        extra = 7 * np.exp(-0.5 * ((self.t - 115) / 4)**2)
        raw = self.model + extra[None, :] + self.rng.normal(size=self.model.shape)
        result = spectral_review(self.data(raw), self.model)
        self.assertNotIn("spectral_model_limited", result["flags"])

    def test_noise_does_not_trigger_spectral_refit(self):
        raw = self.model + self.rng.normal(size=self.model.shape)
        result = spectral_review(self.data(raw), self.model)
        self.assertEqual(result["flags"], [])

    def test_nonrepeating_frequency_time_artifact_is_not_validated(self):
        scales = 1 + 0.7 * np.sin(np.arange(64) * 0.8)
        raw = self.model.copy()
        raw[:, ::2] *= scales[:, None]
        raw += self.rng.normal(size=raw.shape)
        result = spectral_review(self.data(raw), self.model)
        self.assertIn("spectral_model_limited", result["flags"])
        self.assertIn("spectral_amplitude_not_replicated", result["flags"])
        self.assertEqual(result["time_split_status"], "not_replicated")

    def test_channel_baselines_and_negative_frequency_order(self):
        scales = 1 + 0.65 * np.sin(np.arange(64) * 0.8)
        raw = scales[:, None] * self.model + self.rng.normal(size=self.model.shape)
        a = spectral_review(self.data(raw), self.model)
        shifted = raw + np.linspace(-10, 20, 64)[:, None]
        b = spectral_review(self.data(shifted[::-1]), self.model[::-1])
        self.assertEqual(a["flags"], b["flags"])
        np.testing.assert_allclose(a["full_data_gain_change"], b["full_data_gain_change"])

    def test_broadcasting_and_invalid_noise_are_rejected(self):
        with self.assertRaises(ValueError):
            spectral_review(self.data(self.model), self.model[:1])
        with self.assertRaises(ValueError):
            spectral_review(self.data(self.model, np.zeros(64)), self.model)


if __name__ == "__main__":
    unittest.main()
