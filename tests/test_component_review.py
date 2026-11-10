import unittest

import numpy as np

from burstsurveyor.component_review import component_review
from burstsurveyor.science.model import DISPERSION_SECONDS_MHZ2


class ComponentReviewTests(unittest.TestCase):
    def setUp(self):
        self.frequency = np.linspace(400, 800, 65)
        self.time = np.arange(1000)*0.00005

    def params(self, arrival=-0.00766771199983616, sigma=0.02968379356374577, dm=-0.016879369850614015):
        return dict(arrival_time=[arrival], burst_width=[sigma], dm=[dm],
                    dm_index=[-2], ref_freq=[600], scattering_timescale=[0.00023012200268946026],
                    spectral_index=[684.481369238154], spectral_running=[-1666.6666666555557])

    def test_crop_scale_pre_record_component_requests_review(self):
        report = component_review(self.params(), self.frequency, self.time)
        self.assertEqual(report["flags"], ["broad_edge_component"])
        self.assertEqual(report["review_components"], [1])
        self.assertGreater(report["components"][0]["sigma_fraction_of_record"], 0.5)
        self.assertNotIn("reject", report["action"])

    def test_broad_centered_pulse_is_not_automatically_flagged(self):
        report = component_review(self.params(arrival=0.025), self.frequency, self.time)
        self.assertEqual(report["flags"], [])

    def test_narrow_edge_truncation_is_not_the_broad_background_guard(self):
        report = component_review(self.params(sigma=0.001), self.frequency, self.time)
        self.assertEqual(report["flags"], [])
        self.assertTrue(report["components"][0]["all_gaussian_centers_outside_record"])

    def test_real_broad_truncated_signal_is_reviewed_not_rejected(self):
        report = component_review(self.params(arrival=0.055, sigma=0.020, dm=0), self.frequency, self.time)
        self.assertEqual(report["flags"], ["broad_edge_component"])
        self.assertEqual(report["action"], "review crop, off-burst structure and a simpler fit")

    def test_residual_dm_and_frequency_direction_are_used(self):
        # Reference center precedes the crop, but dispersion moves low channels in.
        p = self.params(arrival=-0.001, dm=1.0)
        report = component_review(p, self.frequency[::-1], self.time)
        self.assertEqual(report["flags"], [])
        expected = -0.001 + DISPERSION_SECONDS_MHZ2*(self.frequency**-2 - 600.0**-2)
        np.testing.assert_allclose(report["components"][0]["gaussian_arrival_range_ms"],
                                   [1000*expected.min(), 1000*expected.max()])

    def test_time_origin_and_parameter_lists_are_preserved(self):
        p = self.params(arrival=0.125)
        report = component_review(p, self.frequency, self.time+0.1)
        self.assertEqual(report["flags"], [])
        self.assertEqual(report["component_count"], 1)
        with self.assertRaises(ValueError):
            component_review(dict(p, burst_width=[.1,.2]), self.frequency, self.time)


if __name__ == "__main__":
    unittest.main()
