"""Known-signal controls for the developmental off-burst neighbor screen."""

import unittest

import numpy as np

from burstsurveyor.science.channel_mask import screen_mask_neighbors


class ChannelMaskTests(unittest.TestCase):
    def raw(self, channels=128):
        return np.random.default_rng(171).normal(size=(channels, 320))

    def drift(self, raw, channel):
        raw[channel] = np.linspace(4, -4, raw.shape[1]) + 0.15 * raw[channel]

    def test_measured_drift_excluded_but_quiet_neighbor_kept_without_mutation(self):
        raw = self.raw()
        raw[40] = 0
        self.drift(raw, 39)
        saved = raw.copy()
        ids, receipt = screen_mask_neighbors(raw, np.arange(320))
        np.testing.assert_array_equal(ids, [39])
        self.assertEqual(receipt['status'], 'applied')
        self.assertNotIn(41, ids)
        np.testing.assert_array_equal(raw, saved)

    def test_narrowband_pulse_is_retained_even_if_offburst_detection_misses_it(self):
        for centre in (10, 145, 303):
            raw = self.raw()
            raw[40] = 0
            raw[39] += 35 * np.exp(-0.5 * ((np.arange(320) - centre) / 4) ** 2)
            off = np.flatnonzero(abs(np.arange(320) - centre) > 20)
            for selected in (np.arange(320), off):
                ids, receipt = screen_mask_neighbors(raw, selected)
                self.assertEqual(ids.tolist(), [], (centre, receipt))
                self.assertEqual(receipt['status'], 'unchanged')

    def test_training_omissions_cannot_become_mask_gaps(self):
        raw = self.raw()
        self.drift(raw, 39)
        ids, receipt = screen_mask_neighbors(raw, np.arange(320))
        self.assertEqual(ids.tolist(), [])
        self.assertEqual(receipt['candidate_channels'], 0)

    def test_no_recursive_dilation(self):
        raw = self.raw()
        raw[40] = 0
        self.drift(raw, 42)
        self.drift(raw, 44)
        ids, _ = screen_mask_neighbors(raw, np.arange(320))
        np.testing.assert_array_equal(ids, [42])

    def test_uncertain_insufficient_or_disconnected_offburst_is_noop(self):
        raw = self.raw()
        raw[40] = 0
        self.drift(raw, 39)
        ids, receipt = screen_mask_neighbors(raw, np.arange(320), uncertain=True)
        self.assertEqual(ids.tolist(), [])
        self.assertEqual(receipt['status'], 'skipped_uncertain_offburst')
        for off in (np.arange(30), np.arange(0, 320, 2), np.arange(100, 260)):
            ids, receipt = screen_mask_neighbors(raw, off)
            self.assertEqual(ids.tolist(), [])
            self.assertEqual(receipt['status'], 'skipped_insufficient_offburst')

    def test_excessive_removal_rejects_whole_pass(self):
        raw = self.raw()
        raw[40:50:2] = 0
        for channel in (39, 41, 43, 45, 47, 49):
            self.drift(raw, channel)
        ids, receipt = screen_mask_neighbors(raw, np.arange(320))
        self.assertEqual(ids.tolist(), [])
        self.assertEqual(receipt['status'], 'skipped_safety_cap')
        self.assertGreater(len(receipt['proposed_channels']), receipt['safety_cap'])

    def test_reversed_frequency_order_preserves_decision(self):
        raw = self.raw()
        raw[40] = 0
        self.drift(raw, 39)
        forward, _ = screen_mask_neighbors(raw, np.arange(320))
        backward, _ = screen_mask_neighbors(raw[::-1], np.arange(320))
        np.testing.assert_array_equal(np.sort(127 - backward), forward)

    def test_invalid_time_indices_are_rejected(self):
        raw = self.raw()
        for off in ([0, 0], [-1, 0], [1.5, 2.5], [0, 320], [4, 3]):
            with self.assertRaises(ValueError):
                screen_mask_neighbors(raw, off)


if __name__ == '__main__':
    unittest.main()
