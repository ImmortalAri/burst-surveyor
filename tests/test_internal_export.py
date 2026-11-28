"""Regression checks for smooth-spectrum burst export on the original time grid."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from burstsurveyor.science.fit import _event, load_config
from burstsurveyor.science.model import pulse
from burstsurveyor.science.native import finalize
from burstsurveyor.science.objective import ReconstructionData, score


def two_components(scattered):
    channels, bins = 32, 96
    offsets = np.linspace(-0.00045, 0.00045, channels)
    grid = dict(res_time=0.001, res_freq=10.0, freqs_bin0=420.0, times_bin0=0.0,
                is_dedispersed=True)
    noise = np.random.default_rng(761).normal(0.0, 0.1, (channels, bins))
    event = _event(noise, {"grid": grid, "channel_time_offset_s": offsets}, load_config(), "bounded")
    x = [np.log(40), 38.5, np.log(0.55), 0.2, -0.5,
         np.log(70), 44.2, np.log(2.2), -0.3, 0.1, 0.7]
    if scattered:
        x.append(np.log(1.9))
    x = np.asarray(x)
    event.raw += pulse(x, 2, scattered, event.coords, np.arange(channels), np.arange(bins))[0]
    return ReconstructionData(event), x


def export_row(data, x, scattered, path, backend):
    row = dict(n=2, scatter=scattered, x=x.tolist())
    row["gain"], row["rms_gain"], _ = score(data, row)
    receipt = finalize(data, row, path, pulse_fn=pulse, source_sha256="a" * 64,
                       export_backend=backend)
    if receipt["status"] != "provisional":
        raise AssertionError(receipt["reasons"])
    return receipt


class InternalExportTests(unittest.TestCase):
    def test_independent_strength_adjustment_preserves_geometry_and_channel_times(self):
        with TemporaryDirectory() as tmp:
            for adjust in (False, True):
                with self.subTest(adjust=adjust):
                    data, truth = two_components(scattered=True)
                    incumbent = truth.copy()
                    if adjust:
                        incumbent[5] += np.log(0.6)
                    path = Path(tmp) / f"adjust-{adjust}.npz"
                    receipt = export_row(data, incumbent, True, path, "internal")
                    self.assertIs(receipt["challenger_selected"], adjust)
                    self.assertEqual(receipt["export_backend"], "internal")
                    self.assertEqual(receipt["native_model_passes"], 1 + int(adjust))
                    selected = np.asarray(receipt["selected"]["x"])
                    np.testing.assert_array_equal(np.delete(selected, [0, 5]),
                                                  np.delete(incumbent, [0, 5]))
                    np.testing.assert_allclose(np.exp(selected[[0, 5]] - truth[[0, 5]]),
                                               1.0, atol=0.015)
                    with np.load(path) as saved:
                        self.assertEqual(str(saved["version"]), receipt["version"])
                        np.testing.assert_allclose(saved["channel_time_offset_s"],
                                                   np.linspace(-0.00045, 0.00045, 32), atol=0)
                        signal = pulse(selected, 2, True, data.e.coords, data.indices, data.t)[0]
                        np.testing.assert_allclose(saved["prediction"][data.indices], signal,
                                                   rtol=1e-6, atol=1e-8)
                        error = data.e.raw[data.indices] - saved["prediction"][data.indices]
                        np.testing.assert_allclose(saved["baseline"][data.indices],
                                                   error.mean(axis=1), atol=1e-10)

    def test_stale_winner_score_refuses_export(self):
        data, x = two_components(scattered=False)
        row = dict(n=2, scatter=False, x=x.tolist())
        gain, rms_gain, _ = score(data, row)
        row.update(gain=gain * 1.02, rms_gain=rms_gain)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "stale.npz"
            receipt = finalize(data, row, path, pulse_fn=pulse, source_sha256="a" * 64,
                               export_backend="internal")
            self.assertEqual(receipt["status"], "refused")
            self.assertIn("replayed canonical gain differs from supplied winner", receipt["reasons"])
            self.assertFalse(path.exists())

    @unittest.skipUnless(importlib.util.find_spec("fitburst"), "optional FitBurst reference")
    def test_scattered_and_unscattered_exports_against_pinned_reference(self):
        with TemporaryDirectory() as tmp:
            for scatter in (False, True):
                with self.subTest(scatter=scatter):
                    data, x = two_components(scatter)
                    x[0] += np.log(0.75)
                    outputs = {}
                    for backend in ("internal", "fitburst"):
                        path = Path(tmp) / f"{backend}-{scatter}.npz"
                        receipt = export_row(data, x, scatter, path, backend)
                        self.assertTrue(receipt["challenger_selected"])
                        with np.load(path) as saved:
                            outputs[backend] = saved["prediction"][data.indices]
                    # Compare channel-mean-subtracted signals with the fit weights.
                    def weighted(signal):
                        return (signal - signal.mean(axis=1, keepdims=True)) / np.sqrt(
                            data.var[data.indices, None])
                    reference = weighted(outputs["fitburst"])
                    difference = weighted(outputs["internal"]) - reference
                    self.assertLess(np.linalg.norm(difference) / np.linalg.norm(reference), 0.01)
