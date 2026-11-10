from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from burstsurveyor._storage import read, sha256
from burstsurveyor.review import _display_data, plot_event, plot_verified_event


class DisplayTests(unittest.TestCase):
    def test_weighted_profile_native_axis_direction_mask_and_partial_bins(self):
        nt = 451
        raw = np.tile(np.arange(nt, dtype=float), (5, 1))
        raw[3] *= 5
        baseline = np.array([2.0, 99.0, 99.0, 10.0, 99.0])
        variance = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        data = SimpleNamespace(
            e=SimpleNamespace(raw=raw), indices=np.array([0, 3]), var=variance
        )
        entry = {"grid": {"res_time": 0.001, "res_freq": -3.0, "freqs_bin0": 600.0}}
        signal = raw * 0.6
        view = _display_data(data, entry, signal, baseline)
        expected = ((raw[0] - 2.0) + (raw[3] - 10.0) / 4.0) / 1.25
        self.assertTrue(
            np.allclose(
                view["profile"][0, :-1], expected[:-1].reshape(-1, 2).mean(axis=1)
            )
        )
        self.assertAlmostEqual(view["profile"][0, -1], expected[-1])
        expected_model = (signal[0] + signal[3] / 4.0) / 1.25
        self.assertAlmostEqual(view["profile"][1, -1], expected_model[-1])
        self.assertTrue(np.isnan(view["images"][:, [1, 2, 4]]).all())
        self.assertTrue(np.all(np.diff(view["freq"]) < 0))
        self.assertEqual(view["time"][-1], 450.5)
        np.testing.assert_allclose(view["images"][0, 3, -1], (raw[3, -1] - 10.0) / 2.0)

    def test_exported_example_unresolved_and_source_binding(self):
        from burstsurveyor.batch import fit, verify

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = Path(__file__).resolve().parents[1] / "examples" / "two-pulse.npz"
            (root / "inputs").mkdir()
            (root / "inputs" / src.name).write_bytes(src.read_bytes())
            rng = np.random.default_rng(3434)
            with np.load(src, allow_pickle=False) as example:
                meta = example["metadata"]
            quiet = root / "inputs" / "quiet.npz"
            np.savez(
                quiet,
                data_full=rng.normal(size=(64, 144)).astype("float32"),
                metadata=meta,
            )
            run = root / "batch"
            fit(root / "inputs", run, workers=1)
            events = {row["input_name"]: row for row in verify(run)["events"]}
            for name, row in events.items():
                dest = root / (name + ".png")
                try:
                    rec = plot_verified_event(run, row, root / "inputs" / name, dest)
                except RuntimeError as exc:
                    if "available memory" not in str(exc):
                        raise
                    self.skipTest("Insufficient available RAM for guarded native display test")
                self.assertEqual(rec["source_sha256"], row["source_sha256"])
                self.assertEqual(rec["figure_sha256"], sha256(dest))
                self.assertEqual(dest.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                if name == src.name:
                    native = read(run / row["native_receipt"])
                    self.assertTrue(rec["has_model"])
                    self.assertEqual(
                        rec["prediction_sha256"], native["prediction_sha256"]
                    )
                else:
                    self.assertFalse(rec["has_model"])
                    self.assertIsNone(rec["prediction_sha256"])
                with self.assertRaises(ValueError):
                    plot_event(run, row["id"], root / "inputs" / name, dest)
            with self.assertRaisesRegex(ValueError, "raw NPZ differs"):
                plot_event(run, events[src.name]["id"], quiet, root / "wrong.png")
            self.assertFalse((root / "wrong.png").exists())


if __name__ == "__main__":
    unittest.main()
