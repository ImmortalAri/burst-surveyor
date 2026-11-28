"""Exercise routed rescues through the public CLI and check route selection."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil


def cli(*args, success=True):
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    completed = subprocess.run(
        [sys.executable, "-m", "burstsurveyor", *map(str, args)],
        capture_output=True, timeout=180, env=env,
    )
    if success:
        if completed.returncode:
            raise AssertionError(completed.stderr.decode(errors="replace"))
        return json.loads(completed.stdout)
    if completed.returncode == 0:
        raise AssertionError("Expected the CLI to reject this request")
    return completed.stderr.decode(errors="replace")


class RoutedRescueCliTests(unittest.TestCase):
    def setUp(self):
        if psutil.virtual_memory().available < 1024 * 2**20:
            self.skipTest("Insufficient free memory for the CLI's guarded numerical worker")

    def test_crowded_example_recovers_omitted_injected_peaks(self):
        source = Path(__file__).resolve().parents[1] / "examples/route-rescue-crowded.npz"
        injected_bins = (43, 51, 60, 71, 89, 119)

        def covers(fitted_peaks, injected):
            return any(abs(peak - injected) <= 3 for peak in fitted_peaks)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "bounded"
            alternate = root / "routed"
            cli("fit", source, baseline, "--workers", 1)
            seal = (baseline / "seal.json").read_bytes()

            cli("route-rescue", baseline, source, alternate)
            report = json.loads((alternate / "result.json").read_bytes())
            self.assertEqual(report["route"], "joint")
            self.assertEqual(report["disposition"], "candidate")
            self.assertEqual(report["baseline_components"], 5)
            self.assertEqual(report["candidate_components"], len(injected_bins))
            for feature in injected_bins[:2]:
                self.assertFalse(covers(report["baseline_coverage"]["fitted_peaks"], feature))
            for feature in injected_bins:
                with self.subTest(injected_bin=feature):
                    self.assertTrue(covers(report["native_coverage"]["fitted_peaks"], feature))
            folds = report["channel_folds"]
            self.assertEqual(len(folds), 2)
            for fold in folds:
                self.assertTrue(fold["passed"])
                self.assertGreater(min(fold["error_reduction_pct"]), 0)
            self.assertEqual((baseline / "seal.json").read_bytes(), seal)
            cli("verify", baseline, "--route-rescue", alternate)

    def test_clean_example_retains_fit_and_alternate_receipt_is_verified(self):
        fixture = Path(__file__).resolve().parents[1] / "examples/two-pulse.npz"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / fixture.name
            shutil.copyfile(fixture, source)
            baseline = root / "bounded"
            alternate = root / "routed"
            cli("fit", source, baseline, "--workers", 1)
            seal = (baseline / "seal.json").read_bytes()

            cli("route-rescue", baseline, source, alternate)
            result_path = alternate / "result.json"
            result = json.loads(result_path.read_bytes())
            self.assertEqual(result["disposition"], "retained")
            self.assertEqual(result["baseline_components"], 2)
            self.assertFalse((alternate / "prediction.npz").exists())
            self.assertEqual((baseline / "seal.json").read_bytes(), seal)
            cli("verify", baseline, "--route-rescue", alternate)

            changed = root / "changed"
            changed.mkdir()
            wrong = changed / source.name
            wrong.write_bytes(source.read_bytes() + b"changed bytes")
            rejected = root / "wrong-input"
            cli("route-rescue", baseline, wrong, rejected, success=False)
            self.assertFalse((rejected / "seal.json").exists())
            self.assertEqual((baseline / "seal.json").read_bytes(), seal)

            original = result_path.read_bytes()
            result_path.write_bytes(original + b"\n")
            cli("verify", baseline, "--route-rescue", alternate, success=False)
            cli("verify", baseline)


class RoutedRescuePolicyTests(unittest.TestCase):
    def test_repeated_residual_with_multiple_omissions_selects_joint_route(self):
        from burstsurveyor import rescue_router

        proposal = {"time_bins": [17, 25]}
        features = [{"bin": k} for k in (5, 20, 32)]
        row = {"components": 1}
        before = {"missing": [20, 32]}
        data = SimpleNamespace(c={"max_components": 7})
        folds = ([0], [8])
        signal = object()
        with (
            patch.object(rescue_router, "_discovery", return_value=proposal),
            patch.object(rescue_router, "_evidence", side_effect=[8.0, 10.0]),
            patch.object(rescue_router, "_band_check") as band_check,
        ):
            route, _, selected, scores, _ = rescue_router.choose_route(
                data, row, signal, features, before, folds
            )
        self.assertEqual(route, "joint")
        self.assertEqual(selected, proposal)
        self.assertEqual(scores, [8.0, 10.0])
        band_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
