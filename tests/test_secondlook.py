"""Known-signal and integrity checks for the experimental temporal second look."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import psutil


GRID = {
    "res_time": 0.0005,
    "res_freq": 6.25,
    "freqs_bin0": 400.0,
    "times_bin0": 0.0,
    "is_dedispersed": True,
}


def synthetic(case):
    """Return observed data and the noiseless injected signal on the same native grid."""
    channels, bins = 64, 300
    freq = GRID["freqs_bin0"] + (np.arange(channels) + 0.5) * GRID["res_freq"]
    time = np.arange(bins)
    seeds = {
        "quiet": 7944,
        "single": 8103,
        "separated": 7348,
        "patchy": 7981,
        "seven": 7664,
    }
    rng = np.random.default_rng(seeds[case])
    raw = rng.normal(0, 1, (channels, bins))
    signal = np.zeros_like(raw)
    if case == "seven":
        peaks = [
            (c, f, 8 - 0.55 * i, 2.2, 48)
            for i, (c, f) in enumerate(
                zip(
                    (33, 72, 114, 153, 191, 234, 270),
                    (430, 510, 580, 660, 715, 470, 625),
                )
            )
        ]
    elif case == "quiet":
        peaks = []
    elif case == "separated":
        peaks = [(65, 525, 7, 2.8, 55), (195, 685, 5, 3.7, 52)]
    else:
        peaks = [(114, 580, 6.5, 3.0, 85)]
    for center, pivot, height, width, bandwidth in peaks:
        spectrum = np.exp(-0.5 * ((freq - pivot) / bandwidth) ** 2)
        if case == "patchy":
            spectrum *= 0.2 + 0.8 * np.cos((freq - 400) / 18) ** 2
        signal += (
            height * spectrum[:, None] * np.exp(-0.5 * ((time - center) / width) ** 2)
        )
    return (raw + signal).astype(np.float32), signal


def write_input(path, case):
    raw, truth = synthetic(case)
    np.savez_compressed(path, data_full=raw, metadata=np.array(json.dumps(GRID)))
    return truth


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cli(*args, success=True):
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "burstsurveyor", *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    if success:
        if proc.returncode:
            raise AssertionError(proc.stderr or proc.stdout)
        return json.loads(proc.stdout)
    if proc.returncode == 0:
        raise AssertionError("Expected the CLI to reject this input")
    return proc.stderr


def truth_error(artifact, truth):
    """Signal-only squared error with a profiled constant per channel; same units for all methods."""
    if artifact is None:
        model = np.zeros_like(truth)
    else:
        with np.load(artifact, allow_pickle=False) as z:
            model = z["prediction"].astype(float)
    error = model - truth
    error -= error.mean(axis=1, keepdims=True)
    return float(np.mean(error**2))


class ResidualDiscoveryTests(unittest.TestCase):
    @staticmethod
    def data(raw):
        ids = np.arange(len(raw))
        event = SimpleNamespace(
            raw=raw, length=raw.shape[1],
            tiles=lambda rows: (rows[i:i + 8] for i in range(0, len(rows), 8)),
        )
        return SimpleNamespace(e=event, base=raw.mean(axis=1), var=np.ones(len(raw)), indices=ids)

    def test_broad_weak_pulse_on_fine_grid_replicates_on_other_channels(self):
        from burstsurveyor.secondlook import _discovery, _evidence

        rng = np.random.default_rng(78219)
        time = np.arange(32768)
        raw = rng.normal(size=(64, len(time))).astype(np.float32)
        raw += .055 * np.exp(-.5 * ((time - 12000) / 700)**2)
        data = self.data(raw)
        event = dict(id="broad", source_sha256="0" * 64)
        packet = _discovery(data, data.indices[::2], None, event)
        self.assertIsNotNone(packet)
        first, last = packet["time_bins"]
        self.assertGreater(last - first, 64)
        self.assertLess(first, 12000)
        self.assertGreater(last, 12000)
        self.assertGreater(_evidence(data, data.indices[1::2], None, packet), 3)

    def test_spectral_amplitude_mismatch_is_not_a_new_temporal_pulse(self):
        from burstsurveyor.secondlook import _discovery

        rng = np.random.default_rng(9216)
        time = np.arange(2048)
        pulse = np.exp(-.5 * ((time - 900) / 25)**2)
        model = np.broadcast_to(2 * pulse, (64, len(time))).copy()
        spectrum = 1 + .7 * np.sin(np.arange(64) / 4)
        raw = (spectrum[:, None] * model + rng.normal(size=model.shape)).astype(np.float32)
        data = self.data(raw)
        event = dict(id="spectral", source_sha256="0" * 64)
        self.assertIsNone(_discovery(data, data.indices[::2], model, event))

    def test_missing_pulse_survives_spectral_amplitude_projection(self):
        from burstsurveyor.secondlook import _discovery, _evidence

        rng = np.random.default_rng(44319)
        time = np.arange(4096)
        model = np.broadcast_to(3 * np.exp(-.5 * ((time - 900) / 30)**2), (64, len(time))).copy()
        scale = 1 + .5 * np.cos(np.arange(64) / 3)
        raw = scale[:, None] * model + rng.normal(size=model.shape)
        raw[16:48] += .5 * np.exp(-.5 * ((time - 2800) / 150)**2)
        data = self.data(raw)
        event = dict(id="missing", source_sha256="0" * 64)
        packet = _discovery(data, data.indices[::2], model, event)
        self.assertIsNotNone(packet)
        self.assertLess(packet["time_bins"][0], 2800)
        self.assertGreater(packet["time_bins"][1], 2800)
        self.assertGreater(_evidence(data, data.indices[1::2], model, packet), 3)

    def test_large_native_grid_gets_bounded_work_without_more_iterations(self):
        from burstsurveyor.secondlook import _fit_direction, POLICY

        packet = dict(time_bins=[80, 100], channel_bins=[0, 16])
        training = SimpleNamespace(component_seed=lambda box: [0., 90., 2., 0., -1.])
        granted = []
        with patch("burstsurveyor.science.objective.ReconstructionData", return_value=training), \
             patch("burstsurveyor.science.engine.optimizer", side_effect=lambda *a: (a[-2], a[-1])), \
             patch("burstsurveyor.secondlook.time.monotonic", return_value=100):
            for bins in (300, 25389, 10**7):
                data = SimpleNamespace(e=SimpleNamespace(length=bins, group_size=4))
                iterations, deadline = _fit_direction(data, np.arange(16), None, packet, 9.)
                self.assertEqual(iterations, POLICY["optimizer_iterations"])
                granted.append(deadline - 100)
        self.assertEqual(granted[0], 6.)
        self.assertGreater(granted[1], granted[0])
        self.assertLessEqual(max(granted), 48.)


class SecondLookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if psutil.virtual_memory().available < 1024 * 2**20:
            raise unittest.SkipTest(
                "Insufficient free memory for the CLI's guarded numerical worker"
            )
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cls.inputs = cls.root / "raw"
        cls.inputs.mkdir()
        cls.truth = write_input(cls.inputs / "seven.npz", "seven")
        write_input(cls.inputs / "quiet.npz", "quiet")
        cls.run_dir = cls.root / "fitted"
        cli(
            "fit",
            cls.inputs,
            cls.run_dir,
            "--workers",
            1,
            "--second-look",
            "--limit",
            4,
        )
        cls.extra = cls.root / "fitted.second-look"
        cls.events = {
            e["input_name"]: e for e in read(cls.run_dir / "results.json")["events"]
        }
        cls.original_seal = digest(cls.run_dir / "seal.json")

    def test_separated_island_recovers_signal_and_preserves_baseline(self):
        event = self.events["seven.npz"]
        decision = read(self.extra / event["id"] / "decision.json")
        self.assertEqual(decision["status"], "provisional_challenger")
        self.assertEqual(decision["native"]["selected"]["n"], event["components"] + 1)
        self.assertTrue(
            all(
                v > u
                for u, v in zip(
                    decision["official_gain"]["incumbent"],
                    decision["official_gain"]["proposed"],
                )
            )
        )
        self.assertTrue(
            all(
                v > u
                for u, v in zip(
                    decision["official_channel_check"]["incumbent"],
                    decision["official_channel_check"]["proposed"],
                )
            )
        )
        challenger = self.extra / event["id"] / decision["prediction"]
        self.assertEqual(digest(challenger), decision["prediction_sha256"])
        self.assertLess(
            truth_error(challenger, self.truth),
            truth_error(self.run_dir / event["prediction"], self.truth),
        )
        self.assertEqual(digest(self.run_dir / "seal.json"), self.original_seal)

    def test_quiet_event_abstains(self):
        event = self.events["quiet.npz"]
        decision = read(self.extra / event["id"] / "decision.json")
        self.assertEqual(event["status"], "unresolved")
        self.assertEqual(decision["status"], "retained")
        self.assertIsNone(decision["prediction"])

    def test_false_external_box_and_wrong_hash(self):
        event = self.events["seven.npz"]
        packet = dict(
            schema_version=1,
            event_id=event["id"],
            source_sha256=event["source_sha256"],
            action="add_component",
            time_bins=[1, 4],
            channel_bins=[0, 16],
        )
        proposal = self.root / "false-proposal.json"
        proposal.write_text(json.dumps(packet), encoding="utf-8")
        result_dir = self.root / "false-box"
        cli("challenge", self.run_dir, self.inputs, result_dir, "--proposal", proposal)
        decision = read(result_dir / event["id"] / "decision.json")
        self.assertEqual(decision["status"], "retained")
        self.assertIsNone(decision["prediction"])
        packet["source_sha256"] = "0" * 64
        proposal.write_text(json.dumps(packet), encoding="utf-8")
        error = cli(
            "challenge",
            self.run_dir,
            self.inputs,
            self.root / "wrong-hash",
            "--proposal",
            proposal,
            success=False,
        )
        self.assertIn("match", error.lower())
        self.assertFalse((self.root / "wrong-hash").exists())

    def test_resume_and_tamper_detection(self):
        clone = self.root / "resume-copy"
        shutil.copytree(self.extra, clone)
        seal = digest(clone / "seal.json")
        cli("rescue", self.run_dir, self.inputs, clone, "--resume", "--limit", 4)
        self.assertEqual(digest(clone / "seal.json"), seal)
        cli("verify", self.run_dir, "--second-look", clone)
        artifact = next(clone.glob("*/challenger.npz"))
        with artifact.open("ab") as f:
            f.write(b"corrupt")
        cli("verify", self.run_dir, "--second-look", clone, success=False)
        cli(
            "rescue",
            self.run_dir,
            self.inputs,
            clone,
            "--resume",
            "--limit",
            4,
            success=False,
        )
        self.assertEqual(digest(self.run_dir / "seal.json"), self.original_seal)


if __name__ == "__main__":
    unittest.main()
