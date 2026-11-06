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
