"""Regression checks for batch integrity, interruption and per-input isolation."""

from pathlib import Path
import json
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from burstsurveyor import batch
from burstsurveyor import _dispatch
from burstsurveyor._storage import read, run_lock, sha256
from burstsurveyor.example import create_example


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        cls.inputs = cls.base / "input files"
        cls.inputs.mkdir()
        # Two names with the same bytes remain two individually verifiable inputs.
        create_example(cls.inputs / "two pulses.npz")
        shutil.copyfile(cls.inputs / "two pulses.npz", cls.inputs / "seal.NPZ")
        (cls.inputs / "bad.npz").write_bytes(b"malformed file")
        with np.load(cls.inputs / "two pulses.npz", allow_pickle=False) as valid:
            grid = json.loads(valid["metadata"].item())
            grid["is_dedispersed"] = False
            np.savez(
                cls.inputs / "wrong grid.npz",
                data_full=valid["data_full"],
                metadata=json.dumps(grid),
            )
        cls.run_dir = cls.base / "sealed run"
        # Tiny generated fixtures; the published scheduler's admission and dispatch
        # behavior are exercised separately below with deterministic memory readings.
        if (
            _dispatch.psutil.virtual_memory().available
            <= (_dispatch.RESERVE_MIB + 128) * _dispatch.MIB
        ):
            raise unittest.SkipTest(
                "Insufficient free system memory for the synthetic integration fixture."
            )
        tiny_fixtures = patch.object(_dispatch, "estimate_bytes", return_value=0)
        tiny_fixtures.start()
        cls.addClassCleanup(tiny_fixtures.stop)
        cls.result = batch.fit(cls.inputs, cls.run_dir, workers=2)

    def test_parallel_mixed_inputs(self):
        self.assertEqual(self.result["inputs"], 4)
        self.assertEqual(self.result["statuses"], {"unsupported": 2, "provisional": 2})
        self.assertEqual(self.result["exported"], 2)
        self.assertEqual(batch.verify(self.run_dir), self.result)
        events = {e["input_name"]: e for e in self.result["events"]}
        a, b = events["two pulses.npz"], events["seal.NPZ"]
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(a["source_sha256"], b["source_sha256"])
        self.assertTrue(events["bad.npz"]["reasons"])
        self.assertIsNone(events["bad.npz"]["prediction"])
        self.assertIn("dedispersed", events["wrong grid.npz"]["reasons"][0])
        for event in (a, b):
            self.assertGreater(event["gain"], 0)
            self.assertGreater(event["rms_gain"], 0)
            self.assertTrue((self.run_dir / event["prediction"]).is_file())
            receipt = read(self.run_dir / event["native_receipt"])
            self.assertEqual(event["gain"], receipt["selected_score"]["stored_gain"])
            self.assertEqual(
                event["rms_gain"], receipt["selected_score"]["stored_rms_gain"]
            )
            self.assertEqual(
                receipt["primary_sha256"],
                sha256(self.run_dir / event["primary_receipt"]),
            )

    def test_complete_resume_keeps_both_seals(self):
        names = ("seal.json", "native/seal.json", "primary/seal.json", "results.json")
        before = [sha256(self.run_dir / p) for p in names]
        self.assertEqual(
            batch.fit(self.inputs, self.run_dir, workers=2, resume=True), self.result
        )
        self.assertEqual(before, [sha256(self.run_dir / p) for p in names])
        with self.assertRaises(FileExistsError):
            batch.fit(self.inputs, self.run_dir, workers=2)

    def test_deferred_input_does_not_block_export_of_its_neighbor(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "batch"

            def estimate(event):
                return 2**50 if event["name"] == "seal.NPZ" else 0

            with (
                patch.object(_dispatch, "estimate_bytes", side_effect=estimate),
                patch.object(_dispatch, "MEMORY_WAIT_SECONDS", 0),
            ):
                result = batch.fit(self.inputs, directory, workers=1, defer_memory=True)
            self.assertEqual(batch.verify(directory), result)
            events = {row["input_name"]: row for row in result["events"]}
            self.assertEqual(result["exported"], 1)
            self.assertIsNotNone(events["two pulses.npz"]["prediction"])
            self.assertEqual(events["seal.NPZ"]["status"], "deferred")
            self.assertIsNone(events["seal.NPZ"]["prediction"])
            self.assertIn("Memory admission", events["seal.NPZ"]["reasons"][0])

    def test_changed_input_refuses_resume(self):
        changed = self.base / "changed"
        shutil.copytree(self.inputs, changed)
        (changed / "bad.npz").write_bytes(b"changed input, same filename")
        with self.assertRaisesRegex(ValueError, "Changed input"):
            batch.fit(changed, self.run_dir, workers=2, resume=True)

    def test_portable_verification_needs_no_inputs_or_hidden_work_files(self):
        with TemporaryDirectory() as tmp:
            moved = Path(tmp) / "moved results"
            shutil.copytree(
                self.run_dir,
                moved,
                ignore=shutil.ignore_patterns(".owner.lock", ".work"),
            )
            self.assertEqual(batch.verify(moved), self.result)
            for item in self.result["events"]:
                for key in ("prediction", "primary_receipt", "native_receipt"):
                    if item[key]:
                        self.assertFalse(Path(item[key]).is_absolute())
                        self.assertNotIn("\\", item[key])

    def test_backend_rejection_does_not_claim_output_name(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "can retry here"
            with patch(
                "burstsurveyor._upstream.verified_sources",
                side_effect=RuntimeError("changed FitBurst source"),
            ):
                with self.assertRaisesRegex(RuntimeError, "changed FitBurst source"):
                    batch.fit(self.inputs / "two pulses.npz", output, workers=1)
            self.assertFalse(output.exists())

    def test_corrupt_export_and_receipt_are_rejected(self):
        event = next(e for e in self.result["events"] if e["prediction"])
        for rel in (event["prediction"], event["primary_receipt"]):
            with self.subTest(file=rel), TemporaryDirectory() as tmp:
                copied = Path(tmp) / "run"
                shutil.copytree(self.run_dir, copied)
                with (copied / rel).open("ab") as stream:
                    stream.write(b" ")
                with self.assertRaisesRegex(ValueError, "Changed"):
                    batch.verify(copied)

    def test_resume_after_event_committed(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "partly fitted"
            commit = batch._commit

            def interrupt(*args):
                commit(*args)
                raise KeyboardInterrupt

            with (
                patch.object(batch, "_commit", interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                batch.fit(self.inputs / "two pulses.npz", output, workers=1)
            self.assertFalse((output / "seal.json").exists())
            prior = next((output / "primary/events").iterdir()) / "receipt.json"
            original_sha = sha256(prior)
            result = batch.fit(
                self.inputs / "two pulses.npz", output, workers=1, resume=True
            )
            self.assertEqual(result["exported"], 1)
            self.assertEqual(batch.verify(output), result)
            self.assertEqual(sha256(prior), original_sha)
            attempts = [
                read(p)["status"] for p in (output / "primary/attempts").glob("*.json")
            ]
            self.assertCountEqual(attempts, ["interrupted", "complete"])


class SchedulerTests(unittest.TestCase):
    def test_inadequate_memory_leaves_work_pending(self):
        event = {"input_bytes": 1024, "expanded_bytes": 1024}
        with (
            patch.object(_dispatch.psutil, "virtual_memory") as memory,
            patch.object(_dispatch, "MEMORY_WAIT_SECONDS", 0),
        ):
            memory.return_value.available = 512 * _dispatch.MIB
            with self.assertRaisesRegex(RuntimeError, "Resume"):
                _dispatch.dispatch([event], 1, self.fail, self.fail, self.fail)

    def test_assistant_defers_unadmitted_events_without_starting_a_worker(self):
        event = {"id": "large", "input_bytes": 4096, "expanded_bytes": 4096}
        delayed = []
        with (
            patch.object(_dispatch, "MEMORY_WAIT_SECONDS", 0),
            patch.object(_dispatch.psutil, "virtual_memory") as memory,
        ):
            memory.return_value.available = 515 * _dispatch.MIB
            stats = _dispatch.dispatch(
                [event],
                1,
                self.fail,
                self.fail,
                self.fail,
                defer=lambda item, need, free: delayed.append((item["id"], need, free)),
            )
        self.assertEqual(delayed, [("large", 896, 515)])
        self.assertEqual(stats["peak_active"], 0)

    def test_lock_ownership_is_visible_to_another_process(self):
        with TemporaryDirectory() as tmp:
            script = (
                "from pathlib import Path\nfrom burstsurveyor._storage import run_lock\nimport sys\n"
                "with run_lock(Path(sys.argv[1])):\n    pass\n"
            )
            with run_lock(Path(tmp)):
                proc = subprocess.run(
                    [sys.executable, "-B", "-c", script, tmp],
                    capture_output=True,
                    text=True,
                )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Another process owns", proc.stderr)
            with run_lock(Path(tmp)):
                pass


if __name__ == "__main__":
    unittest.main()
