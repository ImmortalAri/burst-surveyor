"""A raw-only batch rescues a missed pulse once, then requires final review."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from burstsurveyor import assistant
from burstsurveyor._storage import read, sha256
from burstsurveyor.example import create_example
from test_assistant import box_data, action


class AutomaticRescueTests(unittest.TestCase):
    def test_raw_batch_rescue_resume_and_final_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "inputs"
            raw.mkdir()
            box_data(raw / "seven.npz")
            create_example(raw / "two.npz")
            workspace = root / "work"
            # Small synthetic arrays do not need the production memory estimate.
            with patch("burstsurveyor._dispatch.estimate_bytes", return_value=0):
                prepared = assistant.prepare(raw, workspace, workers=1)
                items = {r["input_name"]: r for r in read(Path(prepared["index"]))["events"]}
                seven = items["seven.npz"]
                two = items["two.npz"]
                self.assertEqual(len(prepared["followups"]), 1)
                followup = prepared["followups"][0]
                self.assertEqual(followup["id"], seven["id"])
                self.assertFalse((workspace / "review" / two["id"] / "challenge.json").exists())
                alternate = workspace / "alternatives" / seven["id"]
                decision = read(alternate / seven["id"] / "decision.json")
                self.assertEqual(decision["status"], "provisional_challenger")
                self.assertEqual(decision["native"]["selected"]["n"], 7)
                for name in ("official_gain", "official_channel_check"):
                    self.assertTrue(all(b > a for a, b in zip(decision[name]["incumbent"], decision[name]["proposed"])))
                baseline = sha256(workspace / "fit/seal.json")
                alternate_seal = sha256(alternate / "seal.json")
                with self.assertRaisesRegex(ValueError, "Pending"):
                    assistant.finish(workspace)
                resumed = assistant.prepare(raw, workspace, workers=1, resume=True)
                self.assertEqual(resumed["followups"], prepared["followups"])
                self.assertEqual(sha256(alternate / "seal.json"), alternate_seal)
                assistant.decide(workspace, raw, [
                    action(followup, "retain", "The recovered late pulse matches the injection."),
                    action(two, "retain", "Both injected pulses match the reconstruction."),
                ])
                final = assistant.finish(workspace)
                self.assertEqual(final, assistant.finish(workspace))
                report = {r["input_name"]: r for r in read(Path(final["report"]))["events"]}
                self.assertEqual(report["seven.npz"]["selected"]["origin"], "challenger")
                self.assertEqual(report["two.npz"]["selected"]["origin"], "baseline")
                self.assertEqual(sha256(workspace / "fit/seal.json"), baseline)
                with self.assertRaisesRegex(ValueError, "immutable"):
                    assistant.decide(workspace, raw, action(two, "review", "Changed decision."))


if __name__ == "__main__":
    unittest.main()
