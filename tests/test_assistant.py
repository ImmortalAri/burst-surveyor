"""The assistant must inspect evidence; the second-look guard alone selects the model."""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from burstsurveyor import assistant
from burstsurveyor._storage import read, sha256
from burstsurveyor.example import create_example


def box_data(path):
    channels, bins = 64, 300
    rng = np.random.default_rng(7664)
    f = 400 + (np.arange(channels) + 0.5) * 6.25
    t = np.arange(bins)
    raw = rng.normal(size=(channels, bins))
    for i, (peak, pivot) in enumerate(
        zip((33, 72, 114, 153, 191, 234, 270), (430, 510, 580, 660, 715, 470, 625))
    ):
        raw += (
            (8 - 0.55 * i)
            * np.exp(-0.5 * ((f - pivot) / 48) ** 2)[:, None]
            * np.exp(-0.5 * ((t - peak) / 2.2) ** 2)
        )
    grid = dict(
        res_time=0.0005,
        res_freq=6.25,
        freqs_bin0=400.0,
        times_bin0=0.0,
        is_dedispersed=True,
    )
    np.savez_compressed(
        path, data_full=raw.astype("float32"), metadata=np.array(json.dumps(grid))
    )


def action(event, choice, observation, **box):
    return {
        **{k: event[k] for k in ("id", "source_sha256", "packet_sha256")},
        "action": choice,
        "observation": observation,
        **box,
    }


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.raw = cls.root / "inputs"
        cls.raw.mkdir()
        box_data(cls.raw / "seven.npz")
        create_example(cls.raw / "two.npz")
        (cls.raw / "invalid.npz").write_bytes(b"invalid NPZ")
        # These tiny controls do not need the production per-worker memory estimate.
        cls.mem = patch("burstsurveyor._dispatch.estimate_bytes", return_value=0)
        cls.mem.start()
        cls.addClassCleanup(cls.mem.stop)
        cls.master = cls.root / "original"
        assistant.prepare(cls.raw, cls.master, workers=1, auto_rescue=False)

    def clone(self):
        d = self.root / self._testMethodName
        shutil.copytree(self.master, d)
        items = {r["input_name"]: r for r in read(d / "review/index.json")["events"]}
        return d, items

    def test_guarded_challenger_and_final_review(self):
        d, items = self.clone()
        with self.assertRaisesRegex(ValueError, "Pending"):
            assistant.finish(d)
        baseline_sha = sha256(d / "fit/seal.json")
        request = action(
            items["seven.npz"],
            "challenge",
            "Positive late residual in the middle-high band.",
            time_bins=[267, 275],
            channel_bins=[32, 48],
        )
        next_packet = assistant.decide(d, self.raw, request)["decisions"][0]
        self.assertEqual(next_packet["challenge_result"], "provisional_challenger")
        from burstsurveyor.triage import verify_queue
        self.assertTrue(verify_queue(d / "review", d / "fit")["verified"])
        selected = next(r for r in read(d / "review/queue.json")["events"] if r["id"] == next_packet["id"])
        self.assertEqual(selected["packet_sha256"], next_packet["packet_sha256"])
        self.assertEqual(selected["stage"], "followup")
        self.assertNotEqual(next_packet["packet_sha256"], request["packet_sha256"])
        with self.assertRaisesRegex(ValueError, "different event or review packet"):
            assistant.decide(
                d,
                self.raw,
                action(items["seven.npz"], "retain", "Old packet should not pass."),
            )
        final = [
            action(
                next_packet,
                "retain",
                "The late feature is reduced on the new residual.",
            ),
            action(
                items["two.npz"], "retain", "Both time peaks match the fitted profile."
            ),
            action(
                items["invalid.npz"],
                "unresolved",
                "The file has no readable NPZ header.",
            ),
        ]
        assistant.decide(d, self.raw, final)
        out = assistant.finish(d)
        self.assertEqual(out, assistant.finish(d))
        self.assertEqual(out["exported"], 2)
        rows = {r["input_name"]: r for r in read(Path(out["report"]))["events"]}
        chosen = rows["seven.npz"]["selected"]
        self.assertEqual((chosen["origin"], chosen["components"]), ("challenger", 7))
        self.assertTrue((d / chosen["prediction"]).is_file())
        self.assertTrue((d / rows["seven.npz"]["figure"]).is_file())
        with Path(out["csv"]).open(newline="", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)
        self.assertEqual(sha256(d / "fit/seal.json"), baseline_sha)
        (d / "report.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Changed final assistant report"):
            assistant.finish(d)

    def test_rejection_and_immutable_evidence(self):
        d, items = self.clone()
        case = items["two.npz"]
        with self.assertRaisesRegex(ValueError, "different event or review packet"):
            assistant.decide(
                d,
                self.raw,
                {
                    **action(case, "retain", "Both pulses match."),
                    "source_sha256": "0" * 64,
                },
            )
        with self.assertRaisesRegex(ValueError, "unresolved or review"):
            assistant.decide(
                d,
                self.raw,
                action(items["invalid.npz"], "retain", "No export available."),
            )
        req = action(
            case,
            "challenge",
            "Check the quiet end of the profile.",
            time_bins=[100, 110],
            channel_bins=[0, 64],
        )
        answer = assistant.decide(d, self.raw, req)["decisions"][0]
        self.assertEqual(answer["challenge_result"], "retained")
        self.assertEqual(answer, assistant.decide(d, self.raw, req)["decisions"][0])
        with self.assertRaisesRegex(ValueError, "Only one fixed proposal"):
            assistant.decide(d, self.raw, {**req, "time_bins": [110, 120]})
        assistant.decide(
            d,
            self.raw,
            action(answer, "review", "The extra proposal lacked replicated support."),
        )
        figure = d / items["seven.npz"]["figure"]
        figure.write_bytes(figure.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "Changed assistant figure"):
            assistant.decide(
                d, self.raw, action(items["seven.npz"], "retain", "Original picture.")
            )

    def test_limited_review_can_resume_before_decision(self):
        d = self.root / "limited"
        # Use a preverified scientific batch while simulating visualization pressure only.
        d.mkdir()
        shutil.copytree(self.master / "fit", d / "fit")
        shutil.copyfile(self.master / "session.json", d / "session.json")
        with patch(
            "burstsurveyor.assistant.load_for_review",
            side_effect=RuntimeError("Review needs more available memory"),
        ):
            assistant.prepare(self.raw, d, workers=1, resume=True, auto_rescue=False)
        entries = {r["input_name"]: r for r in read(d / "review/index.json")["events"]}
        self.assertIsNone(entries["two.npz"]["figure"])
        with self.assertRaisesRegex(ValueError, "Figure unavailable"):
            assistant.decide(
                d,
                self.raw,
                action(entries["two.npz"], "retain", "No figure was available."),
            )
        assistant.prepare(self.raw, d, workers=1, resume=True, auto_rescue=False)
        revised = {r["input_name"]: r for r in read(d / "review/index.json")["events"]}
        self.assertNotEqual(
            entries["two.npz"]["packet_sha256"], revised["two.npz"]["packet_sha256"]
        )
        self.assertTrue((d / revised["two.npz"]["figure"]).is_file())

    def test_audit_resumes_missing_inspection_and_preserves_fit(self):
        from burstsurveyor.triage import audit, verify_queue
        d, _ = self.clone()
        out = self.root / "audit-output"
        before = sha256(d / "fit/seal.json")
        with patch("burstsurveyor.review.load_for_review", side_effect=MemoryError("Unavailable")):
            audit(d / "fit", self.raw, out)
        self.assertTrue(all(r["triage"]["priority"] == 0 for r in read(out / "queue.json")["events"]))
        result = audit(d / "fit", self.raw, out, resume=True)
        self.assertTrue(result["verification"]["verified"])
        self.assertEqual(before, sha256(d / "fit/seal.json"))
        rows = read(out / "queue.json")["events"]
        self.assertEqual(sum(r["inspection_status"] == "figure_ready" for r in rows), 2)
        selected = next(r for r in rows if r["figure"])
        self.assertIn("/inspection/", selected["packet"])
        figure = out / selected["figure"]
        figure.write_bytes(figure.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Changed triage figure"):
            verify_queue(out, d / "fit")

    def test_combined_prepare_produces_queue_and_records_mode(self):
        d = self.root / "combined-workspace"
        assistant.prepare(self.raw / "two.npz", d, workers=1, mode="combined")
        self.assertEqual(read(d / "session.json")["mode"], "combined")
        queue = read(d / "review/queue.json")
        self.assertEqual(queue["inputs"], 1)
        self.assertEqual(queue["events"][0]["input_name"], "two.npz")
        self.assertFalse(list((d / "review").rglob("challenge.json")))


if __name__ == "__main__":
    unittest.main()
