"""Exercise the installed CLI across the text and data boundaries an AI client uses."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PublicContractTests(unittest.TestCase):
    def test_example_roundtrip_with_unicode_and_legacy_console(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = Path(tmp) / "observation-α.npz"
            env = dict(os.environ, PYTHONIOENCODING="cp1252")
            argv = [sys.executable, "-m", "burstsurveyor"]
            created = subprocess.run(
                argv + ["example", str(name)], capture_output=True, check=True, env=env
            )
            self.assertEqual(
                Path(json.loads(created.stdout)["synthetic_input"]), name.resolve()
            )
            info = subprocess.run(
                argv + ["inspect", str(name)], capture_output=True, check=True, env=env
            )
            self.assertEqual(json.loads(info.stdout)["shape"], [64, 144])
            refused = subprocess.run(
                argv + ["example", str(name)], capture_output=True, env=env
            )
            self.assertNotEqual(refused.returncode, 0)

    def test_cli_identity_and_help(self):
        command = [sys.executable, "-m", "burstsurveyor"]
        help_text = subprocess.check_output(command + ["--help"], text=True)
        self.assertNotIn("--version", help_text)
        self.assertIn("{prepare,decide,finish,doctor}", help_text)
        doctor = json.loads(subprocess.check_output(command + ["doctor"]))
        self.assertEqual(doctor["product"], "Burst Surveyor")
        self.assertNotIn("version", doctor)

    def test_machine_schemas(self):
        for kind in ("proposal", "results"):
            out = subprocess.run(
                [sys.executable, "-m", "burstsurveyor", "schema", kind],
                capture_output=True,
                check=True,
            )
            schema = json.loads(out.stdout)
            self.assertEqual(schema["properties"]["schema_version"]["const"], 1)

    def test_supplied_example_is_bound_to_expected_result(self):
        root = Path(__file__).resolve().parents[1] / "examples"
        expected = json.loads((root / "expected.json").read_text())
        digest = hashlib.sha256((root / expected["input"]).read_bytes()).hexdigest()
        self.assertEqual(digest, expected["source_sha256"])
