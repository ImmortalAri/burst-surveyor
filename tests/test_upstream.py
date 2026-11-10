from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from burstsurveyor._upstream import verified_sources


class UpstreamIdentityTests(unittest.TestCase):
    def test_line_endings_and_changed_model(self):
        source = Path(find_spec("fitburst").origin).parent
        original = verified_sources()
        self.assertIn("analysis/model.py", original)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in original:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source / name).read_bytes().replace(b"\r\n", b"\n"))
            fake = SimpleNamespace(origin=str(root / "__init__.py"))
            with patch("burstsurveyor._upstream.util.find_spec", return_value=fake):
                normalized = verified_sources()
                model = root / "analysis/model.py"
                self.assertEqual(
                    normalized["analysis/model.py"],
                    sha256(model.read_bytes()).hexdigest(),
                )
                model.write_bytes(
                    model.read_bytes() + b"\n# changed model implementation\n"
                )
                with self.assertRaisesRegex(
                    RuntimeError, r"Changed FitBurst scientific file analysis/model.py"
                ):
                    verified_sources()
