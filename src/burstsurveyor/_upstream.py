"""Verify the scientific files of the FitBurst dependency before a run or replay."""

from hashlib import sha256
from importlib import metadata, resources, util
import json
from pathlib import Path


def verified_sources() -> dict[str, str]:
    """Return on-disk hashes for the approved FitBurst source; raise RuntimeError on drift.

    Interpreter-generated caches are excluded. The committed upstream text and
    our reference wheel can use different newline conventions. Only exact
    source or equivalent CRLF/LF text is accepted; hashes in receipts are always
    calculated on the actual installed bytes.
    """
    manifest = json.loads(
        resources.files("burstsurveyor")
        .joinpath("data", "fitburst-source.json")
        .read_text(encoding="utf-8")
    )
    hint = f"Install the release's vendor/{manifest['wheel']} and retry."
    try:
        version = metadata.version("fitburst")
        spec = util.find_spec("fitburst")
    except (metadata.PackageNotFoundError, ValueError, ImportError) as error:
        raise RuntimeError(f"FitBurst installation is unavailable. {hint}") from error
    if version != manifest["version"] or spec is None or not spec.origin:
        raise RuntimeError(
            f"Expected FitBurst {manifest['version']} with its Python source files. {hint}"
        )

    root = Path(spec.origin).parent
    expected = manifest["files"]
    actual = {}
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts and path.suffix.lower() == ".pyc":
                continue
            if path.suffix.lower() not in {
                ".py",
                ".yaml",
                ".yml",
                ".pyd",
                ".dll",
                ".so",
                ".pyc",
            }:
                continue
            name = relative.as_posix()
            if name not in expected:
                raise RuntimeError(
                    f"Unexpected FitBurst scientific file {name}. {hint}"
                )
            data = path.read_bytes()
            digest = sha256(data).hexdigest()
            if (
                digest != expected[name]["exact"]
                and sha256(data.replace(b"\r\n", b"\n")).hexdigest()
                != expected[name]["lf"]
            ):
                raise RuntimeError(f"Changed FitBurst scientific file {name}. {hint}")
            actual[name] = digest
    except OSError as error:
        raise RuntimeError(
            f"Cannot read the installed FitBurst sources. {hint}"
        ) from error
    missing = sorted(expected.keys() - actual.keys())
    if missing:
        raise RuntimeError(f"Missing FitBurst scientific file {missing[0]}. {hint}")
    return dict(sorted(actual.items()))
