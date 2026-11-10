"""Verify the distributed FitBurst wheel; optional comparison to a local upstream checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(data):
    return data.replace(b"\r\n", b"\n")


def verify(vendor, source=None):
    record = json.loads(
        (vendor / "FITBURST-PROVENANCE.json").read_text(encoding="utf-8")
    )
    wheel = vendor / record["wheel"]
    if digest(wheel.read_bytes()) != record["wheel_sha256"]:
        raise ValueError("FitBurst wheel checksum differs from the release")
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.startswith("fitburst/")]
        if len(names) != len(record["files"]) or set(names) != set(record["files"]):
            raise ValueError("Unexpected FitBurst scientific payload")
        if source:
            paths = {
                p.relative_to(source).as_posix()
                for p in (source / "fitburst").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            }
            if paths != set(names):
                raise ValueError("Unexpected source tree")
        for name in names:
            data = archive.read(name)
            if digest(data) != record["files"][name]["sha256"]:
                raise ValueError("Changed FitBurst package member: " + name)
            if source and normalized((source / name).read_bytes()) != normalized(data):
                raise ValueError("Scientific source differs from wheel: " + name)
        license_text = archive.read("fitburst-0.1.0.dist-info/licenses/LICENSE")
        if (
            normalized(license_text)
            != normalized((vendor / "FITBURST-LICENSE.txt").read_bytes())
            or digest(normalized(license_text)) != record["license_lf_sha256"]
        ):
            raise ValueError("Upstream license differs from the pinned release")
    print(
        f"Verified {wheel.name} and {len(names)} package members against release checksums."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--source", type=Path, help="optional checkout of the pinned upstream commit"
    )
    args = parser.parse_args()
    verify(args.vendor, args.source)
