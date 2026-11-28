"""Run and verify an opt-in rescue for one observation in a sealed fit."""

from pathlib import Path
import tempfile

import psutil
from threadpoolctl import threadpool_limits

from ._dispatch import MIB, RESERVE_MIB, estimate_bytes, run_isolated
from ._storage import read, sha256, write
from .batch import _method, verify
from .review import find_event


def _worker(source, run, event, output):
    from .rescue_router import run_event

    with threadpool_limits(limits=1):
        return run_event(source, run, event, output)


def _result_files(folder, event):
    report = read(folder / "result.json")
    if report["source_sha256"] != event["source_sha256"]:
        raise ValueError("Rescue receipt is for a different input")
    disposition = report["disposition"]
    files = {name: sha256(folder / name) for name in ("method.json", "result.json")}
    prediction = folder / "prediction.npz"
    if disposition in {"candidate", "review"}:
        native = report["official_export"]
        if (not report["triggered"] or native["status"] != "provisional"
                or native["source_sha256"] != event["source_sha256"]
                or report["native_coverage"]["missing"]
                or min(report["native_error_reduction_pct"]) <= 0):
            raise ValueError("Rescue export does not meet its native-fit checks")
        if disposition == "candidate":
            folds = report["channel_folds"]
            if len(folds) != 2 or not all(f["passed"] and min(f["error_reduction_pct"]) > 0 for f in folds):
                raise ValueError("Rescue candidate does not meet both held-out checks")
        digest = sha256(prediction)
        if native["prediction_sha256"] != digest:
            raise ValueError("Changed rescue prediction")
        files[prediction.name] = digest
    elif disposition != "retained" or prediction.exists():
        raise ValueError("Invalid routed rescue disposition")
    return report, files


def _describe(output, event, report):
    return dict(output=str(output), event_id=event["id"], input_name=event["input_name"],
                route=report["route"], disposition=report["disposition"], reason=report["reason"],
                result=str(output / "result.json"),
                prediction=str(output / "prediction.npz") if report.get("official_export") else None)


def route_rescue(run, source, output):
    """Try one selected rescue and publish its separate, verifiable result."""
    run, source, output = (Path(path).resolve() for path in (run, source, output))
    if output.is_relative_to(run):
        raise ValueError("Use a separate output directory outside the sealed run")
    if output.exists():
        raise FileExistsError(f"Use a new routed rescue directory: {output}")
    if not source.is_file() or source.suffix.lower() != ".npz":
        raise ValueError("Supply one original raw NPZ file")

    stock_seal = sha256(run / "seal.json")
    stock = verify(run)
    manifest = read(run / "manifest.json")
    if manifest["method"]["policy"]["mode"] != "bounded":
        raise ValueError("Routed rescue requires a sealed bounded fit")
    if manifest["method"] != _method():
        raise ValueError("Use the same scientific method and runtime as the sealed fit")
    events = [e for e in stock["events"] if e["input_name"] == source.name]
    if not events or sha256(source) != events[0]["source_sha256"]:
        raise ValueError("Input filename and bytes must match a sealed fit event")
    event = events[0]
    if event["status"] not in {"provisional", "unresolved"}:
        raise ValueError(f"Routed rescue cannot inspect an event marked {event['status']}")
    member = next(e for e in manifest["events"] if e["id"] == event["id"])
    if psutil.virtual_memory().available < estimate_bytes(member) + RESERVE_MIB * MIB:
        raise RuntimeError("Routed rescue needs more available memory")

    here = Path(__file__).resolve().parent
    plan = dict(schema_version=1, baseline_seal_sha256=stock_seal,
                event_id=event["id"], source_sha256=event["source_sha256"], method=manifest["method"],
                sources={name: sha256(here / name) for name in (
                    "route_rescue.py", "rescue_router.py", "_router_peaks.py", "_router_spectral.py",
                    "secondlook.py", "diagnostics.py", "triage.py",
                )})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".routed-rescue-", dir=output.parent) as temporary:
        scratch = Path(temporary)
        stage = scratch / "candidate"
        run_isolated(_worker, (source, run, event, stage), scratch=scratch)
        if sha256(run / "seal.json") != stock_seal:
            raise ValueError("Baseline seal changed during the rescue")
        write(stage / "method.json", plan)
        report, files = _result_files(stage, event)
        write(stage / "seal.json", dict(schema_version=1, files=files))
        if output.exists():
            raise FileExistsError(f"Routed rescue directory already exists: {output}")
        stage.rename(output)
    return _describe(output, event, report)


def verify_route_rescue(run, output, summary=None):
    """Verify the routed rescue and its link to the original fit."""
    run, output = (Path(path).resolve() for path in (run, output))
    if summary is None:
        summary = verify(run)
    method = read(output / "method.json")
    event = find_event(summary, method["event_id"])
    if (method["source_sha256"] != event["source_sha256"]
            or method["baseline_seal_sha256"] != sha256(run / "seal.json")
            or method["method"] != read(run / "manifest.json")["method"]):
        raise ValueError("Rescue is attached to a different sealed fit")
    report, files = _result_files(output, event)
    if read(output / "seal.json") != dict(schema_version=1, files=files):
        raise ValueError("Changed routed rescue result")
    return _describe(output, event, report)
