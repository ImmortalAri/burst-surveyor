"""Fit raw spectra, commit each event and verify the final native-pixel results."""

import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import tempfile
import time
import uuid

from ._dispatch import MAX_WORKERS, RESERVE_MIB, dispatch
from ._storage import read, run_lock, sha256, utc, write

SCHEMA = 1
STATUSES = {"provisional", "unresolved", "unsupported", "refused", "failed", "deferred"}
STAGES = ("primary", "native")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _name_id(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _files(inputs):
    source = Path(inputs).resolve()
    if source.is_file() and source.suffix.lower() == ".npz":
        paths = [source]
    elif source.is_dir():
        paths = sorted(
            (p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".npz"),
            key=lambda p: p.name,
        )
    else:
        raise ValueError("Supply a raw NPZ file or a directory of NPZ files.")
    if not paths:
        raise ValueError(f"No NPZ inputs found in {source}")
    by_id = {_name_id(p.name): p for p in paths}
    _require(len(paths) == len(by_id), "Input filename hash collision.")
    return by_id


def _method():
    from ._upstream import verified_sources

    here = Path(__file__).resolve().parent
    science = here / "science"
    sources = [
        here / name
        for name in ("batch.py", "_dispatch.py", "_storage.py", "_upstream.py")
    ]
    sources.append(here / "data" / "fitburst-source.json")
    sources.extend(
        p for p in science.rglob("*") if p.is_file() and p.suffix in {".py", ".json"}
    )
    try:
        product_version = metadata.version("burst-surveyor")
    except metadata.PackageNotFoundError:
        product_version = "source-tree"
    return {
        "schema_version": SCHEMA,
        "product": "Burst Surveyor",
        "product_version": product_version,
        "python": platform.python_version(),
        "packages": {
            name: metadata.version(name)
            for name in ("fitburst", "numpy", "scipy", "psutil", "threadpoolctl")
        },
        "sources": {p.relative_to(here).as_posix(): sha256(p) for p in sorted(sources)},
        "fitburst_sources": verified_sources(),
        "policy": {
            "mode": "bounded",
            "reserve_mib": RESERVE_MIB,
            "threads_per_worker": 1,
        },
    }


def _inventory(paths):
    from .science import InputError, inspect_npz

    events = []
    for key, path in paths.items():
        item = {
            "id": key,
            "name": path.name,
            "sha256": sha256(path),
            "input_bytes": path.stat().st_size,
            "expanded_bytes": 0,
        }
        try:
            inspected = inspect_npz(path)
            item["expanded_bytes"] = int(inspected["expanded_bytes"])
            item["shape"] = list(inspected["shape"])
        except InputError as exc:
            item["unsupported_reason"] = str(exc)
        events.append(item)
    return events


def _manifest(directory):
    value = read(directory / "manifest.json")
    _require(value["schema_version"] == SCHEMA, "Unknown run manifest schema")
    ids = [event["id"] for event in value["events"]]
    _require(len(set(ids)) == len(ids) and bool(ids), "Invalid input roster")
    for event in value["events"]:
        _require(event["id"] == _name_id(event["name"]), "Invalid event identity")
    return value


def _check_inputs(paths, manifest):
    expected = {e["id"]: e["sha256"] for e in manifest["events"]}
    _require(set(paths) == set(expected), "Input roster differs from the original run")
    for key, path in paths.items():
        _require(sha256(path) == expected[key], "Changed input: " + path.name)


def _folder(directory, stage, key):
    return directory / stage / "events" / key


def _run_task(task):
    from threadpoolctl import threadpool_limits
    from .science import InputError, export_event, fit_event

    stage, raw, output, primary, event, primary_hash = task
    started = time.monotonic()
    incumbent = None
    if stage == "native":
        contents = primary.read_bytes()
        _require(
            hashlib.sha256(contents).hexdigest() == primary_hash,
            "Changed sealed primary before export",
        )
        incumbent = json.loads(contents)
    with tempfile.TemporaryDirectory(dir=output) as tmp, threadpool_limits(limits=1):
        try:
            if stage == "primary":
                result = fit_event(raw, Path(tmp))
            else:
                result = export_event(
                    raw, incumbent, output / (event["id"] + ".npz"), Path(tmp)
                )
        except InputError as exc:
            result = {
                "status": "unsupported",
                "reasons": [str(exc)],
                "source_sha256": sha256(raw),
            }
        except (ValueError, ArithmeticError) as exc:
            # A degenerate spectrum may defeat noise estimation or a solver. Keep that event visible.
            result = {
                "status": "failed",
                "reasons": [f"{type(exc).__name__}: {exc}"],
                "source_sha256": sha256(raw),
            }
    _require(
        result.get("source_sha256") == event["sha256"],
        "Scientific result is not bound to its input",
    )
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def _check_event(folder, stage, event, primary_hash=None):
    commit = read(folder / "commit.json")
    row = read(folder / "receipt.json")
    receipt_hash = sha256(folder / "receipt.json")
    _require(
        commit == {"receipt_sha256": receipt_hash},
        "Changed event receipt: " + event["id"],
    )
    _require(
        row["source_sha256"] == event["sha256"] and row["id"] == event["id"],
        "Input/receipt mismatch",
    )
    _require(row["status"] in STATUSES, "Unknown scientific status")
    allowed = {"commit.json", "receipt.json"}
    if row["status"] == "provisional":
        selected = row.get("selected") or {}
        _require(
            isinstance(selected.get("n"), int) and selected["n"] > 0,
            "Invalid selected component count",
        )
        score = selected if stage == "primary" else row.get("selected_score") or {}
        fields = (
            ("gain", "rms_gain")
            if stage == "primary"
            else ("stored_gain", "stored_rms_gain")
        )
        _require(
            all(
                isinstance(score.get(k), (float, int)) and math.isfinite(score[k])
                for k in fields
            ),
            "Missing finite common or RMS gain",
        )
    if stage == "native":
        _require(
            row["primary_sha256"] == primary_hash,
            "Native receipt detached from primary",
        )
        if row["status"] == "provisional":
            filename = event["id"] + ".npz"
            _require(
                row.get("prediction") == filename, "Unexpected native artifact path"
            )
            _require(
                sha256(folder / filename) == row["prediction_sha256"],
                "Changed native prediction: " + filename,
            )
            allowed.add(filename)
        else:
            _require(
                row.get("prediction") is None,
                "Unavailable event advertises a prediction",
            )
    _require(
        {p.name for p in folder.iterdir()} == allowed,
        "Unexpected artifact in " + folder.name,
    )
    return receipt_hash


def _commit(directory, stage, event, scratch, result, primary_hash):
    result["id"] = event["id"]
    if stage == "native":
        result["primary_sha256"] = primary_hash
    write(scratch / "receipt.json", result)
    write(scratch / "commit.json", {"receipt_sha256": sha256(scratch / "receipt.json")})
    fingerprint = _check_event(scratch, stage, event, primary_hash)
    scratch.rename(_folder(directory, stage, event["id"]))
    return fingerprint


def _receipts(directory, manifest, stage, primary):
    roster = {e["id"]: e for e in manifest["events"]}
    current = directory / stage / "events"
    hashes = {}
    for folder in sorted(current.iterdir()):
        _require(
            folder.is_dir() and folder.name in roster,
            "Unexpected event folder: " + folder.name,
        )
        hashes[folder.name] = _check_event(
            folder, stage, roster[folder.name], primary.get(folder.name)
        )
    return hashes


def _verify_stage(directory, manifest, stage, primary):
    sealed = read(directory / stage / "seal.json")
    _require(
        sealed["manifest_sha256"] == sha256(directory / "manifest.json"),
        "Changed run manifest",
    )
    if stage == "native":
        _require(
            sealed["primary_seal_sha256"] == sha256(directory / "primary/seal.json"),
            "Native seal detached from primary seal",
        )
    hashes = _receipts(directory, manifest, stage, primary)
    _require(
        hashes == sealed["receipts"] and len(hashes) == len(manifest["events"]),
        "Incomplete stage seal",
    )
    _require(
        sealed["attempts"] == _attempt_hashes(directory / stage),
        "Changed stage timing receipt",
    )
    return hashes


def _attempt_hashes(stage_dir):
    return {p.name: sha256(p) for p in sorted((stage_dir / "attempts").glob("*.json"))}


def _run_stage(directory, paths, manifest, stage, workers, primary):
    stage_dir = directory / stage
    (stage_dir / "events").mkdir(parents=True, exist_ok=True)
    if (stage_dir / "seal.json").exists():
        return _verify_stage(directory, manifest, stage, primary)
    hashes = _receipts(directory, manifest, stage, primary)
    todo = []
    started = time.monotonic()
    (stage_dir / "attempts").mkdir(exist_ok=True)
    timing_path = stage_dir / "attempts" / (uuid.uuid4().hex + ".json")
    timing = {
        "started_utc": utc(),
        "completed_before": len(hashes),
        "status": "started",
    }
    write(timing_path, timing)
    scratch_root = directory / ".work"
    scratch_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root) as attempt:

        def task(event):
            area = Path(tempfile.mkdtemp(prefix="event-", dir=attempt))
            primary_path = _folder(directory, "primary", event["id"]) / "receipt.json"
            return (
                stage,
                paths[event["id"]],
                area,
                primary_path,
                event,
                primary.get(event["id"]),
            )

        def progress(status, **extra):
            write(
                stage_dir / "progress.json",
                {
                    "stage": stage,
                    "status": status,
                    "updated_utc": utc(),
                    "completed": len(hashes),
                    "total": len(manifest["events"]),
                    **extra,
                },
            )

        def commit(event, args, row):
            key = event["id"]
            hashes[key] = _commit(
                directory, stage, event, args[2], row, primary.get(key)
            )
            progress("running", last_event=key)

        def defer(event, required_mib, available_mib):
            reason = f"Memory admission requires {required_mib:.0f} MiB; {available_mib:.0f} MiB available."
            commit(
                event,
                task(event),
                dict(
                    status="deferred",
                    reasons=[reason],
                    source_sha256=event["sha256"],
                    elapsed_seconds=0.0,
                ),
            )

        progress("running")
        try:
            for event in manifest["events"]:
                key = event["id"]
                if key in hashes:
                    continue
                if event.get("unsupported_reason"):
                    row = {
                        "status": "unsupported",
                        "reasons": [event["unsupported_reason"]],
                    }
                elif stage == "native":
                    original = read(_folder(directory, "primary", key) / "receipt.json")
                    if original["status"] == "provisional":
                        todo.append(event)
                        continue
                    row = {
                        "status": original["status"],
                        "reasons": original.get("reasons")
                        or ["No selected raw-only candidate"],
                    }
                else:
                    todo.append(event)
                    continue
                row.update(source_sha256=event["sha256"], elapsed_seconds=0.0)
                commit(event, task(event), row)
            metrics = dispatch(
                todo,
                min(workers, max(1, len(todo))),
                task,
                _run_task,
                commit,
                defer=defer if manifest.get("defer_memory") else None,
            )
            _require(
                _method() == manifest["method"],
                "Scientific source or numerical environment changed during fitting",
            )
            _require(len(hashes) == len(manifest["events"]), "Stage has missing events")
            _check_inputs(paths, manifest)
        except BaseException as exc:
            progress("interrupted", reason=f"{type(exc).__name__}: {exc}")
            timing.update(
                status="interrupted",
                elapsed_seconds=time.monotonic() - started,
                completed=len(hashes),
                reason=f"{type(exc).__name__}: {exc}",
            )
            write(timing_path, timing)
            raise
    timing.update(
        status="complete",
        elapsed_seconds=time.monotonic() - started,
        completed=len(hashes),
    )
    write(timing_path, timing)
    seal = {
        "schema_version": SCHEMA,
        "stage": stage,
        "manifest_sha256": sha256(directory / "manifest.json"),
        "sealed_utc": utc(),
        "receipts": hashes,
        "last_attempt_seconds": time.monotonic() - started,
        "attempts": _attempt_hashes(stage_dir),
        "scheduler": metrics,
    }
    if stage == "native":
        seal["primary_seal_sha256"] = sha256(directory / "primary/seal.json")
    write(stage_dir / "seal.json", seal)
    progress("complete")
    return hashes


def _summary(directory, manifest):
    events = []
    counts = {}
    for event in manifest["events"]:
        key = event["id"]
        src = _folder(directory, "primary", key) / "receipt.json"
        dest = _folder(directory, "native", key) / "receipt.json"
        first, native = read(src), read(dest)
        status = native["status"]
        counts[status] = counts.get(status, 0) + 1
        selected = native.get("selected") or {}
        score = native.get("selected_score") or {}
        prediction = dest.parent / (key + ".npz") if status == "provisional" else None
        events.append(
            {
                "id": key,
                "input_name": event["name"],
                "source_sha256": event["sha256"],
                "status": status,
                "reasons": native.get("reasons") or [],
                "components": selected.get("n"),
                "scatter": selected.get("scatter"),
                "gain": score.get("stored_gain"),
                "rms_gain": score.get("stored_rms_gain"),
                "residual_peak": first.get("residual_peak"),
                "primary_seconds": first["elapsed_seconds"],
                "export_seconds": native["elapsed_seconds"],
                "primary_receipt": src.relative_to(directory).as_posix(),
                "native_receipt": dest.relative_to(directory).as_posix(),
                "prediction": prediction.relative_to(directory).as_posix()
                if prediction
                else None,
            }
        )
    return {
        "schema_version": SCHEMA,
        "inputs": len(events),
        "exported": counts.get("provisional", 0),
        "statuses": counts,
        "events": events,
    }


def fit(inputs, output, *, workers=3, resume=False, defer_memory=False):
    """Fit one NPZ or an NPZ directory; return the sealed, JSON-serializable result summary.

    Completed event receipts survive interruption. Resume using the same input roster,
    method, dependency versions and worker policy. Each event uses the validated bounded fit.
    """
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= MAX_WORKERS
    ):
        raise ValueError(f"workers must be an integer between 1 and {MAX_WORKERS}")
    if type(defer_memory) is not bool:
        raise ValueError("defer_memory must be a boolean")
    paths, directory = _files(inputs), Path(output).resolve()
    if not resume:
        if directory.exists():
            raise FileExistsError(
                f"Output already exists: {directory}. Choose a fresh output or resume the original run."
            )
        # Dependency/roster checks can fail before fitting. Keep the requested name
        # available so the same command can be retried after fixing the installation.
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": utc(),
            "method": _method(),
            "workers": workers,
            "defer_memory": defer_memory,
            "events": _inventory(paths),
        }
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir() or not (directory / "manifest.json").is_file():
        raise ValueError("Resume needs an existing output with a run manifest.")
    with run_lock(directory):
        if resume:
            manifest = _manifest(directory)
            _check_inputs(paths, manifest)
            _require(
                manifest["workers"] == workers, "Resume with the original worker count"
            )
            _require(
                manifest.get("defer_memory", False) == defer_memory,
                "Resume with the original memory policy",
            )
            if (directory / "seal.json").exists():
                return verify(directory)
            _require(
                _method() == manifest["method"],
                "Resume requires the original scientific source and environment",
            )
        else:
            write(directory / "manifest.json", manifest)
        primary = _run_stage(directory, paths, manifest, "primary", workers, {})
        _run_stage(directory, paths, manifest, "native", workers, primary)
        result = _summary(directory, manifest)
        write(directory / "results.json", result)
        files = (
            "manifest.json",
            "primary/seal.json",
            "native/seal.json",
            "results.json",
        )
        write(
            directory / "seal.json",
            {
                "schema_version": SCHEMA,
                "sealed_utc": utc(),
                "files": {f: sha256(directory / f) for f in files},
            },
        )
    return verify(directory)


def verify(output):
    """Rehash the sealed predictions, receipts, roster and summary; no original NPZs are required."""
    directory = Path(output).resolve()
    manifest = _manifest(directory)
    root = read(directory / "seal.json")
    expected = (
        "manifest.json",
        "primary/seal.json",
        "native/seal.json",
        "results.json",
    )
    _require(
        root["files"] == {name: sha256(directory / name) for name in expected},
        "Changed final run artifact",
    )
    primary = _verify_stage(directory, manifest, "primary", {})
    _verify_stage(directory, manifest, "native", primary)
    result = read(directory / "results.json")
    _require(
        result == _summary(directory, manifest),
        "Summary differs from sealed event receipts",
    )
    return result
