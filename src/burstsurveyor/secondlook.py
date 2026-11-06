"""Experimental, opt-in temporal rescue. All outputs are separate from the sealed fit."""

import math
from pathlib import Path
import tempfile
import time

import numpy as np

from ._storage import read, run_lock, sha256, write
from .review import find_event, queue

POLICY = {
    "method": "positive-island-1",
    "positive_discovery": 6.0,
    "positive_check": 3.0,
    "validation_floor": 3.0,
    "full_gain_floor": 8.0,
    "fraction": 0.001,
    "optimizer_seconds": 6.0,
    "optimizer_iterations": 10,
}


def validate_proposal(packet, event, shape):
    fields = {
        "schema_version",
        "event_id",
        "source_sha256",
        "action",
        "time_bins",
        "channel_bins",
    }
    if not isinstance(packet, dict) or set(packet) != fields:
        raise ValueError(
            "A proposal must contain exactly the documented proposal fields"
        )
    if (
        type(packet["schema_version"]) is not int
        or packet["schema_version"] != 1
        or packet["action"] != "add_component"
    ):
        raise ValueError("Unsupported proposal schema or action")
    if (
        packet["source_sha256"] != event["source_sha256"]
        or packet["event_id"] != event["id"]
    ):
        raise ValueError("Proposal does not match the sealed event and input hash")
    for field, size in (("time_bins", shape[1]), ("channel_bins", shape[0])):
        span = packet[field]
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(v) is not int for v in span)
            or not 0 <= span[0] < span[1] <= size
        ):
            raise ValueError(
                f"{field} must be a zero-based, half-open interval on this input"
            )


def _profile(data, ids, model):
    profile = np.zeros(data.e.length)
    for ii in data.e.tiles(ids):
        centered = data.e.raw[ii].astype(float) - data.base[ii, None]
        if model is not None:
            pred = model[ii].astype(float)
            centered -= pred - pred.mean(axis=1, keepdims=True)
        profile += (centered / np.sqrt(data.var[ii, None])).sum(axis=0)
    return profile / np.sqrt(len(ids))


def _window_scores(profile, width):
    summed = np.r_[0.0, np.cumsum(profile)]
    values = (summed[width:] - summed[:-width]) / np.sqrt(width)
    center = np.median(values)
    spread = max(1.0, 1.4826 * np.median(np.abs(values - center)))
    return (values - center) / spread


def _discovery(data, train, signal, event):
    nf, nt = data.e.raw.shape
    edges = np.linspace(0, nf, 5, dtype=int)
    best = None
    for lo, hi in [(0, nf), *zip(edges[:-1], edges[1:])]:
        ids = train[(train >= lo) & (train < hi)]
        if len(ids) < 3:
            continue
        profile = _profile(data, ids, signal)
        for width in (1, 2, 4, 8, 16, 32, 64):
            if width > nt // 3:
                continue
            z = _window_scores(profile, width)
            start = int(np.argmax(z))
            if best is None or z[start] > best[0]:
                best = float(z[start]), (int(lo), int(hi)), (start, start + width)
    if best is None or best[0] < POLICY["positive_discovery"]:
        return None
    return dict(
        schema_version=1,
        event_id=event["id"],
        source_sha256=event["source_sha256"],
        action="add_component",
        channel_bins=list(best[1]),
        time_bins=list(best[2]),
    )


def _evidence(data, ids, model, packet):
    lo, hi = packet["channel_bins"]
    first, last = packet["time_bins"]
    ii = ids[(ids >= lo) & (ids < hi)]
    if len(ii) < 3:
        return None
    return float(_window_scores(_profile(data, ii, model), last - first)[first])


def _gain(data, row, ids):
    from .science.model import pulse

    gain = rms = 0.0
    if row:
        for ii in data.e.tiles(ids):
            pred = pulse(
                np.asarray(row["x"]),
                row["n"],
                row["scatter"],
                data.e.coords,
                ii,
                data.t,
            )[0]
            pred -= pred.mean(axis=1, keepdims=True)
            y = data.e.raw[ii].astype(float) - data.base[ii, None]
            term = pred * (2 * y - pred)
            gain += float(np.sum(term / data.var[ii, None]))
            rms += float(np.sum(term / data.rms_var[ii, None]))
    return gain, rms


def _improves(challenger, incumbent, floor):
    return all(
        math.isfinite(c) and c > i + max(floor, POLICY["fraction"] * abs(i))
        for c, i in zip(challenger, incumbent)
    )


def _pixel_gains(data, model, ids):
    total = np.zeros(2)
    if model is not None:
        for ii in data.e.tiles(ids):
            y = data.e.raw[ii].astype(float) - data.base[ii, None]
            p = model[ii].astype(float)
            p -= p.mean(axis=1, keepdims=True)
            term = p * (2 * y - p)
            total += [
                np.sum(term / data.var[ii, None]),
                np.sum(term / data.rms_var[ii, None]),
            ]
    return total.tolist()


def _fit_direction(data, train, incumbent, packet, strength):
    from .science.engine import Template, optimizer
    from .science.objective import ReconstructionData

    training = ReconstructionData(data.e, channels=train)
    t0, t1 = packet["time_bins"]
    lo, hi = packet["channel_bins"]
    box = Template(
        lo // data.e.group_size,
        (hi - 1) // data.e.group_size + 1,
        t0,
        t1 - t0,
        strength,
        1.0,
    )
    extra = training.component_seed(box)
    n = incumbent["n"] if incumbent else 0
    x = np.array(incumbent["x"] if incumbent else [0.0], dtype=float)
    seed = np.r_[x[: 5 * n], extra, x[5 * n :]]
    scatter = bool(incumbent and incumbent["scatter"])
    # Only the discovery channels can adjust the proposed geometry.
    return optimizer(
        training,
        n + 1,
        scatter,
        True,
        seed,
        "second_look",
        POLICY["optimizer_iterations"],
        time.monotonic() + POLICY["optimizer_seconds"],
    )


def _attempt(source, event, native, signal, packet, directory):
    from .science import load_event
    from .science.engine import Limit
    from .science.objective import score
    from .science.native import finalize
    from .science.model import pulse

    result = dict(
        id=event["id"],
        source_sha256=event["source_sha256"],
        status="retained",
        reason=None,
        prediction=None,
        proposal_origin="external" if packet else "automatic",
    )

    def stop(reason):
        result["reason"] = reason
        return result

    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        data, entry = load_event(source, Path(temporary))
        if entry["source_sha256"] != event["source_sha256"]:
            raise ValueError("Second-look input bytes differ from sealed input")
        if packet is not None:
            validate_proposal(packet, event, data.e.raw.shape)
        if len(data.indices) < 12:
            return stop("insufficient_channels_for_split")
        train, held = data.indices[::2], data.indices[1::2]
        incumbent = native.get("selected") if native else None
        if incumbent and incumbent["n"] >= data.c["max_components"]:
            return stop("component_budget")
        if packet is None:
            packet = _discovery(data, train, signal, event)
        if packet is None:
            return stop("no_positive_coherent_island")
        result["proposal"] = packet
        d_z, h_z = (_evidence(data, ids, signal, packet) for ids in (train, held))
        result["window_scores"] = dict(discovery=d_z, check=h_z)
        if (
            d_z is None
            or h_z is None
            or d_z < POLICY["positive_discovery"]
            or h_z < POLICY["positive_check"]
        ):
            return stop("weak_or_unreplicated_island")
        try:
            trial = _fit_direction(data, train, incumbent, packet, d_z)
        except (Limit, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            result["detail"] = str(exc)
            return stop("trial_unavailable")
        result["termination"] = trial["termination"]
        if trial["termination"] in {"wall_limit", "work_limit"}:
            return stop("optimizer_budget")
        check_old, check_new = _gain(data, incumbent, held), _gain(data, trial, held)
        if not np.isfinite([*check_old, *check_new]).all():
            return stop("nonfinite_trial")
        result["channel_check"] = dict(
            channels=len(held), incumbent=check_old, proposed=check_new
        )
        if not _improves(check_new, check_old, POLICY["validation_floor"]):
            return stop("no_material_check_channel_gain")
        gain, rms, energy = score(data, trial)
        row = {
            k: trial[k]
            for k in ("n", "scatter", "x", "name", "termination", "active_bounds")
        }
        row["x"] = np.asarray(row["x"]).tolist()
        row.update(gain=float(gain), rms_gain=float(rms), energy=float(energy))
        incumbent_full = _gain(data, incumbent, data.indices)
        if not _improves((gain, rms), incumbent_full, POLICY["full_gain_floor"]):
            return stop("no_material_native_gain")
        # Render the challenger independently on the original NPZ. Upstream pixel scores own final promotion.
        path = Path(temporary) / "challenger.npz"
        receipt = finalize(
            data, row, path, pulse_fn=pulse, source_sha256=entry["source_sha256"]
        )
        if receipt["status"] != "provisional":
            result["export_reasons"] = receipt["reasons"]
            return stop("official_export_refused")
        base_score = (
            native["selected_score"]
            if native and native.get("selected_score")
            else dict(stored_gain=0.0, stored_rms_gain=0.0)
        )
        new_score = receipt["selected_score"]
        measures = ("stored_gain", "stored_rms_gain")
        result["official_gain"] = dict(
            incumbent=[base_score[k] for k in measures],
            proposed=[new_score[k] for k in measures],
        )
        if not _improves(
            result["official_gain"]["proposed"],
            result["official_gain"]["incumbent"],
            POLICY["full_gain_floor"],
        ):
            return stop("official_gain_guard")
        with np.load(path, allow_pickle=False) as exported:
            held_new = _pixel_gains(data, exported["prediction"], held)
        held_old = _pixel_gains(data, signal, held)
        result["official_channel_check"] = dict(incumbent=held_old, proposed=held_new)
        if not _improves(held_new, held_old, POLICY["validation_floor"]):
            return stop("official_check_channel_guard")
        dest = directory / "challenger.npz"
        path.replace(dest)
        result.update(
            status="provisional_challenger",
            reason="material_gain_on_both_objectives",
            prediction=dest.name,
            prediction_sha256=sha256(dest),
            native=receipt,
        )
        return result


def _decision(folder, event):
    record = read(folder / "decision.json")
    artifact = record.get("prediction")
    expected = {"decision.json": sha256(folder / "decision.json")}
    if record["id"] != event["id"] or record["source_sha256"] != event["source_sha256"]:
        raise ValueError("Detached second-look decision")
    if record["status"] == "provisional_challenger":
        if (
            artifact != "challenger.npz"
            or sha256(folder / artifact) != record["prediction_sha256"]
        ):
            raise ValueError("Changed second-look prediction")
        native = record["native"]
        if (
            native["source_sha256"] != event["source_sha256"]
            or native["prediction_sha256"] != record["prediction_sha256"]
        ):
            raise ValueError("Detached upstream receipt")
        expected[artifact] = record["prediction_sha256"]
    elif record["status"] != "retained" or artifact is not None:
        raise ValueError("Invalid second-look disposition")
    if read(folder / "commit.json") != expected or {
        f.name for f in folder.iterdir()
    } != set(expected) | {"commit.json"}:
        raise ValueError("Changed second-look event commitment")
    return dict(
        id=event["id"],
        status=record["status"],
        reason=record["reason"],
        decision_sha256=expected["decision.json"],
        commit_sha256=sha256(folder / "commit.json"),
    )


def _verified(run, output, summary):
    plan, seal = read(output / "method.json"), read(output / "seal.json")
    if (
        plan["baseline_seal_sha256"] != sha256(run / "seal.json")
        or plan["science"] != read(run / "manifest.json")["method"]
    ):
        raise ValueError("Challenger belongs to a different baseline method or seal")
    ids = plan["event_ids"]
    if len(set(ids)) != len(ids) or {
        p.name
        for p in output.iterdir()
        if p.is_dir() and not p.name.startswith(".pending-")
    } != set(ids):
        raise ValueError("Incomplete or unexpected second-look membership")
    records = [_decision(output / key, find_event(summary, key)) for key in ids]
    if seal != dict(
        schema_version=1, method_sha256=sha256(output / "method.json"), events=records
    ):
        raise ValueError("Changed sealed second-look results")
    return seal


def verify_second_look(run, output):
    """Verify a completed alternate run without loading the original data."""
    from .batch import verify

    run, output = Path(run).resolve(), Path(output).resolve()
    return _verified(run, output, verify(run))


def second_look(run, inputs, output, *, proposal=None, limit=8, resume=False):
    """Evaluate at most one additional pulse per routed event into a separate output."""
    import psutil
    from .batch import _method, verify
    from .science import inspect_npz
    from ._dispatch import estimate_bytes, RESERVE_MIB, MIB
    from threadpoolctl import threadpool_limits

    if type(limit) is not int or limit < 1:
        raise ValueError("Second-look event limit must be positive")
    run, inputs, output = (
        Path(run).resolve(),
        Path(inputs).resolve(),
        Path(output).resolve(),
    )
    if output == run or run in output.parents:
        raise ValueError("Use a second-look output outside the sealed run")
    summary = verify(run)
    if isinstance(proposal, (Path, str)) and Path(proposal).stat().st_size > 65536:
        raise ValueError("A proposal JSON file must be 64 KiB or smaller")
    packet = read(Path(proposal)) if isinstance(proposal, (Path, str)) else proposal
    if packet is not None and not isinstance(packet, dict):
        raise ValueError("Expected a JSON proposal object")
    if packet is not None:
        events = [find_event(summary, packet.get("event_id"))]
        roster = read(run / "manifest.json")["events"]
        entry = next(row for row in roster if row["id"] == events[0]["id"])
        if "shape" not in entry or events[0]["status"] not in {
            "provisional",
            "unresolved",
        }:
            raise ValueError("This event has no usable baseline for a proposal")
        validate_proposal(packet, events[0], entry["shape"])
    else:
        review = queue(run, summary)
        events = [
            find_event(summary, r["id"])
            for r in review
            if r["status"] in {"provisional", "unresolved"}
        ][:limit]
    plan = dict(
        schema_version=1,
        baseline_seal_sha256=sha256(run / "seal.json"),
        policy=POLICY,
        event_ids=[e["id"] for e in events],
        proposal=packet,
        science=_method(),
        controller_sources={
            name: sha256(Path(__file__).with_name(name))
            for name in ("secondlook.py", "review.py")
        },
    )
    if plan["science"] != read(run / "manifest.json")["method"]:
        raise ValueError(
            "Use the same installed scientific method and environment as the sealed baseline"
        )
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    if not output.is_dir() or (resume and not (output / "method.json").is_file()):
        raise ValueError(
            "Resume requires the original second-look directory and method"
        )
    with run_lock(output), threadpool_limits(limits=1):
        method = output / "method.json"
        if method.exists():
            if read(method) != plan:
                raise ValueError(
                    "Second-look baseline, proposal, policy or selected events changed"
                )
        else:
            write(method, plan)
        if (output / "seal.json").exists():
            return _verified(run, output, summary)
        items = []
        for event in events:
            dest = output / event["id"]
            file = dest / "decision.json"
            if not file.exists():
                with tempfile.TemporaryDirectory(
                    prefix=".pending-", dir=output
                ) as temp:
                    directory = Path(temp) / "event"
                    directory.mkdir()
                    start = time.monotonic()
                    source = inputs / event["input_name"] if inputs.is_dir() else inputs
                    if sha256(source) != event["source_sha256"]:
                        raise ValueError(f"Wrong raw input for {event['id']}")
                    if event["status"] not in {"provisional", "unresolved"}:
                        raise ValueError("This input has no usable baseline")
                    info = inspect_npz(source)
                    wait_until = time.monotonic() + 30
                    while (
                        psutil.virtual_memory().available
                        < estimate_bytes(info) + RESERVE_MIB * MIB
                    ):
                        if time.monotonic() >= wait_until:
                            raise RuntimeError(
                                "Second look needs more available memory. Resume when available."
                            )
                        time.sleep(1)
                    native = (
                        read(run / event["native_receipt"])
                        if event.get("native_receipt")
                        else None
                    )
                    signal = None
                    if event.get("prediction"):
                        with np.load(
                            run / event["prediction"], allow_pickle=False
                        ) as z:
                            signal = z["prediction"]
                    result = _attempt(source, event, native, signal, packet, directory)
                    result["seconds"] = time.monotonic() - start
                    write(directory / "decision.json", result)
                    write(
                        directory / "commit.json",
                        {p.name: sha256(p) for p in directory.iterdir() if p.is_file()},
                    )
                    directory.rename(dest)
                    del signal
            items.append(_decision(dest, event))
        seal = dict(schema_version=1, method_sha256=sha256(method), events=items)
        write(output / "seal.json", seal)
        return _verified(run, output, summary)
