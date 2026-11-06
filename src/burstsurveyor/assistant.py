"""Assistant inspection, bounded proposals and the reviewed reconstruction report."""

import csv
import io
import os
from pathlib import Path
import tempfile

import numpy as np

from ._storage import read, run_lock, sha256, write
from .batch import fit, verify
from .diagnostics import diagnostic_data
from .review import find_event, load_for_review, plot_verified_event, queue
from .secondlook import _verified, second_look, validate_proposal


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _source(inputs, event):
    return inputs / event["input_name"] if inputs.is_dir() else inputs


def _raw(inputs, event):
    source = _source(inputs, event)
    _check(
        source.name == event["input_name"] and sha256(source) == event["source_sha256"],
        "Input name or bytes differ from the sealed observation",
    )
    return source


def _policy(workers):
    return {
        "schema_version": 1,
        "workers": workers,
        "maximum_challenges_per_event": 1,
        "review_sources": {
            n: sha256(Path(__file__).with_name(n))
            for n in ("assistant.py", "diagnostics.py", "review.py", "secondlook.py")
        },
    }


def _check_session(ws):
    policy = read(ws / "session.json")
    _check(
        policy == _policy(policy.get("workers"))
        and policy["workers"] == read(ws / "fit/manifest.json")["workers"],
        "Assistant policy, source or sealed worker count changed",
    )


def _selected(ws, event, attempt=None):
    """Select the product allowed by the numerical guard and locate its native parameters."""
    run = ws / "fit"
    if attempt and attempt["status"] == "provisional_challenger":
        path = ws / "alternatives" / event["id"] / event["id"] / "challenger.npz"
        native = attempt["native"]
        origin = "challenger"
    else:
        path = run / event["prediction"] if event["prediction"] else None
        native = read(run / event["native_receipt"])
        origin = "baseline"
    score = native.get("selected_score") or {}
    selected = native.get("selected") or {}
    return dict(
        origin=origin,
        native_status=native["status"],
        prediction=path.relative_to(ws).as_posix() if path else None,
        prediction_sha256=native.get("prediction_sha256"),
        components=selected.get("n"),
        scatter=selected.get("scatter"),
        gain=score.get("stored_gain"),
        rms_gain=score.get("stored_rms_gain"),
        parameters=native.get("fitburst_parameters"),
    )


def _packet(ws, event, source, folder, flags, attempt=None, *, retry=False):
    run = ws / "fit"
    selected = _selected(ws, event, attempt)
    entry = {**event, **{k: selected[k] for k in ("gain", "rms_gain")}}
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent, prefix=".pending-") as temp:
        stage = Path(temp) / "packet"
        stage.mkdir()
        figure = None
        display = None
        diagnostics = None
        inspection = "input_unavailable"
        reason = None
        if event["status"] in {"provisional", "unresolved"}:
            try:
                with tempfile.TemporaryDirectory(dir=temp) as scratch:
                    data, grid = load_for_review(
                        source, Path(scratch), event["source_sha256"], wait_seconds=0
                    )
                    signal = None
                    if selected["prediction"]:
                        with np.load(
                            ws / selected["prediction"], allow_pickle=False
                        ) as z:
                            signal = z["prediction"]
                    diagnostics = diagnostic_data(entry, data, grid, signal)
                    alternative = (
                        ws / "alternatives" / event["id"]
                        if selected["origin"] == "challenger"
                        else None
                    )
                    display = plot_verified_event(
                        run,
                        event,
                        source,
                        stage / "figure.png",
                        alternative=alternative,
                        review_data=(data, grid, signal),
                    )
                    figure = (folder / "figure.png").relative_to(ws).as_posix()
                    inspection = "figure_ready"
                    del data, signal
            except (MemoryError, RuntimeError) as exc:
                if not isinstance(exc, MemoryError) and "available memory" not in str(
                    exc
                ):
                    raise
                if retry:
                    return
                inspection = "resource_limited"
                reason = str(exc) or "Insufficient memory for review pixels"
                (stage / "figure.png").unlink(missing_ok=True)
        record = dict(
            schema_version=1,
            id=event["id"],
            input_name=event["input_name"],
            source_sha256=event["source_sha256"],
            baseline_seal_sha256=sha256(run / "seal.json"),
            stage="followup" if attempt else "initial",
            status=event["status"],
            reasons=event["reasons"],
            flags=flags,
            baseline_residual_peak=event["residual_peak"],
            selected=selected,
            diagnostics=diagnostics,
            figure=figure,
            display=display,
            figure_sha256=sha256(stage / "figure.png") if figure else None,
            inspection_status=inspection,
            inspection_reason=reason,
        )
        if attempt:
            record["challenge"] = {
                k: attempt.get(k)
                for k in (
                    "status",
                    "reason",
                    "proposal",
                    "termination",
                    "window_scores",
                    "channel_check",
                    "official_gain",
                    "official_channel_check",
                )
            }
        write(stage / "packet.json", record)
        write(stage / "commit.json", {"packet_sha256": sha256(stage / "packet.json")})
        stage.rename(folder)


def _open_packet(ws, event, folder):
    path = folder / "packet.json"
    record = read(path)
    digest = sha256(path)
    _check(
        read(folder / "commit.json") == {"packet_sha256": digest}
        and record["id"] == event["id"]
        and record["source_sha256"] == event["source_sha256"]
        and record["baseline_seal_sha256"] == sha256(ws / "fit/seal.json"),
        "Changed assistant review packet",
    )
    if record["figure"]:
        figure = folder / "figure.png"
        _check(
            record["figure"] == figure.relative_to(ws).as_posix()
            and sha256(figure) == record["figure_sha256"],
            "Changed assistant figure",
        )
    else:
        _check(
            (
                record["inspection_status"] == "resource_limited"
                or record["status"] not in {"provisional", "unresolved"}
            )
            and record["figure_sha256"] is None,
            "Missing review figure",
        )
    return record, dict(
        id=event["id"],
        input_name=event["input_name"],
        source_sha256=event["source_sha256"],
        status=event["status"],
        packet=path.relative_to(ws).as_posix(),
        packet_sha256=digest,
        figure=record["figure"],
        inspection_status=record["inspection_status"],
    )


def _base(ws, event):
    root = ws / "review" / event["id"]
    folder = root / "inspection" if (root / "inspection").exists() else root / "initial"
    return _open_packet(ws, event, folder)


def _index(ws, summary):
    descriptors = []
    for event in summary["events"]:
        _, initial = _open_packet(ws, event, ws / "review" / event["id"] / "initial")
        _, entry = _base(ws, event)
        entry["initial_packet_sha256"] = initial["packet_sha256"]
        descriptors.append(entry)
    return dict(
        schema_version=1,
        baseline_seal_sha256=sha256(ws / "fit/seal.json"),
        events=descriptors,
    )


def prepare(inputs, workspace, *, workers=3, resume=False):
    """Produce the sealed fit and evidence packets for assistant inspection."""
    inputs, ws = Path(inputs).resolve(), Path(workspace).resolve()
    if not resume:
        ws.mkdir(parents=True, exist_ok=False)
    elif not (ws / "session.json").is_file():
        raise ValueError("Resume needs the original assistant workspace")
    with run_lock(ws):
        policy = _policy(workers)
        if resume:
            _check(
                read(ws / "session.json") == policy,
                "Assistant policy, sources or worker configuration changed",
            )
        else:
            write(ws / "session.json", policy)
        summary = fit(
            inputs,
            ws / "fit",
            workers=workers,
            resume=(ws / "fit/manifest.json").is_file(),
            defer_memory=True,
        )
        flags = {row["id"]: row["flags"] for row in queue(ws / "fit", summary)}
        index_path = ws / "review/index.json"
        old = read(index_path) if index_path.exists() else None
        for event in summary["events"]:
            folder = ws / "review" / event["id"] / "initial"
            if not folder.exists():
                _packet(
                    ws,
                    event,
                    _source(inputs, event),
                    folder,
                    flags.get(event["id"], []),
                )
            elif resume:
                sealed, _ = _open_packet(ws, event, folder)
                parent = folder.parent
                if (
                    sealed["inspection_status"] == "resource_limited"
                    and not (parent / "inspection").exists()
                    and not (parent / "decision.json").exists()
                    and not (parent / "challenge.json").exists()
                ):
                    _packet(
                        ws,
                        event,
                        _source(inputs, event),
                        parent / "inspection",
                        flags.get(event["id"], []),
                        retry=True,
                    )
        index = _index(ws, summary)
        if old and old != index:
            # Only previously resource-limited, undecided packets may acquire retry evidence.
            _check(
                old["baseline_seal_sha256"] == index["baseline_seal_sha256"]
                and len(old["events"]) == len(index["events"]),
                "Changed initial review roster",
            )
            for a, b in zip(old["events"], index["events"]):
                if a != b:
                    event = find_event(summary, b["id"])
                    _, prior = _open_packet(
                        ws, event, ws / "review" / event["id"] / "initial"
                    )
                    prior["initial_packet_sha256"] = prior["packet_sha256"]
                    _check(
                        a == prior
                        and a["inspection_status"] == "resource_limited"
                        and b["inspection_status"] == "figure_ready"
                        and not (ws / "report-seal.json").exists(),
                        "Changed initial review index",
                    )
        if old != index:
            write(index_path, index)
        followups = []
        for event in summary["events"]:
            folder = ws / "review" / event["id"]
            if (folder / "challenge.json").is_file() and not (
                folder / "decision.json"
            ).exists():
                if (folder / "followup").is_dir():
                    _, desc = _open_packet(ws, event, folder / "followup")
                    followups.append(desc)
                else:
                    followups.append(
                        dict(
                            id=event["id"],
                            resume_request=(folder / "challenge.json")
                            .relative_to(ws)
                            .as_posix(),
                        )
                    )
        return dict(
            workspace=str(ws),
            inputs=summary["inputs"],
            exported=summary["exported"],
            index=str(index_path),
            followups=followups,
            pending=[
                e["id"]
                for e in summary["events"]
                if not (ws / "review" / e["id"] / "decision.json").exists()
            ],
        )


def _validate(action, event, descriptor):
    fields = {"id", "source_sha256", "packet_sha256", "action", "observation"}
    if action.get("action") == "challenge":
        fields.update(("time_bins", "channel_bins"))
    _check(
        set(action) == fields,
        "Expected id, source_sha256, packet_sha256, action, observation and box coordinates only for a challenge",
    )
    _check(
        action["id"] == event["id"]
        and action["source_sha256"] == event["source_sha256"]
        and action["packet_sha256"] == descriptor["packet_sha256"],
        "Decision refers to a different event or review packet",
    )
    _check(
        isinstance(action["observation"], str) and bool(action["observation"].strip()),
        "Record what you observed in the figure, diagnostics or input status",
    )
    _check(
        action["action"] in {"retain", "challenge", "review", "unresolved"},
        "Unknown assistant action",
    )


def _proposal(action, event, shape):
    packet = dict(
        schema_version=1,
        event_id=event["id"],
        source_sha256=event["source_sha256"],
        action="add_component",
        time_bins=action["time_bins"],
        channel_bins=action["channel_bins"],
    )
    validate_proposal(packet, event, shape)
    return packet


def _challenge(ws, source, event, action, summary, flags):
    root = ws / "review" / event["id"]
    _check(
        event["status"] in {"provisional", "unresolved"},
        "This event cannot be challenged",
    )
    roster = read(ws / "fit/manifest.json")["events"]
    shape = next(x["shape"] for x in roster if x["id"] == event["id"])
    proposal = _proposal(action, event, shape)
    request = root / "challenge.json"
    if request.exists():
        _check(
            read(request) == action, "Only one fixed proposal is permitted per event"
        )
    else:
        write(request, action)
    alt = ws / "alternatives" / event["id"]
    second_look(
        ws / "fit",
        source,
        alt,
        proposal=proposal,
        resume=(alt / "method.json").exists(),
    )
    _verified(ws / "fit", alt, summary)
    attempt = read(alt / event["id"] / "decision.json")
    follow = root / "followup"
    if not follow.exists():
        _packet(ws, event, source, follow, flags, attempt)
    _, next_packet = _open_packet(ws, event, follow)
    return {
        **next_packet,
        "pending": "inspect_followup",
        "challenge_result": attempt["status"],
        "reason": attempt["reason"],
    }


def _final_action(action, packet):
    if packet["inspection_status"] == "resource_limited":
        _check(
            action["action"] == "review",
            "Figure unavailable; mark this event for review",
        )
    elif action["action"] == "retain":
        _check(
            packet["selected"]["prediction"] is not None,
            "Use unresolved or review for an event with no export",
        )
    elif action["action"] == "unresolved":
        _check(
            packet["selected"]["prediction"] is None,
            "Use review for an ambiguous exported model",
        )
    else:
        _check(action["action"] == "review", "A final assessment is required")


def decide(workspace, inputs, decision):
    """Record one or several explicit assistant assessments. Challenge outcomes need a final assessment."""
    ws, inputs = Path(workspace).resolve(), Path(inputs).resolve()
    if isinstance(decision, (Path, str)):
        decision = read(decision)
    actions = decision if isinstance(decision, list) else [decision]
    _check(
        actions and all(isinstance(a, dict) for a in actions),
        "Expected an assessment object or nonempty array",
    )
    with run_lock(ws):
        summary = verify(ws / "fit")
        _check_session(ws)
        index = read(ws / "review/index.json")
        _check(index == _index(ws, summary), "Changed initial assistant review index")
        done = []
        for action in actions:
            event = find_event(summary, action.get("id"))
            source = _raw(inputs, event)
            root = ws / "review" / event["id"]
            request = root / "challenge.json"
            if action.get("action") == "challenge":
                _check(
                    not (root / "decision.json").exists(),
                    "Event already has a final assessment",
                )
                entry, desc = _base(ws, event)
                _check(
                    entry["inspection_status"] == "figure_ready",
                    "Inspect a figure before challenging an event",
                )
                _validate(action, event, desc)
                done.append(
                    _challenge(ws, source, event, action, summary, entry["flags"])
                )
                continue
            if request.exists():
                _check(
                    (root / "followup").exists(),
                    "Incomplete challenge; resubmit its unchanged request",
                )
                packet, desc = _open_packet(ws, event, root / "followup")
            else:
                packet, desc = _base(ws, event)
            _validate(action, event, desc)
            _final_action(action, packet)
            path = root / "decision.json"
            if path.exists():
                _check(read(path) == action, "Final assistant assessment is immutable")
            else:
                _check(
                    not (ws / "report-seal.json").exists(),
                    "Assistant report already finalized",
                )
                write(path, action)
            done.append(dict(id=event["id"], action=action["action"], pending=None))
        return dict(decisions=done)


def _report_row(ws, event, summary):
    root = ws / "review" / event["id"]
    baseline, initial = _base(ws, event)
    request = root / "challenge.json"
    attempt = None
    receipts = {"initial": initial["packet_sha256"]}
    if request.exists():
        action = read(request)
        _validate(action, event, initial)
        alt = ws / "alternatives" / event["id"]
        _verified(ws / "fit", alt, summary)
        attempt = read(alt / event["id"] / "decision.json")
        proposal = read(alt / "method.json")["proposal"]
        _check(
            action["action"] == "challenge"
            and action["time_bins"] == proposal["time_bins"]
            and action["channel_bins"] == proposal["channel_bins"],
            "Changed challenge proposal",
        )
        receipts.update(
            challenge=sha256(request), alternate_seal=sha256(alt / "seal.json")
        )
        packet, desc = _open_packet(ws, event, root / "followup")
        receipts["followup"] = desc["packet_sha256"]
    else:
        packet, desc = baseline, initial
    selected = _selected(ws, event, attempt)
    _check(
        packet["selected"] == selected,
        "Packet differs from the numerically selected reconstruction",
    )
    final = read(root / "decision.json")
    _validate(final, event, desc)
    _final_action(final, packet)
    receipts["assessment"] = sha256(root / "decision.json")
    return dict(
        id=event["id"],
        input_name=event["input_name"],
        source_sha256=event["source_sha256"],
        status=selected["native_status"],
        assessment=final["action"],
        observation=final["observation"],
        figure=packet["figure"],
        flags=packet["flags"],
        diagnostics=packet["diagnostics"],
        inspection_status=packet["inspection_status"],
        inspection_reason=packet["inspection_reason"],
        baseline_packet=initial["packet"],
        selected_packet=desc["packet"],
        challenge=read(request) if request.exists() else None,
        challenge_reason=attempt["reason"] if attempt else None,
        selected=selected,
        receipts=receipts,
    )


def finish(workspace):
    """Verify all numerical choices and user-agent assessments, then seal the reviewed report."""
    ws = Path(workspace).resolve()
    with run_lock(ws):
        summary = verify(ws / "fit")
        _check_session(ws)
        _check(
            read(ws / "review/index.json") == _index(ws, summary),
            "Changed initial assistant review index",
        )
        missing = [
            e["id"]
            for e in summary["events"]
            if not (ws / "review" / e["id"] / "decision.json").is_file()
        ]
        _check(not missing, "Pending assistant assessments: " + ", ".join(missing))
        rows = [_report_row(ws, event, summary) for event in summary["events"]]
        report = dict(
            schema_version=1,
            baseline_seal_sha256=sha256(ws / "fit/seal.json"),
            review_index_sha256=sha256(ws / "review/index.json"),
            inputs=len(rows),
            exported=sum(r["selected"]["prediction"] is not None for r in rows),
            events=rows,
        )
        table = io.StringIO(newline="")
        fields = (
            "id",
            "input_name",
            "source_sha256",
            "status",
            "assessment",
            "observation",
            "origin",
            "components",
            "scatter",
            "gain",
            "rms_gain",
            "prediction",
            "figure",
            "challenge_reason",
        )
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: row.get(key, row["selected"].get(key)) for key in fields}
            )
        encoded = table.getvalue().encode("utf-8")
        if (ws / "report-seal.json").exists():
            _check(
                read(ws / "report.json") == report
                and (ws / "report.csv").read_bytes() == encoded,
                "Changed final assistant report",
            )
        else:
            write(ws / "report.json", report)
            temp = ws / ".report.tmp"
            temp.write_bytes(encoded)
            os.replace(temp, ws / "report.csv")
        seal = {
            key: sha256(ws / key)
            for key in (
                "report.json",
                "report.csv",
                "review/index.json",
                "session.json",
                "fit/seal.json",
            )
        }
        if (ws / "report-seal.json").exists():
            _check(
                read(ws / "report-seal.json") == seal,
                "Changed final assistant report seal",
            )
        else:
            write(ws / "report-seal.json", seal)
        return dict(
            report=str(ws / "report.json"),
            csv=str(ws / "report.csv"),
            inputs=report["inputs"],
            exported=report["exported"],
            seal_sha256=sha256(ws / "report-seal.json"),
        )
