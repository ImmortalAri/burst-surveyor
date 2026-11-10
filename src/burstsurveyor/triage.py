"""Non-destructive review priorities. Scores are screening evidence, not probabilities."""

import csv
import gc
from html import escape
import io
from pathlib import Path
import tempfile
import time
import sys

import numpy as np

from ._storage import read, run_lock, sha256, write

POLICY = dict(version=1, tentative_fold_score=2.5, maximum_windows=6)
LEVELS = {0: "unavailable", 1: "rescue", 2: "inspect", 3: "parameters", 4: "spectral", 5: "routine"}


def temporal_screen(data, signal):
    """Look below the fitting threshold, for review only; never propose a fit.

    Per-channel nuisance amplitudes remove simple spectral mismatch. The same
    signed window must recur in alternate usable channels. Band/scale searching,
    correlated channels and full-data masks make these conditional diagnostics.
    """
    from .secondlook import _residual_tile, _window_scores

    nf, nt = data.e.raw.shape
    bands = []
    for split in (1, 2, 4):
        edges = np.linspace(0, nf, split + 1, dtype=int)
        bands.extend(zip(edges[:-1], edges[1:]))
    profiles = np.zeros((len(bands), 2, nt))
    templates = np.zeros((len(bands), nt)) if signal is not None else None
    counts = np.zeros((len(bands), 2), dtype=int)
    for fold in (0, 1):
        ids = data.indices[fold::2]
        for ii in data.e.tiles(ids):
            residual = _residual_tile(data, ii, signal) / np.sqrt(data.var[ii, None])
            if signal is not None:
                template = signal[ii].astype(float)
                template -= np.median(template, axis=1, keepdims=True)
                template = np.maximum(template, 0) / np.sqrt(data.var[ii, None])
            for b, (lo, hi) in enumerate(bands):
                use = (ii >= lo) & (ii < hi)
                profiles[b, fold] += residual[use].sum(axis=0)
                counts[b, fold] += int(use.sum())
                if templates is not None:
                    templates[b] += template[use].sum(axis=0)
    inflation = np.sqrt((1 + max(0., data.rho_freq)) / max(1e-6, 1 - max(0., data.rho_freq)))
    candidates = []
    for b, (lo, hi) in enumerate(bands):
        if counts[b].min() < 6:
            continue
        p = profiles[b] / np.sqrt(counts[b, :, None]) / inflation
        width = 1
        while width <= max(1, nt // 8):
            for sign in (1, -1):
                scores = np.array([_window_scores(sign * q, width) for q in p])
                # The weaker fold determines the location, avoiding a one-fold spike.
                weakest = scores.min(axis=0)
                at = int(np.argmax(weakest))
                if weakest[at] >= POLICY["tentative_fold_score"]:
                    start, stop = at, at + width
                    fraction = (float(templates[b, start:stop].mean() / max(templates[b].max(), 1e-12))
                                if templates is not None else None)
                    candidates.append(dict(
                        sign="positive" if sign == 1 else "negative",
                        time_bins=[start, stop], channel_bins=[int(lo), int(hi)],
                        time_ms=[float((start-.5)*data.e.coords.dt*1000),
                                 float((stop-.5)*data.e.coords.dt*1000)],
                        frequency_mhz=sorted([float(data.e.frequency[lo]), float(data.e.frequency[hi-1])]),
                        fold_scores=scores[:, at].tolist(), strength=float(weakest[at]),
                        model_fraction_of_peak=fraction,
                        location="outside_fitted_emission" if fraction is not None and fraction < .1 else "within_fitted_emission" if fraction is not None else "no_model",
                    ))
            width *= 2
    kept = []
    for c in sorted(candidates, key=lambda c: -c["strength"]):
        lo, hi = c["time_bins"]
        if any(c["sign"] == k["sign"] and min(hi, k["time_bins"][1]) > max(lo, k["time_bins"][0]) for k in kept):
            continue
        kept.append(c)
        if len(kept) >= POLICY["maximum_windows"]:
            break
    return dict(status="screened" if counts.max(initial=0) >= 6 and any(counts.min(axis=1) >= 6) else "insufficient_channels",
                windows=kept, policy=POLICY,
                note="Tentative review windows only; searched scores are not significances, independent validation, or proof of an extra pulse.")


def assess(record):
    """Assess a single immutable packet; cohort ranking and human decisions are separate."""
    selected = record.get("selected") or {}
    diagnostic = record.get("diagnostics") or {}
    flags = set(record.get("flags", [])) | set(diagnostic.get("flags", []))
    reasons = []

    def add(level, code, evidence, action):
        reasons.append(dict(priority=level, code=code, evidence=evidence, action=action))

    if not selected.get("prediction"):
        add(0, "no_export", "; ".join(record.get("reasons", [])) or record.get("status", "No model"),
            "Inspect the observation and failure reason; retry or refit manually when appropriate.")
    if record.get("inspection_status") != "figure_ready" or not diagnostic:
        add(0, "inspection_incomplete", record.get("inspection_reason") or "Native figure or diagnostics unavailable.",
            "Obtain the missing inspection before accepting this event.")
    if "positive_residual_island" in flags:
        add(1, "replicated_positive_residual", "A positive temporal residual passes the discovery/check-channel screen.",
            "Inspect its time-frequency box for a missed pulse or a shape error; use the one permitted rescue if justified.")
    screen = diagnostic.get("temporal_screen") or {}
    windows = screen.get("windows", [])
    if windows:
        pos = [w for w in windows if w["sign"] == "positive"]
        neg = [w for w in windows if w["sign"] == "negative"]
        if pos:
            add(2, "tentative_positive_residual", "A weaker positive window repeats in alternate channels; below automatic rescue criteria.",
                "Inspect the marked windows for faint emission, interference or pulse-shape mismatch. Do not add a component from this flag alone.")
            if any(w.get("location") == "outside_fitted_emission" for w in pos):
                add(2, "possible_separate_emission", "A positive window lies where the band-averaged fitted signal is below 10% of its peak.",
                    "Inspect this distinct-time candidate before in-pulse shape mismatches; confirm it in the observed spectrum and rule out interference.")
        if neg:
            add(2, "replicated_negative_residual", "A negative residual window repeats in alternate channels.",
                "Check whether the model fills a real trough, is mistimed, or has the wrong width/scattering; adding a pulse may not help.")
    if screen.get("status") == "insufficient_channels":
        add(2, "limited_temporal_check", "Too few usable channels for the two-fold tentative temporal screen.",
            "Inspect the full profile manually; the temporal screen cannot clear this event.")
    for flag in ("broad_edge_component", "channel_mask_needs_review", "uncertain_offburst_noise"):
        if flag in flags:
            add(2, flag, flag.replace("_", " "), "Inspect the crop, baseline, mask and off-burst noise before interpreting components.")
    components = (diagnostic.get("component_review") or {}).get("components", [])
    for c in components:
        geometry = c.get("flags", [])
        for flag in ("sub_bin_width", "broad_component", "crop_sensitive_component"):
            if flag in geometry:
                add(3, flag, f"Component {c['component']}: {flag.replace('_', ' ')}.",
                    "Treat intrinsic width, residual DM and scattering as model-dependent; inspect resolution and crop before physical interpretation.")
    for flag in ("budget_limited", "parameter_boundary"):
        if flag in flags:
            add(3, flag, flag.replace("_", " "), "Inspect convergence and parameter limits; a retained export may need a manual refit.")
    if "spectral_model_limited" in flags:
        add(3, "spectral_model_limited", "The spectral model leaves replicated frequency structure.",
            "Inspect spectral amplitudes and temporal shape separately; extra temporal components are not the default remedy.")
    elif flags & {"frequency_amplitude_structure", "spectral_amplitude_not_replicated"}:
        add(4, "spectral_structure", "Frequency-amplitude diagnostics request inspection.",
            "Inspect frequency structure and interference without interpreting every spectral patch as a pulse.")
    if "residual_peak" in flags and not any(r["priority"] <= 2 for r in reasons):
        add(4, "residual_outlier", "The native residual contains a large peak; spatial coherence is not established.",
            "Check the residual spectrum for interference or coherent emission.")
    challenge = record.get("challenge")
    if challenge and reasons:
        reasons.append(dict(priority=min(r["priority"] for r in reasons), code="rescue_used",
                            evidence=f"One permitted rescue was attempted: {challenge.get('status')}; {challenge.get('reason')}.",
                            action="Inspect accepted and rejected trials. Persistent problems stay review; use a separate manual refit rather than repeated automatic challenges."))
    level = min((r["priority"] for r in reasons), default=5)
    return dict(priority=level, disposition=LEVELS[level], reasons=reasons,
                windows=windows, strength=max((w["strength"] for w in windows), default=0.),
                separate_emission=any(w["sign"] == "positive" and w.get("location") == "outside_fitted_emission" for w in windows),
                requires_review=level <= 3,
                note="Heuristic inspection order, not a probability of error. Routine events still need visual review; no flag certifies physical components or parameter stability.")


def ranked(records):
    rows = [dict(r, triage=assess(r)) for r in records]
    rows.sort(key=lambda r: (r["triage"]["priority"], not r["triage"]["separate_emission"], -r["triage"]["strength"], r["id"]))
    for n, row in enumerate(rows, 1):
        row["rank"] = n
    return rows


def publish(folder, records, provenance):
    """Publish a deterministic, offline queue; caller verifies source packets first."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = ranked(records)
    report = dict(schema_version=1, provenance=provenance, inputs=len(rows),
                  exported=sum(bool(r.get("selected", {}).get("prediction")) for r in rows),
                  needs_review=sum(r["triage"]["requires_review"] for r in rows), events=rows)
    write(folder / "queue.json", report)
    table = io.StringIO(newline="")
    writer = csv.writer(table)
    writer.writerow(("rank", "id", "input_name", "priority", "disposition", "components", "assessment", "reasons", "recommended_action", "figure"))
    cards = []
    for row in rows:
        t = row["triage"]
        codes = "; ".join(r["code"] for r in t["reasons"])
        action = " ".join(dict.fromkeys(r["action"] for r in t["reasons"])) or "Inspect the profile, spectrum, residuals and component table."
        count = row.get("selected", {}).get("components")
        writer.writerow((row["rank"], row["id"], row.get("input_name", row["id"]), t["priority"], t["disposition"], count, row.get("assessment", "pending"), codes, action, row.get("figure")))
        evidence = "".join(f"<li><b>{escape(r['code'])}</b>: {escape(r['evidence'])}<br>{escape(r['action'])}</li>" for r in t["reasons"])
        boxes = "".join(f"<li>{escape(w['sign'])}: {w['time_ms'][0]:.4g}–{w['time_ms'][1]:.4g} ms; {w['frequency_mhz'][0]:.4g}–{w['frequency_mhz'][1]:.4g} MHz; native bins {escape(str(w['time_bins']))}, channels {escape(str(w['channel_bins']))}; fold scores {escape(str([round(x, 2) for x in w['fold_scores']]))}</li>" for w in t["windows"])
        figure = f'<img loading="lazy" src="{escape(row["figure"], quote=True)}" alt="Observed spectrum, fitted profile, reconstruction and residual for {escape(row["id"])}">' if row.get("figure") else '<p class="warning">Figure unavailable: inspect the recorded reason.</p>'
        components = ((row.get("diagnostics") or {}).get("component_review") or {}).get("components", [])
        body = "".join('<tr>' + ''.join(f'<td>{escape(str(c.get(k)))}</td>' for k in ("component", "gaussian_arrival_ms", "gaussian_sigma_ms", "residual_dm_pc_cm3", "scattering_ms_at_reference", "flags")) + '</tr>' for c in components)
        label = row.get('input_name', row['id'])
        packet_link = f'<p><a href="{escape(row["packet"], quote=True)}">Evidence packet</a></p>' if row.get('packet') else ''
        cards.append(f'<article id="event-{escape(row["id"], quote=True)}" data-name="{escape(label, quote=True)}" data-level="{t["priority"]}"><h2>#{row["rank"]} · {escape(label)} · {escape(t["disposition"])}</h2><p>Components: {count if count is not None else "unavailable"} · Assessment: {escape(row.get("assessment", "pending visual review"))}</p>{packet_link}<ul>{evidence or "<li>No screening concern found. Visual review still required.</li>"}</ul><details><summary>Conditional residual windows</summary><ul>{boxes or "<li>None identified or unavailable.</li>"}</ul></details>{figure}<details><summary>Component parameters (Gaussian sigma, not observed pulse width)</summary><table><thead><tr><th>Component</th><th>Center ms</th><th>Sigma ms</th><th>Residual DM</th><th>Scattering ms at reference</th><th>Flags</th></tr></thead><tbody>{body}</tbody></table></details></article>')
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Burst Surveyor · Review queue</title><style>body{font:16px system-ui;background:#f4f6f8;color:#172635;max-width:1250px;margin:auto;padding:24px}article{background:white;border:1px solid #c8d2dc;border-left:6px solid #8295a6;padding:20px;margin:24px 0;border-radius:6px}article[data-level="0"],article[data-level="1"]{border-left-color:#b82d35}article[data-level="2"]{border-left-color:#d68c16}img{width:100%;height:auto}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccd5de;text-align:left}li{margin:10px 0}.warning{color:#922b32}input,select{font:inherit;padding:8px}nav{position:sticky;top:0;background:#f4f6f8;padding:12px 0}h2{font-size:1.25rem}details{margin:16px 0}</style><h1>Review queue</h1><p>Inspect the highest priorities first. These are conditional screening warnings, not probabilities or physical component counts. A quiet screen does not certify a fit. All events still need visual review; review parameters separately from pulse coverage.</p>' + f'<p>{len(rows)} inputs · {report["exported"]} exports · {report["needs_review"]} prioritized for review. Human assessments remain separate from automatic priority.</p>' + '<nav><label>Find event <input id="search" type="search"></label> <label>Show <select id="level"><option value="5">All events</option><option value="1">Unavailable / rescue</option><option value="2">Include temporal concerns</option><option value="3">Include parameter concerns</option><option value="4">Include spectral concerns</option></select></label></nav><main>' + ''.join(cards) + '</main><script>const s=document.getElementById("search"),l=document.getElementById("level");function filter(){document.querySelectorAll("article").forEach(a=>a.hidden=Number(a.dataset.level)>Number(l.value)||!(a.id+" "+a.dataset.name).toLowerCase().includes(s.value.trim().toLowerCase()))}s.addEventListener("input",filter);l.addEventListener("change",filter);</script></html>'
    (folder / "queue.csv").write_text(table.getvalue(), encoding="utf-8", newline="")
    (folder / "review.html").write_text(html, encoding="utf-8")
    seal = {name: sha256(folder / name) for name in ("queue.json", "queue.csv", "review.html")}
    write(folder / "queue-seal.json", seal)
    return dict(report=str(folder / "queue.json"), csv=str(folder / "queue.csv"), html=str(folder / "review.html"), inputs=len(rows), needs_review=report["needs_review"])


def verify_queue(output, run=None):
    """Verify queue artifacts, bound packets, figures and deterministic ordering."""
    out = Path(output).resolve()
    for name, digest in read(out / "queue-seal.json").items():
        if sha256(out / name) != digest:
            raise ValueError("Changed triage queue artifact")
    report = read(out / "queue.json")
    rows = report["events"]
    for row in rows:
        for key in ("packet", "figure"):
            if row.get(key):
                (out / row[key]).resolve().relative_to(out)
        packet = out / row["packet"]
        if sha256(packet) != row["packet_sha256"]:
            raise ValueError("Changed triage evidence packet")
        original = read(packet)
        if (original["id"] != row["id"] or original["source_sha256"] != row["source_sha256"]
                or original["selected"] != row["selected"] or original["diagnostics"] != row["diagnostics"]):
            raise ValueError("Queue differs from selected evidence packet")
        if row.get("figure") and sha256(out / row["figure"]) != original["figure_sha256"]:
            raise ValueError("Changed triage figure")
    expected = ranked(rows)
    if ([ (r["id"], r["rank"], r["triage"]) for r in rows ] !=
            [ (r["id"], r["rank"], r["triage"]) for r in expected ]
            or len({r["id"] for r in rows}) != len(rows)
            or report["inputs"] != len(rows)
            or report["exported"] != sum(bool(r["selected"].get("prediction")) for r in rows)
            or report["needs_review"] != sum(r["triage"]["requires_review"] for r in rows)):
        raise ValueError("Triage order or counts differ from evidence")
    if run is not None:
        from .batch import verify
        summary = verify(run)
        provenance = report["provenance"]
        if sha256(Path(run) / "seal.json") != provenance.get("fit_seal_sha256", provenance.get("baseline_seal_sha256")):
            raise ValueError("Queue refers to another numerical run")
        if {(r["id"], r["source_sha256"]) for r in rows} != {(r["id"], r["source_sha256"]) for r in summary["events"]}:
            raise ValueError("Queue roster differs from numerical run")
    return dict(verified=True, inputs=report["inputs"], exported=report["exported"], needs_review=report["needs_review"],
                queue_seal_sha256=sha256(out / "queue-seal.json"))


def _audit_packet(run, event, source, path, out, flags, *, retry=False):
    from .component_review import component_review
    from .diagnostics import diagnostic_data
    from .review import check_review_grid, load_for_review, plot_verified_event

    started = time.monotonic()
    native = read(run / event["native_receipt"]) if event.get("native_receipt") else {}
    selected = dict(prediction=event.get("prediction"), components=event.get("components"), parameters=native.get("fitburst_parameters"))
    record = dict(id=event["id"], input_name=event["input_name"], source_sha256=event["source_sha256"], status=event["status"], reasons=event["reasons"],
                  selected=selected, flags=flags, figure=None, figure_sha256=None, diagnostics=None, inspection_status="input_unavailable")
    with tempfile.TemporaryDirectory(dir=out) as temp:
        stage = Path(temp) / "packet"
        stage.mkdir()
        if event["status"] in {"provisional", "unresolved"}:
            try:
                data, grid = load_for_review(source, Path(temp), event["source_sha256"], wait_seconds=5)
                signal = None
                if event.get("prediction"):
                    with np.load(run / event["prediction"], allow_pickle=False) as z:
                        check_review_grid(data, grid, z)
                        signal = z["prediction"]
                diag = diagnostic_data(event, data, grid, signal)
                diag["component_review"] = component_review(selected["parameters"], data.e.frequency[data.indices], data.t * data.e.coords.dt) if selected["parameters"] else dict(components=[], flags=[])
                diag["flags"].extend(diag["component_review"]["flags"])
                plot_verified_event(run, event, source, stage / "figure.png", review_data=(data, grid, signal))
                record.update(diagnostics=diag, figure=(path / "figure.png").relative_to(out).as_posix(),
                              figure_sha256=sha256(stage / "figure.png"), inspection_status="figure_ready")
                del data, signal
            except (MemoryError, RuntimeError) as exc:
                if not isinstance(exc, MemoryError) and "available memory" not in str(exc):
                    raise
                if retry:
                    return None
                record.update(inspection_status="resource_limited", inspection_reason=str(exc))
        record["triage_seconds"] = time.monotonic() - started
        write(stage / "packet.json", record)
        seal = {"packet.json": sha256(stage / "packet.json")}
        if record.get("figure"):
            seal["figure.png"] = sha256(stage / "figure.png")
        write(stage / "seal.json", seal)
        stage.rename(path)
    return record


def audit(run, inputs, output, *, resume=False):
    """Triage a sealed numerical run without refitting or changing its exports."""
    from .batch import verify
    from .review import queue

    run, inputs, out = Path(run).resolve(), Path(inputs).resolve(), Path(output).resolve()
    summary = verify(run)
    provenance = dict(fit_seal_sha256=sha256(run / "seal.json"), policy=POLICY,
                      sources={p.relative_to(Path(__file__).parent).as_posix(): sha256(p) for p in sorted(Path(__file__).parent.rglob("*.py"))})
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise ValueError("Resume requires the original triage output")
    with run_lock(out):
        if (out / "queue-seal.json").exists():
            verify_queue(out, run)
        if (out / "method.json").exists():
            if read(out / "method.json") != provenance:
                raise ValueError("Triage sources or sealed run changed")
        else:
            write(out / "method.json", provenance)
        flags = {r["id"]: r["flags"] for r in queue(run, summary)}
        records = []
        for event in summary["events"]:
            path = out / event["id"]
            source = inputs / event["input_name"] if inputs.is_dir() else inputs
            if source.name != event["input_name"] or sha256(source) != event["source_sha256"]:
                raise ValueError("Triage input differs from sealed observation")
            if path.exists():
                if (path / "inspection").is_dir():
                    path = path / "inspection"
                record = read(path / "packet.json")
                if record["id"] != event["id"] or record["source_sha256"] != event["source_sha256"]:
                    raise ValueError("Triage packet differs from the sealed event")
                expected = {"packet.json": sha256(path / "packet.json")}
                if record.get("figure"):
                    expected["figure.png"] = sha256(out / record["figure"])
                if read(path / "seal.json") != expected:
                    raise ValueError("Changed triage evidence")
                if resume and record["inspection_status"] == "resource_limited":
                    replacement = _audit_packet(run, event, source, path / "inspection", out, flags.get(event["id"], []), retry=True)
                    if replacement is not None:
                        path = path / "inspection"
                        record = replacement
            else:
                record = _audit_packet(run, event, source, path, out, flags.get(event["id"], []))
            gc.collect()
            record = dict(record, packet=(path / "packet.json").relative_to(out).as_posix(), packet_sha256=sha256(path / "packet.json"))
            records.append(record)
            print(f"Triaged {event['id']}: {assess(record)['disposition']}", flush=True, file=sys.stderr)
        if verify(run) != summary:
            raise ValueError("Fit changed during triage")
        result = publish(out, records, provenance)
        result["verification"] = verify_queue(out, run)
        return result
