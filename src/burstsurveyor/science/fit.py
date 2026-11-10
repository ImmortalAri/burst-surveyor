"""Raw-only fitting and official prediction export."""

import json
from pathlib import Path
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from .bounded import refine
from .engine import Event, Limit, optimizer
from .input import load_raw
from .model import pulse
from .native import finalize
from .objective import ReconstructionData, choose, score


def load_config():
    return json.loads(
        Path(__file__).with_name("config.json").read_text(encoding="utf-8")
    )


def load_event(path, scratch_directory):
    """Build a raw-only native objective and its source/grid receipt."""
    raw, entry = load_raw(path, scratch_directory)
    return (ReconstructionData(Event(raw, entry["grid"], load_config())), entry)


def find_regions(d):
    """Discover islands on clipped standardized profiles. Fit and score the original pixels."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    # Share each standardized tile across the full-, half-, and quarter-band profiles.
    # Only discovery pixels are clipped. Objective pixels/weights are untouched.
    bands = [c for splits in (1, 2, 4) for c in np.array_split(d.indices, splits)]
    profiles = [np.zeros(d.e.length) for _ in bands]
    for ii in d.e.tiles(d.indices):
        y = d.e.raw[ii].astype(float)
        y -= np.median(y, axis=1, keepdims=True)
        y = np.clip(y / np.sqrt(d.var[ii, None]), -5, 5)
        for band, profile in zip(bands, profiles):
            keep = (ii >= band[0]) & (ii <= band[-1])
            if np.any(keep):
                profile += y[keep].sum(axis=0)
    candidates = []
    for channels, total in zip(bands, profiles):
        profile = total / len(channels)
        for width in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            if width > max(1, d.e.length / 9):
                break
            trace = gaussian_filter1d(profile, float(width))
            trace -= np.median(trace)
            trace /= max(1e-06, 1.4826 * np.median(abs(trace)))
            peaks = np.unique(np.r_[find_peaks(trace)[0], trace.argmax()])
            for t in peaks[trace[peaks] >= 6]:
                candidates.append(
                    (
                        float(trace[t]),
                        int(t),
                        width,
                        int(channels[0]),
                        int(channels[-1]),
                    )
                )
    # The first-stage time deduplication is retained for a matched incumbent.
    # The new spectral hook separately uses all frequency-specific profiles.
    regions = []
    for strength, t, width, lo, hi in sorted(candidates, reverse=True):
        if regions and strength < 0.12 * regions[0][0]:
            break
        if all(abs(t - old[1]) > max(3, 2 * max(width, old[2])) for old in regions):
            regions.append((strength, t, width, lo, hi))
        if len(regions) == 3:
            break
    return regions


def region_parts(d, regions):
    import numpy as np
    from .engine import Template

    parts = []
    for strength, t, sigma, lo, hi in regions:
        width = min(d.e.length, max(2, round(2.5 * sigma)))
        start = max(0, min(d.e.length - width, round(t - width / 2)))
        box = Template(
            lo // d.e.group_size, hi // d.e.group_size + 1, start, width, strength, 1.0
        )
        part = d.component_seed(box)
        part[1], part[2] = (t, float(np.log(sigma)))
        parts.extend(part)
    return parts


def fit_event(path, scratch_directory, *, mode="bounded"):
    """Fit raw intensity pixels. Return the selected row and every search receipt."""
    if mode not in {"fast", "bounded", "scintillation", "combined"}:
        raise ValueError("mode must be 'fast', 'bounded', 'scintillation' or 'combined'")
    with threadpool_limits(limits=1):
        return _fit_event(path, scratch_directory, mode)


def _fit_event(path, scratch_directory, mode):
    started = time.monotonic()
    raw, entry = load_raw(path, scratch_directory)
    stage_seconds = {"staging": time.monotonic() - started}
    stage_start = time.monotonic()
    config = load_config()
    d = ReconstructionData(Event(raw, entry["grid"], config))
    stage_seconds["data_preparation"] = time.monotonic() - stage_start
    if mode in {"scintillation", "combined"}:
        if mode == 'combined':
            from .combined import fit
        else:
            from .scintillation import fit
        stage_start = time.monotonic()
        result = fit(d, entry["source_sha256"])
        stage_seconds['predictive_selection'] = time.monotonic()-stage_start
        return dict(result, source_sha256=entry['source_sha256'], grid=entry['grid'],
                    mode=mode, fast_selected=None, bounded_review=None,
                    status='provisional' if result['selected'] else 'unresolved',
                    seconds=time.monotonic()-started, stage_seconds=stage_seconds,
                    work=dict(d.e.budget))
    stage_start = time.monotonic()
    regions = find_regions(d)
    stage_seconds["discovery"] = time.monotonic() - stage_start
    search_start = time.monotonic()
    rows, failures = ([], [])

    def trial(name, parts, scatter=False, native=False, iterations=16):
        n = (len(parts) - 1 - int(scatter)) // 5
        try:
            row = optimizer(
                d,
                n,
                scatter,
                native,
                np.array(parts),
                name,
                iterations,
                time.monotonic() + (16 if native else 10),
            )
            gain, rms, energy = score(d, row)
            if not np.isfinite([gain, rms, energy]).all():
                raise ValueError("Nonfinite native score")
            result = {
                k: row[k]
                for k in ("name", "n", "scatter", "x", "termination", "active_bounds")
            }
            result.update(gain=gain, rms_gain=rms, energy=energy)
            rows.append(result)
            return result
        except (Limit, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(dict(name=name, reason=str(exc)))
            return None

    def tail(name, parts):
        x = np.array(parts, dtype=float)
        tau = max(0.5, float(np.median(np.exp(x[2:-1:5]))) * 0.7)
        x[1:-1:5] -= tau
        return trial(name, np.r_[x, np.log(tau)], scatter=True)

    # Compare pulse shape at the same location, even when no residual peak crossed threshold.
    pos, _ = d.scan(d.y)
    if regions:
        parts = region_parts(d, regions)
    elif pos and pos.score >= 5:
        parts = d.component_seed(pos)
    else:
        parts = []
    if parts:
        x = np.r_[parts, 0.0]
        trial("profile", x)
        tail("profile_tail", x)
        if len(parts) > 5:
            # A broad tail can masquerade as several islands in a noisy profile.
            single = np.r_[parts[:5], 0.0]
            tail("brightest_tail", single)
        if rows:
            inc = choose(rows, config)
            p, q = d.scan(d.group_parts(np.array(inc["x"]), inc["n"], inc["scatter"]))
            residual = max(p.score if p else 0, q.score if q else 0)
            if residual >= 6 or inc["gain"] <= 0:
                if pos:
                    peak = np.r_[d.component_seed(pos), 0.0]
                    trial("rectangular", peak)
                    tail("rectangular_tail", peak)
                inc = choose(rows, config)
                p, _ = d.scan(
                    d.group_parts(np.array(inc["x"]), inc["n"], inc["scatter"])
                )
                if p and p.score >= 6 and inc["n"] < 4:
                    n = inc["n"]
                    extra = np.r_[
                        inc["x"][: 5 * n], d.component_seed(p), inc["x"][5 * n :]
                    ]
                    trial("residual_island", extra, scatter=inc["scatter"])
            inc = choose(rows, config)
            trial(
                "native_" + inc["name"],
                inc["x"],
                inc["scatter"],
                native=True,
                iterations=4,
            )

    selected = choose(rows, config) if rows else None
    if selected and selected["gain"] <= 0:
        selected = None
    stage_seconds["first_stage_search"] = time.monotonic() - search_start
    residual = None
    residual_pair = None
    residual_start = time.monotonic()
    if selected:
        p, q = d.scan(
            d.group_parts(np.array(selected["x"]), selected["n"], selected["scatter"])
        )
        residual = max(p.score if p else 0, q.score if q else 0)
        residual_pair = (p, q)
    stage_seconds["first_stage_residual"] = time.monotonic() - residual_start
    fast_selected = selected
    bounded_review = None
    if selected and mode == "bounded":
        bounded_start = time.monotonic()
        selected, bounded_review = refine(
            d,
            rows,
            selected,
            residual,
            source_sha256=entry["source_sha256"],
            residual_pair=residual_pair,
        )
        stage_seconds["bounded_review"] = time.monotonic() - bounded_start
        residual_start = time.monotonic()
        identical = (
            selected["n"] == fast_selected["n"]
            and selected["scatter"] == fast_selected["scatter"]
            and np.array_equal(selected["x"], fast_selected["x"])
        )
        if not identical:
            p, q = d.scan(
                d.group_parts(
                    np.array(selected["x"]), selected["n"], selected["scatter"]
                )
            )
            residual = max(p.score if p else 0, q.score if q else 0)
        stage_seconds["final_residual_scan"] = time.monotonic() - residual_start
        bounded_review["final_residual_reused"] = identical
    record = dict(
        source_sha256=entry["source_sha256"],
        grid=entry["grid"],
        selected=selected,
        mode=mode,
        fast_selected=fast_selected,
        bounded_review=bounded_review,
        regions=regions,
        trials=rows,
        failures=failures,
        residual_peak=residual,
        status="provisional" if selected else "unresolved",
        seconds=time.monotonic() - started,
        stage_seconds=stage_seconds,
        work=dict(d.e.budget),
    )
    try:
        memory = psutil.Process().memory_info()
        record["memory"] = dict(
            rss_mib=memory.rss / 2**20,
            peak_working_set_mib=getattr(memory, "peak_wset", memory.rss) / 2**20,
        )
    except (psutil.Error, OSError):
        record["memory"] = None
    return record


def export_event(path, primary, prediction_path, scratch_directory):
    """Render the selected upstream pixels and return their independently scored receipt."""
    started = time.monotonic()
    with threadpool_limits(limits=1):
        data, source = load_event(path, scratch_directory)
        if (
            source["source_sha256"] != primary["source_sha256"]
            or source["grid"] != primary["grid"]
        ):
            raise ValueError(
                "The raw bytes or acquisition grid differ from the primary fit"
            )
        row = finalize(
            data,
            primary["selected"],
            prediction_path,
            pulse_fn=pulse,
            source_sha256=source["source_sha256"],
        )
        if primary.get('mode')=='combined' and row.get('status')=='provisional':
            from .pulse_coverage import assess_pixels,residual_degradation
            with np.load(prediction_path,allow_pickle=False) as archive:
                if (str(archive['source_sha256'])!=source['source_sha256'] or
                    not np.array_equal(archive['channels'],data.indices) or
                    not np.array_equal(archive['frequency'],data.e.frequency) or
                    not np.array_equal(archive['time_s'],data.t*data.e.coords.dt)):
                    raise ValueError('Native coverage artifact grid/source mismatch')
                coverage=assess_pixels(data,archive['prediction'])
                parent=primary.get('selection_review',{}).get('pre_simplification_full_row')
                if parent:
                    row['native_simplification_guard']=residual_degradation(data,parent,archive['prediction'])
            row['native_peak_coverage']=coverage
            if not coverage['passed'] or not row.get('native_simplification_guard',{}).get('passed',True):
                # The unique generated worker artifact is not a sealed run.
                # The coordinator must not publish an inadequate prediction.
                Path(prediction_path).unlink(missing_ok=True)
                row['status']='unresolved'
                row['reasons']=list(row.get('reasons',[]))+['Stored native reconstruction fails peak coverage or residual-preservation guard']
                row['prediction_rejected']=True
                row.pop('prediction',None)
                row.pop('prediction_sha256',None)
    row["seconds"] = time.monotonic() - started
    return row
