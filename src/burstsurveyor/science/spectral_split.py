"""Raw-only asymmetric spectral/temporal proposals on an existing Data object.

Grouped representative predictions nominate and screen candidates only. Every
completed candidate is rescored by the unchanged native common/RMS scorer.
There is no Data reconstruction, raw rehash, noise estimation, pixel masking,
reference input, event identity or per-event policy in this module.
"""

import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .engine import Limit, optimizer
from .model import pulse
from .objective import score
from .amplitudes import amplitude_solve


VERSION = "spectral-split-1"


def _proxy(d, row):
    p = pulse(np.asarray(row["x"]), row["n"], row["scatter"], d.e.coords, d.reps, d.t)[
        0
    ]
    return (p - p.mean(axis=1, keepdims=True)) * d.group_norm[:, None]


def _bands(d):
    z = d.e.coords.z[d.reps]
    lo, hi = (float(z.min()), float(z.max()))
    cuts = (
        ("low_half", 0.0, 0.5),
        ("high_half", 0.5, 1.0),
        ("low_third", 0.0, 1 / 3),
        ("middle_third", 1 / 3, 2 / 3),
        ("high_third", 2 / 3, 1.0),
        ("low_two_thirds", 0.0, 2 / 3),
        ("high_two_thirds", 1 / 3, 1.0),
        ("full", 0.0, 1.0),
    )
    out = {}
    for name, a, b in cuts:
        mask = (z >= lo + a * (hi - lo) - 1e-12) & (z <= lo + b * (hi - lo) + 1e-12)
        if np.count_nonzero(mask) >= 2:
            out[name] = mask
    return out


def _project(d, plane, mask):
    w = d.group_norm[mask]
    return w @ plane[mask] / max(float(np.linalg.norm(w)), 1e-30)


def _widths(d, incumbent):
    x = np.asarray(incumbent["x"])
    sigmas = np.exp(x[2 : 5 * incumbent["n"] : 5])
    scale = float(np.median(sigmas))
    tail = float(np.exp(x[-1])) if incumbent["scatter"] else 0.0
    maximum = max(0.5, min(128.0, len(d.t) / 5))
    values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 0.5 * scale, scale, 2 * scale, tail]
    return sorted({float(np.clip(v, 0.5, maximum)) for v in values if v > 0})


def _peak(d, profile, widths, window=None):
    """Empirically scaled grouped matched-filter diagnostic, not a p-value."""
    best = None
    eligible = np.ones(len(profile), bool) if window is None else window
    if not np.any(eligible):
        return dict(
            score=0.0,
            arrival=float(d.t[0]),
            sigma=float(widths[0]),
            off_scale=1.0,
            eligible=False,
        )
    off = np.asarray(d.noise_time)[np.asarray(d.train)]
    for sigma in widths:
        radius = max(2, int(np.ceil(4 * sigma)))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        filtered = gaussian_filter1d(profile, sigma, mode="constant", truncate=4.0)
        filtered /= max(float(np.linalg.norm(kernel)), 1e-30)
        sample = filtered[off]
        middle = float(np.median(sample)) if len(sample) else 0.0
        spread = 1.4826 * float(np.median(abs(sample - middle))) if len(sample) else 1.0
        normalized = (filtered - middle) / max(1.0, spread)
        index = int(np.argmax(np.where(eligible, normalized, -np.inf)))
        item = dict(
            score=float(normalized[index]),
            arrival=float(d.t[index]),
            sigma=float(sigma),
            off_scale=max(1.0, spread),
        )
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def _align(d, plane, dm):
    shifted = np.empty_like(plane)
    for g, ii in enumerate(d.reps):
        shifted[g] = np.interp(
            d.t + dm * d.e.coords.dispersion[ii], d.t, plane[g], left=0.0, right=0.0
        )
    return shifted


def _spectral_component(d, mask, peak, scatter, tau):
    z = d.e.coords.z[d.reps[mask]]
    weights = d.group_norm[mask] ** 2
    pivot = float(np.average(z, weights=weights))
    spread = max(0.08, float(np.ptp(z)) * 0.5)
    curvature = max(float(d.c.get("curvature_min", -150.0)), -0.5 / spread**2)
    slope = -2 * curvature * pivot
    sigma, arrival = (peak["sigma"], peak["arrival"])
    if scatter:
        band_tau = tau * float(
            np.average(d.e.coords.tau_ratio[d.reps[mask]], weights=weights)
        )
        arrival -= 0.55 * band_tau
        sigma = np.sqrt(max(0.5**2, sigma**2 - (0.6 * band_tau) ** 2))
    sigma = max(float(d.c.get("sigma_min", 0.065)), sigma)
    # NNLS will set amplitudes jointly. The initial maximum in-band fluence is
    # finite and positive even when the reference lies outside this passband.
    log_peak = np.log(max(d.fluence_scale, 1e-30) * max(1.0, sigma))
    amplitude = log_peak + curvature * pivot * pivot
    return np.array([amplitude, arrival, np.log(sigma), slope, curvature])


def _family_row(components, dm, scatter, tau, name):
    x = np.r_[np.asarray(components).ravel(), dm]
    if scatter:
        x = np.r_[x, np.log(max(0.018, tau))]
    return dict(name=name, n=len(components), scatter=bool(scatter), x=x.tolist())


def _choose(incumbent, candidates):
    safe = [incumbent] + [
        r
        for r in candidates
        if r["gain"] >= incumbent["gain"] and r["rms_gain"] >= incumbent["rms_gain"]
    ]
    best = max(safe, key=lambda r: r["gain"])
    ceiling = max(0.02, 0.001 * max(0.0, best["gain"]))
    return min(
        (r for r in safe if r["gain"] >= best["gain"] - ceiling),
        key=lambda r: (5 * r["n"] + int(r["scatter"]), -r["gain"], r.get("name", "")),
    )


def _timed(review, category, action, trial, field):
    started = time.monotonic()
    try:
        return action()
    finally:
        elapsed = time.monotonic() - started
        trial[field] = elapsed
        review["timing"][category] += elapsed


def refine(d, rows, incumbent, residual, deadline):
    """Return (protected selection, full review), appending completed candidates.

    ``deadline`` is absolute monotonic time. ``residual`` may be a cached exact
    grouped residual plane, a dict with ``plane``/``peak``, a scalar peak, or None.
    Four grouped geometry/family trials and at most two native family trials
    share this deadline; numerical calls may overrun it.
    All masks and authoritative weights come from the caller's existing d.
    """
    begin = time.monotonic()
    selected = incumbent
    review = dict(
        version=VERSION,
        status="not_started",
        routed=False,
        elapsed_seconds=0.0,
        candidates=[],
        trials=[],
        seed_prices=[],
        failures=[],
        timing=dict(
            diagnostics_seconds=0.0,
            grouped_seed_seconds=0.0,
            grouped_refit_seconds=0.0,
            native_refit_seconds=0.0,
            native_score_seconds=0.0,
        ),
        objective="caller native common and alternate RMS; grouped nomination only",
        mask_changed=False,
        preprocessing_reused=True,
        interpretation="In-sample reconstruction protection; component identity and known-truth error remain provisional.",
    )
    budget_before = dict(d.e.budget)
    candidates = []
    active_stage = None
    stage_started = begin
    try:
        if begin >= deadline:
            review["status"] = "budget_expired"
            return (selected, review)
        n = int(incumbent["n"])
        if n >= int(d.c.get("max_components", 7)):
            review["status"] = "component_budget"
            return (selected, review)
        x = np.asarray(incumbent["x"], float)
        if (
            x.shape != (5 * n + 1 + int(incumbent["scatter"]),)
            or not np.isfinite(x).all()
        ):
            raise ValueError("invalid incumbent")
        start = time.monotonic()
        active_stage, stage_started = ("diagnostics_seconds", start)
        components = x[: 5 * n].reshape(n, 5)
        planes = [
            _proxy(
                d,
                _family_row(
                    [part],
                    x[5 * n],
                    incumbent["scatter"],
                    np.exp(x[-1]) if incumbent["scatter"] else 0.5,
                    "component",
                ),
            )
            for part in components
        ]
        model = np.sum(planes, axis=0)
        approximate_loss = float(np.sum((d.y - model) ** 2))
        supplied = residual.get("plane") if isinstance(residual, dict) else residual
        if isinstance(supplied, np.ndarray) and supplied.shape == d.y.shape:
            plane = supplied
            review["residual_source"] = "cached_exact_grouped_plane"
        else:
            plane = d.y - model
            review["residual_source"] = "representative_proxy"
        bands = _bands(d)
        widths = _widths(d, incumbent)
        aligned_residual = _align(d, plane, x[5 * n])
        # The raw-derived signal exclusion was already estimated by the caller.
        # Carry it into reference-time coordinates without estimating noise again.
        observed_signal = ~np.asarray(d.noise_time)[np.asarray(d.train)]
        signal_window = np.zeros(len(d.t), bool)
        for ii in d.reps:
            signal_window |= (
                np.interp(
                    d.t + x[5 * n] * d.e.coords.dispersion[ii],
                    d.t,
                    observed_signal.astype(float),
                    left=0.0,
                    right=0.0,
                )
                > 0.5
            )
        in_peaks, off_peaks, band_peaks = ({}, {}, {})
        for name, mask in bands.items():
            profile = _project(d, aligned_residual, mask)
            in_peaks[name] = _peak(d, profile, widths, signal_window)
            off_peaks[name] = _peak(d, profile, widths, ~signal_window)
            inside = in_peaks[name]["score"] / 3.0 >= off_peaks[name]["score"] / 5.0
            band_peaks[name] = dict(
                in_peaks[name] if inside else off_peaks[name],
                window="signal" if inside else "off_signal",
                route_threshold=3.0 if inside else 5.0,
            )
        best_band = max(
            band_peaks,
            key=lambda name: band_peaks[name]["score"]
            / band_peaks[name]["route_threshold"],
        )
        # Nominate the component with the greatest residual projection relative
        # to its own supported time region, retaining all other components.
        overlap = []
        for k, part in enumerate(components):
            width = max(2.0, np.exp(part[2]))
            tau = np.exp(x[-1]) if incumbent["scatter"] else 0.0
            window = abs(d.t - part[1]) <= 4 * width + 4 * tau
            leverage = float(np.sum(aligned_residual[:, window] ** 2))
            overlap.append(leverage / max(1.0, float(np.sum(planes[k] ** 2))) ** 0.25)
        target = int(np.argmax(overlap))
        target_data = _align(d, plane + planes[target], x[5 * n])
        raw_peaks = {
            name: _peak(d, _project(d, target_data, mask), widths)
            for name, mask in bands.items()
        }
        expected_target = _align(d, planes[target], x[5 * n])
        expected_peaks = {
            name: _peak(d, _project(d, expected_target, mask), widths)
            for name, mask in bands.items()
        }
        pair_names = (
            ("low_half", "high_half"),
            ("low_third", "high_two_thirds"),
            ("low_two_thirds", "high_third"),
            ("middle_third", "full"),
        )
        pairs = [(a, b) for a, b in pair_names if a in bands and b in bands]
        differences = []
        for a, b in pairs:
            pa, pb = (raw_peaks[a], raw_peaks[b])
            ea, eb = (expected_peaks[a], expected_peaks[b])
            observed_contrast = pa["arrival"] - pb["arrival"]
            expected_contrast = ea["arrival"] - eb["arrival"]
            separation = abs(observed_contrast - expected_contrast) / max(
                0.5, min(ea["sigma"], eb["sigma"])
            )
            width_ratio = max(pa["sigma"], pb["sigma"]) / max(
                0.5, min(pa["sigma"], pb["sigma"])
            )
            width_excess = float(
                np.exp(
                    abs(
                        np.log(pa["sigma"] / pb["sigma"])
                        - np.log(ea["sigma"] / eb["sigma"])
                    )
                )
            )
            differences.append(
                dict(
                    bands=[a, b],
                    score=min(pa["score"], pb["score"]),
                    observed_arrival_contrast=observed_contrast,
                    expected_arrival_contrast=expected_contrast,
                    excess_separation_sigma=separation,
                    observed_width_ratio=width_ratio,
                    excess_width_ratio=width_excess,
                )
            )
        morphology = any(
            v["score"] >= 5
            and (v["excess_separation_sigma"] >= 0.5 or v["excess_width_ratio"] >= 1.5)
            for v in differences
        )
        residual_route = (
            band_peaks[best_band]["score"] >= band_peaks[best_band]["route_threshold"]
        )
        review.update(
            band_residual_peaks=band_peaks,
            band_raw_peaks=raw_peaks,
            band_expected_peaks=expected_peaks,
            signal_residual_peaks=in_peaks,
            off_signal_residual_peaks=off_peaks,
            morphology=differences,
            target_component=target,
            route_thresholds=dict(
                signal_residual_proxy_score=3.0,
                off_signal_residual_proxy_score=5.0,
                raw_band_score=5.0,
                excess_relative_arrival=0.5,
                excess_width_ratio=1.5,
            ),
            routed=bool(residual_route or morphology),
        )
        review["timing"]["diagnostics_seconds"] = time.monotonic() - start
        active_stage = None
        if not review["routed"]:
            review["status"] = "no_coherent_band_or_shape_evidence"
            return (selected, review)
        review["route_reason"] = "spectral_shape" if morphology else "band_residual"
        if time.monotonic() >= deadline:
            review["status"] = "diagnostics_used_budget"
            return (selected, review)
        start = time.monotonic()
        active_stage, stage_started = ("grouped_seed_seconds", start)
        tau = (
            float(np.exp(x[-1]))
            if incumbent["scatter"]
            else max(0.4, float(np.median(np.exp(components[:, 2]))) * 0.6)
        )
        seeds = []
        for scatter in (False, True):
            for a, b in pairs:
                keep = [part.copy() for k, part in enumerate(components) if k != target]
                parts = keep + [
                    _spectral_component(d, bands[name], raw_peaks[name], scatter, tau)
                    for name in (a, b)
                ]
                seeds.append(
                    _family_row(
                        parts,
                        x[5 * n],
                        scatter,
                        tau,
                        "spectral_split_"
                        + a
                        + "_"
                        + b
                        + ("_tail" if scatter else "_gaussian"),
                    )
                    | dict(kind="split")
                )
            if residual_route:
                addition = _spectral_component(
                    d, bands[best_band], band_peaks[best_band], scatter, tau
                )
                seeds.append(
                    _family_row(
                        [part.copy() for part in components] + [addition],
                        x[5 * n],
                        scatter,
                        tau,
                        "spectral_residual_"
                        + best_band
                        + ("_tail" if scatter else "_gaussian"),
                    )
                    | dict(kind="addition")
                )
        screened = []
        for seed in seeds:
            if time.monotonic() >= deadline:
                break
            t = time.monotonic()
            q, price = amplitude_solve(
                d, np.asarray(seed["x"]), seed["n"], seed["scatter"], False
            )
            seed["x"] = q.tolist()
            seed["grouped_loss"] = price["loss"]
            receipt = dict(
                name=seed["name"],
                kind=seed["kind"],
                n=seed["n"],
                scatter=seed["scatter"],
                x=seed["x"],
                grouped_improvement=approximate_loss - price["loss"],
                amplitude=price,
                elapsed_seconds=time.monotonic() - t,
            )
            review["seed_prices"].append(receipt)
            screened.append(seed)
        review["timing"]["grouped_seed_seconds"] = time.monotonic() - start
        active_stage = None
        best = [
            min(
                (r for r in screened if r["scatter"] == family and r["kind"] == kind),
                key=lambda r: r["grouped_loss"],
                default=None,
            )
            for kind in ("split", "addition")
            for family in (False, True)
        ]
        best = [r for r in best if r is not None]
        # Preserve a joint split independently of a locally attractive addition.
        # Complete grouped comparisons before spending on original native pixels.
        grouped_end = time.monotonic() + 0.55 * max(0.0, deadline - time.monotonic())
        promoted = []
        for index, seed in enumerate(best):
            if time.monotonic() >= deadline:
                break
            trial = dict(
                name=seed["name"],
                kind=seed["kind"],
                scatter=seed["scatter"],
                n=seed["n"],
                status="started",
            )
            trial_begin = time.monotonic()
            family_deadline = min(
                deadline,
                max(
                    trial_begin + 0.001,
                    trial_begin + (grouped_end - trial_begin) / (len(best) - index),
                ),
            )
            review["trials"].append(trial)
            try:
                grouped = _timed(
                    review,
                    "grouped_refit_seconds",
                    lambda: optimizer(
                        d,
                        seed["n"],
                        seed["scatter"],
                        False,
                        np.asarray(seed["x"]),
                        seed["name"],
                        12,
                        family_deadline,
                    ),
                    trial,
                    "grouped_seconds",
                )
                improvement = approximate_loss - grouped["loss"]
                trial.update(
                    grouped_improvement=improvement,
                    grouped_termination=grouped["termination"],
                )
                threshold = max(0.5, 0.0005 * max(0.0, incumbent["gain"]))
                trial["native_promotion_threshold"] = threshold
                if improvement <= threshold:
                    trial["status"] = "grouped_no_material_improvement"
                    continue
                trial["status"] = "grouped_eligible_for_native"
                promoted.append((grouped, trial))
            except (Limit, ValueError, np.linalg.LinAlgError) as exc:
                trial.update(status="trial_failed", failure=repr(exc))
                review["failures"].append(dict(name=seed["name"], reason=repr(exc)))
            finally:
                trial["elapsed_seconds"] = time.monotonic() - trial_begin
        native_best = [
            max(
                (item for item in promoted if item[0]["scatter"] == family),
                key=lambda item: item[1]["grouped_improvement"],
                default=None,
            )
            for family in (False, True)
        ]
        native_best = [item for item in native_best if item is not None]
        for index, (grouped, trial) in enumerate(native_best):
            t = time.monotonic()
            if t >= deadline:
                trial["status"] = "no_native_budget"
                continue
            native_begin = t
            family_deadline = min(
                deadline, t + (deadline - t) / (len(native_best) - index)
            )
            try:
                native = _timed(
                    review,
                    "native_refit_seconds",
                    lambda: optimizer(
                        d,
                        grouped["n"],
                        grouped["scatter"],
                        True,
                        np.asarray(grouped["x"]),
                        grouped["name"] + "_native",
                        3,
                        family_deadline,
                    ),
                    trial,
                    "native_seconds",
                )
                native["gain"], native["rms_gain"], native["energy"] = _timed(
                    review,
                    "native_score_seconds",
                    lambda: score(d, native),
                    trial,
                    "native_score_seconds",
                )
                candidate = {
                    k: native[k]
                    for k in (
                        "name",
                        "n",
                        "scatter",
                        "x",
                        "gain",
                        "rms_gain",
                        "energy",
                        "termination",
                        "active_bounds",
                    )
                }
                if not np.isfinite(
                    [candidate["gain"], candidate["rms_gain"], candidate["energy"]]
                ).all():
                    raise ValueError("nonfinite native score")
                candidate["proposal_source"] = VERSION
                candidates.append(candidate)
                rows.append(candidate)
                trial.update(
                    status="native_scored",
                    common_gain=candidate["gain"],
                    rms_gain=candidate["rms_gain"],
                    common_change=candidate["gain"] - incumbent["gain"],
                    rms_change=candidate["rms_gain"] - incumbent["rms_gain"],
                    both_score_guard=bool(
                        candidate["gain"] >= incumbent["gain"]
                        and candidate["rms_gain"] >= incumbent["rms_gain"]
                    ),
                    native_termination=candidate["termination"],
                )
            except (Limit, ValueError, np.linalg.LinAlgError) as exc:
                trial.update(status="trial_failed", failure=repr(exc))
                review["failures"].append(dict(name=grouped["name"], reason=repr(exc)))
            finally:
                trial["elapsed_seconds"] += time.monotonic() - native_begin
        selected = _choose(incumbent, candidates)
        review["status"] = (
            "improved" if selected is not incumbent else "incumbent_retained"
        )
        review["candidates"] = candidates
        review["selected_name"] = selected.get("name")
        return (selected, review)
    except (
        Limit,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as exc:
        review["status"] = "proposal_failed"
        review["failures"].append(repr(exc))
        selected = _choose(incumbent, candidates)
        review["candidates"] = candidates
        return (selected, review)
    finally:
        if active_stage is not None:
            review["timing"][active_stage] += time.monotonic() - stage_started
        review["elapsed_seconds"] = time.monotonic() - begin
        review["deadline_overrun_seconds"] = max(0.0, time.monotonic() - deadline)
        review["work_delta"] = {
            key: d.e.budget.get(key, 0) - budget_before.get(key, 0)
            for key in d.e.budget
        }
