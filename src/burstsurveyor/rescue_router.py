"""Try one rescue against a sealed v0.2.0 burst fit, with a two-fold check."""

from pathlib import Path
import tempfile
import time

import numpy as np

from ._router_peaks import candidates, coverage, discover, error, profile_noise
from ._router_spectral import fit_spectral
from ._storage import read, write
from .diagnostics import _band_check
from .science.engine import Limit
from .science.fit import load_event
from .science.model import pulse
from .science.native import finalize
from .science.objective import ReconstructionData, score
from .science.pulse_coverage import assess
from .secondlook import _discovery, _evidence, _fit_direction
from .triage import component_omissions


BLOCK_CHANNELS = 8
RESIDUAL_SCORE = 6.0
BAND_GAIN_FRACTION = 0.01


def channel_sets(data):
    index = data.indices
    parity = (index // BLOCK_CHANNELS) % 2
    return [index[parity == i] for i in (0, 1)]


def peak_coverage(data, pred, features, noise, band_feature=None):
    if features:
        use = data.indices
        profile = np.average(pred[use], axis=0, weights=1 / data.var[use])
        result = coverage(profile, features, noise)
    else:
        result = {"fitted_peaks": [], "missing": []}
    if band_feature is not None:
        band_test = assess(data, None, [band_feature], pixels=pred)
        result["band_peak_check"] = band_test
        result["missing"] = sorted(set(result["missing"] + [x["time_bin"] for x in band_test["missing"]]))
    return result


def repeated_positive(data, sets, signal, row):
    proposal = _discovery(data, sets[0], signal, row)
    if proposal is None:
        return None, []
    values = [_evidence(data, use, signal, proposal) for use in sets]
    if all(score is not None and score >= RESIDUAL_SCORE for score in values):
        return proposal, values
    return None, values


def choose_route(data, row, signal, features, original, sets):
    """A strong replicated temporal mismatch takes priority over spectral scaling."""
    missing = original["missing"]
    proposal, evidence = repeated_positive(data, sets, signal, row)
    if proposal:
        lo, hi = proposal["time_bins"]
        aligned = any(lo - 1 <= peak <= hi for peak in missing)
        if aligned and len(missing) >= 2 and row["components"] < len(features) <= data.c["max_components"]:
            return "joint", "multiple supported profile peaks omitted", proposal, evidence, None
        if aligned and len(missing) == 1 and row["components"] < data.c["max_components"]:
            return "single", "isolated supported profile peak omitted", proposal, evidence, None
        if len(missing) <= 1 and row["components"] < data.c["max_components"]:
            # A feature confined to a sub-band can be absent from the full-band profile.
            omitted = component_omissions(data, signal, row)["missing"]
            if len(omitted) == 1:
                feature = omitted[0]
                local = feature["proposal"]
                scores = [_evidence(data, use, signal, local) for use in sets]
                if all(v is not None and v >= RESIDUAL_SCORE for v in scores):
                    return "single", "isolated supported sub-band peak omitted", local, scores, feature
        return "none", "replicated residual without a supported rescue pattern", proposal, evidence, None

    if features and not missing and not hasattr(data.e.coords, "channel_time_offset_bins"):
        # Trigger only for substantial frequency structure that predicts the other fold.
        band = _band_check(data, signal)
        floor = [max(4.0, BAND_GAIN_FRACTION * abs(row[k] or 0.0)) for k in ("gain", "rms_gain")]
        if band and all(b["common"] > floor[0] and b["rms"] > floor[1] for b in band):
            return "spectral", "repeatable band-amplitude mismatch with peak timing covered", None, band, None
    return "none", "no qualifying rescue pattern", None, evidence, None


def add_component(data, incumbent, proposal, strength):
    previous = data.e.budget
    data.e.budget = {"evaluations": 0, "native_work": 0}
    try:
        row = _fit_direction(data, data.indices, incumbent, proposal, strength)
        if row["termination"] in {"wall_limit", "work_limit"}:
            return None
        row["gain"], row["rms_gain"], _ = score(data, row)
        return row
    except (Limit, ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None
    finally:
        data.e.budget = previous


def temporal_folds(data, sets, solve, old_prediction, old_base):
    folds = []
    for part in (0, 1):
        train, held = sets[part], sets[1-part]
        fit = solve(ReconstructionData(data.e, channels=train))
        if fit is None:
            folds.append(dict(held_channels=len(held), passed=False, reason="training fit unavailable"))
            continue

        def predict(tile):
            return pulse(np.asarray(fit["x"]), fit["n"], fit["scatter"], data.e.coords, tile, data.t)[0]

        before = error(data, held, old_prediction, old_base)
        after = error(data, held, predict)
        change = 100 * (1 - after / before)
        folds.append(dict(held_channels=len(held), error_reduction_pct=change.tolist(),
                          selected_start=fit["name"], passed=bool(np.min(change) > 0)))
    return folds


def run_event(source, baseline_run, baseline_row, output, exporter):
    """Inspect and optionally refit one member of a verified stock run."""
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(source_sha256=baseline_row["source_sha256"], baseline_components=baseline_row["components"],
                  route="none", route_reason="no baseline fit available", triggered=False)

    def finish(disposition, reason):
        elapsed = time.monotonic() - started
        report.setdefault("screen_seconds", elapsed)
        report.update(disposition=disposition, reason=reason, seconds=elapsed)
        write(output / "result.json", report)
        return report

    with tempfile.TemporaryDirectory(dir=output) as scratch:
        data, entry = load_event(source, Path(scratch))
        if entry["source_sha256"] != baseline_row["source_sha256"]:
            raise ValueError("Input and sealed baseline hashes differ")
        if not baseline_row["prediction"]:
            return finish("retained", "no baseline fit available")
        sets = channel_sets(data)
        if min(map(len, sets)) < 12 or data.noise_uncertain or data.channel_screening["status"] == "skipped_safety_cap":
            report["route_reason"] = "data mask or noise estimate needs inspection"
            return finish("retained", report["route_reason"])
        with np.load(baseline_run / baseline_row["prediction"], allow_pickle=False) as original:
            if not np.array_equal(data.indices, original["channels"]):
                raise ValueError("Sealed baseline channel mask differs")
            old_prediction, old_base = original["prediction"], original["baseline"]
        features, noise = discover(data)
        old_coverage = peak_coverage(data, old_prediction, features, noise)
        route, reason, proposal, evidence, band_feature = choose_route(data, baseline_row, old_prediction,
                                                                        features, old_coverage, sets)
        report.update(route=route, route_reason=reason, features=features, baseline_coverage=old_coverage,
                      proposal=proposal, route_evidence=evidence, res_time=data.e.coords.dt)
        if route == "none":
            return finish("retained", reason)
        report.update(triggered=True, screen_seconds=time.monotonic() - started, export_attempts=[])
        incumbent = read(baseline_run / baseline_row["native_receipt"])["selected"]
        old_error = error(data, data.indices, old_prediction, old_base)
        destination = Path(scratch) / "prediction.npz"

        def adequate(artifact, name):
            if not artifact or artifact["status"] != "provisional":
                report["export_attempts"].append(dict(name=name, reasons=(artifact or {}).get("reasons", [])))
                return False
            with np.load(destination, allow_pickle=False) as archive:
                pred, base = archive["prediction"], archive["baseline"]
                native = peak_coverage(data, pred, features, noise, band_feature)
                reduction = (100 * (1 - error(data, data.indices, pred, base) / old_error)).tolist()
            report["export_attempts"].append(dict(name=name, coverage=native, error_reduction_pct=reduction))
            if native["missing"] or min(reduction) <= 0:
                destination.unlink()
                return False
            report.update(official_export=artifact, native_coverage=native, selected_start=name,
                          native_error_reduction_pct=reduction, candidate_components=artifact["selected"]["n"])
            return True

        if route == "spectral":
            fit = fit_spectral(data, incumbent, old_prediction, old_base, destination, entry["source_sha256"])
            if not adequate(fit["official_export"], "fixed-timing spectral amplitudes"):
                return finish("retained", fit["reason"] or "spectral export failed peak or loss check")
            folds = fit["channel_folds"]
        else:
            if route == "joint":
                trial_rows, history = candidates(data, features, noise)
                report["full_trials"] = history

                def solve(d):
                    ranked, _ = candidates(d, features, profile_noise(d.profile, d.noise_time))
                    return ranked[0] if ranked else None
            else:
                strength = min(evidence)

                def solve(d):
                    return add_component(d, incumbent, proposal, strength)

                row = solve(data)
                trial_rows = [] if row is None else [row]
            for row in trial_rows:
                encoded = finalize(data, row, destination, pulse_fn=pulse,
                                   source_sha256=entry["source_sha256"], export_backend=exporter)
                if adequate(encoded, row["name"]):
                    break
            else:
                return finish("retained", "no native export improves both losses and covers supported peaks")
            folds = temporal_folds(data, sets, solve, old_prediction, old_base)

        report["channel_folds"] = folds
        report["channel_fold_scope"] = "candidate optimization excludes held channels; diagnostics see the observation"
        destination.replace(output / "prediction.npz")
        if len(folds) == 2 and all(fold["passed"] for fold in folds):
            return finish("candidate", "native rescue passed both held-out channel checks")
        return finish("review", "native rescue passed; held-out channel checks incomplete or adverse")
