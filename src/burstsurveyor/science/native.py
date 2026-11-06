"""Verify and export a selected native reconstruction with upstream FitBurst."""

import contextlib
from functools import lru_cache
import hashlib
from importlib.metadata import version
import inspect
import io
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits


VERSION = "burstsurveyor-native-1"
PARITY_LIMIT = 0.01


def _measured(timings, label, function, *args, **kwargs):
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timings[label] += time.perf_counter() - started


def _parameters(row, coords, normalize_pbf):
    x = np.asarray(row["x"], dtype=float)
    n = row["n"]
    physical = coords.parameters(x, n, row["scatter"])
    parts = x[: 5 * n].reshape(n, 5)
    if row["scatter"] and not normalize_pbf:
        log_amplitude = parts[:, 0] - np.log(2.0) - x[-1]
    else:
        log_amplitude = parts[:, 0] - 0.5 * np.log(2 * np.pi) - parts[:, 2]
    return dict(
        amplitude=(log_amplitude / np.log(10.0)).tolist(),
        arrival_time=physical["arrival_seconds_relative"],
        burst_width=physical["gaussian_sigma_seconds"],
        spectral_index=physical["spectral_index"],
        spectral_running=physical["spectral_running"],
        scattering_timescale=[float(physical["tau_seconds"])] * n,
        scattering_index=[-4.0] * n,
        dm=[float(physical["residual_dm_pc_cm3"])] * n,
        dm_index=[-2.0] * n,
        ref_freq=[float(physical["reference_mhz"])] * n,
    )


def _components(model_class, parameters, freq, time_s, n, quadrature):
    # Factor_freq_upsample=1 evaluates the actual centres, even across a mask
    # gap or descending band. Upstream reads freq[1] to initialize its grid.
    count = len(freq)
    if count == 1:
        freq = np.r_[freq, freq[0] + 0.001]
    model = model_class(
        freq,
        time_s,
        is_dedispersed=True,
        num_components=n,
        factor_time_upsample=quadrature,
        factor_freq_upsample=1,
    )
    model.update_parameters(parameters)
    with contextlib.redirect_stdout(io.StringIO()):
        model.compute_model()
    components = model.spectrum_per_component[:count]
    if not np.isfinite(components).all():
        raise ValueError("nonfinite FitBurst model")
    return components


def _score(data, row, quadrature, pulse_fn, model_class, normalize_pbf, moments=None):
    gains = np.zeros(2)
    integrated = np.zeros(2)
    exported_gains = np.zeros(2)
    energy = difference = 0.0
    prediction = np.full(data.e.raw.shape, np.nan, dtype=np.float32)
    baseline = np.full(len(data.e.raw), np.nan)
    x = np.asarray(row["x"], dtype=float)
    n = row["n"]
    parameters = _parameters(row, data.e.coords, normalize_pbf)
    time_s = data.t * data.e.coords.dt
    tile = max(1, min(16, 250000 // (len(time_s) * n)))
    for start in range(0, len(data.indices), tile):
        ii = data.indices[start : start + tile]
        parts = _components(
            model_class, parameters, data.e.frequency[ii], time_s, n, quadrature
        )
        signal = parts.sum(axis=2)
        prediction[ii] = signal
        stored = prediction[ii].astype(float)
        if not np.isfinite(stored).all():
            raise ValueError("native signal exceeds prediction storage range")
        baseline[ii] = data.base[ii] - stored.mean(axis=1)
        stored -= stored.mean(axis=1, keepdims=True)
        signal -= signal.mean(axis=1, keepdims=True)
        canonical = pulse_fn(x, n, row["scatter"], data.e.coords, ii, data.t)[0]
        canonical -= canonical.mean(axis=1, keepdims=True)
        y = data.e.raw[ii] - data.base[ii, None]
        if moments is not None:
            parts -= parts.mean(axis=1, keepdims=True)
        for k, variance in enumerate((data.var, data.rms_var)):
            vari = variance[ii, None]
            gains[k] += np.sum(signal * (2 * y - signal) / vari)
            exported_gains[k] += np.sum(stored * (2 * y - stored) / vari)
            integrated[k] += np.sum(canonical * (2 * y - canonical) / vari)
            if moments is not None:
                weight = 1 / np.sqrt(vari)
                design = (parts * weight[:, :, None]).reshape(-1, n)
                moments[0][k] += design.T @ design
                moments[1][k] += design.T @ (y * weight).ravel()
        difference += np.sum((canonical - signal) ** 2 / data.var[ii, None])
        energy += np.sum(canonical**2 / data.var[ii, None])
    rmse = np.sqrt(difference / max(1e-12, energy))
    if not np.isfinite([*gains, *integrated, *exported_gains, rmse]).all():
        raise ValueError("nonfinite official/canonical score")
    score = dict(
        gain=float(gains[0]),
        rms_gain=float(gains[1]),
        stored_gain=float(exported_gains[0]),
        stored_rms_gain=float(exported_gains[1]),
        canonical_gain=float(integrated[0]),
        canonical_rms_gain=float(integrated[1]),
        relative_model_rmse=float(rmse),
    )
    return (score, prediction, baseline)


def _nonnegative_scales(gram, rhs):
    norms = np.sqrt(np.maximum(0.0, np.diag(gram)))
    active = norms > np.sqrt(np.finfo(float).tiny)
    scales = np.zeros_like(rhs)
    if np.any(active):
        norm = norms[active]
        normalized = gram[np.ix_(active, active)] / norm[:, None] / norm
        eigenvalues, vectors = np.linalg.eigh((normalized + normalized.T) / 2)
        keep = eigenvalues > np.finfo(float).eps * len(norm) * max(1.0, eigenvalues[-1])
        if np.any(keep):
            root = np.sqrt(eigenvalues[keep])
            design = root[:, None] * vectors[:, keep].T
            target = vectors[:, keep].T @ (rhs[active] / norm) / root
            scales[active] = (
                nnls(design, target, maxiter=max(100, 10 * len(norm)))[0] / norm
            )
    return np.maximum(scales, np.finfo(float).tiny)


def _check_data(data, row):
    x = np.asarray(row["x"], dtype=float)
    n = row["n"]
    if (
        not isinstance(n, (int, np.integer))
        or isinstance(n, bool)
        or n < 1
        or not isinstance(row["scatter"], (bool, np.bool_))
        or x.shape != (5 * n + 1 + int(row["scatter"]),)
        or not np.isfinite(x).all()
    ):
        raise ValueError("invalid canonical parameter row")
    if not data.e.grid["is_dedispersed"]:
        raise ValueError("requires a dedispersed acquisition grid")
    if (
        not np.array_equal(data.e.frequency, data.e.coords.frequency)
        or np.any(data.e.frequency <= 0)
        or not np.isfinite(data.e.frequency).all()
        or data.e.coords.dt <= 0
        or not np.isfinite(data.e.coords.dt)
    ):
        raise ValueError("invalid native frequency or time coordinates")
    if not np.array_equal(data.t, np.arange(data.e.raw.shape[1])) or len(data.t) < 2:
        raise ValueError("requires the full native time grid")
    if len(data.indices) == 0:
        raise ValueError("empty fitting mask")
    for variance in (data.var[data.indices], data.rms_var[data.indices]):
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError("invalid fitting variance")
    if not np.isfinite([row["gain"], row["rms_gain"]]).all() or row["gain"] <= 0:
        raise ValueError("invalid canonical anchor gains")


def _parity_reasons(score):
    reasons = []
    if score["relative_model_rmse"] > PARITY_LIMIT:
        reasons.append("profiled official/canonical model RMSE exceeds 1%")
    for field in ("gain", "rms_gain"):
        integrated = score["canonical_" + field]
        if score[field] < integrated - PARITY_LIMIT * max(abs(integrated), 1.0):
            reasons.append(
                "official " + field + " falls below canonical parity tolerance"
            )
        if abs(score["stored_" + field] - score[field]) > 1e-5 * max(
            1.0, abs(score[field])
        ):
            reasons.append(
                "float32 artifact " + field + " differs from official rendering"
            )
    return reasons


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def _backend_receipt(model_class):
    source = Path(inspect.getfile(model_class)).resolve()
    package = source.parents[1]
    bundle = hashlib.sha256()
    paths = sorted(
        (path for path in package.rglob("*") if path.suffix in (".py", ".yaml", ".yml"))
    )
    for path in paths:
        bundle.update(path.relative_to(package).as_posix().encode("utf8") + b"\x00")
        bundle.update(bytes.fromhex(_hash(path)))
    return dict(
        backend_version=version("fitburst"),
        backend_source_sha256=_hash(source),
        backend_bundle_sha256=bundle.hexdigest(),
    )


def _publish(path, arrays):
    # Publish only a complete artifact. Concurrent identical retries are safe.
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, prefix=".native-", suffix=".tmp"
        ) as handle:
            tmp_name = handle.name
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(tmp_name)
        digest = _hash(temporary)
        try:
            if os.name == "nt":
                # Windows rename fails atomically if the destination exists.
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if _hash(path) != digest:
                raise FileExistsError(
                    "refusing to replace conflicting native prediction: " + str(path)
                ) from None
        return digest
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)


def finalize(data, selected, prediction_path, *, pulse_fn, source_sha256):
    """Finalize one winner with official FitBurst on identical pixels and weights.

    Return a JSON-safe receipt. A refused or unresolved result publishes nothing.
    Filesystem errors propagate so the coordinator cannot seal an incomplete run.
    Use a unique per-event .npz destination with an existing parent directory.
    """
    start = time.perf_counter()
    stage_seconds = dict(
        incumbent_render_seconds=0.0,
        amplitude_solve_seconds=0.0,
        amplitude_pricing_seconds=0.0,
        conditional_render_seconds=0.0,
        publish_seconds=0.0,
    )
    path = Path(prediction_path)
    if path.suffix.lower() != ".npz":
        raise ValueError("expected an .npz artifact destination")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_sha256)
    ):
        raise ValueError("expected lowercase raw NPZ SHA-256")
    receipt = dict(
        version=VERSION,
        status="refused",
        certified=False,
        source_sha256=source_sha256,
        official_incumbent=None,
        selected_score=None,
        reasons=[],
        prediction=None,
        official_geometry_trials=0,
        native_model_passes=0,
        stage_seconds=stage_seconds,
    )
    if selected is None:
        return receipt | dict(
            status="unresolved",
            reasons=["no selected raw-only candidate"],
            elapsed_s=time.perf_counter() - start,
        )
    try:
        from fitburst.analysis.model import SpectrumModeler
        from fitburst.backend import general

        _check_data(data, selected)
        normalize_pbf = bool(general["flags"]["normalize_pbf"])
        receipt.update(
            **_backend_receipt(SpectrumModeler),
            normalize_pbf=normalize_pbf,
            analytic_incumbent=dict(
                gain=float(selected["gain"]),
                rms_gain=float(selected["rms_gain"]),
                termination=selected.get("termination"),
                name=selected.get("name"),
                active_bounds=selected.get("active_bounds"),
            ),
        )
        n = int(selected["n"])
        row = dict(
            n=n, scatter=bool(selected["scatter"]), x=np.asarray(selected["x"]).tolist()
        )
        narrowest = np.exp(min(row["x"][2 : 5 * n : 5]))
        quadrature = min(64, max(8, int(np.ceil(3 / max(0.01, narrowest)))))
        receipt.update(
            quadrature=quadrature, native_model_passes=1, challenger_selected=False
        )
        moments = (np.zeros((2, n, n)), np.zeros((2, n)))
        fields = ("gain", "rms_gain")
        with threadpool_limits(1):
            control, prediction, baseline = _measured(
                stage_seconds,
                "incumbent_render_seconds",
                _score,
                data,
                row,
                quadrature,
                pulse_fn,
                SpectrumModeler,
                normalize_pbf,
                moments,
            )
            reasons = _parity_reasons(control)
            for field in fields:
                if not np.isclose(
                    control["canonical_" + field], selected[field], rtol=1e-8, atol=1e-6
                ):
                    reasons.append(
                        "replayed canonical " + field + " differs from supplied winner"
                    )
            receipt.update(official_incumbent=control)
            if reasons:
                return receipt | dict(
                    reasons=reasons, elapsed_s=time.perf_counter() - start
                )
            score = control
            try:
                scales = _measured(
                    stage_seconds,
                    "amplitude_solve_seconds",
                    _nonnegative_scales,
                    moments[0][0],
                    moments[1][0],
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                # A failed optional challenge cannot displace a verified incumbent.
                scales = np.ones(n)
                receipt.update(challenger_error=str(exc))
            pricing_started = time.perf_counter()
            priced = {
                field: float(2 * scales @ rhs - scales @ gram @ scales)
                for field, gram, rhs in zip(fields, *moments)
            }
            if not np.isfinite([*scales, *priced.values()]).all():
                scales = np.ones(n)
                priced = {field: control[field] for field in fields}
                receipt.update(challenger_error="nonfinite fixed-amplitude proposal")
            margins = {field: max(1.0, 0.001 * abs(control[field])) for field in fields}
            receipt.update(
                amplitude_scales=scales.tolist(),
                priced_amplitudes=priced,
                required_gains=margins,
            )
            stage_seconds["amplitude_pricing_seconds"] += (
                time.perf_counter() - pricing_started
            )
            if all(priced[k] > control[k] + margins[k] for k in fields):
                x = np.asarray(row["x"], dtype=float)
                x[: 5 * n : 5] += np.log(scales)
                proposal = row | dict(x=x.tolist())
                receipt.update(native_model_passes=2)
                try:
                    trial, trial_prediction, trial_baseline = _measured(
                        stage_seconds,
                        "conditional_render_seconds",
                        _score,
                        data,
                        proposal,
                        quadrature,
                        pulse_fn,
                        SpectrumModeler,
                        normalize_pbf,
                    )
                    matched = all(
                        (
                            np.isclose(trial[k], priced[k], rtol=1e-08, atol=1e-08)
                            for k in fields
                        )
                    )
                    rejections = _parity_reasons(trial)
                    if not matched:
                        rejections.append(
                            "official amplitude replay disagrees with priced gain"
                        )
                    if not all(trial[k] > control[k] + margins[k] for k in fields):
                        rejections.append(
                            "official amplitude replay lacks material dual-gain improvement"
                        )
                    receipt.update(
                        official_challenger=trial,
                        challenger_price_matched=bool(matched),
                        challenger_rejections=rejections,
                        challenger_selected=not rejections,
                    )
                    if not rejections:
                        row, score, prediction, baseline = (
                            proposal,
                            trial,
                            trial_prediction,
                            trial_baseline,
                        )
                except (
                    ValueError,
                    ArithmeticError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                ) as exc:
                    receipt.update(challenger_error=str(exc))
        receipt.update(
            selected=row,
            selected_score=score,
            fitburst_parameters=_parameters(row, data.e.coords, normalize_pbf),
        )
    except (
        ImportError,
        ValueError,
        ArithmeticError,
        RuntimeError,
        np.linalg.LinAlgError,
    ) as exc:
        return receipt | dict(
            reasons=[type(exc).__name__ + ": " + str(exc)],
            elapsed_s=time.perf_counter() - start,
        )
    mask = np.zeros(len(data.e.raw), dtype=bool)
    mask[data.indices] = True
    arrays = dict(
        prediction=prediction,
        baseline=baseline,
        variance=data.var,
        rms_variance=data.rms_var,
        channels=data.indices,
        channel_mask=mask,
        offburst_time_mask=data.noise_time,
        frequency=data.e.frequency,
        time_s=data.t * data.e.coords.dt,
        times_bin0=np.float64(data.e.grid["times_bin0"]),
        freqs_bin0=np.float64(data.e.grid["freqs_bin0"]),
        res_time=np.float64(data.e.grid["res_time"]),
        res_freq=np.float64(data.e.grid["res_freq"]),
        source_sha256=np.array(source_sha256),
        version=np.array(VERSION),
    )
    digest = _measured(stage_seconds, "publish_seconds", _publish, path, arrays)
    return receipt | dict(
        status="provisional",
        prediction=path.name,
        prediction_sha256=digest,
        elapsed_s=time.perf_counter() - start,
    )
