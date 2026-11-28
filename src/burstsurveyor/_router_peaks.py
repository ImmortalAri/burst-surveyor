"""Joint peak refit and scoring for the experimental rescue router."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from .science.engine import Limit, optimizer
from .science.model import pulse
from .science.objective import score


def profile_noise(profile, mask):
    off = profile[mask]
    return float(max(1e-12, 1.4826 * np.median(np.abs(off - np.median(off)))))


def profile_for(data, channels):
    profile = np.zeros(data.e.length)
    for tile in data.e.tiles(channels):
        profile += (data.e.raw[tile] - data.base[tile, None]).T @ (1 / data.var[tile])
    return profile / np.sum(1 / data.var[channels])


def prominence_peaks(profile, noise, ratio):
    return find_peaks(profile, prominence=ratio * noise, distance=4, wlen=15)


def discover(data):
    """Use the full-band profile and require local replication in both channel sets."""
    profile = gaussian_filter1d(profile_for(data, data.indices), 0.5)
    noise = profile_noise(profile, data.noise_time)
    peaks, props = prominence_peaks(profile, noise, 2.0)
    center = np.median(profile[data.noise_time])
    anchor = (np.max(profile) - center) / noise
    if anchor < 5:
        return [], noise
    supported = []
    for parity in (0, 1):
        channels = data.indices[parity::2]
        smoothed = gaussian_filter1d(profile_for(data, channels), 0.5)
        fold_noise = profile_noise(smoothed, data.noise_time)
        fold_peaks, fold_props = prominence_peaks(smoothed, fold_noise, 1.4)
        supported.append((fold_peaks, fold_props["prominences"] / fold_noise))
    features = []
    widths = peak_widths(
        profile, peaks,
        prominence_data=(props["prominences"], props["left_bases"], props["right_bases"]),
    )[0] / 2.355
    for i, peak in enumerate(peaks):
        # Prominence alone gives isolated background fluctuations too many starts.
        if profile[peak] - center < 2.5 * noise:
            continue
        radius = 2 if widths[i] >= 2 else 1
        fold_strengths = [float(np.max(z[np.abs(f - peak) <= radius], initial=0)) for f, z in supported]
        if min(fold_strengths) < 1.4:
            continue
        features.append(dict(bin=int(peak), width=float(max(0.55, widths[i])),
                             left=int(props["left_bases"][i]), right=int(props["right_bases"][i]),
                             strength=float(props["prominences"][i] / noise), fold_strengths=fold_strengths))
    return features, noise


def spectrum_seed(data, feature):
    """Estimate the spectral location of excess intensity over neighboring troughs."""
    k, lo, hi = (feature[key] for key in ("bin", "left", "right"))
    channels = data.indices
    frac = (k - lo) / (hi - lo)
    contrast = np.mean(data.e.raw[channels, max(0, k-1):k+2], axis=1)
    contrast -= 0.5 * ((1-frac) * data.e.raw[channels, lo] + frac * data.e.raw[channels, hi])
    weights = np.zeros(len(data.e.raw))
    weights[channels] = 1 / data.var[channels]
    weighted = np.zeros_like(weights)
    weighted[channels] = contrast * weights[channels]
    sigma = max(1, len(weights) / 32)
    local = gaussian_filter1d(weighted, sigma) / np.sqrt(np.maximum(gaussian_filter1d(weights, sigma), 1e-12))
    center = channels[np.argmax(local[channels])]
    # Start with modest spectral curvature. The optimizer can move both parameters.
    z = data.e.coords.z[center]
    spectral_width = max(0.10, np.ptp(data.e.coords.z[channels]) / 7)
    curvature = -0.5 / spectral_width**2
    return float(z), float(curvature), float(data.e.coords.tau_ratio[center])


def fit_seed(data, features, spectra, tau, localized):
    parts = []
    for f, (z, curvature, ratio) in zip(features, spectra):
        k = f["bin"]
        width = f["width"]
        height = max(data.noise_scale * 0.025, data.profile[k])
        amplitude = np.log(height * 2.5 * width)
        lag = min(1.3 * width, tau * ratio)
        if localized:
            part = [amplitude + curvature*z*z, k-lag, np.log(width), -2*curvature*z, curvature]
        else:
            part = [amplitude, k-lag, np.log(width), 0, 0]
        parts.extend(part)
    return np.array(parts + [0.] + ([np.log(tau)] if tau else []))


def model_profile(data, row):
    weights = 1 / data.var[data.indices]
    out = np.zeros(data.e.length)
    for tile in data.e.tiles(data.indices):
        signal = pulse(np.asarray(row["x"]), row["n"], row["scatter"], data.e.coords, tile, data.t)[0]
        out += signal.T @ (1 / data.var[tile])
    return out / weights.sum()


def coverage(profile, features, noise):
    maxima = find_peaks(profile, prominence=0.5 * noise)[0]
    bins = np.array([f["bin"] for f in features])
    hit = set()
    for peak in maxima:
        distances = np.abs(bins - peak)
        nearest = int(distances.argmin())
        if distances[nearest] <= 3:
            hit.add(int(bins[nearest]))
    return dict(fitted_peaks=maxima.tolist(), missing=[int(k) for k in bins if k not in hit])


def candidates(data, features, noise):
    spectra = [spectrum_seed(data, f) for f in features]
    rows, trials = [], []
    previous_budget = data.e.budget
    original_limits = {k: data.c[k] for k in ("max_evaluations", "max_native_work")}
    data.c.update(data.c["bounded_refinement_work"])

    def run(name, initial, scatter, native, iterations):
        data.e.budget = dict(evaluations=0, native_work=0)
        try:
            row = optimizer(data, len(features), scatter, native, initial, name, iterations, float("inf"))
            row["gain"], row["rms_gain"], _ = score(data, row)
            row["coverage"] = coverage(model_profile(data, row), features, noise)
            rows.append(row)
            trials.append({k: row[k] for k in ("name", "gain", "rms_gain", "coverage", "termination")})
            trials[-1]["work"] = data.e.budget.copy()
        except (Limit, ValueError, np.linalg.LinAlgError) as exc:
            trials.append(dict(name=name, failure=str(exc)))

    def rank(row):
        return (len(row["coverage"]["missing"]), -row["gain"], row["name"])

    try:
        for localized, kind in ((True, "localized"), (False, "flat")):
            for tau in (0., 0.55, 1.5):
                run(f"{kind}-tau-{tau:g}", fit_seed(data, features, spectra, tau, localized),
                    bool(tau), False, 26)
        for row in sorted(rows, key=rank)[:3]:
            run(row["name"] + "-native", row["x"], row["scatter"], True, 28)
    finally:
        data.e.budget = previous_budget
        data.c.update(original_limits)
    ready = [row for row in rows if row["name"].endswith("-native") and not row["coverage"]["missing"]]
    return sorted(ready, key=rank), trials


def error(data, channels, prediction, baseline=None):
    result = np.zeros(2)
    for tile in data.e.tiles(channels):
        y = data.e.raw[tile].astype(float)
        if baseline is None:
            z = prediction(tile)
            resid = y - data.base[tile, None] - (z - np.mean(z, axis=1, keepdims=True))
        else:
            resid = y - prediction[tile] - baseline[tile, None]
        result += (np.sum(resid**2 / data.var[tile, None]), np.sum(resid**2 / data.rms_var[tile, None]))
    return result
