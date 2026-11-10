"""Conditional review of frequency amplitudes; never changes an exported fit."""

import numpy as np


def spectral_review(data, signal):
    """Probe existing pulse shape with channel scales, then check disjoint times.

    The per-channel scales are diagnostic nuisance coefficients, not a fitted
    spectrum or new physical components. Geometry and noise came from the full
    data, so the time split is conditional evidence, not independent validation.
    """
    if signal is None:
        return {"flags": [], "reason": "no exported model"}
    raw, ids = data.e.raw, np.asarray(data.indices, dtype=int)
    if np.shape(signal) != raw.shape or raw.ndim != 2:
        raise ValueError("Spectral review requires matching two-dimensional pixels")
    if len(ids) < 6 or raw.shape[1] < 16:
        return {"flags": [], "reason": "too few channels or time samples"}
    weights = np.array([data.var[ids], data.rms_var[ids]])
    if not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("Spectral review requires positive finite noise variances")
    # Full record and alternating time bins; each comparison profiles the same
    # channel baseline independently. Retain sufficient statistics only.
    pp, py = np.zeros((3, len(ids))), np.zeros((3, len(ids)))
    tile = max(1, min(32, 200000 // raw.shape[1]))
    for start in range(0, len(ids), tile):
        stop = min(start + tile, len(ids))
        y, p = raw[ids[start:stop]].astype(float), signal[ids[start:stop]].astype(float)
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("Nonfinite usable spectral-review pixels")
        for fold, sl in enumerate((slice(None), slice(0, None, 2), slice(1, None, 2))):
            yy, mm = y[:, sl], p[:, sl]
            yy = yy - yy.mean(axis=1, keepdims=True)
            mm = mm - mm.mean(axis=1, keepdims=True)
            pp[fold, start:stop] = np.sum(mm * mm, axis=1)
            py[fold, start:stop] = np.sum(mm * yy, axis=1)

    def scales(fold):
        gram, rhs = pp[fold] / weights[0], py[fold] / weights[0]
        # Fixed ridge towards the existing scale; finite bounds keep the probe
        # conservative in weak channels. No search over ridge or spectral bins.
        return (np.clip((rhs + 20.0) / (gram + 20.0), 0.0, 4.0),
                float(np.clip((rhs.sum() + 20.0) / (gram.sum() + 20.0), 0.0, 4.0)))

    def gain(scale, fold):
        term = 2 * (scale - 1) * py[fold] - (scale**2 - 1) * pp[fold]
        return np.sum(term[None, :] / weights, axis=1)

    informative = int(np.count_nonzero(pp[0] / weights[0] >= 4.0))
    full, global_full = scales(0)
    fitted_gain = gain(full, 0)
    incremental = fitted_gain - gain(global_full, 0)
    base_gain = np.sum((2 * py[0] - pp[0])[None, :] / weights, axis=1)
    # A review threshold, not an information criterion or component p-value.
    # A global amplitude error alone must not masquerade as spectral structure.
    threshold = np.maximum(max(8.0, 2.0 * informative), 0.001 * np.abs(base_gain))
    aligned = informative >= 6 and bool(np.all(incremental > threshold))
    checks = []
    for train, test in ((1, 2), (2, 1)):
        alpha, global_alpha = scales(train)
        added = gain(alpha, test)
        versus_global = added - gain(global_alpha, test)
        checks.append({"train_parity": train - 1, "test_parity": test - 1,
                       "gain_change": added.tolist(),
                       "gain_beyond_global_scale": versus_global.tolist()})
    # Both directions and both documented objectives must support a correction.
    supported = [all(c["gain_change"][k] > 4.0 and
                     c["gain_beyond_global_scale"][k] > 4.0 for c in checks)
                 for k in (0, 1)]
    replicated = all(supported)
    flags = ["spectral_model_limited"] if aligned else []
    if aligned and not replicated:
        flags.append("spectral_amplitude_not_replicated")
    return dict(
        flags=flags, informative_channels=informative,
        full_data_gain_change=fitted_gain.tolist(),
        full_data_gain_beyond_global_scale=incremental.tolist(),
        review_threshold=threshold.tolist(), time_split=checks,
        time_split_status=("replicated" if replicated else
                           "weight_disagreement" if any(supported) else "not_replicated"),
        weight_order=["primary", "alternate"],
        action=("review spectral amplitudes; do not add a pulse for aligned residuals"
                if aligned and replicated else
                "review frequency and time structure together; amplitude-only correction failed"
                if aligned else "no spectral action from this diagnostic"),
        note="Conditional fixed-shape probe only. No correction is exported; no physical interpretation is established.",
    )
