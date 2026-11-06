"""Raw-data review of residual islands and repeatable frequency-dependent amplitudes."""

from pathlib import Path
import tempfile

import numpy as np

from .batch import verify
from .review import find_event, load_for_review


def _band_check(data, signal):
    """Estimate a shrunk amplitude change from one channel split and test on the other."""
    ids = data.indices
    folds = np.full(len(data.e.raw), -1, dtype=int)
    folds[ids] = np.arange(len(ids)) % 2
    # For each band, fold and noise weight, store sufficient statistics for scaling a fixed pulse.
    gram, rhs = np.zeros((8, 2, 2)), np.zeros((8, 2, 2))
    for ii in data.e.tiles(ids):
        bands = np.minimum(7, (ii * 8) // len(data.e.raw))
        y = data.e.raw[ii].astype(float) - data.base[ii, None]
        p = signal[ii].astype(float)
        p -= p.mean(axis=1, keepdims=True)
        py, pp = (p * y).sum(axis=1), (p * p).sum(axis=1)
        for k, weights in enumerate((data.var, data.rms_var)):
            np.add.at(gram[:, :, k], (bands, folds[ii]), pp / weights[ii])
            np.add.at(rhs[:, :, k], (bands, folds[ii]), py / weights[ii])
    changes = []
    for fit_fold in (0, 1):
        test = 1 - fit_fold
        # A fixed ridge penalty shrinks weak-channel estimates towards scale one.
        scale = np.clip(
            (rhs[:, fit_fold, 0] + 20.0) / (gram[:, fit_fold, 0] + 20.0), 0.25, 4.0
        )
        added = (
            2 * (scale - 1)[:, None] * rhs[:, test]
            - (scale**2 - 1)[:, None] * gram[:, test]
        ).sum(axis=0)
        changes.append(
            dict(
                trained_fold=fit_fold,
                scales=scale.tolist(),
                common=float(added[0]),
                rms=float(added[1]),
            )
        )
    return changes


def diagnose(run, event_id, source):
    """Compare two explanations for a residual on the native grid; leave the fit intact."""
    run, source = Path(run), Path(source)
    event = find_event(verify(run), event_id)
    if event["status"] not in {"provisional", "unresolved"}:
        return dict(
            event_id=event_id, disposition=event["status"], reasons=event["reasons"]
        )
    with tempfile.TemporaryDirectory() as tmp:
        data, entry = load_for_review(source, Path(tmp), event["source_sha256"])
        signal = None
        if event.get("prediction"):
            with np.load(run / event["prediction"], allow_pickle=False) as archive:
                signal = archive["prediction"]
        return diagnostic_data(event, data, entry, signal)


def diagnostic_data(event, data, entry, signal):
    """Inspect a verified event already loaded for its diagnostic figure."""
    from .secondlook import _discovery, _evidence

    train, held = data.indices[::2], data.indices[1::2]
    packet = (
        _discovery(data, train, signal, event)
        if min(len(train), len(held)) >= 6
        else None
    )
    check = _evidence(data, held, signal, packet) if packet else None
    directional = bool(check is not None and check >= 3.0)
    bands = (
        _band_check(data, signal)
        if signal is not None and len(data.indices) >= 12
        else []
    )
    gains = (float(event["gain"] or 0.0), float(event["rms_gain"] or 0.0))
    spectral = bool(
        bands
        and all(
            r["common"] > max(4.0, 0.0005 * abs(gains[0]))
            and r["rms"] > max(4.0, 0.0005 * abs(gains[1]))
            for r in bands
        )
    )
    flags = []
    if directional:
        flags.append("positive_residual_island")
    if spectral:
        flags.append("frequency_amplitude_structure")
    if data.noise_uncertain:
        flags.append("uncertain_offburst_noise")
    return dict(
        schema_version=1,
        event_id=event["id"],
        source_sha256=entry["source_sha256"],
        usable_channels=len(data.indices),
        offburst_bins=int(data.noise_time.sum()),
        flags=flags,
        temporal_proposal=packet if directional else None,
        check_channel_score=check,
        band_scaling_diagnostic=bands,
        note="Cross-channel diagnostics are conditional on the existing fit. Band scales are review evidence only.",
    )
