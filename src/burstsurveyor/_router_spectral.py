"""Fixed-timing spectral rescue using Burst Surveyor v0.2.0 channel amplitudes."""

import numpy as np

from .science.model import pulse
from .science.native import finalize
from .science.objective import score
from .science.scintillation import MODEL, coefficients, shapes
from .secondlook import POLICY as RESCUE_POLICY


CHANNEL_BLOCK = 8
MIN_FOLD_CHANNELS = 12
MAX_COMPONENT_CELLS = 12_000_000


def _basis(data, geometry, channels):
    return shapes(
        np.asarray(geometry["x"]), geometry["n"], geometry["scatter"], data.e.coords,
        channels, data.t,
    )[0]


def _train_amplitudes(data, geometry, channels):
    """Fit the nonnegative amplitudes only on the channels passed by the caller."""
    fitted = []
    for tile in data.e.tiles(channels, geometry["n"]):
        y = data.e.raw[tile].astype(float) - data.base[tile, None]
        h = _basis(data, geometry, tile)
        h -= h.mean(axis=1, keepdims=True)
        fitted.append(coefficients(h, y))
    return np.concatenate(fitted)


def _held_prediction(data, geometry, train, held):
    frequency = data.e.frequency
    order = np.argsort(frequency[train], kind="stable")
    freq_train = frequency[train][order]
    amps_train = _train_amplitudes(data, geometry, train)[order]
    return np.column_stack([
        np.interp(frequency[held], freq_train, amp) for amp in amps_train.T
    ])


def _check_fold(data, geometry, train, held, old_prediction, old_baseline):
    if min(len(train), len(held)) < MIN_FOLD_CHANNELS:
        return dict(held_channels=len(held), passed=False, reason="insufficient channels in spectral fold")

    predicted_amps = _held_prediction(data, geometry, train, held)
    old_sse, new_sse, null_sse = np.zeros(2), np.zeros(2), np.zeros(2)
    amp_index = 0
    for tile in data.e.tiles(held, geometry["n"]):
        h = _basis(data, geometry, tile)
        h -= h.mean(axis=1, keepdims=True)
        estimate = np.einsum("ctn,cn->ct", h, predicted_amps[amp_index:amp_index + len(tile)])
        amp_index += len(tile)

        # Use the same profiled constant background as the incumbent. Pulse
        # shapes and spectral predictions are fixed before reading held data.
        y = data.e.raw[tile].astype(float)
        centered = y - data.base[tile, None]
        previous = y - old_prediction[tile] - old_baseline[tile, None]
        proposed = centered - estimate
        for k, var in enumerate((data.var, data.rms_var)):
            old_sse[k] += np.sum(previous**2 / var[tile, None])
            new_sse[k] += np.sum(proposed**2 / var[tile, None])
            null_sse[k] += np.sum(centered**2 / var[tile, None])

    if not np.isfinite([old_sse, new_sse, null_sse]).all() or np.any(old_sse <= 0):
        return dict(held_channels=len(held), passed=False, reason="invalid spectral fold scores")

    change = old_sse - new_sse
    needed = np.maximum(
        RESCUE_POLICY["validation_floor"], RESCUE_POLICY["fraction"] * np.abs(null_sse - old_sse),
    )
    return dict(
        held_channels=len(held), error_reduction_pct=(100 * change / old_sse).tolist(),
        improvement=change.tolist(), required_improvement=needed.tolist(),
        passed=bool(np.all(change > needed)),
    )


def fit_spectral(data, incumbent, old_prediction, old_baseline, destination, source_sha256):
    """Try spectral amplitudes with fixed incumbent timing and test held blocks.

    Both folds share the baseline geometry, grid, mask and noise weights. No
    fitted amplitude on a held channel is used to construct the held prediction.
    The caller checks the full native export and determines its disposition.
    """
    geometry = dict(n=incumbent["n"], scatter=incumbent["scatter"], x=list(incumbent["x"]))
    x = np.asarray(geometry["x"], dtype=float)
    n = geometry["n"]
    x[:5*n:5] = 0
    x[3:5*n:5] = 0
    x[4:5*n:5] = 0
    geometry["x"] = x.tolist()
    channels = data.indices
    result = dict(
        official_export=None, channel_folds=[], candidate_components=n,
        channel_fold_scope=(
            "fixed baseline timing and shared noise estimates; channel amplitudes trained "
            "only outside held 8-channel blocks and interpolated into those blocks"
        ),
    )
    if getattr(data.e.coords, "channel_time_offset_bins", None) is not None:
        return dict(result, reason="spectral export requires a shared time grid")
    if len(channels) * data.e.length * n > MAX_COMPONENT_CELLS:
        return dict(result, reason="spectral component-pixel budget exceeded")

    fold_id = (channels // CHANNEL_BLOCK) % 2
    try:
        for parity in (0, 1):
            train = channels[fold_id == parity]
            held = channels[fold_id != parity]
            result["channel_folds"].append(_check_fold(
                data, geometry, train, held, old_prediction, old_baseline,
            ))
        amplitudes = _train_amplitudes(data, geometry, channels)
        model = dict(
            geometry, model=MODEL, name="fixed-timing-channel-amplitudes",
            amplitude_channels=channels.tolist(), channel_amplitudes=amplitudes.tolist(),
        )
        model["gain"], model["rms_gain"], _ = score(data, model)
        encoded = finalize(data, model, destination, pulse_fn=pulse, source_sha256=source_sha256)
    except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return dict(result, reason=f"spectral refit failed: {exc}")
    result["official_export"] = encoded
    return dict(result, reason=("fixed timing with fitted channel amplitudes" if encoded["status"] == "provisional"
                                else "native spectral export refused"))
