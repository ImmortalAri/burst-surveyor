"""Raw-derived channel masks, noise weights and native reconstruction gain."""

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d

from .engine import Data, InputError
from .model import pulse


def _noise_estimate(raw, channels=None):
    if np.issubdtype(raw.dtype, np.integer):
        # Integer peak-to-peak subtraction may overflow in the input dtype.
        good = np.any(raw != raw[:, :1], axis=1) & (
            np.sum(raw, axis=1, dtype=float) != 0
        )
    else:
        good = (
            np.isfinite(raw).all(axis=1)
            & (np.ptp(raw, axis=1) > 0)
            & (np.sum(raw, axis=1) != 0)
        )
    if channels is not None:
        mask = np.zeros(len(raw), bool)
        mask[channels] = True
        good &= mask
    ids = np.flatnonzero(good)
    if len(ids) < 6:
        raise InputError("Fewer than six usable channels")
    profile = np.zeros(raw.shape[1])
    for ii in np.array_split(ids, max(1, int(np.ceil(len(ids) / 32)))):
        y = raw[ii].astype(float)
        profile += (y - np.median(y, axis=1, keepdims=True)).sum(axis=0)
    profile /= len(ids)
    smooth = gaussian_filter1d(profile, max(1, len(profile) / 700))
    mid = np.median(smooth)
    mad = 1.4826 * np.median(abs(smooth - mid))
    signal = smooth > mid + max(3 * mad, 0.08 * (smooth.max() - mid))
    signal = binary_dilation(signal, iterations=max(2, len(profile) // 60))
    off = np.flatnonzero(~signal)
    uncertain = len(off) < 16
    if uncertain:
        off = np.sort(np.argsort(smooth, kind="stable")[: max(4, len(profile) // 5)])
    var = np.full(len(raw), np.nan)
    rms = var.copy()
    for ii in np.array_split(ids, max(1, int(np.ceil(len(ids) / 32)))):
        var[ii] = np.var(raw[np.ix_(ii, off)], axis=1, ddof=1, dtype=float)
        rms[ii] = np.mean(raw[ii, :-1].astype(float) ** 2, axis=1)
    pool = np.median(var[ids][var[ids] > 0])
    if not np.isfinite(pool):
        raise InputError("Cannot estimate positive channel noise")
    shrink = len(off) / (len(off) + 20)
    var[ids] = np.maximum(shrink * var[ids] + (1 - shrink) * pool, pool * 0.0001)
    rms[ids] = np.maximum(rms[ids], np.median(rms[ids]) * 0.0001)
    return (ids, var, rms, off, uncertain)


def reporting_noise(raw, channels=None, *, return_audit=False):
    from .channel_mask import screen_mask_neighbors

    initial = _noise_estimate(raw)
    extra, audit = screen_mask_neighbors(raw, initial[3], uncertain=initial[4])
    ids = np.setdiff1d(initial[0], extra, assume_unique=True)
    if channels is not None:
        ids = np.intersect1d(ids, channels)
    result = _noise_estimate(raw, ids) if len(extra) or channels is not None else initial
    return (*result, audit) if return_audit else result


class ReconstructionData(Data):
    def __init__(self, event, channels=None):
        train = np.ones(event.length, bool)
        super().__init__(event, train, ~train, np.arange(event.length) // 4, channels)
        ids, var, rms, off, uncertain, self.channel_screening = reporting_noise(
            event.raw, channels, return_audit=True
        )
        self.indices, self.var, self.rms_var = (ids, var, rms)
        self.sqrt_weight = 1 / np.sqrt(var)
        self.noise_uncertain = uncertain
        self.noise_time = np.zeros(event.length, bool)
        self.noise_time[off] = True
        self.off_pairs = int(np.sum(self.noise_time[1:] & self.noise_time[:-1]))
        self.base = np.full(len(event.raw), np.nan)
        for ii in event.tiles(ids):
            self.base[ii] = event.raw[ii].mean(axis=1, dtype=float)
        labels = ids // event.group_size
        self.groups = np.split(ids, np.flatnonzero(np.diff(labels)) + 1)
        self.group_labels = labels[np.r_[0, np.flatnonzero(np.diff(labels)) + 1]]
        self.lookup = np.full(len(event.raw), -1, int)
        self.lookup[ids] = np.searchsorted(self.group_labels, labels)
        self.group_norm = np.sqrt(np.bincount(self.lookup[ids], weights=1 / var[ids]))
        self.reps = []
        self.y = np.empty((len(self.groups), event.length))
        for g, ii in enumerate(self.groups):
            w = 1 / var[ii]
            f = event.frequency[ii]
            self.reps.append(ii[np.argmin(abs(f - np.average(f, weights=w)))])
            self.y[g] = w @ (event.raw[ii] - self.base[ii, None]) / self.group_norm[g]
        self.reps = np.asarray(self.reps)
        self.profile = self.group_norm @ self.y / np.sum(self.group_norm**2)
        self.noise_scale = (
            float(np.sqrt(pool)) if (pool := np.median(var[ids])) > 0 else 1.0
        )
        self.fluence_scale = max(self.noise_scale, float(np.max(self.profile)))


def predict(d, r, ii):
    if r.get('model') == 'channel-amplitudes-v1':
        from .scintillation import prediction
        return prediction(d, r, ii)
    return pulse(np.asarray(r["x"]), r["n"], r["scatter"], d.e.coords, ii, d.t)[0]


def score(d, r):
    gain = rms_gain = energy = 0.0
    for ii in d.e.tiles(d.indices):
        p = predict(d, r, ii)
        p -= p.mean(axis=1, keepdims=True)
        y = d.e.raw[ii] - d.base[ii, None]
        term = p * (2 * y - p)
        gain += float(np.sum(term / d.var[ii, None]))
        rms_gain += float(np.sum(term / d.rms_var[ii, None]))
        energy += float(np.sum(p * p / d.var[ii, None]))
    return (gain, rms_gain, energy)


def choose(rows, c):
    # Common native gain only. Prefer fewest parameters within practical tolerance.
    def utility(r):
        return r["gain"] - c["selection_penalty"] * (5 * r["n"] + int(r["scatter"]))

    best = max(rows, key=utility)
    credible = [
        r
        for r in rows
        if utility(r)
        >= utility(best)
        - max(c["selection_floor"], c["selection_fraction"] * max(0, best["gain"]))
    ]
    return min(
        credible, key=lambda r: (5 * r["n"] + int(r["scatter"]), -r["gain"], r["name"])
    )
