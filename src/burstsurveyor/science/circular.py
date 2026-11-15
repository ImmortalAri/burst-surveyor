"""Circular coordinates for one folded rotation; observations are never cropped."""
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def phase_origin(raw):
    """Place a reproducible raw-derived crest at mid-phase before bounded search.

    Channel medians/MADs and circular smoothing commute with integer rotations.
    Tie breaking compares complete rotated profiles, not their input positions.
    This is a coordinate permutation, with no interpolation or fitted template.
    """
    length = raw.shape[1]
    profile = np.zeros(length)
    for start in range(0, len(raw), 32):
        y = raw[start:start+32].astype(float)
        good = np.isfinite(y).all(axis=1) & (np.ptp(y, axis=1) > 0)
        y = y[good]
        if not len(y):
            continue
        y -= np.median(y, axis=1, keepdims=True)
        scale = 1.4826*np.median(abs(y), axis=1)
        # Constant-background sparse signals still contribute finite discovery power.
        scale = np.maximum(scale, np.max(abs(y), axis=1)*1e-6)
        profile += np.clip(y/scale[:, None], -5, 5).sum(axis=0)
    trace = gaussian_filter1d(profile, max(1., length/700), mode='wrap')
    peaks = np.flatnonzero(trace == trace.max())
    if len(peaks) == length:
        return 0
    peak = int(min(peaks, key=lambda p: hashlib.sha256(np.roll(profile, -int(p)).tobytes()).digest()))
    return (peak-length//2) % length


def distance(a, b, period):
    return abs((np.asarray(a)-b+period/2) % period-period/2)


def peaks(y, **kwargs):
    """Find peaks and trough intervals across phase zero, without duplicated peaks.

    Returned bases are unwrapped around the peak and may be outside [0, N).
    Windows are shorter than one rotation; modulo indexing restores native bins.
    """
    n = len(y)
    indices, properties = find_peaks(np.tile(y, 3), **kwargs)
    keep = (indices >= n) & (indices < 2*n)
    properties = {k: v[keep] for k, v in properties.items()}
    for key in ('left_bases', 'right_bases', 'left_ips', 'right_ips'):
        if key in properties:
            properties[key] = properties[key]-n
    return indices[keep]-n, properties


def restore_record(value, origin, period):
    """Copy scientific receipts back to input phase, including rejected rows.

    Wrapped intervals retain their length and can end beyond the final bin.
    Parameter vectors use modulo-period arrival coordinates on the original grid.
    """
    if isinstance(value, list):
        return [restore_record(v, origin, period) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: restore_record(v, origin, period) for k, v in value.items()}
    n = out.get('n')
    if isinstance(n, int) and isinstance(out.get('x'), list) and len(out['x']) >= 5*n+1:
        x = list(out['x'])
        for k in range(1, 5*n, 5):
            x[k] = float((x[k]+origin) % period)
        out['x'] = x
        initial = out.get('initial')
        if isinstance(initial, list) and len(initial) == len(x):
            initial = list(initial)
            for k in range(1, 5*n, 5):
                initial[k] = float((initial[k]+origin) % period)
            out['initial'] = initial
    for key in ('time_bin', 'arrival'):
        if isinstance(out.get(key), (int, float)):
            out[key] = (out[key]+origin) % period
    if 'start' in out and 'width' in out:
        out['start'] = (out['start']+origin) % period
    if 'time_bins' in out:
        lo, hi = out['time_bins']; start = (lo+origin) % period
        out['time_bins'] = [start, start+hi-lo]
    return out
