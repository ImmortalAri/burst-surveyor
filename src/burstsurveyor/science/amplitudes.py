"""Pivoted parameter coordinates and nonnegative, profiled amplitude solves."""

import numpy as np
from scipy.optimize import nnls

from .model import LOG_FREQUENCY_SCALE, REFERENCE_MHZ, pulse


def to_pivot(x, n, pivots):
    q = np.asarray(x, dtype=float).copy()
    for i, z in enumerate(pivots):
        a, s, c = (x[5 * i], x[5 * i + 3], x[5 * i + 4])
        q[5 * i] = a + s * z + c * z * z
        q[5 * i + 3] = s + 2 * c * z
    return q


def from_pivot(q, n, pivots):
    x = np.asarray(q, dtype=float).copy()
    for i, z in enumerate(pivots):
        a, s, c = (q[5 * i], q[5 * i + 3], q[5 * i + 4])
        x[5 * i] = a - s * z + c * z * z
        x[5 * i + 3] = s - 2 * c * z
    return x


def pivot_matrix(size, n, pivots):
    transform = np.eye(size)
    for i, z in enumerate(pivots):
        transform[5 * i, 5 * i + 3] = -z
        transform[5 * i, 5 * i + 4] = z * z
        transform[5 * i + 3, 5 * i + 4] = -2 * z
    return transform


def training_pivots(data, x, n, scatter):
    pivots = []
    for comp in np.asarray(x[: 5 * n]).reshape(n, 5):
        safe = comp.copy()
        z = data.e.coords.z[data.reps]
        safe[0] = -float(np.max(safe[3] * z + safe[4] * z * z))
        p = pulse(
            np.r_[safe, x[5 * n :]], 1, scatter, data.e.coords, data.reps, data.t
        )[0]
        p -= p.mean(axis=1, keepdims=True)
        correlation = np.maximum(0.0, np.sum(p * data.y, axis=1))
        energy = np.sum(p * p, axis=1) * data.group_norm**2
        # Shrink a noisy matched estimate toward the candidate emitting support.
        weight = correlation**2 + 0.15 * energy / max(float(energy.max()), 1e-20) * max(
            float((correlation**2).max()), 1.0
        )
        pivots.append(
            float(np.average(z, weights=weight))
            if weight.sum() > 0
            else float(np.mean(z))
        )
    return np.asarray(pivots)


def amplitude_solve(data, x, n, scatter, native):
    """Exact nonnegative linear amplitudes after profiling one baseline/channel.

    Accumulate only an n-by-n Gram matrix, then solve its whitened NNLS system.
    Negative unit-template means are essential: the common channel intercept is
    profiled under precisely the same pixel objective as nonlinear refinement.
    """
    x = np.asarray(x, dtype=float).copy()
    comp = x[: 5 * n].reshape(n, 5)
    z = data.e.coords.z[data.indices]
    offsets = -np.max(comp[:, 3, None] * z + comp[:, 4, None] * z * z, axis=1)
    gram = np.zeros((n, n))
    rhs = np.zeros(n)
    null = 0.0
    groups = data.e.tiles(data.indices, n) if native else [data.reps]
    for ii in groups:
        rows = []
        for k in range(n):
            piece = comp[k].copy()
            piece[0] = offsets[k]
            p = pulse(np.r_[piece, x[5 * n :]], 1, scatter, data.e.coords, ii, data.t)[
                0
            ]
            p -= p.mean(axis=1, keepdims=True)
            p *= data.sqrt_weight[ii, None] if native else data.group_norm[:, None]
            rows.append(p.ravel())
        matrix = np.asarray(rows)
        y = (
            (
                (data.e.raw[np.ix_(ii, data.train)] - data.base[ii, None])
                * data.sqrt_weight[ii, None]
            ).ravel()
            if native
            else data.y.ravel()
        )
        gram += matrix @ matrix.T
        rhs += matrix @ y
        null += float(y @ y)
    eigen, vectors = np.linalg.eigh(gram)
    cutoff = max(1e-30, float(eigen.max()) * data.c.get("nnls_relative_cutoff", 1e-12))
    good = eigen > cutoff
    amplitudes = np.zeros(n)
    if np.any(good):
        factor = np.sqrt(eigen[good])[:, None] * vectors[:, good].T
        target = vectors[:, good].T @ rhs / np.sqrt(eigen[good])
        amplitudes = nnls(factor, target, maxiter=max(50, 20 * n))[0]
    # Exact zero has no finite logarithm. This floor is negligible at all locked
    # acquisition scales and is explicitly recorded as a collapsed component.
    x[np.arange(n) * 5] = offsets + np.log(np.maximum(amplitudes, 1e-100))
    loss = float(null - 2 * amplitudes @ rhs + amplitudes @ gram @ amplitudes)
    data.evaluations += 1
    data.e.budget["evaluations"] += 1
    if native:
        data.native_work += len(data.indices) * len(data.t) * n
        data.e.budget["native_work"] += len(data.indices) * len(data.t) * n
    return x, {
        "loss": loss,
        "amplitudes": amplitudes.tolist(),
        "unit_log_fluence": offsets.tolist(),
        "rank": int(good.sum()),
        "collapsed": int(np.count_nonzero(amplitudes <= 1e-90)),
        "objective": "native" if native else "grouped_native_time",
    }


def spectral_diagnostics(data, x, n, scatter):
    z = data.e.coords.z[data.indices]
    f = data.e.frequency[data.indices]
    out = []
    pivots = training_pivots(data, np.asarray(x), n, scatter)
    for k, comp in enumerate(np.asarray(x[: 5 * n]).reshape(n, 5)):
        logsed = comp[0] + comp[3] * z + comp[4] * z * z
        weights = np.exp(2 * (logsed - logsed.max())) / data.var[data.indices]
        active = weights >= weights.max() * 0.01
        neff = weights.sum() ** 2 / max(float(weights @ weights), 1e-30)
        out.append(
            {
                "pivot_mhz": float(
                    REFERENCE_MHZ * np.exp(LOG_FREQUENCY_SCALE * pivots[k])
                ),
                "effective_bandwidth_mhz": float(neff * abs(data.e.grid["res_freq"])),
                "in_band_channels": int(active.sum()),
                "emitting_frequency_range_mhz": [
                    float(f[active].min()),
                    float(f[active].max()),
                ],
                "reference_outside_emitting_band": bool(
                    not (f[active].min() <= REFERENCE_MHZ <= f[active].max())
                ),
                "common_reference_extrapolation_log_ratio": float(
                    comp[0] - logsed.max()
                ),
                "maximum_in_band_log_fluence": float(logsed.max()),
                "spectral_shape_unidentifiable": bool(neff < 3),
                "extreme_common_reference_parameters": bool(
                    abs(comp[3]) > 12 or comp[0] < np.log(data.fluence_scale) - 10
                ),
            }
        )
    return out
