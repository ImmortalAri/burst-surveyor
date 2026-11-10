"""Grouped and native residuals, parameter bounds, and damped nonlinear refinement."""

from dataclasses import dataclass
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .model import Coordinates, DISPERSION_SECONDS_MHZ2, pulse
from .amplitudes import (
    amplitude_solve,
    to_pivot,
    from_pivot,
    pivot_matrix,
    training_pivots,
    spectral_diagnostics,
)


class InputError(ValueError):
    pass


class Limit(RuntimeError):
    pass


def correlation(left, right):
    a = left.ravel()
    b = right.ravel()
    sa, sb = (np.std(a), np.std(b))
    return (
        float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
        if sa > 0 and sb > 0
        else 0.0
    )


class Event:
    def __init__(self, raw, grid, config):
        if raw.ndim != 2 or min(raw.shape) < 6:
            raise InputError("Unsupported raw array dimensions")
        self.raw, self.grid, self.config = (raw, grid, config)
        self.budget = config.get("_shared_budget", {"evaluations": 0, "native_work": 0})
        self.length = raw.shape[1]
        self.frequency = (
            grid["freqs_bin0"] + (np.arange(len(raw)) + 0.5) * grid["res_freq"]
        )
        self.coords = Coordinates(self.frequency, grid["res_time"], self.frequency)
        self.group_size = max(1, int(np.ceil(len(raw) / config["frequency_groups"])))

    def tiles(self, indices, parameters=1):
        cells = self.config["tile_derivative_cells"]
        step = max(
            1,
            min(self.config["channel_tile"], cells // max(1, self.length * parameters)),
        )
        return (indices[i : i + step] for i in range(0, len(indices), step))


@dataclass
class Template:
    first: int
    last: int
    start: int
    width: int
    score: float
    coverage: float

    def record(self):
        return vars(self)


class Data:
    def __init__(self, event, train, held, blocks, channels=None):
        self.e, self.c = (event, event.config)
        self.train, self.held, self.blocks = (train, held, blocks)
        self.t = np.flatnonzero(train).astype(float)
        self.ht = np.flatnonzero(held).astype(float)
        if len(self.t) < self.c["min_train"]:
            raise InputError("Too few training time bins")
        adjacent = np.flatnonzero(np.diff(self.t) == 1)
        if len(adjacent) < 4:
            raise InputError("Too few consecutive training time bins for channel noise")
        raw = event.raw
        self.base = np.full(len(raw), np.nan)
        variance = np.full(len(raw), np.nan)
        centered_sum = np.zeros(event.length)
        for ii in event.tiles(np.arange(len(raw)) if channels is None else channels):
            y = raw[np.ix_(ii, train)].astype(float)
            good = np.isfinite(y).all(axis=1) & (np.ptp(y, axis=1) > 0)
            y = y[good]
            if not len(y):
                continue
            d = y[:, adjacent + 1] - y[:, adjacent]
            mad = 1.4826 * np.median(abs(d - np.median(d, axis=1)[:, None]), axis=1)
            std = np.std(d, axis=1)
            variance[ii[good]] = np.where(mad > 0, mad, std) ** 2 / 2
            self.base[ii[good]] = y.mean(axis=1)
            centered_sum[train] += np.sum(y - np.median(y, axis=1)[:, None], axis=0)
        self.indices = np.flatnonzero(np.isfinite(variance) & (variance > 0))
        if len(self.indices) < self.c["min_channels"]:
            raise InputError("Too few finite varying training channels")
        split_profile = centered_sum[train] / len(self.indices)
        med = np.median(split_profile)
        spread = 1.4826 * np.median(abs(split_profile - med))
        # Weakly exclude bright training bins from the noise estimate. No held pixel
        # is read, even when estimating the shape of the pulse or its support.
        noise_time = np.zeros(event.length, bool)
        noise_time[train] = split_profile <= med + max(
            2.8 * spread, 0.035 * np.ptp(split_profile)
        )
        adjacent_off = np.flatnonzero(noise_time[1:] & noise_time[:-1])
        self.off_pairs = len(adjacent_off)
        rho_values = []
        if len(adjacent_off) >= self.c["min_noise_pairs"]:
            for ii in event.tiles(self.indices):
                left = raw[np.ix_(ii, adjacent_off)].astype(float)
                right = raw[np.ix_(ii, adjacent_off + 1)].astype(float)
                dif = right - left
                mad = 1.4826 * np.median(
                    abs(dif - np.median(dif, axis=1)[:, None]), axis=1
                )
                valid = mad > 0
                variance[ii[valid]] = mad[valid] ** 2 / 2
                a = left - np.median(left, axis=1)[:, None]
                b = right - np.median(right, axis=1)[:, None]
                bound = 4 * np.sqrt(variance[ii, None])
                a, b = (np.clip(a, -bound, bound), np.clip(b, -bound, bound))
                num = np.sum(a * b, axis=1)
                den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
                rho_values.extend(
                    np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                )
        self.rho = (
            float(np.clip(np.median(rho_values), 0.0, self.c["max_noise_rho"]))
            if rho_values
            else 0.0
        )
        self.noise_time = noise_time
        variance /= 1 - self.rho
        pool = float(np.median(variance[self.indices]))
        shrink = len(adjacent) / (len(adjacent) + self.c["noise_prior"])
        self.var = np.maximum(
            shrink * variance + (1 - shrink) * pool, pool * self.c["noise_floor"]
        )
        self.sqrt_weight = 1 / np.sqrt(self.var)
        labels = self.indices // event.group_size
        self.groups = np.split(self.indices, np.flatnonzero(np.diff(labels)) + 1)
        self.group_labels = labels[np.r_[0, np.flatnonzero(np.diff(labels)) + 1]]
        self.lookup = np.full(len(raw), -1, dtype=int)
        self.lookup[self.indices] = np.searchsorted(self.group_labels, labels)
        self.group_norm = np.sqrt(
            np.bincount(self.lookup[self.indices], weights=1 / self.var[self.indices])
        )
        self.reps = []
        self.y = np.empty((len(self.groups), len(self.t)))
        for g, ii in enumerate(self.groups):
            weights = 1 / self.var[ii]
            freq = event.frequency[ii]
            self.reps.append(
                ii[np.argmin(abs(freq - np.average(freq, weights=weights)))]
            )
            y = raw[np.ix_(ii, train)] - self.base[ii, None]
            self.y[g] = weights @ y / self.group_norm[g]
        self.reps = np.asarray(self.reps)
        self.rho_freq = 0.0
        pairs = np.diff(self.group_labels) == 1
        if np.any(pairs) and noise_time.sum() >= 8:
            yoff = self.y[:, noise_time[train]]
            yoff -= np.median(yoff, axis=1)[:, None]
            self.rho_freq = float(
                np.clip(correlation(yoff[:-1][pairs], yoff[1:][pairs]), 0.0, 0.65)
            )
        self.noise_scale = np.sqrt(pool)
        self.profile = self.group_norm @ self.y / np.sum(self.group_norm**2)
        positive_peak = max(float(np.max(self.profile)), self.noise_scale * 0.015)
        self.fluence_scale = max(self.noise_scale, positive_peak)
        self.block_ids = np.unique(blocks[held])
        self.frequency_test = None
        self.evaluations = 0
        self.native_work = 0

    def scan(self, residual):
        """Search both signs using rectangular templates and empirical noise scales."""
        ntime = self.e.length
        nf = (len(self.e.raw) + self.e.group_size - 1) // self.e.group_size
        plane = np.zeros((nf, ntime))
        good = np.zeros((nf, ntime), dtype=int)
        plane[np.ix_(self.group_labels, self.train)] = residual
        good[np.ix_(self.group_labels, self.train)] = 1
        summed = np.pad(plane.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        count = np.pad(good.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        tcount = np.r_[0, self.train.cumsum()]
        offcount = np.r_[0, self.noise_time.cumsum()]
        pos = neg = None
        for band in sorted(set(min(nf, value) for value in self.c["bands"]) | {nf}):
            f0 = np.unique(
                np.r_[np.arange(0, nf - band + 1, max(1, band // 2)), nf - band]
            )
            a, b = (f0[:, None], (f0 + band)[:, None])
            f_corr = 1 + 2 * self.rho_freq * (band - 1) / band
            for width in self.c["widths"]:
                if width > max(1, ntime // 3):
                    break
                start = np.arange(0, ntime - width + 1)
                stop = start + width
                exposure = (
                    count[b, stop] - count[b, start] - count[a, stop] + count[a, start]
                )
                coverage = exposure / (band * width)
                eligible = (coverage >= self.c["template_coverage"]) & (
                    exposure >= self.c["template_min_cells"]
                )
                if not np.any(eligible):
                    continue
                raw = (
                    summed[b, stop]
                    - summed[b, start]
                    - summed[a, stop]
                    + summed[a, start]
                )
                t_corr = (
                    1
                    + 2 * self.rho * (width - 1) / width
                    + 2 * self.rho**2 * max(width - 2, 0) / width
                )
                value = raw / np.sqrt(np.maximum(exposure, 1) * f_corr * t_corr)
                off = eligible & (
                    offcount[stop] - offcount[start] == tcount[stop] - tcount[start]
                )
                sample = value[off]
                if len(sample) >= self.c["min_template_noise"]:
                    med = np.median(sample)
                    spread = 1.4826 * np.median(abs(sample - med))
                    value /= max(1.0, float(spread))
                for sign in (1, -1):
                    oriented = np.where(eligible, sign * value, -np.inf)
                    r, j = np.unravel_index(np.argmax(oriented), oriented.shape)
                    score = float(oriented[r, j])
                    best = pos if sign == 1 else neg
                    if best is None or score > best.score:
                        template = Template(
                            int(f0[r]),
                            int(f0[r] + band),
                            int(j),
                            width,
                            score,
                            float(coverage[r, j]),
                        )
                        if sign == 1:
                            pos = template
                        else:
                            neg = template
        return (pos, neg)

    def group_parts(self, x, n, scatter):
        pred = np.zeros_like(self.y)
        for ii in self.e.tiles(self.indices):
            p, _ = pulse(x, n, scatter, self.e.coords, ii, self.t)
            group = self.lookup[ii]
            starts = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
            pred[group[starts]] += np.add.reduceat(
                p / self.var[ii, None], starts, axis=0
            )
        pred /= self.group_norm[:, None]
        pred -= pred.mean(axis=1)[:, None]
        return self.y - pred

    def bounds(self, n, scatter):
        scale = np.log(self.fluence_scale)
        length = self.e.length
        dm_bins = (
            self.c["max_dm"]
            * DISPERSION_SECONDS_MHZ2
            * self.e.coords.inverse_square_span
            / self.e.coords.dt
        )
        dm_bins = min(dm_bins, self.c["dm_duration_fraction"] * length)
        # Common-reference bounds are absolute numerical guards. Physical signal
        # bounds are imposed at a training-derived in-band pivot by optimizer.
        lo = [
            -600.0,
            -0.15 * length,
            np.log(self.c["sigma_min"]),
            -600.0,
            self.c.get("curvature_min", -150.0),
        ] * n + [-dm_bins]
        hi = [
            self.c.get("absolute_log_signal_bound", 200.0),
            1.12 * length,
            np.log(length * 0.6),
            600.0,
            self.c.get("curvature_max", 8.0),
        ] * n + [dm_bins]
        if scatter:
            lo.append(np.log(self.c["tau_min"]))
            hi.append(np.log(length))
        return (np.array(lo), np.array(hi))

    def component_seed(self, template):
        if template is None:
            i = int(np.argmax(gaussian_filter1d(self.profile, 1)))
            return [np.log(self.fluence_scale * 3), self.t[i], np.log(1.5), 0.0, 0.0]
        active = (self.t >= template.start) & (self.t < template.start + template.width)
        band = (self.group_labels >= template.first) & (
            self.group_labels < template.last
        )
        sigma = max(0.55, template.width / 2.5)
        arrival = np.mean(self.t[active])
        # Frequency shape is seeded from the band centre, then fitted.
        z = self.e.coords.z[self.reps[band]]
        curvature = (
            -min(8.0, 1 / max(0.35, float(np.ptp(z))) ** 2)
            if band.sum() < len(band) * 0.75
            else 0.0
        )
        slope = -2 * curvature * float(np.mean(z))
        height = max(
            self.noise_scale * 0.025,
            abs(template.score)
            / max(1.0, np.linalg.norm(self.group_norm[band]) * np.sqrt(active.sum())),
        )
        return [
            np.log(height * max(1.0, sigma * 2.5)) + curvature * float(np.mean(z)) ** 2,
            arrival,
            np.log(sigma),
            slope,
            curvature,
        ]

    def evaluation(self, x, n, scatter, native, derivative):
        self.evaluations += 1
        self.e.budget["evaluations"] += 1
        if native:
            self.native_work += (
                len(self.indices) * len(self.t) * (n if derivative else 1)
            )
            self.e.budget["native_work"] += (
                len(self.indices) * len(self.t) * (n if derivative else 1)
            )
        if not native:
            p, j = pulse(x, n, scatter, self.e.coords, self.reps, self.t, derivative)
            p -= p.mean(axis=-1, keepdims=True)
            r = (self.y - p * self.group_norm[:, None]).ravel()
            if not derivative:
                return (float(r @ r), None, None)
            j -= j.mean(axis=-1, keepdims=True)
            j *= self.group_norm[:, None]
            j = j.reshape(len(x), -1)
            return (float(r @ r), j @ r, j @ j.T)
        loss, gradient, gram = (0.0, np.zeros(len(x)), np.zeros((len(x), len(x))))
        for ii in self.e.tiles(self.indices, len(x) if derivative else 1):
            p, j = pulse(x, n, scatter, self.e.coords, ii, self.t, derivative)
            p -= p.mean(axis=-1, keepdims=True)
            y = self.e.raw[np.ix_(ii, self.train)] - self.base[ii, None]
            r = ((y - p) * self.sqrt_weight[ii, None]).ravel()
            loss += float(r @ r)
            if derivative:
                j -= j.mean(axis=-1, keepdims=True)
                j *= self.sqrt_weight[ii, None]
                j = j.reshape(len(x), -1)
                gradient += j @ r
                gram += j @ j.T
        return (loss, gradient, gram)


def optimizer(data, n, scatter, native, seed, name, iterations, deadline):
    """Damped native-compatible GN in emitting-band coordinates, with NNLS starts."""
    before = time.monotonic()
    lo, hi = data.bounds(n, scatter)
    settings = data.c
    pivots = training_pivots(data, np.asarray(seed), n, scatter)
    transform = pivot_matrix(len(seed), n, pivots)
    qlo, qhi = (lo.copy(), hi.copy())
    qlo[0 : 5 * n : 5] = np.log(data.fluence_scale) - settings["log_fluence_below"] - 10
    qhi[0 : 5 * n : 5] = min(
        settings.get("absolute_log_signal_bound", 200.0),
        np.log(data.fluence_scale)
        + np.log(data.e.length)
        + settings["log_fluence_above"],
    )
    qlo[3 : 5 * n : 5] = -settings.get("pivot_slope_bound", 80.0)
    qhi[3 : 5 * n : 5] = settings.get("pivot_slope_bound", 80.0)
    z = data.e.coords.z
    maximum_log_signal = float(qhi[0])

    def project(q):
        q = np.clip(q, qlo + 1e-09, qhi - 1e-09)
        for i, zp in enumerate(pivots):
            zz = z - zp
            logs = q[5 * i] + q[5 * i + 3] * zz + q[5 * i + 4] * zz * zz
            q[5 * i] -= max(0.0, float(logs.max()) - maximum_log_signal)
        return q

    def exhausted():
        return (
            time.monotonic() > deadline
            or data.e.budget["native_work"] > settings["max_native_work"]
            or data.e.budget["evaluations"] >= settings["max_evaluations"]
        )

    if exhausted():
        raise Limit("Solve exceeded shared budget before any iterate")
    q = project(to_pivot(np.asarray(seed), n, pivots))
    x = from_pivot(q, n, pivots)
    initial = x.copy()
    initial_loss = data.evaluation(x, n, scatter, native, False)[0]
    amp_receipts = []
    # Every initial temporal/spectral geometry receives the same exact linear solve.
    ax, receipt = amplitude_solve(data, x, n, scatter, native)
    aq = project(to_pivot(ax, n, pivots))
    ax = from_pivot(aq, n, pivots)
    amp_loss = data.evaluation(ax, n, scatter, native, False)[0]
    receipt["constrained_loss"] = amp_loss
    amp_receipts.append(receipt)
    if amp_loss < initial_loss:
        x, q, loss = (ax, aq, amp_loss)
    else:
        loss = initial_loss
    calls, accepted, damping = (3, 0, settings["initial_damping"])
    message = "iteration_limit"
    for iteration in range(iterations):
        if exhausted():
            message = "wall_limit" if time.monotonic() > deadline else "work_limit"
            break
        loss, gradient, gram = data.evaluation(x, n, scatter, native, True)
        calls += 1
        gradient = transform.T @ gradient
        gram = transform.T @ gram @ transform
        scale = 1 / np.sqrt(np.maximum(np.diag(gram), 1e-12))
        active = (q - qlo <= 1e-08) & (gradient < 0) | (qhi - q <= 1e-08) & (
            gradient > 0
        )
        free = np.flatnonzero(~active)
        g = gradient[free] * scale[free]
        if not len(g) or np.max(abs(g)) < settings["gradient_stop"]:
            message = "projected_gradient"
            break
        scaled = (gram * scale * scale[:, None])[np.ix_(free, free)]
        moved = False
        for attempt in range(settings["damping_attempts"]):
            if exhausted():
                message = "wall_limit" if time.monotonic() > deadline else "work_limit"
                break
            step = (
                np.linalg.solve(scaled + damping * np.eye(len(free)), g) * scale[free]
            )
            trialq = q.copy()
            trialq[free] += step
            trialq = project(trialq)
            trial = from_pivot(trialq, n, pivots)
            trial_loss = data.evaluation(trial, n, scatter, native, False)[0]
            calls += 1
            improvement = loss - trial_loss
            if improvement > 0:
                moved = True
                accepted += 1
                x, q, loss = (trial, trialq, trial_loss)
                damping = max(1e-06, damping * 0.35)
                if improvement < settings["step_loss_stop"]:
                    message = "small_training_gain"
                break
            damping = min(1000000000.0, damping * 8)
        if message != "iteration_limit":
            break
        if not moved:
            message = "stalled"
            break
    # A second separable solve closes remaining amplitude error after geometry
    # movement; keep the previous finite point if safety constraints worsen it.
    if accepted and not exhausted():
        ax, receipt = amplitude_solve(data, x, n, scatter, native)
        aq = project(to_pivot(ax, n, pivots))
        ax = from_pivot(aq, n, pivots)
        aloss = data.evaluation(ax, n, scatter, native, False)[0]
        calls += 2
        receipt["constrained_loss"] = aloss
        amp_receipts.append(receipt)
        if aloss < loss:
            x, q, loss = (ax, aq, aloss)
    if not np.isfinite(loss):
        raise Limit("No finite iterate")
    order = np.argsort(x[1 : 5 * n : 5], kind="stable")
    x = np.r_[x[: 5 * n].reshape(n, 5)[order].ravel(), x[5 * n :]]
    q = np.r_[q[: 5 * n].reshape(n, 5)[order].ravel(), q[5 * n :]]
    pivots = pivots[order]
    return {
        "name": name,
        "n": n,
        "scatter": scatter,
        "initial": initial.tolist(),
        "initial_objective": float(initial_loss),
        "x": x.tolist(),
        "loss": float(loss),
        "calls": calls,
        "accepted": accepted,
        "termination": message,
        "seconds": time.monotonic() - before,
        "active_bounds": np.flatnonzero(
            np.minimum(q - qlo, qhi - q) <= 0.002 * (qhi - qlo)
        ).tolist(),
        "bounds": [lo.tolist(), hi.tolist()],
        "pivot_bounds": [qlo.tolist(), qhi.tolist()],
        "pivot_z": pivots.tolist(),
        "maximum_in_band_log_fluence_bound": maximum_log_signal,
        "amplitude_solves": amp_receipts,
        "spectral_support": spectral_diagnostics(data, x, n, scatter),
    }
