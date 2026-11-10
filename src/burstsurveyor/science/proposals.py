"""Bounded residual and pulse-shape proposals for an existing native objective."""

import hashlib
import json
import time
import numpy as np

from .engine import Limit, optimizer
from .model import pulse
from .objective import score


def parameter_key(row):
    return (
        int(row["n"]),
        bool(row["scatter"]),
        np.asarray(row["x"], dtype=np.float64).tobytes(),
    )


def choose_single_ceiling(rows, fraction=0.001, floor=0.02):
    """Prefer the simplest candidate within the stated native-gain tolerance."""
    finite = [r for r in rows if np.isfinite(r["gain"])]
    if not finite:
        raise ValueError("No finite candidate")
    best = max(finite, key=lambda r: r["gain"])
    allowance = max(floor, fraction * max(0.0, best["gain"]))
    return min(
        (r for r in finite if r["gain"] >= best["gain"] - allowance),
        key=lambda r: (5 * r["n"] + int(r["scatter"]), -r["gain"], r.get("name", "")),
    )


class NativeProbeCache:
    """One immutable Data instance; cache keys bind raw, grid, mask, weights/split."""

    def __init__(self, data, pulse, raw_sha256):
        if not raw_sha256:
            raise ValueError("Raw identity is required")
        self.d, self.pulse = (data, pulse)
        h = hashlib.sha256(raw_sha256.encode())
        for a in (
            data.indices,
            data.var,
            data.base,
            data.t,
            data.e.frequency,
            data.train,
        ):
            h.update(np.asarray(a).tobytes())
        h.update(json.dumps(data.e.grid, sort_keys=True).encode())
        self.objective = h.hexdigest()
        self.values = {}
        self.scans = 0
        self.probe_passes = 0
        self.hits = 0

    def predict(self, row, ii):
        return self.pulse(
            np.asarray(row["x"]),
            row["n"],
            row["scatter"],
            self.d.e.coords,
            ii,
            self.d.t,
        )[0]

    def residual(self, row):
        key = parameter_key(row)
        if key in self.values:
            self.hits += 1
            return self.values[key]
        plane = self.d.group_parts(np.asarray(row["x"]), row["n"], row["scatter"])
        pos, neg = self.d.scan(plane)
        item = {"positive": pos, "negative": neg}
        self.values[key] = item
        self.scans += 1
        if len(self.values) > 8:
            del self.values[next(iter(self.values))]
        return item

    def price(self, row, descriptors):
        from .probe import probe

        item = self.residual(row)
        bank_key = tuple(
            (
                (k, v["kind"], parameter_key(v["candidate"]))
                for k, v in descriptors.items()
            )
        )
        if item.get("bank_key") == bank_key:
            self.hits += 1
            return item["prices"]
        current = {}

        def incumbent(ii):
            key = np.asarray(ii).tobytes()
            if current.get("key") != key:
                current.update(key=key, prediction=self.predict(row, ii))
            return current["prediction"]

        bank = {}
        for name, desc in descriptors.items():

            def generate(ii, desc=desc):
                p = self.predict(desc["candidate"], ii)
                return p - incumbent(ii) if desc["kind"] == "replacement" else p

            bank[name] = (
                generate,
                1.0 if desc["kind"] == "replacement" else float("inf"),
            )
        prices = probe(
            self.d.e.raw,
            self.d.indices,
            self.d.var,
            incumbent,
            bank,
            self.d.c["channel_tile"],
        )
        item.update(bank_key=bank_key, prices=prices)
        self.probe_passes += 1
        return prices


def nuisance_diagnostics(d, row, pulse):
    """Uncertainty flags, never deletion/masking of narrowband data."""
    count = 0
    block = max(16, min(128, d.e.length // 32))
    nb = d.e.length // block
    means = []
    for ii in d.e.tiles(d.indices):
        p = pulse(np.asarray(row["x"]), row["n"], row["scatter"], d.e.coords, ii, d.t)[
            0
        ]
        p -= p.mean(axis=1, keepdims=True)
        z = (d.e.raw[ii] - d.base[ii, None] - p) / np.sqrt(d.var[ii, None])
        count += int(np.count_nonzero(abs(z[:, d.noise_time]) > 8.0))
        if nb:
            means.append(z[:, : nb * block].reshape(len(ii), nb, block).mean(axis=2))
    fraction = count / max(1, len(d.indices) * int(d.noise_time.sum()))
    flags = []
    if count >= 2 and fraction > 0.0001:
        flags.append("sparse_outlier_ambiguity")
    if d.rho > 0.3:
        flags.append("correlated_noise_domain")
    if means:
        m = np.vstack(means)
        energy = np.sum(m * m, axis=0) * block
        rough = np.mean(np.diff(m, axis=0) ** 2, axis=0) / np.maximum(
            2 * np.mean(m * m, axis=0), 1e-12
        )
        balance = np.mean(m > 0, axis=0)
        if np.any(
            (energy > max(8 * len(d.indices), 0.05 * max(0, row["gain"])))
            & (rough > 0.55)
            & (balance > 0.2)
            & (balance < 0.8)
        ):
            flags.append("channel_offset_ambiguity")
    return {
        "flags": flags,
        "off_pulse_outlier_fraction": fraction,
        "off_pulse_outlier_pixels": count,
        "mask_changed": False,
        "meaning": "Heuristics validated only on the fixed development fixtures; not a population acceptance calibration.",
    }


def _directions(d, row, cache):
    x = np.asarray(row["x"])
    n = row["n"]
    s = row["scatter"]
    item = cache.residual(row)
    p = item["positive"]
    directions = {}
    if p:
        directions["residual_island"] = {
            "kind": "addition",
            "candidate": {
                "x": np.r_[d.component_seed(p), x[5 * n]].tolist(),
                "n": 1,
                "scatter": False,
            },
        }
        at = p.start + p.width / 2
        for width in sorted(set((max(2.0, p.width / 2.5), 12.0, 32.0))):
            directions["broad_" + str(width)] = {
                "kind": "addition",
                "candidate": {
                    "x": [
                        float(np.log(d.fluence_scale * width)),
                        at,
                        float(np.log(width)),
                        0.0,
                        -1.0,
                        float(x[5 * n]),
                    ],
                    "n": 1,
                    "scatter": False,
                },
            }
    if n < d.c.get("max_components", 7):
        comp = x[: 5 * n].reshape(n, 5).copy()
        k = int(np.argmax(comp[:, 2]))
        b = comp[k].copy()
        w = np.exp(b[2])
        comp[k, 1] -= 0.7 * w
        b[1] += 0.7 * w
        comp[k, 2] -= 0.45
        b[2] -= 0.45
        comp[k, 0] -= np.log(2.0)
        b[0] -= np.log(2.0)
        directions["split"] = {
            "kind": "replacement",
            "candidate": {
                "x": np.r_[comp.ravel(), b, x[5 * n :]].tolist(),
                "n": n + 1,
                "scatter": s,
            },
        }
    if not s:
        tau = max(0.4, float(np.median(np.exp(x[2 : 5 * n : 5]))) * 0.7)
        q = np.r_[x, np.log(tau)]
        q[1 : 5 * n : 5] -= tau
        directions["tail"] = {
            "kind": "replacement",
            "candidate": {"x": q.tolist(), "n": n, "scatter": True},
        }
    return directions


def propose(data, incumbent, deadline, source_sha256, residual_pair=None):
    """Return one optional challenger, or None. All failures retain incumbent.

    The deadline is an absolute monotonic timestamp. The caller supplies its scored incumbent.
    """
    begin = time.monotonic()
    receipt = dict(
        challenger=None,
        automatic_acceptance=False,
        reason="not_started",
        elapsed_seconds=0.0,
        failures=[],
    )
    shared_state = None
    try:
        if deadline <= begin:
            receipt["reason"] = "budget_expired"
            return receipt
        d = data
        config = data.c
        raw_sha = source_sha256
        receipt.update(
            source_sha256=source_sha256,
            raw_identity_kind="bound_raw_npz_sha256",
            prepared_data_reused=True,
        )
        preparation = time.monotonic()
        n = int(incumbent["n"])
        s = bool(incumbent["scatter"])
        x = np.asarray(incumbent["x"], dtype=float)
        if n < 1 or len(x) != 5 * n + 1 + int(s) or (not np.isfinite(x).all()):
            raise ValueError("Invalid incumbent parameters")
        # Work counters remain cumulative in d; receipt deltas isolate this hook.
        shared_state = (d, d.e.budget, d.native_work, d.evaluations)
        # This proposal stage has its own work allowance.
        # Reusing pixels/preprocessing must not silently consume its limit with
        # first-stage counters; restore cumulative telemetry in finally.
        d.e.budget = {"evaluations": 0, "native_work": 0}
        d.native_work = 0
        d.evaluations = 0
        budget_before = dict(d.e.budget)
        native_before = 0
        evaluations_before = 0
        inc = dict(incumbent)
        cache = NativeProbeCache(d, pulse, raw_sha)
        if residual_pair is not None:
            cache.values[parameter_key(inc)] = {
                "positive": residual_pair[0],
                "negative": residual_pair[1],
            }
        receipt["objective_fingerprint"] = cache.objective
        receipt["preparation_seconds"] = time.monotonic() - preparation
        noise = nuisance_diagnostics(d, inc, pulse)
        receipt["nuisance"] = noise
        if noise["flags"]:
            receipt["reason"] = "provisional_nuisance_domain"
            return receipt
        if time.monotonic() >= deadline:
            receipt["reason"] = "preparation_used_budget"
            return receipt
        directions = _directions(d, inc, cache)
        prices = cache.price(inc, directions)
        receipt["probes"] = prices
        name = max(prices, key=lambda k: prices[k]["loss_reduction"])
        receipt["winning_direction"] = name
        hurdle = max(20.0, 0.01 * max(0, inc["gain"]))
        receipt["trial_hurdle"] = hurdle
        if prices[name]["loss_reduction"] <= hurdle:
            receipt["reason"] = "no_material_priced_challenge"
            return receipt
        desc = directions[name]
        row = desc["candidate"]
        q = np.asarray(row["x"])
        nn = row["n"]
        ss = row["scatter"]
        if desc["kind"] == "addition":
            if n >= config.get("max_components", 7):
                receipt["reason"] = "component_budget"
                return receipt
            comp = q[:5].copy()
            comp[0] += np.log(max(1e-08, prices[name]["amplitude"]))
            q = np.r_[x[: 5 * n], comp, x[5 * n :]]
            nn = n + 1
            ss = s
        candidates = []

        def solve(q, nn, ss, label):
            if time.monotonic() >= deadline:
                return
            try:
                row = optimizer(
                    d,
                    nn,
                    ss,
                    False,
                    q,
                    label,
                    16,
                    min(deadline, time.monotonic() + 8.0),
                )
                if time.monotonic() < deadline:
                    row = optimizer(
                        d,
                        nn,
                        ss,
                        True,
                        np.asarray(row["x"]),
                        label + "-native",
                        4,
                        min(deadline, time.monotonic() + 8.0),
                    )
                row["gain"], row["rms_gain"], row["energy"] = score(d, row)
                candidates.append(
                    {
                        k: row[k]
                        for k in (
                            "name",
                            "n",
                            "scatter",
                            "x",
                            "gain",
                            "rms_gain",
                            "energy",
                            "termination",
                            "active_bounds",
                        )
                    }
                )
            except (ValueError, Limit, np.linalg.LinAlgError) as ex:
                receipt["failures"].append(str(ex))

        solve(q, nn, ss, "residual_" + name)
        # One alternate pulse shape at the SAME newly supported count; no ladder.
        if candidates and not ss and time.monotonic() < deadline:
            best = max(candidates, key=lambda a: a["gain"])
            q = np.asarray(best["x"])
            tau = max(0.4, float(np.median(np.exp(q[2 : 5 * nn : 5]))) * 0.7)
            q = np.r_[q, np.log(tau)]
            q[1 : 5 * nn : 5] -= tau
            solve(q, nn, True, "residual_supported_tail")
        receipt["candidate_count"] = len(candidates)
        receipt["candidates"] = candidates
        if candidates:
            best = max(candidates, key=lambda a: a["gain"])
            if best["gain"] > inc["gain"]:
                receipt["challenger"] = best
                receipt["reason"] = "native_rescore_required_by_central_selector"
            else:
                receipt["reason"] = "challenger_did_not_improve_native_gain"
        else:
            receipt["reason"] = "no_completed_challenger"
        receipt["work"] = {
            k: d.e.budget.get(k, 0) - v for k, v in budget_before.items()
        }
        receipt["native_work_delta"] = d.native_work - native_before
        receipt["evaluations_delta"] = d.evaluations - evaluations_before
        receipt["cache"] = {
            "residual_scans": cache.scans,
            "probe_passes": cache.probe_passes,
            "hits": cache.hits,
        }
    except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        receipt["reason"] = "proposal_failure"
        receipt["failures"].append(repr(exc))
    finally:
        if shared_state is not None:
            d, original_budget, original_native, original_evaluations = shared_state
            hook_work = dict(d.e.budget)
            receipt["work"] = hook_work
            receipt["native_work_delta"] = d.native_work
            receipt["evaluations_delta"] = d.evaluations
            for key, value in hook_work.items():
                original_budget[key] = original_budget.get(key, 0) + value
            d.e.budget = original_budget
            d.native_work = original_native + d.native_work
            d.evaluations = original_evaluations + d.evaluations
        receipt["elapsed_seconds"] = time.monotonic() - begin
        receipt["deadline_overrun_seconds"] = max(0.0, time.monotonic() - deadline)
    return receipt
