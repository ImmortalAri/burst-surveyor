"""Conservative, one-pass screening of original mask neighbors from off-burst data."""

import numpy as np


def _raw_valid(raw):
    valid = np.zeros(len(raw), dtype=bool)
    for start in range(0, len(raw), 32):
        block = raw[start : start + 32]
        if np.issubdtype(block.dtype, np.integer):
            good = np.any(block != block[:, :1], axis=1) & (
                np.sum(block, axis=1, dtype=float) != 0
            )
        else:
            good = (
                np.isfinite(block).all(axis=1)
                & (np.ptp(block, axis=1) > 0)
                & (np.sum(block, axis=1) != 0)
            )
        valid[start : start + len(block)] = good
    return valid


def _evidence(values, off, adjacent, quarter):
    y = np.asarray(values, dtype=float)
    sample = y[off]
    early = float(np.median(sample[:quarter]))
    late = float(np.median(sample[-quarter:]))
    shift = abs(early - late)
    spread = 1.4826 * float(np.median(abs(sample - np.median(sample))))
    differences = y[1:][adjacent] - y[:-1][adjacent]
    noise = 1.4826 * float(np.median(abs(differences - np.median(differences)))) / np.sqrt(2)
    floor = np.finfo(float).eps * max(1.0, float(np.max(abs(sample))))
    noise = max(noise, floor)
    # Median standard error under independent Gaussian noise is only a scale
    # here, not a p value: temporal correlation is itself part of the screen.
    drift = shift / (noise * np.sqrt(np.pi / quarter))
    relative_shift = shift / max(spread, floor)
    left = y[:-1][adjacent].copy()
    right = y[1:][adjacent].copy()
    left -= left.mean()
    right -= right.mean()
    scale = max(float(np.max(abs(left))), float(np.max(abs(right))), floor)
    left /= scale
    right /= scale
    norm = np.sqrt(np.dot(left, left) * np.dot(right, right))
    correlation = float(np.dot(left, right) / norm) if norm > 0 else 0.0
    return dict(
        early_median=early,
        late_median=late,
        offburst_noise_sigma=float(noise),
        drift_noise_scale=float(drift),
        drift_relative_to_offburst_mad=float(relative_shift),
        adjacent_offburst_correlation=correlation,
    )


def screen_mask_neighbors(raw, off, *, uncertain=False):
    """Return ``(extra_channel_ids, receipt)`` without modifying data or masks.

    ``off`` must come from the existing raw-only off-burst estimator on the
    original data, before any channel train/validation split. Original invalid
    channels are derived here, so a training omission cannot create a mask gap.
    This developmental policy requires drift AND correlation AND proximity to
    an original invalid channel; it never expands a gap unconditionally.
    """
    raw = np.asarray(raw)
    off = np.asarray(off)
    if raw.ndim != 2 or raw.shape[1] < 2 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("Expected a numeric channel-by-time array")
    if (
        off.ndim != 1
        or not np.issubdtype(off.dtype, np.integer)
        or (len(off) and (off[0] < 0 or off[-1] >= raw.shape[1]))
        or np.any(np.diff(off) <= 0)
    ):
        raise ValueError("Off-burst indices must be sorted, unique native time indices")
    policy = dict(
        neighbor_distance=2,
        minimum_offburst_samples=64,
        minimum_adjacent_pairs=48,
        minimum_offburst_fraction=0.5,
        minimum_drift_noise_scale=6.0,
        minimum_drift_relative_to_offburst_mad=1.0,
        minimum_adjacent_correlation=0.35,
        maximum_excluded_fraction=0.03,
        maximum_excluded_channels=24,
    )
    receipt = dict(
        method="offburst_mask_neighbors",
        developmental=True,
        status="unchanged",
        policy=policy,
        offburst_samples=int(len(off)),
        original_usable_channels=0,
        candidate_channels=0,
        extra_masked_channels=[],
        evidence=[],
    )
    empty = np.empty(0, dtype=int)
    if uncertain:
        receipt["status"] = "skipped_uncertain_offburst"
        return empty, receipt
    time_mask = np.zeros(raw.shape[1], bool)
    time_mask[off] = True
    adjacent = time_mask[:-1] & time_mask[1:]
    receipt["adjacent_offburst_pairs"] = int(adjacent.sum())
    quarter = len(off) // 4
    if (
        len(off) < policy["minimum_offburst_samples"]
        or len(off) < policy["minimum_offburst_fraction"] * raw.shape[1]
        or adjacent.sum() < policy["minimum_adjacent_pairs"]
        or np.median(off[:quarter]) > 0.3 * (raw.shape[1] - 1)
        or np.median(off[-quarter:]) < 0.7 * (raw.shape[1] - 1)
    ):
        receipt["status"] = "skipped_insufficient_offburst"
        return empty, receipt
    valid = _raw_valid(raw)
    receipt["original_usable_channels"] = int(valid.sum())
    neighbors = np.zeros(len(raw), bool)
    for offset in range(1, policy["neighbor_distance"] + 1):
        neighbors[offset:] |= ~valid[:-offset]
        neighbors[:-offset] |= ~valid[offset:]
    candidates = np.flatnonzero(valid & neighbors)
    receipt["candidate_channels"] = int(len(candidates))
    invalid = np.flatnonzero(~valid)
    flagged = []
    for channel in candidates:
        result = _evidence(raw[channel], off, adjacent, quarter)
        if (
            result["drift_noise_scale"] >= policy["minimum_drift_noise_scale"]
            and result["drift_relative_to_offburst_mad"] >= policy["minimum_drift_relative_to_offburst_mad"]
            and result["adjacent_offburst_correlation"] >= policy["minimum_adjacent_correlation"]
        ):
            flagged.append(int(channel))
            receipt["evidence"].append(
                dict(
                    channel=int(channel),
                    distance_to_original_invalid=int(np.min(abs(invalid - channel))),
                    reason="correlated_offburst_drift_near_original_mask",
                    **result,
                )
            )
    cap = min(policy["maximum_excluded_channels"], int(policy["maximum_excluded_fraction"] * valid.sum()))
    receipt["safety_cap"] = int(cap)
    if len(flagged) > cap or valid.sum() - len(flagged) < 6:
        receipt["status"] = "skipped_safety_cap"
        receipt["proposed_channels"] = flagged
        return empty, receipt
    receipt["extra_masked_channels"] = flagged
    if flagged:
        receipt["status"] = "applied"
    return np.asarray(flagged, dtype=int), receipt
