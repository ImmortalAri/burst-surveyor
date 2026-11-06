"""Measure what one fixed template could add before paying for a nonlinear fit."""

import numpy as np


def probe(raw, channels, variances, incumbent, templates, tile_channels=32):
    """Return exact profiled weighted loss reductions for fixed template additions.

    incumbent(ii) returns a pulse prediction on channels ii. Each template is
    (callable, maximum amplitude). For a replacement, use (challenger-incumbent)
    and maximum amplitude 1. For an additive positive pulse use +infinity.
    All candidates share support and channel weights. Inputs may be train-only.
    """
    channels = np.asarray(channels)
    if (
        channels.ndim != 1
        or not len(channels)
        or len(np.unique(channels)) != len(channels)
    ):
        raise ValueError("Expected unique channel indices")
    if (
        not np.issubdtype(channels.dtype, np.integer)
        or channels.min() < 0
        or channels.max() >= len(raw)
    ):
        raise ValueError("Invalid channel index")
    if tile_channels < 1 or not templates:
        raise ValueError("Expected positive tile size and at least one template")
    products = {name: [0.0, 0.0] for name in templates}
    for start in range(0, len(channels), tile_channels):
        ii = channels[start : start + tile_channels]
        y = np.asarray(raw[ii], dtype=float)
        p = np.asarray(incumbent(ii), dtype=float)
        var = np.asarray(variances)[ii, None]
        if (
            p.shape != y.shape
            or not np.isfinite(y).all()
            or not np.isfinite(p).all()
            or not np.isfinite(var).all()
            or (var <= 0).any()
        ):
            raise ValueError("Invalid data, prediction, or noise support")
        residual = y - p
        residual -= residual.mean(axis=1, keepdims=True)
        for name, (generate, limit) in templates.items():
            z = np.asarray(generate(ii), dtype=float)
            if z.shape != y.shape or not np.isfinite(z).all() or not limit > 0:
                raise ValueError("Invalid template or amplitude bound")
            centred = z - z.mean(axis=1, keepdims=True)
            products[name][0] += float(np.sum(centred * residual / var))
            products[name][1] += float(np.sum(centred**2 / var))
    scored = {}
    for name, (cross, energy) in products.items():
        amplitude = (
            min(templates[name][1], max(0.0, cross / energy)) if energy > 0 else 0.0
        )
        scored[name] = {
            "amplitude": amplitude,
            "loss_reduction": amplitude * (2 * cross - amplitude * energy),
        }
    return scored
