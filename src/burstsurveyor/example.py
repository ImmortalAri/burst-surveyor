"""A deterministic, redistributable, two-pulse synthetic observation."""

import json
from pathlib import Path

import numpy as np


def create_example(destination):
    """Create an example NPZ without overwriting an existing file."""
    path = Path(destination)
    if path.suffix.lower() != ".npz":
        raise ValueError("Use a filename ending in .npz for the example")
    channels, samples = 64, 144
    grid = dict(
        res_time=0.0005,
        res_freq=6.25,
        freqs_bin0=400.0,
        times_bin0=0.0,
        is_dedispersed=True,
    )
    frequencies = grid["freqs_bin0"] + (np.arange(channels) + 0.5) * grid["res_freq"]
    bins = np.arange(samples)
    rng = np.random.default_rng(7942)
    raw = rng.normal(size=(channels, samples)) + 0.3 * np.cos(
        frequencies[:, None] / 70.0
    )
    for height, center, width, pivot, bandwidth in (
        (7.0, 49.0, 2.7, 640.0, 90.0),
        (4.5, 73.0, 3.4, 490.0, 47.0),
    ):
        spectral = np.exp(-0.5 * ((frequencies - pivot) / bandwidth) ** 2)
        temporal = np.exp(-0.5 * ((bins - center) / width) ** 2)
        raw += height * spectral[:, None] * temporal[None, :]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, data_full=raw.astype("float32"), metadata=np.array(json.dumps(grid))
        )
    return path
