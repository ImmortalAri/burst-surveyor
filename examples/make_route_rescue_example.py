"""Generate the bundled synthetic example of blended peaks."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr


# Peak height, arrival bin, time width in bins, spectral center and width in MHz.
PULSES = (
    (1.7, 43, 2.0, 565, 80),
    (1.8, 51, 1.9, 550, 77),
    (1.9, 60, 1.6, 515, 79),
    (17, 71, 2.7, 535, 79),
    (15, 89, 2.5, 475, 89),
    (11, 119, 2.25, 455, 70),
)


def generate(path):
    channels, samples = 96, 168
    grid = dict(res_time=.0005, res_freq=400/channels, freqs_bin0=400.0,
                times_bin0=0., is_dedispersed=True)
    frequency = grid["freqs_bin0"] + (np.arange(channels) + .5) * grid["res_freq"]
    bins = np.arange(samples)
    raw = np.random.default_rng(1170).normal(size=(channels, samples))
    raw *= .42
    for height, center, width, pivot, bandwidth in PULSES:
        spectral = np.exp(-.5 * (np.log(frequency / pivot) / (bandwidth / pivot)) ** 2)
        temporal = (ndtr((bins + .5 - center) / width) - ndtr((bins - .5 - center) / width))
        temporal = temporal * width * np.sqrt(2 * np.pi)
        raw += height * spectral[:, None] * temporal[None, :]
    with path.open("xb") as stream:
        np.savez_compressed(stream, data_full=raw.astype("float32"), metadata=np.array(json.dumps(grid)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
