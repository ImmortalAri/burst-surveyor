import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from burstsurveyor.science.engine import InputError
from burstsurveyor.science.input import inspect_npz, load_raw
from burstsurveyor.science.model import Coordinates, pulse


class ChannelTimingTests(unittest.TestCase):
    def test_channel_grid_and_derivatives(self):
        frequencies = np.linspace(420., 760., 7)
        coords = Coordinates(frequencies, .001, frequencies)
        channels = np.arange(len(frequencies))
        times = np.arange(25, dtype=float)
        phases = np.linspace(-.4, .45, len(channels))
        pars = np.array([2., 12., -.5, .2, -.3, .8, -.2])
        for scatter in (False, True):
            x = pars if scatter else pars[:-1]
            base, _ = pulse(x, 1, scatter, coords, channels, times)
            coords.channel_time_offset_bins = np.zeros(len(channels))
            np.testing.assert_array_equal(base, pulse(x, 1, scatter, coords, channels, times)[0])
            coords.channel_time_offset_bins = phases
            actual, grad = pulse(x, 1, scatter, coords, channels, times, True)
            del coords.channel_time_offset_bins
            for i in channels:
                expected, d = pulse(x, 1, scatter, coords, [i], times + phases[i], True)
                np.testing.assert_allclose(actual[i], expected[0], atol=1e-12)
                np.testing.assert_allclose(grad[:, i], d[:, 0], atol=1e-11)

    def test_offset_member_validation(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'obs.npz'
            grid = dict(res_time=.001, res_freq=1., freqs_bin0=420., times_bin0=0., is_dedispersed=True)
            def write(offsets):
                np.savez(path,data_full=np.zeros((21,25)),metadata=json.dumps(grid),channel_time_offset_s=offsets)
            write(np.zeros(25))
            with self.assertRaises(InputError):
                inspect_npz(path)
            write(np.full(21,np.nan))
            with self.assertRaises(InputError):
                load_raw(path,tmp)
            offsets=np.linspace(-.0004,.0004,21)
            write(offsets)
            np.testing.assert_array_equal(load_raw(path,tmp)[1]['channel_time_offset_s'], offsets)
