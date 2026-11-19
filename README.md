# burst surveyor

burst surveyor automatically fits dedispersed radio bursts and folded pulsar profiles. It estimates starting parameters from the observed intensity, compares models with different component counts and with or without scattering, and uses [FitBurst](https://github.com/CHIMEFRB/fitburst) to reconstruct the selected model on the original frequency and time grid.

## Fit an observation

Follow the [installation instructions](docs/guide.md#installation) for Python 3.12, then generate and fit the bundled two-pulse example.

```sh
burstsurveyor example two-pulse.npz
burstsurveyor fit two-pulse.npz run --workers 1
burstsurveyor verify run
```

The fit recovers two components near 24.5 and 36.5 ms and selects zero scattering. `run/results.json` records the component count, scores and path to the reconstruction. To fit your own data, replace `two-pulse.npz` with an [observation in NPZ format](docs/reference.md) or a directory of observations. Use a new output directory for each run.

## How the fit works

For a burst, the search averages usable frequency channels over the full band, halves and quarters. It looks for peaks at several smoothing widths. Their positions and widths supply starting estimates for Gaussian components, which are fitted jointly to the dynamic spectrum. The search tests models with and without a scattering tail and checks the residuals for missed emission. It scores each candidate by the reduction in noise-weighted squared error on the original grid and favors simpler fits when scores are close.

The [methods walkthrough](docs/methods.md) follows the bundled observation from its first peak guesses through the scoring and stopping rules to the exported FitBurst parameters.

In BaseCat2 event 156410110, the initial search found three peaks. After fitting them, a peak near **24.7 ms** remained in the residual (bottom middle). That peak supplied the starting time for a fourth component. After refitting, most of the emission at that time is gone from the residual (bottom right). Gray bands mark masked channels.

[![Observed profile and dynamic spectrum at left. At right, the three- and four-component models are shown above their residuals on the same scales.](examples/event-156410110-fit-stages.png)](https://immortalari.github.io/burst-surveyor/)

The search then found an earlier feature near 16.3 ms. The selected fit used six components, two of which overlap at that early feature and have different spectra. [Step through the four fits.](https://immortalari.github.io/burst-surveyor/)

The [fit gallery](examples/README.md) includes other fast radio bursts, folded pulsar profiles and the bundled example. The [selection guide](docs/selection.md) describes the search in each fitting mode.

## Reviewing fits

Use [`prepare`](docs/guide.md#figures-and-residuals) to generate figures and screen for possible missed components. You can inspect the results yourself or use the [optional AI-assisted review instructions](docs/ai-assisted-review/README.md).

## Citation

If you publish results obtained with burst surveyor, cite [the software](CITATION.cff) and [Fonseca et al. (2024)](https://doi.org/10.3847/1538-4365/ad27d6).
