# burst surveyor

burst surveyor automatically fits dedispersed radio bursts and folded pulsar profiles. It estimates starting parameters from the observed intensity, compares models with different component counts and scattering, and exports the selected reconstruction on the original frequency and time grid using [FitBurst](https://github.com/CHIMEFRB/fitburst).

## How the fit works

For a burst, the search averages usable frequency channels over the full band and over narrower bands, then looks for peaks at several smoothing widths. Those peaks give starting times and widths for a joint fit to the dynamic spectrum. The search tries models with and without a scattering tail and checks the residuals for missed emission.

In BaseCat2 event 156410110, the initial search found three peaks. The three-component fit left a fourth peak near **24.7 ms**. The fitter used that residual to place another component and refit the burst. Compare the lower middle and lower right panels at that time. Gray bands mark masked channels.

[![Observed profile and dynamic spectrum at left, three- and four-component models above their residuals at right. Adding the fourth component removes most of the residual peak near 24.7 ms.](examples/event-156410110-fit-stages.png)](https://immortalari.github.io/burst-surveyor/)

The search then found an earlier feature near 16.3 ms. The selected fit used six components, two of which overlap at that early feature and have different spectra. Candidates are scored by the reduction in noise-weighted squared error on the original grid, with a preference for simpler fits when scores are close. [Explore all four stages](https://immortalari.github.io/burst-surveyor/) · [Model selection](docs/selection.md#default-burst-search)

## Comparison with human fits

In a benchmark of 95 events selected for low human reduced χ², the median reduced χ² was 1.0048 for the automated fits and 1.0052 for the human fits. The automated fits used 247 components in total, compared with 118 in the human fits. The fits disagreed on whether to include scattering in 13 events.

Event 1114368517 illustrates this ambiguity. The two pulse profiles nearly coincide, but the automated fit gives zero scattering and the human reference gives 0.123 ms at 600 MHz. For a folded pulsar example, J1849+0409 shows how the fits follow the main pulse and interpulse over one rotation. [Scattering comparison](examples/README.md#event-1114368517) · [Pulsar comparison](examples/README.md#j18490409) · [Full gallery](examples/README.md)

## Fit an observation

Follow the [installation instructions](docs/guide.md#installation), then run the bundled example in a fresh output directory.

```sh
burstsurveyor example two-pulse.npz
burstsurveyor fit two-pulse.npz run --workers 1
burstsurveyor verify run
```

The expected fit has two components near 24.5 and 36.5 ms, with no scattering selected. `run/results.json` records the component count, scores and path to the reconstruction. To fit your own data, replace `two-pulse.npz` with an observation or a directory of NPZ files. [Example output](examples/README.md#bundled-two-pulse-example) · [Input format](docs/reference.md)

## Review and citation

To generate figures and screen for possible missed components, use `prepare`. AI-assisted review is optional. [Review guide](docs/guide.md#figures-and-residuals) · [AI-assisted review](docs/ai-assisted-review/README.md)

If you publish results obtained with burst surveyor, cite the software and the FitBurst paper by Fonseca et al. (2024). [Software citation](CITATION.cff) · [FitBurst paper](https://doi.org/10.3847/1538-4365/ad27d6)
