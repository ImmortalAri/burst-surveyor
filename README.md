# burst surveyor

burst surveyor fits dedispersed radio bursts and folded pulsar profiles with [FitBurst](https://github.com/CHIMEFRB/fitburst). It automatically fits candidate models to the observed intensity and exports the selected reconstruction on the original frequency and time grid.

## How the fit works

The input is a dynamic spectrum, the observed intensity as a function of frequency and time. To find starting parameters, burst surveyor averages the usable channels into time profiles over the full band, halves and quarters. It searches each profile for peaks at several smoothing widths. A pulse confined to the lower half of the band can stand out there even when it is weak in the full-band average.

The default burst search uses those peaks to initialize Gaussian components, then fits their sum to the dynamic spectrum. Each component has its own arrival time, width, fluence and spectrum. The components share a residual dispersion measure. The search tries models with and without a scattering tail and checks the residuals for structure that could justify another component.

Candidates are compared on the original frequency and time grid. The score measures how much the model reduces the noise-weighted sum of squared residuals relative to a baseline alone. Selection penalizes extra parameters and favors a simpler model when scores are close. FitBurst then renders the selected model on the original grid, and burst surveyor scores the exported reconstruction. The run records the alternatives so the choice can be inspected. [See the search on a real burst](#event-156410110) · [Selection details](docs/selection.md#default-burst-search)

## Fit an observation

In a fresh 64-bit Python 3.12 environment, install burst surveyor and FitBurst, then generate and fit the two-pulse example.

```sh
python -m pip install -c https://github.com/ImmortalAri/burst-surveyor/releases/download/v0.1.0/constraints.txt https://github.com/ImmortalAri/burst-surveyor/releases/download/v0.1.0/fitburst-0.1.0-py3-none-any.whl https://github.com/ImmortalAri/burst-surveyor/releases/download/v0.1.0/burst_surveyor-0.1.0-py3-none-any.whl
burstsurveyor example two-pulse.npz
burstsurveyor fit two-pulse.npz run --workers 1
burstsurveyor verify run
```

`run/results.json` records the fit status, component count and scores for each observation. For an exported fit, the `prediction` field gives the path to the reconstructed array relative to `run`. To fit several observations, pass a directory of [NPZ files](docs/reference.md) in place of the example. Use a new output directory for each run.

### Explore the two-pulse fit

The search finds a peak at 24.5 ms in the full band and a weaker peak at 36.5 ms in the lower half of the band. It tries both a single scattered pulse and a model with two pulses. Open the views below to compare them. All five use the same time, frequency and intensity scales.

<details open>
<summary>Observation</summary>

![Observed intensity after baseline subtraction. The later pulse is brightest near 490 MHz.](examples/two-pulse-views/observation.png)

</details>

<details>
<summary>One-component fit</summary>

![Single scattered component fitted at the brighter peak near 24.5 ms.](examples/two-pulse-views/one-component.png)

</details>

<details>
<summary>One-component residual</summary>

Subtracting the single scattered pulse leaves a patch of emission near 36.5 ms and 490 MHz. Both peaks were found in the initial search. This residual shows what the single component misses.

![Residual intensity with the later, low-frequency pulse still visible.](examples/two-pulse-views/one-component-residual.png)

</details>

<details>
<summary>Selected two-component fit</summary>

![Two fitted Gaussian components, with the later component concentrated at lower frequencies.](examples/two-pulse-views/two-components.png)

</details>

<details>
<summary>Two-component residual</summary>

![Residual intensity after subtracting the selected fit. The patch near 36.5 ms is largely removed.](examples/two-pulse-views/two-component-residual.png)

</details>

The single scattered pulse gives a gain of 5417.72. Two Gaussian components give 7085.52, and adding scattering gives 7085.45. The search selects two components without scattering. After refinement their arrival times at 600 MHz are 24.49 and 36.58 ms.

![Two-pulse synthetic observation, automated profile and model spectrum, and observed minus modeled intensity.](examples/two-pulse-fit.png)

## Fast radio burst examples

The three BaseCat2 examples compare the automated fit with a human reference on the same time grid. Each figure shows the pulse profiles, the observed dynamic spectrum, both model spectra and the residuals (observed minus modeled intensity). Gray bands mark masked channels. Model components can overlap in time and frequency, so they should not automatically be interpreted as separate physical pulses.

### Event 156410110

The initial search found three peaks. After fitting them, the residual still showed emission around 24.7 and 16.3 ms. Open the fits in sequence to see how the search accounted for each feature. The panels show the same time window and intensity scales throughout.

<details open>
<summary>Three components · initial peaks</summary>

The search found peaks near 18.7, 21.6 and 30.1 ms. Of the two three-component trials, the model with scattering gave the higher gain. It follows the tails of those peaks but misses most of the emission near 24.7 ms.

![Three-component fit with scattering. The profile and residual spectrum show a missed peak near 24.7 ms.](examples/event-156410110-views/3-components.png)

</details>

<details>
<summary>Four components · peak near 24.7 ms</summary>

A positive residual supplied the starting point for a fourth component. After refitting, the gain rose from 6928 to 8561. A smaller peak near 16.3 ms remained.

![Four-component fit and residual. The model now follows the peak near 24.7 ms but misses the earlier feature.](examples/event-156410110-views/4-components.png)

</details>

<details>
<summary>Five components · earlier peak near 16.3 ms</summary>

A second residual search added a component at the earlier peak. The gain rose to 9176.

![Five-component fit with the earlier peak included.](examples/event-156410110-views/5-components.png)

</details>

<details>
<summary>Six components · frequency structure of the early peak</summary>

The search tried splitting the early component into two with different spectra. Their fitted arrival times at 600 MHz were 16.20 and 16.35 ms. The gain rose to 9228, a smaller improvement than either of the previous additions. This trial was selected for export. The two overlapping terms describe how the early feature varies across the band. The number of physical pulses remains uncertain.

![Exported six-component fit. Two overlapping terms model the frequency structure of the early feature.](examples/event-156410110-views/6-components.png)

</details>

Both fits follow the main peaks. The automated fit uses six components and follows the smaller peak near 16.5 ms more closely than the five-component human reference.

![Observed profile and residuals for event 156410110. Both model spectra and their residuals are shown below the profiles.](examples/event-156410110-comparison.png)

### Event 300938875

The automated fit recovers a peak near 3.2 ms that is absent from the saved human reference candidate. The atlas records no final human selection for this event. The weaker feature near 4.6 ms remains uncertain.

![Observed profile and residuals for event 300938875. The residual spectrum of the human reference candidate shows a missed peak near 3.2 ms.](examples/event-300938875-comparison.png)

### Event 1114368517

Both fits use one component and give similar pulse profiles. At 600 MHz, the automated fit selects zero scattering and the human reference gives 0.123 ms. The time bin is 0.655 ms.

![Observed profile and residuals for event 1114368517. The profiles and model spectra nearly coincide.](examples/event-1114368517-comparison.png)

### FRB 20190412A

The observed pulse profile and spectrum are shown alongside the fitted reconstruction and residuals.

![FRB 20190412A pulse profile, observed spectrum, fitted reconstruction and residuals](examples/frb20190412a.png)

## Pulsars

[Pulsar mode](docs/selection.md#pulsar-profiles) fits one complete rotation, including emission across the phase boundary, and compares models with and without scattering.

### J1849+0409

The two fits follow the main pulse and interpulse. The largest difference between the profiles occurs near the rotation boundary.

![J1849+0409 over one rotation. The profiles include the human reference. The spectral panels show the observation, automated model and residual.](examples/pulsar-j1849-0409-comparison.png)

### J0824+0028

![J0824+0028 pulse profile, observed spectrum, fitted reconstruction and residuals](examples/pulsar-j0824-0028.png)

## Optional review

To generate figures and screen for possible missed components, use `prepare` as described in the [guide](docs/guide.md#figures-and-residuals). For review with a coding agent, see [AI-assisted review](docs/ai-assisted-review/README.md).

## Citation

If you publish results obtained with burst surveyor, cite the software and the FitBurst paper by Fonseca et al. (2024). [Software citation](CITATION.cff) · [FitBurst repository](https://github.com/CHIMEFRB/fitburst) · [FitBurst paper](https://doi.org/10.3847/1538-4365/ad27d6)
