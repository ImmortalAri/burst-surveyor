# Fit gallery

## Fast radio burst examples

The three BaseCat2 figures show archived comparisons of automated and human fits on the same time grid. Each figure shows the pulse profiles, the observed dynamic spectrum, both model spectra and the residuals (observed minus modeled intensity). Gray bands mark masked channels. Model components can overlap in time and frequency, so they should not automatically be interpreted as separate physical pulses.

### Event 156410110

Both fits follow the main peaks. The automated fit uses six components and follows the smaller peak near 16.5 ms more closely than the five-component human reference. [Follow the v0.2.0 worked example through the selected fit.](../README.md#worked-example-for-basecat2-event-156410110)

![Observed profile and residuals for event 156410110. Both model spectra and their residuals are shown below the profiles.](event-156410110-comparison.png)

### Event 300938875

The automated fit recovers a peak near 3.2 ms that is absent from the saved human reference candidate. The atlas records no final human selection for this event. The weaker feature near 4.6 ms remains uncertain.

![Observed profile and residuals for event 300938875. The residual spectrum of the human reference candidate shows a missed peak near 3.2 ms.](event-300938875-comparison.png)

### Event 1114368517

Both fits use one component and give similar pulse profiles. At 600 MHz, the automated fit selects zero scattering and the human reference gives 0.123 ms. The time bin is 0.655 ms.

![Observed profile and residuals for event 1114368517. The profiles and model spectra nearly coincide.](event-1114368517-comparison.png)

### FRB 20190412A

The observed pulse profile and spectrum are shown alongside the fitted reconstruction and residuals.

![FRB 20190412A pulse profile, observed spectrum, fitted reconstruction and residuals](frb20190412a.png)

## Pulsars

[Pulsar mode](../docs/selection.md#pulsar-profiles) fits one complete rotation, including emission across the phase boundary, and compares models with and without scattering.

### J1849+0409

The two fits follow the main pulse and interpulse. The largest difference between the profiles occurs near the rotation boundary.

![J1849+0409 over one rotation. The profiles include the human reference. The spectral panels show the observation, automated model and residual.](pulsar-j1849-0409-comparison.png)

### J0824+0028

![J0824+0028 pulse profile, observed spectrum, fitted reconstruction and residuals](pulsar-j0824-0028.png)

## Bundled two-pulse example

The search finds a peak at 24.5 ms in the full band and a weaker peak at 36.5 ms in the lower half of the band. It tries both a single scattered pulse and a model with two pulses. [Compare the observation, fits and residuals.](https://immortalari.github.io/burst-surveyor/two-pulse.html)

The single scattered pulse gives a gain of 5417.72. Two Gaussian components give 7085.52, and adding scattering gives 7085.45. The search selects two components without scattering. After refinement their arrival times at 600 MHz are 24.49 and 36.58 ms. [Expected scores and checksum of the bundled NPZ](expected.json)

![Two-pulse synthetic observation, automated profile and model spectrum, and observed minus modeled intensity.](two-pulse-fit.png)

## Bundled rescue example

[`route-rescue-crowded.npz`](route-rescue-crowded.npz) contains six simulated pulses over Gaussian noise. The default fit blends the peaks at **21.5 ms** and **25.5 ms** into one hump. The router detects the missing structure and jointly refits all six peaks.

```sh
burstsurveyor fit examples/route-rescue-crowded.npz synthetic-fit --workers 1
burstsurveyor route-rescue synthetic-fit examples/route-rescue-crowded.npz synthetic-rescue
burstsurveyor verify synthetic-fit --route-rescue synthetic-rescue
```

The routed result is a `candidate` with six distinct fitted peaks. Weighted squared error falls by about **6.16%** across the burst and by **5.32%** and **4.99%** on the two held-out channel sets under the primary weights. Both held-out checks also improve under the alternate weights. The [generator](make_route_rescue_example.py) can recreate the example in a new file.
