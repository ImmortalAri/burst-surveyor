# How burst surveyor fits a burst

This walkthrough follows one dedispersed observation through `burstsurveyor fit` in the default `bounded` mode. Burst surveyor chooses the candidate models, fits their parameters and compares their predictions. By default it evaluates the selected pulse model on the original grid for export. Folded pulsar profiles and the experimental spectral modes have different search rules. [Selection modes](selection.md)

## From intensity to starting guesses

An observation supplies intensity indexed by frequency channel and time sample, along with the acquisition grid. `load_raw` reads the array and grid. `ReconstructionData` identifies usable channels, estimates their noise from off-burst samples, and groups nearby channels for initial fitting. Peak discovery uses the observed time bins. When channel offsets are supplied, the native fit and export evaluate the model at each channel's sample times. [Input format and timing](reference.md#channel-timing) and [noise estimation](../src/burstsurveyor/science/objective.py)

To locate a pulse, `find_regions` subtracts each usable channel’s median, divides by its estimated noise, and clips values to the interval [−5, 5] for peak finding. It forms seven time profiles by averaging the full set of usable channels, its two halves, and its four quarters in channel order. A pulse confined to part of the band can stand out in one of these narrower profiles.

Each profile is smoothed with Gaussians of increasing width, starting at one time bin and doubling up to the configured search range permitted by the observation length. The peak score is the height above the smoothed profile’s median divided by 1.4826 times its median absolute deviation. Local maxima scoring at least 6 enter the candidate list. The strongest is taken first. To keep a second peak, its time must differ from each retained peak by more than the larger of three bins and twice the larger smoothing width of the pair. The search keeps up to three regions and drops candidates weaker than 12% of the strongest retained peak. If no region qualifies, a rectangular time–frequency template with a score of at least 5 can supply a starting guess. [Discovery and seed construction](../src/burstsurveyor/science/fit.py)

## From a starting guess to a fitted model

`region_parts` converts each region into a starting arrival time, width, fluence and frequency spectrum. Each component has a Gaussian time profile and a smooth fluence spectrum. With $x=\ln(\nu/600\,\mathrm{MHz})$, its spectrum is $F(\nu)=F_{600}\exp(\gamma x+r x^2)$, where $\gamma$ is the spectral index and $r$ the spectral running. A peak spanning most of the band starts with a flat spectrum. For a narrower peak, the initial curvature and slope concentrate emission near the band where it was found. Both parameters are then free to change.

The components share a residual dispersion measure. A scattered trial also fits one shared exponential scattering time at 600 MHz, scaling as frequency to the power −4. For any proposed parameters, `pulse` calculates the summed intensity integrated over each time bin. At fixed component shapes the optimizer can solve for nonnegative strengths. It uses bounded, damped Gauss–Newton updates to fit the shapes jointly. A solve can end when the projected gradient or loss improvement is small, when a step fails to improve the fit, or when its iteration or work allowance is exhausted. The trial records its termination reason. [Pulse model](../src/burstsurveyor/science/model.py) and [optimizer](../src/burstsurveyor/science/engine.py)

## Choosing a model and deciding when to stop

The first trials use one Gaussian per starting region, then repeat with scattering. If there is more than one region, the search also tries a single scattered pulse at the strongest region. Every trial is scored on the same usable pixels of the original grid. Its gain $G$ is the decrease in channel-noise-weighted squared error relative to a constant baseline per channel.

For $N$ components and scattering flag $s$ (0 or 1), the initial ranking score is $U=G-4(5N+s)$. Let $G_b$ be the gain of the trial with the highest $U$. Among trials within $\max(4,0.015\max(0,G_b))$ of that best ranking score, the code selects the one with the fewest parameters, breaking ties by gain. The shared residual DM is present in every trial. These are reconstruction criteria. The default search assigns no F-test p-value to adding scattering or a component. [Scoring and initial selection](../src/burstsurveyor/science/objective.py)

A residual can suggest a model the initial search missed. The code subtracts the current prediction and sums residual intensity in rectangular time–frequency windows of several sizes. In the first stage, a positive residual-window score of at least 6 can seed one more component if the current count is below four. The components are refitted together. A refinement on the original grid then tests the preferred candidate.

Bounded refinement can also propose changes to a single-component fit or a fit with a strong residual, and test frequency-dependent structure that could justify splitting a component. A replacement must preserve or improve gain under both the primary and alternate RMS weights. Among eligible candidates, the simplest within $\max(0.02,0.001\max(0,G_{\rm best}))$ of the best gain is preferred. The search ends after its scheduled trials, subject to fixed work budgets and a ceiling of seven components. The final residual scan records remaining structure. A selected fit can still have a visible residual. [Refinement](../src/burstsurveyor/science/bounded.py), [spectral splits](../src/burstsurveyor/science/spectral_split.py) and [search budgets](../src/burstsurveyor/science/config.json)

## Worked example with two pulses

Run the [quickstart](../README.md#fit-an-observation) to generate the observation and write a fit to `run`. The generator puts pulses at 24.5 and 36.5 ms. The fitter receives intensity and grid metadata. [Example generator](../src/burstsurveyor/example.py)

| Function | Input | Result for this observation |
|---|---|---|
| `load_raw` | Two-pulse NPZ | 64 channels by 144 time samples, spaced by 6.25 MHz and 0.5 ms |
| `ReconstructionData` | Raw pixels and grid | 64 usable channels, 110 off-burst bins, 22 frequency groups for initial fitting |
| `find_regions` | Seven profiles searched across smoothing widths | Bin 49 in the full band scores 62.52. Bin 73 in the lower half scores 28.37. Both have a smoothing width of two bins. |
| `region_parts` | Two peak regions | Two five-parameter component seeds with initial widths of 1 ms. A shared dispersion offset is initialized to zero. |
| `optimizer` and `score` | Each candidate and its starting parameters | Fitted parameters and original-grid scores for the trials below |

| Trial in the saved record | Components | Scattering | Gain |
|---|---:|---|---:|
| `profile` | 2 | Zero | 7085.52 |
| `profile_tail` | 2 | Fitted | 7085.45 |
| `brightest_tail` | 1 | Fitted | 5417.72 |
| `native_profile` | 2 | Zero | 7085.71 |

The first three rows are competing explanations. Two Gaussians recover substantially more intensity than the single scattered pulse, and adding scattering to the two gives no improvement. The largest positive and negative residual-window scores after the initial selection have magnitudes 3.70 and 5.21. Both are below 6. The fourth row refines the two Gaussians on the original grid. The spectral check finds no evidence for a split. Both pulses in this example were found before the first fit. [Compare the observation, models and residuals](https://immortalari.github.io/burst-surveyor/two-pulse.html)

For export, Surveyor holds the chosen pulse shapes fixed and tests one nonnegative amplitude adjustment. It adopts the adjustment only if both noise-weighted gains improve by more than the larger of one gain unit and 0.1% of the corresponding starting gain, and the recomputed scores pass the export checks. The default exporter integrates each component over the observed bins. The optional FitBurst exporter evaluates the same geometry with FitBurst. [Exporter comparison](validation.md#exporter-agreement)

## Inspect the decision for your own run

In `run/results.json`, each observation has a status. For an exported fit, `primary_receipt` points to the starting regions, trials and selection. `native_receipt` identifies the exporter and records the checks on the saved pixels, and `prediction` points to the saved NPZ. Run this Python snippet from the directory containing the quickstart’s `run` folder.

```python
import json
from pathlib import Path

run = Path("run")
event = json.loads((run / "results.json").read_text())["events"][0]
primary = json.loads((run / event["primary_receipt"]).read_text())
native = json.loads((run / event["native_receipt"]).read_text())

print(primary["regions"])  # score, time bin, smoothing width, first and last channel
for trial in primary["trials"]:
    print(trial["name"], trial["n"], trial["scatter"], round(trial["gain"], 2))
print(primary["selected"]["name"])
print([round(t * 1000, 3) for t in native["fitburst_parameters"]["arrival_time"]])
```

For the example, the selected trial is `native_profile`. The exported arrival times at 600 MHz are **24.487 and 36.579 ms**, with Gaussian widths of **1.335 and 1.621 ms** and zero scattering. The exported modeled intensity is `prediction + baseline[:, None]`. Its gain, computed from the stored pixels, is 7085.71. `burstsurveyor verify run` checks the integrity of the saved files. The residuals show which features of the observation the model captures. [Export implementation](../src/burstsurveyor/science/native.py) and [output fields](reference.md#export)
