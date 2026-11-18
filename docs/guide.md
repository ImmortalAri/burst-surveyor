# Using burst surveyor

## Installation

Use a fresh 64-bit Python 3.12 environment. From the repository root, install the bundled FitBurst wheel and burst surveyor, then check the numerical runtime.

```sh
python -m pip install -c constraints.txt ./vendor/fitburst-0.1.0-py3-none-any.whl .
python -m burstsurveyor doctor
```

## Fit an observation

```sh
python -m burstsurveyor fit examples/two-pulse.npz run --workers 1
python -m burstsurveyor verify run
```

The default mode fits dedispersed bursts without periodic wrapping. Supply one [NPZ file](reference.md) or a directory of NPZ files and a new output directory. To resume an interrupted run, repeat the fit command with the same inputs, mode and worker count and add `--resume`.

`run/results.json` contains one entry per observation. For an exported fit, the entry gives the component count, scattering model, fit scores and path to the prediction NPZ. Modeled intensity is `prediction + baseline[:, None]` on the original pixels. An entry without an export records the status and reasons. The [reference](reference.md#export) describes the arrays and units.

For the bundled example, the two-component reconstruction is shown in the [README](../README.md#fit-an-observation). To plot the exported fit from this run, use the event ID from `results.json`.

```sh
python -m burstsurveyor plot run --event 179014e0591b80d892ece0ac --input examples/two-pulse.npz --out two-pulse-fit.png
```

## Folded pulsar profiles

For a profile covering one complete rotation, include the period in seconds as `period_s` in the acquisition metadata and select [pulsar mode](selection.md#pulsar-profiles).

```sh
python -m burstsurveyor fit folded-profile.npz pulsar-run --mode pulsar --workers 1
```

## Figures and residuals

To generate figures and screen the residuals as part of a fit, use `prepare` with a new working directory.

```sh
python -m burstsurveyor prepare examples/two-pulse.npz review-run --workers 1
```

Open `review-run/review/review.html` to inspect the observed spectrum, pulse profile, fitted reconstruction and residuals. In bounded mode, `prepare` can test one additional component per event when a peak is missing from the fit and leaves positive residuals in both channel folds. The queue shows the selected reconstruction after the follow-up. The initial fit is in `review-run/fit`. Resume interrupted preparation with the same arguments and `--resume`.

For a completed `fit` run, [triage](review-queue.md#existing-runs) generates a review queue without refitting. To record inspection decisions and produce `report.csv` and `report.json` for a prepared run, follow the [review procedure](review-queue.md#record-an-inspection). Observations and generated reports stay local.

## Scores and interpretation

`gain` is the reduction in weighted squared error relative to a constant baseline per channel. `rms_gain` uses alternate RMS weights. A bounded rescue must pass a separate channel check and improve both scores on the exported pixels. A frequency-amplitude mismatch alone does not justify another pulse.

Components can overlap, and their count is not a count of physical pulses. Width, residual DM, scattering and intrinsic shape can trade off, so retaining a reconstruction does not establish unique physical parameters or calibrated uncertainties. Human fits are useful comparisons, not definitive ground truth.

## Further details

- [Input format and exported arrays](reference.md)
- [Review priorities and triage of existing runs](review-queue.md)
- [Pulsar and experimental selection modes](selection.md)
