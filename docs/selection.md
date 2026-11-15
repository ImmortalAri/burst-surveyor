# Selection modes

`prepare INPUTS WORK --workers 1` uses `--mode bounded` by default. It permits at most one temporal rescue per event when the residual evidence justifies it. All modes preserve rejected trials and enforce resource guards.

## Pulsar profiles

Use `--mode pulsar` for one complete folded rotation. Acquisition metadata must contain a positive `period_s` matching `data_full.shape[1] * res_time` within relative tolerance 1e-9. Cropped pulses and multiple rotations are not supported by this mode.

The model sums bin-integrated Gaussian or scattered profiles across rotations, including tails from earlier pulses. Discovery, residual windows, peak checks and component merging use circular coordinates. An internal phase origin makes initialization and noise screening consistent under integer rotations without cropping or interpolation. Exported pixels, arrival times and trial records return to the original phase. Native export independently checks the reconstruction using translated upstream FitBurst profiles.

The search compares zero and nonzero scattering at proposed component counts, starting with the current model's count. Merging nearby terms is allowed only when the reconstruction and peak-coverage checks pass. Scattering is never forced to be positive.

A coarse nuisance-parameter profile reports scattering as conditionally constrained, a conditional upper limit, or ambiguous. It uses a fixed component count and estimated noise; its nominal intervals are not calibrated physical confidence intervals. Incomplete searches remain ambiguous. A zero-scattering selection does not establish physical absence.

One scattering timescale is shared across components, with frequency exponent fixed at -4. Intrinsic shapes are Gaussian, with no independent rise-time parameter. Component count and time resolution affect widths, arrival times, DM and scattering; broad terms can absorb baseline variation. Inspect the reconstruction and residuals before interpreting those parameters.

## Experimental spectral models

`--mode combined` compares smooth spectra with profiled nonnegative channel amplitudes. It fits geometry and amplitudes before held-time evaluation, then chooses a simpler eligible model when predictive support permits. Simplification cannot discard replicated observed peaks or materially worsen replicated positive residuals. Native-grid acceptance is still required.

`--mode scintillation` uses flexible spectral amplitudes and predictive temporal selection without the combined peak-coverage search. Fewer terms can reflect spectral flexibility rather than fewer physical pulses. Held-time checks are conditional on full-data screening, not independent experiments.

## Follow-ups and interrupted runs

Pulsar and experimental spectral modes use internal searches and do not support external `challenge` requests. Leave persistent ambiguity marked `review`; do not repeatedly add components until residuals become quiet.

If memory admission fails in pulsar mode, the run remains incomplete and resumable. Retry the identical command with `--resume` once RAM is available. Completed fits are reused; do not relax the memory guard.
