# Selection modes

`prepare --mode bounded` is the stable default. Its native reconstruction, resource guards and fitting thresholds are preserved. One external temporal challenge per event may be attempted when the residual evidence justifies it. Rejected trials remain available for inspection.

`prepare --mode combined` is an experimental comparison of smooth spectra and profiled, nonnegative spectral amplitudes at bounded channel resolutions. Geometry and amplitudes are fitted before held-time evaluation. Predictive parsimony chooses among eligible candidates; a smaller model cannot win when it fails replicated peak coverage. Backward simplification is bounded and vetoed when it introduces a materially worse replicated positive residual. Final native-grid acceptance remains required. All candidates, rejection reasons and stopping conditions are retained in receipts.

`prepare --mode scintillation` uses flexible spectral amplitudes and predictive temporal selection without the combined coverage search. Its component count can be smaller when spectral flexibility accounts for structure, but this alone does not establish a better physical model.

Both alternatives use the same observations, masks, noise estimates, FitBurst pulse convention and resource guards. Held-time checks are conditional on full-data screening; they are not independent experiments. Flexible spectra, intrinsic width, residual DM and scattering can be degenerate. The time grid and crop can limit identifiability. Neither mode guarantees every pulse is recovered or every selected component is physical.

Alternative modes have internal bounded searches and do not support the legacy external `challenge` workflow. A request is rejected before it consumes an allowance. Inspect residual warnings and leave persistent ambiguity `review`; a separate manual refit can test another explanation. Do not repeatedly add components until the residual becomes quiet.

## Pulsar profiles

`prepare INPUTS WORK --workers 1 --mode pulsar` adds an opt-in paired search to
the default bounded search. It attempts null and scattering-enabled fits at
each proposed count, starting with the incumbent count and using the same
raw-derived objective. Nearby terms can
be merged as proposals; existing peak-coverage and residual guards veto lost
structure. Scattering is never forced to be positive. All trials are recorded.

The native export checks actual stored-pixel peak coverage. A bounded nuisance
profile labels scattering conditionally constrained, a conditional upper limit,
or ambiguous. These nominal intervals use a fixed count and plug-in noise;
they are not calibrated physical confidence. Incomplete searches remain
ambiguous. Pulsar mode requires a positive `period_s` in the acquisition metadata
matching exactly one sampled rotation (`number of time bins * res_time`, within
relative tolerance 1e-9). It sums bin-integrated periodic Gaussian or scattered
profiles, including tails from earlier rotations. Native export independently
sums translated upstream FitBurst profiles and checks agreement. The scattering
index remains fixed at -4. The default FRB mode is unchanged.

Discovery, residual windows, peak-coverage checks and component merging use
circular coordinates. A raw-derived internal phase origin makes initialization
and noise screening reproducible under integer phase rotations. This only
permutes the samples; it does not crop or interpolate them. Arrival times,
rejected candidates and exported pixels are restored to the original phase.

The resource-bounded search can still be incomplete. Such scattering profiles
remain ambiguous rather than becoming detections or upper limits. Retain a
reconstruction only after inspecting its observed structure and residuals.
A zero-scattering selection is not evidence that propagation scattering is
physically absent. Multiple terms can describe one pulse shape, and broad terms
can absorb a varying baseline. The fixed scattering law and intrinsic pulse
shape can trade off with DM and width; phase stability does not remove these
physical ambiguities or calibrate parameter uncertainties.

The fitted scattering timescale is shared across components. Intrinsic shapes
are Gaussian, with no independent rise-time parameter. Component count and time
resolution can therefore change inferred widths, arrival times, DM and
scattering. Human fits are useful comparisons, not definitive ground truth.
Reliable scattering detection requires more than agreement between two fits.

If memory admission fails, pulsar mode leaves an incomplete, resumable run;
retry the identical command with `--resume` once RAM is available. Completed
fits are reused and the existing memory guard is never relaxed.
