# Selection modes

`prepare --mode bounded` is the stable default. Its native reconstruction, resource guards and fitting thresholds are preserved. One external temporal challenge per event may be attempted when the residual evidence justifies it. Rejected trials remain available for inspection.

`prepare --mode combined` is an experimental comparison of smooth spectra and profiled, nonnegative spectral amplitudes at bounded channel resolutions. Geometry and amplitudes are fitted before held-time evaluation. Predictive parsimony chooses among eligible candidates; a smaller model cannot win when it fails replicated peak coverage. Backward simplification is bounded and vetoed when it introduces a materially worse replicated positive residual. Final native-grid acceptance remains required. All candidates, rejection reasons and stopping conditions are retained in receipts.

`prepare --mode scintillation` uses flexible spectral amplitudes and predictive temporal selection without the combined coverage search. Its component count can be smaller when spectral flexibility accounts for structure, but this alone does not establish a better physical model.

Both alternatives use the same observations, masks, noise estimates, FitBurst pulse convention and resource guards. Held-time checks are conditional on full-data screening; they are not independent experiments. Flexible spectra, intrinsic width, residual DM and scattering can be degenerate. The time grid and crop can limit identifiability. Neither mode guarantees every pulse is recovered or every selected component is physical.

Alternative modes have internal bounded searches and do not support the legacy external `challenge` workflow. A request is rejected before it consumes an allowance. Inspect residual warnings and leave persistent ambiguity `review`; a separate manual refit can test another explanation. Do not repeatedly add components until the residual becomes quiet.
