# Burst Surveyor

Burst Surveyor fits dedispersed radio bursts with FitBurst. Give a coding agent this package and a folder of NPZ files, and ask it to follow [AGENTS.md](AGENTS.md). It will run the fits, inspect the residuals and handle follow-up fits where needed, then return `report.csv`, `report.json` and the figures for review.

The main check is to look at the fitted pulse profile and residuals, first with the coding agent and then yourself. Each decision in `decisions.json` includes `source_sha256` and `packet_sha256`, which link it to the exact input, fit and figure that were inspected.

The reported `gain` measures how much the fit reduces the weighted squared error compared with a constant baseline in each channel. `rms_gain` uses alternate RMS weights. Both are measured on the fitted data, so a higher gain alone does not justify adding another component.

**FRB 20190412A.** This five-component fit describes three overlapping peaks and a tail. Its gain is 9.7% higher than an archived two-component FitBurst fit produced by a coding agent, using the same pixels and weights for the comparison. Some frequency structure remains in the residuals. See the [observation](https://chime-frb.ca/catalogs/basecat1/FRB20190412A) and [catalog paper](https://arxiv.org/abs/2311.00111).

![Observed burst, reconstruction, residuals and fitted profile](examples/frb20190412a.png)

**Seven synthetic pulses.** The initial fit missed the pulse at 135 ms. The residual check prompted a follow-up fit that recovered it and passed the numerical acceptance checks. The [NPZ file](examples/seven-pulse.npz) is included.

![Seven-pulse reconstruction and residuals](examples/seven-pulse.png)

**Two synthetic pulses.** Both pulses were recovered in the initial fit, so no follow-up was needed. The [NPZ file](examples/two-pulse.npz) is included.

![Two-pulse reconstruction and residuals](examples/two-pulse.png)
