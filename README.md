# Burst Surveyor

Burst Surveyor fits dedispersed radio bursts with FitBurst and ranks the results for human inspection. Give a coding agent this package and a folder of [NPZ files](docs/reference.md), and ask it to follow [AGENTS.md](AGENTS.md). Observations and generated reports stay local.

Use a fresh Python 3.12 environment:

```sh
python -m pip install -c constraints.txt ./vendor/fitburst-0.1.0-py3-none-any.whl .
python -m burstsurveyor doctor
python -m burstsurveyor prepare INPUTS WORK --workers 1
```

Open `WORK/review/review.html`. The queue puts unavailable fits or inspections first, followed by possible missed-component alerts. Other diagnostics appear in an information panel. Each warning explains what to inspect and suggests an action. The profile, observed spectrum, reconstruction, residuals and component parameters are available together. The queue uses the selected follow-up when a rescue has been attempted, including a rejected rescue.

The pipeline attempts at most one guarded temporal rescue per event in the default mode. Persistent problems remain flagged for review or a separate manual refit. An alert requires a resolved observed peak that repeats across channel folds, lacks a nearby modeled peak, and leaves a repeated positive residual after channel-amplitude projection. A weaker residual screen supplies information without triggering an alert. Spectral mismatch alone does not request another pulse. Routine events still require visual review: the queue is an inspection aid, not an assurance that every pulse was found.

Record explicit assessments using `decide`, then run `finish` to produce verified `report.csv` and `report.json`. Input and packet hashes bind each decision to the exact observation, fit and figure. See [the review workflow](AGENTS.md) and [priority definitions](docs/review-queue.md).

For an existing sealed numerical run, generate a queue without repeating fits:

```sh
python -m burstsurveyor triage RUN INPUTS NEW_OUTPUT
```

Resume interrupted work with the same command and `--resume`. Resource guards remain active; unavailable inspections are prioritized rather than silently treated as clean.

The reported `gain` measures the reduction in weighted squared error relative to a constant baseline per channel. `rms_gain` uses alternate RMS weights. A higher gain alone does not justify another component, and component count need not equal visible peak count. Widths below the time resolution, broad components and crop-sensitive fits receive informational notes that do not force a review decision. Residual DM, scattering and intrinsic widths can trade off; these warnings do not measure parameter stability or replace uncertainty analysis.

The stable default is `--mode bounded`. The opt-in `--mode combined` selector compares smooth and flexible spectral amplitudes with predictive parsimony, replicated pulse coverage and a guard against simplifications that worsen residual islands. `--mode scintillation` offers flexible spectral amplitudes without the combined coverage search. Both alternatives are experimental: they can change DM, scattering and widths, use conditional validation, and do not guarantee physical components. They use internal bounded searches; legacy external challenges are available only in bounded mode. See [selection modes](docs/selection.md).

**Seven synthetic pulses.** A local test input for inspecting pulse coverage and the guarded rescue workflow. The [NPZ file](examples/seven-pulse.npz) is included.

![Seven-pulse reconstruction and residuals](examples/seven-pulse.png)

**Two synthetic pulses.** A simple control for inspecting a reconstruction. The [NPZ file](examples/two-pulse.npz) is included.

![Two-pulse reconstruction and residuals](examples/two-pulse.png)
