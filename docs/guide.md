# Using burst surveyor

Give your coding agent the repository, a folder of dedispersed observations and a new working directory, then ask it to follow [AGENTS.md](../AGENTS.md). That file contains the installation commands and full workflow; the agent does not need to read every document or source file before starting.

## Results

The agent fits each event, inspects its spectrum, pulse profile and residuals, and attempts a permitted rescue when a pulse appears to be missing. It then returns the following local outputs.

| Output | Contents |
|---|---|
| `review/review.html` | Ranked figures, component tables and diagnostic notes |
| Figures | Observed spectra, fitted reconstructions, profiles and residuals |
| `decisions.json` | Inspection evidence linked to the input and fit hashes |
| `report.csv`, `report.json` | Verified results and review decisions |

Missing fits and unavailable inspections come first in the queue, followed by possible missed components. Other notes are informational. The agent still inspects routine events and leaves persistent or ambiguous problems marked for review, so you can concentrate on cases that need attention.

## Interpreting a fit

The default bounded mode allows at most one temporal rescue per event. A frequency-amplitude mismatch alone does not request another pulse, and a component count is not a count of physical pulses. Intrinsic width, residual DM, scattering and overlapping components can describe similar reconstructions, so retaining a fit does not establish a unique physical interpretation or reliable parameter uncertainties.

### Scores and interpretation

`gain` is the reduction in weighted squared error relative to a constant baseline per channel. `rms_gain` uses alternate RMS weights. A follow-up must pass a separate channel check and improve both scores on the exported pixels; a higher gain alone does not establish another physical pulse.

## Further details

- [Input format and exported arrays](reference.md)
- [Review priorities and triage of existing runs](review-queue.md)
- [Optional experimental selection modes](selection.md)

If preparation is interrupted, the agent can resume with the same arguments and `--resume`. Observations and generated reports stay local. The README shows real observations and a synthetic rescue example; the bundled [two-pulse](../examples/two-pulse.npz) and [seven-pulse](../examples/seven-pulse.npz) inputs are available as small controls.
