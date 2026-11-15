# Using burst surveyor

Give your coding agent the repository, a folder of dedispersed observations and a new working directory, then ask it to follow [AGENTS.md](../AGENTS.md). That file contains the installation commands and workflow; the agent does not need to read every document or source file before starting.

For folded pulsars, request [pulsar mode](selection.md#pulsar-profiles) and supply one complete rotation with its period in the acquisition metadata. The default mode fits bursts without periodic wrapping.

## Results

The agent fits each event, inspects its spectrum, pulse profile and residuals, and attempts a permitted rescue when a pulse appears to be missing. It returns these local outputs:

| Output | Contents |
|---|---|
| `review/review.html` | Ranked figures, component tables and diagnostic notes |
| `decisions.json` | Inspection evidence linked to the input and fit hashes |
| `report.csv`, `report.json` | Verified results and review decisions |

Missing fits and unavailable inspections come first in the queue, followed by possible missed components. Other notes are informational. The agent still inspects routine events and leaves persistent or ambiguous problems marked for review.

## Scores and interpretation

`gain` is the reduction in weighted squared error relative to a constant baseline per channel. `rms_gain` uses alternate RMS weights. A bounded rescue must pass a separate channel check and improve both scores on the exported pixels. A frequency-amplitude mismatch alone does not justify another pulse.

Components can overlap, and their count is not a count of physical pulses. Width, residual DM, scattering and intrinsic shape can trade off, so retaining a reconstruction does not establish unique physical parameters or calibrated uncertainties. Human fits are useful comparisons, not definitive ground truth.

## Further details

- [Input format and exported arrays](reference.md)
- [Review priorities and triage of existing runs](review-queue.md)
- [Pulsar and experimental selection modes](selection.md)

Resume interrupted preparation with the same arguments and `--resume`. The bundled [two-pulse](../examples/two-pulse.npz) and [seven-pulse](../examples/seven-pulse.npz) inputs are small synthetic controls. Observations and generated reports stay local.
