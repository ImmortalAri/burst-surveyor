# Review priority

`prepare` writes `review/review.html`, `review/queue.json` and `review/queue.csv` before terminal assessments are required. The queue includes every input. It uses the current selected packet after a rescue, even when the trial was rejected. Decisions remain explicit and bound to packet hashes; automatic priority never changes a human assessment.

| Priority | Meaning | Inspection |
|---|---|---|
| 0 · unavailable | No export, missing native diagnostics or unavailable figure | Inspect the failure and observation; retry when resources permit or refit separately. |
| 1 · rescue | A positive temporal residual passes the existing discovery/check screen | Inspect the native time-frequency box; consider the one permitted rescue. |
| 2 · inspect | A tentative positive/negative window repeats across channel folds, or noise/mask/coverage checks are limited | Check faint emission, filled troughs, interference and shape errors. A warning does not authorize an extra component. |
| 3 · parameters | Width below a native bin, broad/crop-sensitive component, convergence limit or spectral model limitation | Review DM, scattering, width, crop and model dependence before physical interpretation. |
| 4 · spectral | Frequency-amplitude structure or an isolated residual outlier | Inspect the frequency spectrum and interference; avoid using temporal components to absorb spectral mismatch. |
| 5 · routine | No screening concern identified | Inspect all panels and parameters; the screen can miss weak or unusual structure. |

Within a priority, possible separate emission is inspected before mismatches within the fitted pulse, then the weaker channel-fold residual score orders events, followed by event ID for reproducibility. A positive window outside fitted emission means its band-averaged fitted signal is below 10% of that band's peak; it can still be interference or a shape error. These searched scores are not probabilities or calibrated detection significances. Alternate channels can be correlated; screening, masks and the existing fit use the observation. No test here establishes physical component count.

The tentative screen uses per-channel amplitude projection of the existing temporal prediction, full/half/quarter channel bands, power-of-two time windows up to one eighth of the record, and both residual signs. A window must score at least 2.5 in each fold after robust scaling and frequency-correlation inflation. This threshold changes review sensitivity only: fitting, acceptance thresholds, pixels and exported parameters remain untouched. Fewer than six usable channels in either fold makes that band unavailable to this screen. DM drift, very narrow frequency structure and faint correlated signals can escape detection.

Component geometry warns when Gaussian sigma is below one sample, at least one quarter of the record, or when the center plus two sigma and three reference scattering times reaches the crop. This is a conservative heuristic, not an uncertainty interval. A valid narrow or truncated burst can trigger it. DM/scattering/width stability requires separate sensitivity or uncertainty analysis.

The HTML is local and works without a server or external assets. It filters by event name and priority and links the observed profile, model and residual figure with the component table. JSON/CSV/HTML hashes are recorded together. An existing sealed numerical run can be audited with `triage RUN INPUTS OUTPUT`; `--resume` verifies the same run, sources, inputs and saved evidence and can recover resource-limited inspections without overwriting the initial evidence. This does not refit or create terminal decisions. Verify the queue and its bound evidence with `triage-verify OUTPUT --run RUN`. Hash verification checks integrity, not scientific correctness.
