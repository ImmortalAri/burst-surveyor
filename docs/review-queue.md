# Review queue

Alerts target likely missing components. Other diagnostics are information and do not force review.

| Status | Meaning |
|---|---|
| Unavailable | No export, or a native figure/diagnostic inspection could not be completed. A separate workflow failure. |
| Rescue | A replicated resolved observed peak lacks a nearby modeled peak and leaves repeated positive residuals. Inspect for omitted emission or interference. |
| Routine | No missing-component alert found. Inspect the reconstruction; physical parameter accuracy is not certified. |

The native prominence screen uses full/half/quarter channel bands, multiple smoothing scales, observed peak/trough contrasts, off-burst noise and frequency-correlation inflation. A feature requires pooled prominence at least6 and contrast at least3 in both channel folds. An omission must lose at least half the local contrast and at least6 noise units. A nearby model peak with prominence at least20% of observed contrast within two smoothing scales (minimum two bins) counts as represented. An alert also requires amplitude-profiled positive residual scores at least3 in both folds. These conditional screening scores are not calibrated significances. Narrow-band, faint or unusual structure can be missed.

Spectral mismatch, negative/weak residuals, crop sensitivity, broad/sub-bin components, parameter bounds, noise concerns and fit budgets appear in a collapsed information panel. They can affect parameter interpretation without requiring another temporal component. A visually adequate reconstruction can be retained while recording these limitations. A persistent possible-missing-component alert requires review.

The local review/review.html, queue.json and queue.csv include every event and the latest selected follow-up, including rejected rescues. One external rescue is allowed in bounded mode. Explicit assessments remain separate from automatic priority and bound to input/packet hashes.

An existing sealed numerical run can be diagnosed without refitting: triage RUN INPUTS OUTPUT. Resume with identical sources and --resume; resource-limited inspections can recover without overwriting initial evidence. Verify using triage-verify OUTPUT --run RUN. Hash verification checks integrity, not scientific correctness. HTML uses no external assets or server.
