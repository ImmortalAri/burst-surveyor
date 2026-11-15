# Review queue

Alerts target likely missing components. Other diagnostics are informational.

| Status | Meaning |
|---|---|
| Unavailable | A fit, figure or diagnostic could not be produced. |
| Rescue | A resolved observed peak lacks a nearby modeled peak and leaves positive residuals in both channel folds. Inspect for omitted emission or interference. |
| Routine | No missing-component alert found. Visual inspection is still required. |

The screen checks multiple frequency bands and smoothing scales, allowing for off-burst noise and channel correlations. These are conditional screening scores, not calibrated significances. Faint, narrowband or unusual structure can be missed. The implementation defines the [peak-coverage](../src/burstsurveyor/science/pulse_coverage.py) and [alert](../src/burstsurveyor/triage.py) thresholds.

Spectral mismatch, negative or weak residuals, broad or sub-bin components, parameter bounds, noise concerns and fitting budgets appear in a collapsed information panel. They can affect interpretation without requiring another component. A visually adequate reconstruction can be retained with these limitations; a persistent missing-component alert requires review.

`review/review.html`, `queue.json` and `queue.csv` include every event and its latest fit, including rejected rescues. Explicit inspection decisions are separate from automatic priority and linked to input and packet hashes.

## Existing runs

Diagnose a sealed numerical run without refitting:

```sh
python -m burstsurveyor triage RUN INPUTS OUTPUT
python -m burstsurveyor triage-verify OUTPUT --run RUN
```

Resume with identical sources and `--resume` to retry resource-limited inspections without overwriting initial evidence. Hash verification checks integrity, not scientific correctness. The HTML requires no server or external assets.
