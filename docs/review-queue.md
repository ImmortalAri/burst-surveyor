# Review queue

Alerts target likely missing components. Other diagnostics are informational.

| Status | Meaning |
|---|---|
| Unavailable | A fit, figure or diagnostic could not be produced. |
| Rescue | A resolved observed peak lacks a nearby modeled peak and leaves positive residuals in both channel folds. Inspect for omitted emission or interference. |
| Routine | No missing-component alert found. Inspect the figure before recording a decision. |

The screen checks multiple frequency bands and smoothing scales, allowing for off-burst noise and channel correlations. These are conditional screening scores, not calibrated significances. Faint, narrowband or unusual structure can be missed. The implementation defines the [peak-coverage](../src/burstsurveyor/science/pulse_coverage.py) and [alert](../src/burstsurveyor/triage.py) thresholds.

Spectral mismatch, negative or weak residuals, broad or sub-bin components, parameter bounds, noise concerns and fitting budgets appear in a collapsed information panel. They can affect interpretation without requiring another component. A visually adequate reconstruction can be retained with these limitations; a persistent missing-component alert requires review.

`review/review.html`, `queue.json` and `queue.csv` include every event and its latest fit, including rejected rescues. Explicit inspection decisions are separate from automatic priority and linked to input and packet hashes.

## Record an inspection

For a prepared run, open `WORK/review/review.html` and inspect each figure and component table. Write the decisions in a JSON object or array. Copy `id`, `source_sha256` and the current `packet_sha256` from `WORK/review/queue.json`. Add an `observation` describing the evidence and an `action` for each event.

| Action | Use |
|---|---|
| `retain` | The reconstruction covers the observed pulse structure. |
| `review` | Structure remains ambiguous, a missing-component alert persists, or the figure is unavailable. |
| `unresolved` | No model was exported. Record why. |
| `challenge` | A distinct positive temporal residual spans neighboring channels. Supply its zero-based, half-open `time_bins` and `channel_bins` intervals on the original grid. |

```sh
python -m burstsurveyor decide WORK INPUTS decisions.json
```

Bounded mode permits one challenge per event, including an automatic follow-up. After a challenge, inspect the result and submit a final decision using the returned packet hash. Once every event has a final decision, write `WORK/report.csv` and `WORK/report.json`.

```sh
python -m burstsurveyor finish WORK
```

## Existing runs

Diagnose a sealed numerical run without refitting:

```sh
python -m burstsurveyor triage RUN INPUTS OUTPUT
python -m burstsurveyor triage-verify OUTPUT --run RUN
```

Resume with identical sources and `--resume` to retry resource-limited inspections without overwriting initial evidence. Hash verification checks integrity, not scientific correctness. The HTML requires no server or external assets.
