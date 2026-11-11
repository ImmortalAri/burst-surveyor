# Fit and review a batch

This is the complete default workflow. Consult linked docs only when an input, diagnostic or optional mode needs explanation.

Use a fresh 64-bit Python 3.12 environment. `INPUTS` is one [NPZ](docs/reference.md) or a directory; `WORK` is an unused path. Keep observations and reports local and respect the resource guards.

```sh
python -m pip install -c constraints.txt ./vendor/fitburst-0.1.0-py3-none-any.whl .
python -m burstsurveyor doctor
python -m burstsurveyor prepare INPUTS WORK --workers 1
```

Resume interrupted preparation with identical arguments and `--resume`. Keep the default bounded mode unless another mode was requested. `prepare` can attempt one guarded rescue for a likely missing pulse.

## Inspect

Open `WORK/review/review.html` and work from highest priority down. For every event, open the current packet, selected figure and component table. Inspect the observed spectrum, fitted reconstruction, residuals and pulse profile, including routine events and accepted or rejected follow-ups. Record the time, frequency range and sign of coherent residual features.

Use the latest packet hash in `review/queue.json`; the original index retains initial hashes. An unavailable export, figure or diagnostic cannot receive `retain`. A persistent missing-component alert requires `review`. Spectral mismatch, broad or sub-bin components, crop sensitivity and parameter limits are informational and do not by themselves require another component. Overlapping model terms are not necessarily separate physical pulses, and a negative residual is not evidence for another pulse.

## Decide and finish

Write `decisions.json` as an object or array. Each entry contains `id`, `source_sha256`, `packet_sha256`, `action` and an `observation` describing the inspection evidence.

| Action | Use |
|---|---|
| `retain` | The reconstruction covers the observed pulse structure. |
| `review` | Structure remains ambiguous or an inspection is unavailable. |
| `unresolved` | No model was exported; record why. |
| `challenge` | A distinct positive temporal residual spans neighboring channels. |

A challenge also needs `time_bins` and `channel_bins` as zero-based `[start, stop)` intervals on the original grid; account for the sign of `res_freq`. At most one challenge is allowed per event, including automatic rescue. Spectral mismatch alone does not justify one.

```sh
python -m burstsurveyor decide WORK INPUTS decisions.json
```

After a challenge, inspect its result even if rejected and submit a terminal action with the returned packet hash. Leave persistent problems `review` rather than repeating challenges. When every event has a terminal decision:

```sh
python -m burstsurveyor finish WORK
```

Return verified `report.csv`, `report.json`, decisions and figures. Count exports, challenges, unresolved or unavailable cases, and missed inspections. Never publish raw observations, reference fits, local reports, private paths or experiment history without explicit user authorization.

For an existing sealed run, see [triage](docs/review-queue.md). Experimental [selection modes](docs/selection.md) have internal searches and do not support external challenges.
