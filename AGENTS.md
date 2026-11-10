# Fit a batch

Use a fresh 64-bit Python 3.12 environment: it is the tested CI runtime; 3.13 is allowed by packaging but unverified. `INPUTS` is one [NPZ](docs/reference.md) or a directory. Supply an unused path for `WORK`. Keep raw NPZ files on this machine; do not upload or attach them to chats. Inspect generated figures with local file tools.

```sh
python -m pip install -c constraints.txt ./vendor/fitburst-0.1.0-py3-none-any.whl .
python -m burstsurveyor doctor
python -m burstsurveyor prepare INPUTS WORK
```

Resume with the same arguments and `--resume`. `prepare` attempts one guarded rescue when a positive temporal residual repeats across channel splits. Read its returned follow-ups first, then the index. Open every selected packet and figure relative to `WORK`. Compare profiles and inspect residuals across channels. Record the time, frequency range and sign of coherent features.

Component count is not peak count: overlapping components can describe different frequency bands. Inspect the component table. `broad_edge_component` or `spectral_model_limited` requires `review`. A horizontal off-burst trend is not an extra pulse. The pipeline screens contaminated mask neighbors from off-burst evidence; it never expands every gap. Frequency-amplitude residuals require spectral review, not more temporal components. These diagnostics do not establish physical components.

Write `decisions.json` as an object or array. Copy `id`, `source_sha256` and `packet_sha256` from the index. Add an `observation` stating the evidence and one `action`.

| Action | Use |
|---|---|
| `retain` | Profiles and spectrum support the exported fit. |
| `review` | Ambiguous structure, a model limitation, or a figure you cannot inspect. |
| `unresolved` | No model was exported. Record the reason. |
| `challenge` | A positive residual at a distinct time spans neighboring channels. Add `time_bins` and `channel_bins` as zero-based `[start, stop)` intervals on the original grid. Account for the sign of `res_freq`. |

```sh
python -m burstsurveyor decide WORK INPUTS decisions.json
```

One challenge per event, including automatic rescue. Inspect the follow-up, including rejected trials. Submit a terminal action with its **returned** `packet_sha256`; the index retains the initial hash. Persistent structure after that attempt stays `review`. Do not keep adding components until the residual looks quiet.

Finish when every event has a terminal decision.

```sh
python -m burstsurveyor finish WORK
```

Return the verified `report.csv`, `report.json` and figures. Count exports and unavailable fits. Identify challenges, persistent residuals and any missed inspections.
