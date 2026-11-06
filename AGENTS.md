# Fit a batch

From the checkout, use a fresh 64-bit Python 3.12 environment. `INPUTS` is one [NPZ](docs/reference.md) or a directory. Supply an unused path for `WORK`. Keep observations local.

```sh
python -m pip install -c constraints.txt ./vendor/fitburst-0.1.0-py3-none-any.whl .
python -m burstsurveyor doctor
python -m burstsurveyor prepare INPUTS WORK
```

Resume an interrupted run with the same arguments and `--resume`. Read the index returned by `prepare`. Open each packet and figure relative to `WORK`. Compare the pulse profiles, then inspect the residual across channels. Record the time, frequency range and sign of any coherent feature.

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

One challenge per event. Inspect the follow-up packet and figure returned by `decide`, including rejected trials. Submit a terminal action with the **returned** `packet_sha256`. The index retains the initial hash.

Finish when every event has a terminal decision.

```sh
python -m burstsurveyor finish WORK
```

Return the verified `report.csv`, `report.json` and figures. Count exports and unavailable fits. Identify challenges, persistent residuals and any missed inspections.
