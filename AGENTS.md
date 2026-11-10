# Fit a batch

Use a fresh 64-bit Python 3.12 environment: it is the tested CI runtime; 3.13 is allowed by packaging but unverified. `INPUTS` is one [NPZ](docs/reference.md) or a directory. Supply an unused path for `WORK`. Keep raw NPZ files on this machine; do not upload or attach them to chats. Inspect generated figures with local file tools.

```sh
python -m pip install -c constraints.txt ./vendor/fitburst-0.1.0-py3-none-any.whl .
python -m burstsurveyor doctor
python -m burstsurveyor prepare INPUTS WORK
```

Resume with the same arguments and `--resume`. `prepare` attempts one guarded rescue when a positive temporal residual repeats across channel splits. Open the HTML path returned in `review_queue` (`WORK/review/review.html`); work from highest priority down, using the latest packet hash in `review/queue.json`. The original index retains initial packet hashes. Inspect accepted and rejected follow-ups, then every remaining event. Open every selected packet and figure locally. Compare profiles and inspect residuals across channels. Record the time, frequency range and sign of coherent features.

Priority is an inspection order, not an error probability or an automatic decision. A tentative positive window may be faint emission, interference or a shape error. A negative window can mean a filled trough, incorrect width or scattering; adding a pulse may not fix it. Do not equate component count with physical pulse count. Parameter warnings identify resolution/crop sensitivity, not measured instability. Routine events still need inspection. An unavailable figure or diagnostic must never receive `retain`.

The default `--mode bounded` preserves the stable fitter. `--mode combined` and `--mode scintillation` are experimental alternatives with internal bounded searches; automatic/manual external challenges are unsupported there. Leave persistent issues `review` and propose a separate manual refit. Do not consume a challenge by repeatedly resubmitting a warning.

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

For an already sealed numerical run, `python -m burstsurveyor triage RUN INPUTS NEW_OUTPUT` generates a local ranked queue without fitting. It is a diagnostic audit, not a completed human assessment. Never upload raw observations, reference fits, local reports, private paths or experiment history into the public repository or its documentation.
