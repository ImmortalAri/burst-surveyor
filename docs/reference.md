# NPZ format

`data_full` holds dedispersed intensity on a uniform frequency × time grid, at least 21 × 21. The NPZ and the expanded `data_full.npy` member each have a 160 MiB limit.

`metadata` is a scalar JSON string or legacy scalar dictionary (64 KiB limit).

| Field | Meaning |
|---|---|
| `res_time` | Sample interval in seconds, positive |
| `res_freq` | Signed channel width in MHz |
| `freqs_bin0` | First channel edge in MHz. Centre of channel `i` is `freqs_bin0 + (i + 0.5) * res_freq`. |
| `times_bin0` | Acquisition time origin in seconds |
| `is_dedispersed` | `true` |
| `period_s` | Required only for pulsar mode: rotation period in seconds, matching `data_full.shape[1] * res_time` within relative tolerance 1e-9. |

Pulsar mode expects one complete folded rotation. A cropped pulse, multiple consecutive rotations or an unfolded single-pulse observation should not be labeled as a complete rotation. Missing or incompatible period metadata is rejected rather than inferred from the fit.

## Export

Modeled intensity is `prediction + baseline[:, None]` on the original pixels. The NPZ includes channel and off-burst masks, channel variances, frequency centres in MHz and `time_s` relative to the first sample.

Pulsar search uses an internal circular phase origin, recorded as `phase_origin_bins`; exported pixels and fitted arrival times are restored to the original input phase. Internal review intervals crossing phase zero can extend beyond the last bin and are interpreted modulo the number of time bins. Intrinsic widths are Gaussian sigma; scattering timescales refer to 600 MHz.

See [scores and interpretation](guide.md#scores-and-interpretation) for the meaning and limits of exported scores and component counts.
