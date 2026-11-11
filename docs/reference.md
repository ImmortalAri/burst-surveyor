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

## Export

Modeled intensity is `prediction + baseline[:, None]` on the original pixels. The NPZ includes channel and off-burst masks, channel variances, frequency centres in MHz and `time_s` relative to the first sample.

For score definitions, see [scores and interpretation](guide.md#scores-and-interpretation). A follow-up must pass a separate channel check and improve both scores on the exported pixels. Component counts remain provisional.
