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

## Channel timing

The default bounded burst fit uses a shared time axis unless the NPZ includes
`channel_time_offset_s`, a floating-point array of shape `(num_freq,)`.
Supply one measured, finite offset in seconds per frequency channel, after
removing the dispersion delay already applied to the data. Sample `j` in
channel `i` is evaluated at `j * res_time + channel_time_offset_s[i]` relative
to the nominal time origin. A positive offset means the channel was sampled
later. This representation assumes a constant offset per channel and a common
sample interval. [CHIME/FRB Catalog 2 timing convention](https://arxiv.org/html/2601.09399#A3)

Peak discovery uses the observed time bins. The coarse search averages the model
over four noise-weighted offset quantiles per frequency group. Native fitting,
residual scoring and FitBurst export use each channel's offset. The prediction
NPZ includes the offsets alongside the nominal `time_s` axis.

## Export

Modeled intensity is `prediction + baseline[:, None]` on the original pixels. The NPZ includes channel and off-burst masks, channel variances, frequency centres in MHz and `time_s` relative to the first sample.

Pulsar search uses an internal circular phase origin, recorded as `phase_origin_bins`; exported pixels and fitted arrival times are restored to the original input phase. Internal review intervals crossing phase zero can extend beyond the last bin and are interpreted modulo the number of time bins. Intrinsic widths are Gaussian sigma; scattering timescales refer to 600 MHz.

See [scores and interpretation](guide.md#scores-and-interpretation) for the meaning and limits of exported scores and component counts.

## Bounded fit reproducibility

The default FRB fitter uses candidate, iteration and work limits. The initial
search allows 600 objective evaluations or 250 million estimated native work
units. Each refinement hook has its own allowance of 300 evaluations or 120
million native work units. The optional temporal retry allows ten iterations,
160 evaluations or 120 million native work units. The fitter checks work limits
between numerical calls. Native work weights each evaluation by the number of
pixels and, for derivative evaluations, the number of model components.

Batch workers have a ten-minute emergency watchdog per event and stage. A
timed-out initial fit or export is recorded as failed and the batch continues.
A timed-out retry records the failure alongside the preserved baseline.
Schedule and CPU speed affect completion time. Compare the exported
reconstruction and model parameters with numerical tolerances across platforms.
