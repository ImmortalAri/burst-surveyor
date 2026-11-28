# NPZ format

`data_full` holds dedispersed intensity indexed by frequency channel and time bin, with at least 21 of each. All channels have the same sample interval. The NPZ and the expanded `data_full.npy` member each have a 160 MiB limit.

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

The default `bounded` mode accepts an optional top-level NPZ array,
`channel_time_offset_s`, of floating-point values with shape `(num_freq,)`.
Omit this array for a shared time axis. The `pulsar`, `combined` and
`scintillation` modes require this shared-grid input.
Supply one measured, finite offset in seconds per frequency channel, after
removing the dispersion delay already applied to the data. Sample `j` in
channel `i` is evaluated at `j * res_time + channel_time_offset_s[i]` relative
to the nominal time origin. A positive offset means the channel was sampled
later. This representation assumes a constant offset per channel and a common
sample interval. [CHIME/FRB Catalog 2 timing convention](https://arxiv.org/html/2601.09399#A3)

Peak discovery uses the observed time bins. The coarse search averages the model
over four noise-weighted offset quantiles per frequency group. Native fitting,
residual scoring and both bounded exporters use each channel's offset. The prediction
NPZ includes the offsets alongside the nominal `time_s` axis.

## Export

In bounded mode, `fit` and `prepare` export the internal, bin-integrated model by default. To render with the pinned FitBurst implementation, install the [optional dependency](guide.md#installation) and pass `--exporter fitburst`. The pulsar, scintillation and combined modes use FitBurst. An interrupted run resumes with its recorded exporter.

The run summary records `exporter`. The native event receipt identifies the renderer and its version, and the prediction NPZ records the export format in `version`. Both exporters populate `fitburst_parameters` using the same physical parameter conventions. Historical runs without an exporter field used FitBurst. To reproduce a published result exactly, use the original software version, numerical environment and input.

Modeled intensity is `prediction + baseline[:, None]` on the original pixels. The NPZ includes channel and off-burst masks, channel variances, frequency centres in MHz and the nominal time axis `time_s` in seconds, starting at zero.

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
