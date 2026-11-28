# Validation of the default burst fit

## Reconstruction

Burst Surveyor v0.2.0 and v0.1.1 reconstructed the same 4,455 of 4,464 CHIME/FRB Catalog 2 intensity observations. For 4,445 bursts, we scored both fits and the saved catalog model on identical observed pixels. [Catalog data](https://chime-frb.ca/catalogs/catalog2)

| Fitter | ≥90% of catalog whole-window gain | Higher whole-window gain | Lower on-pulse error | Median on-pulse error reduction |
| --- | ---: | ---: | ---: | ---: |
| v0.2.0 | **4,426/4,441 (99.7%)** | 4,283/4,445 (96.4%) | 4,079/4,386 (93.0%) | 0.0115% |
| v0.1.1 | 3,856/4,441 (86.8%) | 2,753/4,445 (61.9%) | 2,404/4,386 (54.8%) | 0.00137% |

Whole-window gain is the fraction of noise-weighted squared error removed relative to a constant intensity per channel. The 90% threshold applies only to positive catalog gains. On-pulse error uses the same weights within the saved v0.1.1 on-pulse mask. The median includes all 4,386 bursts with on-pulse bins.

## Exporter agreement

Starting from the same bounded fit, the internal and FitBurst exporters agreed on the amplitude-adjustment decision for all 316 tested FRBs, including all 12 archived adjustments in the two comparison runs. The 16-event pilot and separate 300-event follow-up deliberately included difficult cases. In the follow-up (110 Catalog 2 and 190 BaseCat 2 events), all FitBurst replays reproduced the archived prediction files byte for byte. All internal exports ran without FitBurst and preserved the fitted geometry and acquisition grid. The median noise-weighted RMS difference between the final model arrays was 0.0054% of the FitBurst model RMS, and the largest was 0.207%. This comparison covers the default smooth-spectrum burst model. [Exporter regression tests](../tests/test_internal_export.py)

## Component counts

Whole-window gains agreed within 0.1 percentage point for 4,350 of 4,445 v0.2.0 fits (97.9%). Of those fits, 3,093 (71.1%) used a different component count from the catalog.

### Blind injections

The original BaseCat 2 noise files provide a shared time grid. For the timing test, we sampled the same 780 BaseCat 2–like pulse designs at the recorded channel times in 130 CHIME/FRB Catalog 2 intensity noise backgrounds. Both v0.2.0 fits used identical 70-bin cutouts at 0.983 ms per bin, retaining every 16th native frequency channel.

| Noise data | Fitter | Fit time grid | Single pulses split | Weak pairs recovered |
| --- | --- | --- | ---: | ---: |
| Catalog 2 intensity | v0.2.0 | Measured per channel | 108/260 (41.5%) | 98/520 (18.8%) |
| Catalog 2 intensity | v0.2.0 | Shared | 112/260 (43.1%) | 100/520 (19.2%) |
| BaseCat 2 | v0.2.0 and v0.1.1 | Shared | 151/260 (58.1%) | 107/520 (20.6%) |

A single pulse is split when the fitter selects at least two components. A pair is recovered when each injected arrival matches a distinct fitted arrival within the smaller of two bins and 45% of the separation. Both releases exported identical predictions for 779 original BaseCat 2 cases and left the same case unresolved.

Measured timing changed the split rate by −1.5 percentage points (95% interval −5.0 to +1.9) and pair recovery by −0.4 points (−2.1 to +1.3), using a paired bootstrap over noise backgrounds. In the injected pulse window, noise-weighted squared error fell in 720 of 770 paired exports (median reduction 0.0526%). [Protocol, per-injection results and scripts](https://raw.githubusercontent.com/ImmortalAri/burst-surveyor/main/docs/data/v0.2.0-timing-injections.zip)

## Scattering

For 922 BaseCat 2 bursts with selected human fits, we held the archived Surveyor component count fixed and refitted each burst with and without a shared scattering tail. At the nominal Catalog 1 F-test threshold (*p* < 0.001), the test agreed with Surveyor’s scattering decision in 816/922 cases (88.5%). Among the 100 bursts where Surveyor and the human fit disagreed, the test agreed with the human decision in 38 cases at Surveyor’s count and 66 when using the human count and starting geometry. Extra pulse components can account for late emission that a simpler fit assigns to scattering. [Catalog 1, §3.3](https://doi.org/10.3847/1538-4365/ac33ab)

## Channel timing

Measured channel offsets were available for 4,457 of 4,464 observations in the v0.2.0 run. Compared with the v0.1.1 fits on a shared time grid, v0.2.0 had higher whole-window gain in 3,732 of 4,455 fits (83.8%). On-pulse error fell in 3,679 of 4,393 bursts (83.7%), with a median reduction of 0.00849%.

The releases selected the same component count in 3,967 of 4,455 fits. Of the 700 bursts whose scattering decision changed, 581 had a fitted 600 MHz tail shorter than one time bin.
