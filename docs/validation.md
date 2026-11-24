# Validation of the default burst fit

## Reconstruction

Burst Surveyor v0.2.0 and v0.1.1 reconstructed the same 4,455 of 4,464 CHIME/FRB Catalog 2 intensity observations. For 4,445 bursts, we scored both fits and the saved catalog model on identical observed pixels. [Catalog data](https://chime-frb.ca/catalogs/catalog2)

| Fitter | ≥90% of catalog whole-window gain | Higher whole-window gain | Lower on-pulse error | Median on-pulse error reduction |
| --- | ---: | ---: | ---: | ---: |
| v0.2.0 | **4,426/4,441 (99.7%)** | 4,283/4,445 (96.4%) | 4,079/4,386 (93.0%) | 0.0115% |
| v0.1.1 | 3,856/4,441 (86.8%) | 2,753/4,445 (61.9%) | 2,404/4,386 (54.8%) | 0.00137% |

Whole-window gain is the fraction of noise-weighted squared error removed relative to a constant intensity per channel. The 90% threshold applies only to positive catalog gains. On-pulse error uses the same weights within the saved v0.1.1 on-pulse mask. The median includes all 4,386 bursts with on-pulse bins.

## Component counts

Whole-window gains agreed within 0.1 percentage point for 4,350 of 4,445 v0.2.0 fits (97.9%). Of those fits, 3,093 (71.1%) used a different component count from the catalog.

The blind BaseCat 2–like test used 780 injections into 130 observed noise backgrounds on a shared time grid. **v0.2.0 and v0.1.1** each split 151 of 260 single pulses (58.1%) and recovered both arrivals in 107 of 520 weak pairs (20.6%). The 779 exported predictions matched byte for byte between releases. Both left the same case unresolved.

Pair recovery required each injected arrival to match a distinct fitted arrival within the smaller of two time bins and 45% of the pair separation. These rates apply to the tested pulse strengths and separations.

## Channel timing

Measured channel offsets were available for 4,457 of 4,464 observations in the v0.2.0 run. Compared with the v0.1.1 fits on a shared time grid, v0.2.0 had higher whole-window gain in 3,732 of 4,455 fits (83.8%). On-pulse error fell in 3,679 of 4,393 bursts (83.7%), with a median reduction of 0.00849%.

The releases selected the same component count in 3,967 of 4,455 fits. Of the 700 bursts whose scattering decision changed, 581 had a fitted 600 MHz tail shorter than one time bin.
