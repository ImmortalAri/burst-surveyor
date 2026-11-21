# Validation of the default burst fit

## Reconstruction

Burst Surveyor v0.1.1 produced reconstructions for 4,455 of 4,464 CHIME/FRB Catalog 2 intensity observations. For 4,445 bursts, we scored Surveyor and the catalog reference on the same observed pixels. The score is the fraction of noise-weighted squared error removed relative to a constant intensity in each channel.

The whole-window scores agreed within 0.1 percentage points for **4,305 of 4,445 bursts (96.9%)**. A weak burst can account for little of the error across the full window. Surveyor recovered at least 90% of the reference improvement in 86.8% of the 4,441 cases with positive reference gain. Inspect the on-pulse profile and residuals for weak structure. [Catalog data](https://chime-frb.ca/catalogs/catalog2)

## Component counts

Among the 4,305 bursts with close scores, Surveyor and the catalog reference used different component counts in **2,998 (69.6%)**. The fitted count therefore describes one way to reconstruct the emission.

We also added known pulses to real noise backgrounds. In a blind test designed to resemble BaseCat 2, the default fit split 151 of 260 single pulses and recovered both arrival times in 107 of 520 weak pairs. These fractions apply to the injected pulse strengths and separations. Estimating how often real bursts contain multiple pulses requires accounting for both false splits and missed neighbors.
