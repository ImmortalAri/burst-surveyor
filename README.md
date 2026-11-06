# Burst Surveyor

FitBurst reconstruction of dedispersed radio bursts. Give the AI this checkout and the NPZ directory.

> Follow [AGENTS.md](AGENTS.md). Fit `INPUTS` into `WORK`. Return the verified figures and event table.

**FRB 20190412A.** Three overlapping peaks and a long tail. The five-component reconstruction improved weighted gain by 9.7% over our earlier two-component fit. Some frequency structure remains in the residual. [CHIME/FRB observation](https://chime-frb.ca/catalogs/basecat1/FRB20190412A) · [Catalog paper](https://arxiv.org/abs/2311.00111)

![Observed profile, five-component FitBurst reconstruction and residual spectrum of FRB 20190412A](examples/frb20190412a.png)

**Seven synthetic pulses.** The first fit missed the pulse at 135 ms. The follow-up raised weighted reconstruction gain by 8.6%. [NPZ](examples/seven-pulse.npz)

![Seven-pulse profile and dynamic spectrum with the fitted reconstruction and residual](examples/seven-pulse-preview.png)

**Two synthetic pulses.** The first fit recovered both. [NPZ](examples/two-pulse.npz)

![Two-pulse profile and dynamic spectrum with the fitted reconstruction and residual](examples/two-pulse-preview.png)
