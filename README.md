# Burst Surveyor

FitBurst reconstruction of dedispersed radio bursts. Give the AI this checkout and the NPZ directory.

> Follow [AGENTS.md](AGENTS.md). Fit `INPUTS` into `WORK`. Return the verified figures and event table.

**Seven synthetic pulses.** The first fit missed the pulse at 135 ms. The follow-up raised weighted reconstruction gain by 8.6%. [NPZ](examples/seven-pulse.npz)

![Seven-pulse profile and dynamic spectrum with the fitted reconstruction and residual](examples/seven-pulse-preview.png)

**Two synthetic pulses.** The first fit recovered both. [NPZ](examples/two-pulse.npz)

![Two-pulse profile and dynamic spectrum with the fitted reconstruction and residual](examples/two-pulse-preview.png)
