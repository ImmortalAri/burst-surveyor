# Burst Surveyor

**AI-assisted FitBurst fitting, run by your coding agent.**

Give your agent a folder of dedispersed burst observations. It runs the fits, inspects the profiles and residuals, attempts permitted rescues, and returns figures and verified reports. You get a ranked queue of events that may need attention.

## Give this to your agent

Replace the two paths and paste this into your coding agent:

```text
Use https://github.com/ImmortalAri/burst-surveyor and follow AGENTS.md.
Set up a fresh Python 3.12 environment.

Inputs: /absolute/path/to/my/npz-files
Work: /absolute/path/to/a/new/run

Complete the fitting and visual-review workflow, including permitted
follow-ups. Return verified report.csv, report.json, decisions and figures.
Leave persistent or ambiguous problems marked for review.
```

Observations and reports stay local. See the [input format](docs/reference.md), [agent instructions](AGENTS.md), and [detailed guide](docs/guide.md).

## Real bursts

**FRB 20190412A — overlapping peaks and a long tail.**

![Real burst FRB 20190412A: observed pulse profile, fitted reconstruction, spectrum and residuals](examples/frb20190412a.png)

**Event 156410110 — a train of sharp peaks with overlapping tails.** The human reference is overlaid on the pulse profile for context.

![Real multipulse burst: observed profile with AI-assisted FitBurst and human reference curves, observed spectrum, fitted reconstruction and residuals](examples/real-multipulse.png)

Model components need not correspond to separate physical pulses. The agent reviews every exported fit; persistent problems remain visible for your inspection.
