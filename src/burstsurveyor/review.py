"""Build a review queue or render an observed spectrum, model and residual."""

import json
from pathlib import Path
import tempfile
import time


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_event(summary, event_id):
    for event in summary["events"]:
        if event["id"] == event_id:
            return event
    raise ValueError(f"Unknown event ID {event_id!r}. See results.json.")


def queue(run, summary=None, *, sample_clean=0):
    """Report reasons to inspect a result. Flags indicate questions, not probabilities."""
    from .batch import verify

    run = Path(run)
    if summary is None:
        summary = verify(run)
    if type(sample_clean) is not int or sample_clean < 0:
        raise ValueError("The unflagged sample size must be a nonnegative integer")
    rows, controls = [], []
    for event in summary["events"]:
        flags = []
        peak = event.get("residual_peak")
        if event["status"] != "provisional":
            flags.append(event["status"])
        if peak is not None and peak >= 6:
            flags.append("residual_peak")
        if event.get("primary_receipt") and event["status"] == "provisional":
            primary = _read(run / event["primary_receipt"])
            chosen = primary.get("selected") or {}
            if chosen.get("termination") in {"wall_limit", "work_limit"}:
                flags.append("budget_limited")
            if chosen.get("active_bounds"):
                flags.append("parameter_boundary")
        item = dict(
            id=event["id"],
            input_name=event["input_name"],
            source_sha256=event["source_sha256"],
            flags=flags,
            residual_peak=peak,
            status=event["status"],
            reasons=event.get("reasons", []),
        )
        if flags:
            rows.append(item)
        elif sample_clean:
            controls.append(dict(item, flags=["audit_sample"]))
    return (
        sorted(
            rows,
            key=lambda r: (
                r["status"] != "unresolved",
                -(r["residual_peak"] if r["residual_peak"] is not None else -1),
                r["id"],
            ),
        )
        + sorted(controls, key=lambda r: (r["source_sha256"], r["id"]))[:sample_clean]
    )


def load_for_review(path, scratch, source_hash, *, wait_seconds=30):
    import psutil
    from ._dispatch import estimate_bytes, RESERVE_MIB, MIB
    from .science import inspect_npz, load_event

    needed = estimate_bytes(inspect_npz(path)) + RESERVE_MIB * MIB
    deadline = time.monotonic() + wait_seconds
    while psutil.virtual_memory().available < needed:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Review needs {needed / MIB:.0f} MiB of available memory; retry when available"
            )
        time.sleep(1)
    data, entry = load_event(path, scratch)
    if entry["source_sha256"] != source_hash:
        raise ValueError("The raw NPZ differs from the sealed event")
    return data, entry


def _plot_source(run, event, alternative):
    from ._storage import read, sha256
    from .secondlook import verify_second_look

    digest = path = None
    selection = "baseline"
    if alternative is not None:
        alternative = Path(alternative)
        seal = verify_second_look(run, alternative)
        if event["id"] not in {r["id"] for r in seal["events"]}:
            raise ValueError("Event is absent from this second look")
        folder = alternative / event["id"]
        decision = read(folder / "decision.json")
        if decision["status"] == "provisional_challenger":
            path, digest = folder / "challenger.npz", decision["prediction_sha256"]
            selection = "challenger"
    if path is None and event.get("prediction"):
        path = run / event["prediction"]
        digest = read(run / event["native_receipt"])["prediction_sha256"]
    if path is None and event["status"] != "unresolved":
        raise ValueError(f"Event has no usable reconstruction ({event['status']})")
    if path is not None and sha256(path) != digest:
        raise ValueError("Changed upstream artifact")
    return selection, path, digest


def _display_data(data, entry, signal, baseline):
    """Pool pixels for display only; compute the profile on unpooled usable channels."""
    import numpy as np

    nf, nt = data.e.raw.shape
    fs, ts = max(1, (nf + 179) // 180), max(1, (nt + 449) // 450)
    starts = np.arange(0, nt, ts)
    widths = np.minimum(ts, nt - starts)
    ncurves = 1 if signal is None else 2
    images = np.zeros((ncurves, (nf + fs - 1) // fs, len(starts)))
    profile = np.zeros((ncurves, len(starts)))
    counts = np.zeros(images.shape[1], dtype=int)
    weight_sum = 0.0
    for i in data.indices:
        obs = data.e.raw[i].astype(float) - baseline[i]
        curves = np.array([obs] if signal is None else [obs, signal[i]])
        pooled = np.add.reduceat(curves, starts, axis=1) / widths
        weight = 1.0 / data.var[i]
        profile += weight * pooled
        weight_sum += weight
        images[:, i // fs] += pooled * np.sqrt(weight)
        counts[i // fs] += 1
    profile /= weight_sum
    images /= np.where(counts, counts, np.nan)[None, :, None]
    grid = entry["grid"]
    time = (np.r_[starts, nt] - 0.5) * grid["res_time"] * 1000
    freq = grid["freqs_bin0"] + np.r_[np.arange(0, nf, fs), nf] * grid["res_freq"]
    return dict(
        images=images, profile=profile, time=time, freq=freq, time_step=ts, freq_step=fs
    )


def _render(view, event, selected, destination):
    import os
    import numpy as np
    from matplotlib import colormaps, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    images = list(view["images"])
    modeled = len(images) == 2
    if modeled:
        images.append(images[0] - images[1])
    finite = np.abs(images[0][np.isfinite(images[0])])
    limit = max(float(np.percentile(finite, 99.7)), 1e-12)
    mids = (view["time"][:-1] + view["time"][1:]) * 0.5
    labels = ("Observed", "Fitted reconstruction", "Residual")
    with rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig = Figure(
            figsize=(11.6, 5.6) if modeled else (7.0, 5.2), layout="constrained"
        )
        FigureCanvasAgg(fig)
        grid = fig.add_gridspec(
            2,
            len(images) + 1,
            height_ratios=[1.0, 1.65],
            width_ratios=[1.0] * len(images) + [0.035],
        )
        top = fig.add_subplot(grid[0, :-1])
        top.plot(
            mids,
            view["profile"][0],
            color="#26323b",
            linewidth=1.2,
            drawstyle="steps-mid",
            label="Observed",
        )
        if modeled:
            top.plot(
                mids,
                view["profile"][1],
                color="#bb4d29",
                linewidth=1.9,
                label="FitBurst",
            )
        top.set(
            xlim=view["time"][[0, -1]],
            xlabel="Time from first sample (ms)",
            ylabel="Intensity (input units)",
        )
        top.set_title(
            "Pulse profile   ·   inverse-variance channel mean", loc="left", fontsize=11
        )
        top.legend(loc="upper right", frameon=False, ncol=2)
        color = colormaps["RdBu_r"].with_extremes(bad="#e3e5e7")
        first = None
        for j, panel in enumerate(images):
            ax = fig.add_subplot(grid[1, j], sharey=first)
            mesh = ax.pcolormesh(
                view["time"],
                view["freq"],
                panel,
                cmap=color,
                vmin=-limit,
                vmax=limit,
                shading="flat",
                rasterized=True,
            )
            ax.set(xlabel="Time (ms)", ylim=(min(view["freq"]), max(view["freq"])))
            ax.set_title(labels[j], loc="left", fontsize=11)
            if first is None:
                first = ax
                ax.set_ylabel("Frequency (MHz)")
            else:
                ax.tick_params(axis="y", labelleft=False)
        fig.colorbar(
            mesh, cax=fig.add_subplot(grid[1, -1]), label="Signal / channel noise"
        )
        suffix = "  ·  challenger" if selected == "challenger" else ""
        if not modeled:
            suffix = "  ·  no supported model"
        fig.suptitle(event["input_name"] + suffix, fontsize=13)
        # Exclusive atomic publication keeps interrupted or competing writers out.
        with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
            rendered = Path(temp) / "figure.png"
            fig.savefig(rendered, format="png", dpi=145)
            if os.name == "nt":
                os.rename(rendered, destination)
            else:
                os.link(rendered, destination)
    return limit


def plot_verified_event(
    run, event, source, destination, *, alternative=None, review_data=None
):
    """Render one previously verified summary event and return its figure provenance."""
    import numpy as np
    from ._storage import sha256

    run, source, destination = Path(run), Path(source), Path(destination)
    if destination.suffix.lower() != ".png" or destination.exists():
        raise ValueError("Choose a new output filename ending in .png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    selection, artifact, digest = _plot_source(run, event, alternative)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        if review_data is None:
            data, entry = load_for_review(source, Path(scratch), event["source_sha256"])
        else:
            data, entry = review_data[:2]
            if entry["source_sha256"] != event["source_sha256"]:
                raise ValueError("Review data does not match the sealed event")
        signal, baseline = None, data.base
        if artifact is not None:
            with np.load(artifact, allow_pickle=False) as z:
                if (
                    str(z["source_sha256"]) != entry["source_sha256"]
                    or not np.array_equal(z["channels"], data.indices)
                    or not np.array_equal(
                        z["variance"][data.indices], data.var[data.indices]
                    )
                    or not np.array_equal(z["frequency"], data.e.frequency)
                    or not np.array_equal(z["time_s"], data.t * data.e.coords.dt)
                ):
                    raise ValueError("Published model and original grid or mask differ")
                signal, baseline = z["prediction"], z["baseline"]
        view = _display_data(data, entry, signal, baseline)
        limit = _render(view, event, selection, destination)
        return dict(
            id=event["id"],
            file=destination.name,
            figure_sha256=sha256(destination),
            source_sha256=entry["source_sha256"],
            prediction_sha256=digest,
            selected=selection,
            has_model=signal is not None,
            shape=list(data.e.raw.shape),
            grid=entry["grid"],
            usable_channels=len(data.indices),
            color_limit=limit,
            time_bins_per_display_bin=view["time_step"],
            channels_per_display_bin=view["freq_step"],
            normalization="Subtract profiled baseline. Spectra use channel noise units; profile uses inverse-variance weights in input units.",
        )


def plot_event(run, event_id, source, destination, *, alternative=None):
    """Render an event from a verified run and return the output PNG path."""
    from .batch import verify

    run = Path(run)
    event = find_event(verify(run), event_id)
    plot_verified_event(run, event, source, destination, alternative=alternative)
    return Path(destination)
