"""Report component geometry and crop-sensitive terms without rejecting fits."""

import numpy as np

from .science.model import DISPERSION_SECONDS_MHZ2


def component_review(parameters, frequency, time_s):
    """Inspect official FitBurst parameters on the actual usable native grid.

    Arrival parameters are Gaussian centers, before scattering, in seconds.
    Frequency is in MHz; dm is the residual DM of the dedispersed export.
    """
    freq, time = np.asarray(frequency, dtype=float), np.asarray(time_s, dtype=float)
    if (freq.ndim != 1 or not len(freq) or not np.all(np.isfinite(freq) & (freq > 0))
            or time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all()
            or not np.all(np.diff(time) > 0)):
        raise ValueError("Component review requires positive frequencies and ordered native times")
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=1e-7, atol=1e-12):
        raise ValueError("Component review requires uniformly spaced native times")
    first, last = float(time[0] - dt/2), float(time[-1] + dt/2)
    duration = last - first
    n = len(np.atleast_1d(parameters["arrival_time"]))
    if not n:
        raise ValueError("No fitted component parameters")

    def vector(name, default=None):
        values = np.atleast_1d(np.asarray(parameters.get(name, default), dtype=float))
        if values.ndim != 1 or len(values) not in (1, n) or not np.isfinite(values).all():
            raise ValueError(f"Invalid component parameter: {name}")
        return np.repeat(values, n) if len(values) == 1 else values

    arrival, sigma, ref = vector("arrival_time"), vector("burst_width"), vector("ref_freq")
    dm, dm_index = vector("dm", 0.0), vector("dm_index", -2.0)
    tau = vector("scattering_timescale", 0.0)
    slope, running = vector("spectral_index", 0.0), vector("spectral_running", 0.0)
    if np.any(sigma <= 0) or np.any(ref <= 0) or np.any(tau < 0):
        raise ValueError("Invalid component width, reference frequency, or scattering")
    rows = []
    for j in range(n):
        centers = arrival[j] + DISPERSION_SECONDS_MHZ2 * dm[j] * (
            freq**dm_index[j] - ref[j]**dm_index[j]
        )
        if not np.isfinite(centers).all():
            raise ValueError("Nonfinite dispersed component arrival")
        outside = bool(np.all(centers < first) or np.all(centers > last))
        broad_edge = outside and sigma[j] >= duration / 4
        geometry = ["broad_edge_component"] if broad_edge else []
        if sigma[j] < dt:
            geometry.append("sub_bin_width")
        if sigma[j] >= duration / 4:
            geometry.append("broad_component")
        if (centers.min() - 2*sigma[j] < first
                or centers.max() + 2*sigma[j] + 3*tau[j] > last):
            geometry.append("crop_sensitive_component")
        rows.append(dict(
            component=j + 1, reference_mhz=float(ref[j]),
            gaussian_arrival_ms=float(1000 * arrival[j]),
            gaussian_sigma_ms=float(1000 * sigma[j]),
            gaussian_arrival_range_ms=[float(1000*centers.min()), float(1000*centers.max())],
            sigma_fraction_of_record=float(sigma[j] / duration),
            residual_dm_pc_cm3=float(dm[j]), scattering_ms_at_reference=float(1000*tau[j]),
            spectral_index=None if parameters.get('spectral_model') == 'channel-amplitudes-v1' else float(slope[j]),
            spectral_running=None if parameters.get('spectral_model') == 'channel-amplitudes-v1' else float(running[j]),
            all_gaussian_centers_outside_record=outside,
            flags=geometry,
        ))
    flagged = [r["component"] for r in rows if "broad_edge_component" in r["flags"]]
    return dict(
        component_count=n, record_edges_ms=[1000*first, 1000*last], components=rows,
        flags=["broad_edge_component"] if flagged else [], review_components=flagged,
        action="review crop, off-burst structure and a simpler fit" if flagged else "none",
        note="Model components are not counted profile peaks. Flags request review; they do not reject broad or truncated astrophysical emission.",
    )
