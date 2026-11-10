"""Integrated positive-fluence Gaussian and exponentially modified Gaussian."""

import numpy as np
from scipy.special import erfcx, ndtr

REFERENCE_MHZ = 600.0
DISPERSION_SECONDS_MHZ2 = 4149.377593360996
LOG_FREQUENCY_SCALE = 0.3
SQRT_2PI = np.sqrt(2 * np.pi)
SQRT_2 = np.sqrt(2.0)


def gaussian_edge(v, sigma):
    z = v / sigma
    density = np.exp(-0.5 * z * z) / SQRT_2PI
    return (ndtr(z), density / sigma, -z * density)


def scattered_edge(v, sigma, tau):
    z = v / sigma
    r = np.broadcast_to(sigma / tau, z.shape)
    small = r > 55
    # In the narrow-exponential limit use its first two moments. At the
    # transition, the omitted standardized third cumulant is below 1.3e-5.
    regular_r = np.minimum(r, 55.0)
    u = regular_r - z
    positive = u >= 0
    density = np.exp(-0.5 * z * z) / SQRT_2PI
    term = np.empty_like(z)
    term[positive] = SQRT_2PI / 2 * density[positive] * erfcx(u[positive] / SQRT_2)
    term[~positive] = np.exp(
        0.5 * regular_r[~positive] ** 2 - z[~positive] * regular_r[~positive]
    ) * ndtr(-u[~positive])
    cdf = ndtr(z) - term
    dv = term * regular_r / sigma
    ds = regular_r * (density - regular_r * term)
    dtau = -ds - z * regular_r * term
    if np.any(small):
        tb = np.broadcast_to(tau, z.shape)[small]
        eff = np.hypot(sigma, tb)
        cdf[small], dv[small], ds_eff = gaussian_edge(v[small] - tb, eff)
        ds[small] = ds_eff * (sigma / eff) ** 2
        dtau[small] = -tb * dv[small] + ds_eff * (tb / eff) ** 2
    return (cdf, dv, ds, dtau)


class Coordinates:
    def __init__(self, frequency, dt, full_frequency):
        self.frequency = frequency
        self.dt = dt
        self.z = np.log(frequency / REFERENCE_MHZ) / LOG_FREQUENCY_SCALE
        self.inverse_square_span = float(np.ptp(full_frequency ** (-2)))
        self.dispersion = (
            frequency ** (-2) - REFERENCE_MHZ ** (-2)
        ) / self.inverse_square_span
        self.tau_ratio = (frequency / REFERENCE_MHZ) ** (-4)

    def parameters(self, x, n, scatter):
        c = x[: 5 * n].reshape(n, 5)
        return {
            "fluence_intensity_seconds": (np.exp(c[:, 0]) * self.dt).tolist(),
            "arrival_seconds_relative": (c[:, 1] * self.dt).tolist(),
            "gaussian_sigma_seconds": (np.exp(c[:, 2]) * self.dt).tolist(),
            "spectral_index": (c[:, 3] / LOG_FREQUENCY_SCALE).tolist(),
            "spectral_running": (c[:, 4] / LOG_FREQUENCY_SCALE**2).tolist(),
            "residual_dm_pc_cm3": x[5 * n]
            * self.dt
            / (DISPERSION_SECONDS_MHZ2 * self.inverse_square_span),
            "tau_seconds": np.exp(x[-1]) * self.dt if scatter else 0.0,
            "reference_mhz": REFERENCE_MHZ,
            "scattering_index": -4.0,
            "dispersion_index": -2.0,
        }


def pulse(x, n, scatter, coords, channels, t, derivative=False):
    """Predict on centres t in bin units. Truncate only at negligible tail mass."""
    n_freq, n_time = (len(coords.z[channels]), len(t))
    result = np.zeros((n_freq, n_time))
    jac = np.zeros((len(x), n_freq, n_time)) if derivative else None
    z = coords.z[channels]
    disp = coords.dispersion[channels]
    delay = x[5 * n] * disp
    tau = (np.exp(x[-1]) * coords.tau_ratio[channels])[:, None] if scatter else None
    for i, (log_fluence, arrival, log_sigma, slope, curvature) in enumerate(
        x[: 5 * n].reshape(n, 5)
    ):
        sigma = np.exp(log_sigma)
        centres = arrival + delay
        a, b = np.searchsorted(
            t,
            [
                centres.min() - 10 * sigma - 0.5,
                centres.max() + 10 * sigma + 0.5 + (30 * tau.max() if scatter else 0.0),
            ],
        )
        if a == b:
            continue
        v = t[None, a:b] - centres[:, None]
        if scatter:
            hi = scattered_edge(v + 0.5, sigma, tau)
            lo = scattered_edge(v - 0.5, sigma, tau)
        else:
            hi = gaussian_edge(v + 0.5, sigma)
            lo = gaussian_edge(v - 0.5, sigma)
        fluence = np.exp(log_fluence + slope * z + curvature * z * z)[:, None]
        piece = fluence * (hi[0] - lo[0])
        result[:, a:b] += piece
        if derivative:
            arrival_d = fluence * (lo[1] - hi[1])
            jac[5 * i : 5 * i + 5, :, a:b] = (
                piece,
                arrival_d,
                fluence * (hi[2] - lo[2]),
                piece * z[:, None],
                piece * z[:, None] ** 2,
            )
            jac[5 * n, :, a:b] += arrival_d * disp[:, None]
            if scatter:
                jac[-1, :, a:b] += fluence * (hi[3] - lo[3])
    return (result, jac)
