"""Experimental, predictive temporal selection with profiled channel spectra.

Spectral amplitudes are nuisance parameters, not new pulses. Geometry and NNLS
amplitudes are fitted on four of each six native time bins; the other two are
used only for model selection. Channel/noise screening is shared with reporting
and is therefore conditional, not a fully independent detection experiment.
The one-standard-error rule prefers the smallest adequate temporal model.
All accepted and rejected candidates, validation blocks and stopping reasons
are retained. Physical interpretation still requires review.
"""

import copy
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from scipy.signal import find_peaks

from .engine import Limit, optimizer
from .model import pulse

MODEL = "channel-amplitudes-v1"
POLICY = dict(train_bins_per_cycle=4, cycle_bins=6, selection_standard_errors=1.0,
              validation_floor=4.0, discovery_snr=6.0, max_search_seconds=45.0,
              grouped_iterations=24, native_iterations=8, final_iterations=20)


def geometry_indices(n, scatter):
    return np.r_[np.array([[5*i+1, 5*i+2] for i in range(n)]).ravel(),
                 np.arange(5*n, 5*n+1+int(scatter))].astype(int)


def shapes(x, n, scatter, coords, channels, t, derivative=False):
    """Unit-fluence integrated profiles, plus geometry derivatives per component."""
    h, derivatives = [], []
    for i in range(n):
        single = np.r_[0., x[5*i+1:5*i+3], 0., 0., x[5*n:]]
        p, j = pulse(single, 1, scatter, coords, channels, t, derivative)
        h.append(p)
        if derivative:
            derivatives.append(j[[1, 2, 5] + ([6] if scatter else [])])
    return np.stack(h, axis=2), derivatives


def coefficients(h, y):
    """Nonnegative channel amplitudes; channel-constant weights cancel here."""
    norms = np.sqrt(np.sum(h*h, axis=1))
    if h.shape[2] == 1:
        den = norms[:, 0]**2
        return np.maximum(0, np.divide(np.sum(h[:, :, 0]*y, axis=1), den,
                          out=np.zeros(len(h)), where=den > 1e-200))[:, None]
    result = np.zeros((len(h), h.shape[2]))
    for k in range(len(h)):
        active = norms[k] > 1e-100
        if np.any(active):
            result[k, active] = nnls(h[k][:, active]/norms[k, active], y[k],
                                     maxiter=1000)[0]/norms[k, active]
    return result


class Objective:
    def __init__(self, data, train, *, native, spectral_group_size=1, spectral_group_sizes=None):
        self.d, self.train, self.native = data, np.asarray(train, bool), native
        self.t = np.flatnonzero(self.train).astype(float)
        self.spectral_group_size = spectral_group_size if native else 1
        self.spectral_group_sizes = spectral_group_sizes if native else None
        if native:
            self.channels = data.indices
            self.y = None
            self.variance = data.var[self.channels]
        else:
            self.channels = data.reps
            self.y = data.y[:, self.train]/data.group_norm[:, None]
            self.variance = 1/data.group_norm**2

    def evaluate(self, x, n, scatter, derivative=False, keep=False):
        if self.spectral_group_sizes is not None:
            from .combined_spectra import mixed_evaluate
            return mixed_evaluate(self, x, n, scatter, derivative, keep)
        if self.spectral_group_size > 1:
            from .combined_spectra import pooled_evaluate
            return pooled_evaluate(self, x, n, scatter, derivative, keep)
        d = self.d
        d.e.budget['evaluations'] += 1
        work = len(self.channels)*len(self.t)*max(1, n)*(3 if derivative else 1)
        d.e.budget['native_work'] += work
        if (d.e.budget['evaluations'] > d.c['max_evaluations'] or
                d.e.budget['native_work'] > d.c['max_native_work']):
            raise Limit('scintillation fit exceeded shared work budget')
        size = 2*n+1+int(scatter)
        grad, gram = np.zeros(size), np.zeros((size, size))
        loss = 0.
        amps = []
        tile_size = max(1, min(16, d.c['tile_derivative_cells']//max(1, len(self.t)*size)))
        for start in range(0, len(self.channels), tile_size):
            ii = self.channels[start:start+tile_size]
            y = (d.e.raw[np.ix_(ii, self.train)].astype(float) if self.native
                 else self.y[start:start+tile_size].copy())
            y -= y.mean(axis=1, keepdims=True)
            h, derivs = shapes(x, n, scatter, d.e.coords, ii, self.t, derivative)
            h -= h.mean(axis=1, keepdims=True)
            a = coefficients(h, y)
            residual = y-np.einsum('ctn,cn->ct', h, a)
            var = self.variance[start:start+tile_size]
            loss += float(np.sum(residual**2/var[:, None]))
            if keep:
                amps.append(a)
            if derivative:
                jac = np.zeros((size, len(ii), len(self.t)))
                for i, j in enumerate(derivs):
                    j -= j.mean(axis=2, keepdims=True)
                    jac[2*i:2*i+2] = j[:2]*a[None, :, i, None]
                    jac[2*n] += j[2]*a[:, i, None]
                    if scatter:
                        jac[-1] += j[3]*a[:, i, None]
                # Project nuisance directions out of the Gauss-Newton curvature.
                if n == 1:
                    den = np.sum(h[:, :, 0]**2, axis=1)
                    projection = np.einsum('ct,pct->pc', h[:, :, 0], jac)
                    projection = np.divide(projection, den[None, :],
                        out=np.zeros_like(projection), where=(den > 1e-200)[None, :])
                    projection[:, a[:, 0] <= 0] = 0.
                    jac -= projection[:, :, None]*h[None, :, :, 0]
                else:
                    for k in range(len(ii)):
                        active = a[k] > 0
                        if np.any(active):
                            design = h[k][:, active]
                            jac[:, k] -= (design @ np.linalg.lstsq(design, jac[:, k].T,
                                                                  rcond=1e-10)[0]).T
                grad += np.einsum('pct,ct,c->p', jac, residual, 1/var)
                gram += np.einsum('pct,qct,c->pq', jac, jac, 1/var)
        return loss, grad, gram, np.concatenate(amps) if keep else None


def optimize(data, seed, train, *, native, iterations, deadline):
    n, scatter = seed['n'], seed['scatter']
    x = np.asarray(seed['x'], float).copy()
    idx = geometry_indices(n, scatter)
    lo, hi = data.bounds(n, scatter)
    objective = Objective(data, train, native=native,
                          spectral_group_size=seed.get('spectral_group_size', 1),
                          spectral_group_sizes=seed.get('spectral_group_sizes'))
    loss = objective.evaluate(x, n, scatter)[0]
    damping, accepted, termination = .01, 0, 'iteration_limit'
    for _ in range(iterations):
        if time.monotonic() >= deadline:
            termination = 'wall_limit'
            break
        loss, g, h, _ = objective.evaluate(x, n, scatter, derivative=True)
        scale = 1/np.sqrt(np.maximum(np.diag(h), 1e-12))
        if np.max(abs(g*scale)) < .01:
            termination = 'projected_gradient'
            break
        moved = False
        for _ in range(6):
            step = np.linalg.solve(h*scale[:, None]*scale[None, :]+damping*np.eye(len(idx)),
                                   g*scale)*scale
            trial = x.copy()
            trial[idx] = np.clip(x[idx]+step, lo[idx]+1e-8, hi[idx]-1e-8)
            trial_loss = objective.evaluate(trial, n, scatter)[0]
            improvement = loss-trial_loss
            if improvement > 0:
                x, loss = trial, trial_loss
                accepted += 1
                damping = max(1e-6, damping*.35)
                moved = True
                if improvement < .001:
                    termination = 'small_training_gain'
                break
            damping = min(1e9, damping*8)
            if time.monotonic() >= deadline:
                break
        if not moved or termination != 'iteration_limit':
            if not moved:
                termination = 'stalled'
            break
    loss, _, _, a = objective.evaluate(x, n, scatter, keep=True)
    # Canonical amplitude fields are placeholders in this explicitly tagged model.
    x[:5*n:5] = 0.
    x[3:5*n:5] = 0.
    x[4:5*n:5] = 0.
    return dict(seed, x=x.tolist(), model=MODEL, termination=termination,
                accepted=accepted, training_loss=loss, active_bounds=np.flatnonzero(
                    np.minimum(x[idx]-lo[idx], hi[idx]-x[idx]) < .002*(hi[idx]-lo[idx])).tolist(),
                amplitude_channels=objective.channels.tolist(), channel_amplitudes=a.tolist())


def prediction(data, row, channels=None, t=None):
    channels = data.indices if channels is None else np.asarray(channels)
    t = data.t if t is None else t
    if row.get('model') != MODEL:
        return pulse(np.asarray(row['x']),row['n'],row['scatter'],data.e.coords,channels,t)[0]
    lookup = np.full(len(data.e.raw), -1, int)
    lookup[np.asarray(row['amplitude_channels'])] = np.arange(len(row['amplitude_channels']))
    if np.any(lookup[channels] < 0):
        raise ValueError('channel amplitudes missing requested channels')
    h, _ = shapes(np.asarray(row['x']), row['n'], row['scatter'], data.e.coords, channels, t)
    return np.einsum('ctn,cn->ct', h, np.asarray(row['channel_amplitudes'])[lookup[channels]])


def validate(data, row, train):
    """Predict held bins with both geometry AND amplitudes fixed from training."""
    ntime = data.e.length
    losses = np.zeros((2, ntime))
    null_losses = np.zeros(ntime)
    gains = np.zeros(2)
    energy = 0.
    for ii in data.e.tiles(data.indices):
        p = prediction(data, row, ii)
        y = data.e.raw[ii].astype(float)
        baseline = (y[:, train]-p[:, train]).mean(axis=1)
        residual = y-p-baseline[:, None]
        null = y-y[:, train].mean(axis=1, keepdims=True)
        null_losses += np.sum(null**2/data.var[ii, None], axis=0)
        for split in range(2):
            keep = np.arange(len(data.e.raw))[ii] % 2 == split
            losses[split] += np.sum(residual[keep]**2/data.var[ii[keep], None], axis=0)
        pc = p-p.mean(axis=1, keepdims=True)
        yc = y-y.mean(axis=1, keepdims=True)
        gains[0] += np.sum(pc*(2*yc-pc)/data.var[ii, None])
        gains[1] += np.sum(pc*(2*yc-pc)/data.rms_var[ii, None])
        energy += float(np.sum(pc**2/data.var[ii, None]))
    blocks = np.arange(ntime)//POLICY['cycle_bins']
    held = ~train
    split_blocks = [np.bincount(blocks[held], weights=loss[held]).tolist() for loss in losses]
    return dict(validation_split_block_losses=split_blocks,
                validation_loss=float(losses[:, held].sum()), gain=float(gains[0]),
                validation_gain=float(null_losses[held].sum()-losses[:, held].sum()),
                rms_gain=float(gains[1]), energy=energy)


def paired_noise_se(data, row, best, train):
    """Conditional noise uncertainty, without treating burst signal as noise.

    For fixed validation predictions, Var(SSE_a-SSE_b)=4||a-b||_W^2
    under independent unit-variance noise. Use the existing off-burst temporal
    and frequency correlations as conservative AR(1) variance inflation.
    Baselines are fitted on training samples and never refitted on validation.
    This is a working-noise comparison, not a calibrated component p-value.
    """
    energy = 0.
    for ii in data.e.tiles(data.indices):
        delta = prediction(data,row,ii)-prediction(data,best,ii)
        delta -= delta[:,train].mean(axis=1,keepdims=True)
        energy += float(np.sum(delta[:,~train]**2/data.var[ii,None]))
    inflation = 1.
    for rho in (data.rho,data.rho_freq):
        inflation *= (1+rho)/max(1e-6,1-rho)
    return 2*np.sqrt(max(0.,energy)*inflation),energy,inflation


def select(rows, data=None, train=None, cache=None):
    rows = [r for r in rows if r.get('validation_gain', 1.) > 0]
    if not rows:
        return None
    best = min(rows, key=lambda r: (r['validation_loss'], r['n'], r['scatter']))
    credible = []
    for row in rows:
        delta = np.asarray(row['validation_split_block_losses']).sum(axis=0)-np.asarray(
            best['validation_split_block_losses']).sum(axis=0)
        if data is not None:
            key=(row['name'],row.get('model'),best['name'],best.get('model'))
            if row is best:
                se,energy,inflation=0.,0.,1.
            elif cache is not None and key in cache:
                se,energy,inflation=cache[key]
            else:
                se,energy,inflation = paired_noise_se(data,row,best,train)
                if cache is not None:cache[key]=(se,energy,inflation)
            row['validation_prediction_difference_energy'] = energy
            row['validation_noise_variance_inflation'] = inflation
        else:
            reference=dict(name=best['name'],model=best.get('model'))
            if row is best:
                se=0.
            elif row.get('validation_uncertainty_reference')==reference:
                se=row['validation_difference_se']
            else:
                raise ValueError('Native data or a matching conditional-noise receipt is required')
        row['validation_uncertainty_reference']=dict(name=best['name'],model=best.get('model'))
        row['validation_excess_vs_best'] = float(delta.sum())
        row['validation_difference_se'] = se
        row['within_one_se'] = bool(delta.sum() <= max(POLICY['validation_floor'], se))
        if row['within_one_se']:
            credible.append(row)
    return min(credible, key=lambda r: (r['n'], r['scatter'], r['validation_loss'], r['name']))


def training_data(data, train):
    d = copy.copy(data)
    d.train, d.held = train, ~train
    d.t, d.ht = np.flatnonzero(train).astype(float), np.flatnonzero(~train).astype(float)
    d.y = data.y[:, train].copy()
    d.y -= d.y.mean(axis=1, keepdims=True)
    d.profile = d.group_norm @ d.y / np.sum(d.group_norm**2)
    d.base = data.base.copy()
    for ii in data.e.tiles(data.indices):
        d.base[ii] = data.e.raw[np.ix_(ii,train)].mean(axis=1,dtype=float)
    d.fluence_scale = max(d.noise_scale,float(np.max(d.profile)))
    return d


def discovery(data, train):
    d = training_data(data, train)
    profiles = []
    for splits in (1, 2, 4):
        for band in np.array_split(np.arange(len(d.y)), splits):
            plane = np.zeros(data.e.length)
            plane[train] = d.y[band].sum(axis=0)/np.sqrt(len(band))
            profiles.append(plane)
    candidates = []
    for profile in profiles:
        for width in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            if width > data.e.length/9:
                break
            norm = gaussian_filter1d(train.astype(float), float(width))
            trace = gaussian_filter1d(profile, float(width))/np.maximum(norm, .01)
            trace -= np.median(trace)
            trace /= max(1e-6, 1.4826*np.median(abs(trace)))
            peaks = np.unique(np.r_[find_peaks(trace)[0], trace.argmax()])
            for t in peaks[trace[peaks] >= POLICY['discovery_snr']]:
                candidates.append((float(trace[t]), int(t), width))
    regions = []
    for score, t, width in sorted(candidates, reverse=True):
        if all(abs(t-r[1]) > max(3, 2*max(width, r[2])) for r in regions):
            regions.append((score, t, width))
        if len(regions) == 3:
            break
    if not regions:
        # A broad, band-limited burst can raise profile MAD enough that no
        # smoothed trace crosses the discovery cut. Use the existing bounded
        # rectangular scan on TRAINING pixels, never the held samples.
        positive, _ = d.scan(d.y)
        if positive and positive.score >= POLICY['discovery_snr']:
            regions.append((positive.score, positive.start+(positive.width-1)/2,
                            max(.55, positive.width/2.5)))
    return regions


def residual_group(data, row, train):
    grouped = np.zeros_like(data.y)
    for ii in data.e.tiles(data.indices):
        p = prediction(data, row, ii)
        labels = data.lookup[ii]
        starts = np.r_[0, np.flatnonzero(np.diff(labels))+1]
        grouped[labels[starts]] += np.add.reduceat(p/data.var[ii, None], starts, axis=0)
    grouped /= data.group_norm[:, None]
    d = training_data(data, train)
    p = grouped[:, train]
    p -= p.mean(axis=1, keepdims=True)
    return d, d.y-p


def refit_fixed(data, row, *, train=None, iterations=20, seconds=20):
    """Refit fixed-count profiled geometry; useful for split-channel stability checks."""
    train = np.ones(data.e.length, bool) if train is None else train
    if row.get('model') != MODEL:
        d = training_data(data, train)
        result = optimizer(d,row['n'],row['scatter'],True,np.asarray(row['x']),
                           row['name'] if 'name' in row else 'smooth_refit',iterations,
                           time.monotonic()+seconds)
        return dict(result,model='smooth-spectrum-v1')
    return optimize(data, row, train, native=True, iterations=iterations,
                    deadline=time.monotonic()+seconds)


def fit(data, source_sha256):
    started = time.monotonic()
    deadline = started+POLICY['max_search_seconds']
    train = np.arange(data.e.length) % POLICY['cycle_bins'] < POLICY['train_bins_per_cycle']
    regions = discovery(data, train)
    rows, failures = [], []
    selection_cache={}
    def choose():
        return select(rows,data,train,selection_cache)

    def trial(seed):
        if time.monotonic() >= deadline:
            failures.append(dict(name=seed['name'], reason='search_wall_limit'))
            return None

        before = time.monotonic()
        try:
            grouped = optimize(data, seed, train, native=False,
                               iterations=POLICY['grouped_iterations'], deadline=deadline)
            row = optimize(data, grouped, train, native=True,
                           iterations=POLICY['native_iterations'], deadline=deadline)
            row.update(validate(data, row, train))
            row['seconds'] = time.monotonic()-before
            rows.append(row)
            if row['validation_gain'] <= 0 and time.monotonic() < deadline:
                # Weak channels cannot support a spectrum with hundreds of
                # independent coefficients. Reuse the original smooth-spectrum
                # fitter on training pixels at the SAME temporal count/family.
                fallback_start = time.monotonic()
                td = training_data(data,train)
                fallback = optimizer(td,row['n'],row['scatter'],True,np.asarray(row['x']),
                    row['name']+'_smooth',24,deadline)
                fallback = dict(fallback,model='smooth-spectrum-v1')
                fallback.update(validate(data,fallback,train))
                fallback['seconds'] = time.monotonic()-fallback_start
                rows.append(fallback)
            return row
        except (Limit, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append(dict(name=seed['name'], reason=type(exc).__name__+': '+str(exc)))
            return None

    def compare_shapes(seed):
        row = trial(seed)
        if seed['scatter']:
            x = np.asarray(seed['x'])[:-1].copy()
            x[1:5*seed['n']:5] += np.exp(seed['x'][-1])
            alternate = dict(seed, name=seed['name']+'_no_tail', scatter=False, x=x.tolist())
        else:
            x = np.asarray((row or seed)['x']).copy()
            tau = max(.5, float(np.median(np.exp(x[2:5*seed['n']:5])))*.7)
            x[1:5*seed['n']:5] -= tau
            alternate = dict(seed, name=seed['name']+'_tail', scatter=True,
                             x=np.r_[x, np.log(tau)].tolist())
        trial(alternate)

    if not regions:
        return dict(selected=None, trials=[], failures=failures, regions=[],
                    selection_review=dict(policy=POLICY, reason='no training discovery'),
                    residual_peak=None)
    # Compare intrinsic shape at the same count. Seed the broad-tail solution
    # separately so a scattering tail does not require an extra Gaussian.
    for count in range(1, len(regions)+1):
        parts = np.array([[0., r[1], np.log(max(.55, r[2])), 0., 0.] for r in regions[:count]])
        trial(dict(name=f'profile_{count}', n=count, scatter=False, x=np.r_[parts.ravel(), 0.].tolist()))
        tau = max(.5, np.median(np.exp(parts[:, 2]))*.7)
        parts[:, 1] -= tau
        trial(dict(name=f'profile_{count}_tail', n=count, scatter=True,
                   x=np.r_[parts.ravel(), 0., np.log(tau)].tolist()))
    stop = 'no_positive_validation_gain' if rows and choose() is None else 'no_finite_candidate'
    if rows and choose() is not None:
        incumbent = choose()
        # One residual expansion per count; stop after an unhelpful level.
        while incumbent['n'] < data.c['max_components'] and time.monotonic() < deadline:
            td, residual = residual_group(data, incumbent, train)
            pos, _ = td.scan(residual)
            if pos is None or pos.score < POLICY['discovery_snr']:
                stop = 'no_positive_training_residual'
                break
            n = incumbent['n']
            x = np.asarray(incumbent['x'])
            part = td.component_seed(pos)
            expanded = np.r_[x[:5*n], part, x[5*n:]]
            compare_shapes(dict(name=f'residual_{n+1}', n=n+1, scatter=incumbent['scatter'], x=expanded.tolist()))
            # A coincident subpulse need not create a second profile peak.
            pieces = x[:5*n].reshape(n, 5).copy()
            broad = int(np.argmax(pieces[:, 2]))
            extra = pieces[broad].copy()
            shift = .7*np.exp(extra[2])
            pieces[broad, 1] -= shift
            extra[1] += shift
            pieces[broad, 2] -= .45
            extra[2] -= .45
            split = np.r_[pieces.ravel(), extra, x[5*n:]]
            compare_shapes(dict(name=f'overlap_{n+1}', n=n+1, scatter=incumbent['scatter'], x=split.tolist()))
            chosen = choose()
            if chosen['n'] <= n:
                stop = 'extra_component_not_predictively_supported'
                break
            incumbent = chosen
        else:
            stop = 'component_guard' if incumbent['n'] >= data.c['max_components'] else 'search_wall_limit'
        chosen = choose()
        validation_winner = copy.deepcopy(chosen)
        final_start = time.monotonic()
        try:
            final = refit_fixed(data, chosen, iterations=POLICY['final_iterations'])
            final.update(validate(data, final, np.ones(data.e.length, bool)))
            # Preserve the selection evaluation, which precedes the full-data refit.
            final['selection_validation_loss'] = chosen['validation_loss']
            final['validation_split_block_losses'] = chosen['validation_split_block_losses']
            final['validation_loss'] = chosen['validation_loss']
            final['validation_gain'] = chosen['validation_gain']
            selected = final
        except (Limit, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append(dict(name='full_data_refit', reason=str(exc)))
            selected = None
        final_seconds = time.monotonic()-final_start
        if selected:
            td, residual = residual_group(data, selected, np.ones(data.e.length, bool))
            p, q = td.scan(residual)
            residual_peak = max(p.score if p else 0, q.score if q else 0)
        else:
            residual_peak = None
    else:
        selected, validation_winner, final_seconds, residual_peak = None, None, 0., None
    return dict(selected=selected, trials=rows, failures=failures, regions=regions,
                selection_review=dict(policy=POLICY, model=MODEL, stopping_reason=stop,
                    source_sha256=source_sha256, validation_winner=validation_winner,
                    training_time_bins=np.flatnonzero(train).tolist(),
                    validation_time_bins=np.flatnonzero(~train).tolist(),
                    screening_uses_full_data=True, physical_components_certified=False,
                    full_data_refit_seconds=final_seconds), residual_peak=residual_peak)
