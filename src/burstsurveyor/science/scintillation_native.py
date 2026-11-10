"""Verify profiled temporal geometry with the pinned upstream renderer."""
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .native import (_backend_receipt, _check_data, _components, _parameters,
                     _parity_reasons, _publish, VERSION)
from .scintillation import MODEL, prediction


def finalize_profiled(data, row, destination, source_sha256):
    started = time.perf_counter()
    path = Path(destination)
    if path.suffix.lower() != '.npz':
        raise ValueError('expected .npz destination')
    if len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('expected lowercase raw NPZ SHA-256')
    receipt = dict(version=VERSION, status='refused', certified=False,
                   source_sha256=source_sha256, reasons=[], prediction=None,
                   spectral_model=MODEL, physical_components_certified=False,
                   channel_screening=getattr(data, 'channel_screening', None))
    try:
        from fitburst.analysis.model import SpectrumModeler
        from fitburst.backend import general
        _check_data(data, row)
        channels = np.asarray(row['amplitude_channels'])
        a = np.asarray(row['channel_amplitudes'])
        if (not np.array_equal(channels, data.indices) or a.shape != (len(channels), row['n'])
                or not np.all(np.isfinite(a) & (a >= 0))):
            raise ValueError('invalid native profiled amplitudes')
        normal = bool(general['flags']['normalize_pbf'])
        parameters = _parameters(row, data.e.coords, normal)
        n = row['n']
        narrowest = np.exp(min(row['x'][2:5*n:5]))
        quadrature = min(64, max(8, int(np.ceil(3/max(.01, narrowest)))))
        time_s = data.t*data.e.coords.dt
        stored = np.full(data.e.raw.shape, np.nan, np.float32)
        baseline = np.full(len(data.e.raw), np.nan)
        gains, canonical, exported = np.zeros(2), np.zeros(2), np.zeros(2)
        energy = difference = 0.
        tile = max(1, min(16, 250000//(len(time_s)*n)))
        with threadpool_limits(1):
            for start in range(0, len(channels), tile):
                ii = channels[start:start+tile]
                parts = _components(SpectrumModeler, parameters, data.e.frequency[ii], time_s, n, quadrature)
                signal = np.einsum('ctn,cn->ct', parts, a[start:start+tile])
                stored[ii] = signal
                baseline[ii] = data.base[ii]-stored[ii].mean(axis=1, dtype=float)
                analytic = prediction(data, row, ii)
                y = data.e.raw[ii]-data.base[ii, None]
                analytic -= analytic.mean(axis=1, keepdims=True)
                signal -= signal.mean(axis=1, keepdims=True)
                replay = stored[ii].astype(float)
                replay -= replay.mean(axis=1, keepdims=True)
                for k, var in enumerate((data.var, data.rms_var)):
                    gains[k] += np.sum(signal*(2*y-signal)/var[ii, None])
                    canonical[k] += np.sum(analytic*(2*y-analytic)/var[ii, None])
                    exported[k] += np.sum(replay*(2*y-replay)/var[ii, None])
                energy += float(np.sum(analytic**2/data.var[ii, None]))
                difference += float(np.sum((analytic-signal)**2/data.var[ii, None]))
        score = dict(gain=float(gains[0]), rms_gain=float(gains[1]),
            canonical_gain=float(canonical[0]), canonical_rms_gain=float(canonical[1]),
            stored_gain=float(exported[0]), stored_rms_gain=float(exported[1]),
            relative_model_rmse=float(np.sqrt(difference/max(1e-12, energy))))
        reasons = _parity_reasons(score)
        if not np.isfinite(list(score.values())).all():
            reasons.append('nonfinite native score')
        for field in ('gain', 'rms_gain'):
            if not np.isclose(score['canonical_'+field], row[field], rtol=1e-8, atol=1e-6):
                reasons.append('replayed profiled canonical gain differs from winner')
        receipt.update(**_backend_receipt(SpectrumModeler), normalize_pbf=normal,
                       quadrature=quadrature, official_geometry_trials=0, native_model_passes=1,
                       official_incumbent=score, selected_score=score, reasons=reasons,
                       challenger_selected=False, selected=row,
                       fitburst_parameters=dict(parameters, spectral_model=MODEL),
                       spectral_parameters_note='Spectral index/running/amplitude are unit-profile placeholders; use channel_amplitudes with amplitude_channels.')
        if reasons:
            return dict(receipt, elapsed_s=time.perf_counter()-started)
    except (ImportError, ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as exc:
        return dict(receipt, reasons=[type(exc).__name__+': '+str(exc)], elapsed_s=time.perf_counter()-started)
    mask = np.zeros(len(data.e.raw), bool)
    mask[data.indices] = True
    arrays = dict(prediction=stored, baseline=baseline, variance=data.var,
        rms_variance=data.rms_var, channels=data.indices, channel_mask=mask,
        offburst_time_mask=data.noise_time, frequency=data.e.frequency, time_s=time_s,
        times_bin0=np.float64(data.e.grid['times_bin0']), freqs_bin0=np.float64(data.e.grid['freqs_bin0']),
        res_freq=np.float64(data.e.grid['res_freq']), res_time=np.float64(data.e.grid['res_time']),
        source_sha256=np.array(source_sha256), version=np.array(VERSION),
        spectral_model=np.array(MODEL), channel_amplitudes=a)
    digest = _publish(path, arrays)
    return dict(receipt, status='provisional', prediction=path.name, prediction_sha256=digest,
                elapsed_s=time.perf_counter()-started)
