"""Replicated prominence check: a lower error cannot excuse a missing peak.

This checks observable morphology, not physical component count. Channel folds,
frequency correlation inflation and an off-burst empirical noise scale reduce
sensitivity to individual channels. Full-data screening makes it conditional.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

POLICY = dict(discovery_snr=6.,fold_snr=3.,missing_fraction=.5,
              scales=(1,2,4,8,16,32,64,128,256),max_features=24)


def _smooth(y,mask,sigma):
    norm=gaussian_filter1d(mask.astype(float),sigma,mode='nearest')
    return gaussian_filter1d(y*mask,sigma,mode='nearest')/np.maximum(norm,1e-12)


def catalogue(data, mask=None):
    mask=np.ones(data.e.length,bool) if mask is None else np.asarray(mask,bool)
    features=[]
    noise_mask=np.asarray(data.noise_time)&mask
    inflation=np.sqrt((1+data.rho_freq)/max(1e-6,1-data.rho_freq))
    for splits in (1,2,4):
        for band in np.array_split(data.indices,splits):
            if len(band)<8:continue
            folds=[band[::2],band[1::2]]
            profiles=[]; theoretical=[]
            for ii in folds:
                w=1/data.var[ii]
                y=data.e.raw[ii].astype(float)
                y-=y[:,mask].mean(axis=1,keepdims=True)
                profiles.append(w@y/w.sum())
                theoretical.append(np.sqrt(1/w.sum())*inflation)
            for sigma in POLICY['scales']:
                if sigma>max(1,data.e.length/16):break
                smooth=[_smooth(p,mask,sigma) for p in profiles]
                noise=[]
                for p,floor in zip(smooth,theoretical):
                    off=p[noise_mask]
                    empirical=1.4826*np.median(abs(off-np.median(off))) if len(off)>12 else floor
                    # Native sample variance provides a conservative floor;
                    # do not claim independent samples after smoothing.
                    noise.append(max(empirical,floor/np.sqrt(max(1.,sigma))))
                avg=(smooth[0]+smooth[1])/2
                scale=np.hypot(*noise)/2
                peaks,props=find_peaks(avg,prominence=POLICY['discovery_snr']*scale,
                    distance=max(2,sigma),wlen=max(9,8*sigma+1))
                for k,peak in enumerate(peaks):
                    lo=int(props['left_bases'][k]);hi=int(props['right_bases'][k])
                    if peak-lo<1 or hi-peak<1:continue
                    # The SAME peak and neighboring troughs must recur in both folds.
                    contrast=[p[peak]-max(p[lo],p[hi]) for p in smooth]
                    z=np.asarray(contrast)/noise
                    if z.min()<POLICY['fold_snr']:continue
                    prominence=float(avg[peak]-max(avg[lo],avg[hi]))
                    width=float(peak_widths(avg,[peak],rel_height=.5)[0][0])/2.355
                    features.append(dict(time_bin=int(peak),sigma_bins=max(.55,width),
                        smooth_sigma=int(sigma),time_bins=[lo,hi+1],
                        channel_bins=[int(band.min()),int(band.max())+1],
                        channels=band.tolist(),contrast=prominence,noise=float(scale),
                        snr=prominence/scale,fold_snr=z.tolist()))
    # Retain localized evidence from any band. Duplicates across scales are
    # suppressed only at the same time, never by the brightest pulse's height.
    kept=[]
    for f in sorted(features,key=lambda x:-x['snr']):
        if any(abs(f['time_bin']-g['time_bin'])<=max(2,min(f['smooth_sigma'],g['smooth_sigma']))
               for g in kept):continue
        kept.append(f)
        if len(kept)==POLICY['max_features']:break
    return kept


def assess(data,row,features,mask=None,*,pixels=None):
    from .scintillation import prediction
    mask=np.ones(data.e.length,bool) if mask is None else np.asarray(mask,bool)
    missing=[];evidence=[];profiles={}
    for f in features:
        band=np.asarray(f['channels']); w=1/data.var[band]
        key=tuple(f['channel_bins'])
        if key not in profiles:
            profile=np.zeros(data.e.length)
            for start in range(0,len(band),16):
                ii=band[start:start+16]
                signal=prediction(data,row,ii) if pixels is None else pixels[ii]
                profile+=(signal/data.var[ii,None]).sum(axis=0)
            profiles[key]=profile/w.sum()
        profile=profiles[key]
        p=_smooth(profile,mask,f['smooth_sigma'])
        peak=f['time_bin'];lo,hi=f['time_bins'];hi-=1
        # Compare the same local peak/trough contrast; amplitude mismatch in a
        # correctly shaped bright pulse does not automatically create a pulse.
        contrast=float(p[peak]-max(p[lo],p[hi]))
        omission=f['contrast']-contrast
        failed=(contrast<POLICY['missing_fraction']*f['contrast'] and
                omission/f['noise']>=POLICY['discovery_snr'])
        item={k:v for k,v in f.items() if k!='channels'}
        item.update(model_contrast=contrast,missing_contrast_snr=omission/f['noise'],missing=bool(failed))
        evidence.append(item)
        if failed:missing.append(item)
    return dict(passed=not missing,missing=missing,evidence=evidence,policy=POLICY,
                interpretation='conditional replicated peak coverage; not physical pulse certification')


def assess_pixels(data,pixels):
    """Check the actual stored native reconstruction, not a surrogate renderer."""
    pixels=np.asarray(pixels)
    if pixels.shape!=data.e.raw.shape or not np.isfinite(pixels[data.indices]).all():
        raise ValueError('Native coverage requires finite predictions on the exact observation grid')
    return assess(data,None,catalogue(data),pixels=pixels)


def residual_degradation(data,parent,candidate):
    """Veto simplifications that introduce/worsen a replicated residual island.

    Reuse the package's independent residual discovery/check-channel diagnostic
    in addition to raw-profile prominence. Parent/candidate may be stored pixels
    or fitted rows; a tiled view avoids observation-sized surrogate allocation.
    """
    from ..secondlook import _discovery,_evidence,POLICY as RESIDUAL_POLICY
    from .scintillation import prediction
    class View:
        def __init__(self,row):self.row=row
        def __getitem__(self,ii):return prediction(data,self.row,ii)
    def model(value):return value if isinstance(value,np.ndarray) else View(value)
    left,right=data.indices[::2],data.indices[1::2]
    if min(len(left),len(right))<6:
        return dict(passed=True,status='insufficient_channels_for_residual_comparison')
    proposed=model(candidate);previous=model(parent)
    packet=_discovery(data,left,proposed,dict(id='simplification',source_sha256='0'*64))
    if packet is None:return dict(passed=True,status='no_positive_residual_proposal')
    check=_evidence(data,right,proposed,packet)
    old=_evidence(data,right,previous,packet)
    floor=RESIDUAL_POLICY['positive_check']
    worsening=(check is not None and old is not None and check>=floor and check-old>=floor)
    return dict(passed=not worsening,status='worsened_replicated_residual' if worsening else 'not_materially_worsened',
                time_bins=packet['time_bins'],channel_bins=packet['channel_bins'],
                parent_check_score=old,candidate_check_score=check,check_score_increase=check-old if old is not None and check is not None else None)
