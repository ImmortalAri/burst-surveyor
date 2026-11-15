"""Opt-in paired scattering search on the native pulse objective.

No reference parameters, event-specific rules, manual masks or forced nonzero
scattering. This isolates scattering/count competition; the kernel remains
periodic for one folded rotation, with scattering index -4. All candidates are retained.
"""
import time
import numpy as np

from .engine import Limit, optimizer
from .objective import score
from .proposals import choose_single_ceiling, parameter_key
from .pulse_coverage import catalogue, assess, residual_degradation

VERSION = 'pulsar-circular-2'
POLICY = dict(search_seconds=30., profile_seconds=20., grouped_iterations=24,
              native_iterations=8, profile_iterations=14, profile_native_iterations=4,
              confidence_delta=3.841458820694124, profile_points=9)


def _null_seed(row):
    x=np.asarray(row['x'],float).copy();n=row['n']
    if row['scatter']:
        tau=np.exp(x[-1]);x=x[:-1]
        x[1:5*n:5]+=tau
        x[2:5*n:5]=.5*np.log(np.exp(2*x[2:5*n:5])+tau*tau)
    return dict(n=n,scatter=False,x=x.tolist(),name='null_seed')


def _tail_seed(row,tau):
    seed=_null_seed(row);x=np.asarray(seed['x']);n=seed['n']
    sigma=np.exp(x[2:5*n:5]);x[1:5*n:5]-=tau
    x[2:5*n:5]=.5*np.log(np.maximum(.065**2,sigma*sigma-tau*tau))
    return dict(n=n,scatter=True,x=np.r_[x,np.log(tau)].tolist(),name='tail_seed')


def _merge(row, period=None):
    """Nominate one fewer shape term; fitting and coverage decide its adequacy."""
    n=row['n'];x=np.asarray(row['x'],float);parts=x[:5*n].reshape(n,5).copy()
    if n<2:raise ValueError('Cannot merge a single component')
    distance=np.full((n,n),np.inf)
    for i in range(n):
        for j in range(i+1,n):
            delta=parts[j,1]-parts[i,1]
            if period is not None:delta=(delta+period/2)%period-period/2
            distance[i,j]=abs(delta)/max(.065,np.hypot(np.exp(parts[i,2]),np.exp(parts[j,2])))
    i,j=np.unravel_index(np.argmin(distance),distance.shape)
    if period is not None:parts[j,1]=parts[i,1]+(parts[j,1]-parts[i,1]+period/2)%period-period/2
    loga=np.logaddexp(parts[i,0],parts[j,0]);w=np.exp(parts[[i,j],0]-loga)
    mu=float(w@parts[[i,j],1]);variance=float(w@(np.exp(2*parts[[i,j],2])+(parts[[i,j],1]-mu)**2))
    merged=w@parts[[i,j]];merged[:3]=[loga,mu,.5*np.log(variance)]
    parts[i]=merged;parts=np.delete(parts,j,axis=0)
    return dict(n=n-1,scatter=row['scatter'],x=np.r_[parts.ravel(),x[5*n:]].tolist(),name='merge_seed')


def _run(d,seed,name,deadline,*,native=False,iterations=None,fixed_tau=None):
    iterations=iterations or (POLICY['native_iterations'] if native else POLICY['grouped_iterations'])
    kwargs={} if fixed_tau is None else dict(fixed_log_tau=float(np.log(fixed_tau)))
    row=optimizer(d,seed['n'],seed['scatter'],native,np.asarray(seed['x'],float),name,iterations,deadline,**kwargs)
    row['gain'],row['rms_gain'],row['energy']=score(d,row)
    if not np.isfinite([row['gain'],row['rms_gain'],row['energy']]).all():raise ValueError('Nonfinite paired score')
    return row


def eligible(data,candidates,incumbent,features):
    """The existing practical ceiling cannot excuse a missing observed peak."""
    allowance=max(.02,.001*max(0.,incumbent['gain']))
    rms_allowance=max(.02,.001*max(0.,incumbent['rms_gain']))
    accepted=[];checks=[]
    for row in candidates:
        check=dict(name=row['name'],n=row['n'],scatter=row['scatter'])
        if row['gain']<incumbent['gain']-allowance or row['rms_gain']<incumbent['rms_gain']-rms_allowance:
            check.update(passed=False,reason='native_or_alternate_rms_degradation');checks.append(check);continue
        coverage=assess(data,row,features)
        check['coverage']=coverage
        if not coverage['passed']:
            check.update(passed=False,reason='missing_observed_peak');checks.append(check);continue
        if row['n']<incumbent['n']:
            residual=residual_degradation(data,incumbent,row);check['residual_guard']=residual
            if not residual['passed']:
                check.update(passed=False,reason='worsened_positive_residual');checks.append(check);continue
        check.update(passed=True,reason='native_and_peak_guards_passed');checks.append(check);accepted.append(row)
    return (choose_single_ceiling(accepted) if accepted else incumbent),checks


def summarize_profile(points,dt,*,complete,alternative=False):
    finite=[p for p in points if p.get('gain') is not None and np.isfinite(p['gain'])]
    result=dict(status='ambiguous',reference_mhz=600.,scattering_index=-4.,
                nominal_confidence=.95,delta_threshold=POLICY['confidence_delta'],points=points,
                meaning='Conditional coarse profile at fixed component count with nuisance refits and plug-in noise. Not calibrated physical confidence or a guaranteed upper limit.',
                search_complete=complete,unscattered_other_count_alternative=alternative)
    if not complete or len(finite)<4:return result|dict(reason='incomplete_profile')
    finite=sorted(finite,key=lambda p:p['tau_bins']);best=max(p['gain'] for p in finite)
    for p in finite:p['delta_loss']=max(0.,best-p['gain'])
    accepted=[i for i,p in enumerate(finite) if p['delta_loss']<=POLICY['confidence_delta']]
    if accepted!=list(range(min(accepted),max(accepted)+1)):
        return result|dict(reason='disconnected_profile_support')
    last=max(accepted)
    if last==len(finite)-1:return result|dict(reason='upper_bound_not_bracketed')
    a,b=finite[last:last+2];fraction=(POLICY['confidence_delta']-a['delta_loss'])/(b['delta_loss']-a['delta_loss'])
    upper=(a['tau_bins']+fraction*(b['tau_bins']-a['tau_bins'])) if a['tau_bins']==0 else np.exp(np.log(a['tau_bins'])+fraction*np.log(b['tau_bins']/a['tau_bins']))
    result['upper_ms']=float(upper*dt*1000)
    if finite[0]['tau_bins']!=0:return result|dict(reason='zero_model_unavailable')
    if finite[0]['delta_loss']<=POLICY['confidence_delta']:
        return result|dict(status='conditional_upper_limit',reason='zero_scattering_within_conditional_profile')
    first=min(accepted)
    if first==0 or alternative:return result|dict(reason='intrinsic_shape_scattering_ambiguity')
    a,b=finite[first-1:first+1];fraction=(POLICY['confidence_delta']-a['delta_loss'])/(b['delta_loss']-a['delta_loss'])
    lower=(a['tau_bins']+fraction*(b['tau_bins']-a['tau_bins'])) if a['tau_bins']==0 else np.exp(np.log(a['tau_bins'])+fraction*np.log(b['tau_bins']/a['tau_bins']))
    return result|dict(status='conditionally_constrained',reason='zero_excluded_at_fixed_count',lower_ms=float(lower*dt*1000))


def _profile(d,selected,pairs):
    start=time.monotonic();deadline=start+POLICY['profile_seconds'];n=selected['n']
    same=[r for r in pairs if r['n']==n]
    null=[r for r in same if not r['scatter']];tails=[r for r in same if r['scatter']]
    if not null or not tails:return dict(status='ambiguous',reason='paired_count_unavailable',points=[])
    null=max(null,key=lambda r:r['gain']);tail=max(tails,key=lambda r:r['gain'])
    reference=max(selected['gain'],tail['gain'],null['gain']);allowance=max(.02,.001*max(0.,reference))
    alternative=any(not r['scatter'] and r['n']!=n and r['gain']>=reference-allowance for r in pairs)
    lo,hi=d.bounds(n,True);minimum=np.exp(lo[-1]);maximum=np.exp(hi[-1]);optimum=float(np.exp(tail['x'][-1]))
    grid=np.unique(np.r_[np.geomspace(minimum,maximum,POLICY['profile_points']),optimum*.5,optimum,optimum*2])
    grid=grid[(grid>=minimum)&(grid<=maximum)]
    points=[dict(tau_bins=0.,gain=null['gain'],name=null['name'])];failures=[]
    # Visit the central geometry first, then its neighbors, while keeping every point.
    for tau in sorted(grid,key=lambda v:abs(np.log(v/optimum))):
        if time.monotonic()>=deadline:break
        try:
            seed=dict(tail);seed['x']=list(seed['x']);seed['x'][-1]=float(np.log(tau))
            group=_run(d,seed,f'pulsar_profile_{tau:.7g}',deadline,iterations=POLICY['profile_iterations'],fixed_tau=tau)
            row=_run(d,group,group['name']+'_native',deadline,native=True,iterations=POLICY['profile_native_iterations'],fixed_tau=tau)
            points.append(dict(tau_bins=float(tau),gain=row['gain'],rms_gain=row['rms_gain'],name=row['name'],termination=row['termination'],row=row))
        except (Limit,ValueError,np.linalg.LinAlgError) as exc:failures.append(dict(tau_bins=float(tau),reason=str(exc)))
    complete=len(points)==len(grid)+1 and not failures
    result=summarize_profile(points,d.e.coords.dt,complete=complete,alternative=alternative)
    # A fixed point improving materially on the free solve indicates unsettled geometry.
    best=max((p['gain'] for p in points),default=reference)
    if best>reference+POLICY['confidence_delta']:
        result.update(status='ambiguous',reason='profile_improves_on_free_geometry')
    result.update(component_count=n,failures=failures,elapsed_seconds=time.monotonic()-start,selected_tau_ms=float(np.exp(selected['x'][-1])*d.e.coords.dt*1000) if selected['scatter'] else 0.)
    return result


def refine(d,rows,incumbent):
    start=time.monotonic();deadline=start+POLICY['search_seconds'];original_budget=d.e.budget;original_native=d.native_work;original_evals=d.evaluations
    d.e.budget={'evaluations':0,'native_work':0};d.native_work=0;d.evaluations=0
    review=dict(version=VERSION,policy=POLICY,model='period-summed integrated kernel when period configured; fixed -4 scattering exponent',mask_changed=False,
                trials=[],failures=[],count_pairs=[],status='started',baseline=dict(incumbent),pre_simplification_full_row=dict(incumbent))
    pool=[incumbent];pairs=[]
    try:
        bank={}
        for row in [*rows,incumbent]:
            if row['n'] not in bank or row['gain']>bank[row['n']]['gain']:bank[row['n']]=row
        merged=dict(incumbent)
        while merged['n']>1:
            merged=_merge(merged, getattr(d.e.coords,"period_bins",None));bank.setdefault(merged['n'],merged)
        # Compare scattering at the selected count before spending the bounded
        # allowance on distant, usually inadequate smaller models.
        for n in sorted(bank, key=lambda n: (abs(n-incumbent["n"]), n)):
            if time.monotonic()>=deadline:break
            seed=bank[n];null_seed=_null_seed(seed)
            tau=max(.08,float(np.median(np.exp(np.asarray(null_seed['x'])[2:5*n:5])))*.7)
            starts=[null_seed,_tail_seed(null_seed,.08),_tail_seed(null_seed,min(tau,d.e.length*.25))]
            if seed['scatter']:starts.append(seed)
            grouped={False:[],True:[]};completed=[]
            for index,proposal in enumerate(starts):
                if time.monotonic()>=deadline:break
                try:
                    row=_run(d,proposal,f'pulsar_{n}_{"scatter" if proposal["scatter"] else "null"}_{index}',deadline)
                    review['trials'].append(row);grouped[row['scatter']].append(row)
                except (Limit,ValueError,np.linalg.LinAlgError) as exc:review['failures'].append(dict(n=n,scatter=proposal['scatter'],stage='grouped',reason=str(exc)))
            for state in (False,True):
                if not grouped[state] or time.monotonic()>=deadline:continue
                proposal=max(grouped[state],key=lambda r:r['gain'])
                try:
                    row=_run(d,proposal,proposal['name']+'_native',deadline,native=True)
                    review['trials'].append(row);pool.append(row);pairs.append(row);completed.append(state)
                except (Limit,ValueError,np.linalg.LinAlgError) as exc:review['failures'].append(dict(n=n,scatter=state,stage='native',reason=str(exc)))
            review['count_pairs'].append(dict(n=n,null_completed=False in completed,scattering_completed=True in completed))
        proposed=sorted(bank);done=sorted(r['n'] for r in review['count_pairs'] if r['null_completed'] and r['scattering_completed'])
        review.update(proposed_counts=proposed,paired_counts=done,search_complete=done==proposed)
        features=catalogue(d);selected,checks=eligible(d,pool,incumbent,features)
        review.update(guards=checks,observed_features=[{k:v for k,v in f.items() if k!='channels'} for f in features],selected_name=selected['name'],
                      selected_n=selected['n'],selected_scatter=selected['scatter'],status='paired_search_complete' if done==proposed else 'paired_search_incomplete')
        review['search_seconds']=time.monotonic()-start
        # Profiling has its own bounded allowance; counters remain visible in telemetry.
        for key,value in d.e.budget.items():original_budget[key]=original_budget.get(key,0)+value
        original_native+=d.native_work;original_evals+=d.evaluations
        review['search_work']=dict(d.e.budget)
        d.e.budget={'evaluations':0,'native_work':0};d.native_work=0;d.evaluations=0
        review['scattering']=_profile(d,selected,pairs)
        review['profile_work']=dict(d.e.budget)
        if not review['search_complete']:
            review['scattering'].update(status='ambiguous',reason='paired_search_incomplete')
        return selected,review
    finally:
        for key,value in d.e.budget.items():original_budget[key]=original_budget.get(key,0)+value
        d.e.budget=original_budget;d.native_work+=original_native;d.evaluations+=original_evals
        review['elapsed_seconds']=time.monotonic()-start
