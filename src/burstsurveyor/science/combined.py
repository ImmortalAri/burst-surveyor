"""Bounded, coverage-constrained selection across spectral resolutions.

Human/reference parameters are never inputs. All geometry, amplitudes and
baseline fitting precede held-time evaluation. Screening is conditional.
"""
import copy,time
import numpy as np
from .engine import Limit,optimizer
from .scintillation import (MODEL,optimize,validate,select,training_data,
    discovery,residual_group,refit_fixed,shapes)
from .pulse_coverage import catalogue,assess,residual_degradation

POLICY=dict(max_search_seconds=60.,spectral_groups=(16,1),smooth_iterations=24,
            mixed_new_component_group_size=64,
            simplification_seconds=20.,simplification_iterations=20,
            grouped_iterations=12,native_iterations=16,final_iterations=24,
            cycle_bins=6,train_bins=4,discovery_snr=6.,max_full_refits=3,
            acceptance='positive predictive gain AND replicated peak coverage')


def eligible_select(rows,data,train):
    adequate=[r for r in rows if r.get('coverage',{}).get('passed') and
              not r.get('background_components') and r.get('pruning_guard',{}).get('passed',True) and r['validation_gain']>0]
    if not adequate:return None
    # The existing conditional-noise one-SE parsimony rule is applied only
    # inside the adequate set. Fewer components cannot trump a missing peak.
    row=select(adequate,data,train)
    return row


def background_components(data,row):
    """Do not silently count broad off-burst baseline trends as burst terms."""
    x=np.asarray(row['x']);n=row['n']
    broad=np.flatnonzero(np.exp(x[2:5*n:5])>=data.e.length/8)
    if not len(broad):return []
    energy=np.zeros(n);off=np.zeros(n)
    for ii in data.e.tiles(data.indices):
        h,_=shapes(x,n,row['scatter'],data.e.coords,ii,data.t)
        if row.get('model')==MODEL:
            lookup=np.full(len(data.e.raw),-1,int);lookup[row['amplitude_channels']]=np.arange(len(row['amplitude_channels']))
            a=np.asarray(row['channel_amplitudes'])[lookup[ii]]
        else:
            p=x[:5*n].reshape(n,5);z=data.e.coords.z[ii,None]
            a=np.exp(p[None,:,0]+z*p[None,:,3]+z*z*p[None,:,4])
        h*=a[:,None,:]
        # Uncentered squared signal distinguishes a crop-long baseline term
        # from a burst and its tail. Ambiguous broad emission stays unresolved.
        e=h*h/data.var[ii,None,None]
        energy+=e.sum(axis=(0,1));off+=e[:,data.noise_time].sum(axis=(0,1))
    return [dict(component=int(j+1),offburst_energy_fraction=float(off[j]/max(energy[j],1e-200)))
            for j in broad if off[j]/max(energy[j],1e-200)>.8]


def simplify_training(data,rows,train,features,*,seconds=None):
    """Backward counterfactuals from adequate training candidates, never humans.

    Forward selection can leave a lower-count candidate in a poorer local
    optimum. Removing each term from the current winner gives simpler models
    a fair warm start. Shared original work/evaluation guards are not reset.
    """
    started=time.monotonic();deadline=started+(POLICY['simplification_seconds'] if seconds is None else seconds)
    initial=eligible_select(rows,data,train);best=initial;trials=[];failures=[]
    stop='no_adequate_candidate' if best is None else 'already_single'
    td=training_data(data,train)
    while best and best['n']>1 and time.monotonic()<deadline:
        n=best['n'];x=np.asarray(best['x']);parts=x[:5*n].reshape(n,5)
        for drop in range(n):
            if time.monotonic()>=deadline:break
            before=time.monotonic();name=f'prune_{n-1}_drop{drop+1}_trial{len(trials)+1}'
            seed=dict(name=name,n=n-1,scatter=best['scatter'],
                x=np.r_[np.delete(parts,drop,axis=0).ravel(),x[5*n:]].tolist())
            if best.get('spectral_group_sizes'):
                seed['spectral_group_sizes']=np.delete(best['spectral_group_sizes'],drop).tolist()
            elif 'spectral_group_size' in best:seed['spectral_group_size']=best['spectral_group_size']
            try:
                if best.get('model')==MODEL:
                    candidate=optimize(data,seed,train,native=True,
                        iterations=POLICY['simplification_iterations'],deadline=deadline)
                else:
                    candidate=optimizer(td,n-1,best['scatter'],True,np.asarray(seed['x']),name,
                        POLICY['simplification_iterations'],deadline)
                    candidate['model']='smooth-spectrum-v1'
                candidate.update(validate(data,candidate,train))
                candidate['coverage']=assess(data,candidate,features,train)
                candidate['background_components']=background_components(data,candidate)
                if candidate['coverage']['passed'] and candidate['validation_gain']>0:
                    candidate['pruning_guard']=residual_degradation(data,best,candidate)
                candidate['seconds']=time.monotonic()-before
                candidate['pruning_phase']=True
                rows.append(candidate);trials.append(name)
            except (Limit,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures.append(dict(name=name,reason=str(exc)))
        chosen=eligible_select(rows,data,train)
        if chosen is None or chosen['n']>=n:
            stop='no_predictively_supported_deletion';break
        best=chosen;stop='simplified'
    if best and best['n']>1 and time.monotonic()>=deadline:stop='simplification_wall_limit'
    return dict(initial_count=initial['n'] if initial else None,selected_count=best['n'] if best else None,
                trial_names=trials,failures=failures,stopping_reason=stop,seconds=time.monotonic()-started)


def simplify_record(data,record):
    """One permitted backward follow-up using a sealed forward receipt.

    The caller verifies its parent seal, raw hash and grid. Reuses saved
    TRAINING candidates, never the full-data winner for held-time scoring.
    """
    if (record.get('mode')!='combined' or record.get('simplification_review') or
            record.get('selection_review',{}).get('simplification_review')):
        raise ValueError('Expected a combined forward receipt without a prior simplification pass')
    result=copy.deepcopy(record);started=time.monotonic()
    if not result.get('selected'):
        result['simplification_review']=dict(stopping_reason='unresolved_forward_fit',changed=False,seconds=0.)
        return result
    data.e.budget.update(record['work'])
    train=np.arange(data.e.length)%POLICY['cycle_bins']<POLICY['train_bins']
    expected=record['selection_review']['training_time_bins']
    if np.flatnonzero(train).tolist()!=expected:raise ValueError('Saved training split changed')
    features=catalogue(data,train)
    review=simplify_training(data,result['trials'],train,features)
    chosen=eligible_select(result['trials'],data,train)
    review['changed']=False
    if chosen and chosen['n']<record['selected']['n']:
        before=time.monotonic()
        try:
            final=refit_fixed(data,chosen,iterations=POLICY['final_iterations'],seconds=20.)
            final.update(validate(data,final,np.ones(data.e.length,bool)))
            final['coverage']=assess(data,final,catalogue(data))
            final['background_components']=background_components(data,final)
            final['pruning_guard']=residual_degradation(data,record['selected'],final)
            if final['coverage']['passed'] and final['pruning_guard']['passed'] and not final['background_components'] and final['gain']>0 and final['rms_gain']>0:
                final['selection_validation_loss']=chosen['validation_loss']
                final['validation_loss']=chosen['validation_loss'];final['validation_gain']=chosen['validation_gain']
                result['selected']=final;review['changed']=True
                result['selection_review']['pre_simplification_full_row']=copy.deepcopy(record['selected'])
                result['selection_review']['validation_winner']=copy.deepcopy(chosen)
                rd,residual=residual_group(data,final,np.ones(data.e.length,bool))
                p,q=rd.scan(residual);result['residual_peak']=max(p.score if p else 0,q.score if q else 0)
            else:review['full_refit_rejection']=dict(coverage=final['coverage'],background=final['background_components'],pruning_guard=final['pruning_guard'])
        except (Limit,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            review['full_refit_rejection']=str(exc)
        review['full_refit_seconds']=time.monotonic()-before
        result['selection_review']['full_data_refit_seconds']+=review['full_refit_seconds']
    review['seconds']=time.monotonic()-started
    result['simplification_review']=review;result['work']=dict(data.e.budget)
    return result


def fit(data,source_sha256):
    started=time.monotonic();deadline=started+POLICY['max_search_seconds']
    train=np.arange(data.e.length)%POLICY['cycle_bins']<POLICY['train_bins']
    features=catalogue(data,train)
    td=training_data(data,train)
    regions=discovery(data,train)
    rows=[];failures=[];proposals=[];stop='no_finite_candidate'
    def review(row,before):
        row.update(validate(data,row,train))
        row['coverage']=assess(data,row,features,train)
        row['background_components']=background_components(data,row)
        row['seconds']=time.monotonic()-before
        row['validation_gain_positive']=row['validation_gain']>0
        rows.append(row)
        return row

    def trials(seed,label):
        # Treat both intrinsic shape families at the SAME count, and always
        # try smooth spectra, even when a bright main pulse validates well.
        seeds=[dict(seed,name=label)]
        if seed['scatter']:
            x=np.asarray(seed['x'])[:-1].copy()
            x[1:5*seed['n']:5]+=np.exp(seed['x'][-1])
            seeds.append(dict(seed,scatter=False,x=x.tolist(),name=label+'_gaussian'))
        else:
            x=np.asarray(seed['x']).copy()
            tau=max(.5,float(np.median(np.exp(x[2:5*seed['n']:5])))*.7)
            x[1:5*seed['n']:5]-=tau
            seeds.append(dict(seed,scatter=True,x=np.r_[x,np.log(tau)].tolist(),name=label+'_scattered'))
        for shape in seeds:
            if time.monotonic()>=deadline:break
            before=time.monotonic()
            try:
                smooth=optimizer(td,shape['n'],shape['scatter'],True,np.asarray(shape['x']),
                    shape['name']+'_smooth',POLICY['smooth_iterations'],deadline)
                review(dict(smooth,model='smooth-spectrum-v1'),before)
            except (Limit,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures.append(dict(name=shape['name']+'_smooth',reason=str(exc)))
            if seed.get('spectral_group_sizes') and time.monotonic()<deadline:
                before=time.monotonic()
                try:
                    mixed=optimize(data,dict(shape,name=shape['name']+'_mixed'),train,
                        native=True,iterations=24,deadline=deadline)
                    review(mixed,before)
                except (Limit,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    failures.append(dict(name=shape['name']+'_mixed',reason=str(exc)))
            # Pooling borrows strength across a small frequency interval while
            # leaving all native pixels, masks and noise weights unchanged.
            last=None
            for group_size in POLICY['spectral_groups']:
                if time.monotonic()>=deadline:break
                before=time.monotonic()
                try:
                    start=dict(shape,name=shape['name']+f'_spectral{group_size}',spectral_group_size=group_size)
                    start.pop('spectral_group_sizes',None)
                    if last is not None:start['x']=last['x']
                    else:
                        grouped=optimize(data,start,train,native=False,
                            iterations=POLICY['grouped_iterations'],deadline=deadline)
                        start['x']=grouped['x']
                    last=optimize(data,start,train,native=True,
                        iterations=POLICY['native_iterations'],deadline=deadline)
                    review(last,before)
                except (Limit,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    failures.append(dict(name=shape['name']+f'_spectral{group_size}',reason=str(exc)))

    def part(feature):
        return [0.,float(feature['time_bin']),float(np.log(feature['sigma_bins'])),0.,0.]
    if features:
        initial=part(features[0])
    elif regions:
        initial=[0.,float(regions[0][1]),float(np.log(max(.55,regions[0][2]))),0.,0.]
    else:
        pos,_=td.scan(td.y)
        initial=list(td.component_seed(pos)) if pos and pos.score>=POLICY['discovery_snr'] else None
    if initial is None:
        return dict(selected=None,trials=[],failures=[],regions=[],residual_peak=None,
                    selection_review=dict(policy=POLICY,stopping_reason='no_training_discovery'))
    trials(dict(n=1,scatter=False,x=initial+[0.]),'initial_1')
    # A bounded expansion at each count, seeded by an observed missing peak
    # before a residual rectangle. No repeated component accumulation after a
    # failed level; limits leave an explicit unresolved result.
    while rows and time.monotonic()<deadline:
        incumbent=eligible_select(rows,data,train)
        if incumbent is None:
            positive=[r for r in rows if r['validation_gain']>0]
            if not positive:stop='no_positive_validation_gain';break
            incumbent=min(positive,key=lambda r:(len(r['coverage']['missing']),r['validation_loss']))
        n=incumbent['n']
        if n>=data.c['max_components']:stop='component_guard';break
        missing=incumbent['coverage']['missing']
        rd,residual=residual_group(data,incumbent,train)
        pos,_=rd.scan(residual)
        proposal_type='missing_peak' if missing else 'residual'
        if not missing:
            # A very bright pulse's small shape error can dominate a weaker
            # separated pulse. Search outside explained emission independently
            # (discovery only); never zero or reweight fitting observations.
            predicted=rd.y-residual
            predicted-=np.median(predicted,axis=1,keepdims=True)
            explained=np.linalg.norm(np.maximum(predicted,0),axis=0)>3.
            separated=residual.copy();separated[:,explained]=0.
            distinct,_=rd.scan(separated)
            if distinct and distinct.score>=POLICY['discovery_snr']:
                pos=distinct;proposal_type='separated_residual'
        if not missing and (pos is None or pos.score<POLICY['discovery_snr']):
            stop='covered_no_positive_training_residual';break
        if missing:
            feature=missing[0]
            extra=part(dict(feature,sigma_bins=max(.55,feature['sigma_bins'])))
        else:extra=list(rd.component_seed(pos))
        proposals.append(dict(count=n+1,type=proposal_type,
            time_bin=float(extra[1]),template=pos.record() if pos else None))
        x=np.asarray(incumbent['x'])
        expanded=np.r_[x[:5*n],extra,x[5*n:]]
        before_count=len(rows)
        widths=incumbent.get('spectral_group_sizes',[incumbent.get('spectral_group_size',1)]*n)
        trials(dict(n=n+1,scatter=incumbent['scatter'],x=expanded.tolist(),
                    spectral_group_sizes=list(widths)+[POLICY['mixed_new_component_group_size']]),f'expand_{n+1}')
        chosen=eligible_select(rows,data,train)
        if chosen is not None and chosen['n']<=n:
            stop='covered_extra_level_not_predictively_supported';break
        if len(rows)==before_count:stop='resource_guard';break
        if chosen is None:
            later=[r for r in rows[before_count:] if r['validation_gain']>0]
            if not later or min(len(r['coverage']['missing']) for r in later)>=len(missing):
                stop='coverage_not_resolved_by_one_expansion';break
    else:
        if rows:stop='search_wall_limit'
    before_simplification=copy.deepcopy(eligible_select(rows,data,train)) if rows else None
    simplification=simplify_training(data,rows,train,features) if rows else None
    chosen=eligible_select(rows,data,train) if rows else None
    rejected_best=min(rows,key=lambda r:r['validation_loss']) if rows else None
    selected=None;refits=[];final_start=time.monotonic()
    # Full-data optimization may regress morphology; recheck after refitting.
    # Only a bounded number of previously validated adequate candidates may
    # serve as fallbacks. Failed candidates remain in receipts for review.
    candidates=[r for r in rows if r['coverage']['passed'] and not r.get('background_components') and
                r.get('pruning_guard',{}).get('passed',True) and r['validation_gain']>0]
    ordered=([chosen] if chosen else [])+sorted((r for r in candidates if r is not chosen),
        key=lambda r:(r['n'],r['validation_loss']))
    full_features=catalogue(data)
    for candidate in ordered[:POLICY['max_full_refits']]:
        before=time.monotonic()
        try:
            final=refit_fixed(data,candidate,iterations=POLICY['final_iterations'],seconds=20.)
            final.update(validate(data,final,np.ones(data.e.length,bool)))
            coverage=assess(data,final,full_features)
            final['coverage']=coverage
            final['background_components']=background_components(data,final)
            if before_simplification and final['n']<before_simplification['n']:
                final['pruning_guard']=residual_degradation(data,before_simplification,final)
            final['selection_validation_loss']=candidate['validation_loss']
            final['validation_loss']=candidate['validation_loss']
            final['validation_gain']=candidate['validation_gain']
            refits.append(dict(name=candidate['name'],seconds=time.monotonic()-before,
                               coverage=coverage,termination=final['termination']))
            if coverage['passed'] and final.get('pruning_guard',{}).get('passed',True) and not final['background_components'] and final['gain']>0 and final['rms_gain']>0:
                selected=final;chosen=candidate;break
        except (Limit,ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            failures.append(dict(name='full_'+candidate['name'],reason=str(exc)))
    if selected:
        rd,residual=residual_group(data,selected,np.ones(data.e.length,bool))
        p,q=rd.scan(residual);residual_peak=max(p.score if p else 0,q.score if q else 0)
    else:residual_peak=None
    return dict(selected=selected,trials=rows,failures=failures,regions=regions,residual_peak=residual_peak,
        selection_review=dict(policy=POLICY,stopping_reason=stop,source_sha256=source_sha256,
            validation_winner=copy.deepcopy(chosen),best_predictive_candidate=copy.deepcopy(rejected_best),
            training_features=[{k:v for k,v in f.items() if k!='channels'} for f in features],
            final_full_data_refits=refits,full_data_refit_seconds=time.monotonic()-final_start,
            proposals=proposals,
            training_time_bins=np.flatnonzero(train).tolist(),validation_time_bins=np.flatnonzero(~train).tolist(),
            screening_uses_full_data=True,physical_components_certified=False,
            acceptance_guard_passed=selected is not None,
            simplification_review=simplification,
            pre_simplification_winner=before_simplification,
            unresolved_reason=None if selected else 'No validated candidate passed the final replicated peak-coverage guard'))
