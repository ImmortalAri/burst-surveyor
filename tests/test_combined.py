import time,unittest
import numpy as np
from threadpoolctl import threadpool_limits
from test_scintillation import fixture
from burstsurveyor.science.scintillation import shapes,Objective,optimize
from burstsurveyor.science.combined import fit,eligible_select
from burstsurveyor.science.pulse_coverage import catalogue,assess


class CombinedTests(unittest.TestCase):
    def setUp(self):self.threads=threadpool_limits(1)
    def tearDown(self):self.threads.restore_original_limits()

    def test_resolved_double_peak_cannot_be_merged(self):
        d,x=fixture([(60.,2.,150.),(80.,2.,130.)],seed=4,scatter=False,nfreq=64)
        merged=dict(n=1,scatter=False,x=[np.log(280.),70.,np.log(12.),0.,0.,x[-1]])
        features=catalogue(d)
        self.assertGreaterEqual(len(features),2)
        self.assertFalse(assess(d,merged,features)['passed'])

    def test_weak_separated_pulse_not_hidden_by_strong_main(self):
        d,x=fixture([(55.,2.,150.),(120.,3.,20.)],seed=5,scatter=False,nfreq=64)
        main=dict(n=1,scatter=False,x=np.r_[x[:5],x[-1]].tolist())
        self.assertFalse(assess(d,main,catalogue(d))['passed'])
        result=fit(d,'a'*64)
        self.assertIsNotNone(result['selected'],result['selection_review']['stopping_reason'])
        self.assertEqual(result['selected']['n'],2)

    def test_single_scintillating_control_stays_single(self):
        d,_=fixture([(70.,3.,90.)],seed=42,nfreq=64)
        result=fit(d,'a'*64)
        self.assertIsNotNone(result['selected'])
        self.assertEqual(result['selected']['n'],1)
        self.assertTrue(result['selected']['coverage']['passed'])

    def test_one_bad_channel_cannot_define_replicated_peak(self):
        d,_=fixture([(70.,3.,90.)],seed=42,nfreq=64)
        d.e.raw[10,130]+=5000.
        features=catalogue(d)
        self.assertFalse(any(abs(f['time_bin']-130)<4 for f in features))

    def test_pooled_objective_gradient_matches_finite_difference(self):
        d,x=fixture([(65.,3.,70.)],seed=7,nfreq=32)
        train=np.arange(d.e.length)%6<4
        objective=Objective(d,train,native=True,spectral_group_size=16)
        _,g,_,_=objective.evaluate(x,1,True,derivative=True)
        for j,k in enumerate([1,2,5,6]):
            step=1e-4;left=x.copy();right=x.copy();left[k]-=step;right[k]+=step
            numeric=(objective.evaluate(left,1,True)[0]-objective.evaluate(right,1,True)[0])/(4*step)
            self.assertAlmostEqual(g[j],numeric,delta=max(.05,abs(numeric)*.002))

    def test_inadequate_lower_error_is_ineligible(self):
        row=dict(n=1,validation_gain=100.,coverage=dict(passed=False),validation_loss=0.)
        self.assertIsNone(eligible_select([row],None,None))

    def test_pooled_coefficients_are_shared_only_inside_native_bins(self):
        d,x=fixture([(70.,3.,90.)],seed=4,nfreq=32)
        row=optimize(d,dict(n=1,scatter=True,x=x.tolist(),name='pool',spectral_group_size=16),
            np.ones(d.e.length,bool),native=True,iterations=2,deadline=time.monotonic()+5)
        a=np.asarray(row['channel_amplitudes'])
        np.testing.assert_allclose(a[:16],np.repeat(a[:1],16,axis=0))
        np.testing.assert_allclose(a[16:],np.repeat(a[16:17],16,axis=0))

    def test_residual_seed_accepts_package_python_list(self):
        from unittest.mock import patch
        from burstsurveyor.science.scintillation import residual_group as original
        d,_=fixture([(70.,3.,90.)],seed=42,nfreq=32)
        def force_proposal(data,row,train):
            rd,r=original(data,row,train)
            proposal,_=rd.scan(rd.y)
            rd.scan=lambda unused:(proposal,None)
            return rd,r
        with patch('burstsurveyor.science.combined.residual_group',side_effect=force_proposal):
            result=fit(d,'a'*64)
        self.assertIsNotNone(result['selected'])
        self.assertTrue(any(r['name'].startswith('expand_2') for r in result['trials']))

    def test_failed_final_coverage_cannot_export_training_winner(self):
        from unittest.mock import patch
        from burstsurveyor.science.scintillation import refit_fixed
        d,_=fixture([(70.,3.,90.)],seed=42,nfreq=32)
        def bad_refit(data,row,**kwargs):
            final=refit_fixed(data,row,**kwargs)
            final['x'][1]+=60
            return final
        with patch('burstsurveyor.science.combined.refit_fixed',side_effect=bad_refit):
            result=fit(d,'a'*64)
        self.assertIsNone(result['selected'])
        self.assertFalse(result['selection_review']['acceptance_guard_passed'])

    def test_mixed_spectral_gradient_matches_finite_difference(self):
        d,x=fixture([(55.,3.,90.),(105.,4.,20.)],seed=7,nfreq=32)
        train=np.arange(d.e.length)%6<4
        objective=Objective(d,train,native=True,spectral_group_sizes=[1,16])
        _,g,_,a=objective.evaluate(x,2,True,derivative=True,keep=True)
        np.testing.assert_allclose(a[:16,1],np.repeat(a[:1,1],16))
        for j,k in enumerate([1,2,6,7,10,11]):
            step=1e-4;left=x.copy();right=x.copy();left[k]-=step;right[k]+=step
            numeric=(objective.evaluate(left,2,True)[0]-objective.evaluate(right,2,True)[0])/(4*step)
            self.assertAlmostEqual(g[j],numeric,delta=max(.05,abs(numeric)*.002))

    def _native_gate(self,omit):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from burstsurveyor.science.fit import load_event,export_event
        d,_=fixture([(55.,2.,150.),(110.,2.,100.)],seed=4,nfreq=32,scatter=False)
        noise=np.random.default_rng(4).normal(size=d.e.raw.shape)
        actual=d.e.raw-noise
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'test.npz'
            np.savez(source,data_full=d.e.raw,metadata=json.dumps(d.e.grid))
            data,entry=load_event(source,tmp/'load')
            artifact=tmp/'native.npz'
            def fake_finalize(data,selected,path,**kwargs):
                signal=np.zeros_like(actual) if omit else actual
                np.savez(path,prediction=signal,source_sha256=np.array(entry['source_sha256']),
                         channels=data.indices,frequency=data.e.frequency,time_s=data.t*data.e.coords.dt)
                return dict(status='provisional',reasons=[],prediction=path.name,prediction_sha256='a'*64)
            primary=dict(mode='combined',selected={},source_sha256=entry['source_sha256'],grid=entry['grid'])
            with patch('burstsurveyor.science.fit.finalize',side_effect=fake_finalize):
                result=export_event(source,primary,artifact,tmp/'export')
            return result,artifact.exists()

    def test_actual_native_omission_cannot_be_published(self):
        result,exists=self._native_gate(True)
        self.assertEqual(result['status'],'unresolved')
        self.assertFalse(exists)
        self.assertTrue(result['prediction_rejected'])

    def test_actual_native_adequate_pixels_survive(self):
        result,exists=self._native_gate(False)
        self.assertEqual(result['status'],'provisional')
        self.assertTrue(exists)
        self.assertTrue(result['native_peak_coverage']['passed'])

    def test_combined_batch_seals_actual_native_gate(self):
        import json,tempfile
        from pathlib import Path
        from burstsurveyor.batch import fit as batch_fit,verify
        from burstsurveyor import _dispatch
        if _dispatch.psutil.virtual_memory().available<900*1024**2:
            self.skipTest('Insufficient available RAM for guarded combined batch test')
        d,_=fixture([(70.,3.,90.)],seed=42,nfreq=32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'source.npz'
            np.savez(source,data_full=d.e.raw,metadata=json.dumps(d.e.grid))
            result=batch_fit(source,tmp/'run',workers=1,mode='combined')
            self.assertEqual(result,verify(tmp/'run'))
            self.assertEqual(result['exported'],1)
            receipt=json.loads((tmp/'run'/result['events'][0]['native_receipt']).read_text())
            self.assertTrue(receipt['native_peak_coverage']['passed'])

    def test_backward_pass_removes_redundant_term_without_losing_peak(self):
        from burstsurveyor.science.scintillation import validate
        from burstsurveyor.science.combined import simplify_training,eligible_select
        d,x=fixture([(70.,3.,90.)],seed=42,nfreq=32)
        train=np.arange(d.e.length)%6<4
        doubled=np.r_[x[:5],x[:5],x[5:]]
        row=optimize(d,dict(n=2,scatter=True,x=doubled.tolist(),name='redundant'),train,
                     native=True,iterations=4,deadline=time.monotonic()+5)
        row.update(validate(d,row,train));features=catalogue(d,train)
        row['coverage']=assess(d,row,features,train);row['background_components']=[]
        rows=[row];review=simplify_training(d,rows,train,features,seconds=5)
        best=eligible_select(rows,d,train)
        self.assertEqual(best['n'],1)
        self.assertTrue(best['coverage']['passed'])
        self.assertTrue(review['trial_names'])

    def test_backward_pass_respects_shared_work_guard(self):
        from burstsurveyor.science.scintillation import validate
        from burstsurveyor.science.combined import simplify_training
        d,x=fixture([(55.,2.,90.),(110.,2.,70.)],seed=4,nfreq=32)
        train=np.arange(d.e.length)%6<4
        row=optimize(d,dict(n=2,scatter=True,x=x.tolist(),name='two'),train,
                     native=True,iterations=2,deadline=time.monotonic()+5)
        row.update(validate(d,row,train));features=catalogue(d,train)
        row['coverage']=assess(d,row,features,train);row['background_components']=[]
        d.e.budget['native_work']=d.c['max_native_work']+1
        review=simplify_training(d,[row],train,features,seconds=5)
        self.assertEqual(review['selected_count'],2)
        self.assertTrue(review['failures'])

    def test_simplification_cannot_create_replicated_missing_pulse(self):
        from burstsurveyor.science.pulse_coverage import residual_degradation
        d,x=fixture([(55.,2.,120.),(110.,3.,50.)],seed=4,nfreq=64,scatter=False)
        parent=dict(n=2,scatter=False,x=x.tolist())
        # Use the actual scintillating signal as the adequate parent's pixels.
        noise=np.random.default_rng(4).normal(size=d.e.raw.shape)
        full=d.e.raw-noise;missing=full.copy();missing[:,90:135]=0.
        guard=residual_degradation(d,full,missing)
        self.assertFalse(guard['passed'])
        self.assertGreater(guard['check_score_increase'],3.)

    def test_residual_preservation_accepts_identical_native_pixels(self):
        from burstsurveyor.science.pulse_coverage import residual_degradation
        d,_=fixture([(70.,3.,90.)],seed=4,nfreq=32)
        signal=d.e.raw-np.random.default_rng(4).normal(size=d.e.raw.shape)
        self.assertTrue(residual_degradation(d,signal,signal)['passed'])

if __name__=='__main__':unittest.main()
