import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from burstsurveyor.science.circular import phase_origin, distance, restore_record
from burstsurveyor.science.engine import Event
from burstsurveyor.science.fit import load_config, fit_event, export_event, load_event
from burstsurveyor.science.model import pulse
from burstsurveyor.science.objective import ReconstructionData
from burstsurveyor.science.pulse_coverage import catalogue, assess
from burstsurveyor.science.pulsar import _merge
from burstsurveyor.review import check_review_grid, load_for_review
from burstsurveyor.secondlook import _window_scores


def fixture():
    grid=dict(freqs_bin0=400.,res_freq=12.5,res_time=.001,times_bin0=0.,is_dedispersed=True,period_s=.128)
    config=load_config();config['periodic']=True
    raw=np.random.default_rng(421).normal(0,.1,(32,128))
    e=Event(raw,grid,config)
    x=np.array([np.log(24),127.5,np.log(1.4),0.,0.,np.log(12),61.2,np.log(2.2),0.,0.,0.])
    raw+=pulse(x,2,False,e.coords,np.arange(32),np.arange(128))[0]
    raw[8]=0
    return e,x


class CircularTests(unittest.TestCase):
    def setUp(self):self.threads=threadpool_limits(1)
    def tearDown(self):self.threads.restore_original_limits()

    def test_canonical_origin_and_noise_commute_with_every_rotation(self):
        e,x=fixture();origin=phase_origin(e.raw);canonical=np.roll(e.raw,-origin,axis=1)
        self.assertEqual(phase_origin(canonical),0)
        original=ReconstructionData(e)
        for shift in range(128):
            raw=np.roll(e.raw,shift,axis=1);p=phase_origin(raw)
            np.testing.assert_array_equal(canonical,np.roll(raw,-p,axis=1))
            d=ReconstructionData(Event(raw,e.grid,e.config))
            np.testing.assert_array_equal(d.indices,original.indices)
            np.testing.assert_array_equal(d.var,original.var)
            np.testing.assert_array_equal(d.rms_var,original.rms_var)
            np.testing.assert_array_equal(np.roll(d.noise_time,-shift),original.noise_time)
            np.testing.assert_array_equal(np.roll(d.y,-shift,axis=1),original.y)

    def test_boundary_peak_cannot_disappear_from_coverage(self):
        e,x=fixture();d=ReconstructionData(e);features=catalogue(d)
        self.assertTrue(any(distance(f['time_bin'],0,128)<=2 for f in features))
        row=dict(n=1,scatter=False,x=np.r_[x[5:10],x[-1]].tolist())
        check=assess(d,row,features)
        self.assertFalse(check['passed'])
        self.assertTrue(any(distance(f['time_bin'],0,128)<=2 for f in check['missing']))
        self.assertTrue(assess(d,dict(n=2,scatter=False,x=x.tolist()),features)['passed'])

    def test_circular_residual_scan_and_boundary_seed(self):
        e,x=fixture();d=ReconstructionData(e);a,_=d.scan(d.y)
        self.assertLess(distance(a.start+a.width/2,0,128),4)
        self.assertLess(distance(d.component_seed(a)[1],0,128),4)
        shifted=ReconstructionData(Event(np.roll(e.raw,37,axis=1),e.grid,e.config));b,_=shifted.scan(shifted.y)
        self.assertAlmostEqual(a.score,b.score,places=9)
        self.assertEqual((a.start+37)%128,b.start)
        p=np.sin(np.arange(128)/7.)
        np.testing.assert_allclose(np.roll(_window_scores(p,7,True),37),_window_scores(np.roll(p,37),7,True),atol=1e-12)

    def test_boundary_merge_uses_short_arc(self):
        x=np.r_[[np.log(10),127.,np.log(1),0,0,np.log(10),1.,np.log(1),0,0],0.]
        merged=_merge(dict(n=2,scatter=False,x=x.tolist()),128)
        self.assertLess(distance(merged['x'][1],0,128),1e-10)
        self.assertAlmostEqual(np.exp(merged['x'][2]),np.sqrt(2),places=10)

    def test_missing_wrapped_peak_is_flagged_in_user_review(self):
        from burstsurveyor.triage import component_omissions
        e,x=fixture();d=ReconstructionData(e)
        incomplete=pulse(np.r_[x[5:10],x[-1]],1,False,e.coords,np.arange(32),np.arange(128))[0]
        result=component_omissions(d,incomplete,dict(id='fixture',source_sha256='a'*64))
        self.assertTrue(result['missing'])
        complete=pulse(x,2,False,e.coords,np.arange(32),np.arange(128))[0]
        self.assertFalse(component_omissions(d,complete,dict(id='fixture',source_sha256='a'*64))['missing'])

    def test_restores_all_candidate_coordinates_without_mutation(self):
        row=dict(n=1,scatter=False,x=[0,125.,0,0,0,0],initial=[0,124.,0,0,0,0],time_bins=[124,132])
        result=restore_record(dict(selected=row,trials=[row]),10,128)
        self.assertEqual(result['selected']['x'][1],7)
        self.assertEqual(result['trials'][0]['x'][1],7)
        self.assertEqual(result['selected']['time_bins'],[6,14])
        self.assertEqual(row['x'][1],125)
        self.assertEqual(result['selected']['initial'][1],6)
        self.assertEqual(row['initial'][1],124)

    def test_full_pipeline_phase_export_and_review_consistency(self):
        e,x=fixture();predictions=[];parameters=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for shift in (0,1,31,64,127):
                path=root/f'input-{shift}.npz'
                np.savez(path,data_full=np.roll(e.raw,shift,axis=1),metadata=json.dumps(e.grid))
                primary=fit_event(path,root,mode='pulsar')
                self.assertIsNotNone(primary['selected'])
                output=root/f'prediction-{shift}.npz'
                native=export_event(path,primary,output,root)
                self.assertEqual(native['status'],'provisional')
                self.assertTrue(native['native_peak_coverage']['passed'])
                with np.load(output) as z:
                    d,source=load_for_review(path,root,primary['source_sha256'],mode='pulsar')
                    check_review_grid(d,source,z)
                    predictions.append(np.roll(z['prediction'],-shift,axis=1))
                p=np.asarray(primary['selected']['x']);p[1:5*primary['selected']['n']:5]=(p[1:5*primary['selected']['n']:5]-shift)%128
                parameters.append(p)
        for y,p in zip(predictions[1:],parameters[1:]):
            np.testing.assert_allclose(y,predictions[0],rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(p,parameters[0],rtol=1e-8,atol=1e-8)


if __name__=='__main__':unittest.main()
