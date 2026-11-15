import time,unittest
import numpy as np
from threadpoolctl import threadpool_limits
from burstsurveyor.science.engine import Event,optimizer
from burstsurveyor.science.fit import load_config
from burstsurveyor.science.model import pulse
from burstsurveyor.science.objective import ReconstructionData,score
from burstsurveyor.science.pulsar import refine,eligible,summarize_profile
from burstsurveyor.science.pulse_coverage import catalogue


def fixture(parts,scatter=False,tau=5.,seed=12):
    grid=dict(freqs_bin0=400.,res_freq=400/32,times_bin0=0.,res_time=.00025,is_dedispersed=True)
    raw=np.random.default_rng(seed).normal(size=(32,160))
    event=Event(raw,grid,load_config());n=len(parts)
    x=np.r_[np.array([[np.log(f),t,np.log(s),0.,0.] for t,s,f in parts]).ravel(),0.]
    if scatter:x=np.r_[x,np.log(tau)]
    raw+=pulse(x,n,scatter,event.coords,np.arange(32),np.arange(160))[0]
    return ReconstructionData(event),x


def scored(d,x,n,scatter,name):
    row=dict(x=np.asarray(x).tolist(),n=n,scatter=scatter,name=name,termination='fixture',active_bounds=[])
    row['gain'],row['rms_gain'],row['energy']=score(d,row)
    return row


class PulsarTests(unittest.TestCase):
    def setUp(self):self.threads=threadpool_limits(1)
    def tearDown(self):self.threads.restore_original_limits()

    def test_fixed_tau_is_held_while_nuisance_parameters_move(self):
        d,x=fixture([(65.,2.,150.)],True)
        seed=x.copy();seed[1]+=2;seed[2]=np.log(4.)
        row=optimizer(d,1,True,True,seed,'fixed',20,time.monotonic()+10,fixed_log_tau=np.log(5.))
        self.assertAlmostEqual(row['x'][-1],np.log(5.),places=12)
        self.assertLess(abs(row['x'][1]-65.),abs(seed[1]-65.))

    def test_fixed_tau_cannot_silently_enable_scattering_or_exceed_bounds(self):
        d,x=fixture([(65.,2.,100.)])
        with self.assertRaises(ValueError):optimizer(d,1,False,False,x,'bad',1,time.monotonic()+2,fixed_log_tau=0.)
        with self.assertRaises(ValueError):optimizer(d,1,True,False,np.r_[x,0.],'bad',1,time.monotonic()+2,fixed_log_tau=np.log(1000.))

    def test_single_scattered_pulse_can_replace_two_gaussians(self):
        d,x=fixture([(65.,2.,180.)],True)
        seed=np.r_[[np.log(120.),67.,np.log(3.),0.,0.,np.log(60.),74.,np.log(6.),0.,0.],0.]
        inc=optimizer(d,2,False,True,seed,'two',24,time.monotonic()+10)
        inc.update(zip(('gain','rms_gain','energy'),score(d,inc)))
        row,review=refine(d,[inc],inc)
        self.assertEqual(review['count_pairs'][0]['n'], inc['n'])
        self.assertTrue(review['search_complete'])
        self.assertEqual(review['paired_counts'],[1,2])
        self.assertEqual(row['n'],1)
        self.assertTrue(row['scatter'])
        self.assertAlmostEqual(np.exp(row['x'][-1]),5.,delta=.6)
        self.assertTrue(all(c['null_completed'] and c['scattering_completed'] for c in review['count_pairs']))

    def test_unscattered_control_is_not_forced_to_positive_tau(self):
        d,x=fixture([(65.,2.,180.)],seed=4)
        inc=scored(d,x,1,False,'control');row,review=refine(d,[inc],inc)
        self.assertEqual(row['n'],1)
        self.assertFalse(row['scatter'])
        self.assertIn(review['scattering']['status'],('conditional_upper_limit','ambiguous'))

    def test_high_gain_cannot_excuse_a_missing_real_second_peak(self):
        d,x=fixture([(50.,2.,150.),(115.,2.,40.)])
        inc=scored(d,x,2,False,'two_real')
        bad=scored(d,np.r_[x[:5],x[-1]],1,False,'omits_second')
        bad['gain']=inc['gain']+100.;bad['rms_gain']=inc['rms_gain']+100.
        selected,checks=eligible(d,[bad,inc],inc,catalogue(d))
        self.assertEqual(selected['n'],2)
        self.assertEqual(checks[0]['reason'],'missing_observed_peak')

    def test_zero_compatible_profile_is_an_upper_limit(self):
        points=[dict(tau_bins=t,gain=g) for t,g in [(0,100),(.1,100),(.5,99),(2,92)]]
        result=summarize_profile(points,.001,complete=True)
        self.assertEqual(result['status'],'conditional_upper_limit')
        self.assertGreater(result['upper_ms'],.5);self.assertLess(result['upper_ms'],2.)

    def test_truncated_or_shape_degenerate_profile_is_ambiguous(self):
        points=[dict(tau_bins=t,gain=g) for t,g in [(0,90),(.1,95),(.5,100),(2,90)]]
        self.assertEqual(summarize_profile(points,.001,complete=False)['status'],'ambiguous')
        self.assertEqual(summarize_profile(points,.001,complete=True,alternative=True)['status'],'ambiguous')
        self.assertEqual(summarize_profile(points,.001,complete=True)['status'],'conditionally_constrained')

    def test_pulsar_mode_is_explicit(self):
        from burstsurveyor.cli import _parser
        parser=_parser()
        self.assertEqual(parser.parse_args(['prepare','inputs','work']).mode,'bounded')
        self.assertEqual(parser.parse_args(['prepare','inputs','work','--mode','pulsar']).mode,'pulsar')

    def test_actual_native_missing_peak_is_rejected_in_pulsar_mode(self):
        from test_combined import CombinedTests
        result,exists=CombinedTests()._native_gate(True,mode='pulsar')
        self.assertEqual(result['status'],'unresolved')
        self.assertFalse(exists)

    def test_actual_native_adequate_reconstruction_is_exported(self):
        from test_combined import CombinedTests
        result,exists=CombinedTests()._native_gate(False,mode='pulsar')
        self.assertEqual(result['status'],'provisional')
        self.assertTrue(exists)
        self.assertTrue(result['native_peak_coverage']['passed'])

    def test_memory_failure_keeps_pulsar_workspace_resumable(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from burstsurveyor.assistant import prepare
        with tempfile.TemporaryDirectory() as tmp:
            workspace=Path(tmp)/'run'
            with patch('burstsurveyor.assistant.fit',side_effect=RuntimeError('memory guard')) as fitter:
                with self.assertRaisesRegex(RuntimeError,'memory guard'):
                    prepare(Path(tmp)/'inputs',workspace,workers=1,mode='pulsar')
                self.assertFalse(fitter.call_args.kwargs['defer_memory'])
                self.assertTrue((workspace/'session.json').exists())
                self.assertFalse((workspace/'report-seal.json').exists())


if __name__=='__main__':unittest.main()
