import tempfile
import unittest
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from burstsurveyor.science.engine import Event
from burstsurveyor.science.fit import load_config
from burstsurveyor.science.objective import ReconstructionData, score
from burstsurveyor.science.scintillation import (fit, shapes, optimize, prediction,
                                                select, MODEL)
from burstsurveyor.science.native import finalize
from burstsurveyor.science.model import pulse


def fixture(components, *, seed=42, scatter=True, nfreq=32, length=192):
    rng = np.random.default_rng(seed)
    grid = dict(freqs_bin0=800., res_freq=-400/nfreq, times_bin0=0.,
                res_time=2.56e-6, is_dedispersed=True)
    raw = rng.normal(size=(nfreq, length))
    event = Event(raw, grid, load_config())
    n = len(components)
    x = np.r_[np.array([[0., t, np.log(sigma), 0., 0.] for t, sigma, _ in components]).ravel(),
              1.2, np.log(4.)] if scatter else np.r_[np.array(
                  [[0., t, np.log(sigma), 0., 0.] for t, sigma, _ in components]).ravel(), 1.2]
    h, _ = shapes(x, n, scatter, event.coords, np.arange(nfreq), np.arange(length))
    a = np.array([scale*(.1+rng.exponential(size=nfreq)) for _, _, scale in components]).T
    raw += np.einsum('ctn,cn->ct', h, a)
    return ReconstructionData(event), x


class ScintillationTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(1)

    def tearDown(self):
        self.threads.restore_original_limits()

    def test_single_scintillating_pulse_is_one_component(self):
        d, truth = fixture([(70., 3., 90.)])
        result = fit(d, 'a'*64)
        row = result['selected']
        self.assertEqual(row['n'], 1)
        self.assertTrue(row['scatter'])
        np.testing.assert_allclose(np.asarray(row['x'])[[1, 2, 5, 6]], truth[[1, 2, 5, 6]], atol=.3)

    def test_distinct_real_pulse_is_retained(self):
        d, _ = fixture([(60., 2., 90.), (110., 3., 45.)])
        row = fit(d, 'a'*64)['selected']
        self.assertEqual(row['n'], 2)
        np.testing.assert_allclose(sorted(np.asarray(row['x'])[1:10:5]), [60, 110], atol=1.)

    def test_overlapping_band_dependent_pulses_are_retained(self):
        d, _ = fixture([(70., 2., 110.), (78., 2., 100.)], seed=8)
        row = fit(d, 'a'*64)['selected']
        self.assertEqual(row['n'], 2)

    def test_validation_never_refits_heldout_amplitudes(self):
        from burstsurveyor.science.scintillation import validate
        d, x = fixture([(70., 3., 90.)])
        train = np.arange(d.e.length) % 6 < 4
        seed = dict(n=1, scatter=True, x=x.tolist(), name='known')
        row = optimize(d, seed, train, native=True, iterations=8, deadline=float('inf'))
        before = np.array(row['channel_amplitudes'])
        first = validate(d, row, train)
        d.e.raw[:, ~train] += 20.
        second = validate(d, row, train)
        np.testing.assert_array_equal(row['channel_amplitudes'], before)
        self.assertGreater(second['validation_loss'], first['validation_loss']*10)

    def test_native_export_replays_profiled_model(self):
        d, _ = fixture([(70., 3., 90.)])
        row = fit(d, 'a'*64)['selected']
        with tempfile.TemporaryDirectory() as tmp:
            receipt = finalize(d, row, Path(tmp)/'model.npz', pulse_fn=pulse, source_sha256='a'*64)
            self.assertEqual(receipt['status'], 'provisional', receipt['reasons'])
            self.assertLess(receipt['selected_score']['relative_model_rmse'], .01)
            with np.load(Path(tmp)/'model.npz') as z:
                expected = prediction(d, row)
                np.testing.assert_allclose(z['prediction'][d.indices], expected, atol=.02, rtol=.01)

    def test_one_se_rule_keeps_simpler_adequate_model(self):
        simple = dict(n=1, scatter=True, name='one', validation_loss=100.,
                      validation_difference_se=2.,validation_uncertainty_reference=dict(name='two',model=None),
                      validation_split_block_losses=[[20., 30.], [20., 30.]])
        extra = dict(n=2, scatter=True, name='two', validation_loss=99.,
                     validation_split_block_losses=[[19., 30.5], [19., 30.5]])
        self.assertIs(select([simple, extra]), simple)

    def test_broad_band_limited_pulses_use_matched_filter_discovery(self):
        d, truth = fixture([(55.,12.,90.),(120.,12.,90.)], seed=4)
        rng = np.random.default_rng(4)
        d.e.raw[:] = rng.normal(size=d.e.raw.shape)
        h, _ = shapes(truth,2,True,d.e.coords,np.arange(len(d.e.raw)),d.t)
        a = np.zeros((len(d.e.raw),2))
        a[24:28,0]=250.; a[28:32,1]=250.
        d.e.raw[:] += np.einsum('ctn,cn->ct',h,a)
        d = ReconstructionData(d.e)
        result = fit(d,'a'*64)
        self.assertIsNotNone(result['selected'])
        self.assertEqual(result['selected']['n'],2)

    def test_negative_predictive_gain_is_not_an_exportable_detection(self):
        candidate=dict(n=1,scatter=False,name='noise',validation_loss=100.,validation_gain=-1.,
                       validation_split_block_losses=[[20.,30.],[20.,30.]])
        self.assertIsNone(select([candidate]))

    def test_weak_channels_retain_collective_pulse_with_smooth_fallback(self):
        d, truth = fixture([(60.,3.,2.)],seed=6,nfreq=256,length=144)
        rng=np.random.default_rng(6)
        h,_=shapes(truth,1,True,d.e.coords,np.arange(256),d.t)
        d.e.raw[:] = rng.normal(size=d.e.raw.shape)+2*h[:,:,0]
        d=ReconstructionData(d.e)
        result=fit(d,'a'*64)
        self.assertIsNotNone(result['selected'])
        self.assertEqual(result['selected']['n'],1)
        self.assertTrue(any(r.get('model')=='smooth-spectrum-v1' for r in result['trials']))

    def test_localized_validation_pulse_is_not_counted_as_noise_variation(self):
        from burstsurveyor.science.scintillation import paired_noise_se
        d,_=fixture([(60.,3.,10.)],scatter=False)
        train=np.arange(d.e.length)%6<4
        simple=dict(n=1,scatter=False,name='one',x=[np.log(10.),60.,np.log(3.),0.,0.,0.],
                    validation_gain=1.,validation_loss=0.)
        extra=dict(n=2,scatter=False,name='two',x=[np.log(10.),60.,np.log(3.),0.,0.,
                    np.log(30.),76.,np.log(.2),0.,0.,0.],validation_gain=1.,validation_loss=0.)
        se,energy,inflation=paired_noise_se(d,simple,extra,train)
        excess=energy
        self.assertGreater(excess,se)
        simple['validation_loss']=1000.+excess;extra['validation_loss']=1000.
        blocks=np.zeros((2,32));blocks[:,12]=excess/2
        simple['validation_split_block_losses']=blocks.tolist()
        extra['validation_split_block_losses']=np.zeros((2,32)).tolist()
        self.assertIs(select([simple,extra],d,train),extra)

    def test_scintillation_batch_seal_and_mode_resume_guard(self):
        import json
        from burstsurveyor.batch import fit as batch_fit, verify
        from burstsurveyor import _dispatch
        d, _ = fixture([(70., 3., 90.)])
        if _dispatch.psutil.virtual_memory().available < 900*1024**2:
            self.skipTest('Insufficient free memory for guarded batch test')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp/'synthetic.npz'
            np.savez(source, data_full=d.e.raw, metadata=json.dumps(d.e.grid))
            result = batch_fit(source, tmp/'run', workers=1, mode='scintillation')
            self.assertEqual(result['exported'], 1)
            self.assertEqual(verify(tmp/'run'), result)
            from burstsurveyor.secondlook import second_look
            with self.assertRaisesRegex(ValueError, 'does not support bounded'):
                second_look(tmp/'run', source, tmp/'unsupported-rescue')
            self.assertEqual(batch_fit(source, tmp/'run', workers=1, mode='scintillation', resume=True), result)
            with self.assertRaisesRegex(ValueError, 'original fitting mode'):
                batch_fit(source, tmp/'run', workers=1, mode='bounded', resume=True)

if __name__ == '__main__':
    unittest.main()
