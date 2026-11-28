from importlib.util import find_spec
import unittest
import numpy as np
from burstsurveyor.science.engine import Event,InputError
from burstsurveyor.science.fit import load_config
from burstsurveyor.science.model import pulse
from burstsurveyor.science.native import _score
from burstsurveyor.science.objective import ReconstructionData,score

class PeriodicTests(unittest.TestCase):
    def fixture(self,scatter=True):
        grid=dict(freqs_bin0=400.,res_freq=25.,res_time=.001,times_bin0=0.,is_dedispersed=True,period_s=.128)
        c=load_config();c['periodic']=True
        e=Event(np.random.default_rng(21).normal(0,.1,(16,128)),grid,c)
        x=np.array([np.log(100),126.5,np.log(1.8),0.,0.,.3]+([np.log(7.)] if scatter else []))
        return e,x,np.arange(16),np.arange(128.)

    def test_period_and_phase_invariance_and_fluence(self):
        for scatter in (False,True):
            e,x,ch,t=self.fixture(scatter);y,_=pulse(x,1,scatter,e.coords,ch,t)
            shifted=x.copy();shifted[1]+=128
            np.testing.assert_allclose(y,pulse(shifted,1,scatter,e.coords,ch,t)[0],atol=1e-10)
            shifted[1]=x[1]+37
            np.testing.assert_allclose(np.roll(y,37,axis=1),pulse(shifted,1,scatter,e.coords,ch,t)[0],atol=1e-10)
            np.testing.assert_allclose(y.sum(axis=1),100,rtol=1e-10)

    def test_all_derivatives_at_phase_boundary(self):
        e,x,ch,t=self.fixture();y,j=pulse(x,1,True,e.coords,ch,t,True)
        for k in range(len(x)):
            a=x.copy();b=x.copy();a[k]+=1e-5;b[k]-=1e-5
            fd=(pulse(a,1,True,e.coords,ch,t)[0]-pulse(b,1,True,e.coords,ch,t)[0])/2e-5
            np.testing.assert_allclose(fd,j[k],rtol=2e-4,atol=2e-7)

    def test_tail_longer_than_period_conserves_fluence(self):
        e,x,ch,t=self.fixture();x[-1]=np.log(160.)
        y,_=pulse(x,1,True,e.coords,ch,t)
        np.testing.assert_allclose(y.sum(axis=1),100,rtol=1e-10)
        shifted=x.copy();shifted[1]+=57
        np.testing.assert_allclose(np.roll(y,57,axis=1),pulse(shifted,1,True,e.coords,ch,t)[0],atol=1e-10)

    @unittest.skipUnless(find_spec("fitburst"), "requires the optional FitBurst renderer")
    def test_upstream_export_independent_period_sum(self):
        from fitburst.analysis.model import SpectrumModeler
        from fitburst.backend import general

        for scatter in (False,True):
            e,x,ch,t=self.fixture(scatter);e.raw+=pulse(x,1,scatter,e.coords,ch,t)[0]
            d=ReconstructionData(e);row=dict(n=1,scatter=scatter,x=x.tolist())
            row.update(zip(('gain','rms_gain','energy'),score(d,row)))
            stats,pred,base=_score(d,row,32,pulse,SpectrumModeler,general['flags']['normalize_pbf'])
            self.assertLess(stats['relative_model_rmse'],.001)
            self.assertTrue(np.isfinite(pred[d.indices]).all())

    def test_period_required_and_full_rotation(self):
        e,*_=self.fixture()
        for value in (None,.064):
            grid=dict(e.grid)
            if value is None:del grid['period_s']
            else:grid['period_s']=value
            with self.assertRaises(InputError):Event(e.raw,grid,e.config)

    def test_default_not_periodic(self):
        e,x,ch,t=self.fixture(False);linear=Event(e.raw,e.grid,load_config())
        self.assertFalse(hasattr(linear.coords,'period_bins'))
        self.assertLess(pulse(x,1,False,linear.coords,ch,t)[0].sum(),pulse(x,1,False,e.coords,ch,t)[0].sum())
