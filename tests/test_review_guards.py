"""Unresolved scientific diagnostics cannot be signed off as a clean fit."""
import unittest
from types import SimpleNamespace
import numpy as np

from burstsurveyor.assistant import _final_action
from burstsurveyor.review import check_review_grid


class ReviewGuardTests(unittest.TestCase):
    def test_limitations_require_review_without_discarding_export(self):
        for flag in ('broad_edge_component', 'spectral_model_limited',
                     'channel_mask_needs_review', 'positive_residual_island'):
            with self.subTest(flag=flag):
                packet = dict(inspection_status='figure_ready',
                              selected={'prediction': 'sealed.npz'},
                              diagnostics={'flags': [flag]})
                with self.assertRaisesRegex(ValueError, 'mark this event for review'):
                    _final_action({'action': 'retain'}, packet)
                _final_action({'action': 'review'}, packet)
                self.assertEqual(packet['selected']['prediction'], 'sealed.npz')

    def test_clean_export_can_be_retained(self):
        _final_action({'action': 'retain'}, dict(inspection_status='figure_ready',
                      selected={'prediction': 'sealed.npz'}, diagnostics={'flags': []}))

    def test_saved_masks_noise_and_grid_must_match(self):
        data=SimpleNamespace(indices=np.array([0,2]),var=np.ones(3),rms_var=np.ones(3)*2,
             t=np.arange(10),e=SimpleNamespace(frequency=np.array([400.,500.,600.]),
                                             coords=SimpleNamespace(dt=.01)))
        entry={'source_sha256':'a'*64}
        archive=dict(source_sha256=np.array('a'*64),channels=data.indices.copy(),
                     variance=data.var.copy(),rms_variance=data.rms_var.copy(),
                     frequency=data.e.frequency.copy(),time_s=data.t*.01)
        check_review_grid(data,entry,archive)
        for key in ('channels','variance','rms_variance','frequency','time_s'):
            with self.subTest(key=key):
                wrong={k:v.copy() for k,v in archive.items()}
                wrong[key][0]+=1
                with self.assertRaisesRegex(ValueError,'grid, mask or noise differ'):
                    check_review_grid(data,entry,wrong)
