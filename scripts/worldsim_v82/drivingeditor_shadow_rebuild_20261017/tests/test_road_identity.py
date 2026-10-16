from pathlib import Path
import tempfile,unittest
import torch,numpy as np
from fixture import create
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.losses import geometry_and_reference_terms
from shadow_rebuild.train import preflight


class RoadIdentity(unittest.TestCase):
    def test_positive_road_is_independent_of_unknown_masks_and_does_not_label_other_unknown_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data';cfg=create(path);d=Dataset(path,synthetic=True);f=d.frame(0);model=SceneModel(d.seed,cfg)
            f.mask_known[:]=False;f.road_certified=torch.zeros_like(f.road);f.road_certified[0,0]=True
            _,package=model.buffers(f,len(d.actor_ids),'reference')
            visible=torch.full_like(package['visible_labels'],.4,requires_grad=True);package['visible_labels']=visible
            terms=geometry_and_reference_terms(model,package,f,cfg,'geometry')
            grad=torch.autograd.grad(terms['road_identity'],visible)[0]
            self.assertGreater(float(grad[0,0].abs().sum()),0)
            self.assertEqual(float(grad[~f.road_certified].abs().sum()),0)
            self.assertEqual(float(terms['mask_outside'].detach()),0)
            self.assertEqual(float(terms['mask_inside'].detach()),0)

    def test_enabled_without_positive_data_fails_and_body_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data';cfg=create(path);d=Dataset(path,synthetic=True);cfg.weights.road_identity=.5
            with self.assertRaisesRegex(ValueError,'without actual'):preflight(d,cfg)
            file=path/d.rows[0]['supervision'];v=dict(np.load(file));v['road_certified']=v['masks'].any(-1)
            np.savez_compressed(file,**v)
            with self.assertRaisesRegex(ValueError,'no known body overlap'):d.frame(0)
