import tempfile,unittest
from pathlib import Path
import numpy as np,torch
from fixture import create
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.identity import evidence_probability,reassign_rigid_owner,append_measured_candidates
from shadow_rebuild.math3d import covariance,rotation_z
from shadow_rebuild.train import seed_from_state

class IdentityEvidence(unittest.TestCase):
 def test_missing_motion_is_unknown_and_conflicting_independent_votes_not_forced_static(self):
  p=evidence_probability(np.array([False,True,False,True]),np.array([False,False,True,True]),np.array([False,True,False,True]),np.zeros(4,bool))
  np.testing.assert_allclose(p.sum(-1),1)
  self.assertEqual(int(p[0].argmax()),2);self.assertGreater(p[1,1],.8);self.assertGreater(p[2,0],.7)
  self.assertLess(p[3].max(),.7)

 def test_actual_owner_roundtrip_preserves_anchor_world_shape_and_unaffected_adam(self):
  with tempfile.TemporaryDirectory() as temporary:
   path=Path(temporary)/'data';cfg=create(path);d=Dataset(path,synthetic=True);model=SceneModel(d.seed,cfg);f=model.field;opt=model.optimizer('geometry')
   sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward();opt.step();row=torch.tensor([int((f.owner==0).nonzero()[0])]);old=f.xyz[row].detach().clone();old_cov=covariance(f.quaternion[row],f.scales[row]).detach();alpha=f.opacity_logit.detach().clone();adam=opt.state[f.xyz]['exp_avg'].clone()
   src=torch.eye(4);src[:3,:3]=rotation_z(.4,device=src.device,dtype=src.dtype);src[:3,3]=torch.tensor([1.2,-.7,.3]);world=old@src[:3,:3].T+src[:3,3];worldcov=src[:3,:3][None]@old_cov@src[:3,:3].T[None]
   reassign_rigid_owner(f,row,-1,src,torch.eye(4),opt)
   self.assertEqual(int(f.owner[row]),-1);torch.testing.assert_close(f.xyz[row],world);torch.testing.assert_close(covariance(f.quaternion[row],f.scales[row]),worldcov,atol=1e-6,rtol=1e-5)
   torch.testing.assert_close(f.opacity_logit,alpha);torch.testing.assert_close(opt.state[f.xyz]['exp_avg'][0],adam[0]);self.assertEqual(float(opt.state[f.xyz]['exp_avg'][row].abs().sum()),0)
   reassign_rigid_owner(f,row,0,torch.eye(4),src,opt);torch.testing.assert_close(f.xyz[row],old,atol=1e-6,rtol=1e-5);torch.testing.assert_close(covariance(f.quaternion[row],f.scales[row]),old_cov,atol=1e-6,rtol=1e-5)
   with self.assertRaises(ValueError):reassign_rigid_owner(f,row,-2,src,src,opt)

 def test_raw_measurement_and_probability_provenance_survives_split_and_state_restore(self):
  with tempfile.TemporaryDirectory() as temporary:
   path=Path(temporary)/'data';cfg=create(path);d=Dataset(path,synthetic=True);n=len(d.seed['xyz']);seed=d.seed.copy()
   seed.update(source_frame=np.arange(n)%4,source_raw_row=np.arange(n)+100,source_world=seed['xyz'].copy(),source_candidate_owner=seed['owner'].copy(),identity_probability=np.tile([.1,.8,.1],(n,1)),identity_evidence_status=np.ones(n,np.int64))
   model=SceneModel(seed,cfg);opt=model.optimizer('geometry');model.field.split(torch.tensor([2]),opt);state=model.state_dict();restored=SceneModel(seed_from_state(state),cfg);restored.load_state_dict(state,strict=True)
   self.assertEqual(int(restored.field.source_raw_row[-1]),102);torch.testing.assert_close(restored.field.source_world[-1],torch.tensor(seed['source_world'][2]));torch.testing.assert_close(restored.field.identity_probability[-1],torch.tensor([.1,.8,.1]))

 def test_certified_pool_admission_preserves_existing_asset_and_adam_and_reloads(self):
  with tempfile.TemporaryDirectory() as temporary:
   path=Path(temporary)/'data';cfg=create(path);d=Dataset(path,synthetic=True);seed=d.seed.copy();n=len(seed['xyz'])
   seed.update(source_frame=np.arange(n)%4,source_raw_row=np.arange(n)+100,source_world=seed['xyz'].copy(),source_candidate_owner=seed['owner'].copy(),identity_probability=np.tile([.1,.8,.1],(n,1)),identity_evidence_status=np.ones(n,np.int64))
   model=SceneModel(seed,cfg);f=model.field;opt=model.optimizer('geometry');sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward();opt.step()
   old_parameters={key:getattr(f,key).detach().clone() for key in f.point_parameters};moments={key:{k:v.clone() for k,v in opt.state[getattr(f,key)].items() if torch.is_tensor(v)} for key in f.point_parameters if getattr(f,key) in opt.state}
   row=int(np.flatnonzero(seed['owner']==0)[0]);candidate={key:np.array(value[row:row+1],copy=True) for key,value in seed.items() if isinstance(value,np.ndarray) and value.ndim and len(value)==n};candidate['neighbor']=np.zeros(1,np.int64);candidate['source_raw_row'][:]=4242
   append_measured_candidates(f,candidate,opt)
   self.assertEqual(len(f.xyz),n+1);self.assertEqual(int(f.owner[-1]),0);self.assertEqual(int(f.source_raw_row[-1]),4242)
   self.assertAlmostEqual(float(f.opacity_logit[-1].sigmoid()),cfg.initial_opacity,places=6)
   for key,value in old_parameters.items():torch.testing.assert_close(getattr(f,key)[:n],value,rtol=0,atol=0)
   for key,state in moments.items():
    for name,value in state.items():
     actual=opt.state[getattr(f,key)][name]
     if value.ndim:torch.testing.assert_close(actual[:n],value,rtol=0,atol=0);self.assertEqual(float(actual[-1].abs().sum()),0)
     else:torch.testing.assert_close(actual,value,rtol=0,atol=0)
   self.assertTrue(torch.all(f.owner[f.neighbor]==f.owner))
   state=model.state_dict();restored=SceneModel(seed_from_state(state),cfg);restored.load_state_dict(state,strict=True)
   self.assertEqual(int(restored.field.source_raw_row[-1]),4242)
   invalid=candidate.copy();invalid['source_raw_row']=np.array([-1])
   with self.assertRaises(ValueError):append_measured_candidates(f,invalid,opt)
