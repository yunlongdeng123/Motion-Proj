"""真实C/S小patch：独立teacher-valid/rawmask到Σ采样的CPU梯度，不做optimizer更新。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,unittest
import numpy as np
import torch
from teacher_detail_permission import Q,COMMON_HIGH_BOUND,highpass_cpu,permission_query_cpu
torch.set_num_threads(1)
F=Path(os.environ.get('TEACHER_PERMISSION_FIXTURE',str(Q/'permission/final_r1/cpu_mini_fixture.npz')))
class ActualPermissionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with np.load(F,allow_pickle=False) as d:cls.d={k:d[k].copy() for k in d.files}
 def query(self,case,which='new',readonly=False):
  get=lambda key:torch.as_tensor(self.d[case+'__'+key])
  base,seed=get('base_prior').float(),get('seed').float()
  raw=torch.zeros_like(base,requires_grad=True);latent=get('latent').bool()
  high=highpass_cpu(raw,latent)
  color,mass=permission_query_cpu(base,get('prior_valid').bool(),seed,get('teacher_valid').bool(),
   torch.zeros_like(base),high,get('gate_'+which),get('uv')[None],get('weights'),get('geometry')[None])
  if readonly:color=torch.where(get('write').bool(),color,get('legacy').float()[None])
  color.sum().backward()
  return color.detach(),mass.detach(),raw.grad.detach(),latent
 def test_unknown_prior_qualifies_from_independent_teacher(self):
  self.assertEqual(int(self.d['C_unknown_seed__source'][20,20]),0)
  self.assertFalse(bool(self.d['C_unknown_seed__prior_valid'][20,20]))
  self.assertTrue(bool(self.d['C_unknown_seed__teacher_valid'][20,20]))
  color,mass,grad,_=self.query('C_unknown_seed')
  self.assertTrue(torch.isfinite(color).all());self.assertTrue((mass>0).all())
  self.assertGreater(float(grad.abs().max()),1e-7)
 def test_generated_zero_old_Sigma_gate_gets_new_gradient(self):
  _,_,old,_=self.query('C_generated_zero_old_H','old')
  _,_,new,_=self.query('C_generated_zero_old_H','new')
  self.assertLess(float(old.abs().max()),1e-12);self.assertGreater(float(new.abs().max()),1e-7)
 def test_observed_S_not_rejected_by_prior_label(self):
  self.assertEqual(int(self.d['S_prior_observed__source'][20,20]),1)
  _,_,grad,_=self.query('S_prior_observed')
  self.assertGreater(float(grad.abs().max()),1e-7)
 def test_retrieved_or_pseudo_S_gets_high_gradient(self):
  _,_,grad,_=self.query('S_other_prior');self.assertGreater(float(grad.abs().max()),1e-7)
 def test_raw_latent_outside_common_mask_zero_gradient(self):
  _,_,grad,mask=self.query('C_unknown_seed')
  self.assertTrue(torch.equal(grad[~mask],torch.zeros_like(grad[~mask])))
 def test_conflict_field_point_no_new_teacher_H(self):
  _,_,grad,_=self.query('HF_conflict','teacher')
  self.assertLess(float(grad.abs().max()),1e-12)
 def test_K_readonly_composition_not_colour_GT(self):
  color,_,grad,_=self.query('K_readonly','new',readonly=True)
  self.assertTrue(torch.equal(color,torch.as_tensor(self.d['K_readonly__legacy']).float()[None]))
  self.assertTrue(torch.equal(grad,torch.zeros_like(grad)))
 def test_no_teacher_unknown_stays_readonly(self):
  self.assertFalse(bool(self.d['true_unknown_no_teacher__teacher_valid'][20,20]))
  self.assertFalse(bool(self.d['true_unknown_no_teacher__write']))
  _,_,grad,_=self.query('true_unknown_no_teacher','new',readonly=True)
  self.assertTrue(torch.equal(grad,torch.zeros_like(grad)))
 def test_common_H_bound_and_base_ledger(self):
  self.assertEqual(COMMON_HIGH_BOUND,.34)
  raw=torch.linspace(-8,8,41*41*3).reshape(41,41,3)
  mask=torch.as_tensor(self.d['C_unknown_seed__latent'])
  self.assertLessEqual(float(highpass_cpu(raw,mask).abs().max()),.3400001)
  recipe=json.loads(F.with_name('recipe.json').read_text())
  self.assertFalse(recipe['GPU_authorized']);self.assertEqual(recipe['common_high_bound'],.34)
if __name__=='__main__':unittest.main()

