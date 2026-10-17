"""CPU-only R15 数值与纠错语义核查；真实场景 GPU 验收另行运行。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
import torch
from factor_integral import factor_integral, factor_transmittance

torch.set_num_threads(2)
DT = torch.float64
DETAILS = {}


def tensor(x):
    return torch.as_tensor(x, dtype=DT, device='cpu')


class NumericalChecks(unittest.TestCase):
    def test_rotated_thin_gaussian_clear_and_blocked_rays(self):
        rng=np.random.default_rng(15)
        r=Rotation.random(random_state=rng).as_matrix()
        s=np.array([1.,.02,1e-5]);m=np.array([123.,-37.,2.])
        local=np.array([[8.,0.,-2.],[0.,0.,-2.]])
        o=m+(local*s)@r.T
        d=np.repeat(r[:,2][None],2,axis=0)
        values=factor_transmittance(tensor(o),tensor(d),tensor(m[None]),
            tensor(r[None]),tensor(s[None]),tensor([.6]))
        expected_tau=-math.log1p(-.6)*np.exp(-.5*local[:,0]**2)*(.5*math.erfc(-math.sqrt(2)))
        expected=np.exp(-expected_tau)
        torch.testing.assert_close(values,tensor(expected),atol=2e-10,rtol=2e-10)
        self.assertGreater(float(values[0]),.999999999)
        self.assertLess(float(values[1]),.5)
        DETAILS['high_condition_transmittance']={'covariance_condition':1e10,
            'clear_and_blocked_T':values.tolist(),'max_absolute_T_error':float((values-tensor(expected)).abs().max())}

    def test_center_sphere_half_line_and_segment(self):
        o = tensor([[0., 0., 0.]])
        d = tensor([[0., 0., 1.]])
        r = torch.eye(3, dtype=DT)[None]
        s = tensor([[.2, .2, .2]])
        alpha = tensor([.6])
        total_tau = -math.log1p(-.6)
        half = factor_integral(o, d, o, r, s, alpha)
        segment = factor_integral(o, d, o, r, s, alpha, lower=-.2, upper=.2)
        self.assertAlmostEqual(float(half), total_tau/2, places=13)
        self.assertAlmostEqual(float(segment), total_tau*math.erf(1/math.sqrt(2)), places=13)

    def test_independent_scalar_density_quadrature(self):
        rng = np.random.default_rng(82)
        errors = []
        for _ in range(24):
            r = Rotation.random(random_state=rng).as_matrix()
            s = np.exp(rng.uniform(np.log(.02), np.log(.7), 3))
            m = rng.uniform(-.3, .3, 3)
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            o = m - .5*d + r @ (s*rng.normal(size=3)*.3)
            alpha = float(rng.uniform(.05, .9))
            amp = -math.log1p(-alpha)/(math.sqrt(2*math.pi)*min(s))
            def rho(t):
                y = np.linalg.solve(r, o+t*d-m)/s
                return amp*math.exp(-.5*float(y@y))
            # 有限段直接对原始三维密度作标量积分，避免积分器漏过窄峰。
            expected, _ = quad(rho, 0., 1., points=[.25, .5, .75], epsabs=1e-12, epsrel=1e-11, limit=200)
            actual = float(factor_integral(tensor(o[None]), tensor(d[None]),
                tensor(m[None]), tensor(r[None]), tensor(s[None]), tensor([alpha]), upper=1.))
            errors.append(abs(actual-expected))
            self.assertLessEqual(abs(actual-expected), 2e-10 + 2e-9*abs(expected))
        DETAILS['quadrature'] = {'cases': 24, 'max_absolute_tau_error': max(errors)}

    def test_physical_condition_1e10_avoids_covariance(self):
        rng = np.random.default_rng(15)
        r = Rotation.random(random_state=rng).as_matrix()
        s = np.array([1., .02, 1e-5])
        m = np.array([123., -37., 2.])
        z = np.array([.2, -.3, .5]); u = np.array([.8, .2, .1])
        d = r @ (s*u); d /= np.linalg.norm(d)
        o = m + r @ (s*z)
        alpha = .2
        # 独立标准坐标中的一维数值积分，变量是白化路径长度。
        y = np.linalg.solve(r, o-m)/s
        v = np.linalg.solve(r, d)/s
        speed = np.linalg.norm(v); unit = v/speed
        amp = -math.log1p(-alpha)/(math.sqrt(2*math.pi)*min(s))
        expected, _ = quad(lambda t: amp/speed*math.exp(-.5*float((y+t*unit)@(y+t*unit))),
                           0, np.inf, epsabs=1e-9, epsrel=1e-10)
        actual = float(factor_integral(tensor(o[None]),tensor(d[None]),tensor(m[None]),
                                     tensor(r[None]),tensor(s[None]),tensor([alpha])))
        self.assertLess(abs(actual-expected)/expected, 1e-8)
        DETAILS['high_condition'] = {'physical_covariance_condition': 1e10,
            'absolute_tau_error': abs(actual-expected), 'tau': actual}

    def test_chunk_order_sun_batch_and_gradients_invariant(self):
        gen = torch.Generator(device='cpu').manual_seed(82)
        ng, nr = 47, 15
        o = torch.randn(nr,3,generator=gen,dtype=DT)*.15
        d = torch.randn(nr,3,generator=gen,dtype=DT);d=d/d.norm(dim=-1,keepdim=True)
        m = (torch.randn(ng,3,generator=gen,dtype=DT)*.2).requires_grad_()
        r = torch.eye(3,dtype=DT).repeat(ng,1,1)
        s = torch.rand(ng,3,generator=gen,dtype=DT)*.25+.03
        a = torch.rand(ng,generator=gen,dtype=DT)*.1+.005
        baseline = factor_transmittance(o,d,m,r,s,a,gaussian_chunk=47)
        grad = torch.autograd.grad(baseline.sum(),m)[0]
        errors = []
        for chunk in [1,7,16,47]:
            pred = factor_transmittance(o,d,m,r,s,a,gaussian_chunk=chunk)
            g = torch.autograd.grad(pred.sum(),m)[0]
            torch.testing.assert_close(pred,baseline,atol=1e-12,rtol=1e-11)
            torch.testing.assert_close(g,grad,atol=1e-11,rtol=1e-10)
            errors.append(float((pred-baseline).abs().max().detach()))
        order=torch.randperm(ng,generator=gen)
        shuffled=factor_transmittance(o,d,m[order],r[order],s[order],a[order],gaussian_chunk=7)
        torch.testing.assert_close(shuffled,baseline,atol=1e-12,rtol=1e-11)
        serial=torch.cat([factor_transmittance(o[i:i+3],d[i:i+3],m,r,s,a) for i in range(0,nr,3)])
        torch.testing.assert_close(serial,baseline,atol=1e-12,rtol=1e-11)
        DETAILS['partition_invariance']={'gaussian_chunks':[1,7,16,47],
            'max_absolute_T_error':max(errors),'sun_groups':5,'rays_per_group':3,
            'permutation_and_gradient_checks':True}

    def test_autograd_against_finite_differences(self):
        from shadow_rebuild.math3d import quaternion_matrix
        o=tensor([[.1,-.05,-.2]]).requires_grad_()
        raw_d=tensor([[.1,.2,1.]]).requires_grad_()
        m=tensor([[.02,.01,.15]]).requires_grad_()
        q=tensor([[1.,.1,.05,-.08]]).requires_grad_()
        ls=tensor(np.log([[.2,.1,.07]])).requires_grad_()
        logit=tensor([-.8]).requires_grad_()
        def func(oo,dd,mm,qq,ll,aa):
            return factor_integral(oo,dd/dd.norm(dim=-1,keepdim=True),mm,
                quaternion_matrix(qq),ll.exp(),aa.sigmoid())
        self.assertTrue(torch.autograd.gradcheck(func,(o,raw_d,m,q,ls,logit),
                                              eps=1e-6,atol=2e-6,rtol=2e-4))
        DETAILS['gradcheck']={'dtype':'float64','eps':1e-6,'atol':2e-6,'rtol':2e-4,
            'parameters':['origin','direction','mean','quaternion','log_scale','opacity_logit']}


class MechanismChecks(unittest.TestCase):
    def test_candidate_funnel_can_be_empty_after_reset_without_geometry_change(self):
        from shadow_rebuild.ownership import classify_evidence
        # 固定证据语义的单元反例，不冒称真实 reset 的线性缩放模型。
        owner=torch.tensor([6,6,6,6])
        e=tensor([[0,0,1,2],[1,0,.3,2],[1,0,0,2],[0,0,0,0]])
        f=torch.tensor([[0,0,3],[3,0,3],[3,0,0],[0,0,0]])
        before=classify_evidence(e,f,owner,[6])
        self.assertEqual(before['road_candidate'].tolist(),[True,False,False,False])
        self.assertEqual(before['mixed'].tolist(),[False,True,False,False])
        low_e=e*.01
        low_f=torch.zeros_like(f)
        after=classify_evidence(low_e,low_f,owner,[6])
        self.assertEqual(int(after['road_candidate'].sum()+after['mixed'].sum()),0)
        self.assertTrue(bool(after['unknown'][-1]))
        DETAILS['reset_counterexample']={'interpretation':'synthetic sensitivity, not a real before-reset render',
            'before_road_candidates':1,'before_mixed':1,'low_contribution_candidates':0}

    def test_production_competition_has_no_force_on_remote_static_ghost(self):
        from types import SimpleNamespace
        from shadow_rebuild.competition import competition_terms
        sa=tensor([[.9]]).requires_grad_();da=tensor([[.9]]).requires_grad_()
        f=SimpleNamespace(masks=torch.ones(1,1,1,dtype=torch.bool),
                          mask_known=torch.tensor([True]),road_certified=torch.zeros(1,1,dtype=torch.bool))
        cfg=SimpleNamespace(continuous_ownership=True,ownership_depth_tolerance_m=.15,
                            ownership_warmup_steps=500)
        def objective(gap):
            p={'alpha':sa+da,'static_isolated_alpha':sa,'dynamic_isolated_alpha':da,
               'static_isolated_depth':tensor([[10.+gap]]),'dynamic_isolated_depth':tensor([[10.]]),
               'visible_labels':da[...,None]}
            return sum(competition_terms(p,f,cfg,500).values())
        near=objective(.1);far=objective(3.7)
        self.assertGreater(float(near.detach()),0)
        self.assertEqual(float(far.detach()),0)
        self.assertEqual(float(torch.autograd.grad(far,sa)[0].abs().sum()),0)
        DETAILS['competition_counterexample']={'near_gap_m':.1,'far_gap_m':3.7,
            'far_loss_and_static_alpha_gradient':0,
            'interpretation':'same-surface regularizer cannot supervise a distant static appearance ghost'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();sys.path.insert(0,str(Path(args.source).resolve()))
    start=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'task_id':'WS-V82-R15-CLOSURE-20261017','run_id':'R15-C00-CPU',
        'status':'PASS' if result.wasSuccessful() else 'FAIL','gpu_used':False,'training_updates':0,
        'production_code_modified':False,'python':platform.python_version(),'torch':torch.__version__,
        'threads':2,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'seconds':time.monotonic()-start,'details':DETAILS,
        'scope':'CPU synthetic references and production gate semantics only; not real scene repair acceptance',
        'failure_ledger_refs':['V82-F05','V82-F06'],'failure_ledger_delta':'none',
        'human_verdict':None,'problem_solved':None}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,allow_nan=False))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
