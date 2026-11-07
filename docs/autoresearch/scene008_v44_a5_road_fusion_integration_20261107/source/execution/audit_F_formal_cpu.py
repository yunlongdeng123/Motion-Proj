"""独立CPU复核F400矩阵预算、参数/导出与原生全图几何不变量。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v82_gn11_v42')
from scripts.worldsim_v82.drivingeditor_dynamic_edit.bounded_road_appearance import read_bounded_qualification,weighted_l1_cost


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    formal=ROOT/'F_formal400_r1';returned=json.loads((formal/'returned.json').read_text())
    assert returned['status']=='COMPLETE_FIXED_400_PAIRED_BOUNDED_COLOUR_RESIDUAL_BAKE_PENDING_GATES'
    assert returned['actual_optimizer_updates']==returned['final_step']==400
    assert returned['actual_native_forwards']==808 and returned['actual_native_backwards']==800
    qual_path=ROOT/'F4_bounded_appearance_qualification/qualification_manifest.json'
    q,ids,w,inside,matrix,matrix_path=read_bounded_qualification(qual_path)
    wc=w[ids];domain_matrix=matrix[:,ids];counts=np.asarray([r['pixels'] for r in q['domain_records']],np.float64)
    expected=[];generator=random.Random(20261107)
    for _ in range(100):
        order=[94,105,128,137];generator.shuffle(order);expected.extend(order)
    checks=[];final=None;final_bounds=None
    for step in [100,200,400]:
        path=formal/f'state{step}.pth';s=torch.load(str(path),map_location='cpu',weights_only=False)
        d=s['delta'].numpy();assert d.shape==(len(ids),3) and d.dtype==np.float32 and np.isfinite(d).all() and np.abs(d).max()<=1
        assert s['step']==step and len(s['history'])==step and [h['raw'] for h in s['history']]==expected[:step]
        assert all(0<=h['weighted_l1_cost_cpu_float64']<=1 and h['finite_gradient'] and h['zero_weight_box_unchanged'] for h in s['history'])
        cost=weighted_l1_cost(d,wc);assert cost<=1 and np.isclose(cost,s['history'][-1]['weighted_l1_cost_cpu_float64'],atol=1e-13,rtol=1e-13)
        l1=np.abs(d.astype(np.float64)).sum(1);bounds=(domain_matrix@l1)/(3*counts)
        assert np.all(bounds<=5e-7+1e-14) and np.all(bounds<=5e-7*cost+1e-14)
        opt=s['optimizer'];assert len(opt['param_groups'])==1 and list(opt['state'])==[0]
        group=opt['param_groups'][0];assert group['params']==[0] and group['lr']==.01 and group['betas']==(.9,.999) and group['eps']==1e-8
        assert int(opt['state'][0]['step'])==step
        for key in ['exp_avg','exp_avg_sq']:assert opt['state'][0][key].shape==s['delta'].shape and torch.isfinite(opt['state'][0][key]).all()
        checks.append(dict(step=step,sha256=sha(path),delta_max_abs=float(np.abs(d).max()),
            independently_recomputed_weighted_l1_cost=cost,maximum_domain_analytic_MAE_bound=float(bounds.max()),
            domain_bounds=bounds.tolist(),all_recorded_step_costs_le_one=True,
            all_runtime_w0_preservation_assertions_recorded=True,sampler_schedule_exact=True,
            only_private_delta_Adam=True))
        if step==400:final=s;final_bounds=bounds
    mp=formal/'evaluation_material/appearance_manifest.json';material=json.loads(mp.read_text())
    assert material['paired_optimizer_steps']==400 and material['parameter_bounds']==[-1.,1.]
    assert material['model_binding']==q['model_binding'] and material['qualification_manifest_sha256']==sha(qual_path)
    assert material['mass_matrix_sha256']==sha(matrix_path)
    delta_path=mp.parent/material['material_file'];assert sha(delta_path)==material['material_sha256']
    with np.load(delta_path,allow_pickle=False) as z:
        assert z['global_branch_row_ids'].dtype==np.int64 and np.array_equal(z['global_branch_row_ids'],ids)
        assert np.array_equal(z['rgb_delta'],final['delta'].numpy())
        exported_cost=weighted_l1_cost(z['rgb_delta'],wc)
    assert exported_cost<=1 and exported_cost==material['weighted_l1_cost']
    geometry=['alpha','depth_sum','depth_m','group_alpha_T','added_owner_rgb_T'];frames=[];domain_changes=[]
    for raw in [94,105,128,137]:
        obs_path=ROOT.parent/f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz'
        with np.load(obs_path,allow_pickle=False) as z:
            a5=z['actor_labels']==1;a6=z['visible'][0].astype(bool);a37=z['visible'][1].astype(bool)
            protected=z['protect'].astype(bool)&~a6&~a37&~z['road'].astype(bool)&~z['road_edit'].astype(bool)
            yy=np.nonzero(a5)[0];upper=a5&(np.arange(320)[:,None]<=yy.min()+.25*(yy.max()-yy.min())) if len(yy) else a5.copy()
        for state in ['keep','delete']:
            p=formal/f'evaluation/frames/{raw:03d}/{state}_float.npz';baseline=ROOT/f'baseline4/frames/{raw:03d}/{state}_float.npz'
            with np.load(p,allow_pickle=False) as z,np.load(baseline,allow_pickle=False) as b:
                assert np.isfinite(z['rgb']).all()
                for field in geometry:assert np.array_equal(z[field],b[field]),(raw,state,field)
                difference=np.abs(z['rgb'].astype(np.float64)-b['rgb'].astype(np.float64))
            masks=dict(a5_full=a5,a5_upper=upper,a6_visible=a6,protected_nonroad=protected,
                a37_visible_keep=a37 if state=='keep' else np.zeros_like(a37))
            for i,record in enumerate(q['domain_records']):
                if record['raw']!=raw or record['state']!=state:continue
                mask=masks[record['domain']];assert int(mask.sum())==record['pixels']
                actual=float(difference[:,mask].mean());bound=float(final_bounds[i])
                assert actual<=bound+5e-7,(record,actual,bound)
                domain_changes.append(dict(**record,analytic_RGB_MAE_change_bound=bound,
                    actual_RGB_L1_change=actual,reserved_roundoff_allowance=5e-7,
                    original_MAE_tolerance=1e-6))
            frames.append(dict(raw=raw,state=state,float_sha256=sha(p),all_global_geometry_and_semantic_RGB_exact=True))
    before=json.loads((formal/'source_and_target_guard_before.json').read_text());after=json.loads((formal/'source_and_target_guard_after.json').read_text());assert before==after
    assert not torch.cuda.is_initialized()
    out=dict(status='PASS_INDEPENDENT_CPU_F400_BUDGET_EXPORT_AND_GEOMETRY',
        qualification_manifest_sha256=sha(qual_path),mass_matrix_sha256=sha(matrix_path),
        checkpoints=checks,export_manifest=str(mp),export_manifest_sha256=sha(mp),
        export_delta_and_global_row_ids_exact_checkpoint400=True,export_recomputed_weighted_l1_cost=exported_cost,
        delta_bounds=[-1,1],domain_budget=5e-7,original_tolerance=1e-6,
        per_measured_domain_colour_change=domain_changes,frames=frames,exact_geometry_fields=geometry,
        all_original_inputs_and_code_bytes_and_mtime_unchanged=True,
        formal_private_RGB_updates=400,formal_native_forwards=808,formal_native_backwards=800,
        original_Gaussian_updates=0,new_GPU_forwards_for_this_audit=0,cuda_initialized=False,
        original_quality_or_visual_acceptance_claim=False,
        limitations=['Independent budget rechecks cover serialized100/200/400 states; other steps are checked via frozen-code runtime assertions/history.',
            'Four calibrated BUILD views only;49frame protection still requires actual evaluation.',
            'RGB-change bound and geometry invariants do not establish seam quality; original4560 scoring and visual review are separate.'])
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=out['status'],cost=exported_cost,max_domain_bound=float(final_bounds.max()),
        max_actual_protected_RGB_change=max(x['actual_RGB_L1_change'] for x in domain_changes),checkpoints=[100,200,400],geometry_exact=True)))


if __name__=='__main__':main()
