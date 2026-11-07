"""独立复核F原域Ni/掩码/逐row质量/w与primary算子容量；无新GPU。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
CAM=ROOT.parent;F=ROOT/'F4_bounded_appearance_qualification';E=ROOT/'E4_appearance_qualification'
sys.path[:0]=['/root/autodl-tmp/motion_proj_v82_gn11_v42',str(ROOT/'source')]
from scripts.worldsim_v82.drivingeditor_dynamic_edit.bounded_road_appearance import read_bounded_qualification
from diagnose_appearance_capacity_cpu import capacity


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    binding=json.loads((E/'model_binding_before.json').read_text())
    q,ids,w,inside,matrix,path=read_bounded_qualification(F/'qualification_manifest.json',binding)
    receipt=json.loads((F/'receipt.json').read_text());protocol=json.loads((F/'qualification_protocol.json').read_text())
    assert receipt['joint_forwards']==receipt['dummy_feature_backwards']==8
    assert receipt['base_RGB_nonnegative_premise'] and receipt['all_original_model_binding_and_files_unchanged']
    assert all(p['negative_rows']==0 and p['finite'] and p['base_RGB_min']>=0 for p in receipt['base_RGB_premise_details'])
    for p,item in protocol['inputs'].items():
        source=Path(p);assert (source.stat().st_size,source.stat().st_mtime_ns)==(item['bytes'],item['mtime_ns']) and sha(source)==item['sha256']
    records={(r['raw'],r['state'],r['domain']):(i,r) for i,r in enumerate(q['domain_records'])}
    empty={(r['raw'],r['state'],r['domain']):r for r in q['empty_domains']}
    with np.load(CAM/'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz',allow_pickle=False) as z:
        ground=z['depth_m'][:,3]>0
    names=['a5_full','a5_upper','a6_visible','protected_nonroad','a37_visible_keep']
    accumulated=np.zeros_like(inside);domain_counts=[];point_files={}
    for raw in q['calibration_raw_ids']:
        with np.load(CAM/f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz',allow_pickle=False) as z:
            a5=z['actor_labels']==1;a6=z['visible'][0].astype(bool);a37=z['visible'][1].astype(bool)
            road=z['road'].astype(bool);edit=z['road_edit'].astype(bool)
            protect=z['protect'].astype(bool)&~a6&~a37&~road&~edit
            yy=np.nonzero(a5)[0];upper=a5&(np.arange(320)[:,None]<=yy.min()+.25*(yy.max()-yy.min())) if len(yy) else a5.copy()
        for state in q['states']:
            p=F/'point_mass'/f'{raw:03d}_{state}.npz';point_files[str(p)]=sha(p)
            with np.load(p,allow_pickle=False) as z:points=z['mass'].copy()
            assert points.shape==(len(inside),6) and points.dtype==np.float32 and np.isfinite(points).all() and np.all(points>=0)
            accumulated+=points[:,0].astype(np.float64)
            masks=[a5,upper,a6,protect,a37 if state=='keep' else np.zeros((320,480),bool)]
            for column,(name,mask) in enumerate(zip(names,masks),1):
                key=(raw,state,name);pixels=int(mask.sum());digest=hashlib.sha256(np.ascontiguousarray(mask,dtype=np.bool_).tobytes()).hexdigest()
                if pixels:
                    i,record=records[key];assert record['pixels']==pixels and record['mask_sha256']==digest
                    assert np.array_equal(matrix[i],points[:,column].astype(np.float64))
                else:
                    assert empty[key]['pixels']==0 and empty[key]['mask_sha256']==digest and not points[:,column].any()
                domain_counts.append(dict(raw=raw,state=state,domain=name,pixels=pixels,mask_sha256=digest))
    assert np.array_equal(inside,accumulated)
    for record in json.loads((F/'progress.json').read_text())['rows']:
        assert max(record['mass_sum_relative_errors'])<=1e-5
        assert all(v==0 for v in record['baseline_parity_max_abs'].values())
    nmin=min(d['pixels'] for d in domain_counts if d['pixels']>0)
    with np.load(E/'safe_baseline_road_rows.npz',allow_pickle=False) as z:
        einside=np.r_[z['static_inside_mass'],z['road_inside_mass']]
        eoutside=np.r_[z['static_outside_mass'],z['road_outside_mass']]
    comparisons={}
    for state in ['keep','delete']:
        with np.load(E/'point_mass'/f'128_{state}.npz',allow_pickle=False) as z:primary=z['mass'][:,5].astype(np.float64)
        variants={'E_v2_union_Nmin_original1e-6':capacity(primary,einside,eoutside,nmin*1e-6),
            'E_v2_union_Nmin_same_half5e-7':capacity(primary,einside,eoutside,nmin*5e-7),
            'F_original_domain_sum_Nmin_half5e-7':capacity(primary,inside,matrix.sum(0),nmin*5e-7),
            'F_registered_max_normalized_domain_half5e-7':capacity(primary,inside,3*w,1.)}
        comparisons[state]=variants
    report=dict(status='PASS_INDEPENDENT_CPU_F_DOMAINS_MASS_BUDGET_AND_CAPACITY',
        qualification_manifest_sha256=sha(F/'qualification_manifest.json'),mass_matrix_sha256=sha(path),
        domain_mass_shape=list(matrix.shape),domain_counts=domain_counts,point_mass_sha256=point_files,
        every_original_mask_Ni_and_SHA_recomputed=True,all28_mass_rows_exact_source_point_grad=True,
        inside8view_aggregate_exact=True,weights_full_matrix_recomputed_exact=True,
        candidate_rows=len(ids),zero_weight_candidate_rows=int((w[ids]==0).sum()),
        candidate_by_branch=receipt['candidate_by_branch'],base_RGB_nonnegative_premise=True,
        original_model_binding_exact_previous_verified_baseline=True,
        protection_budget=5e-7,original_tolerance=1e-6,N_min_for_old_union_comparison=nmin,
        comparisons=comparisons,proof='For every recorded nonempty domain m, A_mr/(3*N_m)<=epsilon*w_r. Triangle inequality and selected ReLU1-Lipschitz at nonnegative base imply MAE regression<=epsilon*sum(w_r*L1(delta_r))<=5e-7.',
        caveats=['F uses original five protected metric domains; old E v2 union additionally contained model-alpha support. Their writable-capacity difference is not solely normalization.',
            'F summed-domain control double-counts overlaps and is conservative, not an exact measured union.',
            'LP is exact for continuous operator-leverage amplitudes, not achievable target error reduction or a chosen field.',
            'Four BUILD views only. No unseen49 guarantee and no quality acceptance.'],
        GPU_forwards_for_this_audit=0,fit_steps_for_this_audit=0)
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','candidate_rows','zero_weight_candidate_rows','domain_mass_shape']},indent=2))


if __name__=='__main__':main()
