"""冻结输入的有限3D边界配对，只读CPU资格记录；不拟合、不渲染。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD = ROOT/'scene008_blue_global_road1k_20261105_r1'
C = ROOT/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1'
AA = ROOT/'scene008_road_seam_contact_20261107_r1/engineer/canonical_n2_mip_candidate_v2'
FINE = ROOT/'scene008_blue_global_M2_final_20261105_r1/fine11x9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(count=int(a.size), min=float(a.min()), mean=float(a.mean()),
                p50=float(np.median(a)), p95=float(np.quantile(a,.95)), max=float(a.max()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    paths=[OLD/'inputs/road_reference.npz',OLD/'formal/checkpoint001000.pt',
           C/'new_road_canonical_candidate.npz',AA/'new_road_canonical_candidate.npz',
           FINE/'coefficients.npz',FINE/'design.json']
    before={str(p):dict(sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
    old=np.load(paths[0],allow_pickle=False)
    saved=torch.load(str(paths[1]),map_location='cpu',weights_only=False,mmap=True)
    state=saved['repair']; new=np.load(paths[2],allow_pickle=False); aa=np.load(paths[3],allow_pickle=False)
    old_ids=np.flatnonzero(old['active'])
    xyz=state['road.xyz'].numpy()
    assert np.array_equal(xyz,old['model_xyz'][old_ids].astype(np.float32))
    assert np.array_equal(state['road.active'].numpy(),np.ones(len(old_ids),bool))
    scale=float(new['meter_to_model'])
    assert np.array_equal(state['road.scaling'].numpy(),(old['gaussian_sigma_m'][old_ids]*scale).astype(np.float32))
    # 资格窗只用于配对审计，不是已授权的优化带宽。
    oi=np.flatnonzero(old['local_xyz'][old_ids,0]>=19.55)
    ni=np.flatnonzero(new['parent_local_xyz'][:,0]<=20.)
    old_local=old['local_xyz'][old_ids]
    new_local=new['parent_local_xyz'].astype(np.float64)
    distance,j=cKDTree(old_local[oi]).query(new_local[ni],k=1)
    paired=oi[j]
    old_normal=old['normal_world'][old_ids]@new['coordinate_anchor'][:3,:3]
    old_normal/=np.linalg.norm(old_normal,axis=1,keepdims=True)
    new_normal=new['parent_normal_local'].astype(np.float64)
    cosine=np.abs((old_normal[paired]*new_normal[ni]).sum(1))
    old_sigma=state['road.scaling'].numpy()/scale
    new_sigma=new['parent_render_scales_model']/scale
    physical=['render_model_xyz','render_scales_model','render_rotation_wxyz','render_opacity_logit','active']
    assert all(np.array_equal(new[k],aa[k]) for k in physical)
    # 原fine域在旧road支持上，严格先于任何候选的未来边界残差。
    fine=np.load(paths[4],allow_pickle=False)
    assert np.array_equal(fine['colour_baseline'],(.5+.28209479177387814*state['road.dc'][:,0]).numpy())
    records=[]
    for n,o,d,cos in zip(ni,paired,distance,cosine):
        records.append(dict(new_parent_row=int(n),new_children=[int(n*4+i) for i in range(4)],
            old_render_row=int(o),old_reference_row=int(old_ids[o]),distance_m=float(d),
            old_local_xyz_m=old_local[o].tolist(),new_local_xyz_m=new_local[n].tolist(),
            old_sigma_m=old_sigma[o].tolist(),new_parent_sigma_m=new_sigma[n].tolist(),
            old_normal_local=old_normal[o].tolist(),new_normal_local=new_normal[n].tolist(),
            absolute_normal_cosine=float(cos),
            old_real_active=bool(old['real_active'][old_ids[o]]),old_inferred=bool(old['inferred'][old_ids[o]]),
            new_measured_support=bool(new['parent_geometry_measured_support'][n]),
            new_actual_return_support_count=int(new['parent_actual_return_support_count'][n]),
            new_material_writeback_eligible=bool(new['writeback_eligible_parents'][n]),
            aa_parent_rgb=aa['parent_render_colours_srgb'][n].tolist(),
            c_parent_rgb=new['parent_render_colours_srgb'][n].tolist()))
    assert not torch.cuda.is_initialized()
    for p in paths:
        assert before[str(p)]==dict(sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    result=dict(status='PASS_READONLY_CPU_BOUNDARY_COORDINATE_AUDIT',
        inputs=before,original_model_geometry_matches_reference_fp32=True,
        original_scaling_matches_reference_fp32=True,old_fine_baseline_matches_saved_road=True,
        old_fine_priority='saved road RGB -> frozen fine11x9 field/clamp -> possible future candidate-only residual',
        old_centre_x_range_m=[float(old_local[:,0].min()),float(old_local[:,0].max())],
        new_parent_centre_x_range_m=[float(new_local[:,0].min()),float(new_local[:,0].max())],
        centre_support_overlap=False,
        audit_window=dict(old_x_min_m=19.55,new_x_max_m=20.,
            old_rows=int(len(oi)),new_parents=int(len(ni)),
            note='CPU association window only; no fit window or optimization authorized by this audit'),
        pairing='nearest Euclidean 3D anchor-local old endpoint centre for each new front parent; no pixel masks',
        pair_distance_m=stats(distance),absolute_normal_cosine=stats(cosine),
        old_paired_inferred_count=sum(r['old_inferred'] for r in records),
        all_new_parents_measured=all(r['new_measured_support'] for r in records),
        paired_records=records,source_eligibility_limits=[
            'Measured road source/support is not independent hidden RGB truth.',
            'Nearest-centre pairs establish locations only; no semantic or continuity quality acceptance.',
            'No nearest-colour extrapolation to the full old road domain is supported.',
            'Any future fit must retain frozen old fine first and guard a5/a6/a37/protected/seam/tire domains.'],
        GPU_forwards=0,optimizer_steps=0,geometry_updates=0,colour_updates=0,cuda_initialized=False)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','paired_records']},indent=2))


if __name__=='__main__':main()
