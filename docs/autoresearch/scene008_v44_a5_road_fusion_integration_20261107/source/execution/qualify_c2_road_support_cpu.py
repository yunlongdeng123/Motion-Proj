"""固定BUILD40的三维Gaussian支持邻近资格；不应用mask，不读图像评分。"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
C=ROOT/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1'
C2=ROOT/'scene008_a6_insert_artifacts_20261105/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz'
POSTERIOR=ROOT/'scene008_a6_insert_artifacts_20261105/surface_fit/masked_depth_v1/posterior.npz'
SH=ROOT/'scene008_a6_asset_controls_20261105/source/sh_asset.py'
LOD=ROOT/'scene008_a6_insert_artifacts_20261105/source/scale/surface_lod_asset.py'
OBS=ROOT/'scene008_instances_20261105_r1/observations/frames'
GEOMETRY_RECEIPT=ROOT/'scene008_v44_full5_integration_20261107_r1/candidate_semantic_a5_49/receipt.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def rotation(q):
    w,x,y,z=q.T
    return np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),1).reshape(-1,3,3)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    start=time.time();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    contract=json.loads((C/'fit_contract.json').read_text());build=contract['BUILD']
    assert build==[v for v in range(89,138) if v not in contract['HELD']]
    read_paths=[C/'new_road_canonical_candidate.npz',C/'fit_contract.json',C2,POSTERIOR,SH,LOD,GEOMETRY_RECEIPT]
    before={str(p):dict(sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in read_paths}
    asset=np.load(read_paths[0],allow_pickle=False)
    road=asset['render_model_xyz'].astype(np.float64)
    scales=asset['render_scales_model'].astype(np.float64)
    R=rotation(asset['render_rotation_wxyz'].astype(np.float64))
    road_cov=(R*scales[:,None,:])@np.swapaxes(R*scales[:,None,:],1,2)
    road_r=np.sqrt(np.linalg.eigvalsh(road_cov)[:,-1])
    sh=module('sh_asset',SH);lod_module=module('_frozen_c2_support_lod',LOD)
    base=sh.SHAsset.from_geometry(C2).requires_grad_(False)
    lod=lod_module.SurfaceLODAsset(base,POSTERIOR)
    receipt=json.loads(GEOMETRY_RECEIPT.read_text())
    records={f['raw']:f for f in receipt['frames']}
    retained=lod.retained.numpy()
    base_xyz=base.xyz.detach().numpy()[retained]
    base_normal=base.normal.detach().numpy()[retained]
    base_sigma=base.sigma.detach().numpy()[retained]
    base_thickness=base.thickness.detach().numpy()[retained]
    n=len(road);minimum=np.full(n,np.inf);which_raw=np.full(n,-1,np.int16);which_row=np.full(n,-1,np.int64)
    min_distance=np.full(n,np.inf);per_frame=[]
    I=np.eye(3)[None]
    for raw in build:
        obs_path=OBS/f'{raw:03d}.npz'
        before[str(obs_path)]=dict(sha256=sha(obs_path),bytes=obs_path.stat().st_size,mtime_ns=obs_path.stat().st_mtime_ns)
        with np.load(obs_path,allow_pickle=False) as obs:
            pose=obs['model_from_local'][0].astype(np.float32)
        # 仅复用已封存的几何采样统计；不读取receipt内RGB/alpha/质量分数。
        meta=records[raw]['states']['keep']['meta'];geom=meta['LOD']
        area=float(geom['median_screen_area_per_source_UV'])
        assert float(geom['du'])==1/math.sqrt(area)
        lod.screen_area=lambda matrix,camera:(np.array([area],np.float64),0)
        sample,rec=lod.sampling(torch.from_numpy(pose),None)
        if sample is None:
            local=base.xyz.detach().numpy();normal=base.normal.detach().numpy()
            sigma=base.sigma.detach().numpy();thin=base.thickness.detach().numpy()
        else:
            local=np.concatenate((base_xyz,sample['local']))
            normal=np.concatenate((base_normal,np.tile([0.,0.,1.],(len(sample['local']),1))))
            sigma=np.concatenate((base_sigma,sample['sigma']))
            thin=np.concatenate((base_thickness,sample['sigma']))
        assert len(local)==int(meta['asset_rows'])
        xyz=(local@pose[:3,:3].T+pose[:3,3]).astype(np.float64)
        normal=normal@pose[:3,:3].T;normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        scale=float(np.linalg.norm(pose[:3,:3],axis=0).mean())
        s=sigma.astype(np.float64)*scale;t=thin.astype(np.float64)*scale
        cov=s[:,None,None]**2*I+(t*t-s*s)[:,None,None]*(normal[:,:,None]*normal[:,None,:])
        # 确定性宽相：combined covariance的最大特征值上界，不依赖图像/评分。
        radius=3*np.sqrt(road_r**2+max(float(s.max()),float(t.max()))**2)
        near=cKDTree(xyz).query_ball_point(road,radius)
        lengths=np.fromiter((len(v) for v in near),np.int64,count=n)
        ri=np.repeat(np.arange(n),lengths)
        ai=np.concatenate([np.asarray(v,np.int64) for v in near if len(v)]) if lengths.sum() else np.empty(0,np.int64)
        frame_min=np.full(n,np.inf);frame_row=np.full(n,-1,np.int64);frame_d=np.full(n,np.inf)
        if len(ri):
            delta=road[ri]-xyz[ai];cs=road_cov[ri]+cov[ai]
            d2=np.einsum('ni,ni->n',delta,np.linalg.solve(cs,delta[...,None])[...,0])
            order=np.lexsort((ai,d2,ri));first=np.r_[True,ri[order][1:]!=ri[order][:-1]];take=order[first]
            frame_min[ri[take]]=d2[take];frame_row[ri[take]]=ai[take]
            frame_d[ri[take]]=np.linalg.norm(delta[take],axis=1)/float(asset['meter_to_model'])
            hit=frame_min<minimum;minimum[hit]=frame_min[hit];which_raw[hit]=raw;which_row[hit]=frame_row[hit];min_distance[hit]=frame_d[hit]
        selected=np.flatnonzero(frame_min<=9.)
        per_frame.append(dict(raw=raw,actual_frozen_LOD_rows=len(xyz),actual_du=geom['du'],
            broad_pairs=int(len(ri)),qualified_child_rows=int(len(selected)),
            qualified_parent_rows=int(len(np.unique(asset['child_to_parent_ids'][selected])))))
        print(raw,'pairs',len(ri),'qualified',len(selected),flush=True)
    selected=np.flatnonzero(minimum<=9.);parents=np.unique(asset['child_to_parent_ids'][selected])
    payload=out/'c2_newroad_3d_proximity_qualification.npz'
    np.savez_compressed(payload,qualified_child_row_ids=selected,qualified_parent_row_ids=parents,
        min_combined_covariance_mahalanobis2=minimum,nearest_raw=which_raw,nearest_C2_LOD_row=which_row,
        nearest_centre_distance_m=min_distance,parent_id_by_child=asset['child_to_parent_ids'],
        parent_local_xyz_m=asset['parent_local_xyz'])
    assert not torch.cuda.is_initialized()
    for path,old in before.items():
        p=Path(path);assert old==dict(sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    report=dict(status='PASS_CPU_FIXED_3D_SUPPORT_PROXIMITY_QUALIFICATION',
        protocol='For each newroad child and actual frozen C2 SurfaceLOD Gaussian in BUILD40, min delta^T (Sigma_road+Sigma_C2)^-1 delta <=9; union over BUILD views.',
        covariance_rule='Saved original road quaternion/scales; original C2 normal/tangent sigma/thickness/pose. No geometry changes.',
        threshold=9.,threshold_role='fixed 3-sigma combined-Gaussian proximity proxy, not proof of visibility/collision or automatic deletion authorization',
        BUILD=build,HELD_not_used=contract['HELD'],
        pose_keys_read=['model_from_local[0]'],frozen_receipt_keys_used=['LOD.du','LOD.median_screen_area_per_source_UV','asset_rows'],
        no_RGB_GT_new_render_alpha_or_quality_score_used_for_selection=True,
        actual_SurfaceLOD_geometry_reconstructed_on_CPU=True,source_ancestry_not_reassigned=True,
        aa_c_share_exact_geometry=True,newroad_total_children=n,newroad_total_parents=10098,
        qualified_children=len(selected),qualified_parents=len(parents),
        qualified_parent_local_x_range_m=[float(asset['parent_local_xyz'][parents,0].min()),float(asset['parent_local_xyz'][parents,0].max())] if len(parents) else None,
        near_boundary_qualified_parents=int((asset['parent_local_xyz'][parents,0]<=20.).sum()),
        near_boundary_qualified_children=int((asset['local_xyz'][selected,0]<=20.).sum()),
        per_frame=per_frame,inputs=before,output_npz=str(payload),output_npz_sha256=sha(payload),
        mask_applied=False,model_updates=0,colour_updates=0,GPU_forwards=0,cuda_initialized=False,
        limits=['Gaussian 3D proximity need not cause native ray occlusion; non-intersection cannot exclude projected low-pass tails.',
            'Road support may physically belong below a vehicle; this qualification alone does not establish wrong ownership.',
            'No claim that removing this list preserves road/seam/vehicle quality; existing fixed 71-ray and protection gates remain necessary.'],
        seconds=time.time()-start)
    (out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('COMPLETE',len(selected),len(parents),flush=True)


if __name__=='__main__':main()
