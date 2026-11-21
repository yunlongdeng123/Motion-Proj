"""RGB-first semantic grouping and exact signed first-ray-hit diagnostics.

All 1030 primary proxy points remain in every original quality metric. Polygon
groups explain possible return types; they do not change geometry or thresholds.
"""
from pathlib import Path
import argparse
import json
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import cv2
import numpy as np
from PIL import Image
import trimesh
import prepare_observation_surface as obs

POLYGONS={
 '86':{'rear_glass_RGB_polygon':[[134,64],[271,63],[287,106],[119,108]],
       'opaque_tailgate_RGB_polygon':[[111,131],[292,131],[292,148],[114,155]]},
 '110':{'rear_glass_RGB_polygon':[[158,74],[323,75],[332,116],[144,117]],
        'opaque_tailgate_RGB_polygon':[[127,147],[334,147],[333,179],[126,188]]}}


def stats(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return dict(count=0)
    return dict(count=len(a),mean=float(a.mean()),p10=float(np.quantile(a,.1)),p50=float(np.quantile(a,.5)),
        p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),minimum=float(a.min()),maximum=float(a.max()))


def first_ray_hits(mesh,points,mapping):
    origin=np.asarray(mapping['camera_position_canonical_m'],float)
    direction=points-origin;measured_range=np.linalg.norm(direction,axis=1);direction/=measured_range[:,None]
    tri=np.asarray(mesh.triangles);center=tri.mean(1)
    radius=np.linalg.norm(tri-center[:,None],axis=2).max(1)
    wc=center-origin;wc2=(wc*wc).sum(1)
    e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    distance=np.full(len(points),np.nan);face=np.full(len(points),-1,np.int64)
    xyz=np.full((len(points),3),np.nan);candidate_counts=[]
    for i,d in enumerate(direction):
        longitudinal=wc@d
        perpendicular2=np.maximum(wc2-longitudinal*longitudinal,0)
        ids=np.flatnonzero((longitudinal+radius>0)&(perpendicular2<=radius*radius+1e-10))
        candidate_counts.append(len(ids))
        pv=np.cross(np.broadcast_to(d,(len(ids),3)),e2[ids]);det=(e1[ids]*pv).sum(1)
        ok=np.abs(det)>1e-12
        inv=np.zeros(len(ids));inv[ok]=1/det[ok]
        sv=origin-tri[ids,0];u=(sv*pv).sum(1)*inv
        qv=np.cross(sv,e1[ids]);v=(qv*d).sum(1)*inv;t=(e2[ids]*qv).sum(1)*inv
        ok&=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>1e-9)
        if ok.any():
            choice=np.flatnonzero(ok)[np.argmin(t[ok])]
            distance[i]=t[choice];face[i]=ids[choice];xyz[i]=origin+d*distance[i]
    return measured_range-distance,face,xyz,measured_range,distance,stats(candidate_counts)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    frozen=dict(schema='a20.RGB_only_surface_group_polygons.v1',
        policy='Manually fixed from original RGB crops only, before grouped residual/depth computation. Coordinates are crop-native pixels. Conservative inner polygons avoid borders.',
        source_raws=[86,110],polygons=POLYGONS,other_points_group='unknown_or_other_RGB_region',
        geometry_or_distance_used_to_choose_polygons=False,quality_thresholds_changed=False,points_may_be_deleted=False)
    obs.write(args.output/'RGB_groups_preregistered.json',frozen)
    mesh=trimesh.load(args.prior,force='mesh',process=False)
    with np.load(args.conditioning/'lidar/aggregated_lidar_train_086_110.npz',allow_pickle=False) as z:
        p=np.asarray(z['xyz_canonical_m'],float);source=z['source_raw'];rows=z['representative_source_row']
    with np.load(args.fusion/'point_metric_partition.npz',allow_pickle=False) as z:
        before=z['prior_point_to_surface_m'];after=z['fused_point_to_surface_m'];used=z['used_for_surface']
    group=np.zeros(1030,np.uint8);signed=np.full(1030,np.nan);face=np.full(1030,-1,np.int64)
    hit_xyz=np.full((1030,3),np.nan);measured_range=np.full(1030,np.nan);hit_range=np.full(1030,np.nan)
    per_frame=[];refs={}
    for raw in obs.TRAIN:
        m,f,r=obs.read_frame(args.conditioning,raw);refs[str(raw)]=r
        ids=np.flatnonzero(source==raw);uv,depth=obs.project(p[ids],m)
        labels=np.zeros(f['visible_mask'].shape,np.uint8)
        for code,name in [(1,'rear_glass_RGB_polygon'),(2,'opaque_tailgate_RGB_polygon')]:
            cv2.fillPoly(labels,[np.asarray(POLYGONS[str(raw)][name],np.int32)],code)
        g,_=obs.pixel_lookup(uv,labels);group[ids]=g
        overlay=f['rgb'].copy()
        for code,color in [(1,[100,150,255]),(2,[90,225,135])]:
            mask=labels==code;overlay[mask]=(.65*overlay[mask]+.35*np.asarray(color)).astype(np.uint8)
        for name in POLYGONS[str(raw)]:cv2.polylines(overlay,[np.asarray(POLYGONS[str(raw)][name],np.int32)],True,(240,240,240),1)
        Image.fromarray(overlay).save(args.output/f'RGB_group_polygons_{raw:03d}.png')
        print(f'Exact two-sided first-ray hit raw{raw:03d}: {len(ids)} measured proxy points',flush=True)
        delta,hit,xyz,mrange,hrange,count=first_ray_hits(mesh,p[ids],m)
        signed[ids]=delta;face[ids]=hit;hit_xyz[ids]=xyz;measured_range[ids]=mrange;hit_range[ids]=hrange
        per_frame.append(dict(raw=raw,ray_hit_points=int((hit>=0).sum()),ray_miss_points=int((hit<0).sum()),candidate_triangle_counts=count))
    cab_rear=(p[:,0]>=-1)&(p[:,0]<=-.2)
    summaries={}
    for name,code in [('rear_glass_RGB_polygon',1),('opaque_tailgate_RGB_polygon',2),('unknown_or_other_RGB_region',0)]:
        mask=group==code;ray=mask&np.isfinite(signed);delta=signed[ray]
        summaries[name]=dict(points=int(mask.sum()),used_for_surface_points=int((mask&used).sum()),
            unused_for_surface_points=int((mask&~used).sum()),cab_rear_band_points=int((mask&cab_rear).sum()),
            source_counts={str(raw):int((mask&(source==raw)).sum()) for raw in obs.TRAIN},
            prior_exact_point_to_surface_m=stats(before[mask]),fused_exact_point_to_surface_m=stats(after[mask]),
            ray_depth_delta_measured_minus_first_hit_m=stats(delta),
            behind_first_hit_points=int((delta>1e-6).sum()),in_front_of_first_hit_points=int((delta< -1e-6).sum()),
            within_numerical_zero_points=int((np.abs(delta)<=1e-6).sum()),
            behind_more_than_0p075m_points=int((delta>.075).sum()),
            in_front_more_than_0p075m_points=int((delta<-.075).sum()),
            within_plus_minus_0p075m_points=int((np.abs(delta)<=.075).sum()),ray_miss_points=int((mask&(face<0)).sum()))
    np.savez_compressed(args.output/'grouped_returns.npz',xyz_canonical_m=p,source_raw=source,
        representative_source_row=rows,RGB_region_group=group,used_for_surface=used,
        prior_point_to_surface_m=before,fused_point_to_surface_m=after,
        signed_measured_minus_first_ray_hit_m=signed,prior_first_ray_hit_face_id=face,
        prior_first_ray_hit_xyz=hit_xyz,measured_range_m=measured_range,prior_first_hit_range_m=hit_range,
        all_1030_points_retained=np.asarray(True))
    receipt=dict(schema='a20.glass_return_diagnostic.cpu.v1',
        group_contract=obs.record(args.output/'RGB_groups_preregistered.json'),groups=summaries,
        source_prior=obs.record(args.prior),point_metric_source=obs.record(args.fusion/'point_metric_partition.npz'),
        original_RGB_and_mapping=refs,ray_definition='Exact two-sided nearest positive Moller-Trumbore ray/triangle hit, with conservative triangle-sphere candidate pruning; origin is each source camera in actor metres. Signed range = measured range - first-hit range; positive means behind the prior exterior.',
        sign_is_relative_to_coarse_prior_not_true_exterior=True,
        cab_rear_band=dict(x_range_m=[-1,-.2],points=int(cab_rear.sum()),
            glass_group_points=int((cab_rear&(group==1)).sum()),
            opaque_tailgate_points=int((cab_rear&(group==2)).sum()),unknown_points=int((cab_rear&(group==0)).sum())),
        per_frame_ray_proof=per_frame,primary_points_retained=1030,excluded_points=0,
        thresholds_changed=False,polygons_used_for_training_or_fusion=False,geometry_modified=False,
        hypothesis_status='Possible glass/interior returns and exterior-shape mismatch remain competing explanations; this diagnostic alone cannot establish return semantics.',
        native_model_or_generated_alpha_used_for_RGB_grouping=False,gpu_used=False,
        torch_imported='torch' in sys.modules,heldout_reads=[])
    obs.write(args.output/'glass_return_diagnostic.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['prior','conditioning','fusion','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
