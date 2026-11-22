"""只读既有C/K与固定路面，手工单stamp目标建议；不渲染或改变运输。"""
from pathlib import Path
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.spatial import cKDTree

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
PRIOR=C/'road_gaussian_transfer_20261122_r2_isometric'
PARENT=C/'road_surface_residual_20261121_r2/appearance_fit/asset'
OUT=C/'road_clean_core_transfer_20261122_r4/target_audit'
sys.path.insert(0,str(PRIOR/'code'))
from fixed_surface import FixedRoadSurface
from rigid_patch import RoadFrames


def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (keys or z.files)}


def project(xyz,cam):
    t=cam['world_to_camera'];q=np.asarray(xyz)@t[:3,:3].T+t[:3,3];u=q@cam['camera_K'].T
    return u[:,:2]/np.maximum(q[:,2:],1e-9)-.5,q[:,2]


def sample(a,uv,outside):
    return map_coordinates(a.astype(np.float32),[uv[:,1],uv[:,0]],order=0,mode='constant',cval=outside)


def short_axis(uv):
    widths=[]
    for e in np.roll(uv,-1,axis=0)-uv:
        n=np.array([-e[1],e[0]]);n/=np.linalg.norm(n);v=uv@n;widths.append(float(v.max()-v.min()))
    return min(widths)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    surface=FixedRoadSurface.load(PRIOR/'target/surface.npz')
    masks={};cams={};grid_choices=[]
    for raw in [86,110]:
        d=load(C/'road_surface_residual_20261121_r2/proposal/domains'/f'{raw:03d}.npz',['appearance_C','appearance_S'])
        cam=load(C/'road_common1152_20261121/maps'/f'{raw:03d}.npz',['camera_K','world_to_camera','K'])
        uv,depth=project(surface.world.reshape(-1,3),cam)
        legal=(sample(d['appearance_C'],uv,0)>.5)&(sample(cam['K'],uv,1)<.5)&(depth>0)
        legal=legal.reshape(surface.target.shape)&surface.target&surface.support
        masks[raw]=dict(C=d['appearance_C'],K=cam['K'],distance_to_K=distance_transform_edt(~cam['K']))
        cams[raw]=cam
        grid_choices.append((str(raw),legal))
    grid_choices.append(('common86_110',grid_choices[0][1]&grid_choices[1][1]))
    xyz=load(PARENT/'provenance.npz',['world_xyz'])['world_xyz'];normal=load(PARENT/'road.npz',['normal_world'])['normal_world']
    flags=load(PRIOR/'target/rows.npz',['geometric_valid','protected_K'])
    forward=surface.matrix[:3,0];forward=forward/np.linalg.norm(forward)
    frames=RoadFrames(xyz,normal,flags['geometric_valid'],forward)
    protected=cKDTree(xyz[flags['protected_K'],:2])
    candidates=[];counts={}
    for label,legal in grid_choices:
        distance=distance_transform_edt(legal,sampling=(surface.cell[1],surface.cell[0]))
        y,x=np.nonzero(distance>.40);order=np.argsort(distance[y,x])[::-1];chosen=[]
        for idx in order:
            xy=surface.world[y[idx],x[idx],:2]
            if chosen and np.min(np.linalg.norm(np.array(chosen)-xy,axis=1))<.55:continue
            chosen.append(xy)
            if len(chosen)>=40:break
        counts[label]=dict(legal_grid_nodes=int(legal.sum()),max_inside_distance_m=float(distance.max()),candidate_centers=len(chosen))
        for xy in chosen:
            target=frames.at(xy,.6)
            if target is None:continue
            test={};passed=[]
            for side in [.8,.6,1.0]:
                axis=np.linspace(-side/2,side/2,int(round(side/.02))+1);xx,yy=np.meshgrid(axis,axis)
                world=target['origin']+np.c_[xx.ravel(),yy.ravel()]@target['frame'][:,:2].T
                local=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*side/2
                quad=target['origin']+local@target['frame'][:,:2].T
                view={}
                for raw in [86,110]:
                    uv,dep=project(world,cams[raw]);quv,_=project(quad,cams[raw]);m=masks[raw]
                    cf=sample(m['C'],uv,0)>.5;kk=sample(m['K'],uv,1)>.5
                    minimum=float(sample(m['distance_to_K'],uv,0).min())
                    view[str(raw)]=dict(C_fraction=float(cf.mean()),K_pixels_hit=int(kk.sum()),
                        min_distance_to_K_1152_px=minimum,in_front=bool((dep>0).all()),
                        projected_quad_1152=quv.tolist(),short_axis_native1920_px=short_axis(quv)/.6)
                test[str(side)]=dict(views=view,world_grid_domain_fraction=float(surface.domain(world[:,:2]).mean()),
                    min_worldXY_distance_to_protected_K_rows_m=float(protected.query(world[:,:2])[0].min()))
            for raw in [86,110]:
                v=test['0.8']['views'][str(raw)]
                if v['C_fraction']==1 and v['K_pixels_hit']==0 and v['in_front']:
                    passed.append(raw)
            if not passed:continue
            best=max(passed,key=lambda r:test['0.6']['views'][str(r)]['short_axis_native1920_px'])
            v=test['0.8']['views'][str(best)]
            margin=test['1.0']['views'][str(best)]
            score=test['0.6']['views'][str(best)]['short_axis_native1920_px']*np.sqrt(max(1,v['min_distance_to_K_1152_px']))
            if margin['C_fraction']==1 and margin['K_pixels_hit']==0:score*=1.5
            candidates.append(dict(center_xy=xy.tolist(),target_frame={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in target.items()},
                default_road_frame_radius_m=.6,same_target_frame_for_both_sources=True,origin_grid=label,
                target_frame_raw=best,fully_C_raw_frames=passed,footprint_checks=test,score=float(score)))
    candidates.sort(key=lambda r:-r['score'])
    result=dict(complete=True,CPU_only=True,GPU_calls=0,optimizer_updates=0,new_diffusion_calls=0,
        grid_statistics=counts,candidate_count=len(candidates),candidates=candidates[:12],recommended=candidates[0] if candidates else None,
        mask_resolution=[1152,768],native_full_C_K_gate_required=True,
        footprint_policy='Both source00 0.8m and source02 0.6m squares share this target center/frame. The1.0m square tests0.1m surrounding margin.',
        prior_core_only_fact=dict(code=str(PRIOR/'code/build_transfer.py'),lines='22-43',
            selected_rows='source[world_row_ids][source[core_mask]]',halo='Source isolation/qualification/gallery only, not transported rows',
            old_algorithm_already_core_only=True),
        transport_unchanged='Use existing rigid_patch.transport_rigid; this script only proposes target inputs and never transports or renders.')
    (OUT/'target_candidates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if candidates:
        r=candidates[0];raw=r['target_frame_raw'];m=masks[raw];canvas=np.zeros((*m['C'].shape,3),np.uint8)
        canvas[m['K']]=[60,68,82];canvas[m['C']]=[30,105,70]
        im=Image.fromarray(canvas);d=ImageDraw.Draw(im)
        for side,color in [('1.0','#ffdf60'),('0.8','#ef75e6'),('0.6','#7acaff')]:
            uv=r['footprint_checks'][side]['views'][str(raw)]['projected_quad_1152'];d.line([tuple(p) for p in uv+[uv[0]]],fill=color,width=2)
        im.save(OUT/'recommended_C_K_overlay_1152.png')
    print(json.dumps(dict(complete=True,counts=counts,candidates=len(candidates),recommended=candidates[0] if candidates else None)))


if __name__=='__main__':main()
