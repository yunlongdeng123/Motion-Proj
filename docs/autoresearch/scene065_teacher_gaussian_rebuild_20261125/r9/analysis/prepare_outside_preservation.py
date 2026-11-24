"""Restore only retired static parents with zero protected footprint in all20 views.

Selection uses native preprocessing radii, not occluded RGB/alpha attribution.
No model parameter is changed; the only output edit is a retirement ID set.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, itertools, json, sys, time
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_background_coverage_fix_20261125_r9')
MASK_ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_native30k_red_rebuild_20261125_r4/masks')
OBS_ROOT=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs/observations')
A9_TRACK='NTjq-OhZ_97OMJ4xwJAxXg'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def pin(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return dict(path=str(path),sha256=h.hexdigest())

def checked(spec):
    path=Path(spec['path']);assert pin(path)['sha256']==spec['sha256'],path
    return path

def load_npz(path):
    with np.load(path,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}

def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b)and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(exact(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):return type(a)is type(b)and len(a)==len(b)and all(exact(x,y)for x,y in zip(a,b))
    return a==b

def paint_rect(roi,xyxy,pad=0):
    h,w=roi.shape;x0,y0,x1,y1=np.asarray(xyxy,np.float64)
    if not np.isfinite([x0,y0,x1,y1]).all():roi[:]=True;return
    x0=max(0,int(np.floor(x0-pad)));y0=max(0,int(np.floor(y0-pad)))
    x1=min(w,int(np.ceil(x1+pad))+1);y1=min(h,int(np.ceil(y1+pad))+1)
    if x1>x0 and y1>y0:roi[y0:y1,x0:x1]=True

def project_numpy(model_xyz,view,projection,width,height):
    homogeneous=np.concatenate([model_xyz,np.ones((len(model_xyz),1))],-1)
    depth=(homogeneous@view)[:,2];clip=homogeneous@projection
    uv=(clip[:,:2]/clip[:,3:4]+1)*np.asarray([width,height])/2-.5
    return uv,depth

def box_roi(observation,base,camera,shape,official_wh):
    """Union exact native3D box rectangles and supplied FRONT rectangles."""
    h,w=shape;roi=np.zeros(shape,bool);view=camera.world_view_transform.detach().cpu().numpy().astype(np.float64)
    projection=camera.full_proj_transform.detach().cpu().numpy().astype(np.float64)
    count=0;a9_volumes=[]
    for box in observation['boxes']:
        pose=np.asarray(box['world_from_actor'],np.float64).reshape(4,4);dim=np.asarray(box['dimensions'],np.float64)
        assert np.isfinite(pose).all()and np.all(dim>0)
        corners=np.asarray(list(itertools.product([-1.,1.],repeat=3)))*dim/2
        world=corners@pose[:3,:3].T+pose[:3,3];model=world@base.transform[:3,:3].T+base.transform[:3,3]
        uv,depth=project_numpy(model,view,projection,w,h)
        if np.any(depth>0):
            if not np.all(depth>0):roi[:]=True
            else:paint_rect(roi,[uv[:,0].min(),uv[:,1].min(),uv[:,0].max(),uv[:,1].max()])
            count+=1
        comparison=box.get('projection_comparison')or{}
        official=comparison.get('official_box')
        if official is not None:
            rectangle=np.asarray(official,np.float64)*np.asarray([w/official_wh[0],h/official_wh[1],w/official_wh[0],h/official_wh[1]])
            paint_rect(roi,rectangle)
        if box['track']==A9_TRACK:a9_volumes.append((pose,dim))
    return roi,count,a9_volumes

def tile_bbox_hits(roi,centres,radii,margin=32,tile=16):
    """Any intersecting pixel in outward-rounded, padded native tile bounds."""
    h,w=roi.shape;n=len(radii);hits=np.zeros(n,bool)
    nonfinite=(radii>0)&(~np.isfinite(centres).all(-1)|~np.isfinite(radii))
    hits[nonfinite]=True
    ids=np.flatnonzero((radii>0)&~nonfinite)
    if not len(ids):return hits,0
    xy=centres[ids];radius=radii[ids,None]+margin
    lower=np.floor((xy-radius)/tile)*tile
    upper=(np.floor((xy+radius)/tile)+1)*tile
    lower=np.maximum(lower,0);upper=np.minimum(upper,np.asarray([w,h]))
    lower=np.minimum(lower,np.asarray([w,h])).astype(np.int64);upper=np.maximum(upper,0).astype(np.int64)
    x0,y0=lower.T;x1,y1=upper.T
    valid=(x1>x0)&(y1>y0)
    integral=np.pad(roi.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    mass=np.zeros(len(ids),np.int64)
    mass[valid]=integral[y1[valid],x1[valid]]-integral[y0[valid],x1[valid]]-integral[y1[valid],x0[valid]]+integral[y0[valid],x0[valid]]
    hits[ids]=mass>0
    return hits,int(valid.sum())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--runtime',type=Path)
    p.add_argument('--mask-root',type=Path,default=MASK_ROOT);p.add_argument('--observations',type=Path,default=OBS_ROOT)
    p.add_argument('--pad-pixels',type=int,default=32);p.add_argument('--official-box-wh',type=int,nargs=2,default=[960,640]);a=p.parse_args()
    assert a.pad_pixels>=32 and not a.output.exists(),'Keep old evidence; use a new output directory'
    a.output.mkdir(parents=True);started=time.perf_counter()
    model_path=a.model or a.root/'training_v1/train/model_final.pt';contract=a.runtime or a.root/'config/runtime.json'
    payload=torch.load(model_path,map_location='cpu',weights_only=False)
    assert payload['logical_step']==2800
    bindings=payload['bindings'];assert pin(contract)==bindings['runtime_contract']
    module_path=checked(bindings['runtime_code']);sys.path.insert(0,str(module_path.parent))
    spec=importlib.util.spec_from_file_location('_outside_frozen_runtime',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rt=module.TeacherGaussianRuntime(contract,a.output/'runtime');rt.restore_model_state(payload['model'],None)
    rt.frozen_assets['cache_manifest']=bindings['cache_manifest']['sha256']
    assert rt.frozen_assets==payload['frozen_assets']
    before=rt.capture_model_state();assert exact(before,payload['model']);base=rt.base
    cfg=read(bindings['configuration']['path']);frames=list(base.frames)
    assert len(frames)==20 and len(set(frames))==20 and set(frames)==set(rt.frames)
    mask_frames=set(cfg['readout_frames']);assert len(mask_frames)==16
    targets=read(checked(bindings['teacher_targets']));byraw={int(r['raw']):r for r in targets['frames']}
    retirement_spec=rt.cfg['context_retirement'];old=load_npz(checked(retirement_spec));removed=np.asarray(old['remove_global_row_ids'],np.int64)
    assert len(np.unique(removed))==len(removed)
    ns=base.native_static_rows;candidates=removed[removed<ns]
    assert np.all(candidates>=0)
    semantic=base.owners(frames[0],semantic=True).detach().cpu().numpy()
    excluded_A9=semantic[candidates]==10
    fragments=load_npz(checked(rt.cfg['retained_fragments']))
    fragment_parents=np.asarray(fragments['parent_global_row_ids'],np.int64)
    excluded_fragment_parent=np.isin(candidates,fragment_parents)
    eligible=~excluded_A9&~excluded_fragment_parent
    candidates=candidates[eligible];candidate_ids=torch.as_tensor(candidates,dtype=torch.int64,device=base.global_row_ids.device)
    rejected=np.zeros(len(candidates),bool);ever_visible=np.zeros(len(candidates),bool);a9_center=np.zeros(len(candidates),bool)
    rows=[];input_masks=[];input_observations=[];geometry_inputs=[]
    with torch.no_grad():
        for raw in frames:
            camera=base.cameras[raw];h=int(camera.image_height);w=int(camera.image_width)
            local=rt.render(raw,component='local');alpha=local['alpha'].detach().cpu().numpy().squeeze()
            assert alpha.shape==(h,w)and np.isfinite(alpha).all()
            local_roi=alpha>0;roi=local_roi.copy();red_count=green_count=geometry_count=0
            if raw in mask_frames:
                path=a.mask_root/f'{raw:03d}'/'native.npz';masks=load_npz(path)
                assert masks['red_rebuild'].shape==masks['keep_green'].shape==(h,w)
                roi|=masks['red_rebuild'].astype(bool)|masks['keep_green'].astype(bool)
                red_count=int(masks['red_rebuild'].sum());green_count=int(masks['keep_green'].sum());input_masks.append(pin(path))
            if raw in cfg['train_frames']:
                geometry_spec=byraw[raw]['background_geometry'];geometry=load_npz(checked(geometry_spec))['bg_geometry_scope'].astype(bool)
                assert geometry.shape==(h,w);roi|=geometry;geometry_count=int(geometry.sum());geometry_inputs.append(geometry_spec)
            observation_path=a.observations/f'{raw:03d}.json';observation=read(observation_path);input_observations.append(pin(observation_path))
            vehicles,box_count,a9_volumes=box_roi(observation,base,camera,(h,w),a.official_box_wh);roi|=vehicles
            full=base.packet(raw);parents=module.subset(full,candidate_ids)
            visible=base.native_visible_mask(raw,full)[candidate_ids]
            parents=dict(parents,mask=visible)
            out=base.scene._bindings().raster(camera,base.pipe,0,parents,base.scene._background)
            radii=out['radii'].detach().cpu().numpy().astype(np.float64);assert len(radii)==len(candidates)
            means=parents['means3D'];homogeneous=torch.cat([means,torch.ones_like(means[:,:1])],1)
            clip=homogeneous@camera.full_proj_transform
            centres=((clip[:,:2]/clip[:,3:4]+1)*clip.new_tensor([w,h])/2-.5).cpu().numpy().astype(np.float64)
            hit,bbox_count=tile_bbox_hits(roi,centres,radii,a.pad_pixels,16)
            ever_visible|=radii>0;new_rejected=int((hit&~rejected).sum());rejected|=hit
            if a9_volumes:
                inv=np.linalg.inv(base.transform);world=means.detach().cpu().numpy().astype(np.float64)@inv[:3,:3].T+inv[:3,3]
                for pose,dimensions in a9_volumes:
                    local_a9=(world-pose[:3,3])@pose[:3,:3]
                    a9_center|=(np.abs(local_a9)<=dimensions/2+.35).all(-1)
            rows.append(dict(raw=raw,ROI_pixels=int(roi.sum()),local_only_alpha_positive_pixels=int(local_roi.sum()),
                original_red_pixels=red_count,original_green_pixels=green_count,BG_geometry_scope_pixels=geometry_count,
                official_vehicle_rectangle_pixels=int(vehicles.sum()),projected_vehicle_boxes=box_count,
                preprocessed_positive_radius_rows=int((radii>0).sum()),nonempty_padded_tile_bboxes=bbox_count,
                intersecting_parent_rows=int(hit.sum()),newly_rejected_parent_rows=new_rejected))
            print(json.dumps(rows[-1]),flush=True)
            del local,full,parents,out
    restored=candidates[~rejected&~a9_center]
    assert np.all(restored<ns)and not np.isin(restored,fragment_parents).any()
    assert not np.any(semantic[restored]==10)
    updated={k:v.copy()for k,v in old.items()};updated['remove_global_row_ids']=np.setdiff1d(removed,restored,assume_unique=True)
    updated['restored_outside_static_global_row_ids']=restored
    assert np.array_equal(np.sort(updated['remove_global_row_ids'][updated['remove_global_row_ids']>=ns]),np.sort(removed[removed>=ns]))
    assert np.isin(fragment_parents,updated['remove_global_row_ids']).all()
    assert exact(before,rt.capture_model_state());rt.assert_fixed()
    retirement_path=a.output/'retirement_preserved.npz';np.savez_compressed(retirement_path,**updated)
    restored_path=a.output/'restored_static_global_row_ids.npz';np.savez_compressed(restored_path,global_row_ids=restored)
    receipt=dict(schema='scene065.outside_static_preservation.v1',complete=True,source_model=pin(model_path),source_runtime=pin(contract),
        source_retirement=retirement_spec,retirement=pin(retirement_path),restored_IDs=pin(restored_path),
        source_final_field_unchanged_exact=True,original_native_parameters_unchanged=True,optimizer_updates=0,training_updates=0,
        source_logical_step=2800,all20_frames=frames,native_static_rows=ns,native_dynamic_rows=base.native_dynamic_rows,
        original_retired_rows=len(removed),static_retired_rows=int((removed<ns).sum()),eligible_static_candidates=len(candidates),
        excluded_semantic_A9_static=int(excluded_A9.sum()),excluded_existing_fragment_parents=int(excluded_fragment_parent.sum()),
        excluded_A9_box_centres=int(a9_center.sum()),rejected_any_ROI_footprint=int(rejected.sum()),restored_static_rows=len(restored),
        restored_never_preprocessed_visible=int((~ever_visible[~rejected&~a9_center]).sum()),final_retired_rows=len(updated['remove_global_row_ids']),
        restored_dynamic_rows=0,original_fragment_recipe_unchanged=True,
        selection='Restore only original static parents with zero overlap against every protected ROI in all20 native cameras; never use occlusion to certify absence',
        ROI='local-only alpha>0 union original16 red+green masks union train8 fullBG geometry_scope union all official vehicle box rectangles',
        radius='Native raster preprocessing radii; bbox expanded by>=32px and rounded outward to16x16 tiles; original native near eligibility retained',
        radius_pad_pixels=a.pad_pixels,tile_size=[16,16],official_box_coordinate_size=a.official_box_wh,
        dynamic_or_A9_restoration=False,no_new_fragments=True,no_color_scale_or_pose_changes=True,
        protected_view_evidence=rows,mask_inputs=input_masks,observation_inputs=input_observations,geometry_inputs=geometry_inputs,
        native_local_forward_calls=rt.calls,native_candidate_radius_calls=len(frames),native_backward_calls=0,
        seconds=time.perf_counter()-started,finished_UTC=datetime.now(timezone.utc).isoformat())
    write(a.output/'receipt.json',receipt);print(json.dumps(dict(complete=True,retirement=str(retirement_path),restored_static_rows=len(restored),optimizer_updates=0,seconds=receipt['seconds'])),flush=True)

if __name__=='__main__':main()
