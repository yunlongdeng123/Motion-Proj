"""008道路观测链的输入适配：8cm父测距认证→2cm继承取色→独立推断。

无场景分支或查询期补色。坐标、原始注释和输入bank全部显式传入。范围由
给定查询射线图表决定，质量归一化用全部BUILD帧；这两项为新街段适配。
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,gaussian_filter,laplace
from scipy.spatial import cKDTree
from .v44_road_observation import (POLICY,LocalRoadSurface,transform,project,build_foreground_guard,
    certify_parent_sources,inherit_child_sources,sample_linear_rgb,source_view_score,
    select_reference,infer_asphalt_neighbours,linear,srgb,ray_box_blocked)
from .v44_handoff_policy import handoff_domains,target_mask_from_labels,rebind_other_vehicles

read=lambda p:json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def camera(bank,obs,frame_id,width):
    lines=(bank/'calib'/(frame_id+'.txt')).read_text().splitlines()
    cal={line.split(':')[0]:np.fromstring(line.split(':')[1],sep=' ') for line in lines}
    scale=width/obs['camera_size'][0];K=cal['P0'].reshape(3,4)[:,:3].copy();K[:2]*=scale
    C=np.eye(4);C[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4)
    return K,C@np.linalg.inv(np.array(obs['ego_world']).reshape(4,4))


def ray_map(K,w2local,wh,plane,xyz,normals,active,keys,cell,max_distance):
    width,height=wh;yy,xx=np.mgrid[:height,:width];inverse=np.linalg.inv(w2local)
    rays=np.stack((xx+.5,yy+.5,np.ones_like(xx)),2)@np.linalg.inv(K).T@inverse[:3,:3].T
    origin=inverse[:3,3];den=rays[:,:,2]-rays[:,:,:2]@plane[:2]
    with np.errstate(divide='ignore',invalid='ignore'):
        depth=(plane[2]+plane[:2]@origin[:2]-origin[2])/den
    lo=keys.min(0);shape=keys.max(0)-lo+1
    # Exactly one row for every 2cm key in the declared rectangle.
    def lookup(point):
        k=np.floor(np.nan_to_num(point[:,:,:2],nan=0,posinf=0,neginf=0)/cell).astype(np.int64)-lo
        inside=np.isfinite(point).all(2)&(k>=0).all(2)&(k<shape).all(2)
        safe=np.clip(k,[0,0],shape-1);index=safe[:,:,0]*shape[1]+safe[:,:,1]
        return index,inside
    for _ in range(8):
        hit=origin+depth[:,:,None]*rays;index,inside=lookup(hit)
        n=normals[index];den=(n*rays).sum(2)
        correction=np.divide((n*(xyz[index]-hit)).sum(2),den,out=np.zeros_like(depth),where=abs(den)>1e-7)
        usable=inside&active[index]&(depth>0)
        depth[usable]+=np.clip(correction[usable],-10,10)
    hit=origin+depth[:,:,None]*rays;index,inside=lookup(hit)
    inside&=(depth>0)&(np.linalg.norm(hit-origin,axis=2)<=max_distance)
    residual=abs((normals[index]*(xyz[index]-hit)).sum(2))
    surface_hit=inside&active[index]&(residual<.03)
    return index,inside,surface_hit,depth


def build(config,output):
    start=time.monotonic();cfg=read(config)
    if cfg.get('schema')!='v44.road_multiview.v1':raise ValueError('Explicit multiview source contract required')
    out=Path(output);out.mkdir(exist_ok=False);(out/'views').mkdir();(out/'parents').mkdir()
    bank,masks,annotation=(Path(cfg[k]) for k in ['bank','masks','raw_annotations'])
    receipt=read(bank/'receipt.json');raws=receipt['build_raws'];width,height=cfg['render_wh']
    if receipt['holdout_raws']:raise ValueError('Unexpected heldout input; current contract is all frames')
    ids={r['raw']:r['id'] for r in receipt['frames']};obs={r:read(bank/'observations'/f'{r:03d}.json') for r in raws}
    boxes=[b for d in obs.values() for b in d['boxes'] if b['track']==cfg['track_id']]
    A=np.array(boxes[0]['world_from_actor']).reshape(4,4);A[:3,3]=np.median([np.array(b['world_from_actor']).reshape(4,4)[:3,3] for b in boxes],axis=0)
    invA=np.linalg.inv(A)
    with np.load(cfg['geometry_seed']) as seed:
        keys=seed['keys'].copy();seed_xyz=seed['local_xyz'].copy();T=seed['model_from_world'].copy()
        assert np.allclose(transform(seed_xyz,A),seed['world_xyz'],atol=1e-7,rtol=0)
    plane=np.linalg.lstsq(np.c_[seed_xyz[:,:2],np.ones(len(seed_xyz))],seed_xyz[:,2],rcond=None)[0]
    low=seed_xyz[:,:2].min(0)-cfg['ground_pool_halo_m'];high=seed_xyz[:,:2].max(0)+cfg['ground_pool_halo_m']
    plane4=np.array([-plane[0],-plane[1],1.,-plane[2]]);plane4/=np.linalg.norm(plane4[:3])
    caches={};pool=[];sharp=[]
    for raw in raws:
        d=obs[raw];ann=read(annotation/f'{raw:03d}.json');assert ann['timestamp_micros']==d['timestamp_micros']
        cloud=np.fromfile(bank/d['lidar_path'],dtype='<f4').reshape(-1,6)[:,:3]
        local=transform(cloud,invA@np.array(d['ego_world']).reshape(4,4))
        centers=np.array([transform(np.array(b['center'])[None],invA)[0] for b in ann['boxes']])
        halves=np.array([b['half'] for b in ann['boxes']]);axes=np.array([invA[:3,:3]@np.array(b['axes']) for b in ann['boxes']])
        ground=abs(local@plane4[:3]+plane4[3])<=.15
        for c,h,ax in zip(centers,halves,axes):
            ix=np.flatnonzero(ground);ground[ix]&=~(abs((local[ix]-c)@ax)<=h).all(1)
        in_pool=ground&(local[:,:2]>=low).all(1)&(local[:,:2]<=high).all(1);pool.append(local[in_pool])
        labels=np.asarray(Image.open(masks/f'{raw:03d}/labels.png'));target=target_mask_from_labels(labels,[cfg['actor_id']])
        foreground=np.asarray(Image.open(masks/f'{raw:03d}/dynamic_union.png'))>0
        scale=width/d['camera_size'][0];rects=np.array([b['xyxy'] for b in ann['front_camera_rectangles']]).reshape(-1,4)*scale
        instance_guard=binary_dilation(foreground | (labels>0),iterations=2)
        guard=build_foreground_guard((height,width),rects,instance_guard,margin_px=2)
        image=np.asarray(Image.open(bank/d['image_path']).resize((width,height),Image.Resampling.LANCZOS),np.float32)/255
        sharp.append(float(np.var(laplace(image.mean(2)))))
        K,w2world=camera(bank,d,ids[raw],width);w2local=w2world@A
        caches[raw]=dict(cloud=local,ground=ground,centers=centers,halves=halves,axes=axes,guard=guard,
            image=image,K=K,w2local=w2local,labels=labels,target=target,foreground=foreground,
            box_tracks=[b['track'] for b in ann['boxes']])
    pool=np.concatenate(pool);pool_count=len(pool);_,unique=np.unique(np.floor(pool[:,:2]/.1).astype(np.int64),axis=0,return_index=True);pool=pool[unique]
    surface=LocalRoadSurface(pool,plane4)
    parent_keys,parent=np.unique(np.floor_divide(keys,4),axis=0,return_inverse=True)
    parent_xy=(parent_keys+.5)*.08;z,supported=surface.height(parent_xy);normal=surface.normal(parent_xy);parent_xyz=np.c_[parent_xy,z]
    xy=(keys+.5)*.02;delta=xy-parent_xy[parent];fine_normal=normal[parent].copy()
    xyz=np.c_[xy,z[parent]-(fine_normal[:,:2]*delta).sum(1)/np.maximum(fine_normal[:,2],.2)]
    P,N=len(parent_keys),len(keys);S=len(raws);quality=np.minimum(np.array(sharp)/max(np.median(sharp),1e-9),1.)
    rgb=np.zeros((S,N,3),np.float32);valid=np.zeros((S,N),bool);score=np.zeros((S,N),np.float32)
    parent_valid=np.zeros((S,P),bool);source_records=[]
    print('PARENT_GEOMETRY',P,'CHILD_GEOMETRY',N,'POOL',pool_count,len(pool),'SUPPORTED',int(supported.sum()),flush=True)
    for i,raw in enumerate(raws):
        c=caches[raw]
        result=certify_parent_sources(lidar_xyz=c['cloud'],lidar_ground=c['ground'],K=c['K'],world_to_camera=c['w2local'],
            query_xyz=parent_xyz,query_normals=normal,foreground_guard=c['guard'],surface_supported=supported,
            box_centers=c['centers'],box_halves=c['halves'],box_axes=c['axes'],source_quality=quality[i])
        parent_valid[i]=result['valid'];uv,depth=project(xyz,c['K'],c['w2local'])
        valid[i]=inherit_child_sources(result['valid'],parent,uv,depth,c['guard'])
        # Guarded finite coordinates; unqualified samples never enter selection.
        safe=np.where(np.isfinite(uv),uv,-1);rgb[i]=sample_linear_rgb(c['image'],safe)
        origin=np.linalg.inv(c['w2local'])[:3,3];score[i]=source_view_score(xyz,fine_normal,origin,depth,valid[i])
        np.savez_compressed(out/'parents'/f'{raw:03d}.npz',valid=result['valid'],candidate=result['candidate'],
            triangle_ids=result['triangle_ids'],measured_depth=result['measured_depth'],measured_conflict=result['measured_conflict'],
            box_blocked=result['box_blocked'],query_depth=result['query_depth'])
        source_records.append(dict(raw=raw,parent_valid=int(parent_valid[i].sum()),child_valid=int(valid[i].sum()),quality=float(quality[i])))
        if i%20==0:print('CERTIFIED_SOURCE',source_records[-1],time.monotonic()-start,flush=True)
    selected=select_reference(rgb,valid,score);real=selected['real'];reference=selected['reference_linear']
    inferred_result=infer_asphalt_neighbours(xyz,reference,real);reference=inferred_result['reference_linear'];inferred=inferred_result['inferred']
    active=real|inferred;geom_parent=parent.copy()
    if real.any() and inferred.any():
        real_ids=np.flatnonzero(real);_,near=cKDTree(xy[real_ids]).query(xy[inferred]);gid=parent[real_ids[near]];geom_parent[inferred]=gid
        fine_normal[inferred]=normal[gid];delta=xy[inferred]-parent_xy[gid]
        xyz[inferred,2]=parent_xyz[gid,2]-(normal[gid,:2]*delta).sum(1)/np.maximum(normal[gid,2],.2)
    # Original fixed clean-neighbor native colour alignment, not per-query paste.
    shape=keys.max(0)-keys.min(0)+1;assert np.prod(shape)==N
    offset_sum=np.zeros((P,3));offset_count=np.zeros(P)
    reference_luma=reference@np.array([.2126,.7152,.0722])
    for i,raw in enumerate(raws):
        c=caches[raw];uv,_=project(xyz,c['K'],c['w2local']);safe=np.where(np.isfinite(uv),uv,-1)
        native=np.load(Path(cfg['native_eval'])/f'{raw:03d}/keep.npz')['rgb'].transpose(1,2,0)
        native_linear=sample_linear_rgb(native,safe);offset=native_linear-rgb[i]
        native_luma=native_linear@np.array([.2126,.7152,.0722])
        shadow_deficit=(reference_luma-native_luma>.025)&(native_luma<.8*np.maximum(reference_luma,1e-6))
        clean=selected['source_agree'][i]&real&inferred_result['asphalt']&(abs(offset).max(1)<.05)&~shadow_deficit
        offset_count+=np.bincount(parent[clean],minlength=P)
        for ch in range(3):offset_sum[:,ch]+=np.bincount(parent[clean],weights=offset[clean,ch],minlength=P)
    parent_shape=parent_keys.max(0)-parent_keys.min(0)+1;assert np.prod(parent_shape)==P
    den=gaussian_filter(offset_count.reshape(parent_shape),(.8/.08,.4/.08))
    parent_correction=np.stack([gaussian_filter(offset_sum[:,ch].reshape(parent_shape),(.8/.08,.4/.08))/np.maximum(den,1e-8) for ch in range(3)],-1).reshape(P,3)
    parent_correction=np.clip(parent_correction,-.025,.025);correction=parent_correction[parent]
    uncalibrated=reference.copy();reference=np.clip(reference+correction,0,1)
    world=transform(xyz,A);normal_world=fine_normal@A[:3,:3].T;scale=np.cbrt(np.linalg.det(T[:3,:3]))
    best=np.where(real,np.array(raws)[selected['chosen_index']],-1);colours=srgb(reference)
    marked=real&((colours.max(1)>=.65)|(np.ptp(colours,axis=1)>=.18))
    np.savez_compressed(out/'reference.npz',active=active,real=real,inferred=inferred,marked=marked,keys=keys,
        local_xyz=xyz,world_xyz=world,model_xyz=transform(world,T),normal_world=normal_world,gaussian_sigma_m=np.tile([.013,.013,.003],(N,1)),
        road_opacity=np.full(N,.99,np.float32),reference_linear=reference,reference_srgb=srgb(reference).astype(np.float32),
        uncalibrated_linear=uncalibrated,correction_linear=correction,source_valid=valid,source_agree=selected['source_agree'],
        source_raw=np.array(raws),best_source_raw=best,model_from_world=T,meter_to_model=scale,
        parent_ids=parent,geometry_parent_ids=geom_parent,parent_keys=parent_keys,parent_xyz=parent_xyz,parent_normals=normal,
        parent_surface_supported=supported,parent_source_valid=parent_valid,cell_m=.02,parent_cell_m=.08,
        parent_native_correction=parent_correction)
    rows=[]
    for raw,c in caches.items():
        index,inside,hit,depth=ray_map(c['K'],c['w2local'],(width,height),plane,xyz,fine_normal,active,keys,.02,cfg['max_ground_distance_m'])
        sem=np.asarray(Image.open(Path(cfg['semantics'])/f'{raw:03d}.png').resize((width,height),Image.Resampling.NEAREST))
        other=rebind_other_vehicles(c['foreground']&~c['target'],c['labels'],[cfg['actor_id']])
        observed=hit&~c['guard']&(sem==cfg['road_semantic_id']);reveal=hit&c['target']&~other
        # Occluded retained objects are not road supervision. OBB is a
        # conservative uncertainty guard, not an opaque surface ground truth.
        inverse=np.linalg.inv(c['w2local']);origin=inverse[:3,3]
        yy,xx=np.mgrid[:height,:width];rays=np.stack((xx+.5,yy+.5,np.ones_like(xx)),2)@np.linalg.inv(c['K']).T@inverse[:3,:3].T
        query_xyz=origin+depth[:,:,None]*rays
        retained=np.array([track!=cfg['track_id'] for track in c['box_tracks']])
        retained_box_guard=np.zeros((height,width),bool);target_ids=np.flatnonzero(c['target'].ravel()&inside.ravel())
        if len(target_ids):
            retained_box_guard.ravel()[target_ids]=ray_box_blocked(query_xyz.reshape(-1,3)[target_ids],origin,
                c['centers'][retained],c['halves'][retained],c['axes'][retained])
        with np.load(Path(cfg['retained_baseline'])/'frames'/f'{raw:06d}/arrays.npz') as base:
            names=base['branch_alpha_names'].tolist()
            if 'nonfocus_dynamic' not in names:raise ValueError('Missing independent dynamic reveal diagnostic')
            dynamic=base['branch_alpha_T'][names.index('nonfocus_dynamic')].squeeze()
        retained_reveal_guard=c['target']&(dynamic>=.25)
        reveal&=~(retained_box_guard|retained_reveal_guard)
        fit=(observed|reveal)*np.where(inferred[index],.25,1.)
        marking=inside&marked[index]&~c['target']&~other
        with np.load(Path(cfg['seed_views'])/f'{raw:03d}.npz') as old:marking|=old['marking']&~c['target']&~other
        fit[other|marking]=0;keep=np.ones((height,width),np.float32);keep[observed|reveal]=0
        dom=handoff_domains(dict(fit=fit.astype(np.float32),keep=keep,inside=inside,other=other,marking=marking),c['target'])
        np.savez_compressed(out/'views'/f'{raw:03d}.npz',index=index,inside=inside,fit=dom['fit'],keep=dom['keep'],marking=marking,other=other,
            target=c['target'],target_actor_ids=np.array([cfg['actor_id']],np.int64),candidate_support=dom['candidate_support'],left=dom['left'],
            unknown_target=dom['unknown_target'],reference_linear=reference[index].transpose(2,0,1).astype(np.float32),
            real_domain=(fit>.05)&real[index],inferred_domain=(fit>.05)&inferred[index],observed_road=observed,ground_depth=depth.astype(np.float32),
            retained_reveal_guard=retained_reveal_guard,retained_box_guard=retained_box_guard)
        rows.append(dict(raw=raw,target_pixels=int(c['target'].sum()),fit_target_pixels=int((c['target']&(fit>.05)).sum()),
            real_target=int((c['target']&(fit>.05)&real[index]).sum()),inferred_target=int((c['target']&(fit>.05)&inferred[index]).sum()),
            unknown_target=int(dom['unknown_target'].sum()),retained_dynamic_guard=int(retained_reveal_guard.sum()),retained_box_guard=int(retained_box_guard.sum())))
    write(out/'views/binding.json',dict(schema='v44.handoff_views.v1',delete_actor_ids=[cfg['actor_id']],track_ids=[cfg['track_id']],
        policy='full_target_candidate_and_preservation_v1',source_method='008_parent_triangle_child_inheritance',reference_added=True))
    write(out/'receipt.json',dict(status='BUILT_008_MULTIVIEW_ROAD_SOURCE_CHAIN',parent_cells=P,child_cells=N,active_cells=int(active.sum()),
        parent_surface_supported=int(supported.sum()),real_cells=int(real.sum()),inferred_cells=int(inferred.sum()),unknown_cells=int((~active).sum()),
        ground_pool_raw=pool_count,ground_pool_deduplicated=len(pool),plane_anchor=plane4.tolist(),coordinate_anchor=A.tolist(),
        source_records=source_records,rows=rows,source_frames=raws,heldout_frames=[],policy=POLICY,
        child_lidar_independently_certified=False,source_rgb='bilinear in sRGB then linear RGB; Lanczos prepared source image',
        inference='8 nearest reliable asphalt colors; nearest distance<=.60m, weight1/max(d,.02)^2, confidence.25; not observed truth',
        adapters=['query-ray-bounded chart supplied explicitly instead of scene008 footprint','all BUILD median sharpness normalization instead of scene00889..137',
            'new input-bank measured plane; original Surface local residual heights/normals retained'],
        source_qualification='actual FRONT camera labels +2pixel and accepted mask+2pixel union; ALL laser OBB; measured parent triangles; not semantic-road-only',
        native_colour_alignment='8cm parent clean asphalt native-minus-reference; .8m/.4m smooth capped±.025; excludes shadow deficit; child inherits',
        query_reveal_guard='remaining OBB intersections and independent rendered nonfocus dynamic>=.25; uncertainty excludes photo, not hidden truth',
        native_updates=0,seconds=time.monotonic()-start,config=cfg))
    print('COMPLETE',int(real.sum()),int(inferred.sum()),time.monotonic()-start,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();build(a.config,a.output)
if __name__=='__main__':main()
