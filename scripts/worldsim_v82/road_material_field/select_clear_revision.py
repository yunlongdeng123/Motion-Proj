"""固定几何上的清晰单源道路图块选择；按原图投影分辨率预先准入，不看模型结果。"""
from __future__ import annotations
import os
if __name__ == '__main__':
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,copy,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
if __package__:
    from .build_road_dataset import make_guard,save_image
    from .world_feature_field import transform_xyz,project_world_to_camera,bilinear_sample,camera_from_observation,camera_ground_rays
else:
    from build_road_dataset import make_guard,save_image
    from world_feature_field import transform_xyz,project_world_to_camera,bilinear_sample,camera_from_observation,camera_ground_rays


def jwrite(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def build(base,output):
    start=time.monotonic();base=Path(base);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name in ['atlas','samples','candidates','sources','observations','all_frame_query_masks']:(out/name).mkdir()
    old=json.loads((base/'dataset.json').read_text());cfg=old['source_config'];m=copy.deepcopy(old)
    bank=Path(cfg['bank']);masks=Path(cfg['masks']);semantics=Path(cfg['semantics']);anns=Path(cfg['annotations'])
    records=json.loads((base/'sources.json').read_text())['sources'];caches={};wh=tuple(cfg['source_wh']);w,h=wh
    for record in records:
        raw=record['raw'];obs=json.loads((bank/'observations'/f'{raw:03d}.json').read_text())
        road,_,_=make_guard(masks,semantics,anns,raw,wh,obs,cfg['road_semantic_id'])
        image=np.asarray(Image.open(record['image_path']).convert('RGB'),np.float32)/255
        caches[raw]=dict(record=record,road=road,image=image)
    shape=old['atlas']['shape_hw'];atlas_matrix=np.asarray(old['atlas']['world_from_uv']);cell=old['atlas']['cell_m']
    world=np.load(base/old['atlas']['world_xyz'],mmap_mode='r');observed=np.load(base/old['atlas']['supervision_only_valid']).reshape(shape)
    coarse=64;size=128;factor=2;min_span=48.;yy,xx=np.mgrid[:size,:size]
    unit=np.c_[xx.ravel()+.5,yy.ravel()+.5,np.zeros(size*size)]
    def project(points,record):
        return project_world_to_camera(points,**{k:record[k] for k in ['camera_K_native','world_to_camera','image_hw','distortion_Brown']})
    def warp(points,raw):
        c=caches[raw];s=c['record'];uv,depth,inside=project(points.reshape(-1,3),s)
        safe=np.clip(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1),-1e6,1e6)
        p=np.clip(np.floor(safe*[w/s['image_hw'][1],h/s['image_hw'][0]]).astype(int),[0,0],[w-1,h-1])
        valid=(inside&c['road'][p[:,1],p[:,0]]).reshape(size,size)
        rgb=bilinear_sample(c['image'],uv,s['image_hw']).reshape(size,size,3).astype(np.float32);rgb[~valid]=.5
        origin=np.linalg.inv(np.asarray(s['world_to_camera']))[:3,3];normal=np.asarray(old['atlas']['coordinate_anchor'])[:3,2]
        view=origin-points.reshape(-1,3);distance=np.linalg.norm(view,axis=1)
        weight=(np.maximum(view@normal,0)/np.maximum(distance,1e-6)/(distance**2+4.)).reshape(size,size)
        return rgb,valid,uv.reshape(size,size,2),weight
    pool=[];scanned=0
    for y in range(0,shape[0]-coarse+1,coarse):
        for x in range(0,shape[1]-coarse+1,coarse):
            if observed[y:y+coarse,x:x+coarse].mean()<.97:continue
            matrix=atlas_matrix.copy();matrix[:3,3]=transform_xyz(np.array([[x,y,0.]]),atlas_matrix)[0];matrix[:3,:2]/=factor
            # 同一质量准入：源图上图块中心的两个切向投影Jacobian最小轴跨度。
            test=np.array([[64.5,64.5,0],[80.5,64.5,0],[48.5,64.5,0],[64.5,80.5,0],[64.5,48.5,0]])
            test=transform_xyz(test,matrix);quality=[]
            for raw,c in caches.items():
                uv,_,inside=project(test,c['record'])
                if not inside.all():continue
                jac=np.column_stack(((uv[1]-uv[2])/32,(uv[3]-uv[4])/32));span=float(np.linalg.svd(jac,compute_uv=False).min()*size)
                if span>=min_span:quality.append((span,raw))
            quality.sort(reverse=True);scanned+=1
            for span,raw in quality:
                points=transform_xyz(unit,matrix).reshape(size,size,3)
                rgb,valid,uv,weight=warp(points,raw)
                if valid.mean()<.98 or not valid[32:96,32:96].all():continue
                sy,sx=np.mgrid[y:y+coarse,x:x+coarse];ids=sy*shape[1]+sx
                ids=np.repeat(np.repeat(ids,2,axis=0),2,axis=1).astype(np.int32)
                marking=((rgb.min(2)>.60)&(np.ptp(rgb,axis=2)<.18)&valid).mean()>.01
                pool.append(dict(x=x,y=y,matrix=matrix,world=points.astype(np.float32),ids=ids,rgb=rgb,valid=valid,
                    uv=uv.astype(np.float32),weight=weight.astype(np.float32),raw=raw,min_axis_span_native_px=span,marking=bool(marking)))
                break
    preferred=[t for t in pool if t['min_axis_span_native_px']>=64.]
    preferred_count=len(preferred);preferred_marking=sum(t['marking'] for t in preferred)
    if preferred_count>=16 and preferred_marking>=4:
        pool=preferred;min_span=64.
    print('CLEAR_POOL',len(pool),'marking',sum(t['marking'] for t in pool),'gate',min_span,'preferred64',preferred_count,
          'seconds',time.monotonic()-start,flush=True)
    if len(pool)<8 or sum(t['marking'] for t in pool)<3:raise ValueError('Insufficient clear/marking strata')
    rng=np.random.default_rng(old['seed']);marked=[i for i,t in enumerate(pool) if t['marking']];plain=[i for i,t in enumerate(pool) if not t['marking']]
    rng.shuffle(marked);rng.shuffle(plain)
    eval_count=4 if len(pool)>=24 and len(marked)>=4 else 2
    eval_idx=marked[:2]+plain[:eval_count-2]
    remaining=[i for i in range(len(pool)) if i not in eval_idx];rng.shuffle(remaining)
    train_count=min(12 if len(pool)>=32 else 8,len(remaining)-6)
    train_marked=[i for i in marked if i not in eval_idx][:max(1,train_count//4)]
    train_idx=train_marked+[i for i in remaining if i not in train_marked][:train_count-len(train_marked)]
    candidate_idx=[i for i in range(len(pool)) if i not in set(train_idx+eval_idx)]
    # 候选按质量排序后固定截断，避免先扫描远处世界区域形成分辨率偏置。
    candidate_idx=sorted(candidate_idx,key=lambda i:-pool[i]['min_axis_span_native_px'])[:48]
    n=len(world);feature_excluded=np.zeros(n,bool);candidate_excluded=np.zeros(n,bool);eval_world=np.zeros(n,bool)
    samples=[]
    def write_tile(tile,directory,hole):
        directory.mkdir();rgb=tile['rgb'];valid=tile['valid'];inp=rgb.copy();inp[hole]=.5
        for name,value in [('target',rgb),('input',inp),('valid',valid),('hole',hole)]:save_image(directory/(name+'.png'),value)
        np.save(directory/'world_xyz.npy',tile['world']);np.save(directory/'point_ids.npy',tile['ids'])
        np.save(directory/'source_uv_native.npy',tile['uv']);np.save(directory/'source_weight.npy',tile['weight'])
        np.savez_compressed(directory/'tile.npz',target=rgb,input=inp,valid=valid,hole=hole,world_xyz=tile['world'],
            atlas_indices=tile['ids'],world_from_uv=tile['matrix'],uv_native=tile['uv'],weight=tile['weight'])
    for number,(index,split) in enumerate([(i,'train') for i in train_idx]+[(i,'eval') for i in eval_idx]):
        tile=pool[index];sid=f'{old["scene_id"]}_{split}_{number:03d}';d=out/'samples'/sid
        hole=((abs(xx-64)<26)&(abs(yy-64)<28)) if number%2==0 else (((xx-64)/27)**2+((yy-64)/29)**2<1)
        assert tile['valid'][hole].all();write_tile(tile,d,hole)
        feature_excluded[tile['ids'][hole]]=True
        if split=='eval':eval_world[np.unique(tile['ids'])]=True
        prefix=f'samples/{sid}/';s=dict(id=sid,split=split,scene_id=old['scene_id'],
            **{k:prefix+v for k,v in [('target','target.png'),('input','input.png'),('reference','reference.png'),('hole_mask','hole.png'),
                ('valid_mask','valid.png'),('reference_valid_mask','reference_valid.png'),('world_xyz','world_xyz.npy'),('point_ids','point_ids.npy'),
                ('source_uv_native','source_uv_native.npy'),('source_weight','source_weight.npy'),('tile_npz','tile.npz')]},
            hole=prefix+'hole.png',valid=prefix+'valid.png',world_from_uv=tile['matrix'].tolist(),
            world_bounds=[tile['world'].min((0,1)).tolist(),tile['world'].max((0,1)).tolist()],
            atlas_pixel_bounds=[tile['x'],tile['y'],tile['x']+coarse,tile['y']+coarse],tile_size_m=2.56,tile_texel_m=.02,atlas_feature_cell_m=.04,
            target_source_raw=tile['raw'],target_source_image=caches[tile['raw']]['record']['image_path'],
            min_axis_span_native_px=tile['min_axis_span_native_px'],marking_stratum=tile['marking'],valid_pixels=int(tile['valid'].sum()),
            hole_pixels=int(hole.sum()),valid_fraction=float(tile['valid'].mean()),unique_atlas_cells=4096,
            provenance=dict(target='single_source_clean_visible_road_planar_warp',reference='different_world_clean_material_reference',
                geometry='fixed_measured_plane_no_residual',evaluation='visible_road_pseudo_occlusion_only',generated=False,true_hidden_ground_truth=False))
        samples.append(s)
    candidates=[]
    for number,index in enumerate(candidate_idx):
        tile=pool[index];cid=f'candidate_{number:03d}';d=out/'candidates'/cid;write_tile(tile,d,np.zeros((size,size),bool))
        prefix=f'candidates/{cid}/';candidates.append(dict(id=cid,input=prefix+'input.png',valid_mask=prefix+'valid.png',tile_npz=prefix+'tile.npz',
            world_xyz=prefix+'world_xyz.npy',point_ids=prefix+'point_ids.npy',atlas_pixel_bounds=[tile['x'],tile['y'],tile['x']+coarse,tile['y']+coarse],
            target_source_raw=tile['raw'],min_axis_span_native_px=tile['min_axis_span_native_px'],marking_stratum=tile['marking'],
            valid_fraction=float(tile['valid'].mean()),world_from_uv=tile['matrix'].tolist(),
            role='different_world_clean_single_source_visible_material_reference',query_or_eval_world_overlap_cells=0))
    # 参考独立取候选库，所有参考满足与train/eval相同的质量准入，且无人工洞。
    for s in samples:
        own=np.load(out/s['point_ids']);owncenter=np.mean(np.asarray(s['world_bounds']),axis=0)
        c=max(candidates,key=lambda c:np.linalg.norm(np.mean(np.load(out/c['world_xyz']),axis=(0,1))-owncenter))
        assert np.intersect1d(own,np.load(out/c['point_ids'])).size==0
        save_image(out/s['reference'],np.asarray(Image.open(out/c['input'])));save_image(out/s['reference_valid_mask'],np.asarray(Image.open(out/c['valid_mask'])))
        s.update(reference_sample_id=c['id'],reference_candidate_id=c['id'],reference_source_raw=c['target_source_raw'],
            reference_world_bounds=[np.load(out/c['world_xyz']).min((0,1)).tolist(),np.load(out/c['world_xyz']).max((0,1)).tolist()],
            reference_overlap_world_cell_count=0,reference_semantics='visible_road_or_lane_marking_appearance')
    candidate_excluded=feature_excluded|eval_world
    for path in (base/'atlas').iterdir():
        if path.name not in ['exclude_world_mask.npy','feature_exclude_world_mask.npy','eval_world_mask.npy','input_rgb.npy','observed_valid.npy']:
            (out/'atlas'/path.name).symlink_to(path.resolve())
    observed_flat=np.asarray(observed).ravel();rgb=np.load(base/old['atlas']['supervision_only_rgb']).copy();rgb[candidate_excluded]=.5
    np.save(out/'atlas/input_rgb.npy',rgb);np.save(out/'atlas/observed_valid.npy',observed_flat&~candidate_excluded)
    for name,value in [('exclude_world_mask',candidate_excluded),('feature_exclude_world_mask',feature_excluded),('eval_world_mask',eval_world)]:np.save(out/'atlas'/(name+'.npy'),value)
    selected_union=np.unique(np.concatenate([pool[i]['ids'].ravel() for i in train_idx+eval_idx+candidate_idx]));query_ids=np.flatnonzero(feature_excluded)
    plane=np.asarray(old['atlas']['plane']);local_from_world=np.linalg.inv(np.asarray(old['atlas']['coordinate_anchor']))
    low_local=np.floor(np.load(base/old['atlas']['local_xyz'],mmap_mode='r')[:1,:2]/cell).astype(int)[0]
    source_by_raw={r['raw']:copy.deepcopy(r) for r in records};mask_rows=[]
    for raw in old['all_input_frames']:
        obs=json.loads((bank/'observations'/f'{raw:03d}.json').read_text());camera=camera_from_observation(obs)
        hits,depth=camera_ground_rays(camera,wh,local_from_world,plane)
        key=np.floor(np.nan_to_num(hits[:,:,:2],nan=0,posinf=0,neginf=0)/cell).astype(np.int64)-low_local
        inside=np.isfinite(hits).all(2)&(depth>.1)&(key>=0).all(2)&(key<np.array(shape[::-1])).all(2)
        safe=np.clip(key,[0,0],np.array(shape[::-1])-1);ids=safe[:,:,1]*shape[1]+safe[:,:,0]
        blocked=inside&feature_excluded[ids]
        uv,_,positive=project(world[query_ids],camera);p=np.clip(np.floor(uv[positive]*[w/camera['image_hw'][1],h/camera['image_hw'][0]]).astype(int),[0,0],[w-1,h-1])
        blocked[p[:,1],p[:,0]]=True;blocked=binary_dilation(blocked,iterations=4)
        save_image(out/'all_frame_query_masks'/f'{raw:03d}.png',blocked);mask_rows.append(dict(raw=raw,query_pixels=int(blocked.sum())))
        if raw not in source_by_raw:continue
        c=caches[raw];record=source_by_raw[raw];allowed=(~blocked)&c['road']
        native_allowed=np.asarray(Image.fromarray(allowed.astype(np.uint8)*255).resize(c['image'].shape[1::-1],Image.Resampling.NEAREST))>0
        encoded=c['image'].copy();encoded[~native_allowed]=.5
        save_image(out/'sources'/f'{raw:03d}_feature_input.png',encoded);save_image(out/'sources'/f'{raw:03d}_feature_allowed.png',native_allowed)
        uv,depth,positive=project(world[selected_union],record)
        safe_uv=np.clip(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1),-1e6,1e6)
        p=np.clip(np.floor(safe_uv*[w/record['image_hw'][1],h/record['image_hw'][0]]).astype(int),[0,0],[w-1,h-1])
        good=positive&allowed[p[:,1],p[:,0]]&~feature_excluded[selected_union]
        ids=selected_union[good];origin=np.linalg.inv(np.asarray(record['world_to_camera']))[:3,3]
        view=origin-world[ids];distance=np.linalg.norm(view,axis=1);normal=np.asarray(old['atlas']['coordinate_anchor'])[:3,2]
        weight=np.maximum(view@normal,0)/np.maximum(distance,1e-6)/(distance**2+4.)
        np.savez_compressed(out/'observations'/f'{raw:03d}.npz',point_ids=ids.astype(np.int32),uv_native=uv[good].astype(np.float32),weight=weight.astype(np.float32),depth=depth[good].astype(np.float32))
        record.update(observations=f'observations/{raw:03d}.npz',feature_input=f'sources/{raw:03d}_feature_input.png',
            feature_allowed_mask=f'sources/{raw:03d}_feature_allowed.png',observations_count=len(ids),
            observation_role='legal_visibility_on_all_clear_material_samples_and_candidates; one observation per world cell per source')
    m.update(samples=samples,candidates=candidates,image_size=[size,size],train=[s['id'] for s in samples if s['split']=='train'],eval=[s['id'] for s in samples if s['split']=='eval'],
        data_revision='r3_clear_source_pixel_support',quality_gate=dict(native_min_axis_span_px=min_span,valid_fraction_min=.98,central_valid_fraction=1.,
            applied_to='all train/eval/reference/candidate tiles before split',pool_size=len(pool),pool_marking_tiles=sum(t['marking'] for t in pool),
            preferred_64px_pool_size=preferred_count,preferred_64px_marking_tiles=preferred_marking,
            min_axis_span_native_px_min=float(min(t['min_axis_span_native_px'] for t in pool)),
            min_axis_span_native_px_median=float(np.median([t['min_axis_span_native_px'] for t in pool])),
            min_axis_span_native_px_max=float(max(t['min_axis_span_native_px'] for t in pool))),
        split_policy=dict(kind='fixed_seed_disjoint_world_tiles_with_predefined_marking_strata',seed=old['seed'],
            train_internal_overlap_world_cells=0,eval_internal_overlap_world_cells=0,train_eval_overlap_world_cells=0,
            eval_marking_tiles=sum(s['marking_stratum'] for s in samples if s['split']=='eval'),cross_street_generalization=False),
        sources='sources.json',source_observation_domain='all selected clear road samples and candidate tiles, fixed .04m feature atlas')
    m['coordinate']['dual_resolution']='road material tile: .02m/texel; feature atlas: .04m/cell; atlas indices repeat2x2 per coarse world cell'
    jwrite(out/'dataset.json',m);jwrite(out/'sources.json',dict(schema='road_material_sources.v1',sources=list(source_by_raw.values())))
    jwrite(out/'train.json',m['train']);jwrite(out/'eval.json',m['eval'])
    def contact(subset,name):
        canvas=Image.new('RGB',(90+size*6,(size+24)*len(subset)),(20,24,32));draw=ImageDraw.Draw(canvas)
        for row,s in enumerate(subset):
            for col,key in enumerate(['target','input','reference','hole_mask','valid_mask','reference_valid_mask']):canvas.paste(Image.open(out/s[key]).convert('RGB'),(90+col*size,row*(size+24)))
            draw.text((2,row*(size+24)+8),s['id'].split('_')[-2]+' '+s['id'].split('_')[-1],fill='white')
            draw.text((90,row*(size+24)+size+3),'target | input | reference | hole | valid | reference valid',fill='white')
        canvas.save(out/name)
    contact(samples[:4]+[s for s in samples if s['split']=='eval'],'contact_sheet.png');contact([s for s in samples if s['split']=='eval'],'contact_eval.png')
    contact([s for s in samples if s['split']=='train'],'contact_train.png')
    jwrite(out/'selection_receipt.json',dict(status='BUILT_CLEAR_SINGLE_SOURCE_FIXED_GEOMETRY_REVISION',seconds=time.monotonic()-start,
        train=len(m['train']),eval=len(m['eval']),candidates=len(candidates),quality_gate=m['quality_gate'],split_policy=m['split_policy'],
        all_frame_masks=mask_rows,selected_unique_world_cells=len(selected_union),selected_unique_world_area_m2=float(len(selected_union)*cell**2),
        extra_model_updates=0,extra_native_updates=0,geometry_changed=False))
    print('R3_READY',out,'seconds',time.monotonic()-start,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.base,a.output)
