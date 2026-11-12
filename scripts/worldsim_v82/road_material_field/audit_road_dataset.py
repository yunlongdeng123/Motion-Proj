"""对实际生成的数据包做空间隔离、来源掩码和投影验收。"""
from __future__ import annotations
import os
if __name__ == '__main__':
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
if __package__:
    from .world_feature_field import project_world_to_camera
else:
    from world_feature_field import project_world_to_camera


def audit(root):
    root=Path(root);manifest=json.loads((root/'dataset.json').read_text())
    sources=json.loads((root/manifest.get('sources','sources.json')).read_text());world=np.load(root/manifest['atlas']['world_xyz'],mmap_mode='r')
    feature_excluded=np.load(root/manifest['atlas']['feature_exclude_world_mask'])
    candidate_excluded=np.load(root/manifest['atlas']['exclude_world_mask'])
    supervised=np.load(root/manifest['atlas']['supervision_only_valid'])
    shape=manifest['atlas']['shape_hw'];n=len(world)
    domains={k:np.zeros(n,bool) for k in ['train','eval']};queries=np.zeros(n,bool)
    sample_by_id={s['id']:s for s in manifest['samples']}
    references_by_id={**sample_by_id,**{s['id']:s for s in manifest.get('candidates',[])}};rows=[]
    for s in manifest['samples']:
        ids=np.load(root/s['point_ids']);hole=np.asarray(Image.open(root/s['hole_mask']))>0
        valid=np.asarray(Image.open(root/s['valid_mask']))>0
        assert not (hole & ~valid).any()
        assert supervised[ids[hole]].all()
        assert not domains[s['split']][ids].any()
        domains[s['split']][ids]=True;queries[ids[hole]]=True
        reference=references_by_id[s['reference_sample_id']]
        if 'point_ids' in reference:
            ref_ids=np.load(root/reference['point_ids'])
        else:
            with np.load(root/reference['tile_npz']) as d:ref_ids=d['atlas_indices']
        overlap=np.intersect1d(ids,ref_ids).size
        assert overlap==0
        s['reference_overlap_world_cell_count']=int(overlap)
        s['reference_role']='different_world_material_reference'
        inp=np.asarray(Image.open(root/s['input']));tar=np.asarray(Image.open(root/s['target']))
        assert (inp[hole]==128).all();assert (inp[~valid]==128).all();assert (tar[~valid]==128).all()
        assert np.array_equal(inp[valid&~hole],tar[valid&~hole])
        rows.append(dict(id=s['id'],split=s['split'],hole_pixels=int(hole.sum()),valid_pixels=int(valid.sum()),reference_overlap_world_cells=0))
    assert not (domains['train'] & domains['eval']).any()
    assert not (queries & ~feature_excluded).any()
    assert candidate_excluded[domains['eval']].all()
    query_ids=np.flatnonzero(queries);source_rows=[];filtered=0
    for source in sources['sources']:
        camera={k:source[k] for k in ['camera_K_native','world_to_camera','image_hw','distortion_Brown']}
        uv,_,inside=project_world_to_camera(world[query_ids],**camera)
        points=np.floor(uv[inside]).astype(int);points=np.clip(points,[0,0],np.asarray(source['image_hw'][::-1])-1)
        allowed=np.asarray(Image.open(root/source['feature_allowed_mask']))>0
        leak=int(allowed[points[:,1],points[:,0]].sum());assert leak==0,(source['raw'],leak)
        image=np.asarray(Image.open(root/source['feature_input']))
        assert (image[~allowed]==128).all()
        path=root/source['observations']
        with np.load(path) as f:values={k:f[k].copy() for k in f.files}
        ids=values['point_ids'];assert not queries[ids].any();assert len(np.unique(ids))==len(ids)
        pp=np.clip(np.floor(values['uv_native']).astype(int),[0,0],np.asarray(source['image_hw'][::-1])-1)
        # 投影保护带中变灰的观测不再参与世界场融合，灰色不构成真实观测。
        keep=allowed[pp[:,1],pp[:,0]];removed=int((~keep).sum());filtered+=removed
        if removed:
            np.savez_compressed(path,**{k:v[keep] for k,v in values.items()})
        source['observations_count']=int(keep.sum())
        source['additional_projection_guard_removed']=removed
        source_rows.append(dict(raw=source['raw'],projected_query_cells=int(inside.sum()),query_rgb_leak_pixels=leak,
            duplicate_world_observations=0,projection_guard_removed=removed,legal_observations=int(keep.sum())))
    # 完整异地候选只从合法观测域扫描，绝不读取supervision_rgb。
    legal=np.load(root/manifest['atlas']['observed_valid']).reshape(shape)
    excluded=candidate_excluded.reshape(shape);size=manifest['image_size'][0];candidates=[]
    for y in range(0,shape[0]-size+1,size//2):
        for x in range(0,shape[1]-size+1,size//2):
            bad=excluded[y:y+size,x:x+size]
            fraction=float(legal[y:y+size,x:x+size].mean())
            if not bad.any() and fraction>=.65:
                candidates.append(dict(id=f'candidate_{x}_{y}',atlas_pixel_bounds=[x,y,x+size,y+size],valid_fraction=fraction,
                    role='different_world_material_reference',query_or_eval_world_overlap_cells=0))
    if manifest.get('candidate_contract'):
        candidates=manifest['candidates']
    else:
        manifest['candidates']=candidates
    (root/'dataset.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (root/manifest.get('sources','sources.json')).write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    receipt=dict(status='PASS_REAL_DATA_WORLD_SPLIT_AND_SOURCE_MASK_AUDIT',train=len(manifest['train']),eval=len(manifest['eval']),
        atlas_shape_hw=shape,world_cells=n,train_eval_overlap_world_cells=int((domains['train']&domains['eval']).sum()),
        train_world_cells=int(domains['train'].sum()),eval_world_cells=int(domains['eval'].sum()),query_world_cells=len(query_ids),
        train_internal_overlap_world_cells=0,eval_internal_overlap_world_cells=0,
        train_unique_world_area_m2=float(domains['train'].sum()*manifest['atlas']['cell_m']**2),
        eval_unique_world_area_m2=float(domains['eval'].sum()*manifest['atlas']['cell_m']**2),
        query_unique_world_area_m2=float(len(query_ids)*manifest['atlas']['cell_m']**2),
        train_supported_visible_area_m2=float(sum(r['valid_pixels'] for r in rows if r['split']=='train')*
            manifest['samples'][0].get('tile_texel_m',manifest['atlas']['cell_m'])**2),
        eval_supported_visible_area_m2=float(sum(r['valid_pixels'] for r in rows if r['split']=='eval')*
            manifest['samples'][0].get('tile_texel_m',manifest['atlas']['cell_m'])**2),
        sources=len(sources['sources']),source_query_rgb_leak_pixels=sum(r['query_rgb_leak_pixels'] for r in source_rows),
        projection_guard_observations_removed=filtered,eligible_candidate_windows=len(candidates),sample_rows=rows,source_rows=source_rows,
        evidence_boundary='Visible-road synthetic occlusion only; no evaluation of genuinely hidden road truth')
    (root/'audit_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    # 仅一次批量接触表，不通过agent逐帧查看。
    selected=[manifest['samples'][i] for i in [0,1,2,3,len(manifest['samples'])-4,len(manifest['samples'])-3,len(manifest['samples'])-2,len(manifest['samples'])-1]]
    keys=['target','input','reference','hole_mask','valid_mask']
    if 'reference_valid_mask' in selected[0]:keys+=['reference_valid_mask']
    width=size*len(keys);canvas=Image.new('RGB',(width+80,(size+24)*len(selected)),(20,24,32));draw=ImageDraw.Draw(canvas)
    for row,s in enumerate(selected):
        for col,key in enumerate(keys):
            tile=Image.open(root/s[key]).convert('RGB');canvas.paste(tile,(80+col*size,row*(size+24)))
        draw.text((0,row*(size+24)+8),s['id'].split('_')[-2]+' '+s['id'].split('_')[-1],fill='white')
        draw.text((80,row*(size+24)+size+3),'target | input | reference | hole | valid | reference valid',fill='white')
    canvas.save(root/'contact_sheet.png')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['sample_rows','source_rows']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);args=p.parse_args();audit(args.dataset)
