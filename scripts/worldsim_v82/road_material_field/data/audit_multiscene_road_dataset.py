"""只读审计最终原始观测道路数据的世界块隔离、源路径和几何映射。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    a=p.parse_args()
    manifest=read(a.dataset)
    assert manifest['schema']=='road_material_dataset.v1'
    counts={}
    groups={}
    rows=[]
    brightness=[]
    geometry_errors=[]
    path_keys=['input','reference','target','hole_mask','valid_mask','world_xyz']
    required={'scene065':(12,4),'scene067':(12,4),'scene079':(12,4),'scene081':(0,4)}
    for sample in manifest['samples']:
        for key in path_keys:
            assert Path(sample[key]).is_file(),(sample['id'],key)
        image={key:np.asarray(Image.open(sample[key])) for key in path_keys if key!='world_xyz'}
        assert all(v.shape[:2]==(128,128) for v in image.values())
        hole=image['hole_mask']>127
        valid=image['valid_mask']>127
        assert (hole&valid).sum()==sample['hole_pixels']
        assert np.array_equal(image['input'][~hole],image['target'][~hole])
        assert np.all(image['input'][hole]==128)
        assert valid[hole].all()
        scene=sample['scene_id']
        c=counts.setdefault(scene,{'train':0,'eval':0,'marking_train':0,'marking_eval':0})
        c[sample['split']]+=1
        c['marking_'+sample['split']]+=int(sample.get('marking_stratum',False))
        assert sample['evaluation_role']==('old_scene_spatial' if scene=='scene065' else 'unseen_scene' if scene=='scene081' else 'new_train_scene_spatial')
        loaded=np.load(sample['world_xyz'])
        xyz=loaded['world_xyz'] if isinstance(loaded,np.lib.npyio.NpzFile) else loaded
        assert xyz.shape==(128,128,3) and np.isfinite(xyz).all()
        if scene!='scene065':
            assert sample['min_axis_span_native_px']>=64
            assert valid.mean()>=.98
            matrix=np.asarray(sample['world_from_uv'])
            yy,xx=np.mgrid[:128,:128]
            unit=np.stack((xx+.5,yy+.5,np.zeros_like(xx)),2)
            expected=unit@matrix[:3,:3].T+matrix[:3,3]
            geometry_error=float(abs(expected-xyz).max())
            geometry_errors.append(geometry_error)
            assert geometry_error<1e-8,(sample['id'],geometry_error)
            assert abs(np.linalg.norm(matrix[:3,0])-.02)<.001
            assert abs(np.linalg.norm(matrix[:3,1])-.02)<.001
            key=tuple(sample['world_block_key'])
            groups.setdefault(scene,[]).append(key)
            assert key!=tuple(sample['reference_world_block_key'])
        rgb=image['target']/255.
        luma=rgb@np.array([.2126,.7152,.0722])
        bright=(luma>np.quantile(luma[valid],.30)+.075)&(rgb.min(2)>.28)&(np.ptp(rgb,axis=2)<.20)&valid
        yellow=(rgb[:,:,0]>.28)&(rgb[:,:,1]>.22)&(rgb[:,:,0]-rgb[:,:,2]>.095)&valid
        # 更严格的纯沥青子集仍是自动代理，不当作人工真值。
        fraction=float((bright|yellow).mean())
        known=~hole&valid
        weak_threshold=float(np.clip(np.median(luma[known])+.12,.38,.85))
        b=dict(id=sample['id'],scene_id=scene,split=sample['split'],marking_stratum=sample.get('marking_stratum',False),
            visible_road_luma_median=float(np.median(luma[known])),root_weak_threshold=weak_threshold,
            candidate_marking_luma_quantiles=(np.quantile(luma[(bright|yellow)&known],[.1,.5,.9]).tolist() if ((bright|yellow)&known).any() else None),
            candidate_visible_marking_pixels=int(((bright|yellow)&known).sum()),
            root_threshold_visible_candidate_pixels=int((known&(luma>=weak_threshold)).sum()),
            label_policy='automatic exposure/contrast proxy; not manual truth')
        brightness.append(b)
        rows.append(dict(id=sample['id'],scene_id=scene,split=sample['split'],
            hole_pixels=int(hole.sum()),valid_fraction=float(valid.mean()),
            min_axis_span_native_px=sample.get('min_axis_span_native_px'),
            automatic_marking_fraction=fraction,pure_asphalt_proxy=fraction<.003,
            evaluation_role=sample['evaluation_role']))
    for scene,expected in required.items():
        assert (counts[scene]['train'],counts[scene]['eval'])==expected,(scene,counts[scene])
    for scene,keys in groups.items():
        assert len(keys)==len(set(keys)),scene
    assert not any(s['scene_id']=='scene081' and s['split']=='train' for s in manifest['samples'])
    legacy=read(manifest['component_manifests'][0])
    current={s['id']:s for s in manifest['samples'] if s['scene_id']=='scene065'}
    normalized_path_keys=['input','reference','target','hole_mask','valid_mask','world_xyz','point_ids','reference_valid_mask','source_uv_native','source_weight','tile_npz']
    for row in legacy['samples']:
        merged=current[row['id']]
        for key,value in row.items():
            if key in normalized_path_keys and value:
                assert Path(merged[key]).resolve()==(Path(manifest['component_manifests'][0]).parent/value).resolve()
            else:
                assert merged[key]==value,(row['id'],key)
    # 新候选在评价时可以可见，但不得属于任何查询/目标世界块。
    components=[]
    for component in manifest['component_manifests'][1:]:
        cm=read(component)
        queries={tuple(s['world_block_key']) for s in cm['samples']}
        refs={tuple(c['key']) for c in cm['candidates']}
        assert not queries&refs
        annotation=read(Path(cm['source_observation_bank'])/'annotation_receipt.json')
        assert len(annotation['frames'])==17
        assert all(x['camera_boxes']+x['projected_lidar_boxes']>0 for x in annotation['frames'])
        components.append(dict(scene_id=cm['scene_id'],sampled_source_frames=len(cm['source_frames']),
            road_pool=cm['quality_gate']['pool_size'],reference_candidates=len(cm['candidates']),
            source_boxes=sum(x['camera_boxes']+x['projected_lidar_boxes'] for x in annotation['frames'])))
    selected=[s for s in manifest['samples'] if s['scene_id']!='scene065' and s['split']=='train' and not s.get('marking_stratum',False)]
    selected=selected[:2]+[s for s in selected if s['scene_id']=='scene079'][:2]
    canvas=Image.new('RGB',(86+128*3,150*len(selected)),(18,22,30))
    draw=ImageDraw.Draw(canvas)
    for i,s in enumerate(selected):
        draw.text((2,150*i+4),s['scene_id']+'\n'+s['id'][-3:],fill='white')
        for col,key in enumerate(['target','input','reference']):
            canvas.paste(Image.open(s[key]).convert('RGB'),(86+128*col,150*i))
    canvas.save(a.dataset.parent/'contact_plain_qa.png')
    receipt=dict(status='PASS_CPU_MULTISCENE_DATA_AUDIT',dataset=str(a.dataset),scene_counts=counts,
        total_train=sum(c['train'] for c in counts.values()),total_eval=sum(c['eval'] for c in counts.values()),
        valid_scoring_pixels=sum(r['hole_pixels'] for r in rows if r['split']=='eval'),
        old_scene_files_and_all_original_sample_fields_preserved=True,
        new_query_reference_world_block_overlap=0,new_train_eval_world_block_overlap=0,
        unseen_scene_training_blocks=0,geometry_training_updates=0,GPU_queries=0,
        new_geometry_mapping_max_error_m=max(geometry_errors),components=components,
        automatic_marking_labels_are_not_manual_truth=True,
        pure_asphalt_proxy_train_counts={scene:sum(r['scene_id']==scene and r['split']=='train' and r['pure_asphalt_proxy'] for r in rows) for scene in required},
        samples=rows,limits=['Measured local planes and predicted semantics are fixed preparation proxies.',
            'Pseudo-occlusion of visible road is not ground truth for physically hidden road.',
            'Same-scene world blocks can share capture frames; cross-scene eval is separately held out.',
            'Reference selection uses only world center distance; target RGB does not select references.'])
    (a.dataset.parent/'audit_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (a.dataset.parent/'marking_brightness.json').write_text(json.dumps(brightness,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['samples']}),flush=True)


if __name__=='__main__':
    main()
