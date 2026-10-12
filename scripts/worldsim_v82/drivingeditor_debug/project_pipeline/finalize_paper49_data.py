#!/usr/bin/env python3
"""Seal already reviewed/validated paper-pilot inputs; never infer actor IDs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
a=p.parse_args();root=a.scene.resolve()
assert root.is_relative_to(Path('/root/autodl-tmp/drivingeditor_repro/data/waymo_paper_scenes'))
read=lambda path:json.loads((root/path).read_text())
prep=read('preparation_manifest.json');static=read('validation.json')
actor=read('dynamic_actor_preparation.json');validation=read('dynamic_actor_validation.json')
selection=read('selected_actor_0.json');sgcm=read('sgcm_input_validation.json')
deva=read('mask/mask_0/inference_manifest.json');masks=read('mask/mask_0/mask_validation.json')
assert static['status']=='PASS' and validation['status']=='passed'
assert actor['status']=='ready' and selection['status']=='visually_reviewed'
assert sgcm['status']=='ground_support_passed'
assert deva['status']=='validated' and masks['status']=='passed'
assert validation['frames']==49 and len(prep['frames'])==49
assert validation['visible_frames']+validation['empty_mask_frames']==49
assert validation['total_actor_points']>0
for name,expected in validation['sha256'].items():
    assert digest(root/name)==expected, f'Input changed after actor validation: {name}'
assert digest(root/'points3d.ply')==static['ply_sha256']
frame_ids=[row['id'] for row in prep['frames']]
for folder,suffix in [('image/image_0','.png'),('sky/sky_0','.png'),('mask/mask_0/Annotations','.png'),
        ('dynamic_image/dynamic_image_0','.png'),('dynamic_masks/dynamic_mask_0','.png'),
        ('calib','.txt'),('pose','.txt'),('velodyne','.bin')]:
    actual=sorted(path.stem for path in (root/folder).glob('*'+suffix))
    assert actual==frame_ids, f'Incomplete loader inputs: {folder}'
for name,folders in validation['frame_hashes'].items():
    for folder,expected in folders.items():
        assert digest(root/folder/(name+'.png'))==expected, f'Image changed after validation: {folder}/{name}'
assert list(root.glob('points3d_dynamic_*.ply'))==[root/'points3d_dynamic_0.ply']
raw=Path(prep['raw_path']);assert raw.stat().st_size==prep['raw_size_bytes']
files=['preparation_manifest.json','validation.json','selected_actor_0.json','dynamic_actor_preparation.json',
    'dynamic_actor_validation.json','sgcm_input_validation.json','points3d.ply','points3d_dynamic_0.ply',
    'mask/mask_0/inference_manifest.json','mask/mask_0/mask_validation.json','paper49_actor_candidate.json']
record={'status':'ready','schema_version':1,'ready_for':'Single-camera paper-method50k pilot from step0',
    'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scene':str(root),'scene_id':root.name,'frames':49,'frame_ids':frame_ids,'camera_indices':[0],
    'raw_source':{'path':str(raw),'size_bytes':raw.stat().st_size,'sha256':prep['raw_sha256'],
        'verification':'Pinned mirror full-file SHA verified during raw preparation; provenance in BASE/data_manifest.json'},
    'paper_alignment':prep['paper_alignment'],
    'model_time_range':[prep['frames'][0]['model_time'],prep['frames'][-1]['model_time']],
    'actor':{'count':1,'index':0,'instance_ids':selection['default_instance_ids'],
        'target':selection['target_description'],'ply':str(root/'points3d_dynamic_0.ply'),
        'points':validation['total_actor_points'],'points_per_frame_min':validation['points_per_frame_min'],
        'points_per_frame_max':validation['points_per_frame_max'],
        'visible_frames':validation['visible_frames'],'empty_mask_frames':validation['empty_mask_frames'],
        'visible_without_lidar_frame_ids':validation['visible_without_lidar_frame_ids'],
        'visibility_semantics':selection['visibility_semantics'],
        'image_boundary_truncation':selection['frame_boundary_truncation']},
    'sgcm':{'ground_support_status':sgcm['status'],'ground_fit':sgcm['ground_fit'],
        'coordinate_normalization':sgcm['coordinate_normalization'],
        'runtime_requirement':'Scene loader must produce its actual PCA transform; SGCM factory validates similarity and derives the model-space distance. No assumed scale.'},
    'deva_solver':deva['consensus_solver'],
    'validation':{'static_inputs':str(root/'validation.json'),'dynamic_inputs':str(root/'dynamic_actor_validation.json'),
        'instance_masks':str(root/'mask/mask_0/mask_validation.json'),'sgcm_ground':str(root/'sgcm_input_validation.json')},
    'sha256':{name:digest(root/name) for name in files},
    'paper_preprocessing_gaps':validation['paper_preprocessing_gaps'],
    'limitations':validation['limitations']+['Single-camera provisional49-frame scene window; exact paper split is unpublished.',
        'Static-only preparation reports retain their original status; this immutable readiness manifest combines later actor validation.'],
    'input_mutation_policy':'All image/mask/LiDAR/PLY/selection assets frozen for this run; further data changes require a new prepared scene.'}
out=root/'formal_training_ready.json'
assert not out.exists(), 'Refuse to overwrite existing sealed readiness manifest'
out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':'ready','marker':str(out),'sha256':digest(out),'actor_points':record['actor']['points']}))
