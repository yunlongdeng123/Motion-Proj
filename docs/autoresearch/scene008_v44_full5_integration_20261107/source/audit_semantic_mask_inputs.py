"""Read-only AutoDL CPU audit of existing five-instance training mask inputs.

Pipe this file into drivingeditor/bin/python - on the recorded host. JSON is
printed to stdout. No files, checkpoints, renderer, or training state are written.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True

import cv2
import numpy as np
from PIL import Image
import torch


CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
PREPARED=CAM/'scene008_full_dynamic_edit_20261107_r1/inputs/scene0008_089_137'
ORIGINAL=CAM/'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137'
OBS=CAM/'scene008_instances_20261105_r1/observations'
ENGINE=CAM/'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
ORDER=[5,26,18,6,37]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    path=ENGINE/'utils/paper_masks.py'
    spec=importlib.util.spec_from_file_location('real_paper_mask_loader_cpu_audit',path)
    masks=importlib.util.module_from_spec(spec);spec.loader.exec_module(masks)
    preparation=json.loads((PREPARED/'dynamic_multi_actor_preparation.json').read_text())
    manifest=json.loads((PREPARED/'preparation_manifest.json').read_text())
    frames=[];failures=[];totals={str(actor):0 for actor in ORDER}
    for raw in range(89,138):
        fid=f'0008{raw:03d}'
        with np.load(OBS/'frames'/f'{raw:03d}.npz',allow_pickle=False) as data:
            labels=data['actor_labels'].copy();rgb=data['rgb'].copy()
        label_path=PREPARED/'matched_training_labels'/(fid+'.png')
        full_labels=np.asarray(Image.open(label_path))
        rgb_path=PREPARED/'image/image_0'/(fid+'.png')
        original_rgb=np.asarray(Image.open(rgb_path).convert('RGB'))
        expected_rgb=cv2.resize(original_rgb/255.,(480,320)).astype(np.float32)
        actor_rows=[];union=np.zeros((320,480),dtype=bool)
        for index,actor in enumerate(ORDER):
            p=PREPARED/'dynamic_actor_assets'/f'actor_{index}'/'body_masks'/(fid+'.png')
            original_p=ORIGINAL/'dynamic_actor_assets'/f'actor_{index}'/'body_masks'/(fid+'.png')
            image=Image.open(p).convert('L');values=np.asarray(image)
            small=np.asarray(image.resize((480,320),Image.Resampling.NEAREST))>0
            # matched_training_labels is already480x320, while original body
            # masks share the1920x1280 RGB grid. Compare after explicit NN lift.
            matched=np.asarray(Image.fromarray((full_labels==index+1).astype(np.uint8)).resize(image.size,Image.Resampling.NEAREST))>0
            checks=dict(binary_values=set(np.unique(values).tolist()).issubset({0,1,255}),
                        original_RGB_dimensions_equal=image.size==(original_rgb.shape[1],original_rgb.shape[0]),
                        matched_training_label_fullres_exact=values.shape==matched.shape and np.array_equal(values>0,matched),
                        observation_actor_label_exact=np.array_equal(small,labels==index+1),
                        original_prepared_source_bytes_equal=sha(p)==sha(original_p))
            if not all(checks.values()):failures.append(dict(raw=raw,actor=actor,checks=checks))
            total=int(small.sum());totals[str(actor)]+=total;union|=small
            actor_rows.append(dict(actor_id=actor,actor_index=index,label=index+1,
                                   body_mask_path=str(p),sha256=sha(p),size=list(image.size),
                                   observed480x320_pixels=total,checks=checks))
        camera=SimpleNamespace(image_path=str(rgb_path),image_width=480,image_height=320,
                               world_view_transform=torch.zeros((4,4),device='cpu'))
        loaded=masks.load_dynamic_mask(str(PREPARED),camera).numpy().squeeze()>0
        shadow=PREPARED/'dynamic_masks/shadow_mask_0'/(fid+'.png')
        expected=union.copy()
        if shadow.exists():expected|=np.asarray(Image.open(shadow).convert('L').resize((480,320),Image.Resampling.NEAREST))>0
        dynamic=PREPARED/'dynamic_masks/dynamic_mask_0'/(fid+'.png')
        stored=np.asarray(Image.open(dynamic).convert('L').resize((480,320),Image.Resampling.NEAREST))>0
        checks=dict(observation_RGB_recomputed_from_same_original_RGB_exact=np.array_equal(rgb,expected_rgb),
                    five_actor_union_equals_saved_training_dynamic_mask=np.array_equal(union,stored),
                    actual_CPU_training_mask_loader_equals_expected=np.array_equal(loaded,expected),
                    all_labels_in_expected_set=set(np.unique(labels).tolist()).issubset(set(range(6))))
        if not all(checks.values()):failures.append(dict(raw=raw,checks=checks))
        frames.append(dict(raw=raw,frame_id=fid,original_RGB_path=str(rgb_path),original_RGB_sha256=sha(rgb_path),
                           matched_training_labels_path=str(label_path),matched_training_labels_sha256=sha(label_path),
                           actual_training_union_path=str(dynamic),actual_training_union_sha256=sha(dynamic),
                           observation_path=str(OBS/'frames'/f'{raw:03d}.npz'),shadow_present=shadow.exists(),
                           actors=actor_rows,checks=checks))
    assert not torch.cuda.is_initialized()
    mapping=[dict(label=i+1,actor_id=actor,actor_index=i,raw_track_id=preparation['actors'][i]['raw_track_id'],
                  description=preparation['actors'][i]['description'],observed_pixel_frames=totals[str(actor)],
                  body_masks=str(PREPARED/'dynamic_actor_assets'/f'actor_{i}'/'body_masks')) for i,actor in enumerate(ORDER)]
    result=dict(status='PASS_EXISTING_FIVE_INSTANCE_MASK_BUILD_INPUT_AND_RGB_ALIGNMENT' if not failures else 'FAIL_EXISTING_INPUT_ALIGNMENT',
                frame_count=49,per_instance_masks_checked=245,label_to_actor=mapping,frames=frames,failures=failures,
                fixed_a5_pixels=totals['5'],fixed_a5_domain_matches121128=totals['5']==121128,
                original_prepared_source=str(ORIGINAL),Full5_training_input=str(PREPARED),observation_bank=str(OBS),
                actor_label_origin=str(CAM/'r15_gn11_stage1_v41_20261023_dense_r7/frames'),
                original_mask_preparation_GT_box_flag=manifest.get('GT_boxes_used_to_create_masks'),
                input_role='Existing BUILD/in-sample instance silhouettes used for geometry preparation and constructing the dynamic union; not new evaluation-only annotations.',
                original_training_consumption='train.py331 loads the union via paper_masks.py24; train.py384-406 applies union/complement RGB and union dynamic alpha. No per-actor current-identity RGB/alpha term in that original hybrid_state.',
                semantic_limit='Source ancestry is not current semantic ownership. Observed silhouettes do not label hidden surfaces and are not independent held-out GT.',
                RGB_alignment_scope='Per-instance masks and RGB share original image dimensions; nearest mask downsample exactly equals saved480x320 labels. Saved observed RGB exactly equals the documented cv2 resize of the same-frame original image. This does not claim every training-loader RGB resampling kernel is bit-identical.',
                source_hashes={str(p):sha(p) for p in [ENGINE/'utils/paper_masks.py',ENGINE/'train.py',OBS.parent/'source/object_observations.py',PREPARED/'dynamic_multi_actor_preparation.json',PREPARED/'preparation_manifest.json']},
                GPU_used=False,CUDA_initialized=False,models_loaded=False,optimizer_updates=0,files_written_remotely=False)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
