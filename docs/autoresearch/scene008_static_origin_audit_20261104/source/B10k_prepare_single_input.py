"""固定真实 a37 身份后准备单车输入；原五车输入和 B 权重只读。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json, shutil
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_only_B10k_20261103_r1')
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137')
PARENT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/arms/B/run30k')
TRACK='vBOasuXpjI7h7d-X_VEQdA'

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
    scene=ROOT/'inputs/scene0008_089_137';scene.mkdir(parents=True,exist_ok=False)
    prep=json.loads((OLD/'preparation_manifest.json').read_text())
    source=json.loads((OLD/'dynamic_multi_actor_preparation.json').read_text())
    actor=source['actors'][4]
    assert actor['actor_index']==4 and actor['raw_track_id']==TRACK
    for name in ['calib','image','pose','sky','velodyne','annotations','timestamp','points3d.ply']:
        (scene/name).symlink_to(OLD/name,target_is_directory=(OLD/name).is_dir())
    # loader 会写 transform_matrix；使用独立副本，禁止经符号链接写旧现场。
    shutil.copy2(OLD/'transform_matrix.txt',scene/'transform_matrix.txt')
    shutil.copy2(OLD/'points3d_dynamic_4.ply',scene/'points3d_dynamic_0.ply')
    asset=scene/'dynamic_actor_assets/actor_0';(asset/'body_masks').mkdir(parents=True)
    shutil.copy2(OLD/'dynamic_actor_assets/actor_4/source_lidar_indices.npz',asset/'source_lidar_indices.npz')
    for name in ['dynamic_image/dynamic_image_0','dynamic_masks/dynamic_mask_0','mask/mask_0/Annotations']:
        (scene/name).mkdir(parents=True)
    masks=[]
    for row in prep['frames']:
        fid=row['id'];p=OLD/'dynamic_actor_assets/actor_4/body_masks'/f'{fid}.png'
        body=np.asarray(Image.open(p))>0
        rgb=np.asarray(Image.open(scene/'image/image_0'/f'{fid}.png').convert('RGB'))
        assert body.shape==rgb.shape[:2] and body.any()
        dynamic=np.where(body[:,:,None],rgb,0).astype(np.uint8)
        for sub in ['dynamic_masks/dynamic_mask_0','mask/mask_0/Annotations','dynamic_actor_assets/actor_0/body_masks']:
            Image.fromarray(body.astype(np.uint8)*255).save(scene/sub/f'{fid}.png')
        Image.fromarray(dynamic).save(scene/'dynamic_image/dynamic_image_0'/f'{fid}.png')
        assert np.array_equal(np.asarray(Image.open(scene/'dynamic_masks/dynamic_mask_0'/f'{fid}.png'))>0,body)
        assert np.array_equal(np.asarray(Image.open(scene/'dynamic_image/dynamic_image_0'/f'{fid}.png')),dynamic)
        masks.append(dict(frame_id=fid,foreground_pixels=int(body.sum()),only_source_actor_index=4,alias=37))
    for key in ['selected_real_track_ids','selected_genuine_raw_track_ids','selected_raw_track_ids']:
        if key in prep:prep[key]=[TRACK]
    prep['selected_real_track_ids']=[TRACK]
    prep['identity_source']=dict(original_scene=str(OLD),source_actor_index=4,new_actor_index=0,alias=37,raw_track_id=TRACK)
    write(scene/'preparation_manifest.json',prep)
    actor=dict(actor,actor_index=0,source_actor_index=4,alias=37)
    source.update(status='PASS_FIXED_SINGLE_A37_INPUT',scene=str(scene),actors=[actor],old_assets_modified=False,
        actor_count=1,selection='only canonical existing actor_4; no posthoc checkpoint selection',dynamic_RGB_and_union_exact=True)
    write(scene/'dynamic_multi_actor_preparation.json',source);write(scene/'dynamic_multi_actor_validation.json',source)
    config=json.loads((PARENT/'effective_config.json').read_text())
    original=json.loads(json.dumps(config))
    config['source_path']=str(scene);config['model_path']=str(ROOT/'formal10k')
    config['followup_scene_spec']['source_path']=str(scene)
    config['followup_scene_spec']['actor_ply_paths']=[str(scene/'points3d_dynamic_0.ply')]
    assert config['iterations']==30000 and config['start_checkpoint'] is None
    write(ROOT/'config.json',config)
    changes={k:dict(parent=original[k],single_actor=config[k]) for k in original if original[k]!=config[k]}
    write(ROOT/'identity_contract.json',dict(task_id='SCENE008-A37-ONLY-B10K-20261103-R1',dynamic_actor_aliases=[37],
        dynamic_raw_track_ids=[TRACK],source_actor_index=4,new_actor_index=0,retained_aliases=[5,26,18,6],
        retained_representation='original B temporal static/background model; initialized and trained from scratch',
        masks=masks,dynamic_ply_files=[str(scene/'points3d_dynamic_0.ply')],
        source_ply_exact_bytes=(OLD/'points3d_dynamic_4.ply').read_bytes()==(scene/'points3d_dynamic_0.ply').read_bytes(),
        source_real_point_count=actor['total_real_points'],canonical_identity_fixed_before_training=True,
        warm_start=False,original_B_read_only=True,formal_updates=10000,learning_rate_horizon=30000,
        config_changes=changes,smoke_accounting='separate from formal10k; same30k schedule; additional checkpoint/evaluation nodes',
        inputs_role='49observedtrain BUILD; no heldout',no_subagents=True,shutdown_allowed=False,human_verdict=None,quality_accepted=False))
    print(json.dumps(dict(status='PASS_FIXED_SINGLE_A37_INPUT',frames=len(masks),dynamic_aliases=[37],points=actor['total_real_points'],config_changes=list(changes))))

if __name__=='__main__':main()
