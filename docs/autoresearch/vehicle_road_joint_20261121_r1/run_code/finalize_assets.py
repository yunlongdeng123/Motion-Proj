"""CPU收尾：全轨迹绑定、紧凑路色patch与审阅/恢复小包。"""
from pathlib import Path
import argparse,json,shutil,tarfile,time,filecmp,os
import numpy as np
from PIL import Image
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1')
ROAD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/appearance_fit/asset')
BANK=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
TRACK='bzFbWRrSGOgEaXxnuGV0tw'
STAGES=['native_before','canonical_init','canonical_prior_init','vehicle_fit','joint_fit','handoff_fixed','handoff_fixed_guarded'];FRAMES=[86,110,134,179,198]
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def metadata():
    scene=read(ROAD/'persistent/scene.json');T=np.loadtxt(scene['world_to_model']);full=[];world=[];model=[];valid=[]
    for frame in scene['frames']:
        obs=read(frame['observation']);box=next((x for x in obs['boxes'] if x['track']==TRACK),None);row=dict(raw=frame['key'],model_time=frame['model_time'],source_time_s=frame['source_time_s'],observation=frame['observation'],available=box is not None)
        P=np.asarray(box['world_from_actor']).reshape(4,4) if box else np.eye(4)
        if box:row.update(world_from_actor=P.tolist(),model_from_actor=(T@P).tolist(),dimensions_lwh_m=box['dimensions'])
        full.append(row);world.append(P);model.append(T@P);valid.append(box is not None)
    ids=load('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/train_inputs/allowed_color_world_rows.npz')['world_row_ids']
    for asset in ['vehicle_fit','joint_fit','handoff_fixed','handoff_fixed_guarded']:
        p=ROOT/'assets'/asset
        if not p.exists():continue
        write(p/'poses_full.json',dict(track=TRACK,actor_id=21,alias='a20',frozen=True,source='original observed boxes, no fitting or interpolation',frames=full))
        np.savez_compressed(p/'poses_full.npz',raw=np.array([x['raw'] for x in full]),available=np.array(valid),world_from_actor=np.stack(world),model_from_actor=np.stack(model),model_from_world=T)
        write(p/'actor_track_map.json',dict(actors=scene['actors'],target=dict(alias='a20',actor_id=21,track_id=TRACK),delete=dict(alias='a9',actor_id=10,track_id='NTjq-OhZ_97OMJ4xwJAxXg')))
        road=load(p/'road.npz');np.savez_compressed(p/'road_rgb_patch.npz',world_row_ids=ids,rgb_srgb=road['reference_srgb'][ids],base_asset=np.asarray(str(ROAD)),base_road_sha256=np.asarray('70cc545db6ce595b24c0ec2ba836d7058edbb73734d6873c880f3e634fb08d9f'))
        binding=read(p/'scene_binding.json');binding.update(full_poses=str(p/'poses_full.npz'),full_poses_metadata=str(p/'poses_full.json'),actor_track_map=str(p/'actor_track_map.json'),compact_road_color_patch=str(p/'road_rgb_patch.npz'));write(p/'scene_binding.json',binding)
    c=read(ROOT/'contract.json');c['mask_content']='accepted repaired body segmentation proxy; body-only without shadow; not manual ground truth for hidden a20';c['quality_acceptance']='not passed: residual a9 content, coarse lower shape prior and diagnostic/other-actor degradation remain';write(ROOT/'contract_final.json',c)
    ledger=read(ROOT/'resource_ledger.json');ledger.update(handoff_strict_native_forward_calls=read(ROOT/'handoff_proof.json')['native_raster_calls'],handoff_unit_feature_backward_calls=15,handoff_guarded_native_forward_calls=read(ROOT/'handoff_guarded_proof.json')['native_forward_calls'],handoff_optimizer_updates=0,power_operations=0,GPU_keep_on=True);write(ROOT/'resource_ledger.json',ledger)
    write(ROOT/'asset_index.json',dict(formal_steps=dict(vehicle=200,joint=200),original_formal_asset=str(ROOT/'assets/joint_fit'),guarded_editable_asset=str(ROOT/'assets/handoff_fixed_guarded'),public_render_entry=str(ROOT/'code/fix_a9_handoff.py'),render_command='/root/autodl-tmp/envs/drivingeditor/bin/python '+str(ROOT/'code/fix_a9_handoff.py')+' --reload-asset '+str(ROOT/'assets/handoff_fixed_guarded'),full_optimizer_checkpoint=str(ROOT/'checkpoints/joint_step00200.pt'),road_base=str(ROAD),quality_pass=False,full_pose_count=len(full),pose_available_count=int(sum(valid)),frames_rendered=FRAMES))
def pack():
    out=ROOT/'takeback';out.mkdir(exist_ok=True);media=[]
    for stage in STAGES:
        for raw in FRAMES:
            for view in ['joint_keep','joint_delete','vehicle_only','road_only','vehicle_alpha_keep','vehicle_alpha_delete','vehicle_alpha_isolated','road_alpha']:
                src=ROOT/'delivery'/stage/f'{raw:03d}'/(view+'.png');dst=out/'delivery'/stage/f'{raw:03d}'/(view+'.webp');dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():Image.open(src).save(dst,format='WEBP',lossless=True,method=1)
                media.append(dict(stage=stage,raw=raw,view=view,path=str(dst.relative_to(out))))
    for raw in FRAMES:
        p=ROOT/'delivery/references'/f'{raw:03d}'
        for src in p.glob('*.png'):
            if src.name=='road_teacher.png' and raw not in (86,110):continue
            dst=out/'delivery/references'/f'{raw:03d}'/(src.stem+'.webp');dst.parent.mkdir(parents=True,exist_ok=True);Image.open(src).save(dst,format='WEBP',lossless=True,method=1)
    for src in ROOT.glob('*.json'):shutil.copy2(src,out/src.name)
    for src in (ROOT/'delivery').glob('*.json'):shutil.copy2(src,out/'delivery'/src.name)
    for src in (ROOT/'smoke').glob('*.json'):
        p=out/'smoke'/src.name;p.parent.mkdir(exist_ok=True);shutil.copy2(src,p)
    for src in (ROOT/'code').glob('*.py'):
        p=out/'code'/src.name;p.parent.mkdir(exist_ok=True);shutil.copy2(src,p)
    for src in (ROOT/'code/frozen_formal_r2').glob('*.py'):
        p=out/'code/frozen_formal_r2'/src.name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
    for p in ['initial_vehicle_prior.npz','visible_support.npz','handoff_candidate_evidence.npz']:
        shutil.copy2(ROOT/p,out/p)
    for name in ['vehicle_fit','joint_fit','handoff_fixed','handoff_fixed_guarded']:
        p=ROOT/'assets'/name;dst=out/'assets'/name;dst.mkdir(parents=True,exist_ok=True)
        fields=['poses.npz','poses_full.npz','poses_full.json','actor_track_map.json','road_rgb_patch.npz','scene_binding.json']
        if name in ['joint_fit','vehicle_fit']:fields+=['vehicle.npz']
        if name.startswith('handoff_'):fields+=['a9_ownership_sidecar.npz']
        for field in fields:
            if (p/field).exists():shutil.copy2(p/field,dst/field)
    checkpoint=out/'checkpoints/joint_step00200.pt';checkpoint.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'checkpoints/joint_step00200.pt',checkpoint)
    write(out/'media_manifest.json',dict(lossless_webp=True,original_float_render_retained_remote=True,rows=media,core_stages=STAGES))
    text='''# Scene065 车辆与道路联合首版恢复入口

正式预算为车辆200步、联合200步；原道路和原native模型均保留。最终视觉质量未通过，不能晋升默认场景。诊断帧和其它车辆局部仍有退化，a9残影与粗下半形状先验仍可见。

```mermaid
flowchart LR
 B[R2道路与原native模型] --> L[持久加载器]
 C[canonical车辆与冻结轨迹] --> L
 P[1191行道路颜色patch] --> L
 O[75行a9归属sidecar] --> L
 L --> R[统一高斯排序渲染]
 R --> K[KEEP与DELETE]
```

小包含无损WebP审阅图、细节指标、完整Adam/RNG checkpoint、canonical车辆、全199帧轨迹登记、1191行道路RGB patch和归属sidecar。大体积基础道路及原native checkpoint仍在远端，asset_index.json给出原始路径与可执行渲染命令；所有原始float渲染留在远端delivery目录。

远端完整资产可直接执行asset_index.json中的render_command，从磁盘加载并复核五帧。小包的handoff资产复用assets/joint_fit/vehicle.npz；道路使用基线R2 road.npz加road_rgb_patch.npz，不包含重复的完整道路文件。完整训练状态是checkpoints/joint_step00200.pt；smoke和post_numeric_probe.json保留恢复与数值连续性证据。

恢复检查证明保存状态逐值精确；后续FP32 CUDA原子累加步为数值近似。下半面板、保险杠和轮胎是受尺寸/可见车宽/遮挡约束的工程形状先验，隐藏区域没有真实RGB真值，repaired masks是分割代理而非人工标注隐藏真值。
'''
    (out/'RESTORE.md').write_text(text)
    # Exact duplicate files use ordinary hard links; all named paths remain present.
    groups={};duplicates=0
    for p in sorted(x for x in out.rglob('*') if x.is_file()):
        size=p.stat().st_size;previous=next((q for q in groups.get(size,[]) if filecmp.cmp(p,q,shallow=False)),None)
        if previous is not None and p.stat().st_ino!=previous.stat().st_ino:p.unlink();os.link(previous,p);duplicates+=1
        else:groups.setdefault(size,[]).append(p)
    target=ROOT/'vehicle_road_joint_review_and_asset_r2.tar.gz'
    with tarfile.open(target,'w:gz',compresslevel=3) as tar:tar.add(out,arcname='vehicle_road_joint_20261121_r1')
    write(ROOT/'takeback_manifest.json',dict(path=str(target),bytes=target.stat().st_size,files=sum(1 for x in out.rglob('*') if x.is_file()),exact_duplicate_payloads_deduplicated=duplicates,original_float_NPZ_included=False,repeated_full_road_included=False,stages=STAGES,complete=True))
    print(target,target.stat().st_size,flush=True)
ap=argparse.ArgumentParser();ap.add_argument('--pack',action='store_true');a=ap.parse_args();metadata()
if a.pack:pack()
