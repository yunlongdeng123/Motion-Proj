"""核对数据来源与真实输入合同；数量不足不靠mask变体冒充新片段。"""
from pathlib import Path
import sys,itertools,numpy as np
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
from delete_full_query import Writer
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r4';manifest=read(R/'manifest.json');identity=read(T/'r1/identity_metadata.json');inv=read(T/'r1/inventory.json')
extra=[]
for s in ['0230','0255']:
    p=Path(f'/root/autodl-tmp/data/dynamic_editing_v2/manifests/scene-{s}_raw_manifest.json');d=read(p);extra.append(dict(scene=f'scene_{s}',raw_log_prefixes=sorted({Path(x['filename']).name.split('__')[0] for x in d['files'] if '__CAM_' in x['filename']})))
pool=[r for r in identity['rows'] if r['split']=='training_pool_candidate'];overlap=[dict(a=a['scene_name'],b=b['scene'],prefixes=sorted(set(a['raw_log_prefixes'])&set(b['raw_log_prefixes']))) for a,b in itertools.product(pool,extra) if set(a['raw_log_prefixes'])&set(b['raw_log_prefixes'])]
eval_roots=[s['root'] for s in read(T/'r1/registration.json')['new_scenes']]+['/root/autodl-tmp/data/v76_vadgs/scene_0230','/root/autodl-tmp/data/v76_vadgs/scene_0255','/root/autodl-tmp/data/v76_vadgs/official_000/000']
eval_tracks={a['id'] for root in eval_roots for a in read(Path(root)/'instances/instances_info.json').values()};track_overlap=[]
for c in manifest['candidates']:
    if any(x['scene']==c['scene'] for x in track_overlap):continue
    tracks={a['id'] for a in read(Path(c['source_root'])/'instances/instances_info.json').values()};track_overlap.append(dict(scene=c['scene'],shared=len(tracks&eval_tracks)))
checks=[]
for c in manifest['candidates']:
    for j in [0,29]:
        f=c['source_frames'][j];p=Path(c['source_root'])/'images'/f'{f:03}_{c["camera"]}.jpg';original=np.array(Image.open(p).convert('RGB').resize((1024,576),Image.Resampling.BILINEAR));mask=np.array(Image.open(Path(c['mask_dir'])/f'{j:05}.png'))>0;inp=original.copy();inp[mask]=0
        assert mask.shape==original.shape[:2] and np.array_equal(inp[~mask],original[~mask]) and (inp[mask]==0).all();checks.append(dict(candidate=c['id'],frame=f,mask_pixels=int(mask.sum()),outside_changed=0))
out=R/'review';out.mkdir(exist_ok=False);video_rows=[]
for sid in ['276','296','827']:
    c=next(c for c in manifest['candidates'] if c['scene']==sid);writers={k:Writer(out/f'{sid}_{k}.mp4',(1024,576)) for k in ['input','target']}
    for j,f in enumerate(c['source_frames']):
        rgb=np.array(Image.open(Path(c['source_root'])/'images'/f'{f:03}_{c["camera"]}.jpg').convert('RGB').resize((1024,576),Image.Resampling.BILINEAR));mask=np.array(Image.open(Path(c['mask_dir'])/f'{j:05}.png'))>0;inp=rgb.copy();inp[mask]=0;writers['input'].write(inp);writers['target'].write(rgb)
    for w in writers.values():w.close()
    video_rows.append(dict(candidate=c['id'],scene=sid,split=c['split'],frames=30))
report=dict(actual_input_checks=len(checks),known_old_0230_0255_log_overlap=overlap,old_log_metadata=extra,all_nine_eval_actor_track_overlap=track_overlap,official000_raw_log='unavailable; actor tracks disjoint is weaker than log isolation; must resolve before training',unknown_pretrained_overlap=True,admitted_training=0,training_steps=0,visual_notes='18 start/end samples inspected. Includes tree/sidewalk/building and dark night background; candidate clean RGB not yet complete per-clip review. Daytime train vs nighttime validation is a domain-shift stress split, not balanced training validation.',examples=video_rows,checks=checks,human_verdict=None)
dump(R/'data_audit.json',report);print('DATA_AUDIT',len(checks),overlap,track_overlap)
