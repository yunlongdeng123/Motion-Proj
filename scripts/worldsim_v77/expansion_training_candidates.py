"""真实RGB监督候选：独立日志按scene切分，GT-free不是人工干净背景认证。"""
from pathlib import Path
import sys,datetime,itertools,collections
import numpy as np,cv2
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera,hull_mask
from video_review import scene_frame
from geometry import transform,box_mask
TASK=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');ROOT=TASK/'r4'

def main():
    assert not (ROOT/'masks').exists();ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2);inv=read(TASK/'r1/inventory.json');identity=read(TASK/'r1/identity_metadata.json')
    eval_logs={p for r in identity['rows'] if r['split']!='training_pool_candidate' for p in r['raw_log_prefixes']}
    registration=dict(task_id=TASK.name,run_id='r4',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),operation='prepare_only',seed=77,
        sources={'276':'train_candidate','296':'train_candidate','827':'validation_candidate'},window_starts=[0,30,60,90,120,150],frames_per_clip=30,fps=10,
        supervision='Original real RGB decoded and resized bilinear to1024x576; mask filled zero input; target identical original RGB. No generated image as GT.',
        masks='Vehicle4.5x1.8x1.5m stationary world envelopes; camera-ray intersection with coarse local LiDAR road plane; project across30frames. Reject any GT hull overlap, near-plane, small/too-large masks.',
        acceptance='Candidate only. Every clip requires visible-background review for unannotated vehicles/shadows/road suitability. No training and no claim200-500 independent source clips.',
        failure_ledger_refs=['V77-F02'],failure_ledger_delta='none',human_verdict=None)
    dump(ROOT/'registration.json',registration);accepted=[];counts={};thumbs=[]
    for sid,split in registration['sources'].items():
        metadata=next(r for r in identity['rows'] if r['processed_id']==sid);assert not(set(metadata['raw_log_prefixes'])&eval_logs)
        row=next(r for r in inv['rows'] if Path(r['root']).name==sid and r.get('complete_first30_sixcams'))
        data=Path(row['root']);views=[]
        for c in range(6):
            k=np.loadtxt(data/'intrinsics'/f'{c}.txt')
            views.append(dict(camera=c,intrinsics=[[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]],original_wh=list(Image.open(data/'images'/f'000_{c}.jpg').size)))
        spec=dict(name=sid,root=str(data),views=views)
        inst=read(data/'instances/instances_info.json');scene_count=0;reasons=collections.Counter()
        for start in registration['window_starts']:
            frames=[scene_frame(spec,f,inst) for f in range(start,start+30)]
            ego=np.mean([np.array(v['c2w'])[:3,3] for v in frames[0]['views']],axis=0)
            pts=transform(np.fromfile(data/'lidar'/f'{start:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(data/'lidar_pose'/f'{start:03}.txt'))
            keep=np.linalg.norm(pts[:,:2]-ego[:2],axis=1)<18
            for b in frames[0]['all_boxes']:keep &= ~box_mask(pts,np.array(b['pose']),b['size_lwh'])
            pp=pts[keep];z0=np.quantile(pp[:,2],.15);ground=pp[np.abs(pp[:,2]-z0)<.25]
            if len(ground)<30:reasons['ground_insufficient']+=1;continue
            coef=np.linalg.lstsq(np.column_stack([ground[:,:2]-ego[:2],np.ones(len(ground))]),ground[:,2],rcond=None)[0]
            n=np.array([-coef[0],-coef[1],1]);d=coef[0]*ego[0]+coef[1]*ego[1]-coef[2]
            for cid in range(6):
                geometry=[camera(fr,cid) for fr in frames];forbidden=[]
                for fr,(cc,kk) in zip(frames,geometry):
                    busy=np.zeros((576,1024),bool)
                    for b in fr['all_boxes']:busy |= hull_mask(b,cc,kk,pad=4)
                    forbidden.append(busy)
                cc,kk=geometry[0]
                for u,v in itertools.product([256,512,768],[390,440,490]):
                    ray=cc[:3,:3]@np.linalg.solve(kk,[u,v,1]);den=n@ray
                    if abs(den)<1e-5:continue
                    t=-(n@cc[:3,3]+d)/den
                    if not 5<t<28:reasons['distance']+=1;continue
                    gp=cc[:3,3]+ray*t;forward=cc[:3,2].copy();forward[2]=0;forward/=np.linalg.norm(forward)
                    pose=np.eye(4);pose[:3,:3]=np.column_stack([forward,[-forward[1],forward[0],0],[0,0,1]]);pose[:3,3]=gp+[0,0,.75]
                    b=dict(pose=pose.tolist(),size_lwh=[4.5,1.8,1.5]);masks=[hull_mask(b,c,k) for c,k in geometry];areas=[int(m.sum()) for m in masks]
                    if min(areas)<384 or max(areas)>1024*576*.25:reasons['coverage']+=1;continue
                    if any(np.any(m&bad) for m,bad in zip(masks,forbidden)):reasons['gt_overlap']+=1;continue
                    if scene_count>=100:continue
                    uid=f's{sid}_f{start:03}_c{cid}_u{u}_v{v}';dest=ROOT/'masks'/uid;dest.mkdir(parents=True)
                    for j,m in enumerate(masks):Image.fromarray(m.astype('uint8')*255).save(dest/f'{j:05}.png')
                    sample=dict(id=uid,scene=sid,split=split,raw_log_prefixes=metadata['raw_log_prefixes'],source_root=str(data),camera=cid,source_frames=list(range(start,start+30)),mask_dir=str(dest),pose_world=pose.tolist(),size_lwh=b['size_lwh'],area_range=[min(areas),max(areas)],gt_overlap_pixels=0,source_window_id=f'{sid}:{start}:{cid}',source_time_window_id=f'{sid}:{start}',human_review=None,training_admitted=False)
                    accepted.append(sample);scene_count+=1
                    if len(thumbs)<18 and sum(t[0]==sid for t in thumbs)<6:
                        panel=Image.new('RGB',(1024,314),(20,30,40));dd=ImageDraw.Draw(panel)
                        for j,f in enumerate([0,29]):
                            im=np.array(Image.open(data/'images'/f'{start+f:03}_{cid}.jpg').convert('RGB').resize((1024,576),Image.Resampling.BILINEAR));im[masks[f]]=(im[masks[f]]*.55+np.array([255,160,0])*.45).astype('uint8');panel.paste(Image.fromarray(im).resize((512,288)),(j*512,26));dd.text((j*512+4,5),f'{uid} f{start+f} | RGB supervision candidate',fill='white')
                        thumbs.append((sid,panel))
        counts[sid]=dict(candidates=scene_count,rejections=dict(reasons));print('DATA_SCENE',sid,counts[sid],flush=True)
    dump(ROOT/'manifest.json',dict(registration=registration,candidates=accepted))
    splits={s:{p for c in accepted if c['split']==s for p in c['raw_log_prefixes']} for s in ['train_candidate','validation_candidate']};assert not(splits['train_candidate']&splits['validation_candidate'])
    report=dict(candidates=len(accepted),unique_camera_clips=len({c['source_window_id'] for c in accepted}),unique_time_windows=len({c['source_time_window_id'] for c in accepted}),scene_count=3,counts=counts,split_counts=dict(collections.Counter(c['split'] for c in accepted)),all_masks_tested=30*len(accepted),train_admitted=0,generated_gt_count=0,known_new_eval_log_overlap=0,old_three_eval_log_audit='pending',training_steps=0,human_verdict=None)
    dump(ROOT/'summary.json',report)
    sheet=Image.new('RGB',(1024,314*len(thumbs)))
    for i,(_,p) in enumerate(thumbs):sheet.paste(p,(0,314*i))
    sheet.save(ROOT/'candidate_review.jpg',quality=93);print('DATA_DONE',report,flush=True)
if __name__=='__main__':main()
