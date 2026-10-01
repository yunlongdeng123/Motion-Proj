"""对有真实空间预案的来源批量SAM2；所有输出待质量验证，不自动放行。"""
from pathlib import Path
import sys,json,os,time
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
from geometry_factory import read,dump
from segment_sources import CHECKPOINT,CONFIG
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r8';F=O/'factory'
def prepare(all_sources=False):
    src={c['source_id']:c for c in read(F/'source_manifest.json')['clips']}
    rows=[read(p) for folder in ['preflight','cross_preflight'] for p in (O/folder).glob('N*.json')]
    ids=sorted(s for s in src if s.startswith('N')) if all_sources else sorted({r['source_id'] for r in rows if r['plans']})
    jobs=[]
    for sid in ids:
        c=src[sid]
        for j,a in enumerate(c['actors']):
            tok=a['instance_token'];jid=sid if j==0 else sid+'_'+tok[:8]
            dest=F/('segmented' if j==0 else 'segmented_secondary')/jid
            jobs.append({'job_id':jid,'source_id':sid,'instance_token':tok,'primary':j==0,'dest':str(dest),'source_frames':[f['filename'] for f in c['frames']],
                         'boxes':[next(a for a in f['actors'] if a['instance_token']==tok)['projection']['box_xyxy'] for f in c['frames']],'prompt_frame':5})
    oldtrain=set(read(O/'source_pool_summary.json')['old_train_scenes'])
    selfscenes=sorted({r['scene'] for r in rows if r['plans'] and any(p['donor_source_id']==r['source_id'] for p in r['plans']) and r['scene'] not in oldtrain})
    evalscenes=selfscenes[:5]
    if (O/'source_split.json').exists():assert read(O/'source_split.json')['synthetic_validation_scenes']==evalscenes
    else:dump(O/'source_split.json',{'synthetic_validation_scenes':evalscenes,'rule':'first five lexicographic self-template geometric feasible receiver scenes absent from all r7 training receiver/donor scenes, fixed before SAM/outputs','old_r7_training_scenes':sorted(oldtrain),'all_remaining_roles':'training candidates; not final','final_used':False})
    dump(O/'sam_queue.json',{'jobs':jobs,'numeric_proposal_sources':ids,'quality_status':'diagnostic masks; final independent five-check QA required before training'})
    print('QUEUE',len(jobs),'sources',len(ids),'scenes',len({src[s]['scene'] for s in ids}),'validation_scenes',evalscenes,flush=True)
    return jobs,src
def main():
    jobs,src=prepare('--all-sources' in sys.argv)
    import fcntl,torch,cv2
    lock=open(O/'sam.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42)
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2_video_predictor
    predictor=None;state=read(O/'sam_state.json') if (O/'sam_state.json').exists() else {'completed':[]}
    state.update(stage='running',pid=os.getpid(),jobs=len(jobs));dump(O/'sam_state.json',state)
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for job in jobs:
                dest=Path(job['dest']);masks=dest/'sam2_raw';files=sorted(masks.glob('*.png'))
                if len(files)==10:
                    for p in files:
                        with Image.open(p) as im:im.load();assert im.size==(1024,576)
                    if job['job_id'] not in {r['job_id'] for r in state['completed']}:
                        state['completed'].append({'job_id':job['job_id'],'reused':True,'source_frames':job['source_frames']});dump(O/'sam_state.json',state)
                    continue
                if predictor is None:predictor=build_sam2_video_predictor(CONFIG,CHECKPOINT,device='cuda')
                start=time.time();rgb=dest/'rgb';rgb.mkdir(exist_ok=True,parents=True);masks.mkdir(exist_ok=True)
                for i,name in enumerate(job['source_frames']):
                    with Image.open(F/'rgb'/name) as im:im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS).save(rgb/f'{i:05}.jpg',quality=96)
                assert {p.name for p in rgb.glob('*.jpg')}=={f'{i:05}.jpg' for i in range(10)}
                track=predictor.init_state(video_path=str(rgb),offload_video_to_cpu=True,offload_state_to_cpu=True)
                predictor.add_new_points_or_box(track,frame_idx=5,obj_id=1,box=np.float32(job['boxes'][5]));raw={}
                for reverse in [False,True]:
                    for fid,_,logits in predictor.propagate_in_video(track,start_frame_idx=5,reverse=reverse):raw[int(fid)]=(logits[0,0]>0).cpu().numpy()
                assert set(raw)==set(range(10))
                for i,m in raw.items():Image.fromarray(m.astype('uint8')*255).save(masks/f'{i:05}.png')
                row=job|{'reused':False,'seconds':time.time()-start,'checkpoint':CHECKPOINT,'config':CONFIG,'postprocess':'official defaults; no clipping/custom cleanup','quality_status':'pending_final_QA'}
                dump(dest/'mask_manifest.json',row);state['completed'].append(row);dump(O/'sam_state.json',state);predictor.reset_state(track);del track,raw;torch.cuda.empty_cache();print('SAM_DONE',job['job_id'],flush=True)
        state.update(stage='complete_pending_QA');dump(O/'sam_state.json',state)
    except Exception as e:state.update(stage='failed',error=repr(e));dump(O/'sam_state.json',state);raise
if __name__=='__main__':main()
