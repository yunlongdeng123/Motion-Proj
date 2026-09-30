"""GPU阶段的可续跑入口。默认CPU验队列；只有显式--execute才加载SAM2。"""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segment_sources import CHECKPOINT,CONFIG

def read(p):return json.loads(p.read_text())
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def validate(root):
    from PIL import Image
    q=read(root/'gpu_queue.json');src={c['source_id']:c for c in read(root/'native10_factory/source_manifest.json')['clips']}
    qa={c['case_id']:c for c in read(root/'independent_preflight_reviews.json')['cases']};jobs=[]
    for j in q['jobs']:
        if not j.get('eligible_after_GPU_authorization'):continue
        assert qa[j['source_id']]['source_status']=='pass'
        c=src[j['source_id']];fs=c['frames'];assert len(fs)==10 and [f['filename'] for f in fs]==j['source_frames']
        stamps=[f['timestamp'] for f in fs];assert all(0<b-a<=180000 for a,b in zip(stamps,stamps[1:]));assert max(f['delta_ms'] for f in fs)<=55
        assert 0<=j['prompt_frame']<len(fs)
        for f,b in zip(fs,j['box_prompts']):
            a=next(a for a in f['actors'] if a['instance_token']==j['instance_token']);assert a['projection']['box_xyxy']==b
            with Image.open(root/'native10_factory/rgb'/f['filename']) as im:im.verify()
        jobs.append(j)
    return jobs,src

def reusable(dest,j):
    from PIL import Image
    p=dest/'mask_manifest.json'
    if not p.exists():return False
    r=read(p)
    if any(r.get(k)!=j[k] for k in ['job_id','source_id','instance_token','source_frames','prompt_frame','box_prompts']):return False
    if r.get('checkpoint')!=CHECKPOINT or r.get('config')!=CONFIG:return False
    try:
        assert len(list((dest/'sam2_raw').glob('*.png')))==10
        for i in range(10):
            with Image.open(dest/'sam2_raw'/f'{i:05}.png') as im:im.load();assert im.size==(1024,576) and im.mode=='L'
    except (OSError,AssertionError):return False
    return True

def main(root,execute):
    jobs,sources=validate(root);print('VALIDATED_RECEIVER_JOBS',len(jobs),'execute',execute,flush=True)
    if not execute:return
    assert jobs,'无独立通过的新实例，不启动模型'
    import fcntl
    lock=(root/'receiver_segmentation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch,cv2,numpy as np,importlib.util
    from PIL import Image
    assert torch.cuda.is_available(),'等待用户开启GPU，不退到CPU推理'
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42)
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor(CONFIG,CHECKPOINT,device='cuda')
    statepath=root/'receiver_segmentation_state.json';state=read(statepath) if statepath.exists() else {'completed':[]}
    state.update(state='running',pid=os.getpid(),seed=42);dump(statepath,state)
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for j in jobs:
                dest=root/'segmented_receivers'/j['job_id']
                if reusable(dest,j):continue
                start=time.monotonic();rgb=dest/'rgb';masks=dest/'sam2_raw';rgb.mkdir(parents=True,exist_ok=True);masks.mkdir(exist_ok=True)
                for i,name in enumerate(j['source_frames']):
                    with Image.open(root/'native10_factory/rgb'/name) as im:im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS).save(rgb/f'{i:05}.jpg',quality=96)
                assert {p.name for p in rgb.iterdir()}=={f'{i:05}.jpg' for i in range(10)}
                track=predictor.init_state(video_path=str(rgb),offload_video_to_cpu=True,offload_state_to_cpu=True)
                predictor.add_new_points_or_box(track,frame_idx=j['prompt_frame'],obj_id=1,box=np.float32(j['box_prompts'][j['prompt_frame']]))
                raw={}
                for reverse in [False,True]:
                    for fid,_,logits in predictor.propagate_in_video(track,start_frame_idx=j['prompt_frame'],reverse=reverse):raw[int(fid)]=(logits[0,0]>0).cpu().numpy()
                assert set(raw)==set(range(10))
                for i,m in raw.items():Image.fromarray(m.astype('uint8')*255).save(masks/f'{i:05}.png')
                r={k:j[k] for k in ['job_id','source_id','instance_token','source_frames','prompt_frame','box_prompts']}
                r.update(checkpoint=CHECKPOINT,config=CONFIG,frames=10,raw_pixels=[int(raw[i].sum()) for i in range(10)],
                         seconds=time.monotonic()-start,official_optional_extension_available=importlib.util.find_spec('sam2._C') is not None,
                         postprocess='official defaults; no custom cleanup/box clipping; inspect log if optional hole-fill skipped',quality_status='pending_mask_QA_and_exact_pair_gates',training_ready=False)
                dump(dest/'mask_manifest.json',r);state['completed']=[x for x in state['completed'] if x['job_id']!=j['job_id']]+[r];dump(statepath,state)
                predictor.reset_state(track);del track,raw;torch.cuda.empty_cache();print('MASK_COMPLETE',j['job_id'],flush=True)
        assert all(reusable(root/'segmented_receivers'/j['job_id'],j) for j in jobs)
        state['state']='complete_pending_exact_mask_checks';dump(statepath,state)
    except Exception as e:
        state.update(state='failed',error=repr(e));dump(statepath,state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();main(a.root,a.execute)
