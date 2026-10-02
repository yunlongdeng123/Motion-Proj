"""在遮后视频上独立重跑SAM2；旧Y分割不能用于方法条件。"""
from pathlib import Path
import os,sys,json,time,fcntl
os.environ['OMP_NUM_THREADS']='2'
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path[:0]=[str(P),'/root/autodl-tmp/third_party/worldsim_v32/sam2']
from segment_sources import CHECKPOINT,CONFIG
import numpy as np,torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r22'
def read(p):return json.loads(p.read_text())
def dump(p,d):q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)

def main():
    assert read(T/'r21/baseline_state.json')['stage']=='complete_pending_review','先结束唯一GPU入口对照'
    lock=open(O/'segmentation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(42)
    queue=read(O/'observed_queue.json')['jobs'];sp=O/'segmentation_state.json'
    state=read(sp) if sp.exists() else {'completed':[]}
    predictor=build_sam2_video_predictor(CONFIG,CHECKPOINT,device='cuda')
    state.update(stage='running',pid=os.getpid(),total_jobs=len(queue));dump(sp,state)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for job in queue:
            out=O/'observed_masks'/job['job_id'];out.mkdir(parents=True,exist_ok=True)
            if any(r['job_id']==job['job_id'] for r in state['completed']):
                for i in range(30):
                    with Image.open(out/f'{i:05}.png') as im:im.load();assert im.size==(1024,576)
                continue
            start=time.monotonic();inp=Path(job['input_folder'])
            track=predictor.init_state(video_path=str(inp/'sam_rgb'),offload_video_to_cpu=True,offload_state_to_cpu=True)
            prompt=job['prompt_frame'];predictor.add_new_points_or_box(track,frame_idx=prompt,obj_id=1,box=np.float32(job['prompt_box']))
            raw={}
            for reverse in [False,True]:
                for fid,_,logits in predictor.propagate_in_video(track,start_frame_idx=prompt,reverse=reverse):raw[int(fid)]=(logits[0,0]>0).cpu().numpy()
            assert len(raw)==30;stats=[]
            for i in range(30):
                H=np.asarray(Image.open(inp/'H'/f'{i:05}.png'))>0;m=raw[i]&~H
                Image.fromarray(m.astype('uint8')*255).save(out/f'{i:05}.png')
                stats.append({'frame':i,'raw_pixels':int(raw[i].sum()),'observed_pixels':int(m.sum()),'discarded_inside_hole':int((raw[i]&H).sum())})
            row=job|{'seconds':time.monotonic()-start,'checkpoint':CHECKPOINT,'config':CONFIG,'stats':stats,'postprocess':'intersect_only_with_legally_visible_pixels; no_GT_crop','identity_quality':'pending_visual_review','human_verdict':None}
            dump(out/'result.json',row);state['completed'].append(row);dump(sp,state);print('OBSERVED_SAM',job['job_id'],row['seconds'],flush=True)
            predictor.reset_state(track);del track,raw;torch.cuda.empty_cache()
    state.update(stage='complete_pending_identity_review');dump(sp,state)

if __name__=='__main__':main()
