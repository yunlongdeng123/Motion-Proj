"""固定帧数SAM执行器；方法输入与Y评价输入显式分目录、分角色。

方法只能读取target guard已擦除的RGB；完整Y只允许quality_labels队列。
"""
from pathlib import Path
import os,sys,time,fcntl,argparse,json
S=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path[:0]=[str(S),str(S/'iteration12'),'/root/autodl-tmp/third_party/worldsim_v32/sam2']
from segment_sources import CHECKPOINT,CONFIG
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


def read(p):return json.loads(p.read_text())
def dump(p,data):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');q.replace(p)


def main(root):
    lock=open(root/'segmentation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=read(root/'observed_queue.json')['jobs'];state=read(root/'segmentation_state.json') if (root/'segmentation_state.json').exists() else {'completed':[]}
    torch.set_num_threads(2);torch.manual_seed(42)
    model=build_sam2_video_predictor(CONFIG,CHECKPOINT,device='cuda');state.update(stage='running',pid=os.getpid(),total_jobs=len(jobs));dump(root/'segmentation_state.json',state)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for job in jobs:
            if job.get('role')=='evaluation_only_full_Y':
                assert root.name=='quality_labels' and job.get('forbidden_for_condition_building') is True
            else:
                assert root.name!='quality_labels','评价队列不得混入方法条件角色'
            n=job['frames'];dest=root/'observed_masks'/job['job_id'];dest.mkdir(parents=True,exist_ok=True)
            if any(r['job_id']==job['job_id'] for r in state['completed']):
                assert all((dest/f'{i:05}.png').exists() for i in range(n));continue
            start=time.time();folder=Path(job['input_folder']);track=model.init_state(video_path=str(folder/'sam_rgb'),offload_video_to_cpu=True,offload_state_to_cpu=True)
            prompt=job['prompt_frame'];model.add_new_points_or_box(track,frame_idx=prompt,obj_id=1,box=np.float32(job['prompt_box']));raw={}
            for reverse in [False,True]:
                for i,_,logits in model.propagate_in_video(track,start_frame_idx=prompt,reverse=reverse):raw[int(i)]=(logits[0,0]>0).cpu().numpy()
            assert set(raw)==set(range(n));stats=[]
            for i in range(n):
                guard=np.asarray(Image.open(folder/'H'/f'{i:05}.png'))>0;m=raw[i]&~guard
                Image.fromarray(m.astype('uint8')*255).save(dest/f'{i:05}.png');stats.append({'frame':i,'visible_pixels':int(m.sum()),'inside_target_guard_removed':int((raw[i]&guard).sum())})
            rec=job|{'seconds':time.time()-start,'stats':stats,'checkpoint':CHECKPOINT,'config':CONFIG,'quality':'pending_visual_review','human_verdict':None}
            dump(dest/'result.json',rec);state['completed'].append(rec);dump(root/'segmentation_state.json',state)
            print('RETAINED',job['job_id'],len(state['completed']),len(jobs),flush=True);model.reset_state(track);del track,raw;torch.cuda.empty_cache()
    state['stage']='complete_pending_identity_review';dump(root/'segmentation_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
