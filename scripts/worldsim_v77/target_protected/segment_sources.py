"""SAM2下一阶段入口。默认只检查队列；当前CPU回合禁止使用 --execute。"""
import argparse
import json
import os
from pathlib import Path
import time

CHECKPOINT='/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt'
CONFIG='configs/sam2.1/sam2.1_hiera_l.yaml'

def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def reusable_result(root,job,row):
    """续跑直接核对来源字段和全部PNG可解码；不用文件指纹。"""
    from PIL import Image
    if any(row.get(k)!=job[k] for k in ['source_id','instance_token','prompt_frame','frame_filenames']):return False
    if row.get('checkpoint')!=CHECKPOINT or row.get('config')!=CONFIG:return False
    if row.get('eligible_source_roles')!=job['eligible_source_roles']:return False
    try:
        for fid in range(30):
            with Image.open(root/'segmented'/job['source_id']/'sam2_raw'/f'{fid:05}.png') as im:
                im.load()
                if im.size!=(1024,576) or im.mode!='L':return False
    except (OSError,ValueError):return False
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    queue=read(a.root/'segmentation_queue.json');assert queue['stage']=='awaiting_gpu'
    sources={c['source_id']:c for c in read(a.root/'source_manifest.json')['clips']}
    gates={c['source_id']:c for c in read(a.root/'source_geometry_validation.json')['clips']}
    review={c['source_id']:c for c in read(a.root/'subagent_source_reviews.json')['clips']}
    assert queue['jobs']
    for job in queue['jobs']:
        sid=job['source_id'];token=job['instance_token']
        assert gates[sid]['primary_geometry_pass']
        assert any(x['instance_token']==token and x['pass'] for x in gates[sid]['actors'])
        assert review[sid]['receiver_status']=='pass' or review[sid]['donor_status']=='pass'
        assert job['role']=='primary_reviewed_actor', '未检查的次要actor不能混入已准入队列'
        assert token==sources[sid]['actors'][0]['instance_token']
        assert job['frame_filenames']==[f['filename'] for f in sources[sid]['frames']]
        assert all((a.root/'rgb'/f['filename']).is_file() for f in sources[sid]['frames'])
    print('QUEUE_VALID',len(queue['jobs']),'execute',a.execute,flush=True)
    if not a.execute:return
    import fcntl
    lock=open(a.root/'segmentation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import sys
    import importlib.util
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    assert torch.cuda.is_available(),'需要用户开启GPU；不要退到CPU批量模型推理'
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42)
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor(CONFIG,CHECKPOINT,device='cuda')
    extension_available=importlib.util.find_spec('sam2._C') is not None
    stpath=a.root/'segmentation_state.json'
    state=read(stpath) if stpath.exists() else {'state':'running','completed':[],'mask_source':'SAM2.1_hiera_large','pid':os.getpid()}
    old={r['source_id']:r for r in state['completed']}
    state.update(state='running',pid=os.getpid());state.pop('error',None);dump(stpath,state)
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for job in queue['jobs']:
                sid=job['source_id']
                if sid in old and reusable_result(a.root,job,old[sid]):continue
                c=sources[sid];start=time.monotonic();dest=a.root/'segmented'/sid;rgb=dest/'rgb';masks=dest/'sam2_raw'
                rgb.mkdir(parents=True,exist_ok=True);masks.mkdir(exist_ok=True)
                for f in c['frames']:
                    path=rgb/f'{f["frame"]:05}.jpg'
                    # 未完整完成的job重建派生输入，不把旧RGB当作当前来源。
                    with Image.open(a.root/'rgb'/f['filename']) as im:
                        im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS).save(path,quality=96)
                assert {p.name for p in rgb.iterdir()}=={f'{fid:05}.jpg' for fid in range(30)}, '旧残留会改变官方视频加载帧序，须先排查'
                prompt=job['prompt_frame'];fr=c['frames'][prompt]
                actor=next(x for x in fr['actors'] if x['instance_token']==job['instance_token'])
                box=np.float32(actor['projection']['box_xyxy'])
                track=predictor.init_state(video_path=str(rgb),offload_video_to_cpu=True,offload_state_to_cpu=True)
                predictor.add_new_points_or_box(track,frame_idx=prompt,obj_id=1,box=box)
                raw={}
                for reverse in [False,True]:
                    for fid,_,logits in predictor.propagate_in_video(track,start_frame_idx=prompt,reverse=reverse):
                        raw[int(fid)]=(logits[0,0]>0).cpu().numpy()
                assert set(raw)==set(range(30))
                for fid,m in raw.items():Image.fromarray(m.astype(np.uint8)*255).save(masks/f'{fid:05}.png')
                row={'source_id':sid,'instance_token':job['instance_token'],'mask_source':'SAM2.1_hiera_large',
                     'checkpoint':CHECKPOINT,'config':CONFIG,'frame_filenames':job['frame_filenames'],
                     'eligible_source_roles':job['eligible_source_roles'],
                     'prompt_frame':prompt,'frames':30,'raw_pixels':[int(raw[i].sum()) for i in range(30)],
                     'postprocess':'official defaults retained; no custom mask cleanup or GT envelope clipping',
                     'official_postprocessing':{'dynamic_multimask_via_stability':True,'binarize_mask_from_pts_for_mem_enc':True,
                                               'fill_hole_area':8,'compiled_extension_available':extension_available,
                                               'hole_fill_execution':'official optional operation; inspect runtime warning if extension fails'},
                     'quality_status':'pending_mask_and_synthetic_QA','seconds':time.monotonic()-start}
                dump(dest/'mask_manifest.json',row)
                state['completed']=[r for r in state['completed'] if r['source_id']!=sid]+[row];dump(stpath,state)
                predictor.reset_state(track);del track,raw;torch.cuda.empty_cache()
        completed={r['source_id']:r for r in state['completed']}
        assert all(reusable_result(a.root,j,completed.get(j['source_id'],{})) for j in queue['jobs'])
        state['state']='complete_pending_quality_review';dump(stpath,state)
    except Exception as exc:
        state.update(state='failed',error=repr(exc));dump(stpath,state);raise
if __name__=='__main__':main()
