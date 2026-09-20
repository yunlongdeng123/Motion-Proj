"""固定相机的对象移除生成：一个子任务一次模型加载，OOM立即终止。"""
import argparse
import json
import os
from pathlib import Path
import time

import av
import numpy as np
from PIL import Image
import torch

from closed_loop_bridge import upload_scene
from common import CAMERA, config
from render_argoverse import render


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL=ROOT/'WS-V75-ACTOR-REMOVAL-QUALIFY-01/20260921-r2'
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-GENERATION-01/20260921-r1'
FRAMES=117
BLOCKS=15


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('LOCAL_FILES_ONLY','1')
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['reference','dvgt_metric','class_prior'])
    p.add_argument('--variant',required=True,choices=['unedited','removed','edited']);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--qualification',type=Path,default=QUAL);a=p.parse_args()
    q=json.loads((a.qualification/'result.json').read_text());raster=json.loads((a.qualification/'raster_result.json').read_text())
    assert q['status']=='qualified' and q['generation_admitted'] and raster['status']=='passed' and raster['generation_admitted']
    protocol=json.loads((a.output/'protocol.json').read_text());assert protocol['qualification']==str(a.qualification)
    assert protocol['seed']==42 and protocol['frames']==FRAMES and protocol['blocks']==BLOCKS
    info=q['selected']['state_inputs'][a.arm];scene_path=Path(info['source'] if a.variant=='unedited' else info['edited'])
    dest=a.output/f'{a.arm}-{a.variant}';assert not dest.exists();dest.mkdir()
    result={'status':'started','arm':a.arm,'variant':a.variant,'scene':str(scene_path),
            'fixed_camera':True,'policy_feedback':False,'human_verdict':None,'failure_ledger_delta':'none'}
    began=time.monotonic()
    try:
        base=Path(q['selected']['base']);conditioning=Path(q['selected']['conditioning'])
        tr=np.load(base/'trajectory.npz');scene=json.loads(scene_path.read_text())
        conditions=np.lib.format.open_memmap(dest/'conditions.npy',mode='w+',dtype=np.uint8,shape=(FRAMES,704,1280,3))
        ctx,sid,fit=upload_scene(scene,tr['timestamps_us'],tr['K'])
        for start in range(0,FRAMES,8):
            stop=min(FRAMES,start+8);conditions[start:stop]=render(ctx,sid,tr['timestamps_us'][start:stop],tr['camera_world'][start:stop])
        conditions.flush();del ctx;torch.cuda.empty_cache()
        cfg=config();cfg.text_encoder=None;cfg.image_encoder=None;cfg.diffusion_model.seed=42
        pipeline=cfg.setup().to('cuda').eval();embeddings=torch.load(conditioning/'embeddings.pt',weights_only=True,map_location='cpu')
        cache=pipeline.initialize_cache_from_embeddings(**embeddings,view_names=[CAMERA]);torch.cuda.reset_peak_memory_stats()
        generated=np.lib.format.open_memmap(dest/'generated.npy',mode='w+',dtype=np.uint8,shape=(FRAMES,704,1280,3))
        cursor=0;timings=[]
        with av.open(str(dest/'generated.mp4'),'w') as writer:
            stream=writer.add_stream('libx264',rate=30);stream.width=1280;stream.height=704;stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
            for block in range(BLOCKS):
                count=5 if block==0 else 8
                source=torch.from_numpy(np.array(conditions[cursor:cursor+count],copy=True)).permute(0,3,1,2)[None,None]
                source=source.to('cuda',dtype=torch.bfloat16)/127.5-1;start=time.monotonic()
                with torch.inference_mode():
                    output=pipeline.generate(autoregressive_index=block,cache=cache,input=source)
                    pipeline.finalize(autoregressive_index=block,cache=cache)
                torch.cuda.synchronize();frames=((output[0,0].float().clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
                assert len(frames)==count;generated[cursor:cursor+count]=frames
                for frame in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):writer.mux(packet)
                cursor+=count;timings.append({'block':block,'last_frame':cursor-1,'elapsed_s':time.monotonic()-start})
                Image.fromarray(frames[-1]).save(dest/f'generated-{cursor-1:03d}.jpg',quality=94)
                print(json.dumps({'arm':a.arm,'variant':a.variant,**timings[-1]}),flush=True)
            for packet in stream.encode():writer.mux(packet)
        generated.flush();assert cursor==FRAMES
        with av.open(str(dest/'generated.mp4')) as video:decoded=sum(1 for _ in video.decode(video=0))
        assert decoded==FRAMES
        result.update(status='complete',frames=FRAMES,decoded_frames=decoded,blocks=BLOCKS,seed=42,
                      generation_forwards=BLOCKS,condition_render_frames=FRAMES,timings=timings,
                      camera_trajectory=str(base/'trajectory.npz'),conditioning=str(conditioning))
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result.update(wall_s=time.monotonic()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(dest/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='timings'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
