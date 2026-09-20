"""显式输入的官方基线；默认仅检查，--execute 才开始世界模型生成。"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('LOCAL_FILES_ONLY','1')
from common import CAMERA, RUN, config

def main():
    import numpy as np
    import torch
    p=argparse.ArgumentParser()
    p.add_argument('--execute',action='store_true')
    p.add_argument('--blocks',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=RUN/'baseline-seed42.mp4')
    a=p.parse_args()
    manifest=json.loads((RUN/'input_manifest.json').read_text())
    conditions=np.load(RUN/'conditions.npy',mmap_mode='r')
    embeddings=torch.load(RUN/'embeddings.pt',map_location='cpu',weights_only=True)
    assert 1<=a.blocks<=manifest['blocks']
    assert conditions.shape==(manifest['input_frames'],704,1280,3) and conditions.dtype==np.uint8
    assert embeddings['text_embeddings'].shape==(1,1,512,100352)
    assert embeddings['image_embeddings'].shape==(1,1,1,16,88,160)
    assert all(torch.isfinite(x).all() for x in embeddings.values() if x is not None)
    plan={'scene':manifest['scene'],'condition_path':str(RUN/'conditions.npy'),
        'embedding_path':str(RUN/'embeddings.pt'),'seed':a.seed,'blocks':a.blocks,
        'input_frames':5+(a.blocks-1)*8,'fps':30,'output':str(a.output),
        'mode':'execute' if a.execute else 'check_only','policy_feedback':False,
        'human_verdict':None}
    print(json.dumps(plan,ensure_ascii=False),flush=True)
    if not a.execute:
        (RUN/'baseline_command_check.json').write_text(json.dumps(plan,indent=2)+'\n')
        return
    if a.output.exists():
        raise FileExistsError(a.output)
    import av
    cfg=config()
    cfg.text_encoder=None
    cfg.image_encoder=None
    cfg.diffusion_model.seed=a.seed
    pipeline=cfg.setup().to('cuda').eval()
    cache=pipeline.initialize_cache_from_embeddings(**embeddings,view_names=[CAMERA])
    torch.cuda.reset_peak_memory_stats()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    cursor=0
    with av.open(str(a.output),mode='w') as writer:
        stream=writer.add_stream('libx264',rate=30)
        stream.width=1280
        stream.height=704
        stream.pix_fmt='yuv420p'
        stream.options={'crf':'18'}
        for block in range(a.blocks):
            count=5 if block==0 else 8
            source=torch.from_numpy(np.array(conditions[cursor:cursor+count],copy=True))
            source=source.permute(0,3,1,2)[None,None].to(device='cuda',dtype=torch.bfloat16)/127.5-1
            began=time.monotonic()
            with torch.inference_mode():
                output=pipeline.generate(autoregressive_index=block,cache=cache,input=source)
                pipeline.finalize(autoregressive_index=block,cache=cache)
            torch.cuda.synchronize()
            frames=((output[0,0].float().clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
            assert len(frames)==count
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):
                    writer.mux(packet)
            cursor+=count
            rows.append({'block':block,'frames':len(frames),'elapsed_s':time.monotonic()-began})
            print(json.dumps(rows[-1]),flush=True)
        for packet in stream.encode(): writer.mux(packet)
    plan.update({'status':'complete','generation_forwards':a.blocks,'frames':cursor,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'timings':rows})
    a.output.with_suffix('.json').write_text(json.dumps(plan,indent=2)+'\n')

if __name__=='__main__':
    main()
