"""冻结协议下的单个配对 rollout；输入不变项显式复用，OOM不重试。"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import av
import numpy as np
import torch
from common import CAMERA, RUN, config
from select_target import P1

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True)
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args()
    protocol=json.loads((P1/'protocol.json').read_text())
    audit=json.loads((P1/'condition_audit.json').read_text())
    assert audit['zero_offset_exactly_equal'] and audit['status']=='passed'
    assert a.case in protocol['cases'] and a.seed in protocol['seed_values']
    stem=f'{a.case}-seed{a.seed}'
    out=P1/'rollouts'
    out.mkdir(exist_ok=True)
    result_path=out/f'{stem}.json'
    for ext in ['.json','.mp4','.npy']:
        assert not (out/f'{stem}{ext}').exists(), f'拒绝覆盖 {stem}{ext}'
    clean=np.load(RUN/'conditions.npy',mmap_mode='r')
    biased=clean if a.case=='clean' else np.load(P1/f'conditions-{a.case.split("_")[0]}.npy',mmap_mode='r')
    embeddings=torch.load(RUN/'embeddings.pt',map_location='cpu',weights_only=True)
    cfg=config()
    cfg.text_encoder=None
    cfg.image_encoder=None
    cfg.diffusion_model.seed=a.seed
    result={'case':a.case,'seed':a.seed,'target_id':protocol['target_id'],
            'protocol_path':str(P1/'protocol.json'),'embedding_path':str(RUN/'embeddings.pt'),
            'clean_condition_path':str(RUN/'conditions.npy'),
            'started_utc':datetime.now(timezone.utc).isoformat(), 'policy_feedback':False,
            'physical_reference_changed':False,'human_verdict':None, 'timings':[]}
    began_all=time.monotonic()
    try:
        pipeline=cfg.setup().to('cuda').eval()
        cache=pipeline.initialize_cache_from_embeddings(**embeddings,view_names=[CAMERA])
        torch.cuda.reset_peak_memory_stats()
        frames_out=np.lib.format.open_memmap(out/f'{stem}.npy',mode='w+',dtype=np.uint8,shape=clean.shape)
        cursor=0
        with av.open(str(out/f'{stem}.mp4'),mode='w') as writer:
            stream=writer.add_stream('libx264',rate=30)
            stream.width,stream.height,stream.pix_fmt=1280,704,'yuv420p'
            stream.options={'crf':'18'}
            for block in range(30):
                n=5 if block==0 else 8
                active=a.case!='clean' and cursor>=protocol['start_frame'] and (
                    a.case.endswith('persistent') or cursor<protocol['restore_frame'])
                data=biased if active else clean
                source=torch.from_numpy(np.array(data[cursor:cursor+n],copy=True))
                source=source.permute(0,3,1,2)[None,None].to('cuda',dtype=torch.bfloat16)/127.5-1
                began=time.monotonic()
                with torch.inference_mode():
                    output=pipeline.generate(autoregressive_index=block,cache=cache,input=source)
                    pipeline.finalize(autoregressive_index=block,cache=cache)
                assert bool(torch.isfinite(output).all()), '非有限生成输出，停止'
                frames=((output[0,0].float().clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
                assert len(frames)==n
                frames_out[cursor:cursor+n]=frames
                for frame in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):
                        writer.mux(packet)
                cursor+=n
                result['timings'].append({'block':block,'through_frame':cursor-1,'biased_condition':active,
                                          'elapsed_s':time.monotonic()-began})
                print(json.dumps(result['timings'][-1]),flush=True)
            for packet in stream.encode():
                writer.mux(packet)
        frames_out.flush()
        result.update(status='complete',frames=cursor,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    except BaseException as e:
        result.update(status='oom_stopped' if isinstance(e,torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(e).__name__,error=str(e),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        raise
    finally:
        result['wall_s']=time.monotonic()-began_all
        result['ended_utc']=datetime.now(timezone.utc).isoformat()
        result_path.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    main()
