"""在已审计输入目录运行编码/单卡clean；遇OOM立即停止，不重试或变更设置。"""
import argparse
import gc
import json
import time
from pathlib import Path
import av
import numpy as np
from PIL import Image
import torch
from common import CAMERA,config

def encode(out):
    assert not (out/'embeddings.pt').exists()
    cfg=config()
    model=cfg.text_encoder.setup().to('cuda').eval()
    with torch.inference_mode():
        text=model([(out/'prompt.txt').read_text().strip()]).unsqueeze(0).cpu()
    assert text.shape==(1,1,512,100352) and torch.isfinite(text).all()
    del model
    gc.collect();torch.cuda.empty_cache()
    model=cfg.image_encoder.setup().to('cuda').eval()
    image=torch.from_numpy(np.asarray(Image.open(out/'initial_rgb.png')).copy())
    image=image.permute(2,0,1)[None,None,None].to('cuda',dtype=torch.bfloat16)/127.5-1
    with torch.inference_mode():
        image=model(image).cpu()
    assert image.shape==(1,1,1,16,88,160) and torch.isfinite(image).all()
    torch.save({'text_embeddings':text,'image_embeddings':image,'negative_text_embeddings':None},out/'embeddings.pt')
    return {'text_shape':list(text.shape),'image_shape':list(image.shape),'generation_calls':0}

def generate(out,seed):
    manifest=json.loads((out/'input_manifest.json').read_text())
    for name in ['clean.npy','clean.mp4']:
        assert not (out/name).exists(), f'拒绝覆盖{name}'
    inputs=np.load(out/'conditions.npy',mmap_mode='r')
    assert inputs.shape==(237,704,1280,3)
    cfg=config();cfg.text_encoder=None;cfg.image_encoder=None;cfg.diffusion_model.seed=seed
    pipeline=cfg.setup().to('cuda').eval()
    embeddings=torch.load(out/'embeddings.pt',weights_only=True,map_location='cpu')
    cache=pipeline.initialize_cache_from_embeddings(**embeddings,view_names=[CAMERA])
    result=np.lib.format.open_memmap(out/'clean.npy',mode='w+',dtype=np.uint8,shape=inputs.shape)
    timings=[];cursor=0
    with av.open(str(out/'clean.mp4'),'w') as writer:
        stream=writer.add_stream('libx264',rate=30)
        stream.width,stream.height,stream.pix_fmt=1280,704,'yuv420p';stream.options={'crf':'18'}
        for block in range(30):
            n=5 if block==0 else 8
            batch=torch.from_numpy(np.array(inputs[cursor:cursor+n],copy=True))
            batch=batch.permute(0,3,1,2)[None,None].to('cuda',dtype=torch.bfloat16)/127.5-1
            began=time.monotonic()
            with torch.inference_mode():
                output=pipeline.generate(autoregressive_index=block,cache=cache,input=batch)
                pipeline.finalize(autoregressive_index=block,cache=cache)
            assert bool(torch.isfinite(output).all())
            frames=((output[0,0].float().clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
            assert len(frames)==n
            result[cursor:cursor+n]=frames
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):
                    writer.mux(packet)
            cursor+=n
            row={'block':block,'frames':cursor,'elapsed_s':time.monotonic()-began}
            timings.append(row);print(json.dumps(row),flush=True)
        for packet in stream.encode():
            writer.mux(packet)
    result.flush()
    return {'frames':cursor,'timings':timings,'seed':seed,'generation_calls':30,
            'model_view_name':CAMERA,'model_view_name_is_not_source_calibration':True,
            'policy_feedback':False,'natural_reconstruction_input':manifest.get('natural_reconstruction_input',False),
            'state_source':manifest.get('state_source','GT / native conditions'),
            'extra_information':manifest.get('extra_information',[])}

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['encode','generate'])
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();out=a.input_dir
    manifest=json.loads((out/'input_manifest.json').read_text())
    assert json.loads((out/'render_result.json').read_text())['status']=='passed'
    path=out/f'{a.phase}_result.json';assert not path.exists(),'拒绝重跑终态任务'
    result={'phase':a.phase,'input_manifest':str(out/'input_manifest.json'),
            'role':manifest['role'],'status':'started','human_verdict':None}
    began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        result.update(encode(out) if a.phase=='encode' else generate(out,a.seed))
        result['status']='complete'
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        result.update(wall_s=time.monotonic()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='timings'}),flush=True)

if __name__=='__main__':
    main()
