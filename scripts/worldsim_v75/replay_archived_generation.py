"""从保留条件、编码与 seed 复现旧生成数组；默认仅 CPU 预检，执行必须另给新输出。"""
import argparse
import gzip
import json
import os
from pathlib import Path
import numpy as np

DEFAULT=Path('/root/autodl-tmp/cleanup_manifests/20260922/replay-recipes.json')
if not DEFAULT.is_file():
    DEFAULT=Path(__file__).resolve().parents[2]/'docs/autoresearch/storage_cleanup_20260922/replay-recipes.json'

def open_array(path):
    path=Path(path).resolve()
    if path.is_file():return path.open('rb')
    return gzip.open(path.with_name(path.name+'.gz'),'rb')

def header(path):
    with open_array(path) as f:
        version=np.lib.format.read_magic(f)
        assert version in [(1,0),(2,0)]
        fn=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,order,dtype=fn(f)
        assert not order and dtype==np.uint8
        return shape

def inspect(recipe):
    shape=header(recipe['conditions'])
    assert shape[1:]==(704,1280,3) and (shape[0]-5)%8==0
    assert isinstance(recipe['seed'],int)
    assert Path(recipe['embeddings']).is_file()
    assert Path(recipe['result']).is_file()
    for seg in recipe['segments']:assert header(seg['file'])==shape
    # MP4 仅保留审阅用途；核对帧数，不用有损解码替代原指标。
    import av
    with av.open(recipe['review_video']) as vid:
        frames=vid.streams.video[0].frames
        assert frames==shape[0],(recipe['original_output'],frames,shape)
    return shape

def execute(recipe,output):
    from common import CAMERA, config
    import torch
    shape=inspect(recipe)
    assert output.resolve()!=Path(recipe['original_output']).resolve()
    assert not output.exists() and not output.with_suffix('.replay.json').exists()
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    arrays={}
    for name in [recipe['conditions']]+[s['file'] for s in recipe['segments']]:
        with open_array(name) as f:arrays[name]=np.load(f,allow_pickle=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'started','original':recipe['original_output'],'seed':recipe['seed'],
            'mode':'saved-condition replay; not a new closed-loop experiment','human_verdict':None}
    try:
        cfg=config();cfg.text_encoder=None;cfg.image_encoder=None;cfg.diffusion_model.seed=recipe['seed']
        pipeline=cfg.setup().to('cuda').eval()
        embeddings=torch.load(recipe['embeddings'],weights_only=True,map_location='cpu')
        cache=pipeline.initialize_cache_from_embeddings(**embeddings,view_names=[CAMERA])
        out=np.lib.format.open_memmap(output,mode='w+',dtype=np.uint8,shape=shape)
        cursor=0
        for block in range(1+(shape[0]-5)//8):
            count=5 if block==0 else 8
            source=arrays[recipe['conditions']]
            for seg in recipe['segments']:
                if cursor>=seg['start'] and (seg['stop'] is None or cursor<seg['stop']):source=arrays[seg['file']]
            batch=torch.from_numpy(np.array(source[cursor:cursor+count],copy=True)).permute(0,3,1,2)[None,None].to('cuda',dtype=torch.bfloat16)/127.5-1
            with torch.inference_mode():
                pred=pipeline.generate(autoregressive_index=block,cache=cache,input=batch)
                pipeline.finalize(autoregressive_index=block,cache=cache)
            assert bool(torch.isfinite(pred).all())
            frames=((pred[0,0].float().clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
            assert len(frames)==count
            out[cursor:cursor+count]=frames;cursor+=count
        out.flush();result.update(status='complete',frames=cursor)
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',error_type=type(exc).__name__)
        raise
    finally:
        output.with_suffix('.replay.json').write_text(json.dumps(result,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipes',type=Path,default=DEFAULT)
    ap.add_argument('--index',type=int)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--output',type=Path)
    a=ap.parse_args();rows=json.loads(a.recipes.read_text())
    selected=rows if a.index is None else [rows[a.index]]
    for r in selected:inspect(r)
    print(json.dumps({'checked':len(selected),'missing_inputs':0,'mode':'CPU preflight'}),flush=True)
    if a.execute:
        assert a.index is not None and a.output is not None
        execute(selected[0],a.output)

if __name__=='__main__':main()
