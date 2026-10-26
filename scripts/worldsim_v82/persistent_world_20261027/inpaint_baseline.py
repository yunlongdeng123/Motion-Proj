"""固定预训练ProPainter端到端基线；只适配PNG读写，保存原生与最终输出。"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from PIL import Image

SOURCE=Path('/root/autodl-tmp/third_party/worldsim_v77/DiffuEraser')
sys.path.insert(0,str(SOURCE))


def main():
    from resource_control import require_start
    require_start()
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True)
    p.add_argument('--input-kind',choices=['native_3d','masked_raw_rgb'],default='native_3d')
    p.add_argument('--rgb-subdir',default='native_rgb');p.add_argument('--mask-subdir',default='mask')
    a=p.parse_args();root=Path(a.root);case=root/a.case;dest=case/'propainter'
    dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'summary.json').exists(),'不覆盖完成结果'
    frames=sorted((case/a.rgb_subdir).glob('*.png'))
    masks=sorted((case/a.mask_subdir).glob('*.png'))
    assert len(frames)==len(masks)>0
    started=time.time();count=len(frames)
    (dest/'status.json').write_text(json.dumps(dict(status='running',pid=os.getpid(),frames=count)))
    def read_frames(path,length):
        seq=[Image.open(f).convert('RGB') for f in sorted(Path(path).glob('*.png'))]
        return seq,10.,seq[0].size,a.input_kind,len(seq)
    def read_masks(path,n,size,flow_mask_dilates=0,mask_dilates=0):
        assert flow_mask_dilates==0 and mask_dilates==0
        seq=[Image.open(f).convert('L').resize(size,Image.Resampling.NEAREST) for f in sorted(Path(path).glob('*.png'))]
        assert len(seq)==n
        return seq,seq
    def capture(seq):
        output=dest/'model_png';output.mkdir(exist_ok=True)
        for source,value in zip(frames,seq):Image.fromarray(np.asarray(value,dtype=np.uint8)).save(output/source.name)
    code=(SOURCE/'propainter/inference.py').read_text()
    needle='        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),'
    assert code.count(needle)==1
    code=code.replace(needle,'        capture(comp_frames)\n'+needle)
    module=types.ModuleType('persistent_world_propainter_io');module.__file__=str(SOURCE/'propainter/inference.py');module.capture=capture
    exec(compile(code,module.__file__,'exec'),module.__dict__)
    module.read_frame_from_videos=read_frames;module.read_mask=read_masks
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.cuda.reset_peak_memory_stats()
    model=module.Propainter('/root/autodl-tmp/models/worldsim_v77_poc_propainter',torch.device('cuda'))
    model.forward(str(case/a.rgb_subdir),str(case/a.mask_subdir),str(dest/'video.mp4'),
        resize_ratio=1.,video_length=count/10,mask_dilation=0,save_fps=10,
        neighbor_length=10,subvideo_length=50,ref_stride=10)
    records=[]
    for f,m in zip(frames,masks):
        old=np.asarray(Image.open(f));mask=np.asarray(Image.open(m))>0
        result=np.asarray(Image.open(dest/'model_png'/f.name))
        records.append(dict(raw=int(f.stem),outside_max_difference=int(np.abs(result.astype(float)-old)[~mask].max()) if (~mask).any() else None,
            edit_pixels=int(mask.sum()),inside_change_MAE=float(np.abs(result.astype(float)-old)[mask].mean()/255) if mask.any() else None))
    summary=dict(status='complete',model='pretrained ProPainter',training_updates=0,
        frames=count,seconds=time.time()-started,rows=records,
        cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),
        mask_extra_dilation=0,seed=0,input_kind=a.input_kind,native_3d_preserved=a.input_kind=='native_3d',
        consumes_existing_gaussians_or_latents=False,
        generated_output_is_3d_state=False,
        limitation='固定全洞mask会丢弃洞内三维颜色；后续DiffuEraser应显式接收原生三维prior，且需做同输入控制。',
        quality_accepted=False,human_verdict=None)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    from gpu_slot import exclusive
    with exclusive():main()
