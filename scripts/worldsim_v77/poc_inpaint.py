"""调用官方 ProPainter：每路相机独立时序补背景，保存逐帧 PNG。"""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import numpy as np
from PIL import Image

BASE=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
SOURCE=pathlib.Path('/root/autodl-tmp/third_party/worldsim_v77/ProPainter-hfspace')
WEIGHTS=pathlib.Path('/root/autodl-tmp/models/worldsim_v77_poc_propainter')

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--camera',type=int,required=True)
    p.add_argument('--smoke-frames',type=int);a=p.parse_args()
    root=BASE/a.scene;sel=json.loads((root/'selection.json').read_text());count=a.smoke_frames or sel['frames']
    original=root/'rgb'/f'cam{a.camera}'
    if not original.is_dir():raise ValueError('此相机没有目标框或 RGB 输入')
    mask_input=root/'inpaint_input'/('smoke_' if a.smoke_frames else '')/f'cam{a.camera}'
    mask_input.mkdir(parents=True,exist_ok=True)
    rgb_input=root/'inpaint_rgb_input'/('smoke_' if a.smoke_frames else '')/f'cam{a.camera}'
    rgb_input.mkdir(parents=True,exist_ok=True)
    for f in range(count):
        name=f'{f:05d}'
        for source,target in [(root/'masks'/f'cam{a.camera}'/(name+'.png'),mask_input/(name+'.png')),
                              (original/(name+'.jpg'),rgb_input/(name+'.jpg'))]:
            if not source.is_file():raise FileNotFoundError(source)
            if not target.exists():target.symlink_to(source)
    link=SOURCE/'weights'
    if not link.exists():link.symlink_to(WEIGHTS,target_is_directory=True)
    output=root/('inpaint_smoke' if a.smoke_frames else 'inpaint')
    output.mkdir(exist_ok=True)
    command=[sys.executable,str(SOURCE/'inference_propainter.py'),
      '--video',str(rgb_input),'--mask',str(mask_input),'--output',str(output),
      '--width','688','--height','384','--save_fps','10','--save_frames','--fp16',
      '--subvideo_length','40','--neighbor_length','10','--ref_stride','10']
    start=time.monotonic()
    print(json.dumps({'scene':a.scene,'camera':a.camera,'count':count,'command':command}),flush=True)
    subprocess.run(command,cwd=SOURCE,check=True)
    frames=output/f'cam{a.camera}'/'frames'
    paths=sorted(frames.glob('*.png'))
    if len(paths)!=count:raise AssertionError(f'ProPainter 输出帧数 {len(paths)} != {count}')
    for f,path in enumerate(paths):
        arr=np.asarray(Image.open(path).convert('RGB'))
        if arr.shape!=(384,688,3):raise AssertionError((path,arr.shape))
    summary={'scene':a.scene,'camera':a.camera,'frames':count,'elapsed_s':time.monotonic()-start,
       'source':str(SOURCE),'weights':str(WEIGHTS),'output':str(frames),'human_verdict':None}
    (output/f'cam{a.camera}'/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
