"""Lossless, fixed-ROI CPU views of saved float packets; no exposure adjustment."""
from pathlib import Path
import json,argparse,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont

FRAMES=(86,110,134,179,198)
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def rgb(a):return Image.fromarray(np.clip(np.moveaxis(a,0,-1)*255+.5,0,255).astype(np.uint8))
def build(gpu,out):
    gpu=Path(gpu);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=read(gpu/'evaluation/frozen_r2/manifest.json');rois=read(gpu/'evaluation/frozen_r2/rois.json')['rois']
    cases={'N_parent':gpu/'evaluation/frozen_r2'}
    for k in ['pixel_initial','pixel_common_A200','avg_initial','avg_common_A200']:cases[k]=gpu/f'formal_r1/{k}/rendered'
    for k in ['M00','M10','M01','M11','T11']:cases[k]=gpu/f'formal_r1/{k}/C800/rendered'
    packets={};frozen={};sources=[]
    for raw in FRAMES:
        f=load(manifest['frames'][str(raw)]['path']);frozen[raw]=f
        for name,d in cases.items():
            p=d/f'{raw:03d}.npz'if name!='N_parent'else d/f'{raw:03d}_baseline_packet.npz'
            packets[name,raw]=load(p);sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            for view in ['road_rgb','joint_rgb']:
                rgb(packets[name,raw][view]).save(out/f'{name}_{raw:03d}_{view}.webp',lossless=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    crops=[]
    for i,r in enumerate(rois):
        if r['bbox_xyxy']is None:continue
        raw=r['raw'];box=tuple(r['bbox_xyxy']);w,h=box[2]-box[0],box[3]-box[1]
        teacher=frozen[raw].get('target_S'if r['key']=='S_full'else'target_C'if r['key']=='C'else'target_A')
        for view in ['road_rgb','joint_rgb']:
            columns=[]
            if raw in (86,110):
                columns.append(('Observed (no GT under car)',rgb(frozen[raw]['target_A'])))
                columns.append(('Frozen pseudo teacher',rgb(teacher)))
            for name in ['N_parent','pixel_initial','pixel_common_A200','M00','M10','M01','M11','T11']:
                columns.append((name,rgb(packets[name,raw][view])))
            canvas=Image.new('RGB',(len(columns)*(w*3+8),h*3+52),(15,19,27));dr=ImageDraw.Draw(canvas)
            for n,(label,im)in enumerate(columns):
                x=n*(w*3+8);dr.text((x+3,4),label,font=font,fill=(231,237,246))
                canvas.paste(im.crop(box).resize((w*3,h*3),resample=Image.Resampling.NEAREST),(x,31))
            name=f'ROI_{raw:03d}_{r["key"]}_{view}.webp';canvas.save(out/name,lossless=True)
            crops.append(dict(**r,view=view,file=name,scale=3,interpolation='nearest; no sharpening',columns=[a for a,b in columns]))
    (out/'manifest.json').write_text(json.dumps(dict(schema='fixed_native_float_preview.v1',ROI_source=str(gpu/'evaluation/frozen_r2/rois.json'),
        color='native sRGB values clipped only for 8bit display; no exposure/contrast/sharpening; all metrics original float',
        teacher_warning='A observed; C/S pseudo. Observed car ROI is not under-car road ground truth.',sources=sources,crops=crops),indent=2,ensure_ascii=False)+'\n')
    print(str(out/'manifest.json'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--gpu-dir',required=True);a.add_argument('--out',required=True);args=a.parse_args();build(args.gpu_dir,args.out)
