"""Same frozen evaluator/ROIs for the separately registered M12 fallback, CPU only."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .material_evaluation import evaluate_arm,load,read,write,FRAMES

def rgb(a):return Image.fromarray(np.clip(np.moveaxis(a,0,-1)*255+.5,0,255).astype(np.uint8))
def main(gpu):
    gpu=Path(gpu);stage=gpu/'source_filtered_training/formal_r1/C800';out=gpu/'evaluation/M12_r1'
    paths={r:stage/f'rendered/{r:03d}.npz'for r in FRAMES}
    missing=[str(p)for p in paths.values()if not p.exists()]
    if missing:raise FileNotFoundError('M12 endpoint incomplete: '+str(missing))
    r=evaluate_arm(paths,gpu/'evaluation/frozen_r2/manifest.json',out,arm_id='M12',stage='C800')
    old=read(gpu/'evaluation/results_r3/M11/metrics.json')
    pairs={}
    for role in ['A','C','S','S_new','S_thin']:
        pairs[role]=dict(M11_photo=old['pooled'][role]['MAE'],M12_photo=r['pooled'][role]['MAE'])
        if role in ['A','C','S','S_new']:
            for label,v in [('M11',old),('M12',r)]:
                pairs[role][label+'_HF1']=v['detail'][role]['1px']['HF_MAE'];pairs[role][label+'_HF_cosine']=v['detail'][role]['1px']['HF_cosine']
                pairs[role][label+'_flat_HF_RMS']=v['detail'][role]['edges']['spurious_HF_RMS_on_teacher_flat']
    stripe=[dict(raw=a['raw'],M11=a['stripe'],M12=b['stripe'])for a,b in zip(old['rows'],r['rows'])]
    write(out/'paired_M11_M12.json',dict(schema='fixed_evaluator_separate_fallback.v1',role='prospectively registered source-footprint-filtered optimization; not original factorial',
        primary5_unchanged=True,extra_formal_physical_updates=600,logical_updates=800,from_shared_pixel_A200=True,mechanical_pass=r['mechanical_pass'],paired=pairs,stripe=stripe,updates_by_evaluator=0))
    views=out/'previews';views.mkdir();manifest=read(gpu/'evaluation/frozen_r2/manifest.json');ROIs=read(gpu/'evaluation/frozen_r2/rois.json')['rois']
    packets={r:load(p)for r,p in paths.items()};font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for raw,z in packets.items():
        for v in ['road_rgb','joint_rgb']:rgb(z[v]).save(views/f'M12_{raw:03d}_{v}.webp',lossless=True)
    for roi in ROIs:
        if roi['bbox_xyxy']is None:continue
        raw=roi['raw'];box=tuple(roi['bbox_xyxy']);w,h=box[2]-box[0],box[3]-box[1]
        for view in ['road_rgb','joint_rgb']:
            columns=[]
            if raw in [86,110]:
                f=load(manifest['frames'][str(raw)]['path']);key='S'if roi['key']=='S_full'else'C'if roi['key']=='C'else'A'
                columns.append(('Fixed teacher (pseudo C/S)',f['target_'+key]))
            for name,path in [('pixel_initial',gpu/f'formal_r1/pixel_initial/rendered/{raw:03d}.npz'),('M01',gpu/f'formal_r1/M01/C800/rendered/{raw:03d}.npz'),('M11',gpu/f'formal_r1/M11/C800/rendered/{raw:03d}.npz'),('M12 source filtered',paths[raw]),('T11',gpu/f'formal_r1/T11/C800/rendered/{raw:03d}.npz')]:columns.append((name,load(path)[view]))
            im=Image.new('RGB',(len(columns)*(3*w+8),3*h+52),(15,19,27));draw=ImageDraw.Draw(im)
            for i,(name,p)in enumerate(columns):
                x=i*(3*w+8);draw.text((x+3,4),name,font=font,fill=(231,237,246));im.paste(rgb(p).crop(box).resize((3*w,3*h),Image.Resampling.NEAREST),(x,31))
            im.save(views/f'M12_ROI_{raw:03d}_{roi["key"]}_{view}.webp',lossless=True)
    print(json.dumps(dict(paired=pairs,stripe=stripe,mechanical_pass=r['mechanical_pass']),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-dir',required=True);a=p.parse_args();main(a.gpu_dir)
