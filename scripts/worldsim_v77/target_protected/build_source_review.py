"""CPU素材页：逐帧真实RGB/GT定位，独立抽帧图；不伪造合成或模型输出。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def font(size):
    for p in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()
def source(root,n):
    for p in [root/'rgb'/n,root/'by_shard/03'/n,root/'by_shard/07'/n]:
        if p.is_file():return p
    return None
def crop(im,box,scale=1.35):
    x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2
    w=(x1-x0)*scale;h=(y1-y0)*scale
    return im.crop((max(0,int(cx-w/2)),max(0,int(cy-h/2)),min(1024,int(cx+w/2)),min(576,int(cy+h/2))))
def fit(im,wh,bg=(20,25,34)):
    out=Image.new('RGB',wh,bg);tmp=im.copy();tmp.thumbnail(wh,Image.Resampling.LANCZOS);out.paste(tmp,((wh[0]-tmp.width)//2,(wh[1]-tmp.height)//2));return out

def build(root,out,partial=False):
    manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    gates={r['source_id']:r for r in json.loads((root/'source_geometry_validation.json').read_text(encoding='utf-8'))['clips']}
    out.mkdir(parents=True,exist_ok=True);(out/'contacts').mkdir(exist_ok=True)
    rows=[];missing=[]
    for c in manifest['clips']:
        sid=c['source_id'];dst=out/'assets'/sid;dst.mkdir(parents=True,exist_ok=True)
        cached=dst/'source_review.json'
        if cached.exists():
            row=json.loads(cached.read_text(encoding='utf-8'))
            same_source=(row.get('scene')==c['scene'] and row.get('primary_instance')==c['actors'][0]['instance_token']
                         and [f['source_filename'] for f in row['frames']]==[f['filename'] for f in c['frames']]
                         and row.get('full_geometry_gate')==gates[sid])
            if row.get('build_revision')==1 and same_source and len(row['frames'])==30 and all((out/f[k]).is_file() for f in row['frames'] for k in ['rgb','geometry','crop']):
                rows.append(row);continue
        paths=[source(root,f['filename']) for f in c['frames']]
        if any(p is None for p in paths):missing.append(sid);continue
        images=[];labeled=[];metrics=[];frames=[]
        valid=True
        for f,p in zip(c['frames'],paths):
            rgb=Image.open(p).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)
            ann=rgb.copy();draw=ImageDraw.Draw(ann)
            actors=f['actors']; primary=actors[0] if actors else None
            if primary is None or primary['projection'] is None:valid=False;continue
            pr=primary['projection'];bb=pr['box_xyxy']
            good=pr['border']>=8 and pr['width']>=72 and pr['height']>=40 and .006<=pr['area_frac']<=.18
            roi=np.asarray(crop(rgb,bb,1.0).convert('L'))
            gray=roi.astype(np.float64)
            lap=gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
            sharp=float(lap.var())
            for ai,a in enumerate(actors):
                if not a['projection']:continue
                b=a['projection']['box_xyxy'];color=(255,204,54) if ai==0 else (64,217,191)
                draw.rectangle(b,outline=color,width=2)
                label=f'{ai} {a["instance_token"][:7]}'
                draw.rectangle((b[0],max(0,b[1]-18),b[0]+120,max(0,b[1]-18)+18),fill=(15,22,31))
                draw.text((b[0]+2,max(0,b[1]-18)),label,fill=color,font=font(14))
            n=f'{f["frame"]:03}'
            rgb.save(dst/f'{n}_rgb.jpg',quality=92)
            ann.save(dst/f'{n}_geometry.jpg',quality=92)
            crop(rgb,bb).save(dst/f'{n}_crop.jpg',quality=95)
            metrics.append({'frame':f['frame'],'laplacian_var':round(sharp,2),'geometry_gate':bool(good),
                            'box_width':pr['width'],'box_height':pr['height'],'border':pr['border'],
                            'area_fraction':pr['area_frac'],'actual_timestamp_us':f['timestamp'],'delta_ms':f['delta_ms']})
            frames.append({'frame':f['frame'],'rgb':f'assets/{sid}/{n}_rgb.jpg','geometry':f'assets/{sid}/{n}_geometry.jpg',
                           'crop':f'assets/{sid}/{n}_crop.jpg','timestamp_us':f['timestamp'],'source_filename':f['filename']})
            images.append(rgb);labeled.append(ann)
        if not valid or len(frames)!=30:
            rows.append({'source_id':sid,'scene':c['scene'],'source_status':'geometry_reject','reason':'missing_projection','frames':frames});continue
        worst=min(metrics,key=lambda r:r['laplacian_var'])['frame']
        ids=list(dict.fromkeys([0,15,29,worst]))
        sheet=Image.new('RGB',(1400,660),(15,21,31));draw=ImageDraw.Draw(sheet)
        for j,fid in enumerate(ids):
            x=(j%2)*700;y=(j//2)*330
            draw.text((x+8,y+5),f'{sid} {c["scene"]} {c["camera"]}  f{fid:02}'+(' WORST-SHARP' if fid==worst else ''),fill='white',font=font(16))
            sheet.paste(labeled[fid].resize((480,270),Image.Resampling.LANCZOS),(x,y+30))
            a=c['frames'][fid]['actors'][0]
            sheet.paste(fit(crop(images[fid],a['projection']['box_xyxy']),(212,270)),(x+488,y+30))
            draw.text((x+8,y+304),f'PRIMARY {a["instance_token"][:12]} | full context + primary crop',fill=(177,193,213),font=font(14))
        sheet.save(out/'contacts'/f'{sid}.jpg',quality=94)
        row={'source_id':sid,'scene':c['scene'],'camera':c['camera'],'location':c['location'],
             'primary_instance':c['actors'][0]['instance_token'],'category':c['actors'][0]['category'],
             'protected_candidates':[a['instance_token'] for a in c['actors']],
             'source_status':'pending_independent_review' if all(r['geometry_gate'] for r in metrics) and gates[sid]['primary_geometry_pass'] else 'geometry_reject',
             'full_geometry_gate':gates[sid],
             'contact':f'contacts/{sid}.jpg','review_frames':ids,'metrics':metrics,'frames':frames,
             'subagent_review':None,'synthetic_quality':'not_created','human_verdict':None,'build_revision':1}
        dump(cached,row)
        rows.append(row)
        if len(rows)%10==0:print('PREVIEW',len(rows),flush=True)
    if not partial:assert not missing,missing
    data={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','stage':'source_only_cpu',
          'source_count':len(rows),'distinct_scenes':len({r['scene'] for r in rows}),
          'sampling_rejected':manifest['sampling_rejected'],'missing_sources':missing,'clips':rows,
          'columns':['rgb','geometry','crop'],'human_scoring_scope':'source_frames_only_not_final_synthetic_acceptance'}
    dump(out/'source_review_manifest.json',data)
    template=Path(__file__).with_name('source_review.html').read_text(encoding='utf-8')
    (out/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8',newline='\n')
    print('SOURCE_REVIEW',len(rows),'MISSING',len(missing),'STATUS',dict(Counter(r['source_status'] for r in rows)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();build(a.root,a.out,a.partial)
