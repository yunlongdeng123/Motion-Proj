"""为通过receiver中的次要完整车辆生成独立来源审核；不自动准入。"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from geometry_factory import read,dump
from build_source_review import crop,fit,source

def main(root,out):
    out.mkdir(parents=True,exist_ok=True);src={r['source_id']:r for r in read(root/'source_manifest.json')['clips']};g={r['source_id']:r for r in read(root/'source_geometry_validation.json')['clips']};rows=[]
    for q in read(root/'subagent_source_reviews.json')['clips']:
        if q['receiver_status']!='pass':continue
        sid=q['source_id'];c=src[sid];gate={x['instance_token']:x['pass'] for x in g[sid]['actors']}
        for ai,a in enumerate(c['actors'][1:],1):
            tok=a['instance_token']
            if not gate.get(tok):continue
            jid=f'{sid}__a{ai}';ims=[];scores=[]
            for f in c['frames']:
                p=source(root,f['filename']);assert p is not None
                im=Image.open(p).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS);aa=next(x for x in f['actors'] if x['instance_token']==tok);b=aa['projection']['box_xyxy'];gray=np.asarray(crop(im,b,1).convert('L'),dtype=float);lap=gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1];scores.append(float(lap.var()));ims.append((im,b))
            fids=list(dict.fromkeys([0,15,29,int(np.argmin(scores))]));sheet=Image.new('RGB',(1400,700),(14,20,29));dr=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
            for j,i in enumerate(fids):
                im,b=ims[i];ann=im.copy();ImageDraw.Draw(ann).rectangle(b,outline=(255,218,44),width=3);x=(j%2)*700;y=(j//2)*350;dr.text((x+8,y+8),f'{jid} {c["scene"]} f{i} | actor {tok[:12]}',font=font,fill='white');sheet.paste(ann.resize((480,270)),(x,y+36));sheet.paste(fit(crop(im,b),(214,270)),(x+485,y+36))
            sheet.save(out/f'{jid}.jpg',quality=95);rows.append({'job_id':jid,'source_id':sid,'actor_index':ai,'instance_token':tok,'review_frames':fids,'sheet':str(out/f'{jid}.jpg'),'geometry_gate_pass':True,'human_verdict':None})
    dump(out/'sheet_index.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','clips':rows});print('SECONDARY_REVIEW',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
