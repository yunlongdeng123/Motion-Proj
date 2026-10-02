"""保留真实纵横比、逐actor审图，并证明审核图与方法输入同源。"""
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageOps
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r34';CID='Q046'
def read(p):return json.loads(p.read_text())

def main(run_root=O,case_id=CID):
    O=run_root;CID=case_id
    case=next(c for c in read(T/'r28/prepared.json')['cases'] if c['case_id']==CID)
    q=next(c for c in read(T/'r32/instance_quality.json')['cases'] if c['case_id']==CID)['quality']
    tokens=sorted(q['protected_instances']);strips={t:[] for t in tokens};rows=[]
    for i in range(30):
        y=np.asarray(Image.open(Path(case['source_Y_quality_only'])/f'{i:05}.png').convert('RGB'))
        x=np.asarray(Image.open(O/'observed'/CID/'rgb'/f'{i:05}.png').convert('RGB'))
        h=np.asarray(Image.open(O/'observed'/CID/'H'/f'{i:05}.png'))>0
        assert np.array_equal(x[~h],y[~h]) and np.all(x[h]==127),(i,'Y/X來源不同')
        state=np.load(O/'state'/CID/f'{i:05}.npz');overview=y.copy();crops={}
        for slot,t in enumerate(tokens,1):
            a=next(v for v in case['retained_instances'] if v['instance_token']==t)
            bb=a['boxes'][i];pred=state['actor_slot']==slot
            obs=np.asarray(Image.open(O/'observed_masks'/(CID+'_'+t[:8])/f'{i:05}.png'))>0
            ys,xs=np.where(pred|obs);pts=[]
            if len(xs):pts.extend([[int(xs.min()),int(ys.min())],[int(xs.max())+1,int(ys.max())+1]])
            if bb is not None:pts.extend([bb[:2],bb[2:]])
            if pts:
                ar=np.array(pts);x0,y0=np.maximum(np.floor(ar.min(0)-12).astype(int),[0,0]);x1,y1=np.minimum(np.ceil(ar.max(0)+12).astype(int),[1024,576])
            else:x0,y0,x1,y1=0,0,0,0
            crops[t]=[int(x0),int(y0),int(x1),int(y1)]
            tile=Image.new('RGB',(600,202),(18,24,32));draw=ImageDraw.Draw(tile)
            role='main B' if t in q['reveal_instances'] else 'keep'
            if x1>x0 and y1>y0:
                overlay=x.copy();cv2.drawContours(overlay,cv2.findContours(obs.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(40,250,125),1)
                rgb=np.full_like(x,48);rgb[pred]=state['F'][pred]
                for j,(label,im) in enumerate([('Y QA only',y),('legal X / own SAM',overlay),('own projected F',rgb)]):
                    draw.text((j*200+4,3),f'f{i} {role} slot{slot}',fill='white');draw.text((j*200+4,17),label,fill='white')
                    crop=ImageOps.contain(Image.fromarray(im[y0:y1,x0:x1]),(200,160))
                    tile.paste(crop,(j*200+(200-crop.width)//2,36+(160-crop.height)//2))
                cv2.rectangle(overview,(x0,y0),(x1,y1),(40,255,125) if role=='main B' else (70,160,255),2)
                cv2.putText(overview,f'{role} slot{slot}',(x0,max(15,y0-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,255,125),1,cv2.LINE_AA)
            else:draw.text((10,15),f'f{i} slot{slot} outside view / no observed projection',fill='white')
            strips[t].append(tile)
        if i in [0,15,29]:Image.fromarray(overview).save(O/f'{CID}_overview_{i:02}.jpg',quality=96)
        rows.append({'frame':i,'outside_H_X_equals_Y':True,'inside_H_RGB_127':True,'per_actor_crop_xyxy':crops})
    files=[]
    for slot,t in enumerate(tokens,1):
        for start in [0,15]:
            sheet=Image.new('RGB',(3000,606),(18,24,32))
            for j,strip in enumerate(strips[t][start:start+15]):sheet.paste(strip,((j%5)*600,(j//5)*202))
            name=f'{CID}_slot{slot}_{start:02}_{start+14:02}.jpg';sheet.save(O/name,quality=96);files.append(name)
    path=O/('review_provenance_check.json' if O.name=='r34' else f'{CID}_review_provenance_check.json');path.write_text(json.dumps({'case_id':CID,'scene':case['scene'],
        'source_Y':case['source_Y_quality_only'],'masked_X':str(O/'observed'/CID/'rgb'),
        'all_30_frames_same_source':True,'original_review_bug':'r34 multi-actor wide crop forcibly resized to square, distorting geometry',
        'fix':'per-actor crop with aspect-preserving contain and full-scene crop locators',
        'inference_condition_changed':False,'metrics_changed':False,'files':files,'frames':rows},ensure_ascii=False,indent=2)+'\n')
    print('REVIEW_PROVENANCE',len(rows),'source-exact',len(files),'identity sheets')

if __name__=='__main__':main()
