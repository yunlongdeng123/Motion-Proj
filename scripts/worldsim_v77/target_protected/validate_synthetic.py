"""逐实际产物校验监督/条件/保护域与可解码性，非视觉质量评分。"""
import argparse,subprocess
from pathlib import Path
import cv2,imageio_ffmpeg
import numpy as np
from PIL import Image
from geometry_factory import Geometry,read,dump
from data_contract import assert_pair_pixels,masked_condition_from_x
from build_pairs import runlen

def arr(path):
    with Image.open(path) as im:im.load();return np.asarray(im).copy()
def validate(root,out):
    cv2.setNumThreads(1);geo=Geometry(root);rows=[]
    for case in read(out/'synthetic_manifest.json')['clips']:
        cid=case['case_id'];sid=case['source_id'];geo.prepare(sid);folder=root/'synthetic'/cid;errors=[];ratios={t:[] for t in case['occlusion_fraction']};frame_count=0
        for i,f in enumerate(geo.sources[sid]['frames']):
            n=f'{i:03}';y=arr(folder/'Y'/f'{n}.png');x=arr(folder/'X'/f'{n}.png');a=arr(folder/'alpha'/f'{n}.png')>=128;influence=arr(folder/'influence'/f'{n}.png')>0;hole=arr(folder/'model_hole'/f'{n}.png')>0
            with Image.open(root/'rgb'/f['filename']) as im:actual=np.asarray(im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
            assert np.array_equal(y,actual),'GT changed'
            assert_pair_pixels(y,x,influence,hole)
            cond=masked_condition_from_x(x[None],hole[None]);altered=y.copy();altered[hole]=np.uint8(255)-altered[hole]
            assert np.array_equal(cond,masked_condition_from_x(altered[None],hole[None])),'Y hidden pixel condition leakage'
            for t in ratios:
                m=arr(folder/'protected'/f'{n}_{t}.png')>0;ratios[t].append(float((a&m).sum()/max(1,m.sum())))
                if (hole&m).sum()/max(1,m.sum())>.85:errors.append({'frame':i,'reason':'protected_in_hole_gt85pct','instance':t})
            for ob in geo.obstacles[sid][i]:
                if ob['instance_token'] in ratios or ob['_projection'] is None:continue
                bb=ob['_projection']['box'];x0,y0=np.floor(bb[:2]).astype(int);x1,y1=np.ceil(bb[2:]).astype(int);x0=max(0,x0);x1=min(1024,x1);y0=max(0,y0);y1=min(576,y1)
                if x1>x0 and y1>y0 and hole[y0:y1,x0:x1].sum()>max(12,hole.sum()*.01):errors.append({'frame':i,'reason':'unreviewed_GT_envelope_hole_overlap','instance':ob['instance_token']})
            for k in ['gt','input','labels','condition']:
                p=out/f'assets/{cid}/{n}_{k}.jpg';assert arr(p).shape==(576,1024,3)
            frame_count+=1
        active=[t for t,r in ratios.items() if max(r)>.01]
        if case['type']=='background' and active:errors.append({'reason':'actual_category_mismatch'})
        if case['type']=='single_actor' and not(len(active)==1 and runlen((np.array(ratios[active[0]])>=.3)&(np.array(ratios[active[0]])<=.8))>=10):errors.append({'reason':'actual_single_occlusion_gate'})
        if case['type']=='dense_actors' and not(len(active)>=2 and runlen(np.all((np.array([ratios[t] for t in active])>=.2)&(np.array([ratios[t] for t in active])<=.7),axis=0))>=10):errors.append({'reason':'actual_dense_occlusion_gate'})
        vids=[]
        for k,p in case['videos'].items():
            reader=imageio_ffmpeg.read_frames(str(out/p),pix_fmt='rgb24');meta=next(reader);count=sum(1 for _ in reader);assert count==frame_count and meta['size']==(1024,576);vids.append({'kind':k,'frames':count})
        row={'case_id':cid,'pass':not errors,'errors':errors,'frames':frame_count,'videos':vids,'GT_exact_redecode':True,'pixel_contract':True,'hidden_Y_perturbation_probe':True}
        rows.append(row);print('VALID',cid,row['pass'],len(errors),flush=True)
    dump(out/'delivery_validation.json',{'case_count':len(rows),'video_count':len(rows)*4,'preview_frame_count':sum(r['frames'] for r in rows)*4,'cases':rows})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();validate(a.root,a.out)
