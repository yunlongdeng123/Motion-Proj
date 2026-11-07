"""Saved Full5 KEEP->DELETE source37 effect on fixed observed identities; CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image,ImageDraw

LABEL_TO_ACTOR={1:5,2:26,3:18,4:6,5:37}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snap(p):
    s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)}


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def stat(a):
    return None if a.size==0 else float(a.astype(np.float64).mean())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=Path,required=True);ap.add_argument('--obs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    paths=[a.case/'receipt.json',a.case/'frame_headers.json',Path(__file__)]
    for raw in [94,105,128,137]:paths += [a.obs/f'{raw:03d}.npz']+[a.case/f'frames/{raw:03d}/{s}_float.npz' for s in ['keep','delete']]
    before=[snap(p) for p in paths];receipt=json.loads((a.case/'receipt.json').read_text())
    result={'schema':'FULL5_FIXED_IDENTITY_KD_AUDIT_V1','status':'FAIL_NATIVE_SOURCE37_QUERY_RETAINED_A6_PRESERVATION',
       'actual_new_gpu_forwards':0,'actual_new_backwards':0,'actual_new_optimizer_updates':0,
       'source_filter_scope':'Native source ancestry37, not a semantic37 classifier. Original ancestry is not changed.',
       'source_renderer_receipt_sha256':sha(a.case/'receipt.json'),'parameters_owners_unchanged_receipt':receipt['parameters_and_owners_unchanged'],
       'rows':[],'figures':[],'quality_accepted':False,'human_verdict':None,
       'limits':['All domains use original observed labels/visible/depth arrays; no alpha-conditioned scoring.',
          'Source37 alpha on observed6 may indicate misassigned or misplaced source-descendant support; it is not alone proof of a specific training-code bug.',
          'The KEEP source37 filter being removed exactly does not establish semantic delete success.',
          'a6 measured-depth errors are joint conditional camera-z in metres, not unsaved actor-specific depth.',
          'Observed full37 RGB is not clean-delete GT; its DELETE error is diagnostic only.']}
    for raw in [94,105,128,137]:
        obs=load(a.obs/f'{raw:03d}.npz');k=load(a.case/f'frames/{raw:03d}/keep_float.npz');d=load(a.case/f'frames/{raw:03d}/delete_float.npz')
        ids=k['group_actor_ids'].tolist();assert ids==d['group_actor_ids'].tolist()==[-1,5,6,18,26,37]
        gt=obs['rgb'].transpose(2,0,1);kg,dg=k['group_alpha_T'],d['group_alpha_T'];i37=ids.index(37)
        assert float(dg[i37].max())==0
        masks={f'full_actor{actor}':obs['actor_labels']==label for label,actor in LABEL_TO_ACTOR.items()}
        masks.update(a6_refined_visible=obs['visible'][0],a37_refined_visible=obs['visible'][1],a6_measured_depth=obs['depth_valid'][0])
        frame={'raw':raw,'source37_DELETE_max_alpha_T':float(dg[i37].max()),'domains':{},'top20_a6_RGB_changes':[]}
        for name,m in masks.items():
            n=int(m.sum());row={'pixels':n}
            if n:
                row.update(KEEP_RGB_observed_MAE=stat(abs(k['rgb'][:,m]-gt[:,m])),DELETE_RGB_observed_MAE=stat(abs(d['rgb'][:,m]-gt[:,m])),
                    RGB_KEEP_DELETE_mean_abs=stat(abs(k['rgb'][:,m]-d['rgb'][:,m])),RGB_KEEP_DELETE_max_abs=float(abs(k['rgb'][:,m]-d['rgb'][:,m]).max()),
                    KEEP_group_alpha_T_mean={str(actor):stat(kg[i,m]) for i,actor in enumerate(ids)},DELETE_group_alpha_T_mean={str(actor):stat(dg[i,m]) for i,actor in enumerate(ids)},
                    KEEP_argmax_source_actor_fraction={str(actor):float((kg[:,m].argmax(0)==i).mean()) for i,actor in enumerate(ids)},
                    DELETE_argmax_source_actor_fraction={str(actor):float((dg[:,m].argmax(0)==i).mean()) for i,actor in enumerate(ids)})
                if name=='a6_measured_depth':
                    row.update(KEEP_joint_depth_MAE_m=stat(abs(k['depth_m'][m]-obs['depth_m'][0,m])),DELETE_joint_depth_MAE_m=stat(abs(d['depth_m'][m]-obs['depth_m'][0,m])))
            else:row.update(KEEP_RGB_observed_MAE=None,DELETE_RGB_observed_MAE=None,RGB_KEEP_DELETE_mean_abs=None,KEEP_group_alpha_T_mean=None,DELETE_group_alpha_T_mean=None)
            frame['domains'][name]=row
        mask=obs['visible'][0];err=abs(k['rgb']-d['rgb']).mean(0);flat=np.flatnonzero(mask)
        flat=flat[np.argsort(-err.ravel()[flat],kind='stable')[:20]]
        for index in flat:
            y,x=np.unravel_index(int(index),mask.shape)
            frame['top20_a6_RGB_changes'].append({'x':int(x),'y':int(y),'actor_label':int(obs['actor_labels'][y,x]),'a6_refined_visible':bool(mask[y,x]),
                'RGB_mean_abs_change':float(err[y,x]),'observed_rgb':gt[:,y,x].tolist(),'KEEP_rgb':k['rgb'][:,y,x].tolist(),'DELETE_rgb':d['rgb'][:,y,x].tolist(),
                'KEEP_source37_alpha_T':float(kg[i37,y,x]),'KEEP_source6_alpha_T':float(kg[ids.index(6),y,x]),'DELETE_source6_alpha_T':float(dg[ids.index(6),y,x])})
        # Display crop is original full6 silhouette bbox+6; no rendered mask/quality crop.
        y,x=np.where(obs['actor_labels']==4)
        if len(x):
            box=(max(0,int(x.min())-6),max(0,int(y.min())-6),min(480,int(x.max())+7),min(320,int(y.max())+7))
            scale=5;w=(box[2]-box[0])*scale;h=(box[3]-box[1])*scale
            board=Image.new('RGB',(max(5*w,1000),h+80),'white');draw=ImageDraw.Draw(board)
            draw.text((6,5),f'raw{raw} observed full6 bbox+6 {box}; nearest x5; source ownership is not semantic truth',fill='black')
            arrays=[('Observed',gt),('KEEP',k['rgb']),('DELETE source37',d['rgb']),('KEEP source37 alpha_T',np.repeat(kg[i37][None],3,0)),('KEEP source6 alpha_T',np.repeat(kg[ids.index(6)][None],3,0))]
            for i,(label,v) in enumerate(arrays):
                im=Image.fromarray(np.rint(np.clip(v.transpose(1,2,0),0,1)*255).astype(np.uint8)).crop(box)
                im=im.resize((w,h),Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST)
                board.paste(im,(i*w,56));draw.text((i*w+4,30),label,fill='black')
            p=a.output/f'raw{raw:03d}_a6_source37_damage.png';board.save(p)
            result['figures'].append({'raw':raw,'path':str(p),'sha256':sha(p),'size':list(board.size),'crop_xyxy':list(box),'crop_definition':'Observed actor_labels4 bbox+6; display only'})
        result['rows'].append(frame)
    after=[snap(p) for p in paths];assert before==after
    result['source_integrity']={'status':'PASS_ALL_INPUT_SIZE_MTIME_SHA_UNCHANGED','files':before}
    p=a.output.parent/(a.output.name+'.json');p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(p),'sha256':sha(p),'status':result['status']}))


if __name__=='__main__':main()
