"""只做输入分布盘点；数值差异不自动升级为效果因果。"""
from pathlib import Path
import sys
from collections import Counter
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(Path(__file__).parent.parent/'iteration7'))
from asset_factory import O,T,read,dump
from audit import protected_mask

def stats(h,b=None):
    yy,xx=np.where(h);box=(xx.max()-xx.min()+1)*(yy.max()-yy.min()+1) if len(xx) else 0
    return {'hole_canvas_fraction':float(h.mean()),'hole_bbox_fill':float(h.sum()/box) if box else 0,'hole_border_contact':bool(h[0].any() or h[-1].any() or h[:,0].any() or h[:,-1].any()),'protected_hidden_canvas_fraction':float((b&h).mean()) if b is not None else None}

def summarize(rows):
    out={'frames':len(rows)}
    for key in ['hole_canvas_fraction','hole_bbox_fill','protected_hidden_canvas_fraction']:
        v=[r[key] for r in rows if r[key] is not None]
        out[key]={'mean':float(np.mean(v)),'q10':float(np.quantile(v,.1)),'q50':float(np.median(v)),'q90':float(np.quantile(v,.9)),'max':float(max(v))} if v else None
    out['hole_border_contact_fraction']=float(np.mean([r['hole_border_contact'] for r in rows])) if rows else None
    return out

def training(root):
    catalog=read(root/'dataset_catalog.json');frames={};train=[c for c in catalog['cases'] if c['split']=='train']
    for c in train:
        folder=Path(c['folder'])
        for i in range(c['frame_count']):
            h=np.asarray(Image.open(folder/'model_hole'/f'{i:03}.png'))>0
            b=protected_mask(folder,i,h.shape,c['type']) if c['type']!='background' else np.zeros(h.shape,bool)
            frames[c['dataset_id'],i]=stats(h,b)
    weighted=[];steps=read_steps(root/'training/steps.jsonl')
    for r in steps:weighted += [frames[r['last_case'],r['last_window']+i] for i in range(10)]
    return {'cases':len(train),'scene_count':len({c['receiver_scene'] for c in train}),'types':dict(Counter(c['type'] for c in train)),'unique_frame_stats':summarize(list(frames.values())),'step_weighted':summarize(weighted),'step_type_counts':dict(Counter(next(c['type'] for c in train if c['dataset_id']==r['last_case']) for r in steps))}

def read_steps(p):return [__import__('json').loads(v) for v in p.read_text().splitlines()]

def main():
    rows=[];realcases=[]
    for c in read(O/'real_evaluation_plan.json')['cases']:
        masks=sorted((Path(c['folder'])/'model_mask').glob('*.png'));ss=[stats(np.asarray(Image.open(masks[i]))>0) for i in c['frames']];rows+=ss;realcases.append({'eval_id':c['eval_id'],'scene':c['scene'],'stats':summarize(ss),'input_difficulty_proxy':c['input_difficulty_proxy']})
    val=[]
    for c in read(O/'dataset_catalog.json')['cases']:
        if c['split']=='validation':
            f=Path(c['folder']);val += [stats(np.asarray(Image.open(f/'model_hole'/f'{i:03}.png'))>0) for i in range(10)]
    result={'r7_training':training(T/'r7/encoder_fixed_lowres'),'r8_training':training(O),'synthetic_validation':summarize(val),'real_DEVELOPMENT':summarize(rows),'real_cases':realcases,'boundary':'input pixel shares/bbox shapes, not latent gradient contribution or causal explanation; real actor-free GT unavailable; no re-tuning by model outputs'}
    dump(O/'input_distribution.json',result);print({k:v.get('step_weighted',v) for k,v in result.items() if isinstance(v,dict)},flush=True)
if __name__=='__main__':main()
