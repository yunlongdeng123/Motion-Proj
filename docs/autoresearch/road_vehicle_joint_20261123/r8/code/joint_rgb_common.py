"""CPU preflight and immutable inputs for the fixed-geometry joint colour fit."""
from pathlib import Path
import json,hashlib
import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def configuration(path):
    cfg=read(path);base=read(cfg['baseline_contract']);base['frames']=cfg['frames']
    assert cfg['train_frames']==[86,110] and cfg['evaluation_only_frames']==[134,179,198]
    assert cfg['formal_steps']==400 and cfg['RGB_prior']==1e-4
    assert set(cfg['frames'])==set(cfg['train_frames']+cfg['evaluation_only_frames'])
    assert cfg['geometry_sigma_opacity_normal_tracks_fixed'] is True
    return cfg,base


def source_gate(cfg,strict=True):
    """No CUDA/torch import is reachable before both SHA-bound gates pass."""
    source=cfg['source'];issues=[];records=[]
    path=Path(source['field'])
    if not path.is_file():issues.append('shared source field missing')
    elif sha(path)!=source['sha256']:issues.append('shared source SHA mismatch')
    for role in ['numeric','visual']:
        p=Path(source[role+'_gate'])
        if not p.is_file():issues.append(role+' source gate missing');continue
        g=read(p)
        if source.get(role+'_gate_sha256') and sha(p)!=source[role+'_gate_sha256']:
            issues.append(role+' gate SHA mismatch')
        if not(g.get('complete') is True and g.get('decision')=='PASS'):
            issues.append(role+' source gate not PASS')
        if g.get('final_atlas_sha256')!=source['sha256']:issues.append(role+' gate refers to another field')
        if g.get('failed_checks'):issues.append(role+' gate contains failed checks')
        records.append(dict(role=role,path=str(p),sha256=sha(p),decision=g.get('decision')))
    p=Path(source['root_review'])
    if not p.is_file():issues.append('root source approval missing')
    else:
        g=read(p)
        if sha(p)!=source['root_review_sha256']:issues.append('root source approval SHA mismatch')
        if not(g.get('complete') is True and g.get('decision')=='PASS' and g.get('root_concur') is True and g.get('student_fit_authorized') is True):
            issues.append('root did not authorize fitting this source')
        if g.get('final_atlas_sha256')!=source['sha256']:issues.append('root approval refers to another field')
        for role,key in [('numeric','numeric_acceptance'),('visual','independent_visual_review')]:
            if g.get(key,{}).get('sha256')!=source[role+'_gate_sha256']:issues.append('root approval '+role+' binding differs')
        records.append(dict(role='root approval',path=str(p),sha256=sha(p),decision=g.get('decision')))
    result=dict(pass_before_CUDA=not issues,issues=issues,field=source['field'],field_sha256=source['sha256'],gates=records)
    if strict and issues:raise RuntimeError('Joint colour GPU action refused before native loading: '+'; '.join(issues))
    return result


def actor_ranges(rt,raw):
    cursor=len(rt.context[raw][0]['means3D']);result=[]
    for index,(spec,field) in enumerate(rt.actors):
        result.append((index,spec,cursor,cursor+len(field.xyz)));cursor+=len(field.xyz)
    return result


def srgb_to_linear_numpy(x):
    x=np.asarray(x);return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def linear_to_srgb_numpy(x):
    x=np.maximum(np.asarray(x),0);return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)


def all4_and_sample(image,uv,valid=None):
    """UV texel centres; no boundary clamp counts as valid support."""
    image=np.asarray(image);uv=np.asarray(uv);h,w=image.shape[:2]
    finite=np.isfinite(uv).all(-1);safe=np.where(finite[...,None],uv,0)
    x=np.floor(safe[...,0]).astype(np.int64);y=np.floor(safe[...,1]).astype(np.int64)
    inside=finite&(x>=0)&(y>=0)&(x+1<w)&(y+1<h)
    x=np.clip(x,0,w-2);y=np.clip(y,0,h-2);fx=safe[...,0]-x;fy=safe[...,1]-y
    weight=[(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy]
    index=[(y,x),(y,x+1),(y+1,x),(y+1,x+1)]
    if valid is not None:
        for iy,ix in index:inside&=valid[iy,ix]
    out=sum(image[iy,ix]*a[...,None] if image.ndim==3 else image[iy,ix]*a for (iy,ix),a in zip(index,weight))
    return out.astype(np.float32),inside
