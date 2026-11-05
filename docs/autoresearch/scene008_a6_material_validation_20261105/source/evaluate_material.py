"""Sealed pseudo material evaluation and real all49 frozen-scene validation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,time,gzip,argparse
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz
from scipy.ndimage import binary_erosion
from PIL import Image,ImageDraw
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');OLD=CAM/'scene008_a6_asset_controls_20261105';RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(OLD/'source'))
from sh_asset import SHAsset
from scene_runtime import Runtime,raster,tagged
from metrics import evaluate_pair,pool
from evaluate_control import aggregate
from fit_material import CASES,METHODS

def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cpu(v):return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
def metric(v,mask):
    a=np.asarray(v)[mask].astype(np.float64);return dict(count=len(a),sum=float(a.sum()),mean=float(a.mean()) if len(a) else None,max=float(a.max()) if len(a) else None)
def solo(rt,raw,asset):
    d=rt.raw(raw);rows=tagged(asset.rows(d['pose'],d['cam'],rt.pipe,16),'asset')
    out=raster(d['cam'],rt.pipe,3,rows,rt.n.bg.new_zeros(3));out['component_alpha_T']=out['extra'][:4];return out
def strip(values,names,crop,path,height=250):
    pieces=[]
    for value,name in zip(values,names):
        a=cpu(value)
        if a.shape[0]==3:a=a.transpose(1,2,0)
        p=Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5)).crop(crop)
        p=p.resize((round(p.width*height/p.height),height),Image.Resampling.NEAREST)
        q=Image.new('RGB',(p.width,height+33),(240,245,248));q.paste(p,(0,33));ImageDraw.Draw(q).text((6,9),name,fill=(19,41,55));pieces.append(q)
    image=Image.new('RGB',(sum(x.width for x in pieces),height+33));x=0
    for p in pieces:image.paste(p,(x,0));x+=p.width
    image.save(path,quality=94)

def pseudo():
    dest=RUN/'pseudo_evaluation';dest.mkdir(exist_ok=False);start=time.time()
    A=load_npz(RUN/'data/all_design.npz');data=dict(np.load(RUN/'data/sealed_targets.npz'));raw=data['raw'];target=data['target']
    build=np.isin(raw,data['BUILD']);held=~build;reports=[]
    rt=Runtime()
    for case in CASES:
        spec=dict(np.load(RUN/'cases'/case/'split.npz'));H=spec['hidden'];unknown=spec['unknown']
        fraction=np.asarray(A[:,H].sum(1)).ravel()/np.maximum(np.asarray(A.sum(1)).ravel(),1e-8)
        hit=A[:,H].getnnz(axis=1)>0;dominant=hit&(fraction>=.9);mixed=hit&~dominant
        train=build&~hit;assert int(train.sum())==int(spec['train_rows'].sum())
        assets={name:SHAsset.from_geometry(RUN/'formal'/case/'extensions'/f'{name}_asset.npz') for name in METHODS}
        baseline=A@assets['median'].rgb.numpy();rows=[]
        for name,asset in assets.items():
            prediction=A@asset.rgb.numpy();error=np.abs(prediction-target).mean(1)
            assert np.array_equal(prediction[train],baseline[train])
            provenance=dict(np.load(RUN/'formal'/case/'extensions'/f'{name}_colours.npz'))
            row=dict(method=name,
                BUILD_spatial_withheld=metric(error,build&dominant),HELD_spatial_withheld=metric(error,held&dominant),
                BUILD_mixed=metric(error,build&mixed),HELD_mixed=metric(error,held&mixed),
                retained_train=metric(error,train),HELD_full_observed=metric(error,held),
                BUILD_full_observed=metric(error,build),all_observed=metric(error,np.ones(len(raw),bool)),
                hidden_primitives=int(H.sum()),unknown_columns=int(unknown.sum()),
                propagated=int(provenance['propagated'].sum()),unreachable=int(provenance['unreachable'].sum()),
                unknown_inside_H=int((unknown&H).sum()),unknown_outside_H=int((unknown&~H).sum()),
                propagated_inside_H=int((provenance['propagated']&H).sum()),
                retained_training_prediction_bitwise_same=True)
            rows.append(row)
        for r in (105,113,130):
            d=rt.raw(r);rendered=[]
            for asset in assets.values():rendered.append(solo(rt,r,asset.cuda()))
            domain=(cpu(rendered[0]['alpha'][0])>.01)|cpu(d['a6_visible']).astype(bool)
            yy,xx=np.where(domain);crop=(max(0,int(xx.min())-9),max(0,int(yy.min())-9),min(480,int(xx.max())+10),min(320,int(yy.max())+10))
            marked=cpu(d['rgb']*d['a6_visible'][None]).copy();mask=np.zeros((320,480),bool);select=(raw==r)&dominant
            mask.ravel()[data['pixels'][select]]=True;edge=mask&~binary_erosion(mask)
            marked[:,edge]=np.array([1.,.5,0.])[:,None]
            strip([marked]+[p['rgb'] for p in rendered],['Observed / hidden ROI']+list(METHODS),crop,dest/f'{case}_{r:03d}.jpg')
            for asset in assets.values():asset.cpu()
            rt._data.clear();torch.cuda.empty_cache()
        reports.append(dict(case=case,BUILD_eval_rows=int((build&dominant).sum()),HELD_eval_rows=int((held&dominant).sum()),methods=rows))
    assert rt.modelunchanged()
    bymethod={name:dict(HELD_macro_mean=float(np.mean([next(x for x in r['methods'] if x['method']==name)['HELD_spatial_withheld']['mean'] for r in reports])),
        BUILD_macro_mean=float(np.mean([next(x for x in r['methods'] if x['method']==name)['BUILD_spatial_withheld']['mean'] for r in reports]))) for name in METHODS}
    write(dest/'summary.json',dict(status='COMPLETE_SEALED_MATERIAL_EVALUATION',cases=reports,case_equal_macro=bymethod,
        seconds=time.time()-start,scope='Material-only fixed-geometry pseudo-occlusion; not unseen geometry/generalization',
        alpha_not_used_to_shrink_denominator=True,unreachable_pixels_retained=True))

@torch.no_grad()
def real():
    dest=RUN/'real_evaluation';dest.mkdir(exist_ok=False);(dest/'frames').mkdir();start=time.time();rt=Runtime()
    assets={name:SHAsset.from_geometry(RUN/'real'/f'{name}_asset.npz').cuda() for name in METHODS}
    records={name:[] for name in METHODS};alpha_delta=0.;depth_delta=0.;groups_delta=0.
    for raw in range(89,138):
        d=rt.raw(raw);base=solo(rt,raw,assets['median']);bk=rt.render_scene(raw,assets['median']);bd=rt.render_scene(raw,assets['median'],delete=True)
        sols=[base];streets={};frame_records=[]
        for name in METHODS:
            if name=='median':new,nk,nd=base,bk,bd
            else:new=solo(rt,raw,assets[name]);nk=rt.render_scene(raw,assets[name]);nd=rt.render_scene(raw,assets[name],delete=True)
            from metrics import evaluate_frame
            row=dict(raw=raw,solo=evaluate_frame(d,new,base,rt.scale,'solo'),scene=evaluate_pair(d,nk,nd,bk,bd,rt.scale))
            records[name].append(row)
            for current,reference in ((new,base),(nk,bk),(nd,bd)):
                assert torch.equal(current['alpha'],reference['alpha']) and torch.equal(current['depth_sum'],reference['depth_sum'])
                assert torch.equal(current['component_alpha_T'],reference['component_alpha_T'])
            if name!='median':sols.append(new)
            streets[name]=(nk['rgb'],nd['rgb'])
        domain=(base['alpha'][0]>.01)|d['a6_visible'].bool();y,x=torch.where(domain)
        crop=(max(0,int(x.min())-9),max(0,int(y.min())-9),min(480,int(x.max())+10),min(320,int(y.max())+10))
        strip([d['rgb']*d['a6_visible'][None]]+[s['rgb'] for s in sols],['Real observed RGB']+list(METHODS),crop,dest/'frames'/f'{raw:03d}_objects.jpg')
        # Street branch is separate from black-object completion diagnosis.
        if raw in (105,113,130):
            for name in METHODS[1:]:
                strip([streets['median'][0],streets[name][0],streets['median'][1],streets[name][1]],
                    ['M2 keep',name+' keep','M2 delete a37',name+' delete a37'],(0,0,480,320),dest/'frames'/f'{raw:03d}_{name}_street.jpg',height=320)
        print(json.dumps(dict(stage='real_eval',raw=raw)),flush=True)
        del d,base,bk,bd,new,nk,nd,sols,streets;rt._data.clear();torch.cuda.empty_cache()
    assert rt.modelunchanged() and all(a.assert_geometry_unchanged() for a in assets.values())
    with gzip.open(dest/'metrics.json.gz','wt') as f:json.dump(records,f)
    summaries={name:dict(all49=aggregate(rows),HELD=aggregate([r for r in rows if r['raw'] in (89,95,101,105,113,121,125,130,137)])) for name,rows in records.items()}
    write(dest/'summary.json',dict(status='PASS_FROZEN_GEOMETRY_SCENE_EVALUATION',methods=summaries,frames=49,
        alpha_bitwise_same=True,depth_bitwise_same=True,group_contributions_bitwise_same=True,
        native_road_environment_unchanged=True,seconds=time.time()-start,
        scene_adoption=False,hidden_RGB_accuracy_measured=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pseudo','real']);args=p.parse_args()
    if args.stage=='pseudo':
        with torch.no_grad():pseudo()
    else:real()
