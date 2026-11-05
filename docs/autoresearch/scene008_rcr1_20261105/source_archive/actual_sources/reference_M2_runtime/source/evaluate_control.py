"""Fixed full49 evaluation of one unified asset, versus native and SH0 control."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from sh_asset import SHAsset
from scene_runtime import Runtime,raster,tagged
from metrics import evaluate_frame,evaluate_pair,pool
import scene_runtime as _runtime_module
assert Path(_runtime_module.__file__).resolve().parent==Path(__file__).resolve().parent,'Wrong runtime source binding'

RUN=Path(__file__).resolve().parent.parent;PARENT=RUN.parent/'scene008_a6_complete_asset_20261105'
HELD=(89,95,101,105,113,121,125,130,137)
def cpu(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def img(x):
    a=cpu(x)
    if a.ndim==3 and a.shape[0] in (1,3):a=a.transpose(1,2,0)
    if a.ndim==3 and a.shape[2]==1:a=np.repeat(a,3,2)
    return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))
def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def solo(rt,raw,asset):
    d=rt.raw(raw);rows=tagged(asset.rows(d['pose'],d['cam'],rt.pipe,16,precomputed=False),'asset')
    out=raster(d['cam'],rt.pipe,3,rows,rt.n.bg.new_zeros(3));out['component_alpha_T']=out['extra'][:4];return out
def strip(values,labels,crop=None,height=250):
    tiles=[]
    for value,label in zip(values,labels):
        p=img(value)
        if crop is not None:p=p.crop(crop)
        p=p.resize((round(p.width*height/p.height),height),Image.Resampling.NEAREST)
        q=Image.new('RGB',(p.width,height+30),(241,246,249));q.paste(p,(0,30));ImageDraw.Draw(q).text((7,8),label,fill=(22,40,52));tiles.append(q)
    out=Image.new('RGB',(sum(t.width for t in tiles),height+30));x=0
    for q in tiles:out.paste(q,(x,0));x+=q.width
    return out
def leaves(value,prefix=''):
    if isinstance(value,dict):
        if {'count','sum','mean'}.issubset(value):yield prefix,value
        else:
            for k,v in value.items():yield from leaves(v,prefix+'.'+k if prefix else k)
def aggregate(rows):
    keys={key for row in rows for key,_ in leaves(row)}
    flat=[dict(leaves(row)) for row in rows]
    return {key:pool(row[key] for row in flat if key in row) for key in sorted(keys)}

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['m1','m2','m3']);p.add_argument('--asset',required=True);p.add_argument('--compare');args=p.parse_args()
    start=time.time();dest=RUN/args.arm/'evaluation';dest.mkdir(exist_ok=False);frames=dest/'frames';frames.mkdir()
    compare=Path(args.compare) if args.compare else PARENT/'appearance/complete_a6_fitted.npz'
    rt=Runtime();asset=SHAsset.from_geometry(Path(args.asset)).cuda();parent=SHAsset.from_geometry(compare).cuda()
    records=[];conservation=0.;alpha_delta=0.;depth_delta=0.
    for raw in range(89,138):
        d=rt.raw(raw);native=rt.native_reference(raw);base=solo(rt,raw,parent);new=solo(rt,raw,asset)
        bk=rt.render_scene(raw,baseline=True);bd=rt.render_scene(raw,baseline=True,delete=True)
        pk=rt.render_scene(raw,parent);pd=rt.render_scene(raw,parent,delete=True)
        nk=rt.render_scene(raw,asset);nd=rt.render_scene(raw,asset,delete=True)
        record=dict(raw=raw,split='HELD' if raw in HELD else 'BUILD',
           solo_native=evaluate_frame(d,native,native,rt.scale,'solo'),
           solo_SH0=evaluate_frame(d,base,native,rt.scale,'solo'),
           solo_new=evaluate_frame(d,new,native,rt.scale,'solo'),
           scene=evaluate_pair(d,nk,nd,bk,bd,rt.scale),
           incremental_scene=evaluate_pair(d,nk,nd,pk,pd,rt.scale),
           asset_colour_diagnostics=asset.last_diagnostics)
        records.append(record)
        for pkg in (native,base,new,bk,bd,pk,pd,nk,nd):
            conservation=max(conservation,float((pkg['component_alpha_T'].sum(0)-pkg['alpha'][0]).abs().max()))
        alpha_delta=max(alpha_delta,float((new['alpha']-base['alpha']).abs().max()))
        depth_delta=max(depth_delta,float((new['depth_sum']-base['depth_sum']).abs().max()))
        support=(new['alpha'][0]>.01)|(base['alpha'][0]>.01)|(native['alpha'][0]>.01)|d['a6_visible'].bool()
        y,x=torch.where(support);crop=(max(0,int(x.min())-10),max(0,int(y.min())-10),min(480,int(x.max())+11),min(320,int(y.max())+11))
        strip([d['rgb']*d['a6_visible'][None],native['rgb'],base['rgb'],new['rgb']],
          ['Observed a6','Native a6','Complete / SH0',args.arm+' complete asset'],crop).save(frames/f'{raw:03d}_objects.jpg',quality=94)
        strip([pk['rgb'],nk['rgb'],pd['rgb'],nd['rgb']],
          ['SH0 / a37 present',args.arm+' / a37 present','SH0 / delete a37',args.arm+' / delete a37'],None,height=320).save(frames/f'{raw:03d}_street.jpg',quality=92)
        np.savez_compressed(frames/f'{raw:03d}_replay.npz',solo=cpu(new['rgb']).astype(np.float16),alpha=cpu(new['alpha']).astype(np.float16),
          keep=cpu(nk['rgb']).astype(np.float16),delete=cpu(nd['rgb']).astype(np.float16),
          keep_groups=cpu(nk['component_alpha_T']).astype(np.float16),delete_groups=cpu(nd['component_alpha_T']).astype(np.float16))
        print(json.dumps(dict(arm=args.arm,raw=raw,complete=len(records))),flush=True)
        del d,native,base,new,bk,bd,pk,pd,nk,nd;rt._data.clear();torch.cuda.empty_cache()
    assert rt.modelunchanged() and asset.assert_geometry_unchanged() and conservation<2e-5
    if args.arm=='m1':assert asset.geometry_hash==parent.geometry_hash and alpha_delta==0 and depth_delta==0
    write(dest/'metrics.json',dict(rows=records,runtime=rt.meta,geometry_hash=asset.geometry_hash,native_a6_rows=0,
      group_conservation_maxabs=conservation,alpha_change_vs_SH0_maxabs=alpha_delta,depth_sum_change_vs_SH0_maxabs=depth_delta,
      seconds=time.time()-start,all49_complete=True,geometry_unchanged=True,incremental_comparison_asset=str(compare)))
    summary={'all49':aggregate(records),'BUILD':aggregate([r for r in records if r['split']=='BUILD']),
      'HELD':aggregate([r for r in records if r['split']=='HELD'])}
    write(dest/'summary.json',summary)
    print(json.dumps(dict(arm=args.arm,status='PASS_ENGINEERING',seconds=time.time()-start,
      solo_RGB=summary['all49']['solo_new.a6.rgb_error']['mean'],held_RGB=summary['HELD']['solo_new.a6.rgb_error']['mean'])),flush=True)

if __name__=='__main__':main()
