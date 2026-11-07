"""同一联合排序中评价融合目标写回，分别保存接缝与保护证据。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np

BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD=BASE/'scene008_road_seam_contact_20261107_r1'
CANON=OLD/'engineer/canonical_n2_mip_candidate_v2'
RUN=BASE/'scene008_road_fusion_bake_20261107_r1'
KEYS=(105,113,128,130,137)


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def stat(a,m):
    v=np.asarray(a,dtype=np.float64)[m]
    return dict(count=int(len(v)),sum=float(v.sum()),mean=float(v.mean()) if len(v) else None)


def image(p,a):
    from PIL import Image
    if a.ndim==3 and a.shape[0]==3:a=a.transpose(1,2,0)
    Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5)).save(p)


def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--material',action='append',default=[])
    ap.add_argument('--raws',nargs='+',type=int,default=list(range(89,138)))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.45)
    sys.path.insert(0,str(CANON))
    from canonical_road_runtime import CanonicalRoadRuntime
    model=CanonicalRoadRuntime(CANON/'scene_manifest.json',verify_dependencies=False)
    rt=model.rt;materials={'baseline':None};sources={};records=[]
    for arg in args.material:
        name,path=arg.split('=',1);path=Path(path)
        assert name not in materials and name.replace('_','').isalnum()
        with np.load(path,allow_pickle=False) as z:col=z['colours_srgb'].copy()
        assert col.shape==(10098,3) and np.isfinite(col).all() and col.min()>=0 and col.max()<=1
        materials[name]=torch.as_tensor(col.repeat(4,axis=0),device='cuda',dtype=torch.float32)
        sources[name]=dict(path=str(path),sha256=sha(path))
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    import shutil
    for p in (Path(__file__),RUN/'contract/round_scope.json'):
        shutil.copy2(p,snapshot/p.name)
    start=time.time();before=rt.render_calls
    with torch.no_grad():
        for raw in args.raws:
            folder=args.output/f'{raw:03d}';folder.mkdir()
            obs=rt.raw(raw)['obs'];gt=np.asarray(obs['rgb'],dtype=np.float32).transpose(2,0,1)
            if gt.max()>1:gt/=255
            image(folder/'observed.png',gt)
            masks=dict(a6=obs['visible'][0].astype(bool),a37=obs['visible'][1].astype(bool),
                road=obs['road'].astype(bool),protected=obs['protect'].astype(bool)&~obs['visible'].any(0),
                depth_a6=obs['depth_valid'][0].astype(bool),depth_road=obs['road'].astype(bool)&(obs['scene_depth_m']>0))
            for state in ('keep','delete'):
                # Build once; inference only changes this material table, not visibility.
                scene=model.scene(raw,delete=state=='delete');baseline=None;reference_png=None
                for name,material in materials.items():
                    rows=scene['rows']
                    if material is not None:
                        rows=dict(rows);select=rows['features'][:,18]>0
                        assert int(select.sum())==40392
                        cols=rows['colors_precomp'].clone();cols[select]=material
                        feat=rows['features'].clone();feat[select,12:15]=material;feat[select,19:22]=material
                        rows.update(colors_precomp=cols,features=feat)
                    out=rt.render(raw,dict(scene,rows=rows))
                    a={k:out[k].cpu().numpy() for k in ('rgb','alpha','group_alpha_T','group_rgb_T','env_rgb_T')}
                    a['depth_m']=(out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-8)/rt.scale).cpu().numpy()
                    a['newroad_alpha_T']=out['extra'][16].cpu().numpy()
                    a['newroad_rgb_T']=out['extra'][17:20].cpu().numpy()
                    a['oldroad_alpha_T']=a['group_alpha_T'][2]-a['newroad_alpha_T']
                    a['oldroad_rgb_T']=a['group_rgb_T'][2]-a['newroad_rgb_T']
                    err=abs(a['rgb']-gt).mean(0)
                    metrics={k:stat(err,m) for k,m in masks.items() if not k.startswith('depth_')}
                    if state=='delete':metrics['a37']=None
                    metrics['depth_a6']=stat(abs(a['depth_m']-obs['depth_m'][0]),masks['depth_a6'])
                    metrics['depth_road']=stat(abs(a['depth_m']-obs['scene_depth_m']),masks['depth_road'])
                    geom=('alpha','depth_m','group_alpha_T','newroad_alpha_T','oldroad_alpha_T')
                    exact={k:np.array_equal(a[k],baseline[k]) for k in geom} if baseline else {}
                    asset_rgb_exact=np.array_equal(a['group_rgb_T'][0],baseline['group_rgb_T'][0]) if baseline else True
                    a37_rgb_exact=np.array_equal(a['group_rgb_T'][1],baseline['group_rgb_T'][1]) if baseline else True
                    if name=='baseline':baseline=a
                    else:
                        assert all(exact.values()),(raw,state,name,exact)
                        assert asset_rgb_exact and a37_rgb_exact,(raw,state,name,'frozen vehicle RGB changed')
                    png=folder/f'{name}_{state}.png';image(png,a['rgb'])
                    parity=None
                    if name=='baseline':
                        reference_png=OLD/f'results/n2_mip_full49/{raw:03d}/candidate_{state}.png'
                        parity=png.read_bytes()==reference_png.read_bytes()
                        assert parity,(raw,state,'baseline PNG')
                    np.savez_compressed(folder/f'{name}_{state}.npz',**(a if raw in KEYS else {'rgb':a['rgb']}))
                    if raw in KEYS:
                        image(folder/f'{name}_{state}_roadW.png',np.repeat(a['newroad_alpha_T'][...,None],3,2))
                    rec=dict(raw=raw,state=state,variant=name,metrics=metrics,
                        geometry_fields_bitwise_equal_baseline=exact,geometry_readout_hashes={k:ahash(a[k]) for k in geom},
                        frozen_asset_premultiplied_RGB_exact=asset_rgb_exact,frozen_a37_premultiplied_RGB_exact=a37_rgb_exact,
                        baseline_parent_PNG_exact=parity,asset_rows_hash=out['meta']['native_asset_row_hash'],
                        alpha_conservation_maxabs=out['group_conservation_maxabs'],RGB_conservation_maxabs=out['color_reconstruction_maxabs'])
                    records.append(rec)
                    del out
                del scene,baseline
            rt._data.clear();torch.cuda.empty_cache()
            write(args.output/'progress.json',dict(last_raw=raw,records=len(records),seconds=time.time()-start))
            print(json.dumps(dict(raw=raw,seconds=time.time()-start)),flush=True)
    pooled=[]
    for state in ('keep','delete'):
        for name in materials:
            for metric in ('a6','a37','road','protected','depth_a6','depth_road'):
                vals=[r['metrics'][metric] for r in records if r['state']==state and r['variant']==name and r['metrics'][metric] is not None]
                n=sum(v['count'] for v in vals)
                pooled.append(dict(state=state,variant=name,metric=metric,count=n,sum=sum(v['sum'] for v in vals),
                    mean=sum(v['sum'] for v in vals)/n if n else None))
    write(args.output/'receipt.json',dict(schema='FUSION_TARGET_PERSISTENT_3D_WRITEBACK_V1',raws=args.raws,
        variants=list(materials),materials=sources,records=records,pooled=pooled,forwards=rt.render_calls-before,
        seconds=time.time()-start,backwards=0,geometry_and_alpha_updates=0,vehicle_updates=0,parameter_updates=0,
        floating_point_storage='Full decomposition only 5 keys; all49 lossless float RGB for fixed seam evaluation.',
        runtime=model.receipt(),human_verdict=None,whole_scene_adopted=False,AutoDL='KEEP_ON'))


if __name__=='__main__':main()
