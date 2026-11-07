"""分别记录道路覆盖、路面外观、车轮贡献和已观测保护；不合并质量标签。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1')
PARENT=ROOT.parent/'scene008_road_r0_v44_20261107_r1'
MANIFEST=PARENT/'candidates/baked_cellfit_strip1000_candidate_v1/scene_manifest.json'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def stats(array,mask):
    v=np.asarray(array,dtype=np.float64)[mask]
    return dict(count=len(v),sum=float(v.sum()),mean=float(v.mean()) if len(v) else None,
        min=float(v.min()) if len(v) else None,max=float(v.max()) if len(v) else None)

def image(path,array):
    from PIL import Image
    if array.ndim==3 and array.shape[0]==3:array=array.transpose(1,2,0)
    if array.ndim==2:array=np.repeat(array[:,:,None],3,axis=2)
    Image.fromarray(np.uint8(np.clip(array,0,1)*255+.5)).save(path)

def arrays(out,scale):
    d={k:out[k].detach().cpu().numpy() for k in ['rgb','alpha','group_alpha_T','group_rgb_T','env_rgb_T']}
    d.update(newroad_alpha_T=out['extra'][16].cpu().numpy(),newroad_rgb_T=out['extra'][17:20].cpu().numpy())
    d['depth_m']=(out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-8)/scale).cpu().numpy()
    d['oldroad_alpha_T']=d['group_alpha_T'][2]-d['newroad_alpha_T']
    d['oldroad_rgb_T']=d['group_rgb_T'][2]-d['newroad_rgb_T']
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--factor',type=int,choices=[1,2,4],default=2)
    ap.add_argument('--raws',type=int,nargs='+',default=[105,113,128,130,137])
    ap.add_argument('--relative-mip',action='store_true')
    ap.add_argument('--material',type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.5)
    sys.path.insert(0,str(ROOT/'engineer'))
    from refined_road_runtime import RefinedRoadRuntime
    kwargs={}
    if args.relative_mip:
        from relative_mip import RelativeMip
        kwargs=dict(opacity_fn=RelativeMip(),opacity_policy_id='PARENT_RELATIVE_MIP_NEWROAD_ONLY')
    model=RefinedRoadRuntime(MANIFEST,factor=args.factor,verify_dependencies=False,**kwargs)
    rt=model.rt
    if args.material:
        with np.load(args.material,allow_pickle=False) as z:
            override=z['colours_srgb'].copy()
        assert override.shape==(10098,3) and np.isfinite(override).all()
        assert ((override>=0)&(override<=1)).all()
        override=torch.as_tensor(override,dtype=torch.float32,device='cuda')
    else:override=None
    snap=args.output/'source_snapshot';snap.mkdir()
    source_files=[Path(__file__),ROOT/'engineer/refined_road_runtime.py',ROOT/'engineer/world_surface_sampling.py',ROOT/'contract/round_scope.json']
    if args.relative_mip:source_files.append(ROOT/'engineer/relative_mip.py')
    source_id={}
    for path in source_files:
        shutil.copy2(path,snap/path.name);source_id[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    start=time.time();calls_before=rt.render_calls;records=[]
    with torch.no_grad():
        for raw in args.raws:
            folder=args.output/f'{raw:03d}';folder.mkdir()
            obs=rt.raw(raw)['obs'];gt=np.asarray(obs['rgb'],dtype=np.float32).transpose(2,0,1)
            if gt.max()>1:gt/=255
            image(folder/'observed.png',gt)
            masks=dict(a6=obs['visible'][0].astype(bool),a37=obs['visible'][1].astype(bool),
                measured_a6=obs['depth_valid'][0].astype(bool),road=obs['road'].astype(bool),
                measured_road=obs['road'].astype(bool)&(obs['scene_depth_m']>0),
                protected=obs['protect'].astype(bool)&~obs['visible'].any(0),
                hidden_box_a6=obs['box_candidate'][0].astype(bool)&~obs['visible_reviewed'][0].astype(bool))
            blue=np.zeros((320,480),bool);blue[188:234,165:208]=True
            red=np.zeros_like(blue);red[190:218,232:284]=True
            fixed=np.zeros_like(blue);fixed[204:222,230:270]=True
            solo_rows=model.extension_rows(raw)
            if override is not None:
                solo_rows=dict(solo_rows)
                solo_rows['colors_precomp']=override.repeat_interleave(args.factor**2,dim=0)
                feat=solo_rows['features'].clone();feat[:,12:15]=solo_rows['colors_precomp'];feat[:,19:22]=solo_rows['colors_precomp'];solo_rows['features']=feat
            data=rt.raw(raw)
            solo=rt.render(raw,dict(rows=solo_rows,cam=data['cam'],env=data['env']*0,bg=rt.n.bg*0,meta=dict(raw=raw,solo_newroad=True)))
            solo_alpha=solo['alpha'][0].cpu().numpy();image(folder/'newroad_solo_alpha.png',solo_alpha)
            solo_record=dict(fixed720=stats(solo_alpha,fixed),coverage_lt098_fixed720=int((solo_alpha[fixed]<.98).sum())) if raw==137 else {}
            del solo
            for state in ['keep','delete']:
                outputs={};state_items={}
                for name in ['parent','candidate']:
                    scene=model.parent.scene(raw,delete=state=='delete') if name=='parent' else model.scene(raw,delete=state=='delete')
                    if name=='candidate' and override is not None:
                        rows=dict(scene['rows']);sel=rows['features'][:,18]>0
                        cols=rows['colors_precomp'].clone();cols[sel]=override.repeat_interleave(args.factor**2,dim=0)
                        feat=rows['features'].clone();feat[sel,12:15]=cols[sel];feat[sel,19:22]=cols[sel]
                        rows.update(colors_precomp=cols,features=feat);scene=dict(scene,rows=rows)
                    out=rt.render(raw,scene);a=arrays(out,rt.scale);err=abs(a['rgb']-gt).mean(0)
                    a6d=abs(a['depth_m']-obs['depth_m'][0]);roadd=abs(a['depth_m']-obs['scene_depth_m'])
                    image(folder/f'{name}_{state}.png',a['rgb']);image(folder/f'{name}_{state}_newroadW.png',a['newroad_alpha_T'])
                    if name=='parent':
                        ref=PARENT/'results/full49_corridor_cellfit'/f'{raw:03d}'/f'calibrated_{state}.png'
                        assert (folder/f'{name}_{state}.png').read_bytes()==ref.read_bytes(),('parent parity',raw,state)
                    item=dict(a6_RGB=stats(err,masks['a6']),a37_RGB=stats(err,masks['a37']) if state=='keep' else None,
                        a6_depth_m=stats(a6d,masks['measured_a6']),observed_road_RGB=stats(err,masks['road']),
                        observed_road_depth_m=stats(roadd,masks['measured_road']),protected_RGB=stats(err,masks['protected']),
                        a6_alpha=stats(a['group_alpha_T'][0],masks['a6']),conservation_alpha=out['group_conservation_maxabs'],
                        conservation_RGB=out['color_reconstruction_maxabs'],asset_rows_hash=out['meta'].get('native_asset_row_hash'),
                        newroad_rows=int((scene['rows']['features'][:,18]>0).sum()))
                    domains={'hidden_a6_box':masks['hidden_box_a6']}
                    if raw==128:domains.update(blue_frame_rect=blue,blue_original_observed_road=blue&masks['road'],red_frame_rect=red)
                    if raw==137:domains['fixed720']=fixed
                    item['domains']={label:dict(count=int(mask.sum()),RGB=stats(err,mask),newroadW=stats(a['newroad_alpha_T'],mask),
                        oldroadW=stats(a['oldroad_alpha_T'],mask),staticW=stats(a['group_alpha_T'][3],mask),
                        a6W=stats(a['group_alpha_T'][0],mask),environmentT=stats(1-a['alpha'][0],mask)) for label,mask in domains.items()}
                    if name=='candidate':
                        parent=outputs['parent'];item['versus_parent']={label:dict(RGB_abs_change=stats(abs(a['rgb']-parent['rgb']).mean(0),mask),
                            a6W_loss=stats(np.maximum(parent['group_alpha_T'][0]-a['group_alpha_T'][0],0),mask),
                            a6W_gain=stats(np.maximum(a['group_alpha_T'][0]-parent['group_alpha_T'][0],0),mask)) for label,mask in domains.items()}
                        assert item['asset_rows_hash']==state_items['parent']['asset_rows_hash']
                    if args.factor==1 and override is None and not args.relative_mip and name=='candidate':
                        for key in ['rgb','alpha','group_alpha_T','depth_m']:
                            assert np.array_equal(a[key],outputs['parent'][key]),('n1 parity',raw,state,key)
                    if raw in [105,113,128,130,137]:np.savez_compressed(folder/f'{name}_{state}.npz',**a)
                    outputs[name]=a;state_items[name]=item
                    del out,scene
                records.append(dict(raw=raw,state=state,variants=state_items,solo_newroad=solo_record))
            write(folder/'metrics.json',records[-2:]);write(args.output/'progress.json',dict(completed=sorted(set(r['raw'] for r in records))))
            rt._data.clear();torch.cuda.empty_cache();print(json.dumps(dict(raw=raw,seconds=time.time()-start)),flush=True)
    pooled=[]
    for state in ['keep','delete']:
        for variant in ['parent','candidate']:
            for metric in ['a6_RGB','a37_RGB','a6_depth_m','observed_road_RGB','observed_road_depth_m','protected_RGB']:
                vals=[r['variants'][variant][metric] for r in records if r['state']==state and r['variants'][variant][metric] is not None]
                count=sum(v['count'] for v in vals)
                if count:pooled.append(dict(state=state,variant=variant,metric=metric,count=count,sum=sum(v['sum'] for v in vals),mean=sum(v['sum'] for v in vals)/count))
    write(args.output/'receipt.json',dict(schema='SEPARATE_SEAM_CONTACT_CONTROL_V1',factor=args.factor,relative_mip=args.relative_mip,
        material_override=str(args.material) if args.material else None,raws=args.raws,frames=records,pooled=pooled,
        seconds=time.time()-start,forwards=rt.render_calls-calls_before,executed_sources=source_id,runtime=model.receipt(),
        optimizer_updates=0,original_parameter_updates=0,whole_scene_adoption_claim=False,acceptance_channels=['road coverage','road seam','wheel ordering','observed protection'],
        hidden_GT=False,shutdown_performed=False))

if __name__=='__main__':main()
