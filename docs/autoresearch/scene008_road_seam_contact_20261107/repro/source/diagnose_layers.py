"""冻结场景的联合成像读出；不修改车辆/道路参数，不执行训练。"""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
import sys
import time
import numpy as np

PARENT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_r0_v44_20261107_r1')
BAKED=PARENT/'candidates/baked_cellfit_strip1000_candidate_v1'

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def image(path,array):
    from PIL import Image
    array=np.asarray(array)
    if array.ndim==3 and array.shape[0]==3:array=array.transpose(1,2,0)
    Image.fromarray(np.uint8(np.clip(array,0,1)*255+.5)).save(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--raws',type=int,nargs='+',default=[105,113,128,130,137])
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.50)
    sys.path.insert(0,str(BAKED))
    from baked_road_runtime import BakedRoadRuntime
    start=time.time()
    model=BakedRoadRuntime(BAKED/'scene_manifest.json',verify_dependencies=False)
    rt=model.rt
    records=[]
    with torch.no_grad():
        for raw in args.raws:
            folder=args.output/f'{raw:03d}';folder.mkdir()
            data=rt.raw(raw);obs=data['obs']
            np.savez_compressed(folder/'observations.npz',**{k:v for k,v in obs.items() if isinstance(v,np.ndarray)})
            for state in ('keep','delete'):
                for name,enabled in [('v44',False),('r0',True)]:
                    scene=model.base.scene(raw,delete=state=='delete',extension=enabled)
                    rows=dict(scene['rows'])
                    feature=rows['features'].clone()
                    depth=(rows['means3D']@scene['cam'].world_view_transform[:3,:3]+scene['cam'].world_view_transform[3,:3])[:,2]/rt.scale
                    feature[:,22]=depth*(feature[:,18]>0)
                    feature[:,23]=depth*(feature[:,2]>0)
                    rows['features']=feature;scene=dict(scene,rows=rows)
                    out=rt.render(raw,scene)
                    arrays={k:out[k].cpu().numpy() for k in ['rgb','alpha','depth_sum','group_alpha_T','group_rgb_T','env_rgb_T','constant_background_rgb_T']}
                    arrays.update(newroad_alpha_T=out['extra'][16].cpu().numpy(),newroad_rgb_T=out['extra'][17:20].cpu().numpy(),
                        newroad_centre_depth_T_m=out['extra'][20].cpu().numpy(),asset_centre_depth_T_m=out['extra'][21].cpu().numpy())
                    arrays['depth_m']=arrays['depth_sum'][0]/np.maximum(arrays['alpha'][0],1e-8)/rt.scale
                    arrays['oldroad_alpha_T']=arrays['group_alpha_T'][2]-arrays['newroad_alpha_T']
                    arrays['oldroad_rgb_T']=arrays['group_rgb_T'][2]-arrays['newroad_rgb_T']
                    np.savez_compressed(folder/f'{name}_{state}.npz',**arrays)
                    png=folder/f'{name}_{state}.png';image(png,arrays['rgb'])
                    parentname='calibrated' if enabled else 'baseline'
                    old=PARENT/'results/full49_corridor_cellfit'/f'{raw:03d}'/f'{parentname}_{state}.png'
                    assert png.read_bytes()==old.read_bytes(),(raw,state,name,'Physical RGB changed under diagnostic channels')
                    if raw in [105,113,130,137]:
                        with np.load(old.with_name(old.stem+'_float.npz')) as z:
                            assert np.array_equal(arrays['rgb'],z['rgb'])
                            assert np.array_equal(arrays['group_alpha_T'],z['group_alpha_T'])
                    records.append(dict(raw=raw,state=state,variant=name,PNG_parent_exact=True,alpha_conservation=out['group_conservation_maxabs'],RGB_conservation=out['color_reconstruction_maxabs'],asset_rows_hash=out['meta']['native_asset_row_hash']))
                    del out,scene,rows,feature,depth,arrays
            for name in ['asset','newroad']:
                scene=model.scene(raw,delete=True)
                key=2 if name=='asset' else 18
                mask=scene['rows']['features'][:,key]>0
                rows={k:None if v is None else v[mask].contiguous() for k,v in scene['rows'].items()}
                scene=dict(scene,rows=rows,env=scene['env']*0,bg=scene['bg']*0)
                out=rt.render(raw,scene)
                rgb=out['rgb'].cpu().numpy();alpha=out['alpha'][0].cpu().numpy()
                np.savez_compressed(folder/f'{name}_solo.npz',rgb=rgb,alpha=alpha,
                    depth_m=(out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-8)/rt.scale).cpu().numpy())
                image(folder/f'{name}_solo.png',rgb)
                del scene,rows,out
            rt._data.clear();torch.cuda.empty_cache()
            print(json.dumps(dict(raw=raw,seconds=time.time()-start)),flush=True)
    write(args.output/'receipt.json',dict(schema='SEAM_CONTACT_LAYER_DIAGNOSTICS_V1',raws=args.raws,records=records,
        total_forward=len(records)+2*len(args.raws),seconds=time.time()-start,original_parameter_updates=0,
        newroad_parameter_updates=0,shared_sort=True,depth_tags='alphaT-weighted Gaussian centre depths, not surface first intersection',
        dependency_verification='manifest identity and native runtime checks; full78 dependency payload SHA was previously verified in parent export, not recomputed here',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime=model.receipt(),shutdown_performed=False))

if __name__=='__main__':main()
