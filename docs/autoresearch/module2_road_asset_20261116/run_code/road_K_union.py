"""父版/新版道路在真实合成中对保留车辆K的world-row成像贡献union。"""
from pathlib import Path
import argparse,json,sys,os,gc,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
from drivingeditor_dynamic_edit.native_row_contribution import attribute_rows
SOURCE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene065_closedloop_20261116_r1/handoff_coverage_r2/persistent')
DEST=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/handoff')
BASE=DEST.parent


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def regions_for(scene,manifest,key):
    row=next(r for r in manifest.data['frames'] if r['key']==key)
    camera=scene.camera_provider(key)
    path=Path(row['instance_mask']);path=path if path.is_absolute() else manifest.base/path
    with Image.open(path) as im:
        labels=np.asarray(im.resize((camera.image_width,camera.image_height),Image.Resampling.NEAREST)).copy()
    target=labels==10;protected=(labels>0)&~target
    return dict(target=target,protected=protected,outside=~target&~protected),labels


def collect_one(source,keys):
    scene,m,receipt=load_persistent_handoff(source/'scene.json',source/'proof.json')
    with np.load(m.road) as z:
        active=z['active'].astype(bool);rgb=z['reference_srgb'];transform=z['model_from_world'].copy()
    world=np.flatnonzero(active);massmax=np.zeros(len(active),np.float32);rgbmax=massmax.copy();records=[]
    for key in keys:
        regions,_=regions_for(scene,m,key)
        a=attribute_rows(scene,key,{'K':regions['protected']},delete_actor_ids=(10,))
        mass=a.alpha_mass[a.static_rows+a.dynamic_rows:,0]
        if len(mass)!=len(world):raise RuntimeError('road row order/count differs from complete asset active mapping')
        np.maximum(massmax[world],mass,out=mass) if False else None
        massmax[world]=np.maximum(massmax[world],mass)
        rgbmax[world]=np.maximum(rgbmax[world],mass*np.abs(rgb[world]).max(1))
        record=dict(raw=key,K_pixels=a.region_pixels['K'],road_K_alphaT_mass=float(mass.sum()),contributing_world_rows=int((mass>1e-7).sum()),
            attribution_conservation_relative_error=a.conservation_relative_error)
        records.append(record);print(str(source),json.dumps(record),flush=True)
    scene.assert_persistent_unchanged()
    del scene,m,a;gc.collect();torch.cuda.empty_cache()
    return massmax,rgbmax,active,transform,receipt,records


def main(args):
    start=time.monotonic();torch.set_num_threads(2);keys=[int(v) for v in args.frames.split(',')]
    old=collect_one(args.old,keys);new=collect_one(args.new,keys)
    if old[0].shape!=new[0].shape or not np.array_equal(old[3],new[3]):raise RuntimeError('world grid topology/similarity differs across versions')
    guard=(old[0]>1e-7)|(new[0]>1e-7)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(args.output,world_row_ids=np.arange(len(guard)),old_mass_max=old[0],new_mass_max=new[0],
        old_rgb_mass_max=old[1],new_rgb_mass_max=new[1],old_active=old[2],new_active=new[2],guard=guard)
    receipt=dict(schema='road_protected_world_row_union.v1',frames=keys,validation_scope='pilot5_only',
        old_source=str(args.old),new_source=str(args.new),old_loader=old[4],new_loader=new[4],
        old_frames=old[5],new_frames=new[5],guard_rows=int(guard.sum()),new_active_conflict_rows=int((guard&~old[2]&new[2]).sum()),
        threshold=1e-7,criterion='union of true shared-sort road alphaT contribution to accepted retained actor pixels, target actor10 deleted',
        required_action='persistently retain full old row appearance/geometry/active/depth guard; deactivate newly-active conflicting rows',
        outside_scope_safety_not_claimed=True,seconds=time.monotonic()-start,GPU='keep_on')
    write(args.output.with_suffix('.json'),receipt);print('COMPLETE',json.dumps({k:v for k,v in receipt.items() if k not in ['old_loader','new_loader','old_frames','new_frames']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,default=SOURCE);p.add_argument('--new',type=Path,default=BASE/'asset_r1/persistent')
    p.add_argument('--frames',default='86,110,134,179,198');p.add_argument('--output',type=Path,default=DEST/'road_K_union_r1.npz');main(p.parse_args())
