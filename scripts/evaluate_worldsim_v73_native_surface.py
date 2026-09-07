"""将冻结原生/已微调DPT点图统一规范融合，用同一曲面读出正面对比。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from motion_proj.worldsim_v73.native_pyramid import NativeGeometryPyramid
from motion_proj.worldsim_v73.spatial_queries import ActorSpatialQueryDecoder
from train_worldsim_v73_physical_surface import lidar_patches,evaluate_actor_surface


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--physical-run',type=Path,required=True)
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    out=Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-NATIVE-SURFACE-01')/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value): (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save('status.json',{'status':'running'})
    torch.set_num_threads(6)
    started=time.monotonic()
    try:
        cohort=json.loads((args.physical_run/'cohort.json').read_text())
        scene=next(s for s in torch.load(args.native_run/'build_observations.pt',weights_only=False)
                   if s['scene_id']==cohort['scene'])
        rays=torch.load(args.physical_run/'raw_actor_rays.pt',weights_only=True)
        owner=cohort['owner']; size=next(r['size_lwh_m'] for r in rays if r['role']=='build').cuda()
        views=[(i,v) for i,v in enumerate(scene['views']) if owner in v['world_from_actor']]
        pyramid=NativeGeometryPyramid(args.native_run,scene,[i for i,_ in views])
        patcher=ActorSpatialQueryDecoder().cuda()
        scale=json.loads((args.native_run/'metric_scales.json').read_text())[scene['scene_id']]
        results={}
        with torch.no_grad():
            for label in ['finetuned_native','frozen_native']:
                if label=='frozen_native':
                    from safetensors import safe_open
                    with safe_open('/root/autodl-tmp/models/eas_vggt/vggt/model.safetensors',framework='pt') as handle:
                        state={k[len('depth_head.'):]:handle.get_tensor(k) for k in handle.keys() if k.startswith('depth_head.')}
                    pyramid.head.load_state_dict(state)
                points=[]; counts=[]
                for (i,view),(token_inputs,image,patch_start) in zip(views,pyramid.token_inputs):
                    tokens=[None]*24
                    for j,t in zip([4,11,17,23],token_inputs): tokens[j]=t
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        depth,_=pyramid.head(tokens,image,patch_start)
                    depth=depth[0,0,:,:,0].float()*scale
                    height,width=depth.shape
                    y,x=torch.meshgrid(torch.arange(height,device='cuda'),torch.arange(width,device='cuda'),indexing='ij')
                    uv=torch.stack([x,y,torch.ones_like(x)],-1).float().reshape(-1,3)
                    intrinsics=torch.tensor(view['intrinsics'],device='cuda',dtype=torch.float32)
                    camera=(uv@torch.linalg.inv(intrinsics).T)*depth.reshape(-1,1)
                    actor_from_camera=torch.tensor(np.linalg.inv(view['world_from_actor'][owner])@view['world_from_camera'],device='cuda',dtype=torch.float32)
                    actor=camera@actor_from_camera[:3,:3].T+actor_from_camera[:3,3]
                    # 输出点图按只读Actor框归属融合；不是按held-out质量筛点，记录入框覆盖数。
                    keep=torch.isfinite(actor).all(-1)&(actor.abs()<=size/2+.25).all(-1)&(depth.reshape(-1)>.05)
                    counts.append(int(keep.sum()))
                    points.append(actor[keep])
                points=torch.cat(points)
                if len(points):
                    # 5cm体素平均，减少重复视图的密度偏差；不作凸包。
                    _,inverse=torch.unique((points/.05).round().long(),dim=0,return_inverse=True)
                    total=points.new_zeros(int(inverse.max())+1,3)
                    total.index_add_(0,inverse,points)
                    count=points.new_zeros(len(total)); count.index_add_(0,inverse,torch.ones(len(points),device='cuda'))
                    fused=total/count[:,None]
                    surface=lidar_patches(fused,patcher)
                else:
                    fused=points
                    surface={'vertices_actor_m':points,'faces':torch.empty(0,3,device='cuda',dtype=torch.long),
                             'centers_actor_m':points}
                if len(surface['faces']):
                    result=evaluate_actor_surface(surface,rays)
                else:
                    result=None
                torch.save({k:v.cpu() for k,v in surface.items()},out/f'{label}_surface.pt')
                results[label]={'per_view_in_box_pixels':counts,'voxel_points':len(fused),
                                'patches':len(surface['centers_actor_m']),'metrics':result,'empty_surface':not len(fused)}
        save('summary.json',{'status':'done','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'scene':scene['scene_id'],'owner':owner,'native_run':str(args.native_run),'physical_run':str(args.physical_run),
            'scale':scale,'results':results,'wall_s':time.monotonic()-started,
            'boundary':'same PCA patch readout; box-based native point ownership may retain background; no full-surface precision claim'})
        save('status.json',{'status':'done'})
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__': main()
