"""同一显式曲面上的原生DPT联合适配、coverage/free训练和新时刻真实束读出。"""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.actor_rays import load_actor_rays
from motion_proj.worldsim_v73.native_pyramid import NativeGeometryPyramid
from motion_proj.worldsim_v73.spatial_queries import ActorSpatialQueryDecoder
from motion_proj.worldsim_v73.surface_readout import (first_triangle_intersection,closest_surface_points,
                                                     direct_free_space_loss,first_return_metrics)


def lidar_patches(points,decoder,count=1536):
    # 同样固定尺度三角片读出；LiDAR邻域PCA法向，不取凸包、不学习透明度。
    ids=torch.linspace(0,len(points)-1,min(count,len(points)),device=points.device).long()
    centers=points[ids]
    neighbors=cKDTree(points.cpu().numpy()).query(centers.cpu().numpy(),k=min(20,len(points)))[1]
    neighborhood=points[torch.tensor(neighbors,device=points.device)]
    local=neighborhood-neighborhood.mean(1,keepdim=True)
    _,vectors=torch.linalg.eigh(local.transpose(1,2)@local)
    normal=vectors[:,:,0]
    reference=torch.zeros_like(normal); reference[:,2]=1
    take=normal[:,2].abs()>.9
    reference[take,2]=0; reference[take,1]=1
    tangent=torch.nn.functional.normalize(torch.cross(normal,reference,dim=-1),dim=-1)
    bitangent=torch.cross(normal,tangent,dim=-1)
    grid=decoder.patch_grid*decoder.patch_spacing_m
    vertices=(centers[:,None]+tangent[:,None]*grid[None,:,0:1]+bitangent[:,None]*grid[None,:,1:2]).reshape(-1,3)
    faces=(decoder.patch_faces[None]+torch.arange(len(centers),device=points.device)[:,None,None]*9).reshape(-1,3)
    return {'vertices_actor_m':vertices,'faces':faces,'centers_actor_m':centers}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--scene',default='scene-0100')
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--free-weight',type=float,default=.5)
    parser.add_argument('--mode',choices=['joint','lidar_only','pointwise'],default='joint')
    parser.add_argument('--owner')
    args=parser.parse_args()
    task='WS-V73-M2-PHYSICAL-SURFACE-01'
    out=Path('/root/autodl-tmp/runs/worldsim_v73')/task/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    started=time.monotonic()
    torch.set_num_threads(6)
    torch.manual_seed(7303); np.random.seed(7303)
    save('manifest.json',{'task_id':task,'run_id':args.run_id,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'seed':7303,
        'claim':'single fit Actor geometry/free mechanism; not cross-log method superiority',
        'failure_ledger_refs':['V73-F01','V73-F02','V73-F03','V73-F04','V73-F05']})
    save('status.json',{'status':'running','phase':'raw_rays'})
    try:
        scenes=torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False)
        scene=next(s for s in scenes if s['scene_id']==args.scene)
        counts={}
        for view in scene['views']:
            for owner,actor in zip(view['owners'],view['actor_mask']):
                if actor: counts[owner]=counts.get(owner,0)+1
        owner=args.owner or max(counts,key=counts.get)
        rays=load_actor_rays('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval',args.scene,owner,
                             {v['sample_id'] for v in scene['views']})
        torch.save(rays,out/'raw_actor_rays.pt')
        save('cohort.json',{'scene':args.scene,'owner':owner,
            'frames':[{k:v for k,v in r.items() if not isinstance(v,torch.Tensor)} for r in rays],
            'scan_time_boundary':'single scan timestamp; no per-point time available',
            'free_domain':'all original first-return beams intersecting known box +0.5m; behind first hit unknown'})
        build_rays=[r for r in rays if r['role']=='build']
        build=torch.unique(torch.cat([r['points_actor_m'][r['positive_actor']] for r in build_rays]),dim=0).cuda()
        if not len(build): raise ValueError('Actor无build表面支持')
        size=build_rays[0]['size_lwh_m'].cuda()
        views=[(i,v) for i,v in enumerate(scene['views']) if owner in v['world_from_actor']]
        matrices=torch.tensor(np.stack([np.linalg.inv(v['world_from_camera'])@v['world_from_actor'][owner]
                                      for _,v in views]),device='cuda',dtype=torch.float32)
        intrinsics=torch.tensor(np.stack([v['intrinsics'] for _,v in views]),device='cuda',dtype=torch.float32)
        channel_ids={'CAM_FRONT':0,'CAM_FRONT_RIGHT':1,'CAM_BACK_RIGHT':2,'CAM_BACK':3,'CAM_BACK_LEFT':4,'CAM_FRONT_LEFT':5}
        camera_ids=torch.tensor([channel_ids[v['camera_id']] for _,v in views],device='cuda')
        times=torch.tensor([(v['camera_time_us']-views[0][1]['camera_time_us'])/1e6 for _,v in views],device='cuda')
        pyramid=NativeGeometryPyramid(args.native_run,scene,[i for i,_ in views])
        decoder=ActorSpatialQueryDecoder().cuda()
        if args.mode!='joint': raise NotImplementedError('同信息控制候选将单独接入；本轮仅joint')
        parameters=[*pyramid.parameters(),*decoder.parameters()]
        optimizer=torch.optim.AdamW(parameters,lr=1e-5)
        build_origins=torch.cat([r['origins_actor_m'] for r in build_rays]).cuda()
        build_directions=torch.cat([r['directions_actor'] for r in build_rays]).cuda()
        build_ranges=torch.cat([r['observed_first_range_m'] for r in build_rays]).cuda()
        def predict():
            features=pyramid()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return decoder(build,size,features,matrices,intrinsics,views[0][1]['image'].shape[-2:],camera_ids,times)

        @torch.no_grad()
        def evaluate(prediction):
            vertices=prediction['vertices_actor_m'].float(); faces=prediction['faces']
            rows=[]
            for r in rays:
                origins=r['origins_actor_m'].cuda(); directions=r['directions_actor'].cuda()
                ranges=r['observed_first_range_m'].cuda(); positive=r['positive_actor'].cuda()
                depth,_=first_triangle_intersection(vertices,faces,origins,directions)
                points=r['points_actor_m'][r['positive_actor']].cuda()
                nearest=closest_surface_points(vertices,faces,points)
                distance=(nearest-points).norm(dim=-1)
                rows.append({'sample_index':r['sample_index'],'role':r['role'],
                    'owned_ray':first_return_metrics(depth[positive],ranges[positive]),
                    'all_near_box_rays':len(ranges),
                    'free_intrusion_rate':(torch.isfinite(depth)&(depth<ranges-.2)).float().mean().item(),
                    'mean_free_intrusion_m':direct_free_space_loss(depth,ranges).item(),
                    'positive_points':len(points),'positive_surface_mean_m':distance.mean().item() if len(points) else None,
                    'positive_surface_recall_02':(distance<=.2).float().mean().item() if len(points) else None})
            return rows

        baseline=lidar_patches(build,decoder)
        save('lidar_patch_baseline.json',evaluate(baseline))
        torch.save({k:v.cpu() for k,v in baseline.items()},out/'lidar_patch_surface.pt')
        del baseline
        with torch.no_grad(): initial=predict()
        save('initial.json',evaluate(initial)); del initial
        before=pyramid.head.projects[0].weight.detach().clone()
        rows=[]
        for step in range(args.steps):
            tick=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            prediction=predict()
            vertices=prediction['vertices_actor_m'].float(); faces=prediction['faces']
            chosen=build[torch.randperm(len(build),device='cuda')[:1024]]
            nearest=closest_surface_points(vertices,faces,chosen)
            coverage=(nearest-chosen).norm(dim=-1).mean()
            ids=torch.randperm(len(build_ranges),device='cuda')[:512]
            depth,_=first_triangle_intersection(vertices,faces,build_origins[ids],build_directions[ids])
            free=direct_free_space_loss(depth,build_ranges[ids])
            envelope=(vertices.abs()-size/2-.25).clamp_min(0).square().mean()
            loss=coverage+args.free_weight*free+.05*envelope
            loss.backward()
            native_grad=pyramid.head.projects[0].weight.grad.norm().item()
            normal_grad=decoder.normal.weight.grad.norm().item()
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            optimizer.step()
            row={'step':step+1,'loss':loss.item(),'coverage_m':coverage.item(),'free_intrusion_m':free.item(),
                 'native_project_gradient':native_grad,'normal_gradient':normal_grad,
                 'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'step_s':time.monotonic()-tick}
            rows.append(row)
            with (out/'train.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
            save('status.json',{'status':'running','phase':'physical_train',**row})
            print(json.dumps(row),flush=True)
            if (step+1)%20==0 or step+1==args.steps:
                torch.save({'depth_head':pyramid.head.state_dict(),'query_decoder':decoder.state_dict(),
                            'optimizer':optimizer.state_dict(),'step':step+1},out/'latest.pt')
            del prediction,vertices,faces,loss,coverage,free,envelope,nearest,depth
        with torch.no_grad(): prediction=predict()
        final=evaluate(prediction)
        save('final.json',final)
        torch.save({k:v.cpu() for k,v in prediction.items() if isinstance(v,torch.Tensor)},out/'final_surface.pt')
        result={'status':'done','scene':args.scene,'owner':owner,'views':len(views),'build_points':len(build),
            'steps':args.steps,'free_weight':args.free_weight,'mode':args.mode,'final':final,
            'native_project_max_change':(pyramid.head.projects[0].weight.detach()-before).abs().max().item(),
            'first_step':rows[0],'last_step':rows[-1],
            'wall_s':time.monotonic()-started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'claim':'one Actor within-window held-out timestamps; no sparse-target full-surface precision/F-score claim',
            'free_boundary':'original beam before measured first hit only; no valid no-return beams present',
            'failure_ledger_delta':'pending analysis; risks remain active'}
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps(result),flush=True)
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc)})
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__': main()
