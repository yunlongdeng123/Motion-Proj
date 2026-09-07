"""真实12视图原生DPT到空间曲面的联合反向实验，检查资源与有效几何梯度。"""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import ijson
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.spatial_queries import ActorSpatialQueryDecoder


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--steps',type=int,default=8)
    args=parser.parse_args()
    out=Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-JOINT-GEOMETRY-PATH-01')/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    started=time.monotonic()
    torch.manual_seed(7302)
    torch.set_num_threads(8)
    save('manifest.json',{'task_id':'WS-V73-M2-JOINT-GEOMETRY-PATH-01','run_id':args.run_id,
         'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'native_run':str(args.native_run),'steps':args.steps,'seed':7302,
         'failure_ledger_refs':['V73-F01','V73-F02','V73-F03','V73-F04','V73-F05'],
         'claim':'joint native-DPT geometry gradient/resource experiment, not final method comparison'})
    save('status.json',{'status':'running','phase':'load_build'})
    try:
        scenes=torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False)
        scene=next(s for s in scenes if s['scene_id']=='scene-0100')
        owners=sorted({o for v in scene['views'] for o,m in zip(v['owners'],v['actor_mask']) if m})
        build={}
        for owner in owners:
            parts=[]
            for view in scene['views']:
                take=torch.tensor([o==owner for o in view['owners']]) & ~view['diagnostic_mask']
                parts.append(view['points_actor_m'][take])
            build[owner]=torch.unique(torch.cat(parts),dim=0)
        # 资源/梯度实验按build支持量选Actor，不使用诊断质量挑对象。
        owner=max(owners,key=lambda o:len(build[o]))
        points=build[owner].cuda()
        metadata=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval/sample_annotation.json')
        with metadata.open('rb') as handle:
            for row in ijson.items(handle,'item'):
                if row['instance_token']==owner:
                    size=torch.tensor(np.array(row['size'],dtype=float)[[1,0,2]],device='cuda',dtype=torch.float32)
                    break
        views=[(i,v) for i,v in enumerate(scene['views']) if owner in v['world_from_actor']]
        matrices=torch.tensor(np.stack([np.linalg.inv(v['world_from_camera'])@v['world_from_actor'][owner]
                               for _,v in views]),device='cuda',dtype=torch.float32)
        intrinsics=torch.tensor(np.stack([v['intrinsics'] for _,v in views]),device='cuda',dtype=torch.float32)
        ids={'CAM_FRONT':0,'CAM_FRONT_RIGHT':1,'CAM_BACK_RIGHT':2,'CAM_BACK':3,'CAM_BACK_LEFT':4,'CAM_FRONT_LEFT':5}
        camera_ids=torch.tensor([ids[v['camera_id']] for _,v in views],device='cuda')
        t0=views[0][1]['camera_time_us']
        times=torch.tensor([(v['camera_time_us']-t0)/1e6 for _,v in views],device='cuda',dtype=torch.float32)
        sys.path.insert(0,'/root/autodl-tmp/external/worldsim_v72/vggt')
        from vggt.heads.dpt_head import DPTHead
        head=DPTHead(dim_in=2048,output_dim=2,activation='exp',conf_activation='expp1').cuda()
        loaded=torch.load(args.native_run/'latest.pt',map_location='cpu',weights_only=True)
        head.load_state_dict(loaded['depth_head'])
        del loaded
        decoder=ActorSpatialQueryDecoder().cuda()
        optimizer=torch.optim.AdamW([*head.parameters(),*decoder.parameters()],lr=1e-5)
        token_inputs=[]
        for i,v in views:
            data=torch.load(args.native_run/'frozen_prefix'/f'{scene["scene_id"]}_{i:02}.pt',weights_only=True)
            token_inputs.append((tuple(data['tokens'][j].cuda() for j in [4,11,17,23]),
                                 v['image'][None,None].cuda(),data['patch_start']))
        before=head.projects[0].weight.detach().clone()
        def pyramid(*inputs):
            tokens=[None]*24
            for j,t in zip([4,11,17,23],inputs[:4]): tokens[j]=t
            captured={}
            handles=[]
            for level,name in enumerate(['refinenet4','refinenet3','refinenet2','refinenet1']):
                def hook(module,values,result,key=level): captured[key]=result
                handles.append(getattr(head.scratch,name).register_forward_hook(hook))
            try:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    head(tokens,inputs[4],token_inputs[0][2])
            finally:
                for handle in handles: handle.remove()
            return tuple(captured[j] for j in range(4))
        rows=[]
        for step in range(args.steps):
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            features=[[],[],[],[]]
            for tokens,image,patch_start in token_inputs:
                maps=checkpoint(pyramid,*tokens,image,use_reentrant=False)
                for level,feature in enumerate(maps): features[level].append(feature)
            features=[torch.cat(level) for level in features]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=decoder(points,size,features,matrices,intrinsics,views[0][1]['image'].shape[-2:],camera_ids,times)
            centers=prediction['centers_actor_m'].float()
            # 一侧coverage不给未观测completion表面伪负标签；此阶段尚不宣称free/event成功。
            coverage=torch.stack([torch.cdist(chunk[None],centers[None])[0].amin(1).mean()
                                  for chunk in points.split(256)]).mean()
            envelope=(centers.abs()-size/2-.25).clamp_min(0).square().mean()
            loss=coverage+.05*envelope
            loss.backward()
            native_grad=head.projects[0].weight.grad.norm().item()
            geometry_grad=decoder.displacements[0].weight.grad.norm().item()
            torch.nn.utils.clip_grad_norm_([*head.parameters(),*decoder.parameters()],1.)
            optimizer.step()
            row={'step':step+1,'loss':loss.item(),'coverage_m':coverage.item(),'native_project_gradient':native_grad,
                 'geometry_gradient':geometry_grad,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                 'queries':len(centers),'vertices':len(prediction['vertices_actor_m']),
                 'faces':len(prediction['faces']),'visual_observed_fraction':prediction['visual_observed'].float().mean().item()}
            rows.append(row)
            save('status.json',{'status':'running','phase':'joint_backward',**row})
            print(json.dumps(row),flush=True)
            del prediction,features,maps,loss,coverage,envelope,centers
        torch.save({'depth_head':head.state_dict(),'query_decoder':decoder.state_dict(),'config':vars(args)},out/'joint.pt')
        result={'status':'done','scene':scene['scene_id'],'owner':owner,'build_points':len(points),'views':len(views),
                'steps':rows,'native_project_max_change':(head.projects[0].weight.detach()-before).abs().max().item(),
                'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                'failure_ledger_delta':'none; four method risks remain active',
                'boundary':'real joint gradient/resource only; no free/event loss, held-out score or method victory'}
        save('summary.json',result)
        save('status.json',{'status':'done'})
        print(json.dumps(result),flush=True)
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc),
                           'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__': main()
