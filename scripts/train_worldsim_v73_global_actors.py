"""共享原生几何头/空间查询跨日志训练，开发日志仅评价，冻结前缀驻留CPU。"""
import argparse
import json
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from motion_proj.worldsim_v73.native_pyramid import NativeGeometryPyramid
from motion_proj.worldsim_v73.spatial_queries import ActorSpatialQueryDecoder
from motion_proj.worldsim_v73.surface_seeds import farthest_indices,native_surface_seeds
from motion_proj.worldsim_v73.surface_readout import closest_surface_points,first_triangle_intersection,direct_free_space_loss
from train_worldsim_v73_physical_surface import lidar_patches,evaluate_actor_surface


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--mode',choices=['joint','pointwise','lidar_only'],default='joint')
    parser.add_argument('--completion-init',choices=['native_surface','lidar_surface'],default='native_surface')
    parser.add_argument('--free-weight',type=float,default=.5)
    args=parser.parse_args()
    task='WS-V73-M2-GLOBAL-ACTOR-01'
    out=Path('/root/autodl-tmp/runs/worldsim_v73')/task/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        tmp=out/(name+'.tmp')
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        tmp.replace(out/name)
    started=time.monotonic()
    torch.set_num_threads(6); torch.manual_seed(7304); random.seed(7304)
    save('manifest.json',{'task_id':task,'run_id':args.run_id,'seed':7304,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'shared_parameters':'native DPT and query decoder across fit logs; no development gradient or optimizer update',
        'frozen_prefix':'aggregator 24-view joint tokens, stored CPU; upper aggregation not adapted',
        'source_test_read':False,'external_test_read':False,'failure_ledger_refs':['V73-F01','V73-F02','V73-F03','V73-F04','V73-F05']})
    save('status.json',{'status':'running','phase':'load_contexts'})
    try:
        index=json.loads((args.actor_data/'index.json').read_text())
        save('cohort.json',index['cases'])
        cases=[]
        for row in index['cases']:
            if row['status']=='ready': cases.append(torch.load(args.actor_data/row['file'],map_location='cpu',weights_only=True))
        scenes={s['scene_id']:s for s in torch.load(args.native_run/'build_observations.pt',weights_only=False,map_location='cpu')}
        scales=json.loads((args.native_run/'metric_scales.json').read_text())
        pyramids={}; head=None
        if args.mode!='lidar_only':
            for case in cases:
                key=case['metadata']['owner']; scene=scenes[case['metadata']['scene']]
                pyramid=NativeGeometryPyramid(args.native_run,scene,case['view_indices'],head=head,token_device='cpu')
                head=pyramid.head; pyramids[key]=pyramid
        del scenes
        torch.manual_seed(7304)
        decoder=ActorSpatialQueryDecoder().cuda()
        parameters=[*(head.parameters() if head is not None else []),*decoder.parameters()]
        optimizer=torch.optim.AdamW(parameters,lr=1e-5)
        fit=[c for c in cases if c['metadata']['role']=='fit']
        if not fit: raise ValueError('没有可训练fit Actor')
        for case in cases:
            case['lidar_seed']=case['points_actor_m'][farthest_indices(case['points_actor_m'],len(decoder.coarse))]
        before=head.projects[0].weight.detach().clone() if head is not None else None

        def predict(case):
            points=case['points_actor_m'].cuda(); size=case['size_lwh_m'].cuda()
            matrices=case['camera_from_actor'].cuda(); calibration=case['intrinsics'].cuda()
            if args.completion_init=='native_surface':
                features,depth=pyramids[case['metadata']['owner']](include_depth=True)
                seeds,support=native_surface_seeds(depth,scales[case['metadata']['scene']],matrices,calibration,
                                                    size,points,len(decoder.coarse))
            else:
                features=pyramids[case['metadata']['owner']]() if head is not None else None
                seeds=case['lidar_seed'].cuda(); support={'lidar_fallback':False,'initialization':'lidar_surface'}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=decoder(points,size,features,matrices,calibration,case['image_hw'],case['camera_ids'].cuda(),
                    case['time_offsets_s'].cuda(),use_spatial=args.mode!='pointwise',use_visual=args.mode!='lidar_only',
                    completion_seeds=seeds)
            return result,support

        @torch.no_grad()
        def evaluate(tag,baseline=False):
            rows=[]
            for case in cases:
                if baseline:
                    surface=lidar_patches(case['points_actor_m'].cuda(),decoder)
                    support={}
                else:
                    surface,support=predict(case)
                metrics=evaluate_actor_surface(surface,case['rays'])
                rows.append({'actor':case['metadata'],'surface_patches':len(surface['centers_actor_m']),
                             'seed_support':support,'frames':metrics})
                if tag=='final':
                    torch.save({k:v.cpu() for k,v in surface.items() if isinstance(v,torch.Tensor)},
                               out/(case['metadata']['owner']+'_surface.pt'))
                del surface
            save(tag+'.json',rows)
            return rows

        save('status.json',{'status':'running','phase':'initial_evaluation','fit_actors':len(fit),'all_actors':len(cases)})
        baseline=evaluate('lidar_baseline',True)
        initial=evaluate('initial')
        history=[]
        for epoch in range(args.epochs):
            random.shuffle(fit)
            for case in fit:
                tick=time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                prediction,support=predict(case)
                vertices=prediction['vertices_actor_m'].float(); faces=prediction['faces']
                points=case['points_actor_m'].cuda()
                chosen=points[torch.randperm(len(points),device='cuda')[:1024]]
                nearest=closest_surface_points(vertices,faces,chosen)
                coverage=(nearest-chosen).norm(dim=-1).mean()
                build=[r for r in case['rays'] if r['role']=='build']
                ranges=torch.cat([r['observed_first_range_m'] for r in build]).cuda()
                origins=torch.cat([r['origins_actor_m'] for r in build]).cuda()
                directions=torch.cat([r['directions_actor'] for r in build]).cuda()
                ids=torch.randperm(len(ranges),device='cuda')[:512]
                depth,_=first_triangle_intersection(vertices,faces,origins[ids],directions[ids])
                free=direct_free_space_loss(depth,ranges[ids])
                envelope=(vertices.abs()-case['size_lwh_m'].cuda()/2-.25).clamp_min(0).square().mean()
                loss=coverage+args.free_weight*free+.05*envelope
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not torch.isfinite(grad): raise FloatingPointError('共享几何训练出现非有限梯度')
                output_grad=sum(p.grad.norm().item() for p in head.scratch.output_conv2.parameters() if p.grad is not None) if head is not None else None
                optimizer.step()
                row={'epoch':epoch+1,'scene':case['metadata']['scene'],'owner':case['metadata']['owner'],
                    'loss':loss.item(),'coverage_m':coverage.item(),'free_intrusion_m':free.item(),
                    'gradient_norm_before_clip':grad.item(),'native_output_gradient_after_clip':output_grad,
                    'lidar_fallback':support.get('lidar_fallback',False),'views':len(case['view_indices']),
                    'query_count':len(prediction['centers_actor_m']),'step_s':time.monotonic()-tick,
                    'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
                history.append(row)
                with (out/'train.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
                save('status.json',{'status':'running','phase':'shared_train','elapsed_s':time.monotonic()-started,**row})
                print(json.dumps(row),flush=True)
                del prediction,vertices,faces,loss,coverage,free,envelope,nearest,depth,points,chosen,origins,directions,ranges
            checkpoint={'depth_head':head.state_dict() if head is not None else None,'query_decoder':decoder.state_dict(),
                        'optimizer':optimizer.state_dict(),'epoch':epoch+1,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
            torch.save(checkpoint,out/'latest.tmp.pt'); (out/'latest.tmp.pt').replace(out/'latest.pt')
        final=evaluate('final')
        result={'status':'done','fit_actors':len(fit),'development_actors':len(cases)-len(fit),
            'epochs':args.epochs,'updates':len(history),'mode':args.mode,'completion_initialization':args.completion_init,
            'native_project_max_change':(head.projects[0].weight.detach()-before).abs().max().item() if head is not None else None,
            'first_step':history[0],'last_step':history[-1],'wall_s':time.monotonic()-started,
            'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'baseline':baseline,'initial':initial,'final':final,
            'boundary':'one Actor per existing log, development no gradient; within-window heldout time, not new-source confirmation',
            'failure_ledger_delta':'pending comparison, risks remain active'}
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps({k:v for k,v in result.items() if k not in ['baseline','initial','final']}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc),
                           'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__': main()
