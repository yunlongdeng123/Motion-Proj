"""完整预训练AdaPoinTr的真实稀疏标签微调与统一物理曲面评价。"""
import argparse,json,random,resource,subprocess,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from motion_proj.worldsim_v73.adapointr_baseline import load_official_model,prepare_input,select_surface_centers,sparse_denoising_loss
from motion_proj.worldsim_v73.spatial_queries import ActorSpatialQueryDecoder
from motion_proj.worldsim_v73.surface_readout import closest_surface_points,first_triangle_intersection,direct_free_space_loss
from train_worldsim_v73_physical_surface import lidar_patches,evaluate_actor_surface


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--fit-targets',type=Path,required=True)
    parser.add_argument('--baseline-results',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--repo',type=Path,default=Path('/root/autodl-tmp/build/worldsim_v72/PoinTr_AdaPoinTr_4603257'))
    parser.add_argument('--weights',type=Path,default=Path('/root/autodl-tmp/external/worldsim_v72/checkpoints/AdaPoinTr_PCN.pth'))
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--accumulate',type=int,default=4)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=7307)
    args=parser.parse_args()
    out=Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-ADAPOINTR-01')/args.run_id
    out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); torch.set_num_threads(4)
    torch.manual_seed(args.seed); random.seed(args.seed)
    def save(name,value):
        p=out/name; tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); tmp.replace(p)
    manifest={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'method':'all-parameter official pretrained AdaPoinTr, sparse-observation task fine-tuning; not exact PCN benchmark reproduction',
        'input':'all original build LiDAR points; repeat only when fewer than 512 slots; no target initialization or point-count cap',
        'supervision':'fit-only full-track measured positives and original first-return free; development no gradient',
        'loss':'target-to-matched-patch coverage +0.5 literal free +0.05 soft box envelope +0.1 target-to-coarse coverage +0.1 sparse local denoising coverage',
        'unknown_region':'no global predicted-to-sparse-target penalty; sparse measurements are not complete surfaces',
        'physical_readout':'min(build points,1024)+512 fixed .06m PCA patches from full 16384-point output; same triangle/first-return operators as query method',
        'surface_gradient':'FPS/neighborhood/PCA orientation held fixed per step; selected center positions receive surface/free gradients; PCA recomputed each step',
        'optimizer':'AdamW lr1e-4 weight_decay5e-4; lr x0.9 at epoch21; variable-point actors accumulated in groups4; clip norm10',
        'density':'full native point cloud saved at initial/final for separate density/point-set analysis; primary surface budget matched',
        'source_test_read':False,'external_test_read':False}
    save('manifest.json',manifest); save('status.json',{'status':'running','phase':'load_official_model'})
    try:
        model,source=load_official_model(args.repo,args.weights)
        manifest['official_model']=source; save('manifest.json',manifest)
        model=model.cuda(); model.train()
        # 仅复用固定patch grid/面定义，不训练这个辅助容器的query参数。
        template=ActorSpatialQueryDecoder()
        patch_definition=SimpleNamespace(patch_grid=template.patch_grid.cuda(),patch_faces=template.patch_faces.cuda(),
                                         patch_spacing_m=template.patch_spacing_m)
        del template
        torch.manual_seed(args.seed)
        entries=json.loads((args.actor_data/'index.json').read_text())['cases']; save('cohort.json',entries)
        cases=[]
        for entry in entries:
            case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=True)
            if entry['role']=='fit' and entry['status']=='ready':
                labels=torch.load(args.fit_targets/(entry['scene']+'__'+entry['owner']+'.pt'),map_location='cpu',weights_only=True)
                case['training_points']=labels['target_points_actor_m']; case['training_rays']=labels['target_rays']
            cases.append(case)
        fit=[case for case in cases if case['metadata']['role']=='fit' and case['metadata']['status']=='ready']
        baseline=json.loads(args.baseline_results.read_text()); save('lidar_baseline.json',baseline)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=5e-4)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=21,gamma=.9)

        def matched_surface(fine,case):
            centers=select_surface_centers(fine,min(len(case['points_actor_m']),1024)+512)
            return lidar_patches(fine,patch_definition,centers=centers)

        @torch.no_grad()
        def evaluate(tag):
            model.eval(); rows=[]
            for case in cases:
                entry=case['metadata']; points=case['points_actor_m'].cuda()
                if entry['status']=='ready':
                    xyz,scale=prepare_input(points,case['size_lwh_m'].cuda())
                    coarse,fine=model(xyz); fine=fine[0].float()*scale
                    surface=matched_surface(fine,case)
                else:
                    fine=points.new_empty(0,3)
                    surface={'vertices_actor_m':points.new_empty(0,3),'faces':torch.empty(0,3,dtype=torch.long,device='cuda'),
                             'centers_actor_m':points.new_empty(0,3)}
                row={'actor':entry,'surface_patches':len(surface['centers_actor_m']),'native_output_points':len(fine),
                    'frames':evaluate_actor_surface(surface,case['rays']),
                    'seed_support':{'prediction_unavailable':entry['status']!='ready','lidar_fallback':False},
                    'extra_time_usage':'training_labels' if entry['role']=='fit' and entry['status']=='ready' else 'evaluation_only'}
                rows.append(row)
                torch.save(fine.cpu(),out/(entry['owner']+'_'+tag+'_points.pt'))
                if tag=='final': torch.save({k:v.cpu() for k,v in surface.items()},out/(entry['owner']+'_surface.pt'))
                save('status.json',{'status':'running','phase':tag+'_evaluation','actors_done':len(rows),'elapsed_s':time.monotonic()-started})
            save(tag+'.json',rows); model.train(); return rows

        initial=evaluate('initial'); history=[]; updates=0
        for epoch in range(args.epochs):
            random.shuffle(fit)
            for start in range(0,len(fit),args.accumulate):
                group=fit[start:start+args.accumulate]; optimizer.zero_grad(set_to_none=True); group_rows=[]
                for case in group:
                    tick=time.monotonic(); points=case['points_actor_m'].cuda(); size=case['size_lwh_m'].cuda()
                    xyz,scale=prepare_input(points,size)
                    coarse,denoised_coarse,denoised_fine,fine=model(xyz)
                    coarse=coarse[0].float()*scale; fine=fine[0].float()*scale
                    denoised_coarse=denoised_coarse[0].float()*scale; denoised_fine=denoised_fine[0].float()*scale
                    surface=matched_surface(fine,case); vertices=surface['vertices_actor_m']; faces=surface['faces']
                    targets=case['training_points'].cuda(); chosen=targets[torch.randperm(len(targets),device='cuda')[:1024]]
                    nearest=closest_surface_points(vertices,faces,chosen)
                    coverage=(nearest-chosen).norm(dim=-1).mean()
                    coarse_coverage=torch.cdist(chosen,coarse).min(-1).values.mean()
                    denoising=sparse_denoising_loss(denoised_coarse,denoised_fine,chosen)
                    rays=case['training_rays']; ranges=torch.cat([r['observed_first_range_m'] for r in rays]).cuda()
                    origins=torch.cat([r['origins_actor_m'] for r in rays]).cuda(); directions=torch.cat([r['directions_actor'] for r in rays]).cuda()
                    ids=torch.randperm(len(ranges),device='cuda')[:512]
                    depth,_=first_triangle_intersection(vertices,faces,origins[ids],directions[ids])
                    free=direct_free_space_loss(depth,ranges[ids])
                    envelope=(vertices.abs()-size/2-.25).clamp_min(0).square().mean()
                    loss=coverage+.5*free+.05*envelope+.1*coarse_coverage+.1*denoising
                    (loss/len(group)).backward()
                    row={'epoch':epoch+1,'scene':case['metadata']['scene'],'owner':case['metadata']['owner'],
                        'loss':loss.item(),'coverage_m':coverage.item(),'free_intrusion_m':free.item(),
                        'coarse_coverage_m':coarse_coverage.item(),'denoising_m':denoising.item(),
                        'native_output_points':len(fine),'matched_patches':len(surface['centers_actor_m']),
                        'raw_input_points':len(points),'input_slots':xyz.shape[1],'step_s':time.monotonic()-tick,
                        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
                    group_rows.append(row)
                    del xyz,coarse,fine,denoised_coarse,denoised_fine,surface,vertices,faces,targets,chosen,nearest,loss,coverage,coarse_coverage,denoising,depth,free,envelope,ranges,origins,directions
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(gradient): raise FloatingPointError('AdaPoinTr梯度出现非有限值')
                optimizer.step(); updates+=1
                for row in group_rows:
                    row.update(optimizer_update=updates,group_gradient_norm_before_clip=gradient.item())
                    history.append(row)
                    with (out/'train.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
                save('status.json',{'status':'running','phase':'task_fine_tuning','elapsed_s':time.monotonic()-started,**group_rows[-1]})
                print(json.dumps(group_rows[-1]),flush=True)
            scheduler.step()
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'epoch':epoch+1,'optimizer_updates':updates,'config':manifest['config'],
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(),'rng_python':random.getstate()},out/'latest.tmp.pt')
            (out/'latest.tmp.pt').replace(out/'latest.pt')
        final=evaluate('final')
        save('summary.json',{'status':'done','fit_label_times':'full_track','epochs':args.epochs,
            'actor_presentations':len(history),'optimizer_updates':updates,'fit_actors':len(fit),
            'cohort_actors':len(cases),'baseline':baseline,'initial':initial,'final':final,
            'first_step':history[0],'last_step':history[-1], 'wall_s':time.monotonic()-started,
            'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'boundary':'existing window population; all-parameter official AdaPoinTr task adaptation; sparse-aware auxiliary losses and fixed-PCA position gradients; not exact official PCN protocol or new-source confirmation'})
        save('status.json',{'status':'done'})
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc),
                           'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
        (out/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
