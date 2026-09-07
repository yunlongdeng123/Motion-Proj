"""完整Actor队列的原生DPT+LiDAR融合；每窗口只读解码一次，保留缺输入对象。"""
import argparse,json
from pathlib import Path
import resource,subprocess,sys,time,traceback
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from motion_proj.worldsim_v73.native_pyramid import NativeGeometryPyramid
from motion_proj.worldsim_v73.spatial_queries import ActorSpatialQueryDecoder
from motion_proj.worldsim_v73.surface_seeds import farthest_indices,native_surface_points
from train_worldsim_v73_physical_surface import lidar_patches,evaluate_actor_surface


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--baseline-results',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    out=Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-GLOBAL-FUSION-01')/args.run_id
    out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); torch.set_num_threads(4)
    def save(name,value):
        tmp=out/(name+'.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); tmp.replace(out/name)
    save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'native_run':str(args.native_run),'actor_data':str(args.actor_data),'baseline_results':str(args.baseline_results),
        'method':'M1r3 native DPT fine-tuning + canonical build LiDAR fusion, no additional optimizer or extra fit-time labels',
        'execution':'fixed DPT at evaluation; decode all 24 views once per window, reuse only within frozen evaluation',
        'density':'min(build,1024)+512 native FPS; same fixed PCA patch spacing .06m; native-only when no build LiDAR but native supports exist',
        'source_test_read':False,'external_test_read':False,'boundary':'all original window Actors retained, no prediction-quality selection; known-box ownership proxy'})
    save('status.json',{'status':'running','phase':'load'})
    try:
        entries=json.loads((args.actor_data/'index.json').read_text())['cases']; save('cohort.json',entries)
        scenes=torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False)
        scales=json.loads((args.native_run/'metric_scales.json').read_text())
        decoder=ActorSpatialQueryDecoder().cuda().eval(); head=None; rows=[]
        with torch.no_grad():
            for scene in scenes:
                pyramid=NativeGeometryPyramid(args.native_run,scene,list(range(len(scene['views']))),head=head,token_device='cpu')
                head=pyramid.head; head.eval()
                features,depth=pyramid(include_depth=True); del features
                for entry in [r for r in entries if r['scene']==scene['scene_id']]:
                    case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=True)
                    points=case['points_actor_m'].cuda()
                    if case['view_indices']:
                        native,counts=native_surface_points(depth[case['view_indices']],scales[entry['scene']],
                            case['camera_from_actor'].cuda(),case['intrinsics'].cuda(),case['size_lwh_m'].cuda())
                    else:
                        native=points.new_empty(0,3); counts=[]
                    support=native if len(native) else points
                    if len(support):
                        n=min(len(points),decoder.evidence_queries)
                        evidence=points[torch.linspace(0,max(0,len(points)-1),n,device='cuda').long()]
                        centers=torch.cat([evidence,support[farthest_indices(support,len(decoder.coarse))]])
                        surface=lidar_patches(torch.cat([points,native]),decoder,centers=centers)
                    else:
                        surface={'vertices_actor_m':points.new_empty(0,3),'faces':torch.empty(0,3,dtype=torch.long,device='cuda'),
                            'centers_actor_m':points.new_empty(0,3)}
                    rows.append({'actor':entry,'surface_patches':len(surface['centers_actor_m']),
                        'seed_support':{'native_candidates':len(native),'per_view_native_support':counts,
                            'lidar_fallback':not len(native),'prediction_unavailable':not len(support)},
                        'frames':evaluate_actor_surface(surface,case['rays']),'extra_time_usage':'evaluation_only'})
                    torch.save({key:value.cpu() for key,value in surface.items()},out/(entry['scene']+'__'+entry['owner']+'_surface.pt'))
                    save('status.json',{'status':'running','phase':'population_surface_evaluation','actors_done':len(rows),
                        'scene':entry['scene'],'native_candidates':len(native),'elapsed_s':time.monotonic()-started})
                    print(json.dumps({'scene':entry['scene'],'owner':entry['owner'],'native_candidates':len(native)}),flush=True)
                    del case,points,native,support,surface
                del pyramid,depth
        result={'status':'done','actors':len(rows),'final':rows,
            'baseline':json.loads(args.baseline_results.read_text()),'fit_label_times':'build',
            'wall_s':time.monotonic()-started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'boundary':'full existing window population, fixed M1r3 decoder; no new-source confirmation; extra times evaluation only'}
        save('summary.json',result); save('status.json',{'status':'done'})
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc)})
        (out/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
