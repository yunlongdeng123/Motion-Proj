"""原生几何头微调后与build LiDAR规范融合，使用相同尺度/数量的PCA曲面片。"""
import argparse
import json
from pathlib import Path
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
from motion_proj.worldsim_v73.surface_seeds import farthest_indices,native_surface_points
from train_worldsim_v73_physical_surface import lidar_patches,evaluate_actor_surface


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    task='WS-V73-M2-GLOBAL-FUSION-01'
    out=Path('/root/autodl-tmp/runs/worldsim_v73')/task/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    started=time.monotonic()
    torch.set_num_threads(4)
    save('manifest.json',{'task_id':task,'run_id':args.run_id,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'native_run':str(args.native_run),'actor_data':str(args.actor_data),
        'method':'M1 trained DPT + build LiDAR, canonical point union + local PCA patches; no new optimizer',
        'density':'min(1024, build count) evidence + 512 native FPS, same fixed 0.06m spacing and 8 triangles per patch',
        'source_test_read':False,'external_test_read':False,
        'boundary':'same information simple fusion reference, not CAPA/AdaPoinTr/TSDF reproduction'})
    save('status.json',{'status':'running','phase':'load'})
    try:
        index=json.loads((args.actor_data/'index.json').read_text())
        save('cohort.json',index['cases'])
        scenes={s['scene_id']:s for s in torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False)}
        scales=json.loads((args.native_run/'metric_scales.json').read_text())
        decoder=ActorSpatialQueryDecoder().cuda()
        head=None; rows=[]
        with torch.no_grad():
            for entry in index['cases']:
                if entry['status']!='ready': continue
                case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=True)
                pyramid=NativeGeometryPyramid(args.native_run,scenes[entry['scene']],case['view_indices'],head=head,token_device='cpu')
                head=pyramid.head
                features,depth=pyramid(include_depth=True)
                del features
                points=case['points_actor_m'].cuda()
                native,counts=native_surface_points(depth,scales[entry['scene']],case['camera_from_actor'].cuda(),
                    case['intrinsics'].cuda(),case['size_lwh_m'].cuda(),case.get('valid_image_rect_xyxy'))
                count=min(len(points),decoder.evidence_queries)
                evidence=points[torch.linspace(0,len(points)-1,count,device='cuda').long()]
                support=native if len(native) else points
                seeds=support[farthest_indices(support,len(decoder.coarse))]
                centers=torch.cat([evidence,seeds])
                surface=lidar_patches(torch.cat([points,native]),decoder,centers=centers)
                metrics=evaluate_actor_surface(surface,case['rays'])
                row={'actor':entry,'surface_patches':len(centers),'native_candidates':len(native),
                     'per_view_native_support':counts,'lidar_fallback':not len(native),'frames':metrics}
                rows.append(row)
                torch.save({k:v.cpu() for k,v in surface.items()},out/(entry['owner']+'_surface.pt'))
                save('status.json',{'status':'running','phase':'fusion_evaluation','actors_done':len(rows),
                    'scene':entry['scene'],'native_candidates':len(native)})
                print(json.dumps({'scene':entry['scene'],'owner':entry['owner'],'native_candidates':len(native),
                                  'surface_patches':len(centers)}),flush=True)
                del pyramid,depth,points,native,support,seeds,evidence,centers,surface
        result={'status':'done','actors':len(rows),'final':rows,'wall_s':time.monotonic()-started,
                'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps({k:v for k,v in result.items() if k!='final'}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc)})
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__': main()
