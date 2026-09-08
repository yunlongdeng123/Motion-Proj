"""Evaluate fixed trained geometry on exported Actor windows without adaptation.

The input role is preserved (including external_confirmation). Empty input Actors
remain in the denominator, and every saved surface has the common hard readout.
"""
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
from motion_proj.worldsim_v73.surface_seeds import farthest_indices,native_surface_seeds,native_surface_points
from train_worldsim_v73_physical_surface import lidar_patches,evaluate_actor_surface


def make_query_decoder(config):
    # 旧checkpoint没有query_surface字段，保持原独立patch路径。
    if config.get('query_surface','patches')=='shared_mesh':
        from motion_proj.worldsim_v73.shared_mesh_queries import ActorSharedMeshQueryDecoder
        return ActorSharedMeshQueryDecoder(mesh_level=config.get('mesh_level',3))
    return ActorSpatialQueryDecoder()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--native-run',type=Path,help='prefix and build alignment for these input windows')
    parser.add_argument('--checkpoint',type=Path,help='fixed completed shared-training latest.pt')
    parser.add_argument('--method',choices=['checkpoint','native_fusion','lidar_pca'],default='checkpoint')
    parser.add_argument('--include-visual-only',action='store_true',
                        help='explicitly allow zero-LiDAR Actors with camera poses; does not imply the checkpoint was trained on them')
    parser.add_argument('--input-subset',choices=['all','zero_lidar'],default='all',
                        help='metadata-only scope for a registered zero-LiDAR input-path analysis')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(7304); started=time.monotonic()
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    index=json.loads((args.actor_data/'index.json').read_text())
    original_actors=len(index['cases'])
    if args.input_subset=='zero_lidar':
        index['cases']=[case for case in index['cases'] if case['build_points']==0]
    roles=sorted({case['role'] for case in index['cases']})
    manifest={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'actor_data':str(args.actor_data),'native_run':str(args.native_run),'checkpoint':str(args.checkpoint),
              'method':args.method,'roles':roles,'optimizer_updates':0,'camera_parameters_updated':False,
              'input_boundary':'original build support/images, fixed build alignment and known rigid trajectories only',
              'target_boundary':'heldout measurements used only by common final evaluator, never prediction',
              'empty_input_policy':('zero-LiDAR camera-pose Actors may use native support or existing coarse queries; both-modality-absent Actors remain empty'
                                    if args.include_visual_only else 'all non-ready Actors retained with empty prediction, matching shared model protocol'),
              'include_visual_only':args.include_visual_only,'input_subset':args.input_subset,
              'original_cohort_actors':original_actors,'selected_actors':len(index['cases']),
              'selection_boundary':'input-subset depends only on original build LiDAR count, never prediction or heldout quality',
              'checkpoint_boundary':'explicit visual-only override is input-path migration, not evidence that these Actors participated in checkpoint training',
              'surface_readout':'literal triangles from the selected surface representation; no opacity, hull or export remeshing',
              'external_test_read':'external_confirmation' in roles}
    save('manifest.json',manifest); save('cohort.json',index['cases'])
    save('status.json',{'status':'running','phase':'load'})
    try:
        state=None; head=None; mode=args.method; config={}
        if args.method=='checkpoint':
            state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
            config=state['config']; mode=config['mode']
        decoder=make_query_decoder(config).cuda().eval()
        if state is not None:
            decoder.load_state_dict(state['query_decoder'])
        query_surface=config.get('query_surface','patches')
        parameterization=('lidar_pca_patches' if mode=='lidar_pca' else
                          'native_lidar_pca' if mode in ['native_only','native_fusion'] else query_surface)
        manifest.update(mode=mode,query_surface=query_surface,surface_parameterization=parameterization,
                        mesh_level=config.get('mesh_level',3) if query_surface=='shared_mesh' else None,
                        completion_initialization=config.get('completion_init','native_surface'),
                        surface_patches_boundary='legacy center count; shared_mesh counts vertices, not independent patches')
        visual_only_enabled=args.include_visual_only or bool(config.get('include_visual_only',False))
        manifest['effective_visual_only']=visual_only_enabled
        manifest['checkpoint_visual_only_training']=bool(config.get('include_visual_only',False)) if state is not None else None
        manifest['checkpoint_boundary']=('checkpoint explicitly includes visual-only input training; preserve that input policy at fixed inference'
            if config.get('include_visual_only',False) else
            'explicit visual-only override is input-path migration, not evidence that these Actors participated in checkpoint training')
        if visual_only_enabled:
            manifest['empty_input_policy']='zero-LiDAR camera-pose Actors may use native support or existing coarse queries; both-modality-absent Actors remain empty'
        save('manifest.json',manifest)
        visual=mode not in ['lidar_only','lidar_pca']
        scenes={}; scales={}; prefix_cache={}; depth_cache={}
        if visual:
            scenes={s['scene_id']:s for s in torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False,mmap=True)}
            scales=json.loads((args.native_run/'metric_scales.json').read_text())
        rows=[]
        for entry in index['cases']:
            case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=True)
            case['metadata']['input_status']=entry['status']
            points=case['points_actor_m'].cuda(); support={}
            visual_only=visual_only_enabled and visual and not len(points) and bool(case['view_indices'])
            case['metadata']['visual_only_prediction_enabled']=visual_only
            if entry['status']!='ready' and not visual_only:
                surface={'vertices_actor_m':points.new_empty(0,3),'faces':torch.empty(0,3,device='cuda',dtype=torch.long),
                         'centers_actor_m':points.new_empty(0,3)}
                support={'prediction_unavailable':True,'reason':case['metadata'].get('reason')}
            elif mode=='lidar_pca':
                surface=lidar_patches(points,decoder)
            else:
                size=case['size_lwh_m'].cuda(); matrices=case['camera_from_actor'].cuda()
                calibration=case['intrinsics'].cuda(); rect=case.get('valid_image_rect_xyxy')
                depth=None; features=None; has_views=visual and bool(case['view_indices'])
                if has_views:
                    pyramid=NativeGeometryPyramid(args.native_run,scenes[entry['scene']],case['view_indices'],head=head,
                                                   token_device='cpu',prefix_cache=prefix_cache)
                    if head is None:
                        head=pyramid.head
                        if state is not None: head.load_state_dict(state['depth_head'])
                        head.eval()
                    if mode in ['native_only','native_fusion']:
                        depth=pyramid.depth_only(depth_cache)
                    else:
                        features,depth=pyramid(include_depth=True)
                completion=config.get('completion_init','native_surface')
                if mode in ['native_only','native_fusion']:
                    native,counts=(native_surface_points(depth,scales[entry['scene']],matrices,calibration,size,rect)
                                   if depth is not None else (points.new_empty(0,3),[]))
                    source=native if len(native) else points
                    n=min(len(points),decoder.evidence_queries)
                    evidence=points[torch.linspace(0,len(points)-1,n,device='cuda').long()]
                    if len(source):
                        centers=torch.cat([evidence,source[farthest_indices(source,len(decoder.coarse))]])
                        surface=lidar_patches(torch.cat([points,native]),decoder,centers=centers)
                    else:
                        surface={'vertices_actor_m':points.new_empty(0,3),
                                 'faces':torch.empty(0,3,device='cuda',dtype=torch.long),'centers_actor_m':points.new_empty(0,3)}
                    support={'native_candidates':len(native),'per_view_native_support':counts,'lidar_fallback':not len(native)}
                    if not len(source): support.update(lidar_fallback=False,prediction_unavailable=True,reason='no_native_or_lidar_surface_support')
                else:
                    if completion=='native_surface' and depth is not None:
                        seeds,support=native_surface_seeds(depth,scales[entry['scene']],matrices,calibration,size,
                                                          points,len(decoder.coarse),valid_image_rect=rect)
                    else:
                        seeds=points[farthest_indices(points,len(decoder.coarse))] if len(points) else None
                        support={'native_candidates':None,'lidar_fallback':completion=='native_surface' and len(points)>0,
                                 'coarse_fallback':not len(points)}
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        surface=decoder(points,size,features,matrices,calibration,case['image_hw'],case['camera_ids'].cuda(),
                            case['time_offsets_s'].cuda(),use_spatial=mode!='pointwise',use_visual=features is not None,
                            completion_seeds=seeds,camera_weights=case.get('camera_embedding_weights'),valid_image_rect=rect)
                support['has_actor_camera_pose']=has_views
                support['input_path']='visual_only' if visual_only else 'existing_multimodal_or_lidar'
                if has_views: del pyramid
                del features,depth
            row={'actor':case['metadata'],'surface_patches':len(surface['centers_actor_m']),
                 'surface_parameterization':parameterization,'surface_vertices':len(surface['vertices_actor_m']),
                 'surface_faces':len(surface['faces']),'surface_patches_boundary':manifest['surface_patches_boundary'],
                 'seed_support':support,'frames':evaluate_actor_surface(surface,case['rays']),
                 'extra_time_usage':'evaluation_only'}
            rows.append(row)
            torch.save({k:v.cpu() for k,v in surface.items() if isinstance(v,torch.Tensor)},args.output/(entry['owner']+'_surface.pt'))
            save('status.json',{'status':'running','phase':'fixed_evaluation','actors_done':len(rows),'actors_total':len(index['cases']),
                                'scene':entry['scene'],'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
            print(json.dumps({'scene':entry['scene'],'owner':entry['owner'],'surface_vertices':row['surface_vertices'],
                              'surface_faces':row['surface_faces'],'surface_parameterization':parameterization}),flush=True)
            del case,surface,points
        result={'status':'done','final':rows,'actors':len(rows),'roles':roles,'optimizer_updates':0,
                'mode':mode,'query_surface':query_surface,'mesh_level':manifest['mesh_level'],
                'surface_parameterization':parameterization,'completion_initialization':manifest['completion_initialization'],
                'wall_s':time.monotonic()-started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                'fit_label_times':'evaluation_only','boundary':'fixed shared weights; input roles preserved; no new-source adaptation or target-supported initialization'}
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps({k:v for k,v in result.items() if k!='final'}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__': main()
