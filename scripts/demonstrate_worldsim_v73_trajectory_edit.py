"""Reuse canonical physical surfaces under edited rigid trajectories, without edited GT."""
import argparse,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.av2_geometry import TimedPoses
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH,compose_first


class LateralTrajectory:
    def __init__(self,base,offset_m): self.base,self.offset_m,self.times=base,offset_m,base.times
    def at(self,times):
        poses,known=self.base.at(times); poses=poses.copy()
        poses[...,:3,3]+=poses[...,:3,1]*self.offset_m
        return poses,known


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--scene-data',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True); parser.add_argument('--surface-run',type=Path,required=True)
    parser.add_argument('--scene',required=True); parser.add_argument('--lateral-m',type=float,default=2.)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False); started=time.monotonic(); torch.set_num_threads(2)
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('status.json',{'status':'running','phase':'load'})
    try:
        scene=next(s for s in json.loads((args.scene_data/'index.json').read_text())['scenes'] if s['scene']==args.scene)
        selected=[entry for entry in scene['cohort'] if (entry.get('translation_speed_mps') or 0)>2]
        save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'scene_data':str(args.scene_data),'actor_data':str(args.actor_data),'surface_run':str(args.surface_run),
            'scene':args.scene,'role':scene['role'],'selection':'all build-metadata cohort Actors with known translation speed >2m/s',
            'selected_owners':[e['owner'] for e in selected],'edit':'one Actor at a time, +Y translation in its instantaneous local rigid frame',
            'lateral_m':args.lateral_m,'canonical_geometry':'loaded once and reused in the same BVH; positions, normals, faces and appearance not optimized',
            'composition':'unchanged background and other Actor layers, global nearest intersection on same per-return-time beams',
            'data_boundary':'query directions are the original observed-return beam subset; unavailable no-return beams are not synthesized',
            'counterfactual_ground_truth':False,'accuracy_metrics_computed':False,'optimizer_updates':0,
            'unknown_background':'released rays without a stored background/other-Actor intersection stay unknown; no inpainting or hidden ground-truth fill',
            'claim':'mechanical rigid composition demonstration, not edited-scene realism, collision-free motion or learned-method superiority',
            'references':['https://github.com/zju3dv/street_gaussians',
                'https://openaccess.thecvf.com/content/CVPR2024/papers/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.pdf']})
        folder=args.scene_data/args.scene; bg=np.load(folder/'background.npz')
        background=SurfaceBVH(bg['vertices_world_m'],bg['faces']); actors={}; trajectories={}; sizes={}; centers={}
        for entry in scene['cohort']:
            owner=entry['owner']; path=args.surface_run/(args.scene+'__'+owner+'_surface.pt')
            if not path.is_file(): path=args.surface_run/(owner+'_surface.pt')
            surface=torch.load(path,map_location='cpu',weights_only=False)
            actors[owner]=SurfaceBVH(surface['vertices_actor_m'].numpy(),surface['faces'].numpy())
            centers[owner]=surface['centers_actor_m'].numpy(); del surface
            value=scene['actor_trajectories'][owner]
            trajectories[owner]=TimedPoses(value['timestamps_ns'],value['world_from_actor'])
            case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=False)
            sizes[owner]=case['size_lwh_m'].tolist(); del case
        rows=[]; previews=[]
        for frame in scene['frames']:
            raw=np.load(folder/frame['file']); origins=raw['origin_world_m']; directions=raw['directions_world']
            times=raw['point_timestamps_ns']; sensor_known=raw['sensor_pose_known']; n=len(directions)
            bg_depth=np.full(n,np.inf,np.float32); bg_depth[sensor_known]=background.cast(origins[sensor_known],directions[sensor_known])
            depths={owner:actors[owner].cast_per_time(origins,directions,times,trajectory,sensor_known)[0]
                    for owner,trajectory in trajectories.items()}
            base_depth,base_owner,_=compose_first(bg_depth,[(scene['owner_ids'][owner],depth) for owner,depth in depths.items()])
            for entry in selected:
                owner=entry['owner']; owner_id=scene['owner_ids'][owner]
                edited= LateralTrajectory(trajectories[owner],args.lateral_m)
                actor_depth,known=actors[owner].cast_per_time(origins,directions,times,edited,sensor_known)
                after_depth,after_owner,_=compose_first(bg_depth,[(scene['owner_ids'][key],actor_depth if key==owner else depth)
                                                                  for key,depth in depths.items()])
                before_visible=base_owner==owner_id; after_visible=after_owner==owner_id
                released=before_visible&~after_visible; both=np.isfinite(base_depth)&np.isfinite(after_depth)
                difference=np.abs(after_depth[both]-base_depth[both])
                row={'owner':owner,'sample_index':frame['sample_index'],'raw_query_rays':n,'known_actor_pose_rays':int(known.sum()),
                    'before_actor_first':int(before_visible.sum()),'after_actor_first':int(after_visible.sum()),
                    'introduced_occlusions':int((after_visible&~before_visible).sum()),'released_rays':int(released.sum()),
                    'released_to_background':int((released&(after_owner==0)).sum()),'released_to_unknown':int((released&(after_owner==-1)).sum()),
                    'released_to_other_actor':int((released&(after_owner>0)).sum()),
                    'edited_intersections_hidden_by_other_layers':int((np.isfinite(actor_depth)&~after_visible).sum()),
                    'changed_owner_rays':int((base_owner!=after_owner).sum()),
                    'depth_changed_gt_1mm_among_both_finite':int((difference>.001).sum()),
                    'mean_abs_depth_change_both_finite_m':float(difference.mean()) if len(difference) else None,
                    'counterfactual_accuracy':None}
                rows.append(row)
                file=f'{owner}__{frame["sample_index"]}.npz'
                np.savez(args.output/file,before_depth_m=base_depth,after_depth_m=after_depth,before_owner=base_owner,after_owner=after_owner)
                if frame==scene['frames'][0]:
                    reference,pose_known=trajectories[owner].at(frame['timestamp_ns'])
                    preview={'owner':owner,'owner_id':owner_id,'build_points':entry['build_points'],
                        'speed_mps':entry['translation_speed_mps'],'size_lwh_m':sizes[owner],'file':file,
                        'world_from_reference_actor':reference.tolist(),'reference_pose_known':bool(pose_known),
                        'canonical_centers':centers[owner].tolist(),'statistics':row}
                    previews.append(preview)
                save('status.json',{'status':'running','phase':'rigid_edit','completed_cases':len(rows),'elapsed_s':time.monotonic()-started})
        save('preview.json',{'scene':args.scene,'source_frame':scene['frames'][0],'source_scene_folder':str(folder),
                             'lateral_m':args.lateral_m,'actors':previews})
        result={'status':'done','scene':args.scene,'role':scene['role'],'edited_actors':selected,'frames':rows,
            'total_raw_queries_per_edit':sum(frame['rays'] for frame in scene['frames']),
            'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'boundary':'no counterfactual ground truth, accuracy or physical-feasibility claim; fixed geometry/known rigid time composition only'}
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps({k:v for k,v in result.items() if k not in ['edited_actors','frames']}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
