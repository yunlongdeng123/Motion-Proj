"""Inventory zero-LiDAR Actors using existing inputs and completed fusion outputs.

Camera frustum overlap is a conservative geometric possibility, not visibility.
No network inference, new ray casting, model updates or quality-based selection.
"""
import argparse,json,resource,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]


def camera_box_overlap(case):
    poses=case['camera_from_actor']; calibration=case['intrinsics']
    if not len(poses): return []
    corners=torch.cartesian_prod(torch.tensor([-1.,1.]),torch.tensor([-1.,1.]),torch.tensor([-1.,1.]))
    corners=corners*case['size_lwh_m']/2
    camera=torch.einsum('vij,pj->vpi',poses[:,:3,:3],corners)+poses[:,:3,3][:,None]
    projected=torch.einsum('vij,vpj->vpi',calibration,camera)
    z=camera[...,2]; h,w=case['image_hw']
    rect=case.get('valid_image_rect_xyxy')
    rect=torch.as_tensor(rect) if rect is not None else torch.tensor([[0,0,w,h]]*len(poses))
    planes=torch.stack([z-.05,projected[...,0]-rect[:,0,None]*z,
        (rect[:,2,None]-1)*z-projected[...,0],projected[...,1]-rect[:,1,None]*z,
        (rect[:,3,None]-1)*z-projected[...,1]],-1)
    return (planes.max(1).values>=0).all(-1).tolist()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--fusion-run',type=Path,required=True)
    parser.add_argument('--fit-targets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); tick=time.monotonic()
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'actor_data':str(args.actor_data),'fusion_run':str(args.fusion_run),'fit_targets':str(args.fit_targets),
        'selection':'all original cohort Actors with zero build LiDAR measurements, by existing metadata',
        'camera_overlap':'annotation box corners against 5 calibrated image frustum planes; conservative possibility, no occlusion claim',
        'native_support':'already saved M1r3 native fusion result, not a new inference or correctness oracle',
        'targets':'FIT full-track label availability counted only; no added prediction input or development labels',
        'optimizer_updates':0,'external_data_read':False})
    save('status.json',{'status':'running'})
    entries=json.loads((args.actor_data/'index.json').read_text())['cases']
    fusion=json.loads((args.fusion_run/'summary.json').read_text())['final']
    lookup={(r['actor']['scene'],r['actor']['owner']):r for r in fusion}
    rows=[]
    for entry in entries:
        if entry['build_points']!=0: continue
        case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=True)
        match=lookup[(entry['scene'],entry['owner'])]; support=match['seed_support']
        build=[r for r in case['rays'] if r['role']=='build']
        row={'actor':entry,'camera_poses':len(case['view_indices']),
             'camera_frustum_overlap':camera_box_overlap(case),
             'build_near_box_rays':sum(len(r['observed_first_range_m']) for r in build),
             'build_owned_rays':sum(int(r['positive_actor'].sum()) for r in build),
             'native_candidates':support['native_candidates'],'native_surface_patches':match['surface_patches'],
             'native_prediction_unavailable':support['prediction_unavailable'],
             'native_support_per_view':support['per_view_native_support'],
             'fit_full_track_target_points':None,'fit_full_track_owned_rays':None,'fit_full_track_near_box_rays':None}
        if entry['role']=='fit':
            target=torch.load(args.fit_targets/(entry['scene']+'__'+entry['owner']+'.pt'),map_location='cpu',weights_only=True)
            row['fit_full_track_target_points']=len(target['target_points_actor_m'])
            row['fit_full_track_owned_rays']=sum(int(r['positive_actor'].sum()) for r in target['target_rays'])
            row['fit_full_track_near_box_rays']=sum(len(r['observed_first_range_m']) for r in target['target_rays'])
        rows.append(row)
    stats={}
    for role in sorted({r['actor']['role'] for r in rows}):
        subset=[r for r in rows if r['actor']['role']==role]
        stats[role]={'actors':len(subset),'logs':len({r['actor']['log_id'] for r in subset}),
            'with_camera_pose':sum(r['camera_poses']>0 for r in subset),
            'with_frustum_overlap':sum(any(r['camera_frustum_overlap']) for r in subset),
            'with_native_candidates':sum(r['native_candidates']>0 for r in subset),
            'with_native_surface':sum(r['native_surface_patches']>0 for r in subset),
            'with_build_near_box_rays':sum(r['build_near_box_rays']>0 for r in subset),
            'build_near_box_rays':sum(r['build_near_box_rays'] for r in subset),
            'heldout_owned_rays':sum(r['actor']['heldout_actor_returns'] for r in subset),
            'with_heldout_owned_rays':sum(r['actor']['heldout_actor_returns']>0 for r in subset),
            'with_fit_full_track_target':sum((r['fit_full_track_target_points'] or 0)>0 for r in subset) if role=='fit' else None,
            'fit_target_points':sum(r['fit_full_track_target_points'] or 0 for r in subset) if role=='fit' else None}
    result={'status':'done','rows':rows,'statistics':stats,'wall_s':time.monotonic()-tick,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'boundary':'metadata/input availability audit; saved native candidates do not imply correct Actor surface or real visibility'}
    save('summary.json',result); save('status.json',{'status':'done'})
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__': main()
