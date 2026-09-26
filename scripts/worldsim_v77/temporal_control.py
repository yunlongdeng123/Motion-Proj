"""固定0/20/40帧的GT轨迹规范坐标累积：只用已有解析算子，无学习或对象后处理。"""
import argparse,datetime,json,pathlib
import numpy as np
import torch
from PIL import Image
from geometry import box_mask,edit_actor,resized_intrinsics,target_pose,transform
from evaluate import camera_looking,command_list,dump,labeled,lidar_recall,read_geometry,render


def prepare(base,root):
    original=json.loads((base/'registration.json').read_text());reg=dict(original)
    reg.update(task_id='WS-V77-P0-TEMPORAL-20260926',run_dir=str(root),scenes=[],
        registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_run=str(base),frames=[0,20,40],baseline_prediction_reused_frame=20,
        hypothesis='固定三个时刻的GT刚体规范坐标累积能否补充可见表面；不学习、不调阈值、不填未见面',
        metric_caveat='点集并集使最近距离召回机械地不下降，不能据此单独声称质量改善；必须查看多视角形状及混入物',
        input_role='原24对象继续开发控制，额外0/40帧RGB及GT轨迹、标定、同帧背景LiDAR尺度',
        failure_ledger_delta='none')
    for scene in original['scenes']:
        data=pathlib.Path(scene['root']);instances=json.loads((data/'instances/instances_info.json').read_text())
        for frame in [0,40]:
            s=dict(scene,name=scene['name']+f'_f{frame:03d}',base_scene=scene['name'],frame=frame,views=[],actors=[],all_boxes=[])
            for camera in range(6):
                v=dict(scene['views'][camera],image=str(data/'images'/f'{frame:03d}_{camera}.jpg'),c2w=np.loadtxt(data/'extrinsics'/f'{frame:03d}_{camera}.txt').tolist())
                assert pathlib.Path(v['image']).exists();s['views'].append(v)
            s['origin_world']=np.mean([np.asarray(v['c2w'])[:3,3] for v in s['views']],0).tolist()
            for actor_id,record in instances.items():
                annotation=record['frame_annotations']
                if frame not in annotation['frame_idx']:continue
                j=annotation['frame_idx'].index(frame)
                box={'actor_id':actor_id,'track_id':record['id'],'category':record['class_name'],'pose':annotation['obj_to_world'][j],'size_lwh':annotation['box_size'][j]}
                s['all_boxes'].append(box)
                if actor_id in [a['actor_id'] for a in scene['actors']]:s['actors'].append(box)
            reg['scenes'].append(s)
    root.mkdir(parents=True,exist_ok=True);assert not (root/'registration.json').exists()
    dump(root/'registration.json',reg);print('Registered six additional frozen forward passes',flush=True)


def evaluate(base,root):
    torch.set_num_threads(4);reg=json.loads((root/'registration.json').read_text());original=json.loads((base/'registration.json').read_text())
    out=root/'evaluation';out.mkdir(exist_ok=True);all_rows=[]
    for scene in original['scenes']:
        name=scene['name'];sceneout=out/name;sceneout.mkdir(exist_ok=True);origin=np.array(scene['origin_world'])
        additions={a['actor_id']:[] for a in scene['actors']}
        for extra in [s for s in reg['scenes'] if s['base_scene']==name]:
            d=root/extra['name'];pred=np.load(d/'prediction.npz')
            variants,_,_,_,_,_=read_geometry(extra,pred,d)
            points,colors,_=variants['calibrated_control'];extraorigin=np.array(extra['origin_world'])
            for a in extra['actors']:
                pose=np.array(a['pose']);pose[:3,3]-=extraorigin;mask=box_mask(points,pose,a['size_lwh'])
                additions[a['actor_id']].append({'frame':extra['frame'],'track_id':a['track_id'],'local':transform(points[mask],np.linalg.inv(pose)),'colors':colors[mask]})
            del variants,points,colors,pred
        base_eval=base/'evaluation_v2'/name;cloud=np.load(base_eval/'calibrated_control/scene_points.npz')
        basepoints=cloud['points_local_world'].astype(np.float64);basecolors=cloud['colors']
        bp=np.load(base/name/'prediction.npz');rgb=np.rint(np.moveaxis(bp['images'],1,-1)*255).astype(np.uint8);h,w=rgb.shape[1:3]
        cams=np.array([v['c2w'] for v in scene['views']]);cams[:,:3,3]-=origin
        ks=np.array([resized_intrinsics(v['intrinsics'],v['original_wh'],[h,w]) for v in scene['views']])
        data=pathlib.Path(scene['root']);lidar=transform(np.fromfile(data/'lidar/020.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(data/'lidar_pose/020.txt'))-origin
        for actor in scene['actors']:
            aid=actor['actor_id'];ad=sceneout/('actor_'+aid);ad.mkdir(exist_ok=True)
            src=base_eval/'calibrated_control'/('actor_'+aid);asset=np.load(src/'actor.npz');old=json.loads((src/'metrics.json').read_text())
            parts=[asset['local_points']];colors=[asset['colors']];counts={'20':len(parts[0])}
            for addition in additions[aid]:
                assert addition['track_id']==actor['track_id']
                parts.append(addition['local']);colors.append(addition['colors']);counts[str(addition['frame'])]=len(addition['local'])
            local=np.concatenate(parts).astype(np.float64);color=np.concatenate(colors)
            pose=np.array(actor['pose']);pose[:3,3]-=origin;size=np.array(actor['size_lwh'])
            world=transform(local,pose);original_mask=box_mask(basepoints,pose,size)
            points=np.concatenate([basepoints[~original_mask],world]);allcolors=np.concatenate([basecolors[~original_mask],color])
            selected=np.arange(len(points))>=len(points)-len(world)
            reference=lidar[box_mask(lidar,pose,size)];recall,median=lidar_recall(reference,world)
            metrics={'scene':name,'actor_id':aid,'track_id':actor['track_id'],'source_point_counts':counts,'total_points':len(local),
                'single_time_points':old['point_count'],'single_time_recall':old['visible_lidar_recall_0p2m'],
                'temporal_recall':recall,'lidar_median_nearest_m':median,'missing_requested_frames':sorted(set([0,20,40])-set(map(int,counts))),
                'source_geometry_extent_lwh':old['geometry_extent_lwh'],'temporal_geometry_extent_lwh':np.ptp(local,axis=0).tolist(),
                'reference_box_size_lwh':size.tolist(),'frames_with_points':sum(n>0 for n in counts.values()),'commands':[],
                'human_verdict':None,'no_dedup_no_fill':True}
            np.savez_compressed(ad/'actor_temporal.npz',local_points=local.astype(np.float32),colors=color,pose_local_world=pose)
            for op,xyz,yaw in command_list(reg['edits']):
                target=target_pose(pose,xyz,yaw);p,c,ids=edit_actor(points,allcolors,selected,pose,target,op)
                bg=~selected[ids];bg_error=float(np.max(np.abs(p[bg]-points[ids[bg]])))
                error=None
                if op!='DELETE':
                    transformed=p[selected] if op=='MOVE' else p[len(points):]
                    error=float(np.linalg.norm(transformed-transform(local,target),axis=1).max())
                    assert error<1e-8
                expected_count=len(points)-len(local) if op=='DELETE' else len(points)+len(local) if op=='INSERT' else len(points)
                assert len(p)==expected_count and bg_error==0
                metrics['commands'].append({'operation':op,'delta_world_m':xyz,'yaw_deg':yaw,'point_adherence_error_m':error,'background_change_m':bg_error,'output_point_count':len(p)})
                del p,c,ids
            camera=actor['primary_camera'];images=[rgb[camera],render(points,allcolors,cams[camera],ks[camera],(h,w))[0]]
            for op,xyz,yaw in [('MOVE',[3,0,0],15),('DELETE',[0,0,0],0),('INSERT',[3,0,0],0)]:
                p,c,_=edit_actor(points,allcolors,selected,pose,target_pose(pose,xyz,yaw),op)
                images.append(render(p,c,cams[camera],ks[camera],(h,w))[0]);del p,c
            b=actor['gt_projected_boxes'][camera];lo=np.maximum(np.array(b[:2])-35,0).astype(int);hi=np.minimum(np.array(b[2:])+35,[w,h]).astype(int)
            labeled([a[lo[1]:hi[1],lo[0]:hi[0]] for a in images],['Input RGB','3-time factual','MOVE','DELETE','INSERT'],250).save(ad/'edit_crop.jpg',quality=92)
            canon=[];k=np.array([[220,0,127.5],[0,220,127.5],[0,0,1.]])
            for direction in [(1,0,.2),(-1,0,.2),(0,1,.2),(0,-1,.2),(0,0,1)]:
                eye=np.asarray(direction,dtype=float);eye=eye/np.linalg.norm(eye)*max(size)*1.6
                canon.append(render(local,color,camera_looking(eye,[0,0,0]),k,(256,256))[0])
            labeled(canon,['Front','Back','Left','Right','Top'],256).save(ad/'canonical.jpg',quality=92)
            dump(ad/'metrics.json',metrics);all_rows.append(metrics)
            print(json.dumps({'scene':name,'actor':aid,'counts':counts,'single':metrics['single_time_recall'],'temporal':recall}),flush=True)
        torch.cuda.empty_cache()
    summary={'actors':len(all_rows),'commands':sum(len(r['commands']) for r in all_rows),
             'macro_single_time_recall':float(np.mean([r['single_time_recall'] for r in all_rows])),
             'macro_temporal_recall':float(np.mean([r['temporal_recall'] for r in all_rows])),
             'actors_with_missing_gt_frames':sum(bool(r['missing_requested_frames']) for r in all_rows),
             'actors_with_all_three_nonempty_frames':sum(r['frames_with_points']==3 for r in all_rows),
             'max_adherence_error_m':max(c['point_adherence_error_m'] or 0 for r in all_rows for c in r['commands']),
             'max_background_change_m':max(c['background_change_m'] for r in all_rows for c in r['commands']),
             'claim_boundary':'Recall increase is monotonic for a union; no completeness/purity success claimed from it alone'}
    dump(out/'all_actor_metrics.json',all_rows);dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','evaluate']);p.add_argument('--base-run',required=True);p.add_argument('--run-dir',required=True);a=p.parse_args()
    (prepare if a.phase=='prepare' else evaluate)(pathlib.Path(a.base_run),pathlib.Path(a.run_dir))
