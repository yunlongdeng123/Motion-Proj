"""在查看Ω预测前按GT与文件可用性固定P0开发样本。"""
import argparse, datetime, json, pathlib, subprocess
import numpy as np
from PIL import Image
from geometry import project_bbox, resized_intrinsics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=json.loads(pathlib.Path(args.config).read_text());out=pathlib.Path(cfg['run_dir'])
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'registration.json').exists(), '不可覆盖预登记'
    rows=[]
    for spec in cfg['scenes']:
        root=pathlib.Path(spec['root']);frame=cfg['frame'];views=[]
        for camera in range(6):
            image=root/'images'/f'{frame:03d}_{camera}.jpg'
            wh=Image.open(image).size
            vals=np.loadtxt(root/'intrinsics'/f'{camera}.txt')
            k=np.array([[vals[0],0,vals[2]],[0,vals[1],vals[3]],[0,0,1]])
            views.append({'camera':camera,'image':str(image),'original_wh':list(wh),'intrinsics':k.tolist(),
                          'c2w':np.loadtxt(root/'extrinsics'/f'{frame:03d}_{camera}.txt').tolist()})
        instances=json.loads((root/'instances/instances_info.json').read_text());actors=[];all_boxes=[]
        origin=np.mean([np.asarray(v['c2w'])[:3,3] for v in views],axis=0)
        for actor_id,record in instances.items():
            a=record['frame_annotations']
            if frame not in a['frame_idx']:continue
            j=a['frame_idx'].index(frame);pose=np.asarray(a['obj_to_world'][j]);size=a['box_size'][j]
            box={'actor_id':str(actor_id),'track_id':record['id'],'category':record['class_name'],'pose':pose.tolist(),'size_lwh':size}
            all_boxes.append(box)
            if not record['class_name'].startswith(('vehicle.car','vehicle.truck','vehicle.bus','vehicle.trailer','vehicle.construction')):continue
            distance=float(np.linalg.norm(pose[:3,3]-origin))
            projections=[project_bbox(pose,size,np.array(v['c2w']),resized_intrinsics(v['intrinsics'],v['original_wh'],[384,688]),[384,688]) for v in views]
            areas=[(b[2]-b[0])*(b[3]-b[1]) if b else 0 for b in projections]
            if distance>45 or max(areas)<256:continue
            box=dict(box,distance_m=distance,gt_projected_boxes=projections,primary_camera=int(np.argmax(areas)),gt_frustum_cameras=sum(v>0 for v in areas))
            actors.append(box)
        actors=sorted(actors,key=lambda x:(x['distance_m'],x['actor_id']))
        print(spec['name'],'eligible_GT_vehicles',len(actors),flush=True)
        rows.append(dict(spec,frame=frame,views=views,origin_world=origin.tolist(),actors=actors,all_boxes=all_boxes,
                         camera_timestamp_note='同一processed 10Hz帧；原始camera timestamp未随此导出保留，不宣称精确同步'))
    candidates=sorted([(a['distance_m'],s['name'],a['actor_id']) for s in rows for a in s['actors']])
    assert len(candidates)>=cfg['actor_count'],len(candidates)
    selected={(s,i) for _,s,i in candidates[:cfg['actor_count']]}
    for s in rows:s['actors']=[a for a in s['actors'] if (s['name'],a['actor_id']) in selected]
    record=dict(cfg,scenes=rows,registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        actual_actor_count=sum(len(x['actors']) for x in rows),input_role='re-exposed development, not independent test',
        selection='GT类别、完整图像、距离<=45m、GT投影>=256像素；三个场景合并按距离选最近24车，预测前固定，不保证无遮挡',
        failure_ledger_refs=['V76-F03'],failure_ledger_delta='none',human_verdict=None)
    (out/'registration.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'run':str(out),'actors':record['actual_actor_count'],'selection':[[x['name'],[a['actor_id'] for a in x['actors']]] for x in rows]},ensure_ascii=False))

if __name__=='__main__':main()
