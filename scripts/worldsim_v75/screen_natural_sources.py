"""固定八日志先筛可见性与真实参考支持，不加载重建/生成输出。"""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from prepare_argoverse import ROOT, LOG, CAMERA, QUAT, POS, CORNERS, poses, crop_image

OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATURAL-SOURCES-01/20260920-r1')

def track_pose(annotations,ego_df,track_id,query,origin):
    df=annotations[annotations.track_uuid==track_id].sort_values('timestamp_ns')
    if len(df)<2 or min(query)<df.timestamp_ns.min() or max(query)>df.timestamp_ns.max():return None
    times=df.timestamp_ns.to_numpy(np.int64)
    # 不能跨缺失标注外插/插值。
    for q in query:
        j=np.searchsorted(times,q)
        if j>0 and j<len(times) and times[j]-times[j-1]>150_000_000:return None
    E=poses(ego_df,times); local=np.repeat(np.eye(4)[None],len(df),0)
    local[:,:3,:3]=Rotation.from_quat(df[QUAT].to_numpy()).as_matrix();local[:,:3,3]=df[POS].to_numpy()
    world=E@local;world[:,:3,3]-=origin
    packed=pd.DataFrame({'timestamp_ns':times});packed[POS]=world[:,:3,3];packed[QUAT]=Rotation.from_matrix(world[:,:3,:3]).as_quat()
    return poses(packed,query),df[['length_m','width_m','height_m']].iloc[0].to_numpy(float)

def main():
    assert not OUT.exists(),'拒绝重复筛查/覆盖冻结分母'
    logs=[d.name for d in sorted(ROOT.iterdir()) if d.is_dir() and d.name!=LOG and
          len(list((d/'sensors/cameras'/CAMERA).glob('*.jpg')))>=200 and (d/'annotations.feather').exists()][:8]
    assert len(logs)==8
    OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-NATURAL-SOURCES-01','run_id':'20260920-r1',
              'frozen_utc':datetime.now(timezone.utc).isoformat(),'logs':logs,
              'role':'development_screen; prior exposure not excluded; not independent confirmation',
              'selection':'first eight lexicographic locally complete AV2 val logs excluding previous bridge',
              'start':'first front RGB timestamp >= first RGB +0.5 seconds, fixed once per log',
              'target':'REGULAR_VEHICLE; full projected box in 1280x704 crop at 0,.5,1,1.5,2 seconds; initial >=48x32px and camera depth 5..60m',
              'reference':'latest sweep <= start; per-point timestamp <= start; >=6 points inside target 3D cuboid(+0.15m) and central60% projected box',
              'priority':'within each log lexicographic eligible track UUID; inspect real RGB for occlusion before reconstruction',
              'stop':'all eight exhausted without eligible target: report data gap, do not loosen thresholds or add logs in this screen',
              'models_loaded':False,'human_verdict':None}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    rows=[];sheet=Image.new('RGB',(1280,8*382),'#111827')
    for index,log in enumerate(logs):
        raw=ROOT/log;images=sorted((raw/'sensors/cameras'/CAMERA).glob('*.jpg'))
        initial=next(im for im in images if int(im.stem)>=int(images[0].stem)+500_000_000);start=int(initial.stem)
        query=start+np.rint(np.arange(5)*.5e9).astype(np.int64)
        ego_df=pd.read_feather(raw/'city_SE3_egovehicle.feather');ego=poses(ego_df,query);origin=ego[0,:3,3].copy();ego[:,:3,3]-=origin
        cal=pd.read_feather(raw/'calibration/egovehicle_SE3_sensor.feather').set_index('sensor_name').loc[CAMERA]
        ext=np.eye(4);ext[:3,:3]=Rotation.from_quat(cal[QUAT].to_numpy(float)).as_matrix();ext[:3,3]=cal[POS].to_numpy(float)
        cams=ego@ext
        intr=pd.read_feather(raw/'calibration/intrinsics.feather').set_index('sensor_name').loc[CAMERA]
        top=(intr.height_px-intr.width_px*704/1280)/2;crop=[0,top,float(intr.width_px),top+intr.width_px*704/1280];s=1280/intr.width_px
        K=np.array([intr.fx_px*s,intr.fy_px*s,(intr.cx_px+.5)*s-.5,(intr.cy_px-top+.5)*s-.5])
        ann=pd.read_feather(raw/'annotations.feather')
        lidar_path=max((x for x in (raw/'sensors/lidar').glob('*.feather') if int(x.stem)<=start),key=lambda x:int(x.stem))
        lidar=pd.read_feather(lidar_path);keep=int(lidar_path.stem)+lidar.offset_ns.to_numpy(np.int64)<=start
        LE=poses(ego_df,[int(lidar_path.stem)])[0];LE[:3,3]-=origin
        points=lidar[['x','y','z']].to_numpy(float)[keep]@LE[:3,:3].T+LE[:3,3]
        pc=(points-cams[0,:3,3])@cams[0,:3,:3]
        uv=pc[:,:2]/np.where(abs(pc[:,2:])>1e-8,pc[:,2:],np.nan)*K[:2]+K[2:]
        candidates=[];rejects={};rgb=crop_image(initial,crop);overlay=rgb.copy();draw=ImageDraw.Draw(overlay)
        ids=sorted(ann[ann.category=='REGULAR_VEHICLE'].track_uuid.unique())
        for track in ids:
            found=track_pose(ann,ego_df,track,query,origin)
            if found is None:reason='missing_two_second_track'
            else:
                transforms,dims=found;corners=(CORNERS*dims)[None]@np.transpose(transforms[:,:3,:3],(0,2,1))+transforms[:,None,:3,3]
                camera_points=(corners-cams[:,None,:3,3])@cams[:,:3,:3]
                proj=camera_points[...,:2]/np.where(abs(camera_points[...,2:])>1e-8,camera_points[...,2:],np.nan)*K[:2]+K[2:]
                bounds=np.concatenate([proj.min(1),proj.max(1)],axis=1);b=bounds[0];depth=float(camera_points[0,:,2].mean())
                if (camera_points[...,2]<=.1).any() or not ((bounds[:,:2]>=0).all() and (bounds[:,2]<1280).all() and (bounds[:,3]<704).all()):reason='not_fully_visible_two_seconds'
                elif not (5<=depth<=60 and b[2]-b[0]>=48 and b[3]-b[1]>=32):reason='small_or_out_of_depth_range'
                else:
                    at_lidar=track_pose(ann,ego_df,track,[int(lidar_path.stem)],origin)
                    if at_lidar is None:reason='missing_lidar_time_track'
                    else:
                        T,_=at_lidar;q=(points-T[0,:3,3])@T[0,:3,:3]
                        inside=np.all(abs(q)<=dims/2+.15,axis=1)
                        center=(b[:2]+b[2:])/2;half=(b[2:]-b[:2])*.3
                        support=inside&(pc[:,2]>.1)&np.all(uv>=center-half,axis=1)&np.all(uv<=center+half,axis=1)
                        n=int(support.sum());eligible=n>=6
                        candidates.append({'track':track,'depth_m':depth,'initial_box':b.tolist(),
                                           'all_boxes':bounds.tolist(),'target_lidar_points':int(inside.sum()),
                                           'target_core_lidar_points':n,'eligible':eligible})
                        draw.rectangle(b.tolist(),outline='#55ee88' if eligible else '#f0bb55',width=3)
                        draw.text((b[0],max(0,b[1]-16)),f'{track[:6]} n={n}',fill='white')
                        for point in uv[support]:draw.ellipse((point[0]-1,point[1]-1,point[0]+1,point[1]+1),fill='red')
                        if eligible:continue
                        reason='insufficient_core_lidar'
            rejects[reason]=rejects.get(reason,0)+1
        rgb.save(OUT/f'{log}-rgb.jpg',quality=94);overlay.save(OUT/f'{log}-support.jpg',quality=94)
        eligible=[c for c in candidates if c['eligible']]
        row={'log_id':log,'start_ns':start,'start_offset_s':(start-int(images[0].stem))/1e9,
             'initial_rgb':str(initial),'lidar_file':str(lidar_path),'per_point_causal_lidar':int(keep.sum()),
             'candidate_tracks_total':len(ids),'rejection_counts':rejects,'geometric_candidates':candidates,
             'eligible_target_count':len(eligible),'selected_target':eligible[0]['track'] if eligible else None,
             'rgb_occlusion_verification':'pending','human_verdict':None}
        rows.append(row);(OUT/'screen_partial.json').write_text(json.dumps(rows,indent=2)+'\n')
        y=index*382;ImageDraw.Draw(sheet).text((8,y+5),f'{log[:8]} | start={row["start_offset_s"]:.2f}s | eligible={len(eligible)} | red: causal target LiDAR',fill='white')
        sheet.paste(rgb.resize((640,352)),(0,y+30));sheet.paste(overlay.resize((640,352)),(640,y+30))
        print(json.dumps({'log_id':log,'eligible':len(eligible),'selected':row['selected_target']}),flush=True)
    sheet.save(OUT/'source-screen.jpg',quality=94)
    result={'status':'complete','logs':rows,'screened_logs':8,'logs_with_eligible_targets':sum(r['selected_target'] is not None for r in rows),
            'world_model_calls':0,'reconstruction_calls':0,'human_verdict':None,'failure_ledger_delta':'none',
            'interpretation':'只确认参考与几何可见性；不是模型失败分母，人工/独立确认未完成'}
    (OUT/'screen_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
