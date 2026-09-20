"""冻结生成起点前的真实环视输入和有额外几何先验的定位诊断。"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
from prepare_argoverse import OUT as BASE, ROOT, LOG, QUAT, POS, poses

OUT = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATURAL-01/20260920-r1')
TARGET = '94dede14-59da-4f09-b016-95f19596ac08'

def main():
    global OUT, BASE, TARGET
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--base-dir',type=Path,default=BASE);parser.add_argument('--target',default=TARGET)
    parser.add_argument('--task-id',default='WS-V75-NATURAL-01')
    parser.add_argument('--source-protocol',type=Path)
    a=parser.parse_args();OUT,BASE,TARGET=a.output,a.base_dir,a.target
    assert not OUT.exists(), '拒绝覆盖冻结目录'
    base=json.loads((BASE/'input_manifest.json').read_text())
    raw=ROOT/base['log_id']
    cutoff=base['first_timestamp_ns']
    cal=pd.read_feather(raw/'calibration/egovehicle_SE3_sensor.feather').set_index('sensor_name')
    intr=pd.read_feather(raw/'calibration/intrinsics.feather').set_index('sensor_name')
    ego=pd.read_feather(raw/'city_SE3_egovehicle.feather')
    origin=np.asarray(base['city_origin'])
    views=[]
    for folder in sorted((raw/'sensors/cameras').glob('ring_*')):
        candidates=[p for p in folder.glob('*.jpg') if int(p.stem)<=cutoff]
        assert candidates
        image=max(candidates,key=lambda p:int(p.stem)); stamp=int(image.stem)
        assert cutoff-stamp<50_000_000
        w,h=Image.open(image).size
        nh=round(h*(512/w)/16)*16; cropped=max(0,(nh-512)//2)
        pad=max(0,(512-min(nh,512))//2)
        A=np.array([[512/w,0,(512/w-1)/2],[0,nh/h,(nh/h-1)/2-cropped+pad],[0,0,1]])
        c=intr.loc[folder.name]
        K=np.array([[c.fx_px,0,c.cx_px],[0,c.fy_px,c.cy_px],[0,0,1]])
        ext=np.eye(4);ext[:3,:3]=Rotation.from_quat(cal.loc[folder.name,QUAT].to_numpy(float)).as_matrix()
        ext[:3,3]=cal.loc[folder.name,POS].to_numpy(float)
        E=poses(ego,[stamp])[0];E[:3,3]-=origin
        views.append({'camera':folder.name,'image':str(image),'timestamp_ns':stamp,
                      'delta_to_cutoff_ms':(stamp-cutoff)/1e6,'original_wh':[w,h],
                      'resized_wh':[512,nh],'crop_top':cropped,'pad_top':pad,
                      'raw_to_network':A.tolist(),'K_raw':K.tolist(),'K_network':(A@K).tolist(),
                      'camera_world':(E@ext).tolist(),'ego_world':E.tolist()})
    assert len(views)==7 and views[0]['camera']=='ring_front_center'
    target=next(t for t in json.loads((BASE/'scene.json').read_text())['tracks'] if t['id']==TARGET)
    if (BASE/'evaluation.json').exists():
        detection=json.loads((BASE/'evaluation.json').read_text())['frames'][0]['real']
    else:
        from evaluate_localization import model,predict,match
        projection=next(x for x in json.loads((BASE/'projections.json').read_text()) if x['id']==TARGET)['projections'][0]['bounds']
        detection=match(predict(model(),np.asarray(Image.open(BASE/'initial_rgb.png'))),projection)
    assert detection is not None
    b=np.asarray(detection['box']); crop=base['crop_xyxy']; scale=1280/base['source_image_size'][0]
    original=(b+0.5)/scale-0.5+np.array([crop[0],crop[1],crop[0],crop[1]])
    A=np.array(views[0]['raw_to_network']); net=(original.reshape(2,2)@A[:2,:2].T+A[:2,2]).ravel()
    lidar=max((p for p in (raw/'sensors/lidar').glob('*.feather') if int(p.stem)<=cutoff),key=lambda p:int(p.stem))
    protocol={'task_id':a.task_id,'run_id':'20260920-r1',
              'case_id':base['log_id'],'source_task_id':base['task_id'],
              'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'role':'previously_exposed_development_log_conditional_readout_diagnostic',
              'base_run':str(BASE),'log_id':base['log_id'],'cutoff_ns':cutoff,'views':views,
              'model':'official DVGT-1','repository':'/root/autodl-tmp/external/worldsim_v81/DVGT',
              'revision':'51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236',
              'weights':'/root/autodl-tmp/models/worldsim_v81/dvgt1.pt',
              'preprocess':'official load_and_preprocess_images(mode=crop), one timestep/seven ring views',
              'metric_conversion':'official dataset gt_scale_factor=0.1; divide native points by 0.1; ego RDF to FLU',
              'target':TARGET,'target_selection':'inherited geometric-visibility selection, not ranked by reconstruction error',
              'target_detection':detection,'bbox_original':original.tolist(),'bbox_network':net.tolist(),
              'target_dimensions_oracle':target['dimensions'],
              'target_rotation_world_oracle':Rotation.from_quat(target['quaternions'][0]).as_matrix().tolist(),
              'input_roles':{'reconstructor':'seven raw RGB captures <= cutoff; no calibration, LiDAR or labels',
                             'readout':'initial RGB detector bbox, known calibration; extra GT dimensions and yaw held fixed',
                             'scale_control':'extra pre-cutoff LiDAR background + GT masks excluding all actors',
                             'reference':'GT translation and target LiDAR, never used to fit predicted center',
                             'world_model':'same initial RGB, text, GT map and future actor/ego trajectories; only target initial translation replaced'},
              'lidar_anchor_file':str(lidar),'lidar_delta_ms':(int(lidar.stem)-cutoff)/1e6,
              'readout_rule':'central 60% detector box; >=20 finite positive pixels; ordinary fixed-size/yaw 2D cuboid fit; use its first visible face and known rays to read a robust radial center from DVGT depth',
              'controls':['ordinary RGB bbox cuboid fit with the same oracle size/yaw',
                          'known pixel rays replace learned lateral coordinates',
                          'single median background LiDAR scale, target and all other GT cuboids excluded'],
              'reference_check':'apply identical face readout to target LiDAR; >=6 central points and center error <=0.5m required for causal generation admission',
              'stop_rules':['OOM stops immediately without retry or downsizing',
                            'invalid projection or insufficient target/reference support stops causal generation',
                            'no threshold/grid/seed expansion; if scale control removes meaningful residual, keep negative result',
                            'render/generate only if reliable controlled center residual >0.3m and projected bbox change >2px'],
              'followup_if_admitted':'seed42 metric raw / scale-controlled / ordinary cuboid-fit versus existing GT clean, unchanged other states; then seed43 only for a repeatable nontrivial signal',
              'generation_frames':[0,30,60,90,120,150,180,210,234],
              'training_overlap':'unknown','failure_ledger_refs':['V75-F01','V74-H2-F20','V74-H2-F21','V74-H2-F22'],
              'human_verdict':None}
    if a.source_protocol:
        source=json.loads(a.source_protocol.read_text())
        assert source['task_id']=='WS-V75-VISIBLE-DEV-01'
        protocol.update(source_protocol=str(a.source_protocol),admission_policy='reference_and_raw_support_without_error_ranking',
                        role=source['boundary'],target_selection=source['target_selection'],
                        followup_if_admitted=source['generation'],generation_frames=source['measurement_frames'],
                        stop_rules=['OOM stops immediately without retry or downsizing',
                                    'causal generation requires same reliable target LiDAR readout and raw model support',
                                    'good cases retained; no residual-size or global-scale-failure ranking',source['stop']])
    OUT.mkdir(parents=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    inp=OUT/'native_input/frame_0';inp.mkdir(parents=True)
    for i,v in enumerate(views):
        (inp/f'{i:02d}_{v["camera"]}.jpg').symlink_to(v['image'])
    print(json.dumps({'task_id':protocol['task_id'],'views':len(views),'bbox_network':net.tolist(),
                      'capture_delta_ms':[v['delta_to_cutoff_ms'] for v in views],
                      'extra_information':'size/yaw oracle; LiDAR metric control separately reported'}),flush=True)

if __name__=='__main__': main()
