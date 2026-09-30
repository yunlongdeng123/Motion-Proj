"""CPU有界扫描已有train实例的完整真实track，先索引几何再提取少量RGB。"""
import argparse,json,sys,bisect
from pathlib import Path
from collections import defaultdict,Counter
import ijson,numpy as np
from pyquaternion import Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import transform,projection,view_angles,wrap

def read(p):return json.loads(Path(p).read_text())
def stream(p):
    with Path(p).open('rb') as f:yield from ijson.items(f,'item',use_float=True)
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main(root,factory,meta):
    catalog=read(root/'multiview/catalog.json');wanted={r['instance_token']:r for r in catalog['instances']};samples={r['token']:r for r in read(meta/'sample.json')}
    tracks=defaultdict(list);by_sample=defaultdict(set)
    for a in stream(meta/'sample_annotation.json'):
        if a['instance_token'] in wanted:
            a['timestamp']=samples[a['sample_token']]['timestamp'];a['scene_token']=samples[a['sample_token']]['scene_token'];tracks[a['instance_token']].append(a);by_sample[a['sample_token']].add(a['instance_token'])
    for v in tracks.values():v.sort(key=lambda a:a['timestamp'])
    sensors={r['token']:r['channel'] for r in read(meta/'sensor.json')};cal={r['token']:r for r in read(meta/'calibrated_sensor.json')};data=[];need_ego=set()
    for s in stream(meta/'sample_data.json'):
        if s['is_key_frame'] and s['sample_token'] in by_sample and sensors[cal[s['calibrated_sensor_token']]['sensor_token']].startswith('CAM_'):
            data.append(s);need_ego.add(s['ego_pose_token'])
    ego={r['token']:r for r in stream(meta/'ego_pose.json') if r['token'] in need_ego};rows=[]
    for s in data:
        c=cal[s['calibrated_sensor_token']];e=ego[s['ego_pose_token']];channel=sensors[c['sensor_token']];c2w=transform(e['translation'],e['rotation'])@transform(c['translation'],c['rotation'])
        K=np.array(c['camera_intrinsic'])*.64;K[2,2]=1;fr={'camera_to_world':c2w.tolist(),'_w2c':np.linalg.inv(c2w),'intrinsics_1024':K.tolist()}
        for tok in by_sample[s['sample_token']]:
            track=tracks[tok];times=[a['timestamp'] for a in track];t=s['timestamp'];hi=bisect.bisect_right(times,t)
            if hi==len(times) and t==times[-1]:hi-=1
            if hi==0 or hi>=len(times):continue
            a,b=track[hi-1:hi+1]
            if b['timestamp']-a['timestamp']>600000 or a['visibility_token']!='4' or b['visibility_token']!='4':continue
            u=(t-a['timestamp'])/(b['timestamp']-a['timestamp']);a=a|{'translation':((1-u)*np.array(a['translation'])+u*np.array(b['translation'])).tolist(),'rotation':Quaternion.slerp(Quaternion(a['rotation']),Quaternion(b['rotation']),amount=u).elements.tolist()}
            p=projection(a,fr)
            if p is None:continue
            x0,y0,x1,y1=p['box'];w=x1-x0;h=y1-y0
            if min(w/72,h/40)<1 or not .006<=w*h/(1024*576)<=.18 or min(x0,y0,1024-x1,576-y1)<8:continue
            rows.append({'instance_token':tok,'source_ids':wanted[tok]['source_ids'],'scene_token':a['scene_token'],'camera':channel,
                'filename':s['filename'],'timestamp_us':t,'box_xyxy':p['box'].tolist(),'view_angles_deg':view_angles(a,fr).tolist(),
                'available':(factory/'rgb'/s['filename']).is_file(),'sample_data_token':s['token'],'camera_to_world':c2w.tolist(),
                'intrinsics_1024':K.tolist(),'actor':{k:a[k] for k in ['translation','rotation','size','instance_token']},'area_px':w*h,'mask_status':'not_inferred'})
    grouped=defaultdict(list)
    for r in rows:grouped[r['instance_token']].append(r)
    instances=[];selected=[]
    for tok,rs in grouped.items():
        cameras=sorted({r['camera'] for r in rs});best_pair=max(((a,b) for a in rs for b in rs),key=lambda ab:abs(wrap(ab[0]['view_angles_deg'][0]-ab[1]['view_angles_deg'][0])))
        span=abs(wrap(best_pair[0]['view_angles_deg'][0]-best_pair[1]['view_angles_deg'][0]));instances.append({'instance_token':tok,'source_ids':wanted[tok]['source_ids'],'views':len(rs),'cameras':cameras,'yaw_span_deg':span})
        if len(cameras)>=2 and span>=15:
            # 有限预检：最远两个真实角度+中间最大面积，单实例最多3张。
            picks=[*best_pair,max(rs,key=lambda r:r['area_px'])]
            seen=set()
            for r in picks:
                if r['filename'] not in seen:selected.append(r);seen.add(r['filename'])
    # 防止扩成又一轮全池视觉扫描；只按事先几何排序选最多4个真实实例。
    eligible=sorted([r for r in instances if len(r['cameras'])>=2 and r['yaw_span_deg']>=15],key=lambda r:(-r['yaw_span_deg'],r['instance_token']))[:4]
    selected=[r for r in selected if r['instance_token'] in {i['instance_token'] for i in eligible}]
    summary={'scanned_existing_instances':len(wanted),'geometric_candidate_views':len(rows),'instances_multi_camera':sum(len(i['cameras'])>=2 for i in instances),
        'selected_instances':len(eligible),'selected_views':len(selected),'selected_missing_rgb':sum(not r['available'] for r in selected),
        'scope':'已有train来源实例的完整track；每实例独立原相机、原曝光、GT插值<=0.6s。筛选不等于图像/分割合格。没有把多视角输入DriveEditor网络。'}
    dump(root/'multiview/full_track_catalog.json',{'summary':summary,'instances':instances,'selected':selected,'views':rows})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--factory',type=Path,required=True);p.add_argument('--meta',type=Path,required=True);a=p.parse_args();main(a.root,a.factory,a.meta)
