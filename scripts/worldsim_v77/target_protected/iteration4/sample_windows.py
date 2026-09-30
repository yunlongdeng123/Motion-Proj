"""固定10Hz网格的有序曝光匹配；不复制曝光、不插帧、不放宽误差上限。"""
import bisect,copy,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from pyquaternion import Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_source_pool import read,dump,stream,mat,project,good

def match_exposures(rows,targets,tolerance_us=55000,max_gap_us=180000):
    stamps=[r['timestamp'] for r in rows];states={}
    for i,t in enumerate(targets):
        choices=range(bisect.bisect_left(stamps,t-tolerance_us),bisect.bisect_right(stamps,t+tolerance_us))
        nextstates={}
        for j in choices:
            cost=abs(stamps[j]-t)
            if i==0:nextstates[j]=(cost,(j,));continue
            options=[(v[0]+cost,v[1]+(j,)) for k,v in states.items() if 0<stamps[j]-stamps[k]<=max_gap_us]
            if options:nextstates[j]=min(options)
        states=nextstates
        if not states:return None
    return list(min(states.values())[1])

def resolve_ten(meta,root,selection):
    cache=root.parent/'camera_metadata_cache.json'
    if cache.exists():cached=read(cache);camera={tuple(k.split('|')):v for k,v in cached['camera'].items()};cal=cached['cal']
    else:
        samples={r['token']:r for r in read(meta/'sample.json')};channels={r['token']:r['channel'] for r in read(meta/'sensor.json')};cal={r['token']:r for r in read(meta/'calibrated_sensor.json')}
        wanted={(c['scene_token'],c['camera']) for c in selection['clips']};camera=defaultdict(list)
        for r in stream(meta/'sample_data.json'):
            key=(samples[r['sample_token']]['scene_token'],channels[cal[r['calibrated_sensor_token']]['sensor_token']])
            if key in wanted:camera[key].append({k:r[k] for k in ['token','sample_token','timestamp','filename','width','height','ego_pose_token','calibrated_sensor_token']})
        for rows in camera.values():rows.sort(key=lambda r:r['timestamp'])
        dump(cache,{'camera':{'|'.join(k):v for k,v in camera.items()},'cal':cal})
    clips=[];audits=[];wanted_ego=set()
    for c in selection['clips']:
        rows=camera[c['scene_token'],c['camera']];ts=[r['timestamp'] for r in rows]
        anchor=bisect.bisect_left(ts,c['start_timestamp_us']+50000)
        targets=[ts[anchor]+100000*i for i in range(10,20)]
        nearest=[min(range(max(0,bisect.bisect_left(ts,t)-1),min(len(ts),bisect.bisect_left(ts,t)+1)),key=lambda j:abs(ts[j]-t)) for t in targets]
        nt=[ts[j] for j in nearest]
        oldpass=len(set(nt))==10 and max(np.diff(nt))<=180000 and max(abs(t-s) for t,s in zip(nt,targets))<=55000
        ix=match_exposures(rows,targets)
        audits.append({'source_id':c['source_id'],'nearest_fixed_ten_pass':bool(oldpass),'ordered_fixed_ten_pass':ix is not None,'nearest_timestamps':nt,'targets':targets,'selected_timestamps':[] if ix is None else [ts[j] for j in ix]})
        if ix is None:continue
        d=copy.deepcopy(c);d['frames']=[]
        for i,(t,j) in enumerate(zip(targets,ix)):
            f=rows[j]|{'frame':i,'requested_timestamp_us':t,'delta_ms':abs(t-ts[j])/1000};d['frames'].append(f);wanted_ego.add(f['ego_pose_token'])
        d['source_window']={'selection':'fixed original grid positions 10:20; globally ordered unique exposure assignment','interpolated_images':False};clips.append(d)
    ego={r['token']:r for r in stream(meta/'ego_pose.json') if r['token'] in wanted_ego}
    rejected=[];valid=[]
    for c in clips:
        failures=[]
        for f in c['frames']:
            ca=cal[f['calibrated_sensor_token']];e=ego[f['ego_pose_token']];c2w=mat(e['translation'],e['rotation'])@mat(ca['translation'],ca['rotation']);k=np.array(ca['camera_intrinsic'],float);k[0]*=1024/f['width'];k[1]*=576/f['height']
            f['camera_to_world']=c2w.tolist();f['intrinsics_1024']=k.tolist();f['actors']=[]
            for a in c['actors']:
                stamps=c['keyframe_timestamps'];t=f['timestamp'];ix=bisect.bisect_right(stamps,t)
                if not 0<ix<len(stamps) or stamps[ix]-stamps[ix-1]>600000:failures.append('annotation_unbracketed');continue
                lo,hi=a['keyframe_annotations'][ix-1:ix+1];u=(t-stamps[ix-1])/(stamps[ix]-stamps[ix-1]);obj={'translation':((1-u)*np.array(lo['translation'])+u*np.array(hi['translation'])).tolist(),'size':((1-u)*np.array(lo['size'])+u*np.array(hi['size'])).tolist(),'rotation':Quaternion.slerp(Quaternion(lo['rotation']),Quaternion(hi['rotation']),amount=u).elements.tolist()}
                pr=project(obj,{'w2c':np.linalg.inv(c2w),'k':k});f['actors'].append(obj|{'instance_token':a['instance_token'],'category':a['category'],'projection':pr})
                if not good(pr):failures.append('actual_exposure_geometry')
        if failures:rejected.append({'source_id':c['source_id'],'reasons':dict(Counter(failures))})
        else:valid.append(c)
    dump(root/'sampling_control.json',{'nearest_fixed_ten_pass':sum(a['nearest_fixed_ten_pass'] for a in audits),'ordered_fixed_ten_pass':sum(a['ordered_fixed_ten_pass'] for a in audits),'controls':audits,'geometry_rejected':rejected})
    dump(root/'source_manifest.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r4','clips':valid,'sampling_rejected':[a['source_id'] for a in audits if not a['ordered_fixed_ten_pass']],'geometry_rejected':rejected})
    print('ORDERED_EXPOSURES',len(audits),'nearest',sum(a['nearest_fixed_ten_pass'] for a in audits),'ordered',sum(a['ordered_fixed_ten_pass'] for a in audits),'geometry_pass',len(valid),flush=True)
