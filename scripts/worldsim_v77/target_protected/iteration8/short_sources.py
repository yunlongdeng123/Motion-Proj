"""按实际一秒训练窗口重采well-observed密集来源；窗外弱观测不等于窗内不可用。"""
from pathlib import Path
import sys,os,copy,bisect
from collections import defaultdict,Counter
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from nuscenes.utils.splits import train
from prepare_source_pool import read,dump,stream,mat,project,good,intersects,sharding,EXPOSED,CAMS
from iteration4.sample_windows import resolve_ten
from prepare_context_extract import context,extract
from assemble_sources import link
from asset_factory import O,F,T
META=T.parent/'WS-V77-DELETE-AUDIT-20260928/r1/metadata/v1.0-trainval'
S=O/'short_sources';S.mkdir(exist_ok=True)

def index():
    path=S/'candidate_pool.json'
    if path.exists():return read(path)['candidates']
    blocked=set(read(O/'source_split.json')['synthetic_validation_scenes'])|set(read(O/'source_pool_summary.json')['old_validation_scenes_excluded'])|EXPOSED
    logs={r['token']:r for r in read(META/'log.json')}
    scenes={r['token']:r for r in read(META/'scene.json') if r['name'] in train and r['name'] not in blocked and logs[r['log_token']]['location']=='boston-seaport' and not any(w in r['description'].lower() for w in ['night','rain','dark'])}
    samples={r['token']:r for r in read(META/'sample.json') if r['scene_token'] in scenes};ordered=defaultdict(list)
    for r in samples.values():ordered[r['scene_token']].append(r)
    for arr in ordered.values():arr.sort(key=lambda r:r['timestamp'])
    categories={r['token']:r['name'] for r in read(META/'category.json')};instances={r['token']:categories[r['category_token']] for r in read(META/'instance.json')};anns=defaultdict(dict)
    for a in stream(META/'sample_annotation.json'):
        if a['sample_token'] not in samples or instances[a['instance_token']] not in ['vehicle.car','vehicle.truck','vehicle.bus.rigid','vehicle.bus.bendy']:continue
        anns[a['sample_token']][a['instance_token']]={k:a[k] for k in ['instance_token','translation','rotation','size','visibility_token','num_lidar_pts']}|{'category':instances[a['instance_token']]}
    print('SHORT_ANN_READY',len(scenes),flush=True)
    cal={r['token']:r for r in read(META/'calibrated_sensor.json')};channels={r['token']:r['channel'] for r in read(META/'sensor.json')};sd=defaultdict(dict);wanted=set()
    for r in stream(META/'sample_data.json'):
        if not r['is_key_frame'] or r['sample_token'] not in samples:continue
        ch=channels[cal[r['calibrated_sensor_token']]['sensor_token']]
        if ch not in CAMS:continue
        sd[r['sample_token']][ch]={k:r[k] for k in ['filename','token','timestamp','ego_pose_token','calibrated_sensor_token']};wanted.add(r['ego_pose_token'])
    ego={r['token']:r for r in stream(META/'ego_pose.json') if r['token'] in wanted};shards=sharding();candidates=[]
    for scene,rr in ordered.items():
        for start in [3,7,11,15,19,23,27,31]:
            keys=[r['token'] for r in rr[start:start+8]]
            if len(keys)!=8:continue
            window=keys[2:6];common=set.intersection(*(set(anns[k]) for k in window))
            tracks=[tok for tok in common if all(int(anns[k][tok]['visibility_token'])==4 for k in window) and np.median([anns[k][tok]['num_lidar_pts'] for k in window])>=8]
            if len(tracks)<2:continue
            for cam in CAMS:
                if any(cam not in sd[k] for k in window):continue
                name=sd[window[1]][cam]['filename'];sh=sorted(shards.get(Path(name).name.split('__')[0],set()))
                if not sh or not set(sh)<={'03','07'}:continue
                views=[]
                for k in window:
                    d=sd[k][cam];ca=cal[d['calibrated_sensor_token']];e=ego[d['ego_pose_token']];c2w=mat(e['translation'],e['rotation'])@mat(ca['translation'],ca['rotation']);intr=np.array(ca['camera_intrinsic'],float);intr[0]*=.64;intr[1]*=.64
                    views.append({'w2c':np.linalg.inv(c2w),'k':intr})
                actors=[]
                for tok in sorted(tracks):
                    pp=[project(anns[k][tok],v) for k,v in zip(window,views)]
                    if not all(good(p) for p in pp):continue
                    pr=pp[1];front=[]
                    for other,a in anns[window[1]].items():
                        if other==tok:continue
                        op=project(a,views[1])
                        if op and op['depth']<pr['depth']:front.append(intersects(pr['box_xyxy'],op['box_xyxy'])/(pr['width']*pr['height']))
                    if max(front,default=0)>.08:continue
                    actors.append({'instance_token':tok,'category':anns[window[0]][tok]['category'],'anchor_projection':pr,'keyframe_annotations':[anns[k].get(tok) for k in keys],'keyframe_projections':[None,None]+pp+[None,None],'max_front_box_overlap':max(front,default=0)})
                if len(actors)<2:continue
                actors.sort(key=lambda a:(abs(a['anchor_projection']['area_frac']-.028),a['instance_token']))
                s=scenes[scene];candidates.append({'scene':s['name'],'scene_token':scene,'description':s['description'],'location':'boston-seaport','log_token':s['log_token'],'camera':cam,'start_keyframe':start,'keyframe_tokens':keys,'start_timestamp_us':samples[keys[0]]['timestamp'],'keyframe_timestamps':[samples[k]['timestamp'] for k in keys],'shards':sh,'actors':actors[:4],'anchor_rgb':name,'anchor_sample_data_token':sd[window[1]][cam]['token'],'source_selection_scope':'high visibility/size on four bracket keyframes for actual 10-frame window, not four-second complete-track requirement'})
    dump(path,{'candidates':candidates,'scenes':len({c['scene'] for c in candidates})});print('SHORT_POOL',len(candidates),'scenes',len({c['scene'] for c in candidates}),flush=True);return candidates

def main():
    pool=index();current=read(F/'source_manifest.json')['clips'];identities={(c['scene'],c['camera'],c['start_keyframe']) for c in current if c['source_id'].startswith('N')};counts=Counter();chosen=[]
    byscene=defaultdict(list)
    for c in pool:
        if (c['scene'],c['camera'],c['start_keyframe']) not in identities:byscene[c['scene']].append(c)
    for rank in range(2):
        for scene in sorted(byscene):
            cc=sorted(byscene[scene],key=lambda c:(-len(c['actors']),c['start_keyframe'],c['camera']))
            if rank<len(cc) and len(chosen)<40:chosen.append(copy.deepcopy(cc[rank]))
    for i,c in enumerate(chosen):c['source_id']=f'X{i+1:03}'
    if (S/'source_selection.json').exists():chosen=read(S/'source_selection.json')['clips']
    else:dump(S/'source_selection.json',{'clips':chosen,'budget':40,'sampling_before_model_outputs':True})
    if not (S/'factory/source_manifest.json').exists():resolve_ten(META,S/'factory',{'clips':chosen})
    factory=S/'factory';clips=read(factory/'source_manifest.json')['clips']
    if not (factory/'source_context.json').exists():context(factory,META)
    required=sorted({f['filename'] for c in clips for f in c['frames']}|{c['frames'][i]['sensors']['LIDAR_TOP']['filename'] for c in read(factory/'source_context.json')['clips'] for i in [0,3,6]})
    dump(factory/'all_required_files.json',required);missing=[]
    bases=[F/'rgb',T/'r1/rgb',T/'r1/expanded_factory/rgb',T/'r1/source_extension/rgb',T/'r3/factory/rgb',T/'r4/factory/rgb']
    for n in required:
        dst=factory/'rgb'/n
        src=next((b/n for b in bases if (b/n).is_file()),None)
        if src is not None:link(src,dst)
        elif not dst.is_file():missing.append(n)
    dump(factory/'required_files.json',missing)
    if missing:extract(factory,Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval'))
    dump(factory/'required_files.json',required);link(F/'maps',factory/'maps')
    dump(S/'summary.json',{'selected':len(chosen),'resolved':len(clips),'scenes':len({c['scene'] for c in clips}),'files':len(required),'new_extract_requests':len(missing),'missing':[n for n in required if not (factory/'rgb'/n).is_file()],'stage':'source materialization only; no training admission'})
    print('SHORT_PREPARED',read(S/'summary.json'),flush=True)
if __name__=='__main__':main()
