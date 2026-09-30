"""已选真实多视角各扩固定10曝光窗口；不复制帧、不外推GT、不换实例。"""
import argparse,bisect,copy,json,sys,tarfile,time
from collections import defaultdict,Counter
from pathlib import Path
import ijson,numpy as np
from PIL import Image
from pyquaternion import Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import transform,projection,view_angles,wrap,read,dump
from prepare_source_pool import sharding

def stream(p):
    with p.open('rb') as f:yield from ijson.items(f,'item',use_float=True)

def main(root,meta,parent,pub,window_search=False):
    out=root/('temporal_windows' if window_search else 'temporal');out.mkdir(exist_ok=True);views=read(root/'multiview/selected_views.json')['views']
    samples={r['token']:r for r in read(meta/'sample.json')}; scenes={r['token']:r for r in read(meta/'scene.json')}
    wanted={v['instance_token'] for v in views};wanted_scene={v['scene_token'] for v in views}
    anns=defaultdict(list)
    for a in stream(meta/'sample_annotation.json'):
        if samples[a['sample_token']]['scene_token'] in wanted_scene:
            anns[a['instance_token']].append(a|{'timestamp':samples[a['sample_token']]['timestamp']})
    for track in anns.values():track.sort(key=lambda a:a['timestamp'])
    channels={r['token']:r['channel'] for r in read(meta/'sensor.json')};cal={r['token']:r for r in read(meta/'calibrated_sensor.json')}
    cameras=defaultdict(list)
    for d in stream(meta/'sample_data.json'):
        scene=samples[d['sample_token']]['scene_token'];ch=channels[cal[d['calibrated_sensor_token']]['sensor_token']]
        if scene in wanted_scene and ch.startswith('CAM_'):cameras[scene,ch].append(d)
    for rs in cameras.values():rs.sort(key=lambda d:d['timestamp'])
    need_ego=set();windows=[]
    for i,v in enumerate(views):
        rs=cameras[v['scene_token'],v['camera']];times=[d['timestamp'] for d in rs];window=[]
        for fid in range(19 if window_search else 10):
            t=v['timestamp_us']+(fid-(9 if window_search else 5))*100000;ix=bisect.bisect_left(times,t)
            d=min((rs[j] for j in [ix-1,ix] if 0<=j<len(rs)),key=lambda d:abs(d['timestamp']-t))
            window.append(d|{'frame':fid,'requested_timestamp_us':t,'delta_ms':abs(t-d['timestamp'])/1000});need_ego.add(d['ego_pose_token'])
        windows.append((v,window))
    ego={e['token']:e for e in stream(meta/'ego_pose.json') if e['token'] in need_ego}
    old=read(parent/'source_manifest.json')['clips'];byinst={c['actors'][0]['instance_token']:c for c in old}
    def at(tok,t):
        tr=anns[tok];ix=bisect.bisect_right([a['timestamp'] for a in tr],t)
        if not 0<ix<len(tr):return None
        a,b=tr[ix-1:ix+1]
        if b['timestamp']-a['timestamp']>600000:return None
        u=(t-a['timestamp'])/(b['timestamp']-a['timestamp'])
        return {'translation':((1-u)*np.array(a['translation'])+u*np.array(b['translation'])).tolist(),'rotation':Quaternion.slerp(Quaternion(a['rotation']),Quaternion(b['rotation']),amount=u).elements.tolist(),
                'size':((1-u)*np.array(a['size'])+u*np.array(b['size'])).tolist(),'instance_token':tok,'visibility_ok':a['visibility_token']=='4' and b['visibility_token']=='4'}
    clips=[]
    for i,(v,frames) in enumerate(windows):
        c=copy.deepcopy(byinst[v['instance_token']]);sid=f'V{i+1:03}';c.update(source_id=sid,camera=v['camera'],scene_token=v['scene_token'],scene=scenes[v['scene_token']]['name'],frames=[],seed_view=v,source_roles=['donor'])
        bad=[];ts=[f['timestamp'] for f in frames]
        if len(set(ts))!=10 or min(np.diff(ts))<=0 or max(np.diff(ts))>180000 or max(f['delta_ms'] for f in frames)>55:bad.append('exposure_continuity')
        for f in frames:
            ca=cal[f['calibrated_sensor_token']];e=ego[f['ego_pose_token']];c2w=transform(e['translation'],e['rotation'])@transform(ca['translation'],ca['rotation']);K=np.array(ca['camera_intrinsic']);K[0]*=1024/f['width'];K[1]*=576/f['height']
            fr=f|{'camera_to_world':c2w.tolist(),'intrinsics_1024':K.tolist(),'_w2c':np.linalg.inv(c2w)};a=at(v['instance_token'],f['timestamp']);framebad=[]
            if a is None:framebad.append('GT_unbracketed');fr['actors']=[]
            else:
                p=projection(a,fr);a['category']=c['actors'][0]['category'];fr['actors']=[a]
                if p is None:framebad.append('behind_camera')
                else:
                    b=p['box'];w,h=b[2:]-b[:2];a['projection']={'box_xyxy':b.tolist(),'hull':p['corners'].tolist(),'depth':p['center_depth']};fr['view_angles_deg']=view_angles(a,fr).tolist()
                    if w<128 or h<72 or min(b[0],b[1],1024-b[2])<8 or b[3]+8>=512:framebad.append('size_or_border')
                    if not a['visibility_ok']:framebad.append('GT_visibility')
                    hits=[]
                    for tok in anns:
                        if tok==v['instance_token']:continue
                        other=at(tok,f['timestamp'])
                        if other is None:continue
                        op=projection(other,fr,clip_near=True)
                        if op is None or op['center_depth']>=p['center_depth']:continue
                        ob=op['box'];area=max(0,min(b[2],ob[2])-max(b[0],ob[0]))*max(0,min(b[3],ob[3])-max(b[1],ob[1]))
                        if area/(w*h)>.03:hits.append(tok)
                    if hits:framebad.append('potential_foreground_envelope_overlap')
            fr.pop('_w2c');fr['proposal_flags']=framebad;c['frames'].append(fr)
            if framebad:bad.append({'frame':f['frame'],'flags':framebad})
        attempts=[]
        if window_search:
            options=[]
            for start in range(10):
                fs=c['frames'][start:start+10];tt=[f['timestamp'] for f in fs]
                flags=[{'frame':f['frame'],'flags':f['proposal_flags']} for f in fs if f['proposal_flags']]
                if len(set(tt))!=10 or min(np.diff(tt))<=0 or max(np.diff(tt))>180000 or max(f['delta_ms'] for f in fs)>55:flags.append('exposure_continuity')
                attempts.append({'start_index':start,'flags':flags})
                if not flags:
                    areas=[np.prod(np.diff(np.array(f['actors'][0]['projection']['box_xyxy']).reshape(2,2),axis=0)) for f in fs]
                    options.append((-float(min(areas)),abs(start-4),start))
            if options:
                start=min(options)[2];c['frames']=c['frames'][start:start+10];bad=[]
                for j,f in enumerate(c['frames']):f['frame']=j
            else:bad=['no_valid_containing_seed_window'];c['frames']=c['frames'][4:14]
        c['prompt_frame']=next((f['frame'] for f in c['frames'] if f['filename']==v['filename']),None)
        c['geometry_flags']=bad;c['geometry_pass']=not bad;c['window_provenance']={'seed_view':v['view_id'],'sampling':'bounded 10 windows containing seed, full-window geometry/exposure gate then max minimum area; no mask/output-based selection' if window_search else '10Hz grid centered on fixed r2 seed exposure; no output-based window selection','num_frames':10,'same_instance_multiview':True,'not_simultaneous_views':True,'window_attempts':attempts}
        clips.append(c)
    dump(out/'source_manifest.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':root.name,'clips':clips})
    rgb=out/'rgb';rgb.mkdir(exist_ok=True)
    needed={f['filename'] for c in clips if c['geometry_pass'] for f in c['frames']}
    for n in needed:
        p=rgb/n
        for base in [parent/'rgb',root/'multiview/rgb',root/'temporal/rgb']:
            src=base/n
            if src.is_file() and not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(src.resolve())
    groups=defaultdict(set);mapping=sharding()
    for n in needed:
        if not (rgb/n).is_file():
            for shard in mapping[Path(n).name.split('__')[0]]:groups[shard].add(n)
    extraction=[]
    for shard,names in sorted(groups.items()):
        todo={n for n in names if not (rgb/n).is_file()};start=time.monotonic()
        if not todo:continue
        with tarfile.open(pub/f'v1.0-trainval{int(shard):02}_blobs.tgz','r|gz') as tf:
            for m in tf:
                n=m.name.removeprefix('./')
                if n not in todo:continue
                assert m.isfile() and not Path(n).is_absolute() and '..' not in Path(n).parts
                p=rgb/n;p.parent.mkdir(parents=True,exist_ok=True)
                with tf.extractfile(m) as s,p.open('wb') as d:d.write(s.read())
                with Image.open(p) as im:im.verify()
                todo.remove(n)
                if not todo:break
        extraction.append({'shard':shard,'requested':len(names),'missing':sorted(todo),'seconds':time.monotonic()-start});print('EXTRACT',extraction[-1],flush=True)
    result={'clips':len(clips),'geometry_pass':sum(c['geometry_pass'] for c in clips),'needed_RGB':len(needed),'missing_RGB':[n for n in needed if not (rgb/n).is_file()],'extraction':extraction,'source_ids_pass':[c['source_id'] for c in clips if c['geometry_pass']]}
    dump(out/'preparation.json',result);print('TEMPORAL',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--meta',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--pub',type=Path,required=True);p.add_argument('--window-search',action='store_true');a=p.parse_args();main(a.root,a.meta,a.parent,a.pub,a.window_search)
