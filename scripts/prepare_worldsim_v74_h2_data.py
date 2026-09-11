"""H2 无 GPU 数据准备：FIT 帧块任务，以及盲化 AV2 储备坐标。"""
import argparse,json,os,sys,time,platform
try:
    import resource
except ImportError:
    resource=None
from pathlib import Path
from collections import Counter
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
FIELDS=['origins_actor_m','directions_actor','observed_first_range_m','points_actor_m','positive_actor','ambiguous_owner','point_timestamps_ns']
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def peak_rss_mib():
    if resource is not None:return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    return None
def unpack(path):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['frame_metadata_json']));off=data['frame_offsets']
        return [{**{k:data[k][off[i]:off[i+1]].copy() for k in FIELDS},'metadata':m} for i,m in enumerate(meta)]
def pack(frames):
    counts=[len(f['observed_first_range_m']) for f in frames]
    result={k:np.concatenate([f[k] for f in frames]) for k in FIELDS}
    result['frame_offsets']=np.array([0,*np.cumsum(counts)],np.int64)
    result['frame_metadata_json']=np.array(json.dumps([f['metadata'] for f in frames]))
    return result
def fit(root):
    started=time.monotonic();source=Path('/root/autodl-tmp/data/worldsim_v74/p0_r1')
    roles=json.loads((ROOT/'configs/worldsim_v74/data_roles.json').read_text())['dataset_roles']
    split={d:{'FIT_TRAIN':sorted(r['FIT'])[:-n],'FIT_VAL':sorted(r['FIT'])[-n:],'DEV':r['DEV']} for d,r,n in [(d,roles[d],4 if d=='nuscenes' else 3) for d in roles]}
    cfg={'task':'WS-V74-H2-DATA-01','dataset_roles':split,'seed':7411,
         'selection':'lexicographic existing FIT logs; final 4 nuScenes / 3 AV2 for internal validation',
         'episodes':'all six observed frames sorted in time: first 3 -> last 3; reverse 3 -> 3',
         'supervision_policy':'FIT only; input support recomputed exclusively from A; B loaded only by teacher/loss',
         'historical_exposure':'all FIT already exposed in H1; internal validation is not independent confirmation',
         'dev_index':str(source/'probe_cohort.json'),'final_quality_access':False}
    write(ROOT/'configs/worldsim_v74_h2/data_roles.json',cfg);write(root/'config.json',cfg)
    rows=json.loads((source/'index.json').read_text())['cases'];entries=[];counts=Counter();issues=[]
    for row in rows:
        if row['role']!='FIT':continue
        frames=unpack(row['build_file'])+unpack(row['query_truth_file'])
        frames.sort(key=lambda f:int(f['metadata'].get('timestamp_ns',f['metadata']['timestamp_us']*1000)))
        unique=[f['metadata']['sample_id'] for f in frames]
        if len(set(unique))!=len(unique):raise ValueError('duplicated actual frames: '+row['case_id'])
        if len(frames)<2:issues.append({'case_id':row['case_id'],'reason':'fewer_than_two_frames'});continue
        mid=len(frames)//2
        phase='FIT_TRAIN' if row['log_id'] in split[row['dataset']]['FIT_TRAIN'] else 'FIT_VAL'
        for name,left,right in [('forward',frames[:mid],frames[mid:]),('reverse',frames[mid:],frames[:mid])]:
            folder=root/row['dataset']/phase/(row['case_id']+'__'+name);folder.mkdir(parents=True,exist_ok=True)
            inp=pack(left);sup=pack(right);positive=inp['positive_actor']&~inp['ambiguous_owner']
            inp['support_points_actor_m']=np.unique(inp['points_actor_m'][positive],axis=0)
            with np.load(row['build_file'],allow_pickle=False) as b:inp['size_lwh_m']=b['size_lwh_m']
            np.savez(folder/'input.npz',**inp);np.savez(folder/'supervision.npz',**sup)
            # 元数据归属仍是 box proxy；无自有返回不伪装成正例。
            meta={k:row[k] for k in ['case_id','dataset','log_id','owner','scene','category','timing','ownership']}
            meta.update(phase=phase,episode=name,input_frame_ids=[f['metadata']['sample_id'] for f in left],
                        source_build=row['build_file'],source_fit_supervision=row['query_truth_file'])
            write(folder/'input_metadata.json',meta)
            status='ready' if positive.any() else 'no_input_positive'
            if not sup['positive_actor'].any():status+='__no_supervision_positive'
            counts[row['dataset']+'/'+phase+'/'+status]+=1
            entries.append({**meta,'input':str(folder/'input.npz'),'supervision':str(folder/'supervision.npz'),
                'input_rays':len(inp['observed_first_range_m']),'input_positive':int(positive.sum()),
                'supervision_rays':len(sup['observed_first_range_m']),'supervision_positive':int(sup['positive_actor'].sum()),
                'supervision_frame_ids':[f['metadata']['sample_id'] for f in right],'status':status})
        if len(entries)%200==0:print(json.dumps({'task':'FIT','episodes':len(entries)}),flush=True)
    write(root/'index.json',{'schema':'worldsim_v74_h2.fit_blocks.v1','episodes':entries,'issues':issues})
    summary={'task':'WS-V74-H2-DATA-01','status':'done','episodes':len(entries),'objects':len(entries)//2,'counts':dict(counts),'issues':issues,
        'wall_s':time.monotonic()-started,'peak_rss_mib':peak_rss_mib(),
        'data_root':str(root),'dev_new_quality_evaluated':False,'final_values_loaded':False}
    write(root/'summary.json',summary);write(ROOT/'docs/autoresearch/worldsim_v74_h2/fit_data_summary.json',summary)
    print(json.dumps(summary),flush=True)
def near_box(o,d,ext):
    parallel=np.abs(d)<1e-10;inv=1/np.where(parallel,1,d)
    x=(-ext-o)*inv;y=(ext-o)*inv
    lo=np.where(parallel,-np.inf,np.minimum(x,y)).max(1);hi=np.where(parallel,np.inf,np.maximum(x,y)).min(1)
    return (hi>=np.maximum(lo,0))&~np.any(parallel&(np.abs(o)>ext),axis=1)
def final(root,raw_root=None):
    from motion_proj.worldsim_v73.av2_geometry import AV2MetricLog,RIGID_VEHICLES,sizes_at
    plan=json.loads((ROOT/'configs/worldsim_v74/av2_final.json').read_text());start=time.monotonic();all_entries=[]
    # 仅复用冻结传感器/刚体语义；一次只驻留一束块和一个 sweep，不驻留 all-track × all-return。
    for number,log in enumerate(plan['logs'],1):
        out=root/log['log_id'];done=out/'index.json'
        if done.exists():all_entries.extend(json.loads(done.read_text())['cases']);continue
        data=AV2MetricLog((Path(raw_root or plan['output'])/log['log_id']).resolve());tracks=data.load_tracks()
        stamps={int(k):int(v) for k,v in log['lidar_timestamps_ns'].items()};build_ids=plan['build_lidar_indices']
        selected={owner:t for owner,t in tracks.items() if t['category'] in RIGID_VEHICLES and np.any(t['poses'].at(np.array([stamps[i] for i in build_ids],np.int64))[1])}
        owners_list=list(tracks);actor_frames={x:[] for x in selected}
        for index,stamp in sorted(stamps.items()):
            sweep=data.sweep(data.root/'sensors/lidar'/f'{stamp}.feather');n=sweep['raw_points']
            membership=np.zeros(n,np.int16);owner_idx=np.full(n,-1,np.int32)
            for j,owner in enumerate(owners_list):
                track=tracks[owner]
                for lo in range(0,n,8192):
                    hi=min(lo+8192,n);part={k:sweep[k][lo:hi] for k in ['point_timestamps_ns','points_world_m','origins_world_m','directions_world','valid_sensor_pose']}
                    local,_,_,known=data.actor_coordinates(owner,part)
                    inside=known&np.all(np.abs(local)<=sizes_at(track,part['point_timestamps_ns'])/2+.1,axis=1)
                    ids=np.flatnonzero(inside)+lo;owner_idx[ids[membership[ids]==0]]=j;membership[ids]+=1
            for owner,track in selected.items():
                pieces=[];j=owners_list.index(owner)
                for lo in range(0,n,8192):
                    hi=min(lo+8192,n);part={k:sweep[k][lo:hi] for k in ['point_timestamps_ns','points_world_m','origins_world_m','directions_world','valid_sensor_pose']}
                    local,o,d,known=data.actor_coordinates(owner,part)
                    near=known&near_box(o,d,sizes_at(track,part['point_timestamps_ns'])/2+.5)
                    ids=np.flatnonzero(near)+lo
                    pieces.append({'origins_actor_m':o[near].astype(np.float32),'directions_actor':d[near].astype(np.float32),
                        'observed_first_range_m':sweep['observed_range_m'][ids].astype(np.float32),'points_actor_m':local[near].astype(np.float32),
                        'positive_actor':(owner_idx[ids]==j)&(membership[ids]==1),'ambiguous_owner':membership[ids]>1,
                        'point_timestamps_ns':part['point_timestamps_ns'][near]})
                f={k:np.concatenate([p[k] for p in pieces]) for k in FIELDS}
                pose,known=track['poses'].at(np.int64(stamp))
                f['metadata']={'sample_id':str(stamp),'sample_index':index,'timestamp_ns':stamp,'timestamp_us':stamp//1000,
                    'role':'build' if index in build_ids else 'heldout_time','world_from_actor_at_scan':pose.tolist(),'scan_actor_pose_known':bool(known),
                    'raw_scan_points':n,'near_box_rays':len(f['positive_actor']),'owned_points':int(f['positive_actor'].sum())}
                actor_frames[owner].append(f)
            del sweep
        entries=[]
        for owner,frames in actor_frames.items():
            cid='av2-'+log['log_id']+'__'+owner;folder=out/cid;folder.mkdir(parents=True,exist_ok=True)
            bf=[f for f in frames if f['metadata']['role']=='build'];qf=[f for f in frames if f['metadata']['role']=='heldout_time']
            b=pack(bf);q=pack(qf);b['support_points_actor_m']=np.unique(b['points_actor_m'][b['positive_actor']],axis=0)
            b['size_lwh_m']=np.median(sizes_at(selected[owner],np.array([stamps[i] for i in build_ids],np.int64)),axis=0).astype(np.float32)
            np.savez(folder/'build.npz',**b);np.savez(folder/'query_truth.npz',**q)
            np.savez(folder/'query_rays.npz',**{k:q[k] for k in ['origins_actor_m','directions_actor','point_timestamps_ns','frame_offsets']})
            meta={'case_id':cid,'dataset':'av2','log_id':log['log_id'],'scene':'av2-'+log['log_id'],'owner':owner,'role':'FINAL_RESERVE',
                'category':selected[owner]['category'],'build_points':len(b['support_points_actor_m']),
                'build_frame_ids':[f['metadata']['sample_id'] for f in bf],'ownership':'unique per-return-time box +0.1m proxy; overlap not positive',
                'timing':'per-return AV2 ns; motion-compensated sweep endpoint transformed only once',
                'source_raw_log':str(data.root),'quality_evaluated':False}
            write(folder/'build_metadata.json',meta)
            entries.append({**meta,'build_file':str(folder/'build.npz'),'query_rays_file':str(folder/'query_rays.npz'),'query_truth_file':str(folder/'query_truth.npz')})
        write(done,{'schema':'worldsim_v74.lidar.v1','cases':entries,'model_quality_read':False})
        all_entries.extend(entries)
        print(json.dumps({'task':'FINAL_FORMAT','logs_done':number,'elapsed_s':time.monotonic()-start}),flush=True)
    write(root/'index.json',{'schema':'worldsim_v74.lidar.v1','cases':all_entries})
    # 储备对象的质量/正返回分布不用于筛选或 roadmap；只公开格式完成度。
    summary={'task':'WS-V74-H2-DATA-01','status':'coordinate_ready_quality_sealed','logs':len(plan['logs']),
        'data_root':str(root),'selection_unchanged':True,'model_quality_read':False,'used_in_training_or_teacher':False,
        'wall_s':time.monotonic()-start,'peak_rss_mib':peak_rss_mib(),'host':platform.node(),
        'scope':'raw metadata and lidar values opened for format conversion only; no method quality or selection by outcome'}
    write(root/'summary.json',summary);write(ROOT/'docs/autoresearch/worldsim_v74_h2/av2_reserve_summary.json',summary)
    print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','final']);p.add_argument('--output',type=Path,required=True);p.add_argument('--raw-root',type=Path);args=p.parse_args()
    fit(args.output) if args.mode=='fit' else final(args.output,args.raw_root)
