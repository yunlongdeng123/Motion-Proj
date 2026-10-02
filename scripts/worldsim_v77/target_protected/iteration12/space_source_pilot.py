"""r37：离开旧两分片，先选远处可辨B和实际投影变化，再准备真实30帧。

只改变来源；不新增同来源的速度／位置网格，不运行训练。
"""
from pathlib import Path
import sys,os,json,copy,time,random,fcntl,traceback,shutil,zipfile
from collections import defaultdict,Counter
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
S=Path(__file__).parents[1];sys.path.insert(0,str(S))
sys.path.insert(0,str(S/'iteration4'))
import numpy as np
from prepare_source_pool import read,dump,resolve,good
from prepare_context_extract import context,extract
from nuscenes.utils.splits import train
from sample_windows import match_exposures
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r37';ROOT=O/'factory'
META=T.parent/'WS-V77-DELETE-AUDIT-20260928/r1/metadata/v1.0-trainval'
PUB=Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')
POLICY={'new_shards':['02','04'],'max_sources':20,'max_per_scene':1,'min_primary_depth_m':16.,
        'min_input_projection_change':.20,'frames':30,'split_seed':3701,'dev_fraction':.20,
        'prior_primary_good_gate_unchanged':True,'shape_split':{'train':'sedan','validation':'suv'},
        'image_or_model_output_used_in_source_selection':False}


def provision_maps(clips):
    archive=Path('/root/autodl-tmp/nuScenes-map-expansion-v1.3.zip')
    locations=sorted({c['location'] for c in clips});records=[]
    # 只读取明确命名的地图成员，不把整个压缩包解到工作目录。
    with zipfile.ZipFile(archive) as z:
        for location in locations:
            assert location in {'boston-seaport','singapore-hollandvillage','singapore-onenorth','singapore-queenstown'}
            member=f'expansion/{location}.json';data=z.read(member);meta=json.loads(data)
            dest=ROOT/'maps/expansion'/f'{location}.json';dest.parent.mkdir(exist_ok=True,parents=True)
            if dest.exists():assert dest.read_bytes()==data,'既有地图不同，须单独排查'
            else:dest.write_bytes(data)
            records.append({'location':location,'archive_member':member,'bytes':len(data),'version':meta['version']})
    result={'archive':str(archive),'new_download':False,'maps':records,
            'map_selection':'source location, never Boston fallback for Singapore'}
    if (O/'map_provenance.json').exists():assert read(O/'map_provenance.json')==result
    else:dump(O/'map_provenance.json',result)


def validate_ready():
    from PIL import Image
    clips=read(ROOT/'source_manifest.json')['clips'];required=read(ROOT/'required_files.json');frames=0;lidars=0
    for c in clips:
        for fr in c['frames']:
            with Image.open(ROOT/'rgb'/fr['filename']) as im:im.load();assert im.size==(1600,900)
            assert abs(np.linalg.det(np.asarray(fr['camera_to_world'])[:3,:3])-1)<1e-4
            frames+=1
    for name in required:
        path=ROOT/'rgb'/name;assert path.is_file() and path.stat().st_size>0
        if name.endswith('.bin'):
            points=np.fromfile(path,dtype=np.float32);assert points.size%5==0 and np.isfinite(points).all();lidars+=1
    result={'stage':'ready_for_quarantined_SAM2','source_count':len(clips),'real_frames_decoded':frames,
            'required_files':len(required),'valid_lidar_files':lidars,'synthetic_admission':0,
            'source_role':'offline real Y QA and supervision only; no condition texture access','human_verdict':None}
    if (O/'source_ready.json').exists():
        previous=read(O/'source_ready.json');assert all(previous[k]==v for k,v in result.items())
    else:dump(O/'source_ready.json',result|{'created_unix':time.time()})
    return result


def freeze():
    if (O/'run.json').exists():
        assert read(O/'run.json')['policy']==POLICY
        return read(ROOT/'source_selection.json')
    O.mkdir(exist_ok=True);ROOT.mkdir(exist_ok=True)
    old={c['scene'] for c in read(T/'r8/factory/source_manifest.json')['clips']}
    old|=set(read(T/'r8/source_split.json')['old_r7_training_scenes'])
    by=defaultdict(list);eligible=[]
    pool=read(T/'r1/candidate_pool.json')['candidates']
    for c in pool:
        if c['scene'] in old or not set(c['shards']).issubset(POLICY['new_shards']):continue
        assert c['scene'] in train,'不接nuScenes val/final'
        for a in c['actors']:
            pp=a['keyframe_projections']
            if len(pp)!=8 or not all(good(p) for p in pp):continue
            if not all(a.get('keyframe_annotations',[])):continue
            depth=min(p['depth'] for p in pp)
            if depth<POLICY['min_primary_depth_m']:continue
            bb=np.array([p['box_xyxy'] for p in pp]);wh=bb[:,2:]-bb[:,:2];cc=(bb[:,2:]+bb[:,:2])/2
            change=float(np.linalg.norm(cc[-1]-cc[0])/np.median(wh[:,0])+np.ptp(wh[:,0])/np.median(wh[:,0]))
            if change<POLICY['min_input_projection_change']:continue
            row=copy.deepcopy(c);row['actors']=[copy.deepcopy(a)]+[copy.deepcopy(x) for x in c['actors'] if x['instance_token']!=a['instance_token']]
            row['source_selection_metrics']={'min_depth_m':depth,'projection_change':change,'median_wh':np.median(wh,0).tolist(),
                'role':'clear primary B, no admission of future synthetic A implied'}
            eligible.append({'scene':c['scene'],'camera':c['camera'],'start_timestamp_us':c['start_timestamp_us'],
                             'primary':a['instance_token'],'category':a['category'],**row['source_selection_metrics']})
            by[c['scene']].append(row)
    chosen=[]
    for scene,cc in sorted(by.items()):
        # 先清楚的car，再可辨像素；不再最大化ego运动从而总选近场掠过。
        c=min(cc,key=lambda c:(c['actors'][0]['category']!='vehicle.car',
            -min(c['source_selection_metrics']['median_wh']),-c['source_selection_metrics']['min_depth_m'],
            c['start_keyframe'],c['camera'],c['actors'][0]['instance_token']))
        chosen.append(c)
    chosen=chosen[:POLICY['max_sources']]
    scenes=sorted(c['scene'] for c in chosen);random.Random(POLICY['split_seed']).shuffle(scenes)
    val=set(scenes[:max(1,round(len(scenes)*POLICY['dev_fraction']))])
    for i,c in enumerate(chosen):c.update(source_id=f'F{i+1:03}',source_split='validation' if c['scene'] in val else 'train')
    selection={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r37','clips':chosen,'policy':POLICY,
               'excluded_old_scenes':sorted(old),'eligible_metadata':eligible,'source_count':len(chosen),
               'scenes':len(scenes),'split_counts':dict(Counter(c['source_split'] for c in chosen)),
               'not_unbiased_audit':'training-source collection for physical space and well-observed B','human_verdict':None}
    dump(ROOT/'source_selection.json',selection)
    run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r37','phase':'space_aware_fresh_source_preparation',
         'policy':POLICY,'source_count':len(chosen),'scene_count':len(scenes),'frames':30,'training_steps':0,
         'host':'wm-3090-1001','failure_ledger_refs':['V77-F02'],'human_verdict':None,
         'source_role':'nuScenes train-domain collection, new relative to prior RGB source pool; DEV split frozen before images',
         'stop':'one fixed two-shard cohort; do not expand source thresholds or static-placement grid after results',
         'inputs':'metadata for selection; real Y only QA/supervision; any future condition must use final-H masked RGB',
         'resource_budget':'CPU only extraction; expected <2GB added; require 30GiB free; no weights/raw evidence deleted'}
    dump(O/'run.json',run)
    (O/'plan.md').write_text('''# r37：先补可放置空间的来源，再制造遮挡

旧r23源池主要来自03/07两分片；r33静止A解普遍撞上ego/尺度/地面限制，r35证明不能靠放宽水平截边解决。本轮只新增02/04两个公共分片的固定来源，先选全8关键帧B至少16m、保留旧72×40/边界/面积标准，并有归一化投影中心或尺度变化≥0.20的真实对象；每scene一条、最多20scene，先car，再可辨像素。门槛只是新来源筛选，不等于合成A空间或时序已经合格。

```mermaid
flowchart LR
 P[既有nuScenes train元数据池] --> S[新02/04分片\n远处可辨B＋投影变化]
 S --> F[固定scene split\n30唯一真实曝光]
 F --> R[按清单提取RGB与LiDAR]
 R --> Q[后续实际空间／显露检查]
 Q -.尚未准入.-> C[H遮后合法条件与独立QA]
```

这不是按模型输出选容易的评测集，而是训练来源收集。剔除既有RGB源scene与旧r7训练scene；新scene按seed3701冻结80/20训练／DEV，全部来自官方train，不能称最终测试。mask形状仍train sedan、DEV suv，已知两种旧形状的限制保留。

先准备30真实曝光，无复制／插帧；旧单帧可见性等级不继承为输入AI2。相机/轨迹/LiDAR仍明示几何辅助。完整Y永不作为未来条件输入。来源完成后才登记有界合成检查；本轮训练0，不提前调用DriveEditor，不新增自动化。只读取这两个分片一次，缺文件或错误则记录并停止，不重抽所有归档。
''')
    E=Path('/root/autodl-tmp/motion_proj_v77/docs/autoresearch/worldsim_v77/target_protected_20260929/r37');E.mkdir(exist_ok=True,parents=True)
    for name in ['run.json','plan.md']:shutil.copy2(O/name,E/name)
    dump(E/'source_summary.json',{k:v for k,v in selection.items() if k not in ['clips','eligible_metadata']})
    return selection


def main():
    selection=freeze();lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'stage':'metadata_resolve','pid':os.getpid(),'started':time.time(),'training_steps':0};dump(O/'controller_state.json',state)
    try:
        assert shutil.disk_usage(O).free>30*2**30,'先盘点可恢复临时物'
        if not (ROOT/'source_manifest.json').exists():resolve(META,ROOT,selection,exposure_matcher=match_exposures)
        d=read(ROOT/'source_manifest.json');d['run_id']='r37';kept=[];rejected=d.get('long_primary_rejected',[])
        for c in d['clips']:
            if c['source_geometry_status']=='geometry_reject':rejected.append({'source_id':c['source_id'],'gates':c['actor_geometry_gate']});continue
            bad={t for t,g in c['actor_geometry_gate'].items() if not g['pass']};c['actors']=[a for a in c['actors'] if a['instance_token'] not in bad]
            for fr in c['frames']:fr['actors']=[a for a in fr['actors'] if a['instance_token'] not in bad]
            kept.append(c)
        d['clips']=kept;d['long_primary_rejected']=rejected;dump(ROOT/'source_manifest.json',d)
        if not (ROOT/'source_context.json').exists():context(ROOT,META)
        # 为合法条件提前提取全部窗口内LiDAR；方法仍不能读窗口外RGB。
        names=set(read(ROOT/'required_files.json'));ctx={c['source_id']:c for c in read(ROOT/'source_context.json')['clips']}
        for c in kept:
            ta,tb=c['frames'][0]['timestamp'],c['frames'][-1]['timestamp']
            for k in ctx[c['source_id']]['frames']:
                datum=k['sensors']['LIDAR_TOP']
                if ta<=datum['timestamp']<=tb:names.add(datum['filename'])
        dump(ROOT/'required_files.json',sorted(names));state.update(stage='extracting_frozen_two_shards',actual_sources=len(kept));dump(O/'controller_state.json',state)
        extract(ROOT,PUB)
        ex=read(ROOT/'extract_state.json');assert ex['state']=='complete',ex['state']
        provision_maps(kept);validate_ready()
        state.update(stage='complete_real_sources_pending_synthetic_quality',frames=len(kept)*30,seconds=time.time()-state['started'],
                     materialized_bytes=ex['actual_bytes'],synthetic_admission=0);dump(O/'controller_state.json',state)
        print('SPACE_SOURCE_READY',len(kept),len(kept)*30,flush=True)
    except Exception as error:
        state.update(stage='engineering_error',error=repr(error),traceback=traceback.format_exc());dump(O/'controller_state.json',state);raise


if __name__=='__main__':main()
