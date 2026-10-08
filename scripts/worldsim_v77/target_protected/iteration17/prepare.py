"""只用已有真实RGB/官方元数据冻结train开发目标；不读失败输出。"""
from common import *
from collections import Counter, defaultdict
import random, time
import ijson
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train
from prepare_clips import polygon, choose_instance_prompt
from PIL import Image

SEED = 771050
OLD_NINE = {'scene-0230','scene-0255','scene-0061','scene-0436','scene-0875',
            'scene-0242','scene-0535','scene-0471','scene-0998'}


def stream(name):
    with (META/name).open('rb') as handle:
        yield from ijson.items(handle, 'item', use_float=True)


def cached_sources():
    """缓存限定明确记录，去掉重复曝光；30帧来源统一取中间10帧。"""
    roots = sorted(T.glob('r*/factory/source_manifest.json'))
    available = [p.parent/'rgb' for p in roots]
    source_scenes = set(); sources = []; used = set(); exclusions = Counter()
    for manifest in roots:
        for c in read(manifest).get('clips', []):
            source_scenes.add(c['scene'])
            if c['scene'] not in train or c['scene'] in OLD_NINE:
                exclusions['not_train_or_old_nine'] += 1; continue
            if any(s in c.get('description','').lower() for s in ['night','dark','low light']):
                exclusions['night_metadata'] += 1; continue
            frames = c.get('frames', [])
            if len(frames) < 10:
                exclusions['short_window'] += 1; continue
            start = (len(frames)-10)//2; frames = frames[start:start+10]
            key = (c['scene'], c['camera'], tuple(f['token'] for f in frames))
            if key in used:
                exclusions['duplicate_exposure_window'] += 1; continue
            paths = [next((r/f['filename'] for r in available if (r/f['filename']).is_file()), None) for f in frames]
            if any(p is None for p in paths):
                exclusions['RGB_not_cached'] += 1; continue
            dt = np.diff([f['timestamp'] for f in frames])/1e6
            # 相机曝光是20Hz，旧10Hz目标网格的最近帧会有50/150ms间隔。
            # 保留真实曝光，不插帧；不把正常最近帧量化当作时序跳变。
            if not (np.all((dt >= .04)&(dt <= .17)) and .08<=np.median(dt)<=.12):
                exclusions['invalid_actual_exposure_spacing'] += 1; continue
            used.add(key)
            sources.append({'source_id':f'S{len(sources)+1:04}', 'scene':c['scene'],
                'camera':c['camera'], 'description':c.get('description',''),
                'manifest':str(manifest), 'original_source_id':c.get('source_id'),
                'source_slice':[start,start+10], 'frames':frames,
                'RGB_paths':[str(p) for p in paths]})
    return sources, source_scenes, dict(exclusions)


def official_geometry(sources):
    samples = {r['token']:r for r in read(META/'sample.json')}
    wanted = {f['sample_token'] for s in sources for f in s['frames']}
    wanted |= {samples[t]['prev'] for t in list(wanted) if samples[t]['prev']}
    cat = {r['token']:r['name'] for r in read(META/'category.json')}
    inst = {r['token']:cat[r['category_token']] for r in read(META/'instance.json')}
    annotations = {}; by_sample = defaultdict(list)
    for a in stream('sample_annotation.json'):
        if a['sample_token'] in wanted:
            a = dict(a, category_name=inst[a['instance_token']])
            annotations[a['token']] = a; by_sample[a['sample_token']].append(a['token'])
    print('ANNOTATIONS',len(annotations),flush=True)
    frame_tokens = {f['token'] for s in sources for f in s['frames']}
    sd = {r['token']:r for r in stream('sample_data.json') if r['token'] in frame_tokens}
    assert set(sd) == frame_tokens
    tables = {'sample':{t:dict(samples[t],anns=by_sample[t]) for t in wanted},
              'sample_annotation':annotations,'sample_data':sd}
    sdk = NuScenes.__new__(NuScenes); sdk.get = lambda table,token:tables[table][token]
    for s in sources:
        frames = []
        for i, f in enumerate(s['frames']):
            d = sd[f['token']]
            assert d['filename']==f['filename'] and d['timestamp']==f['timestamp']
            camera = {'w2c':np.linalg.inv(np.array(f['camera_to_world'])),
                      'k':np.array(f['intrinsics_1024'])}
            actors = []
            for box in sdk.get_boxes(d['token']):
                a = annotations[box.token]
                if not a['category_name'].startswith('vehicle.'): continue
                q = {'instance_token':a['instance_token'],'category':a['category_name'],
                     'translation':box.center.tolist(),'size':box.wlh.tolist(),
                     'rotation':box.orientation.elements.tolist()}
                shape = polygon(q,camera)
                if shape is None: continue
                q.update(shape,depth=float((camera['w2c']@np.r_[box.center,1])[2]),
                         visibility=int(a['visibility_token']),annotation_token=box.token)
                actors.append(q)
            frames.append({'frame':i,'filename':f['filename'],'sample_data_token':d['token'],
                'sample_token':d['sample_token'],'timestamp':d['timestamp'],
                'is_key_frame':d['is_key_frame'],
                'pose_source':'SDK associated keyframe' if d['is_key_frame'] else 'SDK previous/current interpolation',
                'camera_to_world':f['camera_to_world'],'intrinsics_1024':f['intrinsics_1024'],
                'actors':actors})
        s['frames'] = frames
    return sources


def overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def candidates(sources):
    pool=[]; rejected=Counter()
    for s in sources:
        tokens={a['instance_token'] for a in s['frames'][5]['actors'] if a['category']=='vehicle.car'}
        for tok in sorted(tokens):
            aa=[next((a for a in f['actors'] if a['instance_token']==tok),None) for f in s['frames']]
            if any(a is None for a in aa): rejected['not_visible_all10']+=1;continue
            boxes=np.array([a['box_xyxy'] for a in aa]);wh=boxes[:,2:]-boxes[:,:2]
            area=np.prod(wh,axis=1)/(1024*576)
            border=np.min(np.c_[boxes[:,:2],1024-boxes[:,2],576-boxes[:,3]])
            if np.min(wh[:,0])<80 or np.min(wh[:,1])<40 or np.median(wh[:,0])<100:
                rejected['small_target']+=1;continue
            if border<4 or np.max(area)>.28:
                rejected['clipped_or_oversized']+=1;continue
            if min(a['visibility'] for a in aa)<3:
                rejected['low_global_GT_visibility']+=1;continue
            anchor=aa[5];box=anchor['box_xyxy'];behind=[];front=[]
            for b in s['frames'][5]['actors']:
                if b['instance_token']==tok:continue
                fraction=overlap(box,b['box_xyxy'])/max(1,np.prod(wh[5]))
                if fraction>=.03 and b['depth']>anchor['depth']+1:
                    bwh=np.diff(np.array(b['box_xyxy']).reshape(2,2),axis=0)[0]
                    if b['category']=='vehicle.car' and min(bwh[0]/64,bwh[1]/32)>=1:
                        behind.append({'instance_token':b['instance_token'],'overlap_fraction':fraction,
                                       'width':float(bwh[0]),'height':float(bwh[1]),'visibility':b['visibility']})
                elif fraction>=.1 and b['depth']<anchor['depth']-1:front.append(fraction)
            if front and max(front)>.3: rejected['large_foreground_overlap_proxy']+=1;continue
            pool.append({'source_id':s['source_id'],'scene':s['scene'],'camera':s['camera'],
                'instance_token':tok,'median_width':float(np.median(wh[:,0])),
                'median_height':float(np.median(wh[:,1])),'median_area_fraction':float(np.median(area)),
                'visibility_min':min(a['visibility'] for a in aa),'border_min_px':float(border),
                'behind_vehicle_proxy':bool(behind),'behind':behind,
                'near_vehicle_count':len(s['frames'][5]['actors'])-1})
    return pool,dict(rejected)


def select(pool, source_scenes):
    rng=random.Random(SEED); by_scene=defaultdict(list)
    for c in pool:by_scene[c['scene']].append(c)
    names=sorted(n for n,cs in by_scene.items() if len({c['instance_token'] for c in cs})>=2)
    rng.shuffle(names)
    # 含后车的场景优先；内部按固定随机顺序。所有决定均发生在推理前。
    names.sort(key=lambda n:not any(c['behind_vehicle_proxy'] for c in by_scene[n]))
    assert len(names)>=20,('eligible scenes',len(names))
    names=names[:20];selected=[];third=[]
    for scene in names:
        rows=by_scene[scene].copy();rng.shuffle(rows)
        rows.sort(key=lambda c:(not c['behind_vehicle_proxy'],-c['visibility_min'],-c['median_width']))
        used=set();best=[]
        for c in rows:
            if c['instance_token'] in used:continue
            used.add(c['instance_token']);best.append(c)
            if len(best)==3:break
        selected.extend(best[:2])
        if len(best)>2:third.append(best[2])
    selected.extend(third[:10])
    assert 40<=len(selected)<=50
    reserve=sorted(set(train)-source_scenes-OLD_NINE)
    rng.shuffle(reserve)
    return selected,names,reserve[:5]


def materialize_case(c, s, cid):
    """追加候选沿用原始 RGB、SDK 几何和 prompt，不重选旧 ID。"""
    dest=O/'inputs'/cid
    (dest/'rgb').mkdir(parents=True,exist_ok=True);frames=[];crop_stats=[]
    for f,path in zip(s['frames'],s['RGB_paths']):
        a=next(a for a in f['actors'] if a['instance_token']==c['instance_token'])
        im=Image.open(path).convert('RGB').resize((1024,576),Image.Resampling.BILINEAR)
        im.save(dest/'rgb'/f"{f['frame']:05}.jpg",quality=96)
        frames.append(dict(f,target=a,neighbors=[b for b in f['actors'] if b['instance_token']!=c['instance_token']]))
        if f['frame'] in [0,5,9]:
            x0,y0,x1,y1=np.rint(a['box_xyxy']).astype(int)
            crop=np.asarray(im)[y0:y1,x0:x1];gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
            crop_stats.append({'frame':f['frame'],'laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                               'mean_luma':float(gray.mean()),'width':int(x1-x0),'height':int(y1-y0)})
    prompt,policy=choose_instance_prompt(frames)
    main_b=max(c['behind'],key=lambda b:b['overlap_fraction']) if c['behind'] else None
    refs=[]
    if main_b:
        for f in frames:
            b=next((b for b in f['neighbors'] if b['instance_token']==main_b['instance_token']),None)
            if b:
                bb=b['box_xyxy'];ov=overlap(bb,f['target']['box_xyxy'])/max(1,(bb[2]-bb[0])*(bb[3]-bb[1]))
                refs.append({'frame':f['frame'],'box_xyxy':bb,'target_bbox_overlap_fraction':ov,'visibility':b['visibility']})
    best_b=min(refs,key=lambda b:(b['target_bbox_overlap_fraction'],-b['visibility'],abs(b['frame']-5))) if refs else None
    flags=[]
    if any(x['laplacian_variance']<20 for x in crop_stats):flags.append('low_target_texture_or_blur_proxy')
    if any(x['mean_luma']<45 for x in crop_stats):flags.append('dark_target_proxy')
    if c['behind_vehicle_proxy']:flags.append('behind_actor_geometry_proxy')
    case=dict(c,case_id=cid,split='real_train_DEV',kind='real',folder=str(dest),frames=frames,
        source_RGB_paths=s['RGB_paths'],source_manifest=s['manifest'],source_slice=s['source_slice'],
        prompt_frame=prompt,prompt_policy=policy,crop_diagnostics=crop_stats,
        protected_actor=main_b,protected_evidence=refs,best_protected_reference=best_b,
        input_difficulty='medium' if flags else 'low',difficulty_factors=flags,
        input_visual_review='pending',hidden_region_GT=None,human_verdict=None)
    dump(dest/'case.json',case)
    return case


def main():
    if (O/'manifest.json').exists():print('ALREADY_PREPARED');return
    cv2.setNumThreads(1); O.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    dump(O/'controller_state.json',{'stage':'CPU_sampling','pid':os.getpid(),'GPU_jobs':0,'training_steps':0})
    cache=O/'CPU_selection_geometry.json'
    if cache.exists():
        state=read(cache);sources=state['sources'];pool=state['pool'];rejections=state['rejections']
        exposed=set(state['exposed']);exclusions=state['exclusions'];chosen=state['chosen']
        scenes=state['scenes'];reserve=state['reserve']
    else:
        sources,exposed,exclusions=cached_sources();print('CACHED_SOURCES',len(sources),exclusions,flush=True)
        assert sources,'没有可用RGB窗，停止元数据扫描'
        sources=official_geometry(sources); pool,rejections=candidates(sources)
        chosen,scenes,reserve=select(pool,exposed)
        dump(cache,{'sources':sources,'pool':pool,'rejections':rejections,'exposed':sorted(exposed),
                    'exclusions':exclusions,'chosen':chosen,'scenes':scenes,'reserve':reserve})
    by_source={s['source_id']:s for s in sources}
    cases=[]
    for i,c in enumerate(chosen):
        cid=f'R{ i+1:03}';s=by_source[c['source_id']]
        cases.append(materialize_case(c,s,cid))
        print('CPU_CASE',cid,c['scene'],c['camera'],flush=True)
    dump(O/'sampling.json',{'seed':SEED,'cached_scene_count':len(exposed),'sources':len(sources),
        'candidate_count':len(pool),'candidate_scene_count':len({c['scene'] for c in pool}),
        'source_exclusions':exclusions,'target_rejections':rejections,'DEV_scenes':scenes,
        'reserve_scenes':reserve,'reserve_role':'scene-isolated future gain check; RGB not loaded, not used for training or method selection',
        'selection_condition':'existing cached source population, not uniform sample of all700 nuScenes train scenes',
        'prior_source_scenes':sorted(exposed),'pool':pool})
    manifest={'task_id':TASK,'run_id':'r50','phase':'expanded_real_DELETE_single_frame_structure',
        'cases':cases,'scene_count':20,'case_count':len(cases),'reserve_scenes':reserve,
        'official_split':'nuScenes train; development/training discovery only',
        'base':'r46 official DriveEditor checkpoint + r21 sam_full_v2',
        'model_checkpoint':'/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor/checkpoints/model.safetensors',
        'SAM_checkpoint':'/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',
        'resolution':[576,1024],'frames':10,'seed':42,'sampling_steps':25,'previous_segment_condition':False,
        'adapter':False,'training_steps':0,'temporal_module_change':False,'factual_reconstruction':False,
        'selection_rule':'cached day train windows,10 unique actual exposures selected around10Hz; real gaps40-170ms, median80-120ms, no interpolatedRGB; one car present all10; min80x40, median width100, border4, area<=.28, GT visibility>=3, reject foreground bbox overlap>.3; 20 scenes with2-3 distinct instances; behind-car layouts first',
        'failure_focus':['film_or_ghost','protected_actor_deformation','car_body_extends_onto_road'],
        'audit_order':'first unify official DELETE; image review structural failures; then generate matched occlusions on independent real Y; finally test spatial layers',
        'synthetic_GT_policy':'only real RGB Y; generated failures never become training truth',
        'stop':'CPU prepare stops before SAM2/DriveEditor; after GPU one fixed window/actor; no per-case seed, hole tuning, temporal training, or automatic fine-tune',
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'none; no generated audit results yet','human_verdict':None}
    dump(O/'manifest.json',manifest)
    dump(O/'controller_state.json',{'stage':'CPU_ready_waiting_GPU','cases':len(cases),'scenes':20,
        'GPU_jobs':0,'training_steps':0,'seconds':time.monotonic()-started,'human_verdict':None})
    print('CPU_READY',len(cases),20,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        dump(O/'controller_state.json',{'stage':'CPU_engineering_error','error':repr(error),'GPU_jobs':0,'training_steps':0})
        raise
