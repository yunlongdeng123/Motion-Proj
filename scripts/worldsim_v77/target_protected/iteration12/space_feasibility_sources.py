"""r43：移除固定B距离代理，先做实际空间检查，再投入实例分割。"""
from pathlib import Path
import sys,copy,time,os,shutil,random,fcntl,traceback
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).parent))
import temporal_source_index as previous
import space_source_pilot as p
import numpy as np

O=p.T/'r43';ROOT=O/'factory';E=Path('/root/autodl-tmp/motion_proj_v77/docs/autoresearch/worldsim_v77/target_protected_20260929/r43')
POLICY=copy.deepcopy(previous.prior.POLICY)
POLICY.update(min_B_depth_m=None,max_sources=6,split_seed=4301,
    depth_proxy_replacement='actual all-frame map/LiDAR/ego/collision/size checks before SAM; no geometric gate weakened')


def select():
    dest=ROOT/'source_manifest.json'
    if dest.exists():return p.read(dest)
    old={c['scene'] for run in ['r8','r37'] for c in p.read(p.T/run/'factory/source_manifest.json')['clips']}
    old|=set(p.read(p.T/'r8/source_split.json')['old_r7_training_scenes'])
    pool=p.read(p.T/'r42/metadata_probe/candidate_pool.json')['candidates']
    rows,counts=previous.metadata_filter(pool,old,POLICY)
    probe=O/'metadata_probe';probe.mkdir(exist_ok=True)
    for i,c in enumerate(rows):c['source_id']=f'W{i+1:03}'
    p.dump(probe/'source_selection.json',{'clips':rows,'policy':POLICY})
    if not (probe/'source_manifest.json').exists():p.resolve(p.META,probe,{'clips':rows},exposure_matcher=p.match_exposures)
    resolved=p.read(probe/'source_manifest.json');by=defaultdict(list);checks=[]
    for c in resolved['clips']:
        cams=np.array([fr['camera_to_world'] for fr in c['frames']])[:,:3,3]
        travel=float(np.linalg.norm(np.diff(cams[:,:2],axis=0),axis=1).sum());reasons=[]
        if c['source_geometry_status']=='geometry_reject':reasons.append('geometry_reject')
        if not POLICY['max_ego_camera_travel_m'][0]<=travel<=POLICY['max_ego_camera_travel_m'][1]:reasons.append('ego_motion_not_slow_parallax')
        if not reasons:
            bb=np.array([fr['actors'][0]['projection']['box_xyxy'] for fr in c['frames']]);wh=bb[:,2:]-bb[:,:2]
            if np.any(wh.min(0)<POLICY['min_B_box_wh']):reasons.append('small_between_keyframes')
        depths=[fr['actors'][0]['projection']['depth'] for fr in c['frames'] if fr['actors'] and fr['actors'][0]['projection']]
        checks.append({'source_id':c['source_id'],'scene':c['scene'],'camera':c['camera'],'ego_camera_travel_m':travel,
            'minimum_primary_depth_m':min(depths) if depths else None,'reasons':reasons})
        if not reasons:
            c['measured_camera_travel_m']=travel;c['minimum_primary_depth_m']=min(depths);by[c['scene']].append(c)
    # 先按同一过程强度取每scene代表，再location轮询；不使用RGB或生成结果。
    buckets=defaultdict(list)
    for scene,cc in sorted(by.items()):
        c=min(cc,key=lambda c:(abs(c['measured_camera_travel_m']-4.),c['start_keyframe'],c['camera'],c['source_id']))
        buckets[c['location']].append(c)
    chosen=[]
    while any(buckets.values()) and len(chosen)<POLICY['max_sources']:
        for loc in sorted(buckets):
            if buckets[loc] and len(chosen)<POLICY['max_sources']:chosen.append(buckets[loc].pop(0))
    scenes=sorted(c['scene'] for c in chosen);random.Random(POLICY['split_seed']).shuffle(scenes)
    val=set(scenes[:max(1,round(len(scenes)*.2))]) if scenes else set()
    for c in chosen:
        c['source_split']='validation' if c['scene'] in val else 'train'
        bad={tok for tok,q in c['actor_geometry_gate'].items() if not q['pass']}
        assert c['actors'][0]['instance_token'] not in bad
        c['actors']=[a for a in c['actors'] if a['instance_token'] not in bad]
        for fr in c['frames']:fr['actors']=[a for a in fr['actors'] if a['instance_token'] not in bad]
    selection={'clips':chosen,'policy':POLICY,'excluded_old_scenes':sorted(old),'source_count':len(chosen),
        'eligible_scenes_before_six_source_cap':len(by),'source_role':'training-source POC, not final test','human_verdict':None}
    p.dump(ROOT/'source_selection.json',selection)
    manifest={'clips':chosen,'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r43','policy':POLICY,
        'selection':str(ROOT/'source_selection.json'),'sampling_rejected':resolved['sampling_rejected']}
    p.dump(dest,manifest)
    p.dump(O/'source_summary.json',{'metadata_actor_windows':len(rows),'resolved_windows':len(resolved['clips']),
        'sampling_rejected':len(resolved['sampling_rejected']),'filters':dict(counts),'resolved_checks':checks,
        'eligible_scenes_before_cap':len(by),'selected_sources':len(chosen),'required_shards':sorted({s for c in chosen for s in c['shards']}),
        'split_counts':dict(Counter(c['source_split'] for c in chosen)),'training_admission':0,'training_steps':0})
    print('SPACE_FEASIBILITY_SOURCES',len(chosen),'of',len(by),'eligible scenes',flush=True)
    return manifest


def spatial_preflight(ready_sources=None):
    import reveal_factory as f
    from prepare_instance_audit import clean
    f.O=O;f.ROOT=ROOT;f.legacy.O=O;f.legacy.ROOT=ROOT
    g=f.legacy.geometry();maps={};directory=O/'source_proposals';directory.mkdir(exist_ok=True)
    for sid,c in sorted(g.sources.items()):
        if ready_sources is not None and sid not in ready_sources:continue
        path=directory/(sid+'.json')
        if path.exists():continue
        ground=g.prepare(sid);rejections=Counter();rows=[];attempts=0
        if ground['pass']:
            f.legacy.support(g,sid);loc=c['location']
            if loc not in maps:maps[loc]=f.NuScenesMap(dataroot=str(ROOT),map_name=loc)
            asset=f.POLICY['split_shape'][c['source_split']];size=[1.85,4.5,1.5] if asset=='sedan' else [1.90,4.6,1.7]
            for q in f.centers(g,maps[loc],c,size):
                # 本轮只验证世界静止A＋运动ego，不追加多个速度救结果。
                attempts+=1;tr,why=f.trajectory(g,c,q,0.,asset)
                if tr is None:rejections[why]+=1
                else:rows.append(clean(tr))
        else:rejections['ground_fit_unavailable']+=1
        p.dump(path,{'source_id':sid,'scene':c['scene'],'attempts':attempts,'spatial_trajectories':rows,
            'rejects':dict(rejections),'ground':clean(ground),'SAM_not_run':True})
        print('GEOMETRY_BEFORE_SAM',sid,len(rows),dict(rejections),flush=True)
    if ready_sources is not None:
        result={'completed_sources':sorted(ready_sources),'not_full_cohort':True,'training_admission':0}
        p.dump(O/'space_partial.json',result)
        return result
    stats=[p.read(x) for x in sorted(directory.glob('*.json'))];cases=[]
    for row in stats:
        for tr in row['spatial_trajectories']:
            cases.append({'case_id':f'G{len(cases)+1:03}','source_id':row['source_id'],'trajectory':tr,
                'reuse_saved_pixels':False,'sorted_first_rejection':'not_a_selector_pending_complete_instance_QA'})
    assert len(stats)==len(g.sources)
    p.dump(O/'candidate_roster.json',{'selection_id':'space_before_segmentation_static_A_v1','cases':cases,'count':len(cases),
        'saved_pixel_reuse':0,'sources':sorted(g.sources),'selection_frozen_before_extra_Y_SAM':True,
        'training_steps':0,'training_admission':0})
    summary={'sources':len(stats),'spatial_candidates':len(cases),'candidate_scenes':len({c['trajectory']['scene'] for c in cases}),
        'attempts':sum(r['attempts'] for r in stats),'rejects':dict(sum((Counter(r['rejects']) for r in stats),Counter())),
        'mask_QA_and_condition_not_done':True,'training_steps':0,'training_admission':0}
    p.dump(O/'space_summary.json',summary);print('SPACE_SUMMARY',summary,flush=True)
    return summary


def main():
    O.mkdir(exist_ok=True);ROOT.mkdir(exist_ok=True);E.mkdir(exist_ok=True,parents=True)
    lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r43','phase':'replace_distance_proxy_with_actual_space_check',
        'policy':POLICY,'input_role':'official train metadata, GT poses and LiDAR POC; real Y remains QA/supervision only',
        'change':'remove heuristic B minimum16m from source prefilter; retain clear/static-B, ego motion and all actual A-space gates',
        'bounded_factory':'max6 new scenes, fixed max24 lane centers, only world-static A; no speed/offset grids',
        'infrastructure_change':'world/map/ground/collision feasibility before expensive Y-SAM; full instance QA still mandatory',
        'resource_budget':'CPU source extraction and spatial preflight only; selected files only, require30GiB free; GPU separate stage',
        'stop':'once fixed six-source cohort evaluated, record all denominators; no threshold tuning or admission from map checks alone',
        'training_steps':0,'training_admission':0,'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    if (O/'run.json').exists():assert p.read(O/'run.json')==run
    else:p.dump(O/'run.json',run)
    p.dump(E/'run.json',run);state={'stage':'metadata_resolve','pid':os.getpid(),'started':time.time(),'training_steps':0};p.dump(O/'controller_state.json',state)
    try:
        assert shutil.disk_usage(O).free>30*2**30
        data=select();clips=data['clips']
        if clips:
            if not (ROOT/'source_context.json').exists():p.context(ROOT,p.META)
            names=set(p.read(ROOT/'required_files.json'));context={c['source_id']:c for c in p.read(ROOT/'source_context.json')['clips']}
            for c in clips:
                lo,hi=c['frames'][0]['timestamp'],c['frames'][-1]['timestamp']
                for frame in context[c['source_id']]['frames']:
                    d=frame['sensors']['LIDAR_TOP']
                    if lo<=d['timestamp']<=hi:names.add(d['filename'])
            p.dump(ROOT/'required_files.json',sorted(names));state['stage']='selected_public_archive_extraction';p.dump(O/'controller_state.json',state)
            allowed_shards=p.read(O/'source_summary.json')['required_shards']
            p.extract(ROOT,p.PUB,allowed_shards=allowed_shards,max_workers=2);assert p.read(ROOT/'extract_state.json')['state']=='complete'
            p.O=O;p.ROOT=ROOT;p.provision_maps(clips);p.validate_ready()
            state['stage']='spatial_preflight_no_SAM';p.dump(O/'controller_state.json',state);spatial_preflight()
        state.update(stage='CPU_complete_pending_geometry_result',selected_sources=len(clips),seconds=time.time()-state['started']);p.dump(O/'controller_state.json',state)
        for name in ['source_summary.json','source_ready.json','map_provenance.json','space_summary.json']:
            if (O/name).exists():p.dump(E/name,p.read(O/name))
    except Exception as error:
        state.update(stage='engineering_error',error=repr(error),traceback=traceback.format_exc());p.dump(O/'controller_state.json',state);raise


if __name__=='__main__':main()
